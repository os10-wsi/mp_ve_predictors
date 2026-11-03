"""Scoring one protein with every requested score.

The shape is deliberately flat: resolve the inputs once, then ask each
score for a column. Nothing here knows how a particular score works; that
lives in ``scores.py`` and the runner scripts.

Two behaviours are worth stating because they are what make a long run
survivable. Every score's output is cached per protein, so re-running
fills only the gaps and costs nothing for what already succeeded. And a
score that cannot run, or fails, leaves its column absent rather than
taking the protein down with it, with the reason recorded next to the
result.

ThermoMPNN is the one score with two sources. The published table is
preferred when it covers the protein, and the model runs when it does
not. Which happened is recorded, because a column assembled from two
sources should never leave you guessing which one a number came from.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vepbench.config import PATHS, Paths, write_json
from vepbench.protein import ProteinContext
from vepbench.scores import (
    BUILTIN,
    MODEL,
    SCORES,
    STRUCTURE,
    TABLE,
    Score,
    ScoreError,
    applicable,
)
from vepbench.structure import StructureError, load_structure
from vepbench.uniprot import fetch_uniprot
from vepbench.variants import saturation_variants

log = logging.getLogger(__name__)

#: Scores whose cache was written under a different name by the previous
#: pipeline. Reading those saves re-running a model that already ran.
CACHE_ALIASES = {
    "esm1v": ("esm1v_full",),
    "blosum62": ("blosum62_baseline",),
}

OK = "ok"
CACHED = "cached"
UNAVAILABLE = "unavailable"
FAILED = "failed"


@dataclass
class ScoreOutcome:
    """What became of one score on one protein."""

    name: str
    status: str
    seconds: float = 0.0
    n_scored: int = 0
    source: str = ""
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "score": self.name, "status": self.status,
            "seconds": round(self.seconds, 2), "n_scored": self.n_scored,
            "source": self.source, "message": self.message[:400],
        }


@dataclass
class ProteinResult:
    uniprot_id: str
    table: pd.DataFrame
    outcomes: list[ScoreOutcome] = field(default_factory=list)
    is_human: bool = False

    @property
    def scored(self) -> list[str]:
        return [o.name for o in self.outcomes if o.status in (OK, CACHED)]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def build_context(uniprot_id: str, *, paths: Paths = PATHS, needs_structure: bool,
                  offline: bool = False) -> ProteinContext:
    """Fetch the sequence, and the structure when something needs one."""
    record = fetch_uniprot(uniprot_id, paths=paths, offline=offline)
    work_dir = paths.cache / "work" / uniprot_id
    work_dir.mkdir(parents=True, exist_ok=True)

    structure = None
    if needs_structure:
        try:
            structure = load_structure(uniprot_id, record.sequence, paths=paths,
                                       offline=offline)
            if structure.coverage < 0.9:
                log.warning(
                    "%s: the structure covers %.0f%% of the sequence; positions "
                    "without a mapped residue cannot be scored by the "
                    "structure-based scores", uniprot_id, 100 * structure.coverage,
                )
        except StructureError as exc:
            log.warning("%s: no structure (%s)", uniprot_id, exc)

    return ProteinContext(
        uniprot=record, work_dir=work_dir, structure=structure,
        weights_dir=paths.weights, third_party_dir=paths.third_party,
    )


def _cache_path(name: str, uniprot_id: str, paths: Paths) -> Path:
    return paths.predictions_cache(uniprot_id, name)


def _read_cache(
    score: Score, protein: ProteinContext, variants: pd.DataFrame, paths: Paths
) -> tuple[pd.Series | None, str]:
    """Cached values for this score, including under a previous name.

    A cache written by the old pipeline carries no fingerprint this one can
    reproduce, so instead of trusting a hash the rows are checked against
    the sequence: every cached wild-type residue has to be the residue the
    protein actually has at that position. That is the same gate the
    ClinVar mapping uses, and it catches the case that matters -- a cache
    belonging to a different isoform.
    """
    for name in (score.name, *CACHE_ALIASES.get(score.name, ())):
        path = _cache_path(name, protein.uniprot_id, paths)
        if not path.exists():
            continue
        try:
            cached = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            log.warning("unreadable cache at %s (%s); ignoring", path, exc)
            continue
        if not {"position", "wt_aa", "mut_aa", "score"} <= set(cached.columns):
            continue

        positions = pd.to_numeric(cached["position"], errors="coerce")
        if not _matches_sequence(positions, cached["wt_aa"], protein.sequence):
            log.warning("%s: cached %s does not match the current sequence; "
                        "ignoring it", protein.uniprot_id, name)
            continue

        lookup = dict(zip(
            [f"{w}{int(p)}{m}" for w, p, m in
             zip(cached["wt_aa"], positions, cached["mut_aa"], strict=True)],
            pd.to_numeric(cached["score"], errors="coerce"), strict=True,
        ))
        values = _apply_lookup(variants, lookup)
        if values.notna().any():
            return values, name
    return None, ""


def _matches_sequence(positions: pd.Series, wild_type: pd.Series, sequence: str) -> bool:
    """Whether every cached wild-type residue is the one the sequence carries."""
    for position, residue in zip(positions, wild_type, strict=True):
        if pd.isna(position):
            return False
        index = int(position)
        if index < 1 or index > len(sequence) or sequence[index - 1] != residue:
            return False
    return True


def _apply_lookup(variants: pd.DataFrame, lookup: dict[str, float]) -> pd.Series:
    keys = [f"{w}{int(p)}{m}" for w, p, m in
            zip(variants["wt_aa"], variants["position"], variants["mut_aa"],
                strict=True)]
    return pd.Series([lookup.get(k, float("nan")) for k in keys],
                     index=variants.index, dtype=float)


def _write_cache(
    score: Score, protein: ProteinContext, variants: pd.DataFrame,
    values: pd.Series, paths: Paths, extra: dict,
) -> None:
    path = _cache_path(score.name, protein.uniprot_id, paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = variants.loc[:, ["position", "wt_aa", "mut_aa"]].copy()
    frame.insert(0, "uniprot_id", protein.uniprot_id)
    frame["variant"] = [f"{w}{p}{m}" for w, p, m in
                        zip(frame["wt_aa"], frame["position"], frame["mut_aa"],
                            strict=True)]
    frame["score"] = values.to_numpy()
    frame = frame.loc[frame["score"].notna()]
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    path.with_suffix(".meta.json").write_text(json.dumps(
        {"score": score.name, "units": score.units, "written": time.time(), **extra},
        indent=2, default=str))


def compute_score(
    score: Score, protein: ProteinContext, variants: pd.DataFrame, *,
    paths: Paths = PATHS, force: bool = False, context: dict | None = None,
    clinvar_variants: pd.DataFrame | None = None, options: dict | None = None,
) -> tuple[pd.Series | None, ScoreOutcome]:
    """Produce one score's column, from cache when possible."""
    started = time.perf_counter()

    runnable, why = applicable(score, is_human=protein.is_human,
                               has_structure=protein.has_structure)
    if not runnable:
        return None, ScoreOutcome(score.name, UNAVAILABLE, message=why)

    if not force:
        values, source = _read_cache(score, protein, variants, paths)
        if values is not None:
            return values, ScoreOutcome(
                score.name, CACHED, time.perf_counter() - started,
                int(values.notna().sum()),
                source=f"cache ({source})" if source != score.name else "cache",
            )

    target = variants
    if score.default_scope == "clinvar":
        if clinvar_variants is None or clinvar_variants.empty:
            return None, ScoreOutcome(
                score.name, UNAVAILABLE,
                message="scored only at ClinVar positions, and this protein has none",
            )
        target = clinvar_variants

    try:
        if score.kind == BUILTIN:
            values = score.compute(target, context or {})
            source = "computed"
        elif score.kind == TABLE:
            values, source = _from_table(score, protein, target, paths, options or {})
        elif score.kind == MODEL:
            values, source = _from_model(score, protein, target, paths, options or {})
        else:  # pragma: no cover - the registry is closed
            raise ScoreError(f"unknown score kind {score.kind!r}")
    except Exception as exc:  # noqa: BLE001 - one score must not stop the protein
        log.error("%s failed on %s: %s", score.name, protein.uniprot_id, exc)
        return None, ScoreOutcome(score.name, FAILED,
                                  time.perf_counter() - started, message=str(exc))

    if values is None:
        return None, ScoreOutcome(score.name, FAILED,
                                  time.perf_counter() - started,
                                  message="no values produced")

    if target is not variants:
        # A ClinVar-scoped score produced a shorter column; widen it back.
        lookup = dict(zip(
            [f"{w}{int(p)}{m}" for w, p, m in zip(
                target["wt_aa"], target["position"], target["mut_aa"], strict=True)],
            values, strict=True))
        values = _apply_lookup(variants, lookup)

    _write_cache(score, protein, variants, values, paths, {"source": source})
    return values, ScoreOutcome(
        score.name, OK, time.perf_counter() - started,
        int(values.notna().sum()), source=source,
    )


def _from_table(
    score: Score, protein: ProteinContext, variants: pd.DataFrame, paths: Paths,
    options: dict,
) -> tuple[pd.Series | None, str]:
    """AlphaMissense and ThermoMPNN, both read from published tables."""
    if score.name == "alphamissense":
        from vepbench.alphamissense import load_slice

        published = load_slice(protein.uniprot_id, paths=paths)
        if not published:
            raise ScoreError(
                f"no AlphaMissense slice for {protein.uniprot_id}; run "
                f"`vepbench fetch alphamissense`")
        return _apply_lookup(
            variants, {k: v[0] for k, v in published.items()}), "published table"

    if score.name == "thermompnn":
        from vepbench.thermompnn_table import lookup as table_lookup

        index = options.get("thermompnn_index") or {}
        published = index.get(protein.uniprot_id, {})
        if published:
            values, coverage = table_lookup(variants, published, protein.uniprot_id)
            log.info("%s", coverage.describe())
            if coverage.usable:
                return values, "published table"
            log.info("%s: coverage too low to use, computing instead",
                     protein.uniprot_id)
        else:
            log.info("%s: not in the published ThermoMPNN table, computing",
                     protein.uniprot_id)
        return _from_model(score, protein, variants, paths, options)

    raise ScoreError(f"no table source for {score.name!r}")


def _from_model(
    score: Score, protein: ProteinContext, variants: pd.DataFrame, paths: Paths,
    options: dict,
) -> tuple[pd.Series | None, str]:
    """Run a score in its isolated environment, over a subprocess."""
    from vepbench.envs import run_score

    frame = run_score(
        score, protein, variants, paths=paths,
        options=options.get("runtime", {}),
    )
    lookup = dict(zip(
        [f"{w}{int(p)}{m}" for w, p, m in
         zip(frame["wt_aa"], frame["position"], frame["mut_aa"], strict=True)],
        pd.to_numeric(frame["score"], errors="coerce"), strict=True))
    return _apply_lookup(variants, lookup), "computed"


def score_protein(
    uniprot_id: str, requested: list[Score], *, paths: Paths = PATHS,
    force: bool = False, offline: bool = False, options: dict | None = None,
) -> ProteinResult:
    """Score one protein, returning the wide table and what happened."""
    options = options or {}
    needs_structure = any(STRUCTURE in s.needs for s in requested)
    protein = build_context(uniprot_id, paths=paths, needs_structure=needs_structure,
                            offline=offline)
    variants = saturation_variants(protein.sequence)
    log.info("%s: %s, %d residues, %d variants, %d scores requested",
             uniprot_id, protein.uniprot.organism or "unknown organism",
             protein.length, len(variants), len(requested))

    labels = _clinvar_labels(protein, variants, paths=paths) if protein.is_human else None
    clinvar_variants = (
        variants.loc[labels.notna().to_numpy()] if labels is not None else None
    )

    context: dict = {}
    if protein.structure is not None:
        from vepbench.structure import relative_sasa

        context["rsasa"] = relative_sasa(protein.structure)

    table = variants.copy()
    table.insert(0, "uniprot_id", uniprot_id)
    outcomes: list[ScoreOutcome] = []
    for score in requested:
        values, outcome = compute_score(
            score, protein, variants, paths=paths, force=force, context=context,
            clinvar_variants=clinvar_variants, options=options,
        )
        outcomes.append(outcome)
        log.info("  %-22s %-12s %7.1fs %s", outcome.name, outcome.status,
                 outcome.seconds, outcome.source or outcome.message[:60])
        if values is not None:
            table[score.name] = values.to_numpy()
            table[f"{score.name}__damaging"] = score.harmonize(values).to_numpy()

    # The table is everything known about this protein, not just what was
    # asked for on this invocation. Writing only the requested columns would
    # make `--scores esm1v` silently drop the other nine from the file, even
    # though their values are sitting in the cache.
    for name, score in SCORES.items():
        if name in table.columns:
            continue
        cached, source = _read_cache(score, protein, variants, paths)
        if cached is None:
            continue
        table[name] = cached.to_numpy()
        table[f"{name}__damaging"] = score.harmonize(cached).to_numpy()
        outcomes.append(ScoreOutcome(name, CACHED, 0.0, int(cached.notna().sum()),
                                     source=f"cache ({source})"))

    if labels is not None:
        table["clinical_significance"] = _clinvar_significance(
            protein, variants, paths=paths)
        table["label"] = labels.to_numpy()

    return ProteinResult(uniprot_id=uniprot_id, table=table, outcomes=outcomes,
                         is_human=protein.is_human)


def _clinvar_frame(protein: ProteinContext, paths: Paths) -> pd.DataFrame | None:
    from vepbench.clinvar_store import load_for_protein

    return load_for_protein(protein.uniprot_id, paths=paths)


def _clinvar_labels(
    protein: ProteinContext, variants: pd.DataFrame, *, paths: Paths
) -> pd.Series | None:
    frame = _clinvar_frame(protein, paths)
    if frame is None or frame.empty:
        return None
    lookup = dict(zip(frame["variant"], frame["label"], strict=True))
    keys = [f"{w}{p}{m}" for w, p, m in
            zip(variants["wt_aa"], variants["position"], variants["mut_aa"],
                strict=True)]
    return pd.Series([lookup.get(k) for k in keys], index=variants.index,
                     dtype="Float64")


def _clinvar_significance(
    protein: ProteinContext, variants: pd.DataFrame, *, paths: Paths
) -> pd.Series:
    frame = _clinvar_frame(protein, paths)
    lookup = (dict(zip(frame["variant"], frame["clinical_significance"], strict=True))
              if frame is not None and not frame.empty else {})
    keys = [f"{w}{p}{m}" for w, p, m in
            zip(variants["wt_aa"], variants["position"], variants["mut_aa"],
                strict=True)]
    return pd.Series([lookup.get(k, "") for k in keys], index=variants.index)


def write_protein(result: ProteinResult, *, paths: Paths = PATHS) -> Path:
    """Write one protein's table and its status record."""
    directory = paths.output_dir(result.uniprot_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scores.tsv"
    tmp = path.with_name(path.name + ".tmp")
    result.table.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)
    write_json(directory / "status.json", {
        "uniprot_id": result.uniprot_id,
        "is_human": result.is_human,
        "generated": time.time(),
        "n_variants": len(result.table),
        "scores": [o.as_dict() for o in result.outcomes],
    })
    return path

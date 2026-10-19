"""Orchestration: build the inputs once, run every predictor, write the outputs.

Three rules shape this module:

1. A failure in one predictor must not affect the others. Each runs inside its
   own try/except and its outcome is recorded in ``status.json``.
2. Finished work is never repeated. Each result is cached against a
   fingerprint of exactly the inputs that predictor consumes, so re-running
   recomputes only what actually changed.
3. Nothing is invented. A predictor that cannot run is marked ``unavailable``
   with the reason and contributes no rows.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import vepbench
from vepbench import VEPBENCH
from vepbench.config import RunConfig, cache_key, write_json
from vepbench.features import DEFAULT_FEATURES, build_features, inputs_for_features
from vepbench.msa import MSAError, build_msa
from vepbench.outputs import (
    load_existing_long,
    merge_long,
    merge_status_entries,
    output_lock,
    write_csv,
)
from vepbench.predictors import REGISTRY, resolve_names
from vepbench.predictors.base import Predictor
from vepbench.protein import ProteinContext
from vepbench.schema import concat_long, to_wide, validate_long_frame
from vepbench.structure import StructureError, load_structure
from vepbench.uniprot import choose_domain, fetch_uniprot
from vepbench.variants import saturation_variants

log = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_CACHED = "cached"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"


def git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


@dataclass
class PredictorOutcome:
    name: str
    status: str
    seconds: float = 0.0
    n_rows: int = 0
    message: str = ""
    model_version: str = ""
    traceback: str = ""

    def as_dict(self) -> dict:
        payload = {
            "predictor": self.name, "status": self.status,
            "seconds": round(self.seconds, 2), "n_rows": self.n_rows,
            "model_version": self.model_version,
        }
        if self.message:
            payload["message"] = self.message
        if self.traceback:
            payload["traceback"] = self.traceback
        return payload


@dataclass
class RunResult:
    uniprot_id: str
    long_frame: pd.DataFrame
    wide_frame: pd.DataFrame
    outcomes: list[PredictorOutcome] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    output_dir: Path | None = None

    @property
    def succeeded(self) -> list[str]:
        return [o.name for o in self.outcomes if o.status in (STATUS_OK, STATUS_CACHED)]

    @property
    def failed(self) -> list[str]:
        return [o.name for o in self.outcomes if o.status == STATUS_FAILED]

    @property
    def unavailable(self) -> list[str]:
        return [o.name for o in self.outcomes if o.status == STATUS_UNAVAILABLE]


def build_context(uniprot_id: str, config: RunConfig) -> ProteinContext:
    """Fetch sequence, structure, alignment and domain boundaries for one protein."""
    paths = config.paths
    paths.ensure()
    work_dir = paths.cache / "work" / uniprot_id
    work_dir.mkdir(parents=True, exist_ok=True)

    record = fetch_uniprot(uniprot_id, paths=paths, force=config.force,
                           offline=config.offline)
    if config.sequence_override:
        log.warning("overriding the UniProt sequence with the one supplied")
        record.sequence = config.sequence_override.strip().upper()

    needed = {
        requirement for name in config.predictors if name in REGISTRY
        for requirement in REGISTRY[name].spec.requires
    }
    # A requested feature is as good a reason to fetch an input as a predictor.
    needed |= inputs_for_features(config.features or list(DEFAULT_FEATURES))

    structure = None
    if "structure" in needed or config.structure_override:
        try:
            structure = load_structure(
                uniprot_id, record.sequence, override=config.structure_override,
                paths=paths, force=config.force, offline=config.offline,
            )
            if structure.coverage < 0.9:
                log.warning(
                    "the structure covers only %.0f%% of the sequence; positions "
                    "without a mapped residue cannot be scored by structure-based "
                    "predictors", 100 * structure.coverage,
                )
        except (StructureError, Exception) as exc:  # noqa: BLE001 - reported, not raised
            log.error("structure unavailable for %s: %s", uniprot_id, exc)

    msa = None
    if "msa" in needed or config.msa_override:
        try:
            msa = build_msa(
                uniprot_id, record.sequence, backend=config.msa_backend, paths=paths,
                override=config.msa_override, force=config.force, offline=config.offline,
                database=config.msa_database,
                email=os.environ.get("VEPBENCH_EMAIL", ""),
            )
        except (MSAError, Exception) as exc:  # noqa: BLE001 - reported, not raised
            log.error("alignment unavailable for %s: %s", uniprot_id, exc)

    domain, domain_note = None, ""
    if "domain" in needed or config.domain:
        prefer = "transmembrane" if record.is_membrane_protein else "auto"
        domain, domain_note = choose_domain(
            record, override=config.domain, prefer=prefer, paths=paths
        )
        log.info("domain window %s-%s (%s)", domain[0], domain[1], domain_note)

    return ProteinContext(
        uniprot=record, work_dir=work_dir, structure=structure, msa=msa,
        domain=domain, domain_note=domain_note, weights_dir=paths.weights,
        third_party_dir=paths.third_party,
        extras={"runtime_options": config.runtime_options()},
    )


def _fingerprint(predictor: Predictor, protein: ProteinContext, config: RunConfig) -> str:
    """Hash of exactly the inputs this predictor consumes.

    Only declared inputs are included, so choosing a domain window for a
    domain-level model does not invalidate a full-length model's cache.
    Hardware options are excluded deliberately: running on a GPU instead of a
    CPU should not change the score, and if it does that is a bug worth seeing.
    """
    requires = predictor.spec.requires
    return cache_key(
        predictor.name, protein.sequence,
        (protein.structure.identifier, protein.structure.path.name)
        if ("structure" in requires and protein.structure) else None,
        protein.msa.path.name if ("msa" in requires and protein.msa) else None,
        protein.domain if "domain" in requires else None,
        config.include_wt, getattr(predictor, "options", {}),
        predictor.spec.model_version,
    )


def _load_cached(path: Path, fingerprint: str) -> pd.DataFrame | None:
    meta_path = path.with_suffix(".meta.json")
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    if meta.get("fingerprint") != fingerprint:
        log.info("cache for %s is stale (inputs changed), recomputing", path.stem)
        return None
    return pd.read_csv(path)


def _store_cached(path: Path, frame: pd.DataFrame, fingerprint: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(
        json.dumps({"fingerprint": fingerprint, "written": time.time(), **extra},
                   indent=2, default=str)
    )


def run_predictor(
    predictor: Predictor, protein: ProteinContext, variants: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame | None, PredictorOutcome]:
    """Run one predictor, returning its long frame and an outcome record."""
    paths = config.paths
    cache_path = paths.predictions_cache(protein.uniprot_id, predictor.name)
    fingerprint = _fingerprint(predictor, protein, config)

    if not config.force:
        cached = _load_cached(cache_path, fingerprint)
        if cached is not None:
            version = str(cached["model_version"].iloc[0]) if len(cached) else ""
            return cached, PredictorOutcome(
                predictor.name, STATUS_CACHED, n_rows=len(cached), model_version=version
            )

    availability = predictor.availability(protein, paths=paths)
    if not availability.available:
        log.warning("%s unavailable: %s", predictor.name, availability.reason)
        return None, PredictorOutcome(
            predictor.name, STATUS_UNAVAILABLE, message=availability.reason
        )

    started = time.time()
    try:
        frame = predictor.predict(protein, variants)
        validate_long_frame(frame, protein.sequence)
    except Exception as exc:  # noqa: BLE001 - isolation is the point
        log.error("%s failed: %s", predictor.name, exc)
        return None, PredictorOutcome(
            predictor.name, STATUS_FAILED, seconds=time.time() - started,
            message=str(exc)[:500], traceback=traceback.format_exc()[-2000:],
        )

    elapsed = time.time() - started
    version = str(frame["model_version"].iloc[0]) if len(frame) else ""
    _store_cached(cache_path, frame, fingerprint, {
        "predictor": predictor.name, "model_version": version,
        "seconds": round(elapsed, 2),
    })
    _store_member_scores(predictor, protein, paths)
    return frame, PredictorOutcome(
        predictor.name, STATUS_OK, seconds=elapsed, n_rows=len(frame),
        model_version=version,
    )


def _store_member_scores(predictor: Predictor, protein: ProteinContext, paths) -> None:
    """Keep per-replicate scores, such as each of the five ESM-1v models."""
    source = protein.work_dir / predictor.name / "scores.csv"
    if not source.exists():
        return
    frame = pd.read_csv(source)
    members = [c for c in frame.columns if c.startswith("score_member")]
    if len(members) < 2:
        return
    destination = paths.output_dir(protein.uniprot_id) / "per_member"
    destination.mkdir(parents=True, exist_ok=True)
    columns = [c for c in ["position", "wt_aa", "mut_aa", *members] if c in frame.columns]
    frame.loc[:, columns].to_csv(destination / f"{predictor.name}.csv", index=False)


def run_protein(uniprot_id: str, config: RunConfig) -> RunResult:
    """Run the requested predictors for one protein and write all outputs."""
    started = time.time()
    names = resolve_names(config.predictors)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown predictors: {unknown}. Try `{VEPBENCH} list`.")
    config.predictors = names

    protein = build_context(uniprot_id, config)
    variants = saturation_variants(protein.sequence, include_wt=config.include_wt)
    log.info("%s: %d residues, %d variants, %d predictors requested",
             uniprot_id, protein.length, len(variants), len(names))

    frames: list[pd.DataFrame] = []
    outcomes: list[PredictorOutcome] = []
    for name in names:
        frame, outcome = run_predictor(REGISTRY[name], protein, variants, config)
        outcomes.append(outcome)
        if frame is not None:
            frames.append(frame)
        log.info("  %-28s %-12s %6.1fs %s", name, outcome.status, outcome.seconds,
                 outcome.message[:80])

    long_frame = concat_long(frames)
    feature_frame = build_features(protein, variants,
                                   config.features or list(DEFAULT_FEATURES))

    output_dir = config.paths.output_dir(uniprot_id)
    metadata = build_metadata(protein, config, outcomes, time.time() - started,
                              len(variants))

    # Merge onto whatever earlier runs left here rather than replacing it, so a
    # run of one predictor adds a column instead of removing the rest. The lock
    # covers the whole read-modify-write: another run on this protein may be
    # about to do the same thing.
    with output_lock(output_dir):
        long_path = output_dir / "predictions_long.csv"
        previous = load_existing_long(long_path, protein.sequence)
        long_frame = merge_long(previous, long_frame)
        wide_frame = to_wide(long_frame, feature_frame)
        recovered = sorted(set(previous["predictor"].unique())
                           - {o.name for o in outcomes}) if len(previous) else []
        if recovered:
            log.info("kept %d predictor(s) from earlier runs: %s",
                     len(recovered), ", ".join(recovered))

        write_csv(long_path, long_frame)
        write_csv(output_dir / "predictions_wide.csv", wide_frame)

        write_json(output_dir / "metadata.json", metadata)
        write_json(output_dir / "status.json",
                   _status_payload(uniprot_id, outcomes, output_dir / "status.json"))

    return RunResult(uniprot_id=uniprot_id, long_frame=long_frame, wide_frame=wide_frame,
                     outcomes=outcomes, metadata=metadata, output_dir=output_dir)


def _status_payload(uniprot_id: str, outcomes: list[PredictorOutcome], path: Path) -> dict:
    """Record this run's outcomes alongside the ones already on file.

    `vepbench status` reads these to tell "never tried" apart from "tried and
    failed". A narrow run that replaced the list would make every predictor it
    did not request look untried.
    """
    previous: list[dict] = []
    if path.exists():
        try:
            previous = json.loads(path.read_text()).get("predictors", [])
        except (OSError, ValueError) as exc:
            log.warning("could not read %s (%s); recording this run's results only",
                        path, exc)
    entries = merge_status_entries(previous, [o.as_dict() for o in outcomes])
    return {
        "uniprot_id": uniprot_id, "generated": time.time(),
        "predictors": entries,
        "counts": {
            "ok": sum(1 for e in entries if e["status"] == STATUS_OK),
            "cached": sum(1 for e in entries if e["status"] == STATUS_CACHED),
            "unavailable": sum(1 for e in entries if e["status"] == STATUS_UNAVAILABLE),
            "failed": sum(1 for e in entries if e["status"] == STATUS_FAILED),
        },
    }


def build_metadata(
    protein: ProteinContext, config: RunConfig, outcomes: list[PredictorOutcome],
    seconds: float, n_variants: int,
) -> dict:
    caveats: list[str] = []
    if protein.is_membrane_protein:
        caveats.append(
            "This is a membrane protein. The stability predictors here (FoldX, RaSP, "
            "ThermoMPNN) and the inverse-folding models (ProteinMPNN, ESM-IF1) "
            "were trained or parameterised largely on soluble, globular proteins. "
            "Their scores for lipid-facing residues are out of distribution and should "
            "not be read as free energies of folding in a bilayer."
        )
    if protein.msa is not None and protein.msa.is_shallow:
        caveats.append(
            f"The alignment is shallow (Neff {protein.msa.n_effective:.0f}, "
            f"{protein.msa.neff_per_length:.2f} per residue). Evolutionary models are "
            f"unreliable at this depth."
        )
    if protein.structure is not None and protein.structure.coverage < 0.9:
        caveats.append(
            f"The structure covers only {100 * protein.structure.coverage:.0f}% of the "
            f"sequence; uncovered positions have no structure-based scores."
        )
    if protein.structure is not None and protein.structure.is_predicted:
        caveats.append(
            "The structure is an AlphaFold prediction. For a membrane protein the "
            "model has no bilayer, so burial and surface features describe a protein "
            "in water."
        )
    leaky = [n for n in config.predictors
             if n in REGISTRY and REGISTRY[n].spec.trained_on_megascale]
    if leaky:
        caveats.append(
            "These predictors were trained on the Tsuboyama et al. 2023 mega-scale "
            f"stability dataset: {', '.join(leaky)}. Treat any overlap with that data "
            "as leakage when benchmarking."
        )

    return {
        "vepbench_version": vepbench.__version__, "git_hash": git_hash(),
        "generated": time.time(), "runtime_seconds": round(seconds, 1),
        "python": platform.python_version(), "platform": platform.platform(),
        "uniprot": protein.uniprot.as_dict(),
        "structure": protein.structure.as_dict() if protein.structure else None,
        "msa": protein.msa.as_dict() if protein.msa else None,
        "domain": {"window": list(protein.domain) if protein.domain else None,
                   "source": protein.domain_note},
        "n_variants": n_variants, "include_wt": config.include_wt,
        "features": list(config.features or DEFAULT_FEATURES),
        "hardware_options": config.runtime_options(),
        "predictors": {n: REGISTRY[n].spec.as_dict() for n in config.predictors
                       if n in REGISTRY},
        "outcomes": [o.as_dict() for o in outcomes],
        "failures": {o.name: o.message for o in outcomes if o.status == STATUS_FAILED},
        "unavailable": {o.name: o.message for o in outcomes
                        if o.status == STATUS_UNAVAILABLE},
        "caveats": caveats,
    }

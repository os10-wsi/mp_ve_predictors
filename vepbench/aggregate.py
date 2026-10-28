"""Turning thousands of per-protein outputs into one analysable dataset.

The per-protein CSVs under ``outputs/<ACC>/`` stay the record of what ran;
this module only derives from them, so it can be rerun at any point and
after any number of partial job arrays. That matters because a proteome run
is never finished all at once: tasks fail, get resubmitted, and a predictor
gets added weeks later. Aggregation therefore reports coverage rather than
assuming it, and the analysis table says plainly how much of the proteome it
rests on.

The join onto ClinVar is done one protein at a time. The labelled set is
small -- a few hundred thousand rows -- but the predictions it is drawn from
are tens of millions, and reading those into one frame would need more
memory than any sensible machine has.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vepbench import tables
from vepbench.config import PATHS, Paths, read_json
from vepbench.schema import HARMONIZED_SUFFIX, WIDE_KEY_COLUMNS

log = logging.getLogger(__name__)

PREDICTIONS_DIRNAME = "predictions"


class AggregateError(RuntimeError):
    pass


@dataclass
class Coverage:
    """What actually ran, as opposed to what was asked for."""

    proteins_expected: int = 0
    proteins_with_output: int = 0
    variants: int = 0
    per_predictor: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    failed: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{self.proteins_with_output:,} of {self.proteins_expected:,} proteins "
            f"have predictions ({self.variants:,} variant rows)"
        ]
        if self.per_predictor:
            width = max(len(name) for name in self.per_predictor)
            for name, count in sorted(self.per_predictor.items(), key=lambda kv: -kv[1]):
                share = 100 * count / max(self.proteins_with_output, 1)
                lines.append(f"  {name:<{width}}  {count:>6,} proteins  ({share:5.1f}%)")
        if self.missing:
            shown = ", ".join(self.missing[:8])
            more = f" and {len(self.missing) - 8} more" if len(self.missing) > 8 else ""
            lines.append(f"  no output at all: {len(self.missing):,} ({shown}{more})")
        return "\n".join(lines)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"predictor": name, "proteins": count,
              "fraction": count / max(self.proteins_with_output, 1)}
             for name, count in sorted(self.per_predictor.items())]
        )


def protein_predictions(uniprot_id: str, *, paths: Paths = PATHS) -> pd.DataFrame | None:
    """One protein's wide table, or None when it has not been run."""
    path = paths.output_dir(uniprot_id) / "predictions_wide.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "uniprot_id" not in frame.columns:
        frame.insert(0, "uniprot_id", uniprot_id)
    return frame


def protein_status(uniprot_id: str, *, paths: Paths = PATHS) -> dict:
    path = paths.output_dir(uniprot_id) / "status.json"
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except ValueError:
        log.warning("%s has an unreadable status.json", uniprot_id)
        return {}


def dataset_root(name: str = "human_membrane", *, paths: Paths = PATHS) -> Path:
    return paths.outputs / "proteome" / name / PREDICTIONS_DIRNAME


def build_dataset(
    accessions: list[str], *, name: str = "human_membrane", paths: Paths = PATHS,
    force: bool = False,
) -> Coverage:
    """Copy each protein's wide table into the partitioned dataset.

    Proteins already present are skipped unless ``force``, so this is cheap
    to rerun after another batch of array tasks lands.
    """
    root = dataset_root(name, paths=paths)
    root.mkdir(parents=True, exist_ok=True)
    present = set(tables.dataset_members(root)) if not force else set()

    coverage = Coverage(proteins_expected=len(accessions))
    for accession in accessions:
        frame = protein_predictions(accession, paths=paths)
        if frame is None or frame.empty:
            coverage.missing.append(accession)
            continue
        coverage.proteins_with_output += 1
        coverage.variants += len(frame)
        for predictor in _predictor_columns(frame):
            coverage.per_predictor[predictor] = coverage.per_predictor.get(predictor, 0) + 1

        status = protein_status(accession, paths=paths)
        failures = [
            entry.get("predictor", "?") for entry in status.get("predictors", [])
            if entry.get("status") == "failed"
        ]
        if failures:
            coverage.failed[accession] = failures

        if accession not in present:
            tables.write_partition(root, accession, frame)

    log.info("%s", coverage.summary())
    return coverage


def _predictor_columns(frame: pd.DataFrame) -> list[str]:
    """Predictor names that actually carry a score in this table."""
    out = []
    for column in frame.columns:
        if not column.endswith(HARMONIZED_SUFFIX):
            continue
        if frame[column].notna().any():
            out.append(column[: -len(HARMONIZED_SUFFIX)])
    return out


def join_clinvar(
    clinvar_frame: pd.DataFrame, *, name: str = "human_membrane", paths: Paths = PATHS,
    labelled_only: bool = False,
) -> tuple[pd.DataFrame, Coverage]:
    """Attach predictions and features to every ClinVar variant.

    Done per protein so memory stays bounded by the largest single protein
    rather than by the proteome. Variants whose protein was never run are
    kept with empty prediction columns and counted, because dropping them
    silently would make coverage look better than it is.
    """
    if clinvar_frame.empty:
        return clinvar_frame, Coverage()

    root = dataset_root(name, paths=paths)
    if labelled_only:
        clinvar_frame = clinvar_frame.loc[clinvar_frame["label"].notna()]

    accessions = sorted(clinvar_frame["uniprot_id"].unique())
    coverage = Coverage(proteins_expected=len(accessions))
    pieces: list[pd.DataFrame] = []

    # The dataset is derived and may not have been built yet, or may lag
    # behind the per-protein outputs by a batch of array tasks. Either way
    # the per-protein CSVs are the record, so fall back to them.
    has_dataset = root.exists()
    for accession in accessions:
        variants = clinvar_frame.loc[clinvar_frame["uniprot_id"] == accession]
        predictions = tables.read_dataset(root, [accession]) if has_dataset else pd.DataFrame()
        if predictions.empty:
            predictions = protein_predictions(accession, paths=paths)
        if predictions is None or predictions.empty:
            coverage.missing.append(accession)
            pieces.append(variants)
            continue

        coverage.proteins_with_output += 1
        coverage.variants += len(variants)
        for predictor in _predictor_columns(predictions):
            coverage.per_predictor[predictor] = coverage.per_predictor.get(predictor, 0) + 1

        # 'variant' is derived from the same three keys on both sides, so
        # joining on it as well would be redundant; drop it from the right.
        extra = [c for c in predictions.columns if c not in WIDE_KEY_COLUMNS]
        right = predictions.loc[:, ["position", "wt_aa", "mut_aa", *extra]]
        merged = variants.merge(right, how="left", on=["position", "wt_aa", "mut_aa"])
        pieces.append(merged)

    joined = pd.concat(pieces, ignore_index=True)
    log.info("joined %d ClinVar variants against %d proteins; %d proteins had no "
             "predictions", len(joined), coverage.proteins_with_output,
             len(coverage.missing))
    return joined, coverage


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    """Harmonized score columns present in an analysis table, sorted."""
    return sorted(c for c in frame.columns if c.endswith(HARMONIZED_SUFFIX))


def drop_empty_predictors(frame: pd.DataFrame, min_rows: int = 50) -> tuple[pd.DataFrame, list[str]]:
    """Remove predictor columns too sparse to score, and name them.

    A predictor present for a handful of variants produces an AUROC with a
    confidence interval spanning most of the possible range. Reporting it
    beside a predictor covering the whole set invites a comparison that the
    numbers do not support.
    """
    dropped = []
    for column in predictor_columns(frame):
        if frame[column].notna().sum() < min_rows:
            dropped.append(column[: -len(HARMONIZED_SUFFIX)])
            frame = frame.drop(columns=[column])
    if dropped:
        log.info("dropped %d predictor(s) with fewer than %d scored variants: %s",
                 len(dropped), min_rows, ", ".join(dropped))
    return frame, dropped

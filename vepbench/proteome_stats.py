"""Scoring predictors against ClinVar labels, pooled and stratified.

``benchmark.py`` scores against a continuous DMS measurement. ClinVar gives
a binary label instead, drawn from a population of variants that is nothing
like a random sample: pathogenic variants are enriched in well-studied
genes, in conserved positions, and in the proteins somebody had a reason to
sequence. Pooled numbers across the proteome therefore flatter every
predictor at once, which is why everything here is also reported per
protein and stratified by structural context.

Two warnings are built into the output rather than left to the reader.
AUROC is reported beside average precision, because the class balance runs
anywhere from 1:1 to 1:50 depending on the gene and AUROC hides that.
Predictors calibrated on ClinVar -- AlphaMissense among them -- are flagged
in the results table, because their numbers here are not measuring the same
thing as an unsupervised predictor's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vepbench.aggregate import predictor_columns
from vepbench.benchmark import _bootstrap_ci, auroc
from vepbench.schema import HARMONIZED_SUFFIX

log = logging.getLogger(__name__)

#: A protein needs this many of each class before a per-protein AUROC means
#: anything. Below it the number is noise with a wide interval.
MIN_PER_CLASS = 5

#: Stratifications applied to the pooled table. Each maps a column in the
#: analysis table to a function producing the group label.
RESULT_COLUMNS = [
    "stratum", "group", "predictor", "n", "n_pathogenic", "n_benign",
    "auroc", "auroc_ci_low", "auroc_ci_high", "average_precision",
    "trained_on_clinvar",
]


class StatsError(RuntimeError):
    pass


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, by the step-wise sum.

    More informative than AUROC when positives are rare, which they are in
    most single genes.
    """
    positives = labels.astype(bool)
    n_positive = int(positives.sum())
    if n_positive == 0 or n_positive == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = positives[order]
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_positive)


@dataclass(frozen=True)
class Stratum:
    """One slice of the data, and the rows it selects."""

    name: str
    group: str
    mask: pd.Series

    @property
    def size(self) -> int:
        return int(self.mask.sum())


def strata(frame: pd.DataFrame) -> list[Stratum]:
    """Every slice we report, skipping ones the table cannot support.

    A stratification whose column is absent is skipped rather than faked:
    the structural columns only exist for proteins that had a structure.
    """
    out = [Stratum("all", "all", pd.Series(True, index=frame.index))]

    if "in_transmembrane" in frame:
        membrane = frame["in_transmembrane"].fillna(False).astype(bool)
        out.append(Stratum("topology", "transmembrane", membrane))
        out.append(Stratum("topology", "non_transmembrane", ~membrane))

    if "rsasa" in frame:
        rsasa = pd.to_numeric(frame["rsasa"], errors="coerce")
        out.append(Stratum("burial", "buried", rsasa < 0.25))
        out.append(Stratum("burial", "intermediate", (rsasa >= 0.25) & (rsasa < 0.5)))
        out.append(Stratum("burial", "exposed", rsasa >= 0.5))

    if "plddt" in frame:
        plddt = pd.to_numeric(frame["plddt"], errors="coerce")
        out.append(Stratum("confidence", "plddt_high", plddt >= 90))
        out.append(Stratum("confidence", "plddt_medium", (plddt >= 70) & (plddt < 90)))
        out.append(Stratum("confidence", "plddt_low", plddt < 70))

    if "membrane_topology" in frame:
        for label in sorted(frame["membrane_topology"].dropna().unique()):
            out.append(Stratum("architecture", str(label),
                               frame["membrane_topology"] == label))

    if "review_stars" in frame:
        stars = pd.to_numeric(frame["review_stars"], errors="coerce")
        out.append(Stratum("review", "two_stars_or_better", stars >= 2))

    return out


def evaluate_labelled(
    frame: pd.DataFrame, *, label_col: str = "label", n_boot: int = 1000,
    seed: int = 0, min_per_class: int = MIN_PER_CLASS,
    supervised: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Pooled metrics per predictor, for every stratum the table supports."""
    columns = predictor_columns(frame)
    if not columns:
        raise StatsError(
            "no harmonized predictor columns in the analysis table; expected "
            f"columns ending in {HARMONIZED_SUFFIX!r}"
        )
    labels_all = pd.to_numeric(frame[label_col], errors="coerce")

    rows: list[dict] = []
    for stratum in strata(frame):
        subset = stratum.mask.fillna(False)
        for column in columns:
            name = column[: -len(HARMONIZED_SUFFIX)]
            scores = pd.to_numeric(frame[column], errors="coerce")
            keep = subset & scores.notna() & labels_all.notna()
            n_positive = int((labels_all[keep] == 1).sum())
            n_negative = int((labels_all[keep] == 0).sum())
            if n_positive < min_per_class or n_negative < min_per_class:
                continue
            x = scores[keep].to_numpy(dtype=float)
            y = labels_all[keep].to_numpy(dtype=float)
            low, high = _bootstrap_ci(
                lambda a, b: auroc(a, b.astype(bool)), x, y, n_boot=n_boot, seed=seed
            )
            rows.append({
                "stratum": stratum.name, "group": stratum.group, "predictor": name,
                "n": n_positive + n_negative, "n_pathogenic": n_positive,
                "n_benign": n_negative, "auroc": auroc(x, y.astype(bool)),
                "auroc_ci_low": low, "auroc_ci_high": high,
                "average_precision": average_precision(x, y.astype(bool)),
                "trained_on_clinvar": name in supervised,
            })

    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return result.sort_values(
        ["stratum", "group", "auroc"], ascending=[True, True, False]
    ).reset_index(drop=True)


def evaluate_per_protein(
    frame: pd.DataFrame, *, label_col: str = "label",
    min_per_class: int = MIN_PER_CLASS, supervised: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """AUROC within each protein, for predictors with enough of both classes.

    Pooled AUROC across the proteome can be high purely because some genes
    carry mostly pathogenic variants and others mostly benign ones: a
    predictor that only distinguished genes, not variants, would score well.
    Within a protein that shortcut is unavailable, so this is the stricter
    measurement, and the one to lead with.
    """
    columns = predictor_columns(frame)
    labels_all = pd.to_numeric(frame[label_col], errors="coerce")

    rows: list[dict] = []
    for accession, group in frame.groupby("uniprot_id"):
        labels = labels_all.loc[group.index]
        for column in columns:
            name = column[: -len(HARMONIZED_SUFFIX)]
            scores = pd.to_numeric(group[column], errors="coerce")
            keep = scores.notna() & labels.notna()
            n_positive = int((labels[keep] == 1).sum())
            n_negative = int((labels[keep] == 0).sum())
            if n_positive < min_per_class or n_negative < min_per_class:
                continue
            rows.append({
                "uniprot_id": accession, "predictor": name,
                "n": n_positive + n_negative, "n_pathogenic": n_positive,
                "n_benign": n_negative,
                "auroc": auroc(scores[keep].to_numpy(dtype=float),
                               labels[keep].to_numpy(dtype=float).astype(bool)),
                "average_precision": average_precision(
                    scores[keep].to_numpy(dtype=float),
                    labels[keep].to_numpy(dtype=float).astype(bool)),
                "trained_on_clinvar": name in supervised,
            })
    return pd.DataFrame(rows)


def summarise_per_protein(per_protein: pd.DataFrame) -> pd.DataFrame:
    """Median and spread of the per-protein AUROCs, per predictor."""
    if per_protein.empty:
        return pd.DataFrame(columns=["predictor", "n_proteins", "median_auroc"])
    grouped = per_protein.groupby("predictor")["auroc"]
    out = pd.DataFrame({
        "n_proteins": grouped.size(),
        "median_auroc": grouped.median(),
        "mean_auroc": grouped.mean(),
        "q1_auroc": grouped.quantile(0.25),
        "q3_auroc": grouped.quantile(0.75),
        "fraction_above_chance": grouped.apply(lambda s: float((s > 0.5).mean())),
    }).reset_index()
    supervised = per_protein.groupby("predictor")["trained_on_clinvar"].any()
    out["trained_on_clinvar"] = out["predictor"].map(supervised)
    return out.sort_values("median_auroc", ascending=False).reset_index(drop=True)


def supervised_predictors() -> frozenset[str]:
    """Predictors whose training touched ClinVar, from the registry."""
    from vepbench.predictors import REGISTRY

    return frozenset(
        name for name, predictor in REGISTRY.items()
        if predictor.spec.trained_on_clinvar or predictor.spec.supervised
    )


def write_results(directory: Path, frames: dict[str, pd.DataFrame]) -> list[Path]:
    """Write each results table as a TSV, returning the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, frame in frames.items():
        path = directory / f"{stem}.tsv"
        tmp = path.with_suffix(".tsv.tmp")
        frame.to_csv(tmp, sep="\t", index=False)
        tmp.replace(path)
        written.append(path)
    return written

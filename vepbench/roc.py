"""ROC analysis: one AUC per predictor per protein, with an error.

A thousand proteins times ten predictors is ten thousand AUCs, so the
error cannot come from bootstrapping. It uses the Hanley and McNeil
(1982) standard error instead, which is analytic, exact under their
assumptions and costs one sort. The curves themselves come from the same
sort, so a protein is scored once and both the number and the plot fall
out of it.

Labels come from ClinVar: pathogenic and likely pathogenic are the
positive class, benign and likely benign the negative. Variants of
uncertain significance are carried through the tables but never scored,
because they are the cases a predictor is most useful for and treating a
guess about them as truth would measure the guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Below this many of either class, an AUC is too unstable to report. Two
#: pathogenic variants can put the number anywhere.
MIN_PER_CLASS = 5


@dataclass(frozen=True)
class RocResult:
    """One predictor on one protein."""

    uniprot_id: str
    predictor: str
    n_pathogenic: int
    n_benign: int
    auc: float
    se: float
    fpr: np.ndarray
    tpr: np.ndarray

    @property
    def ci(self) -> tuple[float, float]:
        """Normal-approximation 95% interval, clipped to the unit interval."""
        if not np.isfinite(self.se):
            return (float("nan"), float("nan"))
        return (max(0.0, self.auc - 1.96 * self.se),
                min(1.0, self.auc + 1.96 * self.se))

    def as_row(self) -> dict:
        low, high = self.ci
        return {
            "uniprot_id": self.uniprot_id, "predictor": self.predictor,
            "n_pathogenic": self.n_pathogenic, "n_benign": self.n_benign,
            "n": self.n_pathogenic + self.n_benign,
            "auc": self.auc, "se": self.se, "ci_low": low, "ci_high": high,
        }


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, which is what makes the AUC identity exact."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def auc_score(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve, by the rank-sum identity."""
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = _ranks(scores)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def hanley_mcneil_se(auc: float, n_positive: int, n_negative: int) -> float:
    """Standard error of an AUC, Hanley & McNeil 1982.

    Their Q1 and Q2 are the probabilities that two randomly drawn
    positives, or two negatives, both rank above a case of the other
    class. Expressing them in terms of the AUC is the approximation that
    makes this analytic rather than a resampling job.
    """
    if n_positive < 1 or n_negative < 1 or not np.isfinite(auc):
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (n_positive - 1) * (q1 - auc * auc)
        + (n_negative - 1) * (q2 - auc * auc)
    ) / (n_positive * n_negative)
    return float(np.sqrt(variance)) if variance > 0 else 0.0


def roc_curve(scores: np.ndarray, positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates, highest score first."""
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-scores, kind="mergesort")
    hits = positive[order]
    true_positive = np.concatenate([[0.0], np.cumsum(hits)])
    false_positive = np.concatenate([[0.0], np.cumsum(~hits)])
    return false_positive / n_negative, true_positive / n_positive


def evaluate(
    frame: pd.DataFrame, predictors: list[str], *, uniprot_id: str,
    label_col: str = "label", min_per_class: int = MIN_PER_CLASS,
) -> list[RocResult]:
    """Score every predictor on one protein's labelled variants.

    Predictors with too few of either class, or no scores at all, are
    skipped rather than reported with an interval spanning most of the
    possible range.
    """
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    results: list[RocResult] = []
    for name in predictors:
        if name not in frame.columns:
            continue
        scores = pd.to_numeric(frame[name], errors="coerce")
        keep = scores.notna() & labels.notna()
        positive = (labels[keep] == 1).to_numpy(dtype=bool)
        n_positive, n_negative = int(positive.sum()), int((~positive).sum())
        if n_positive < min_per_class or n_negative < min_per_class:
            continue
        values = scores[keep].to_numpy(dtype=float)
        area = auc_score(values, positive)
        false_positive, true_positive = roc_curve(values, positive)
        results.append(RocResult(
            uniprot_id=uniprot_id, predictor=name,
            n_pathogenic=n_positive, n_benign=n_negative, auc=area,
            se=hanley_mcneil_se(area, n_positive, n_negative),
            fpr=false_positive, tpr=true_positive,
        ))
    return results


def long_table(results: list[RocResult]) -> pd.DataFrame:
    """One row per protein and predictor."""
    return pd.DataFrame([r.as_row() for r in results])


def wide_table(results: list[RocResult], predictors: list[str]) -> pd.DataFrame:
    """One row per protein, two columns per predictor: the AUC and its error.

    This is the shape to read across a thousand proteins at a glance; the
    long table is the one to plot from.
    """
    if not results:
        return pd.DataFrame(columns=["uniprot_id"])
    rows: dict[str, dict] = {}
    for result in results:
        row = rows.setdefault(result.uniprot_id, {"uniprot_id": result.uniprot_id})
        row[f"{result.predictor}_auc"] = result.auc
        row[f"{result.predictor}_se"] = result.se
        row["n_pathogenic"] = result.n_pathogenic
        row["n_benign"] = result.n_benign
    columns = ["uniprot_id", "n_pathogenic", "n_benign"]
    for name in predictors:
        columns += [f"{name}_auc", f"{name}_se"]
    frame = pd.DataFrame(list(rows.values()))
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame.loc[:, columns].sort_values("uniprot_id").reset_index(drop=True)


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    """Across proteins, per predictor: the median AUC and how often it beats chance."""
    if long.empty:
        return pd.DataFrame(columns=["predictor", "n_proteins", "median_auc"])
    grouped = long.groupby("predictor")["auc"]
    out = pd.DataFrame({
        "n_proteins": grouped.size(),
        "median_auc": grouped.median(),
        "mean_auc": grouped.mean(),
        "q1_auc": grouped.quantile(0.25),
        "q3_auc": grouped.quantile(0.75),
        "fraction_above_chance": grouped.apply(lambda s: float((s > 0.5).mean())),
    }).reset_index()
    return out.sort_values("median_auc", ascending=False).reset_index(drop=True)

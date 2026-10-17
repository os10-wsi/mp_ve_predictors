"""Benchmark predictions against measured values.

Metrics are computed on ``score_harmonized``, so every predictor is oriented
the same way. Assays differ in direction: a biogenesis or abundance readout
usually has higher meaning more functional. ``assay_direction`` states which
way the measured column runs, and correlations are reported in the orientation
where a good predictor scores positively.

Everything here is pure numpy, so no scipy or scikit-learn is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vepbench.schema import HARMONIZED_SUFFIX

log = logging.getLogger(__name__)

HIGHER_IS_MORE_FUNCTIONAL = "higher_is_more_functional"
HIGHER_IS_MORE_DAMAGING = "higher_is_more_damaging"


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denominator) if denominator > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    return pearson(_rank(x), _rank(y))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve by the rank-sum identity, ties handled."""
    positives = labels.astype(bool)
    n_positive, n_negative = int(positives.sum()), int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = _rank(scores)
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def ndcg(scores: np.ndarray, relevance: np.ndarray, k: int | None = None) -> float:
    """Normalised discounted cumulative gain.

    Relevance is shifted to be non-negative, the convention ProteinGym uses.
    """
    if len(scores) < 2:
        return float("nan")
    gains = relevance - np.nanmin(relevance)
    if np.nanmax(gains) == 0:
        return float("nan")
    k = k or len(scores)
    order = np.argsort(-scores, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    actual = float((gains[order] * discounts).sum())
    ideal_order = np.argsort(-gains, kind="mergesort")[:k]
    ideal = float((gains[ideal_order] * discounts).sum())
    return actual / ideal if ideal > 0 else float("nan")


def top_k_recall(scores: np.ndarray, relevance: np.ndarray, k: int = 100) -> float:
    """Fraction of the truly top-k variants recovered in the predicted top k."""
    k = min(k, len(scores))
    if k == 0:
        return float("nan")
    predicted = set(np.argsort(-scores, kind="mergesort")[:k].tolist())
    actual = set(np.argsort(-relevance, kind="mergesort")[:k].tolist())
    return len(predicted & actual) / k


@dataclass
class MetricResult:
    predictor: str
    n: int
    spearman: float
    pearson: float
    auroc: float
    ndcg: float
    top_k_recall: float
    spearman_ci: tuple[float, float] = (float("nan"), float("nan"))
    pearson_ci: tuple[float, float] = (float("nan"), float("nan"))
    auroc_ci: tuple[float, float] = (float("nan"), float("nan"))
    group: str = "pooled"

    def as_row(self) -> dict:
        return {
            "group": self.group, "predictor": self.predictor, "n_variants": self.n,
            "spearman": self.spearman,
            "spearman_ci_low": self.spearman_ci[0],
            "spearman_ci_high": self.spearman_ci[1],
            "pearson": self.pearson,
            "pearson_ci_low": self.pearson_ci[0],
            "pearson_ci_high": self.pearson_ci[1],
            "auroc": self.auroc,
            "auroc_ci_low": self.auroc_ci[0],
            "auroc_ci_high": self.auroc_ci[1],
            "ndcg": self.ndcg, "top_k_recall": self.top_k_recall,
        }


def _bootstrap_ci(
    function, x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    if n_boot <= 0 or len(x) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    size = len(x)
    for i in range(n_boot):
        index = rng.integers(0, size, size)
        values[i] = function(x[index], y[index])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def evaluate(
    merged: pd.DataFrame, score_col: str, *, predictors: list[str] | None = None,
    assay_direction: str = HIGHER_IS_MORE_FUNCTIONAL,
    binarize_threshold: float | None = None, binarize_quantile: float | None = 0.2,
    n_boot: int = 1000, seed: int = 0, top_k: int = 100, group: str = "pooled",
) -> list[MetricResult]:
    """Score every predictor column in ``merged`` against the measured column."""
    if score_col not in merged.columns:
        raise ValueError(f"measured column {score_col!r} not found in the merged table")

    columns = predictors or [c for c in merged.columns if c.endswith(HARMONIZED_SUFFIX)]
    if not columns:
        raise ValueError(
            "no harmonized predictor columns found; expected columns ending in "
            f"{HARMONIZED_SUFFIX!r}"
        )

    measured_raw = pd.to_numeric(merged[score_col], errors="coerce").to_numpy(dtype=float)
    if assay_direction == HIGHER_IS_MORE_FUNCTIONAL:
        damage = -measured_raw
    elif assay_direction == HIGHER_IS_MORE_DAMAGING:
        damage = measured_raw
    else:
        raise ValueError(f"unknown assay_direction {assay_direction!r}")

    results: list[MetricResult] = []
    for column in columns:
        predicted = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(predicted) & np.isfinite(damage)
        n = int(keep.sum())
        if n < 3:
            log.warning("%s: only %d usable rows, skipping", column, n)
            continue
        p, d = predicted[keep], damage[keep]

        if binarize_threshold is not None:
            labels = (
                (measured_raw[keep] <= binarize_threshold)
                if assay_direction == HIGHER_IS_MORE_FUNCTIONAL
                else (measured_raw[keep] >= binarize_threshold)
            )
        elif binarize_quantile is not None:
            labels = d >= np.quantile(d, 1 - binarize_quantile)
        else:
            labels = np.zeros(n, dtype=bool)

        name = column[: -len(HARMONIZED_SUFFIX)] if column.endswith(HARMONIZED_SUFFIX) else column
        results.append(MetricResult(
            predictor=name, n=n,
            spearman=spearman(p, d), pearson=pearson(p, d),
            auroc=auroc(p, labels), ndcg=ndcg(p, d),
            top_k_recall=top_k_recall(p, d, k=top_k),
            spearman_ci=_bootstrap_ci(spearman, p, d, n_boot=n_boot, seed=seed),
            pearson_ci=_bootstrap_ci(pearson, p, d, n_boot=n_boot, seed=seed),
            auroc_ci=_bootstrap_ci(
                lambda a, b: auroc(a, b.astype(bool)), p, labels.astype(float),
                n_boot=n_boot, seed=seed,
            ),
            group=group,
        ))
    return sorted(results, key=lambda r: (-r.spearman if np.isfinite(r.spearman) else 0))


def evaluate_by_group(
    merged: pd.DataFrame, score_col: str, group_col: str | None = None, **kwargs
) -> pd.DataFrame:
    """Per-protein and pooled metrics in one table."""
    rows = [r.as_row() for r in evaluate(merged, score_col, group="pooled", **kwargs)]
    if group_col and group_col in merged.columns:
        for key, subset in merged.groupby(group_col):
            rows.extend(r.as_row() for r in evaluate(subset, score_col, group=str(key), **kwargs))
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_path: Path, group: str = "pooled") -> Path | None:
    """Horizontal bar chart of Spearman with bootstrap intervals.

    Returns None when matplotlib is absent, rather than failing the run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib is not installed; skipping the plot "
                    "(pip install 'vepbench[bench]')")
        return None

    subset = summary.loc[summary["group"] == group].dropna(subset=["spearman"])
    if subset.empty:
        log.warning("nothing to plot for group %r", group)
        return None
    subset = subset.sort_values("spearman")

    figure, axis = plt.subplots(figsize=(8.0, max(3.0, 0.32 * len(subset) + 1.2)))
    y = np.arange(len(subset))
    low = (subset["spearman"] - subset["spearman_ci_low"]).clip(lower=0)
    high = (subset["spearman_ci_high"] - subset["spearman"]).clip(lower=0)
    axis.barh(y, subset["spearman"], color="#4C72B0", height=0.7)
    axis.errorbar(subset["spearman"], y, xerr=np.vstack([low, high]), fmt="none",
                  ecolor="#333333", elinewidth=1, capsize=3)
    axis.set_yticks(y)
    axis.set_yticklabels(subset["predictor"], fontsize=9)
    axis.axvline(0, color="#888888", linewidth=0.8)
    axis.set_xlabel("Spearman correlation with the measured values "
                    "(positive means the predictor is right)")
    axis.set_title(f"Predictor performance ({group})", fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path

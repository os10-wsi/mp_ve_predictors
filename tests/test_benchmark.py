"""Metric correctness and orientation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vepbench.benchmark import (
    HIGHER_IS_MORE_DAMAGING,
    auroc,
    evaluate,
    evaluate_by_group,
    ndcg,
    pearson,
    spearman,
    top_k_recall,
)


def test_pearson_extremes():
    x = np.arange(50, dtype=float)
    assert pearson(x, x) == pytest.approx(1.0)
    assert pearson(x, -x) == pytest.approx(-1.0)


def test_spearman_is_rank_based():
    x = np.arange(50, dtype=float)
    assert spearman(x, np.exp(x / 10)) == pytest.approx(1.0)


def test_spearman_handles_ties():
    values = np.array([1.0, 1.0, 2.0, 3.0])
    assert spearman(values, values) == pytest.approx(1.0)


def test_auroc_matches_a_known_value():
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.75)
    assert auroc(-scores, labels) == pytest.approx(0.25)


def test_auroc_is_nan_without_both_classes():
    scores = np.array([1.0, 2.0, 3.0])
    assert np.isnan(auroc(scores, np.zeros(3)))
    assert np.isnan(auroc(scores, np.ones(3)))


def test_ndcg_is_one_for_a_perfect_ranking():
    relevance = np.array([3.0, 2.0, 1.0, 0.0])
    assert ndcg(relevance, relevance) == pytest.approx(1.0)


def test_top_k_recall():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    relevance = np.array([5.0, 4.0, 0.0, 0.0, 0.0])
    assert top_k_recall(scores, relevance, k=2) == pytest.approx(1.0)
    assert top_k_recall(-scores, relevance, k=2) == pytest.approx(0.0)


def _toy(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "fitness": 0.8 * x + 0.6 * rng.normal(size=n),
        # harmonized: higher means more damaging, so a good predictor is -x
        "good__harmonized": -x, "bad__harmonized": x,
        "noise__harmonized": rng.normal(size=n),
    })


def test_orientation_for_a_functional_assay():
    results = {r.predictor: r for r in evaluate(_toy(), "fitness", n_boot=0)}
    assert results["good"].spearman > 0.5
    assert results["bad"].spearman < -0.5
    assert abs(results["noise"].spearman) < 0.2
    assert results["good"].auroc > 0.7
    assert results["bad"].auroc < 0.3


def test_orientation_for_a_damaging_assay():
    frame = _toy()
    frame["damage"] = -frame["fitness"]
    results = {r.predictor: r for r in
               evaluate(frame, "damage", n_boot=0,
                        assay_direction=HIGHER_IS_MORE_DAMAGING)}
    assert results["good"].spearman > 0.5


def test_bootstrap_interval_brackets_the_estimate():
    good = {r.predictor: r for r in evaluate(_toy(), "fitness", n_boot=300)}["good"]
    low, high = good.spearman_ci
    assert low < good.spearman < high
    assert high - low < 0.4


def test_explicit_threshold_is_used():
    results = evaluate(_toy(), "fitness", n_boot=0, binarize_threshold=0.0)
    assert all(np.isfinite(r.auroc) for r in results)


def test_grouped_evaluation_includes_pooled_and_each_group():
    frame = _toy()
    frame["uniprot_id"] = ["P1"] * 150 + ["P2"] * 150
    summary = evaluate_by_group(frame, "fitness", group_col="uniprot_id", n_boot=0)
    assert set(summary["group"]) == {"pooled", "P1", "P2"}
    assert set(summary["predictor"]) == {"good", "bad", "noise"}


def test_missing_measured_column_is_an_error():
    with pytest.raises(ValueError, match="not found"):
        evaluate(_toy(), "absent", n_boot=0)


def test_no_harmonized_columns_is_an_error():
    with pytest.raises(ValueError, match="harmonized"):
        evaluate(pd.DataFrame({"fitness": [1.0, 2.0, 3.0]}), "fitness", n_boot=0)

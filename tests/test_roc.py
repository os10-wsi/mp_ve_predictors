"""AUC, its error, and the curve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vepbench.roc import (
    auc_score,
    evaluate,
    hanley_mcneil_se,
    long_table,
    roc_curve,
    summarise,
    wide_table,
)


class TestAuc:
    def test_a_perfect_ranking_scores_one(self):
        assert auc_score(np.array([4.0, 3, 2, 1]),
                         np.array([1, 1, 0, 0], bool)) == 1.0

    def test_a_reversed_ranking_scores_zero(self):
        assert auc_score(np.array([4.0, 3, 2, 1]),
                         np.array([0, 0, 1, 1], bool)) == 0.0

    def test_all_ties_score_exactly_chance(self):
        assert auc_score(np.ones(4), np.array([1, 1, 0, 0], bool)) == 0.5

    def test_it_agrees_with_the_pairwise_definition(self):
        # The rank-sum identity is an optimisation, not an approximation.
        rng = np.random.default_rng(0)
        scores = rng.normal(size=80)
        positive = rng.random(80) < 0.4
        pairwise = np.mean([(a > b) + 0.5 * (a == b)
                            for a in scores[positive] for b in scores[~positive]])
        assert auc_score(scores, positive) == pytest.approx(pairwise)

    def test_one_class_alone_has_no_auc(self):
        assert np.isnan(auc_score(np.array([1.0, 2.0]), np.array([1, 1], bool)))


class TestError:
    def test_the_error_shrinks_as_the_sample_grows(self):
        wide = hanley_mcneil_se(0.8, 10, 10)
        narrow = hanley_mcneil_se(0.8, 1000, 1000)
        assert wide > narrow > 0

    def test_a_perfect_auc_has_no_spread(self):
        assert hanley_mcneil_se(1.0, 50, 50) == pytest.approx(0.0, abs=1e-9)

    def test_an_undefined_auc_has_an_undefined_error(self):
        assert np.isnan(hanley_mcneil_se(float("nan"), 10, 10))

    def test_the_interval_cannot_leave_the_unit_range(self):
        from vepbench.roc import RocResult

        result = RocResult("P1", "x", 6, 6, 0.99, 0.2, np.array([0.]), np.array([0.]))
        low, high = result.ci
        assert 0.0 <= low <= high <= 1.0


class TestCurve:
    def test_a_perfect_separation_reaches_the_corner(self):
        fpr, tpr = roc_curve(np.array([4.0, 3, 2, 1]),
                             np.array([1, 1, 0, 0], bool))
        assert tpr[np.argmin(np.abs(fpr))] == pytest.approx(0.0)
        assert tpr.max() == pytest.approx(1.0)
        assert fpr.max() == pytest.approx(1.0)

    def test_the_curve_is_monotone(self):
        rng = np.random.default_rng(1)
        fpr, tpr = roc_curve(rng.normal(size=50), rng.random(50) < 0.5)
        assert (np.diff(fpr) >= -1e-12).all()
        assert (np.diff(tpr) >= -1e-12).all()

    def test_the_area_under_the_drawn_curve_matches_the_auc(self):
        rng = np.random.default_rng(2)
        scores, positive = rng.normal(size=300), rng.random(300) < 0.4
        fpr, tpr = roc_curve(scores, positive)
        assert np.trapezoid(tpr, fpr) == pytest.approx(auc_score(scores, positive),
                                                       abs=1e-9)


def labelled(n=60, seed=0, skill=1.5, predictors=("good", "useless")):
    rng = np.random.default_rng(seed)
    label = (rng.random(n) < 0.5).astype(int)
    frame = pd.DataFrame({"label": label})
    frame["good"] = label * rng.normal(skill, 0.5, n) + rng.normal(0, 1, n)
    frame["useless"] = rng.normal(0, 1, n)
    return frame.loc[:, ["label", *predictors]]


class TestEvaluate:
    def test_a_predictor_tracking_the_label_beats_one_that_does_not(self):
        results = {r.predictor: r for r in
                   evaluate(labelled(300), ["good", "useless"], uniprot_id="P1")}
        assert results["good"].auc > 0.8
        assert 0.35 < results["useless"].auc < 0.65

    def test_a_protein_with_one_class_is_skipped(self):
        frame = pd.DataFrame({"label": [1] * 40, "good": np.arange(40.0)})
        assert evaluate(frame, ["good"], uniprot_id="P1") == []

    def test_too_few_of_either_class_is_skipped(self):
        frame = labelled(40)
        frame.loc[frame.index[2:], "label"] = 1
        assert evaluate(frame, ["good"], uniprot_id="P1", min_per_class=5) == []

    def test_a_predictor_absent_from_the_table_is_skipped_not_invented(self):
        assert evaluate(labelled(100), ["missing"], uniprot_id="P1") == []

    def test_a_predictor_of_only_nulls_is_skipped(self):
        frame = labelled(100)
        frame["good"] = np.nan
        assert [r.predictor for r in
                evaluate(frame, ["good", "useless"], uniprot_id="P1")] == ["useless"]

    def test_the_counts_reported_are_the_scored_ones(self):
        frame = labelled(100)
        frame.loc[frame.index[:20], "good"] = np.nan
        result = evaluate(frame, ["good"], uniprot_id="P1")[0]
        assert result.n_pathogenic + result.n_benign == 80


class TestTables:
    def _results(self):
        out = []
        for index, accession in enumerate(["P2", "P1"]):
            out += evaluate(labelled(200, seed=index), ["good", "useless"],
                            uniprot_id=accession)
        return out

    def test_the_long_table_has_a_row_per_protein_and_predictor(self):
        frame = long_table(self._results())
        assert len(frame) == 4
        assert {"uniprot_id", "predictor", "auc", "se", "ci_low", "ci_high"} <= set(
            frame.columns)

    def test_the_wide_table_has_one_row_per_protein(self):
        # This is the shape asked for: a thousand proteins, a thousand rows.
        frame = wide_table(self._results(), ["good", "useless"])
        assert len(frame) == 2
        assert list(frame["uniprot_id"]) == ["P1", "P2"]
        for name in ("good", "useless"):
            assert f"{name}_auc" in frame.columns
            assert f"{name}_se" in frame.columns

    def test_a_predictor_absent_everywhere_still_gets_its_columns(self):
        frame = wide_table(self._results(), ["good", "useless", "never_ran"])
        assert frame["never_ran_auc"].isna().all()

    def test_the_summary_ranks_predictors_across_proteins(self):
        summary = summarise(long_table(self._results()))
        assert summary.iloc[0]["predictor"] == "good"
        assert summary.iloc[0]["n_proteins"] == 2
        assert 0.0 <= summary.iloc[0]["fraction_above_chance"] <= 1.0

    def test_summarising_nothing_returns_an_empty_frame(self):
        assert summarise(pd.DataFrame()).empty
        assert wide_table([], ["good"]).empty

"""Aggregating per-protein outputs, joining ClinVar, and scoring the result."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vepbench import aggregate, proteome_stats
from vepbench.schema import HARMONIZED_SUFFIX


def write_protein(paths, accession, predictors, n=12, with_status=True):
    """A small wide table for one protein, as the runner would leave it."""
    rng = np.random.default_rng(abs(hash(accession)) % 1000)
    frame = pd.DataFrame({
        "uniprot_id": accession,
        "position": np.arange(1, n + 1),
        "wt_aa": list("ACDEFGHIKLMN")[:n],
        "mut_aa": list("WYVTSRQPNMLK")[:n],
    })
    frame["variant"] = [f"{w}{p}{m}" for w, p, m in
                        zip(frame["wt_aa"], frame["position"], frame["mut_aa"], strict=True)]
    for name in predictors:
        frame[f"{name}{HARMONIZED_SUFFIX}"] = rng.normal(size=n)
    frame["rsasa"] = rng.random(n)
    frame["in_transmembrane"] = rng.random(n) < 0.5
    directory = paths.output_dir(accession)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "predictions_wide.csv", index=False)
    if with_status:
        (directory / "status.json").write_text(json.dumps({
            "uniprot_id": accession,
            "predictors": [{"predictor": n_, "status": "ok"} for n_ in predictors],
        }))
    return frame


class TestBuildDataset:
    def test_every_run_protein_lands_in_the_dataset(self, paths):
        write_protein(paths, "P00001", ["esm2_650m"])
        write_protein(paths, "P00002", ["esm2_650m"])
        coverage = aggregate.build_dataset(["P00001", "P00002"], paths=paths)
        assert coverage.proteins_with_output == 2
        assert coverage.per_predictor["esm2_650m"] == 2

    def test_a_protein_that_never_ran_is_named_not_assumed(self, paths):
        write_protein(paths, "P00001", ["esm2_650m"])
        coverage = aggregate.build_dataset(["P00001", "P00002"], paths=paths)
        assert coverage.missing == ["P00002"]
        assert coverage.proteins_expected == 2
        assert "P00002" in coverage.summary()

    def test_coverage_counts_each_predictor_separately(self, paths):
        write_protein(paths, "P00001", ["esm2_650m", "ankh"])
        write_protein(paths, "P00002", ["esm2_650m"])
        coverage = aggregate.build_dataset(["P00001", "P00002"], paths=paths)
        assert coverage.per_predictor == {"esm2_650m": 2, "ankh": 1}

    def test_a_predictor_column_of_only_nulls_does_not_count_as_coverage(self, paths):
        frame = write_protein(paths, "P00001", ["esm2_650m", "ankh"])
        frame[f"ankh{HARMONIZED_SUFFIX}"] = np.nan
        frame.to_csv(paths.output_dir("P00001") / "predictions_wide.csv", index=False)
        coverage = aggregate.build_dataset(["P00001"], paths=paths)
        assert "ankh" not in coverage.per_predictor

    def test_failed_predictors_are_recorded_per_protein(self, paths):
        write_protein(paths, "P00001", ["esm2_650m"])
        (paths.output_dir("P00001") / "status.json").write_text(json.dumps({
            "predictors": [{"predictor": "rasp", "status": "failed"},
                           {"predictor": "esm2_650m", "status": "ok"}],
        }))
        coverage = aggregate.build_dataset(["P00001"], paths=paths)
        assert coverage.failed == {"P00001": ["rasp"]}

    def test_rerunning_after_more_tasks_land_picks_up_the_new_proteins(self, paths):
        write_protein(paths, "P00001", ["esm2_650m"])
        aggregate.build_dataset(["P00001", "P00002"], paths=paths)
        write_protein(paths, "P00002", ["esm2_650m"])
        coverage = aggregate.build_dataset(["P00001", "P00002"], paths=paths)
        assert coverage.proteins_with_output == 2
        assert coverage.missing == []

    def test_an_unreadable_status_file_does_not_stop_aggregation(self, paths):
        write_protein(paths, "P00001", ["esm2_650m"])
        (paths.output_dir("P00001") / "status.json").write_text("{ truncated")
        assert aggregate.build_dataset(["P00001"], paths=paths).proteins_with_output == 1


class TestJoinClinVar:
    def _clinvar(self, accession="P00001"):
        return pd.DataFrame({
            "uniprot_id": [accession, accession],
            "position": [1, 2], "wt_aa": ["A", "C"], "mut_aa": ["W", "Y"],
            "variant": ["A1W", "C2Y"], "label": pd.array([1, 0], dtype="Int64"),
        })

    def test_predictions_and_features_reach_the_variants(self, paths):
        write_protein(paths, "P00001", ["esm2_650m"])
        aggregate.build_dataset(["P00001"], paths=paths)
        joined, coverage = aggregate.join_clinvar(self._clinvar(), paths=paths)
        assert f"esm2_650m{HARMONIZED_SUFFIX}" in joined.columns
        assert "rsasa" in joined.columns
        assert joined[f"esm2_650m{HARMONIZED_SUFFIX}"].notna().all()
        assert coverage.proteins_with_output == 1

    def test_variants_in_a_protein_that_never_ran_are_kept_and_counted(self, paths):
        joined, coverage = aggregate.join_clinvar(self._clinvar("P09999"), paths=paths)
        assert len(joined) == 2
        assert coverage.missing == ["P09999"]

    def test_unlabelled_variants_can_be_excluded(self, paths):
        write_protein(paths, "P00001", ["esm2_650m"])
        aggregate.build_dataset(["P00001"], paths=paths)
        frame = self._clinvar()
        frame.loc[1, "label"] = pd.NA
        joined, _ = aggregate.join_clinvar(frame, paths=paths, labelled_only=True)
        assert joined["variant"].tolist() == ["A1W"]

    def test_an_empty_clinvar_table_returns_empty(self, paths):
        joined, coverage = aggregate.join_clinvar(pd.DataFrame(), paths=paths)
        assert joined.empty and coverage.proteins_expected == 0

    def test_the_join_falls_back_to_the_per_protein_output(self, paths):
        # The per-protein CSVs are the record; the dataset is derived, so a
        # protein aggregated later must still join.
        write_protein(paths, "P00001", ["esm2_650m"])
        joined, coverage = aggregate.join_clinvar(self._clinvar(), paths=paths)
        assert coverage.proteins_with_output == 1
        assert joined[f"esm2_650m{HARMONIZED_SUFFIX}"].notna().all()


class TestSparsePredictors:
    def test_a_predictor_with_too_few_scores_is_dropped_and_named(self):
        frame = pd.DataFrame({
            f"good{HARMONIZED_SUFFIX}": np.arange(100.0),
            f"sparse{HARMONIZED_SUFFIX}": [1.0] + [np.nan] * 99,
        })
        trimmed, dropped = aggregate.drop_empty_predictors(frame, min_rows=50)
        assert dropped == ["sparse"]
        assert f"sparse{HARMONIZED_SUFFIX}" not in trimmed.columns


class TestMetrics:
    def test_average_precision_is_one_for_a_perfect_ranking(self):
        scores = np.array([3.0, 2.0, 1.0, 0.0])
        assert proteome_stats.average_precision(scores, np.array([1, 1, 0, 0])) == 1.0

    def test_average_precision_reflects_a_reversed_ranking(self):
        scores = np.array([3.0, 2.0, 1.0, 0.0])
        reversed_ap = proteome_stats.average_precision(scores, np.array([0, 0, 1, 1]))
        assert reversed_ap < 0.6

    def test_average_precision_is_undefined_without_both_classes(self):
        scores = np.array([1.0, 2.0])
        assert np.isnan(proteome_stats.average_precision(scores, np.array([1, 1])))


class TestLabelledEvaluation:
    def _frame(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        label = rng.integers(0, 2, n)
        return pd.DataFrame({
            "uniprot_id": rng.choice(["P1", "P2", "P3"], n),
            "label": label,
            "in_transmembrane": rng.random(n) < 0.5,
            "rsasa": rng.random(n),
            f"strong{HARMONIZED_SUFFIX}": label + rng.normal(0, 0.4, n),
            f"useless{HARMONIZED_SUFFIX}": rng.normal(0, 1, n),
        })

    def test_a_predictor_tracking_the_label_beats_one_that_does_not(self):
        result = proteome_stats.evaluate_labelled(self._frame(), n_boot=100)
        overall = result.loc[result["stratum"] == "all"].set_index("predictor")
        assert overall.at["strong", "auroc"] > 0.85
        assert 0.4 < overall.at["useless", "auroc"] < 0.6

    def test_every_supported_stratum_is_reported(self):
        result = proteome_stats.evaluate_labelled(self._frame(), n_boot=50)
        assert {"all", "topology", "burial"} <= set(result["stratum"])

    def test_a_stratum_whose_column_is_absent_is_skipped_not_faked(self):
        frame = self._frame().drop(columns=["rsasa"])
        result = proteome_stats.evaluate_labelled(frame, n_boot=50)
        assert "burial" not in set(result["stratum"])

    def test_a_group_without_enough_of_both_classes_is_omitted(self):
        frame = self._frame(n=60)
        frame.loc[frame.index[:58], "label"] = 1
        result = proteome_stats.evaluate_labelled(frame, n_boot=20, min_per_class=5)
        assert result.empty

    def test_clinvar_trained_predictors_are_flagged_in_the_results(self):
        result = proteome_stats.evaluate_labelled(
            self._frame(), n_boot=20, supervised=frozenset({"strong"})
        )
        flags = result.set_index("predictor")["trained_on_clinvar"]
        assert flags.loc["strong"].all()
        assert not flags.loc["useless"].any()

    def test_a_table_with_no_predictor_columns_says_so(self):
        with pytest.raises(proteome_stats.StatsError, match="harmonized"):
            proteome_stats.evaluate_labelled(pd.DataFrame({"label": [0, 1]}))

    def test_alphamissense_is_recognised_as_calibrated_on_clinvar(self):
        assert "alphamissense" in proteome_stats.supervised_predictors()


class TestPerProtein:
    def test_per_protein_auroc_is_reported_for_proteins_with_enough_variants(self):
        rng = np.random.default_rng(1)
        pieces = []
        for accession in ("P1", "P2"):
            label = rng.integers(0, 2, 80)
            pieces.append(pd.DataFrame({
                "uniprot_id": accession, "label": label,
                f"strong{HARMONIZED_SUFFIX}": label + rng.normal(0, 0.4, 80),
            }))
        frame = pd.concat(pieces, ignore_index=True)
        per_protein = proteome_stats.evaluate_per_protein(frame)
        assert sorted(per_protein["uniprot_id"]) == ["P1", "P2"]

        summary = proteome_stats.summarise_per_protein(per_protein)
        assert summary.at[0, "n_proteins"] == 2
        assert summary.at[0, "median_auroc"] > 0.85

    def test_a_protein_with_one_class_only_is_skipped(self):
        frame = pd.DataFrame({
            "uniprot_id": "P1", "label": [1] * 40,
            f"strong{HARMONIZED_SUFFIX}": np.arange(40.0),
        })
        assert proteome_stats.evaluate_per_protein(frame).empty

    def test_summarising_nothing_returns_an_empty_frame_not_an_error(self):
        assert proteome_stats.summarise_per_protein(pd.DataFrame()).empty

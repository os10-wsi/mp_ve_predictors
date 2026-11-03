"""Scoring a protein: caching, the table that gets written, and ClinVar."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vepbench.runner import CACHED, OK, UNAVAILABLE, score_protein, write_protein
from vepbench.scores import resolve

BUILTIN_ONLY = ["blosum62", "delta_volume", "delta_charge", "delta_hydrophobicity"]


@pytest.fixture
def human(paths):
    """P26678 with its cached sequence and structure, as the fixtures hold it."""
    return "P26678"


def run(accession, names, paths, **kwargs):
    return score_protein(accession, resolve(names), paths=paths, offline=True,
                         **kwargs)


class TestScoring:
    def test_every_builtin_score_produces_a_column(self, paths, human):
        result = run(human, BUILTIN_ONLY, paths)
        assert sorted(result.scored) == sorted(BUILTIN_ONLY)
        for name in BUILTIN_ONLY:
            assert name in result.table.columns
            assert result.table[name].notna().any()

    def test_the_table_carries_a_damage_oriented_column_for_each(self, paths, human):
        result = run(human, ["blosum62"], paths)
        # BLOSUM62 runs the other way, so the oriented column is its negation.
        assert (result.table["blosum62__damaging"]
                == -result.table["blosum62"]).all()

    def test_saturation_means_nineteen_substitutions_per_residue(self, paths, human):
        result = run(human, ["blosum62"], paths)
        per_position = result.table.groupby("position").size()
        assert set(per_position) == {19}

    def test_a_structural_score_without_a_structure_is_unavailable(self, paths,
                                                                   monkeypatch):
        monkeypatch.setattr("vepbench.runner.load_structure",
                            lambda *a, **k: (_ for _ in ()).throw(
                                __import__("vepbench.structure", fromlist=["x"])
                                .StructureError("none")))
        result = run("P26678", ["solvent_exposure"], paths)
        outcome = result.outcomes[0]
        assert outcome.status == UNAVAILABLE
        assert "structure" in outcome.message

    def test_alphamissense_is_unavailable_for_a_non_human_protein(self, paths, human,
                                                                  monkeypatch):
        monkeypatch.setattr("vepbench.protein.ProteinContext.is_human",
                            property(lambda self: False))
        result = run(human, ["alphamissense"], paths)
        assert result.outcomes[0].status == UNAVAILABLE
        assert "human" in result.outcomes[0].message


class TestCaching:
    def test_a_second_run_reads_the_cache(self, paths, human):
        run(human, ["blosum62"], paths)
        again = run(human, ["blosum62"], paths)
        assert again.outcomes[0].status == CACHED

    def test_force_recomputes(self, paths, human):
        run(human, ["blosum62"], paths)
        again = run(human, ["blosum62"], paths, force=True)
        assert again.outcomes[0].status == OK

    def test_a_cache_written_under_the_old_name_is_reused(self, paths, human):
        # esm1v was called esm1v_full by the previous pipeline. Re-running a
        # model that already ran would waste GPU-weeks.
        legacy = paths.predictions_cache(human, "esm1v_full")
        legacy.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"uniprot_id": human, "position": [1, 1], "wt_aa": ["M", "M"],
                      "mut_aa": ["A", "C"], "score": [-1.5, -2.5]}).to_csv(
            legacy, index=False)
        result = run(human, ["esm1v"], paths)
        assert result.outcomes[0].status == CACHED
        assert "esm1v_full" in result.outcomes[0].source
        assert result.table["esm1v"].notna().sum() == 2

    def test_a_cache_from_a_different_sequence_is_refused(self, paths, human):
        # The wild-type residue check is what catches a cache belonging to
        # another isoform; no fingerprint from the old pipeline is reproducible.
        legacy = paths.predictions_cache(human, "esm1v_full")
        legacy.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"uniprot_id": human, "position": [1], "wt_aa": ["W"],
                      "mut_aa": ["A"], "score": [-1.5]}).to_csv(legacy, index=False)
        result = run(human, ["esm1v"], paths)
        assert result.outcomes[0].status != CACHED

    def test_an_unreadable_cache_is_ignored_rather_than_fatal(self, paths, human):
        path = paths.predictions_cache(human, "blosum62")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not a csv")
        assert run(human, ["blosum62"], paths).outcomes[0].status == OK


class TestWholeTable:
    def test_scoring_one_name_keeps_the_columns_already_cached(self, paths, human):
        # `--scores blosum62` must not drop the other nine from the file.
        run(human, BUILTIN_ONLY, paths)
        result = run(human, ["blosum62"], paths)
        for name in BUILTIN_ONLY:
            assert name in result.table.columns

    def test_the_written_file_and_status_describe_the_same_run(self, paths, human):
        result = run(human, BUILTIN_ONLY, paths)
        path = write_protein(result, paths=paths)
        assert path.exists()
        table = pd.read_csv(path, sep="\t")
        assert len(table) == len(result.table)
        status = json.loads((path.parent / "status.json").read_text())
        assert status["uniprot_id"] == human
        assert {s["score"] for s in status["scores"]} >= set(BUILTIN_ONLY)


class TestClinVarLabels:
    def _slice(self, paths, accession, rows):
        from vepbench.clinvar_store import slice_path

        path = slice_path(accession, paths=paths)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)

    def test_labels_are_joined_for_a_human_protein(self, paths, human):
        self._slice(paths, human, [
            {"variant": "M1A", "position": 1, "wt_aa": "M", "mut_aa": "A",
             "clinical_significance": "Pathogenic", "label": 1},
            {"variant": "M1C", "position": 1, "wt_aa": "M", "mut_aa": "C",
             "clinical_significance": "Benign", "label": 0},
        ])
        result = run(human, ["blosum62"], paths)
        assert "label" in result.table.columns
        labelled = result.table.loc[result.table["label"].notna()]
        assert sorted(labelled["variant"]) == ["M1A", "M1C"]

    def test_variants_clinvar_has_not_seen_stay_unlabelled(self, paths, human):
        self._slice(paths, human, [
            {"variant": "M1A", "position": 1, "wt_aa": "M", "mut_aa": "A",
             "clinical_significance": "Pathogenic", "label": 1}])
        result = run(human, ["blosum62"], paths)
        assert result.table["label"].isna().sum() == len(result.table) - 1

    def test_a_protein_with_no_slice_has_no_label_column(self, paths, human):
        result = run(human, ["blosum62"], paths)
        assert "label" not in result.table.columns

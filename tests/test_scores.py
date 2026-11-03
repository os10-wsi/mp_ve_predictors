"""The score registry, and the one thing all ten must agree on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vepbench.scores import (
    BUILTIN,
    HUMAN,
    MODEL,
    SCORES,
    STRUCTURE,
    TABLE,
    ScoreError,
    all_names,
    applicable,
    resolve,
)


def frame(pairs):
    wt, mut = zip(*pairs, strict=True)
    return pd.DataFrame({"position": range(1, len(wt) + 1),
                         "wt_aa": list(wt), "mut_aa": list(mut)})


class TestRegistry:
    def test_the_ten_asked_for_are_all_present(self):
        assert set(all_names()) == {
            "alphamissense", "saprot", "esm1v", "foldx", "blosum62",
            "delta_volume", "delta_charge", "delta_hydrophobicity",
            "solvent_exposure", "thermompnn",
        }

    def test_every_score_declares_which_way_it_points(self):
        # They all end up on one ROC curve, so a score that does not say
        # would be silently plotted backwards.
        for score in SCORES.values():
            assert isinstance(score.higher_is_damaging, bool), score.name
            assert score.units, score.name
            assert score.description, score.name

    def test_every_model_score_names_an_environment_and_runner(self):
        for score in SCORES.values():
            if score.kind == MODEL:
                assert score.env and score.runner, score.name

    def test_builtin_scores_carry_a_function_and_need_no_environment(self):
        for score in SCORES.values():
            if score.kind == BUILTIN:
                assert score.compute is not None, score.name
                assert not score.env, score.name

    def test_resolving_nothing_gives_everything(self):
        assert len(resolve(None)) == len(SCORES)
        assert len(resolve(["all"])) == len(SCORES)

    def test_an_unknown_name_lists_what_is_available(self):
        with pytest.raises(ScoreError, match="blosum62"):
            resolve(["nonsense"])

    def test_resolution_keeps_the_order_asked_for(self):
        assert [s.name for s in resolve(["esm1v", "blosum62"])] == ["esm1v", "blosum62"]


class TestOrientation:
    """Harmonizing turns every raw score into 'higher means more damaging'."""

    def test_a_tolerance_score_is_negated(self):
        # BLOSUM62 positive means a favoured substitution, so a high raw
        # score must become a low damage score.
        raw = pd.Series([4.0, -3.0])
        assert SCORES["blosum62"].harmonize(raw).tolist() == [-4.0, 3.0]

    def test_a_damage_score_is_left_alone(self):
        raw = pd.Series([0.9, 0.1])
        assert SCORES["alphamissense"].harmonize(raw).tolist() == [0.9, 0.1]

    def test_harmonizing_preserves_missing_values(self):
        out = SCORES["esm1v"].harmonize(pd.Series([1.0, None]))
        assert out.isna().tolist() == [False, True]

    def test_non_numeric_becomes_missing_rather_than_raising(self):
        out = SCORES["foldx"].harmonize(pd.Series(["oops", "2.0"]))
        assert out.isna().tolist() == [True, False]


class TestBuiltins:
    def test_blosum62_rates_a_conservative_swap_above_a_drastic_one(self):
        values = SCORES["blosum62"].compute(frame([("L", "I"), ("W", "P")]), {})
        assert values.iloc[0] > values.iloc[1]

    @pytest.mark.parametrize("name", ["delta_volume", "delta_charge",
                                      "delta_hydrophobicity"])
    def test_the_deltas_are_magnitudes_not_signed(self, name):
        # Bigger and smaller are both disruptive; the sign would cancel one
        # against the other on a ROC curve.
        both = SCORES[name].compute(frame([("G", "W"), ("W", "G")]), {})
        assert both.iloc[0] == pytest.approx(both.iloc[1])
        assert (both >= 0).all()

    def test_delta_volume_separates_a_large_change_from_a_small_one(self):
        values = SCORES["delta_volume"].compute(frame([("G", "W"), ("I", "L")]), {})
        assert values.iloc[0] > values.iloc[1]

    def test_delta_charge_notices_a_charge_reversal(self):
        values = SCORES["delta_charge"].compute(frame([("D", "K"), ("A", "G")]), {})
        assert values.iloc[0] == pytest.approx(2.0)
        assert values.iloc[1] == pytest.approx(0.0)

    def test_solvent_exposure_is_a_property_of_the_position(self):
        # Every substitution at one residue shares a value; that is the
        # question it asks.
        variants = frame([("A", "V"), ("A", "W")])
        variants["position"] = [7, 7]
        values = SCORES["solvent_exposure"].compute(variants, {"rsasa": {7: 0.12}})
        assert values.tolist() == [0.12, 0.12]

    def test_solvent_exposure_without_a_structure_says_so(self):
        with pytest.raises(ScoreError, match="structure"):
            SCORES["solvent_exposure"].compute(frame([("A", "V")]), {})

    def test_an_unknown_residue_gives_nan_rather_than_a_wrong_number(self):
        values = SCORES["delta_volume"].compute(frame([("X", "V")]), {})
        assert np.isnan(values.iloc[0])


class TestApplicability:
    def test_alphamissense_is_refused_for_a_non_human_protein(self):
        ok, why = applicable(SCORES["alphamissense"], is_human=False,
                             has_structure=True)
        assert not ok and "human" in why

    def test_a_structural_score_is_refused_without_a_structure(self):
        ok, why = applicable(SCORES["saprot"], is_human=True, has_structure=False)
        assert not ok and "structure" in why

    def test_a_sequence_only_score_runs_for_anything(self):
        assert applicable(SCORES["esm1v"], is_human=False, has_structure=False)[0]

    def test_every_human_only_score_is_marked_as_such(self):
        assert HUMAN in SCORES["alphamissense"].needs
        assert all(HUMAN not in s.needs for n, s in SCORES.items()
                   if n != "alphamissense")

    def test_structure_requirements_are_declared(self):
        needs_structure = {n for n, s in SCORES.items() if STRUCTURE in s.needs}
        assert needs_structure == {"solvent_exposure", "saprot", "foldx", "thermompnn"}


class TestScope:
    def test_foldx_is_limited_to_clinvar_positions(self):
        # Saturation FoldX is days per protein and nothing reads the scores
        # away from a ClinVar position.
        assert SCORES["foldx"].default_scope == "clinvar"

    def test_everything_else_runs_at_saturation(self):
        assert all(s.default_scope == "saturation"
                   for n, s in SCORES.items() if n != "foldx")

    def test_the_two_table_scores_are_the_expected_ones(self):
        assert {n for n, s in SCORES.items() if s.kind == TABLE} == {
            "alphamissense", "thermompnn"}

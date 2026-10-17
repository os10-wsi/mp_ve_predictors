"""Schema, harmonization and the numbering assertion."""

from __future__ import annotations

import pandas as pd
import pytest

from vepbench.schema import (
    LONG_COLUMNS,
    MORE_DELETERIOUS,
    MORE_TOLERATED,
    SchemaError,
    ScoreConvention,
    build_long_frame,
    harmonize,
    to_wide,
    validate_long_frame,
)


def _frame():
    return pd.DataFrame({"position": [1, 2], "wt_aa": ["M", "E"],
                         "mut_aa": ["A", "A"], "score": [-2.0, 1.5]})


def test_harmonize_flips_only_tolerated_scores():
    assert harmonize(2.5, MORE_DELETERIOUS) == 2.5
    assert harmonize(2.5, MORE_TOLERATED) == -2.5
    assert harmonize(-1.0, MORE_DELETERIOUS) == -1.0
    assert harmonize(-1.0, MORE_TOLERATED) == 1.0


def test_harmonize_rejects_unknown_direction():
    with pytest.raises(SchemaError):
        harmonize(1.0, "sideways")


def test_convention_rejects_unknown_direction():
    with pytest.raises(SchemaError):
        ScoreConvention("x", "sideways", True, "test")


def test_build_long_frame_shape_and_harmonization():
    out = build_long_frame(
        uniprot_id="P1", frame=_frame(), predictor="demo",
        convention=ScoreConvention("log_likelihood_ratio", MORE_TOLERATED, True, "t"),
        model_version="v1",
    )
    assert list(out.columns) == LONG_COLUMNS
    assert out["variant"].tolist() == ["M1A", "E2A"]
    assert out["score_harmonized"].tolist() == [2.0, -1.5]
    assert (out["score"] == [-2.0, 1.5]).all()


def test_build_long_frame_requires_columns():
    with pytest.raises(SchemaError, match="missing columns"):
        build_long_frame(
            uniprot_id="P1", frame=pd.DataFrame({"position": [1]}), predictor="d",
            convention=ScoreConvention("x", MORE_TOLERATED, True, "t"),
            model_version="v1",
        )


def _long():
    return build_long_frame(
        uniprot_id="P1", frame=_frame(), predictor="d",
        convention=ScoreConvention("x", MORE_TOLERATED, True, "t"), model_version="v1",
    )


def test_validate_catches_wild_type_mismatch():
    validate_long_frame(_long(), "ME")
    with pytest.raises(SchemaError, match="disagree with the UniProt sequence"):
        validate_long_frame(_long(), "MA")


def test_validate_catches_out_of_range():
    with pytest.raises(SchemaError, match="outside the sequence"):
        validate_long_frame(_long(), "M")


def test_to_wide_has_raw_and_harmonized_blocks():
    one = _long()
    two = build_long_frame(
        uniprot_id="P1", frame=_frame(), predictor="b",
        convention=ScoreConvention("x", MORE_TOLERATED, True, "t"), model_version="v1",
    )
    wide = to_wide(pd.concat([one, two], ignore_index=True))
    assert {"d", "b", "d__harmonized", "b__harmonized"} <= set(wide.columns)
    assert len(wide) == 2


def test_to_wide_joins_features():
    features = pd.DataFrame({"position": [1, 2], "wt_aa": ["M", "E"],
                             "mut_aa": ["A", "A"], "rsasa": [0.5, 0.2]})
    assert to_wide(_long(), features)["rsasa"].tolist() == [0.5, 0.2]

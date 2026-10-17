"""Joining predictions onto a DMS table, including every failure mode."""

from __future__ import annotations

import pandas as pd
import pytest

from vepbench.merge import (
    MATCHED,
    MULTI_MUTANT,
    NO_PREDICTION,
    NONSENSE,
    OUT_OF_RANGE,
    SYNONYMOUS,
    UNPARSEABLE,
    WT_MISMATCH,
    MergeError,
    merge_with_dms,
)

SEQUENCE = "MKVLA"


@pytest.fixture
def wide():
    return pd.DataFrame({
        "uniprot_id": ["P1"] * 3, "position": [1, 2, 3], "wt_aa": ["M", "K", "V"],
        "mut_aa": ["A", "A", "A"], "variant": ["M1A", "K2A", "V3A"],
        "demo": [1.0, 2.0, 3.0], "demo__harmonized": [-1.0, -2.0, -3.0],
    })


def test_every_failure_mode_is_labelled(wide):
    dms = pd.DataFrame({
        "variant": ["M1A", "p.Lys2Ala", "V3A", "Q4A", "M1A,K2A", "rubbish",
                    "A9G", "K2*", "M1M", "L4A"],
        "fitness": range(10),
    })
    merged, report = merge_with_dms(dms, wide, SEQUENCE, variant_col="variant")
    assert merged["match_status"].tolist() == [
        MATCHED, MATCHED, MATCHED, WT_MISMATCH, MULTI_MUTANT, UNPARSEABLE,
        OUT_OF_RANGE, NONSENSE, SYNONYMOUS, NO_PREDICTION,
    ]
    assert report.matched == 3
    assert report.unmatched == 7


def test_nothing_is_dropped_by_default(wide):
    dms = pd.DataFrame({"variant": ["M1A", "rubbish"], "fitness": [1.0, 2.0]})
    merged, _ = merge_with_dms(dms, wide, SEQUENCE, variant_col="variant")
    assert len(merged) == len(dms)


def test_drop_unmatched_when_asked(wide):
    dms = pd.DataFrame({"variant": ["M1A", "rubbish"], "fitness": [1.0, 2.0]})
    merged, _ = merge_with_dms(dms, wide, SEQUENCE, variant_col="variant",
                               keep_unmatched=False)
    assert merged["match_status"].tolist() == [MATCHED]


def test_original_columns_are_preserved(wide):
    dms = pd.DataFrame({"variant": ["M1A"], "fitness": [0.42], "replicate": ["r1"]})
    merged, _ = merge_with_dms(dms, wide, SEQUENCE, variant_col="variant")
    assert merged["fitness"].iloc[0] == 0.42
    assert merged["replicate"].iloc[0] == "r1"
    assert merged["demo"].iloc[0] == 1.0


def test_separate_columns_including_three_letter_codes(wide):
    dms = pd.DataFrame({"pos": [1, 2, 99], "wt": ["M", "Lys", "A"],
                        "mut": ["A", "Ala", "G"], "score": [0.1, 0.2, 0.3]})
    merged, report = merge_with_dms(dms, wide, SEQUENCE, position_col="pos",
                                    wt_col="wt", mut_col="mut")
    assert merged["variant_normalised"].tolist() == ["M1A", "K2A", "A99G"]
    assert report.matched == 2
    assert merged["match_status"].iloc[2] == OUT_OF_RANGE


def test_wt_mismatch_message_names_both_residues(wide):
    merged, _ = merge_with_dms(pd.DataFrame({"variant": ["Q4A"], "fitness": [1.0]}),
                               wide, SEQUENCE, variant_col="variant")
    detail = merged["match_detail"].iloc[0]
    assert "Q" in detail and "L" in detail and "4" in detail


def test_missing_column_is_an_error(wide):
    with pytest.raises(MergeError, match="not found"):
        merge_with_dms(pd.DataFrame({"v": ["M1A"]}), wide, SEQUENCE,
                       variant_col="variant")


def test_requires_a_column_specification(wide):
    with pytest.raises(MergeError, match="--variant-col"):
        merge_with_dms(pd.DataFrame({"v": ["M1A"]}), wide, SEQUENCE)


def test_report_summary_mentions_each_failure(wide):
    dms = pd.DataFrame({"variant": ["M1A", "rubbish", "Q4A"], "fitness": [1, 2, 3]})
    summary = merge_with_dms(dms, wide, SEQUENCE, variant_col="variant")[1].summary()
    assert "1/3" in summary
    assert UNPARSEABLE in summary
    assert WT_MISMATCH in summary

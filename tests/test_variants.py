"""Variant enumeration and notation parsing."""

from __future__ import annotations

import pytest

from vepbench.variants import (
    VariantParseError,
    check_against_sequence,
    parse_variant,
    saturation_variants,
)


@pytest.mark.parametrize("text,expected", [
    ("A123G", "A123G"), ("p.A123G", "A123G"), ("p.Ala123Gly", "A123G"),
    ("Ala123Gly", "A123G"), ("a123g", "A123G"), ("  A123G  ", "A123G"),
    ("p.Met1Ter", "M1*"), ("M1*", "M1*"),
])
def test_parse_accepted_notations(text, expected):
    assert parse_variant(text).variant == expected


@pytest.mark.parametrize("text", [
    "A123G,B45C", "A123G;B45C", "A123G+B45C", "nonsense", "p.Xyz12Gly",
    "A0G", "", None,
])
def test_parse_rejects(text):
    with pytest.raises(VariantParseError):
        parse_variant(text)


def test_multi_mutant_is_distinguishable():
    with pytest.raises(VariantParseError, match="multi-mutant"):
        parse_variant("A1G,C2D")


def test_saturation_is_complete(sequence):
    frame = saturation_variants(sequence)
    assert len(frame) == len(sequence) * 19
    assert frame["position"].nunique() == len(sequence)
    assert not (frame["wt_aa"] == frame["mut_aa"]).any()
    for _position, group in frame.groupby("position"):
        assert len(group) == 19
        assert group["mut_aa"].nunique() == 19


def test_saturation_with_wild_type(sequence):
    frame = saturation_variants(sequence, include_wt=True)
    assert len(frame) == len(sequence) * 20
    assert frame["is_wt"].sum() == len(sequence)


def test_saturation_wild_type_matches_sequence(sequence):
    assert check_against_sequence(saturation_variants(sequence), sequence).all()


def test_saturation_subset_of_positions(sequence):
    assert len(saturation_variants(sequence, positions=[1, 2, 3])) == 3 * 19


def test_saturation_rejects_nonstandard_residues():
    with pytest.raises(ValueError, match="non-standard"):
        saturation_variants("MKVLX")


def test_saturation_rejects_out_of_range_position(sequence):
    with pytest.raises(ValueError, match="outside the sequence"):
        saturation_variants(sequence, positions=[len(sequence) + 1])

"""Per-position and per-variant feature columns."""

from __future__ import annotations

import pytest

from vepbench.constants import KD_HYDROPATHY, blosum62, grantham_distance
from vepbench.features import ALL_FEATURES, build_features
from vepbench.variants import saturation_variants


@pytest.mark.parametrize("wt,mut,expected", [
    ("S", "R", 110), ("L", "P", 98), ("W", "C", 215), ("A", "V", 64), ("D", "E", 45),
])
def test_grantham_matches_published_values(wt, mut, expected):
    assert grantham_distance(wt, mut) == pytest.approx(expected, abs=1.0)


def test_grantham_is_symmetric_and_zero_on_the_diagonal():
    assert grantham_distance("K", "E") == pytest.approx(grantham_distance("E", "K"))
    assert grantham_distance("K", "K") == pytest.approx(0.0)


@pytest.mark.parametrize("wt,mut,expected", [
    ("W", "W", 11), ("A", "A", 4), ("D", "E", 2), ("C", "C", 9), ("W", "D", -4),
])
def test_blosum62_matches_the_published_matrix(wt, mut, expected):
    assert blosum62(wt, mut) == expected


def test_feature_block_covers_every_variant(protein, sequence):
    variants = saturation_variants(sequence)
    frame = build_features(protein, variants, ["physchem", "blosum", "grantham"])
    assert len(frame) == len(variants)
    assert frame["blosum62"].notna().all()
    assert frame["grantham"].notna().all()


def test_physchem_deltas_have_the_right_sign(protein, sequence):
    frame = build_features(protein, saturation_variants(sequence), ["physchem"])
    row = frame.loc[(frame["wt_aa"] == "L") & (frame["mut_aa"] == "D")].iloc[0]
    assert row["delta_hydropathy"] == pytest.approx(
        KD_HYDROPATHY["D"] - KD_HYDROPATHY["L"]
    )
    assert row["delta_charge"] == pytest.approx(-1.0)
    assert row["introduces_charge"]
    assert row["delta_volume"] < 0


def test_proline_flag(protein, sequence):
    frame = build_features(protein, saturation_variants(sequence), ["physchem"])
    assert frame.loc[frame["mut_aa"] == "P", "introduces_proline"].all()
    assert not frame.loc[frame["mut_aa"] != "P", "introduces_proline"].any()


def test_membrane_annotations_follow_uniprot(protein, sequence, uniprot_record):
    frame = build_features(protein, saturation_variants(sequence), ["membrane"])
    start, end = uniprot_record.transmembrane_segments[0]
    assert int(frame["in_transmembrane"].sum()) == (end - start + 1) * 19
    inside = frame.loc[frame["in_transmembrane"]]
    assert inside["position"].min() == start
    assert inside["position"].max() == end
    assert (inside["residues_to_nearest_tm"] == 0).all()


def test_topology_labels_come_through(protein, sequence):
    frame = build_features(protein, saturation_variants(sequence), ["membrane"])
    assert "Cytoplasmic" in set(frame["topology"])


def test_rsasa_present_when_a_structure_is_given(protein, sequence):
    frame = build_features(protein, saturation_variants(sequence), ["rsasa"])
    assert frame["rsasa"].notna().all()
    assert (frame["rsasa"] >= 0).all()


def test_structure_features_are_skipped_without_a_structure(uniprot_record, tmp_path):
    from vepbench.protein import ProteinContext

    protein = ProteinContext(uniprot=uniprot_record, work_dir=tmp_path)
    frame = build_features(protein, saturation_variants(uniprot_record.sequence),
                           ["rsasa", "plddt", "physchem"])
    # skipped, not filled with a placeholder
    assert "rsasa" not in frame.columns
    assert "plddt" not in frame.columns
    assert "delta_volume" in frame.columns


def test_plddt_comes_through_for_a_real_alphafold_model(
    uniprot_record, alphafold_path, paths, tmp_path
):
    from vepbench.protein import ProteinContext
    from vepbench.structure import load_structure

    structure = load_structure("P26678", uniprot_record.sequence, paths=paths)
    assert structure.is_predicted
    protein = ProteinContext(uniprot=uniprot_record, work_dir=tmp_path,
                             structure=structure)
    frame = build_features(protein, saturation_variants(uniprot_record.sequence),
                           ["plddt", "membrane"])
    assert frame["plddt"].notna().all()
    by_segment = frame.groupby("in_transmembrane")["plddt"].mean()
    assert by_segment[True] > by_segment[False]


def test_conservation_features_need_an_alignment(protein, sequence, alignment):
    from vepbench.msa import describe_alignment

    variants = saturation_variants(sequence)
    assert "msa_entropy_bits" not in build_features(protein, variants, ["entropy"]).columns

    protein.msa = describe_alignment(alignment, backend="test")
    frame = build_features(protein, variants, ["entropy", "gaps"])
    assert frame["msa_entropy_bits"].notna().all()
    assert (frame["msa_entropy_bits"] >= 0).all()


def test_unknown_feature_is_rejected(protein, sequence):
    with pytest.raises(ValueError, match="unknown features"):
        build_features(protein, saturation_variants(sequence), ["not_a_feature"])


def test_every_declared_feature_is_accepted(protein, sequence, alignment):
    from vepbench.msa import describe_alignment

    protein.msa = describe_alignment(alignment, backend="test")
    variants = saturation_variants(sequence, positions=[1, 2])
    build_features(protein, variants, list(ALL_FEATURES))

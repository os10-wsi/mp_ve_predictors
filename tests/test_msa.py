"""Alignment parsing, depth statistics, conservation and the EBI conversion."""

from __future__ import annotations

import json

import pytest

from vepbench.msa import (
    LOW_NEFF_THRESHOLD,
    alignment_statistics,
    blast_json_to_a3m,
    describe_alignment,
    hsp_to_a3m_row,
    per_position_entropy,
    per_position_gap_fraction,
    read_a3m,
    read_alignment,
    read_stockholm,
    to_matched_columns,
)


def write(tmp_path, text, name="test.a3m"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_read_a3m(tmp_path):
    path = write(tmp_path, ">query\nMKVLA\n>s1\nMKVLG\n")
    assert read_a3m(path) == [("query", "MKVLA"), ("s1", "MKVLG")]


def test_read_a3m_joins_wrapped_lines(tmp_path):
    assert read_a3m(write(tmp_path, ">query\nMKV\nLA\n"))[0][1] == "MKVLA"


def test_read_stockholm(tmp_path):
    path = write(tmp_path, "# STOCKHOLM 1.0\nquery MKVLA\ns1 MKVLG\n//\n", name="t.sto")
    assert dict(read_stockholm(path)) == {"query": "MKVLA", "s1": "MKVLG"}


def test_read_alignment_dispatches_on_suffix(tmp_path):
    path = write(tmp_path, "# STOCKHOLM 1.0\nq MKVLA\n//\n", name="x.sto")
    assert read_alignment(path)[0][1] == "MKVLA"


def test_insertions_are_dropped_from_matched_columns(tmp_path):
    # lower case marks an insertion relative to the query
    path = write(tmp_path, ">query\nMKVLA\n>s1\nMKqVLA\n")
    assert to_matched_columns(read_a3m(path)).shape[1] == 5


def test_neff_counts_clusters_not_sequences(tmp_path):
    path = write(tmp_path, ">query\nMKVLA\n" + ">a\nMKVLA\n" * 4 + ">b\nWWWWW\n")
    _, neff, _, _ = alignment_statistics(read_a3m(path))
    assert 1.5 < neff < 2.5


def test_identical_sequences_give_neff_one(tmp_path):
    path = write(tmp_path, ">query\nMKVLA\n" + ">a\nMKVLA\n" * 9)
    _, neff, _, _ = alignment_statistics(read_a3m(path))
    assert neff == pytest.approx(1.0)


def test_shallow_alignment_is_flagged(tmp_path):
    record = describe_alignment(write(tmp_path, ">query\nMKVLA\n>s1\nMKVLG\n"),
                                backend="test")
    assert record.is_shallow
    assert record.n_effective < LOW_NEFF_THRESHOLD


def test_diverse_alignment_has_high_neff(alignment):
    record = describe_alignment(alignment, backend="test")
    assert record.n_sequences == 61
    assert record.n_effective > 50


def test_entropy_is_zero_for_a_conserved_column(tmp_path):
    entropy = per_position_entropy(read_a3m(
        write(tmp_path, ">query\nMKVLA\n>a\nMAVLA\n>b\nMDVLA\n")
    ))
    assert entropy[1] == pytest.approx(0.0, abs=1e-6)   # all M
    assert entropy[2] > 1.0                              # K, A, D


def test_gap_fraction(tmp_path):
    gaps = per_position_gap_fraction(read_a3m(
        write(tmp_path, ">query\nMKVLA\n>a\nMK-LA\n>b\nMK-LA\n")
    ))
    assert gaps[3] == pytest.approx(2 / 3)
    assert gaps[1] == pytest.approx(0.0)


# --- the EBI BLAST to A3M conversion ------------------------------------

def test_a3m_row_handles_offset_and_insertion():
    # query columns 3..7 aligned, with one residue inserted in the hit
    assert hsp_to_a3m_row(10, "MK-VL", "MKQVL", 3, 7) == "--MKqVL---"


def test_a3m_row_pads_both_ends():
    assert hsp_to_a3m_row(6, "AB", "CD", 3, 4) == "--CD--"


def test_a3m_row_keeps_gaps_in_the_hit():
    assert hsp_to_a3m_row(4, "ABCD", "A-CD", 1, 4) == "A-CD"


def test_blast_json_becomes_a_valid_a3m(tmp_path):
    sequence = "MKVLAGG"
    payload = {"hits": [
        {"hit_acc": "H1", "hit_uni_de": "first hit", "hit_hsps": [
            {"hsp_bit_score": 40, "hsp_qseq": "MKVLAGG", "hsp_hseq": "MKVLAGG",
             "hsp_query_from": 1, "hsp_query_to": 7},
        ]},
        {"hit_acc": "H2", "hit_uni_de": "partial hit", "hit_hsps": [
            {"hsp_bit_score": 20, "hsp_qseq": "VLA", "hsp_hseq": "VMA",
             "hsp_query_from": 3, "hsp_query_to": 5},
            {"hsp_bit_score": 10, "hsp_qseq": "GG", "hsp_hseq": "AA",
             "hsp_query_from": 6, "hsp_query_to": 7},
        ]},
    ]}
    out = tmp_path / "ebi.a3m"
    assert blast_json_to_a3m(payload, sequence, out) == 2
    records = read_a3m(out)
    assert records[0] == ("query", sequence)
    # only the best-scoring HSP per hit is used
    assert records[2][1] == "--VMA--"
    # every row projects to exactly the query length
    assert to_matched_columns(records).shape == (3, len(sequence))


def test_blast_json_skips_duplicate_accessions(tmp_path):
    payload = {"hits": [
        {"hit_acc": "H1", "hit_hsps": [{"hsp_bit_score": 1, "hsp_qseq": "MK",
                                        "hsp_hseq": "MK", "hsp_query_from": 1,
                                        "hsp_query_to": 2}]},
        {"hit_acc": "H1", "hit_hsps": [{"hsp_bit_score": 1, "hsp_qseq": "MK",
                                        "hsp_hseq": "LL", "hsp_query_from": 1,
                                        "hsp_query_to": 2}]},
    ]}
    assert blast_json_to_a3m(payload, "MK", tmp_path / "x.a3m") == 1


def test_record_dict_round_trips(alignment):
    payload = json.loads(json.dumps(describe_alignment(alignment, backend="t").as_dict()))
    assert payload["backend"] == "t"
    assert payload["n_effective"] > 0


# --- where a local database is found -------------------------------------

def test_explicit_database_wins(monkeypatch):
    from vepbench.msa import resolve_database

    monkeypatch.setenv("VEPBENCH_MSA_DATABASE", "/from/env.fasta")
    assert resolve_database("jackhmmer", "/explicit.fasta") == "/explicit.fasta"


def test_environment_variable_is_used_when_no_flag(monkeypatch):
    from vepbench.msa import resolve_database

    monkeypatch.setenv("VEPBENCH_MSA_DATABASE", "/from/env.fasta")
    assert resolve_database("jackhmmer", None) == "/from/env.fasta"
    assert resolve_database("hhblits", None) == "/from/env.fasta"


def test_conventional_location_is_found(monkeypatch, tmp_path):
    """A database left in data/databases/ needs no flag."""
    import vepbench.msa as msa_module
    from vepbench.config import Paths

    monkeypatch.delenv("VEPBENCH_MSA_DATABASE", raising=False)
    monkeypatch.setenv("VEPBENCH_DATA", str(tmp_path / "data"))
    paths = Paths(root=tmp_path)
    paths.databases.mkdir(parents=True, exist_ok=True)
    (paths.databases / "uniref90.fasta").write_text(">a\nMK\n")
    monkeypatch.setattr(msa_module, "PATHS", paths)
    assert msa_module.resolve_database("jackhmmer", None) == str(
        paths.databases / "uniref90.fasta"
    )


def test_no_database_for_the_remote_backends(monkeypatch):
    from vepbench.msa import resolve_database

    monkeypatch.setenv("VEPBENCH_MSA_DATABASE", "/from/env.fasta")
    # a local database is meaningless for a server-side search
    assert resolve_database("colabfold", None) is None
    assert resolve_database("ebi", None) is None


def test_jackhmmer_without_a_database_explains_the_setup(monkeypatch, tmp_path):
    import vepbench.msa as msa_module
    from vepbench.config import Paths
    from vepbench.msa import MSAError, build_msa

    monkeypatch.delenv("VEPBENCH_MSA_DATABASE", raising=False)
    monkeypatch.setenv("VEPBENCH_DATA", str(tmp_path / "data"))
    paths = Paths(root=tmp_path)
    paths.ensure()
    monkeypatch.setattr(msa_module, "PATHS", paths)
    with pytest.raises(MSAError, match="setup_uniref90"):
        build_msa("P1", "MKVLA", backend="jackhmmer", paths=paths)

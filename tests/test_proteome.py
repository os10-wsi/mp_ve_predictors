"""The manifest, the table helpers and the AlphaMissense lookup."""

from __future__ import annotations

import pandas as pd
import pytest

from vepbench import tables
from vepbench.proteome import (
    QUERIES,
    ProteomeError,
    _parse_row,
    _topology_label,
    load_manifest,
    manifest_directory,
    resolve_query,
)


class TestQueryResolution:
    def test_named_sets_resolve(self):
        name, query = resolve_query("human_membrane")
        assert name == "human_membrane"
        assert "KW-0812" in query

    def test_a_raw_query_is_passed_through(self):
        name, query = resolve_query("(organism_id:10090) AND (reviewed:true)")
        assert name == "custom"
        assert query.startswith("(organism_id")

    def test_an_unknown_bare_name_is_refused_with_the_options(self):
        with pytest.raises(ProteomeError, match="human_membrane"):
            resolve_query("mouse")

    def test_every_named_query_is_restricted_to_reviewed_entries(self):
        # An unreviewed set is two orders of magnitude bigger and full of
        # fragments; running the proteome against it by accident would be
        # expensive and silent.
        for name, query in QUERIES.items():
            assert "reviewed:true" in query, name


class TestManifestParsing:
    def test_transmembrane_segments_are_counted_and_measured(self):
        row = _parse_row({
            "Entry": "A0PK11", "Entry Name": "CLRN2_HUMAN",
            "Gene Names (primary)": "CLRN2", "Protein names": "Clarin-2",
            "Length": "232",
            "Transmembrane": ('TRANSMEM 10..30; /evidence="ECO:0000255"; '
                              'TRANSMEM 101..121; /evidence="ECO:0000255"'),
            "Intramembrane": "",
            "Subcellular location [CC]": "SUBCELLULAR LOCATION: Multi-pass membrane protein",
        })
        assert row["n_tm_segments"] == 2
        assert row["tm_residues"] == 42
        assert row["tm_evidence"] == "predicted"
        assert row["membrane_topology"] == "multipass"

    def test_experimental_evidence_is_distinguished_from_prediction(self):
        row = _parse_row({
            "Entry": "P1", "Length": "100",
            "Transmembrane": 'TRANSMEM 10..30; /evidence="ECO:0000269|PubMed:1"',
            "Intramembrane": "", "Subcellular location [CC]": "",
        })
        assert row["tm_evidence"] == "experimental"

    def test_a_protein_name_keeps_only_the_recommended_name(self):
        row = _parse_row({
            "Entry": "P1", "Length": "10",
            "Protein names": "Elongase 7 (EC 2.3.1.199) (ELOVL 7)",
            "Transmembrane": "", "Intramembrane": "", "Subcellular location [CC]": "",
        })
        assert row["protein_name"] == "Elongase 7"

    def test_topology_falls_back_to_the_segment_count(self):
        assert _topology_label("", 4) == "multipass"
        assert _topology_label("", 1) == "singlepass"
        assert _topology_label("Peripheral membrane protein", 0) == "peripheral"
        assert _topology_label("", 0) == "unclassified"


class TestManifestOnDisk:
    def test_a_missing_manifest_names_the_command_that_makes_one(self, paths):
        with pytest.raises(ProteomeError, match="proteome define"):
            load_manifest("human_membrane", paths=paths)

    def test_a_written_manifest_round_trips(self, paths):
        frame = pd.DataFrame({
            "uniprot_id": ["P26678"], "gene": ["PLN"], "length": [52],
        })
        directory = manifest_directory("human_membrane", paths=paths)
        directory.mkdir(parents=True)
        frame.to_csv(directory / "manifest.tsv", sep="\t", index=False)
        assert load_manifest("human_membrane", paths=paths)["gene"].tolist() == ["PLN"]


class TestTables:
    def test_a_table_round_trips_in_whichever_format_is_available(self, tmp_path):
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        written = tables.write_table(tmp_path / "thing", frame)
        assert written.exists()
        pd.testing.assert_frame_equal(tables.read_table(tmp_path / "thing"), frame)

    def test_reading_a_table_that_was_never_written_names_both_candidates(self, tmp_path):
        with pytest.raises(tables.TableError, match="parquet"):
            tables.read_table(tmp_path / "absent")

    def test_partitions_keep_proteins_separate_and_restore_the_accession(self, tmp_path):
        root = tmp_path / "dataset"
        tables.write_partition(root, "P1", pd.DataFrame({"uniprot_id": ["P1"], "v": [1]}))
        tables.write_partition(root, "P2", pd.DataFrame({"uniprot_id": ["P2"], "v": [2]}))

        assert tables.dataset_members(root) == ["P1", "P2"]
        both = tables.read_dataset(root)
        assert both["uniprot_id"].tolist() == ["P1", "P2"]
        # The accession lives in the directory name, not in every row.
        one = tables.read_dataset(root, ["P2"])
        assert one["uniprot_id"].tolist() == ["P2"] and one["v"].tolist() == [2]

    def test_rewriting_a_partition_replaces_it_rather_than_appending(self, tmp_path):
        root = tmp_path / "dataset"
        tables.write_partition(root, "P1", pd.DataFrame({"v": [1, 2, 3]}))
        tables.write_partition(root, "P1", pd.DataFrame({"v": [9]}))
        assert tables.read_dataset(root)["v"].tolist() == [9]

    def test_an_interrupted_write_is_not_read_back(self, tmp_path):
        root = tmp_path / "dataset"
        tables.write_partition(root, "P1", pd.DataFrame({"v": [1]}))
        (root / "uniprot_id=P1" / "part-0.parquet.tmp").write_bytes(b"half a file")
        assert tables.read_dataset(root)["v"].tolist() == [1]

    def test_an_empty_dataset_reads_as_an_empty_frame(self, tmp_path):
        (tmp_path / "dataset").mkdir()
        assert tables.read_dataset(tmp_path / "dataset").empty


class TestAccessionLists:
    """Building a manifest from a file of accessions rather than a query."""

    def test_blank_lines_and_comments_are_ignored(self, tmp_path):
        from vepbench.proteome import read_accession_list

        path = tmp_path / "ids.txt"
        path.write_text("P00001\n\n# a note\n  P00002  \nP00003\n")
        assert read_accession_list(path) == ["P00001", "P00002", "P00003"]

    def test_duplicates_are_collapsed_keeping_order(self, tmp_path):
        from vepbench.proteome import read_accession_list

        path = tmp_path / "ids.txt"
        path.write_text("P00002\nP00001\nP00002\n")
        assert read_accession_list(path) == ["P00002", "P00001"]

    def test_a_tsv_or_fasta_header_yields_the_accession(self, tmp_path):
        from vepbench.proteome import read_accession_list

        path = tmp_path / "ids.txt"
        path.write_text(">sp|P00001|NAME_YEAST Some protein\nP00002\textra\tcolumns\n")
        assert read_accession_list(path) == ["P00001", "P00002"]

    def test_an_empty_file_is_refused_by_name(self, tmp_path):
        from vepbench.proteome import ProteomeError, read_accession_list

        path = tmp_path / "ids.txt"
        path.write_text("\n# nothing but a comment\n")
        with pytest.raises(ProteomeError, match="no accessions"):
            read_accession_list(path)

    def test_yeast_is_a_named_set(self):
        from vepbench.proteome import QUERIES

        assert "559292" in QUERIES["yeast_membrane"]


class TestBatchVerification:
    """A batch can answer 200 with only part of what was asked for."""

    def test_accessions_missing_from_a_batch_are_retried_individually(
        self, paths, monkeypatch
    ):
        # The real failure seen against UniProt: a batch of 50 came back 200
        # with 2 rows. Trusting that would silently shrink the manifest.
        from vepbench import proteome

        monkeypatch.setattr(proteome, "ACCESSION_BATCH", 100)
        calls: list[list[str]] = []

        def flaky(batch, *, timeout):
            calls.append(list(batch))
            if len(batch) > 1:
                return [_row("P00001")]          # the batch under-delivers
            return [_row(batch[0])]              # individual requests are honest

        monkeypatch.setattr(proteome, "_fetch_accession_batch", flaky)
        frame, _ = proteome.fetch_manifest_for(
            ["P00001", "P00002", "P00003"], "retryset", paths=paths,
        )
        assert sorted(frame["uniprot_id"]) == ["P00001", "P00002", "P00003"]
        assert calls[1:] == [["P00002"], ["P00003"]]

    def test_an_accession_nobody_returns_is_recorded_not_dropped_quietly(
        self, paths, monkeypatch
    ):
        import json

        from vepbench import proteome

        monkeypatch.setattr(
            proteome, "_fetch_accession_batch",
            lambda batch, *, timeout: [_row(a) for a in batch if a != "GONE"],
        )
        frame, _ = proteome.fetch_manifest_for(["P00001", "GONE"], "missingset",
                                               paths=paths)
        assert frame["uniprot_id"].tolist() == ["P00001"]
        recorded = json.loads(
            (paths.outputs / "proteome/missingset/manifest.json").read_text())
        assert recorded["not_found"] == ["GONE"]
        assert recorded["n_requested"] == 2

    def test_an_existing_manifest_is_reused_without_refetching(self, paths, monkeypatch):
        from vepbench import proteome

        monkeypatch.setattr(
            proteome, "_fetch_accession_batch",
            lambda batch, *, timeout: [_row(a) for a in batch])
        proteome.fetch_manifest_for(["P00001"], "cachedset", paths=paths)

        monkeypatch.setattr(proteome, "_fetch_accession_batch", _must_not_fetch)
        frame, info = proteome.fetch_manifest_for(["P00001"], "cachedset", paths=paths)
        assert len(frame) == 1
        assert info.retrieved == "cached"

    def test_the_length_filter_applies_to_a_list_too(self, paths, monkeypatch):
        from vepbench import proteome

        monkeypatch.setattr(
            proteome, "_fetch_accession_batch",
            lambda batch, *, timeout: [_row("P00001", length=100),
                                       _row("P00002", length=9000)])
        frame, _ = proteome.fetch_manifest_for(["P00001", "P00002"], "lenset",
                                               paths=paths, max_length=1499)
        assert frame["uniprot_id"].tolist() == ["P00001"]


def _row(accession: str, length: int = 400) -> dict:
    return {
        "uniprot_id": accession, "entry_name": f"{accession}_YEAST",
        "gene": accession, "protein_name": "Test", "length": length,
        "n_tm_segments": 2, "tm_residues": 40, "n_intramem_segments": 0,
        "tm_evidence": "predicted", "membrane_topology": "multipass",
    }


def _must_not_fetch(batch, *, timeout):
    raise AssertionError("a cached manifest must not be refetched")

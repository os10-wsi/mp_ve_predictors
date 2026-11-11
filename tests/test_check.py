"""Deciding whether a run finished, and saying what stopped it."""

from __future__ import annotations

import json

import pytest

from vepbench.check import check, normalise, write_missing


def protein(paths, accession, entries, *, table=True, status=True, variants=100):
    directory = paths.output_dir(accession)
    directory.mkdir(parents=True, exist_ok=True)
    if table:
        (directory / "scores.tsv").write_text("uniprot_id\n")
    if status:
        (directory / "status.json").write_text(json.dumps(
            {"uniprot_id": accession, "n_variants": variants, "scores": entries}))
    return directory


def ok(name):
    return {"score": name, "status": "ok", "seconds": 1.0}


def failed(name, message=""):
    return {"score": name, "status": "failed", "message": message}


def unavailable(name, message="human only"):
    return {"score": name, "status": "unavailable", "message": message}


class TestCompleteness:
    def test_a_clean_run_is_complete(self, paths):
        protein(paths, "P1", [ok("esm1v")])
        report = check(["P1"], paths=paths)
        assert report.complete
        assert "run is complete" in "\n".join(report.lines())

    def test_a_protein_with_no_table_is_missing_not_failed(self, paths):
        # Its task died or never started, which is a different problem from
        # a score that broke inside a protein that otherwise worked.
        protein(paths, "P1", [ok("esm1v")])
        report = check(["P1", "P2"], paths=paths)
        assert report.missing == ["P2"]
        assert not report.complete
        assert not report.status_counts.get("esm1v", {}).get("failed")

    def test_a_failed_score_stops_the_run_being_complete(self, paths):
        protein(paths, "P1", [ok("esm1v"), failed("saprot", "boom")])
        assert not check(["P1"], paths=paths).complete

    def test_an_unavailable_score_is_not_a_failure(self, paths):
        # AlphaMissense has no yeast entries and FoldX needs ClinVar
        # positions; both being absent is correct, not broken.
        protein(paths, "P1", [ok("esm1v"), unavailable("alphamissense")])
        report = check(["P1"], paths=paths)
        assert report.complete
        assert report.status_counts["alphamissense"]["unavailable"] == 1

    def test_a_cached_score_counts_as_having_run(self, paths):
        protein(paths, "P1", [{"score": "esm1v", "status": "cached"}])
        assert check(["P1"], paths=paths).complete

    def test_a_table_without_a_status_file_is_flagged(self, paths):
        protein(paths, "P1", [], status=False)
        report = check(["P1"], paths=paths)
        assert report.unreadable == ["P1"]
        assert not report.complete

    def test_a_truncated_status_file_is_flagged_not_fatal(self, paths):
        directory = protein(paths, "P1", [ok("esm1v")])
        (directory / "status.json").write_text("{ truncated")
        report = check(["P1"], paths=paths)
        assert report.unreadable == ["P1"]

    def test_variant_rows_are_totalled(self, paths):
        protein(paths, "P1", [ok("esm1v")], variants=7980)
        protein(paths, "P2", [ok("esm1v")], variants=1020)
        assert check(["P1", "P2"], paths=paths).variants == 9000


class TestFailureReporting:
    def test_identical_failures_group_into_one_line(self, paths):
        # Byte counts and paths differ every time; without normalising, one
        # recurring cause becomes thousands of unique strings.
        for index in range(5):
            protein(paths, f"P{index}", [failed(
                "esm1v", f"CUDA out of memory. Tried to allocate {index}.5 GiB "
                         f"on /lustre/x/P{index}")])
        report = check([f"P{i}" for i in range(5)], paths=paths)
        assert len(report.reasons["esm1v"]) == 1
        assert report.reasons["esm1v"].most_common(1)[0][1] == 5

    def test_an_intermittent_failure_is_distinguished_from_a_total_one(self, paths):
        protein(paths, "P1", [ok("esm1v"), failed("prosst", "broken")])
        protein(paths, "P2", [failed("esm1v", "oom"), failed("prosst", "broken")])
        text = "\n".join(check(["P1", "P2"], paths=paths).lines())
        assert "intermittent" in text
        assert "every attempt" in text

    def test_a_failure_with_no_message_still_appears(self, paths):
        protein(paths, "P1", [failed("esm1v")])
        text = "\n".join(check(["P1"], paths=paths).lines())
        assert "(no message recorded)" in text

    @pytest.mark.parametrize(("raw", "expected_in"), [
        ("runner exited 1: CUDA out of memory. Tried to allocate 2.5 GiB",
         "CUDA out of memory"),
        ("(HF Xet disabled) | could not import prosst from /lustre/a/b",
         "could not import prosst"),
    ])
    def test_normalising_keeps_the_cause_and_drops_the_noise(self, raw, expected_in):
        assert expected_in in normalise(raw)
        assert "/lustre" not in normalise(raw)


class TestMissingList:
    def test_the_missing_accessions_are_written_for_resubmission(self, paths,
                                                                 tmp_path):
        protein(paths, "P1", [ok("esm1v")])
        report = check(["P1", "P2", "P3"], paths=paths)
        path = write_missing(report, tmp_path / "missing.txt")
        assert path.read_text().split() == ["P2", "P3"]

    def test_nothing_is_written_when_nothing_is_missing(self, paths, tmp_path):
        protein(paths, "P1", [ok("esm1v")])
        assert write_missing(check(["P1"], paths=paths),
                             tmp_path / "missing.txt") is None


class TestCli:
    def test_an_incomplete_run_exits_non_zero(self, paths, capsys):
        from vepbench.cli import main

        protein(paths, "P1", [ok("esm1v")])
        assert main(["check", "--uniprot", "P1", "P2"]) == 1
        out = capsys.readouterr().out
        assert "1 of 2 proteins" in out
        assert "resubmit with" in out

    def test_a_complete_run_exits_zero(self, paths, capsys):
        from vepbench.cli import main

        protein(paths, "P1", [ok("esm1v")])
        assert main(["check", "--uniprot", "P1"]) == 0
        assert "run is complete" in capsys.readouterr().out

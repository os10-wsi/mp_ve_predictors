"""The `vepbench proteome` stages, driven through the CLI."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from vepbench.cli import main
from vepbench.schema import HARMONIZED_SUFFIX

SEQUENCE = "MGRGLMFVKQ"


@pytest.fixture
def small_set(paths, monkeypatch):
    """A two-protein manifest with cached UniProt entries, as define leaves it."""
    directory = paths.outputs / "proteome" / "testset"
    directory.mkdir(parents=True)
    manifest = pd.DataFrame({
        "uniprot_id": ["P00001", "P00002"],
        "entry_name": ["ONE_HUMAN", "TWO_HUMAN"],
        "gene": ["GENEA", "GENEB"],
        "protein_name": ["One", "Two"],
        "length": [10, 10], "n_tm_segments": [2, 1], "tm_residues": [4, 2],
        "n_intramem_segments": [0, 0], "tm_evidence": ["predicted"] * 2,
        "membrane_topology": ["multipass", "singlepass"],
    })
    manifest.to_csv(directory / "manifest.tsv", sep="\t", index=False)
    for accession in manifest["uniprot_id"]:
        (paths.annotations / f"{accession}.json").write_text(json.dumps({
            "uniprot_id": accession, "sequence": SEQUENCE, "features": [],
        }))
    return directory


def write_clinvar_source(path, genes=("GENEA", "GENEB")):
    header = ["Type", "Name", "GeneSymbol", "ClinicalSignificance", "ReviewStatus",
              "NumberSubmitters", "PhenotypeList", "VariationID", "Origin",
              "OriginSimple", "Assembly"]
    rows = []
    identifier = 0
    for gene in genes:
        for position, (wt, mut, significance) in enumerate([
            ("Met", "Val", "Pathogenic"), ("Gly", "Ser", "Benign"),
            ("Arg", "Trp", "Pathogenic"), ("Gly", "Ala", "Benign"),
            ("Leu", "Pro", "Pathogenic"), ("Met", "Thr", "Benign"),
            ("Phe", "Ser", "Pathogenic"), ("Val", "Asp", "Benign"),
            ("Lys", "Glu", "Pathogenic"), ("Gln", "His", "Benign"),
        ], start=1):
            identifier += 1
            rows.append([
                "single nucleotide variant",
                f"NM_1.1({gene}):c.1A>G (p.{wt}{position}{mut})", gene, significance,
                "criteria provided, multiple submitters, no conflicts", "3", "Cond",
                str(identifier), "germline", "germline", "GRCh38",
            ])
    with gzip.open(path, "wt") as handle:
        handle.write("\t".join(header) + "\n")
        for item in rows:
            handle.write("\t".join(item) + "\n")
    return path


def write_alphamissense_source(path, accessions=("P00001", "P00002")):
    with gzip.open(path, "wt") as handle:
        handle.write("# stand-in\n")
        handle.write("uniprot_id\tprotein_variant\tam_pathogenicity\tam_class\n")
        for accession in accessions:
            for position, aa in enumerate(SEQUENCE, start=1):
                for mut in "ACDEFGHIKLMNPQRSTVWY":
                    if mut == aa:
                        continue
                    handle.write(f"{accession}\t{aa}{position}{mut}\t0.5\tambiguous\n")
    return path


def write_predictions(paths, accession, predictors=("alpha", "beta"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for position, wt in enumerate(SEQUENCE, start=1):
        for mut in "ACDEFGHIKLMNPQRSTVWY":
            if mut == wt:
                continue
            rows.append({"uniprot_id": accession, "position": position,
                         "wt_aa": wt, "mut_aa": mut, "variant": f"{wt}{position}{mut}"})
    frame = pd.DataFrame(rows)
    for name in predictors:
        frame[f"{name}{HARMONIZED_SUFFIX}"] = rng.normal(size=len(frame))
    frame["in_transmembrane"] = rng.random(len(frame)) < 0.5
    directory = paths.output_dir(accession)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "predictions_wide.csv", index=False)
    return frame


class TestDefine:
    def test_count_only_reaches_no_further_than_the_count(self, paths, monkeypatch):
        seen = {}

        def fake_count(query, **kwargs):
            seen["query"] = query
            return 4242

        monkeypatch.setattr("vepbench.proteome.count_matches", fake_count)
        monkeypatch.setattr("vepbench.proteome.fetch_manifest", _must_not_run)
        assert main(["proteome", "define", "--set", "human_membrane",
                     "--count-only"]) == 0
        assert "KW-0812" in seen["query"]

    def test_an_unknown_set_name_fails_before_any_download(self, paths, capsys):
        from vepbench.proteome import ProteomeError

        with pytest.raises(ProteomeError, match="human_membrane"):
            main(["proteome", "define", "--set", "nonsense"])


def _must_not_run(*args, **kwargs):
    raise AssertionError("this stage should not have run")


class TestFetchData:
    def test_clinvar_is_mapped_and_both_artefacts_are_written(self, paths, small_set,
                                                              tmp_path, capsys):
        source = write_clinvar_source(tmp_path / "variant_summary.txt.gz")
        code = main(["proteome", "fetch-data", "--set", "testset",
                     "--only", "clinvar", "--clinvar-file", str(source)])
        assert code == 0
        assert (small_set / "clinvar" / "mapping_report.tsv").exists()
        out = capsys.readouterr().out
        assert "pathogenic" in out

    def test_rows_outside_the_protein_set_are_counted_separately(
        self, paths, small_set, tmp_path, capsys
    ):
        # They are reported, but not as a failure: ClinVar covers the whole
        # genome and most of it was never a candidate.
        source = write_clinvar_source(tmp_path / "vs.txt.gz", genes=("GENEA", "ABSENT"))
        main(["proteome", "fetch-data", "--set", "testset", "--only", "clinvar",
              "--clinvar-file", str(source)])
        out = capsys.readouterr().out
        assert "rows were in genes outside the set" in out
        assert "gene_not_in_manifest:" not in out

    def test_alphamissense_is_sliced_per_accession(self, paths, small_set, tmp_path):
        from vepbench.alphamissense import slice_path

        source = write_alphamissense_source(tmp_path / "am.tsv.gz")
        assert main(["proteome", "fetch-data", "--set", "testset",
                     "--only", "alphamissense",
                     "--alphamissense-file", str(source)]) == 0
        assert slice_path("P00001", paths=paths).exists()
        assert slice_path("P00002", paths=paths).exists()

    def test_a_sliced_protein_can_then_be_scored(self, paths, small_set, tmp_path):
        from vepbench.alphamissense import load_slice

        write_alphamissense_source(tmp_path / "am.tsv.gz")
        main(["proteome", "fetch-data", "--set", "testset", "--only", "alphamissense",
              "--alphamissense-file", str(tmp_path / "am.tsv.gz")])
        lookup = load_slice("P00001", paths=paths)
        assert lookup["M1V"] == (0.5, "ambiguous")

    def test_mapping_without_cached_sequences_fails_with_the_fix(
        self, paths, small_set, tmp_path, capsys
    ):
        for accession in ("P00001", "P00002"):
            (paths.annotations / f"{accession}.json").unlink()
        source = write_clinvar_source(tmp_path / "vs.txt.gz")
        code = main(["proteome", "fetch-data", "--set", "testset", "--only", "clinvar",
                     "--clinvar-file", str(source)])
        assert code == 1
        assert "no UniProt sequences are cached" in capsys.readouterr().err


class TestRun:
    def test_without_submit_the_command_is_printed_and_nothing_runs(
        self, paths, small_set, capsys
    ):
        assert main(["proteome", "run", "--set", "testset"]) == 0
        out = capsys.readouterr().out
        assert "submit_job.sh" in out
        assert "--predictors proteome" in out
        assert (small_set / "to_run.txt").read_text().split() == ["P00001", "P00002"]

    def test_proteins_already_run_can_be_left_out(self, paths, small_set, capsys):
        write_predictions(paths, "P00001")
        main(["proteome", "run", "--set", "testset", "--skip-done"])
        assert (small_set / "to_run.txt").read_text().split() == ["P00002"]

    def test_long_proteins_can_be_excluded(self, paths, small_set, capsys):
        assert main(["proteome", "run", "--set", "testset", "--max-length", "5"]) == 0
        assert "nothing left to run" in capsys.readouterr().out


class TestAggregateAndAnalyse:
    def _prepare(self, paths, small_set, tmp_path, predictors=("alpha", "beta")):
        write_predictions(paths, "P00001", predictors, seed=1)
        write_predictions(paths, "P00002", predictors, seed=2)
        source = write_clinvar_source(tmp_path / "vs.txt.gz")
        main(["proteome", "fetch-data", "--set", "testset", "--only", "clinvar",
              "--clinvar-file", str(source)])

    def test_aggregate_reports_coverage_per_predictor(self, paths, small_set,
                                                      tmp_path, capsys):
        self._prepare(paths, small_set, tmp_path)
        assert main(["proteome", "aggregate", "--set", "testset"]) == 0
        out = capsys.readouterr().out
        assert "2 of 2 proteins have predictions" in out
        assert (small_set / "analysis" / "coverage.tsv").exists()

    def test_analyse_writes_every_metrics_table(self, paths, small_set, tmp_path):
        self._prepare(paths, small_set, tmp_path)
        main(["proteome", "aggregate", "--set", "testset"])
        assert main(["proteome", "analyse", "--set", "testset",
                     "--min-per-class", "3", "--min-rows", "5",
                     "--n-boot", "10", "--tracks", "0"]) == 0
        analysis = small_set / "analysis"
        for stem in ("metrics_pooled", "metrics_stratified", "metrics_per_protein",
                     "metrics_summary"):
            assert (analysis / f"{stem}.tsv").exists(), stem

    def test_analyse_without_a_clinvar_table_says_which_command_is_missing(
        self, paths, small_set, capsys
    ):
        assert main(["proteome", "analyse", "--set", "testset"]) == 1
        assert "fetch-data" in capsys.readouterr().err

    def test_analyse_draws_the_requested_per_protein_tracks(self, paths, small_set,
                                                            tmp_path):
        self._prepare(paths, small_set, tmp_path)
        main(["proteome", "aggregate", "--set", "testset"])
        main(["proteome", "analyse", "--set", "testset", "--min-per-class", "3",
              "--min-rows", "5", "--n-boot", "10", "--tracks", "1"])
        tracks = list((small_set / "plots" / "per_protein").glob("*.png"))
        assert len(tracks) == 1

    def test_the_analysis_joins_manifest_columns_for_stratifying(self, paths, small_set,
                                                                 tmp_path):
        from vepbench import tables

        self._prepare(paths, small_set, tmp_path)
        main(["proteome", "aggregate", "--set", "testset"])
        main(["proteome", "analyse", "--set", "testset", "--min-per-class", "3",
              "--min-rows", "5", "--n-boot", "10", "--tracks", "0"])
        joined = tables.read_table(small_set / "analysis" / "labelled_variants")
        assert "membrane_topology" in joined.columns
        assert "in_transmembrane" in joined.columns


class TestUniProtRetry:
    """A long serial fetch draws throttling; a blip must not cost a gene."""

    def _manifest(self, paths, accessions):
        directory = paths.outputs / "proteome" / "retryset"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "uniprot_id": accessions, "gene": [f"G{i}" for i in range(len(accessions))],
            "length": [10] * len(accessions),
        }).to_csv(directory / "manifest.tsv", sep="\t", index=False)
        return pd.read_csv(directory / "manifest.tsv", sep="\t")

    def test_an_entry_that_fails_once_is_retried_and_kept(self, paths, monkeypatch):
        from vepbench import cli_proteome
        from vepbench.uniprot import UniProtError

        monkeypatch.setattr(cli_proteome, "UNIPROT_BACKOFF_SECONDS", 0)
        attempts: dict[str, int] = {}

        def flaky(accession, **kwargs):
            attempts[accession] = attempts.get(accession, 0) + 1
            if accession == "P00002" and attempts[accession] == 1:
                raise UniProtError("503 from UniProt")
            (paths.annotations / f"{accession}.json").write_text(
                json.dumps({"uniprot_id": accession, "sequence": "MG", "features": []})
            )

        monkeypatch.setattr("vepbench.uniprot.fetch_uniprot", flaky)
        manifest = self._manifest(paths, ["P00001", "P00002"])
        cli_proteome._fetch_uniprot(manifest, offline=False)

        assert attempts["P00002"] == 2
        assert (paths.annotations / "P00002.json").exists()

    def test_an_entry_that_never_comes_back_is_written_down(self, paths, monkeypatch,
                                                            caplog):
        from vepbench import cli_proteome
        from vepbench.uniprot import UniProtError

        monkeypatch.setattr(cli_proteome, "UNIPROT_BACKOFF_SECONDS", 0)

        def always_fails(accession, **kwargs):
            raise UniProtError("gone")

        monkeypatch.setattr("vepbench.uniprot.fetch_uniprot", always_fails)
        manifest = self._manifest(paths, ["P00001"])
        cli_proteome._fetch_uniprot(manifest, offline=False)

        record = paths.annotations / "uniprot_fetch_failures.txt"
        assert record.read_text().split() == ["P00001"]

    def test_a_successful_pass_does_not_retry(self, paths, monkeypatch):
        from vepbench import cli_proteome

        calls: list[str] = []

        def works(accession, **kwargs):
            calls.append(accession)
            (paths.annotations / f"{accession}.json").write_text(
                json.dumps({"uniprot_id": accession, "sequence": "MG", "features": []})
            )

        monkeypatch.setattr("vepbench.uniprot.fetch_uniprot", works)
        manifest = self._manifest(paths, ["P00001", "P00002"])
        cli_proteome._fetch_uniprot(manifest, offline=False)
        assert calls == ["P00001", "P00002"]

    def test_offline_does_not_retry(self, paths, monkeypatch):
        # Nothing about waiting makes a cache miss resolve itself.
        from vepbench import cli_proteome
        from vepbench.uniprot import UniProtError

        monkeypatch.setattr(cli_proteome, "UNIPROT_BACKOFF_SECONDS", 0)
        calls: list[str] = []

        def not_cached(accession, **kwargs):
            calls.append(accession)
            raise UniProtError("not cached and --offline was requested")

        monkeypatch.setattr("vepbench.uniprot.fetch_uniprot", not_cached)
        manifest = self._manifest(paths, ["P00001"])
        cli_proteome._fetch_uniprot(manifest, offline=True)
        assert calls == ["P00001"]


class TestProgress:
    """A multi-day run needs its remaining time measured, not guessed."""

    def _setup(self, paths, accessions, done, *, lengths=None, spacing=3600.0,
               seconds=None, failed=()):
        import time

        directory = paths.outputs / "proteome" / "progset"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "uniprot_id": accessions,
            "gene": [f"G{i}" for i in range(len(accessions))],
            "length": lengths or [400] * len(accessions),
        }).to_csv(directory / "manifest.tsv", sep="\t", index=False)

        base = time.time() - spacing * len(done)
        for index, accession in enumerate(done):
            write_predictions(paths, accession, ("alpha", "beta"), seed=index)
            entries = [{"predictor": n, "status": "ok", "seconds": s}
                       for n, s in (seconds or {"alpha": 60.0, "beta": 120.0}).items()]
            entries += [{"predictor": n, "status": "failed"} for n in failed]
            (paths.output_dir(accession) / "status.json").write_text(json.dumps({
                "uniprot_id": accession, "generated": base + spacing * index,
                "predictors": entries,
            }))
        return directory

    def test_it_counts_what_is_done_against_what_was_submitted(self, paths, capsys):
        self._setup(paths, ["P1", "P2", "P3", "P4"], ["P1", "P2"])
        assert main(["proteome", "progress", "--set", "progset"]) == 0
        assert "2 of 4 proteins complete (50.0%), 2 to go" in capsys.readouterr().out

    def test_the_denominator_follows_the_length_cap_that_was_used(self, paths, capsys):
        # Projecting against proteins that were never submitted would make
        # the run look permanently behind.
        self._setup(paths, ["P1", "P2", "P3", "P4"], ["P1"],
                    lengths=[400, 400, 9000, 9000])
        main(["proteome", "progress", "--set", "progset", "--max-length", "1499"])
        assert "1 of 2 proteins complete" in capsys.readouterr().out

    def test_it_projects_the_remaining_time_from_observed_throughput(self, paths,
                                                                     capsys):
        # Four done, one per hour, six left -> about six hours.
        self._setup(paths, [f"P{i}" for i in range(10)],
                    [f"P{i}" for i in range(4)], spacing=3600.0)
        main(["proteome", "progress", "--set", "progset"])
        out = capsys.readouterr().out
        assert "proteins/hour" in out
        assert "6.0 hours" in out

    def test_it_reports_which_predictors_cost_the_most(self, paths, capsys):
        self._setup(paths, ["P1", "P2"], ["P1", "P2"],
                    seconds={"cheap": 5.0, "expensive": 500.0})
        main(["proteome", "progress", "--set", "progset"])
        out = capsys.readouterr().out
        lines = [ln for ln in out.splitlines() if "expensive" in ln or "cheap" in ln]
        assert "expensive" in lines[0]        # most costly first
        assert "505.0s" in out

    def test_a_predictor_failing_everywhere_is_named(self, paths, capsys):
        self._setup(paths, ["P1", "P2"], ["P1", "P2"], failed=("prosst",))
        main(["proteome", "progress", "--set", "progset"])
        assert "prosst: 2" in capsys.readouterr().out

    def test_nothing_finished_yet_projects_nothing(self, paths, capsys):
        self._setup(paths, ["P1", "P2"], [])
        main(["proteome", "progress", "--set", "progset"])
        assert "nothing has finished yet" in capsys.readouterr().out

    def test_a_finished_run_says_so(self, paths, capsys):
        self._setup(paths, ["P1"], ["P1"])
        main(["proteome", "progress", "--set", "progset"])
        assert "run complete" in capsys.readouterr().out

    def test_a_failed_predictor_contributes_no_timing(self, paths, capsys):
        # A predictor that failed in 0.2s must not drag the mean down and
        # make the remaining work look cheaper than it is.
        self._setup(paths, ["P1", "P2"], ["P1", "P2"],
                    seconds={"real": 300.0}, failed=("prosst",))
        capsys.readouterr()          # discard anything the setup printed
        main(["proteome", "progress", "--set", "progset"])
        text = capsys.readouterr().out
        assert "300.0s" in text
        assert "prosst" in text.split("failing on every protein")[1]


class TestSchedulerResources:
    """Resource flags belong to the submitter, not to `vepbench run`."""

    def _manifest(self, paths):
        directory = paths.outputs / "proteome" / "resset"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"uniprot_id": ["P1", "P2"], "gene": ["A", "B"],
                      "length": [100, 200]}).to_csv(
            directory / "manifest.tsv", sep="\t", index=False)
        return directory

    def test_memory_and_cores_reach_the_submitter(self, paths, capsys):
        self._manifest(paths)
        main(["proteome", "run", "--set", "resset", "--mem", "10000",
              "--cores", "2", "--time", "8:00"])
        out = capsys.readouterr().out
        assert "--mem 10000" in out
        assert "--cores 2" in out
        assert "--time 8:00" in out

    def test_they_come_before_the_handover_to_vepbench_run(self, paths, capsys):
        # Anything after `--` is passed to `vepbench run`, which would reject
        # --mem outright.
        self._manifest(paths)
        main(["proteome", "run", "--set", "resset", "--mem", "10000",
              "--extra", "--offline"])
        out = capsys.readouterr().out
        assert out.index("--mem") < out.index(" -- ")

    def test_resources_are_omitted_when_not_given(self, paths, capsys):
        self._manifest(paths)
        main(["proteome", "run", "--set", "resset"])
        out = capsys.readouterr().out
        assert "--mem" not in out
        assert "--cores" not in out

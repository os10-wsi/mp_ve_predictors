"""Command line behaviour."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from vepbench.cli import main

FIXTURES = Path(__file__).parent / "fixtures"
ACCESSION = "P26678"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A temporary working directory with the cached inputs in place."""
    monkeypatch.setenv("VEPBENCH_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("VEPBENCH_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("VEPBENCH_WEIGHTS", str(tmp_path / "weights"))

    from vepbench import config
    importlib.reload(config)
    config.PATHS.ensure()
    shutil.copy(FIXTURES / f"{ACCESSION}.uniprot.json",
                config.PATHS.annotations / f"{ACCESSION}.json")

    from vepbench import cli
    importlib.reload(cli)
    return tmp_path, cli


def test_list_runs(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "esm1v_full" in out
    assert "predictors" in out


def test_list_json_is_valid(capsys):
    assert main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["esm1v_full"]["higher_is"] == "more_tolerated"
    assert "citation" in payload["foldx"]
    assert payload["esm2_3b"]["approximate_vram_gb"] == 13.0


def test_list_filtered_by_category(capsys):
    assert main(["list", "--category", "stability"]) == 0
    out = capsys.readouterr().out
    assert "foldx" in out


def test_check_conventions(capsys):
    assert main(["check-conventions"]) == 0
    out = capsys.readouterr().out
    assert "NEEDS CONFIRMATION" in out
    assert "ddmut" in out


def test_status_lists_environments(capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "esm" in out and "installed" in out


def test_install_env_dry_run(capsys):
    assert main(["install-env", "esm", "--dry-run"]) == 0
    assert "ESM-1v" in capsys.readouterr().out


def test_shared_flags_work_after_the_subcommand(capsys):
    """`vepbench list -q` must parse as readily as `vepbench -q list`."""
    assert main(["list", "-q"]) == 0


def test_run_merge_benchmark_end_to_end(workspace):
    tmp_path, cli = workspace
    structure = str(FIXTURES / f"{ACCESSION}_synthetic_helix_TESTONLY.pdb")

    assert cli.main([
        "run", "-q", "--uniprot", ACCESSION, "--predictors", "blosum62_baseline",
        "--pdb", structure, "--msa-backend", "none", "--offline",
    ]) == 0
    wide_path = tmp_path / "outputs" / ACCESSION / "predictions_wide.csv"
    assert wide_path.exists()

    predictions = pd.read_csv(wide_path).head(20)
    dms = pd.DataFrame({
        "variant": list(predictions["variant"]) + ["Z999A"],
        "abundance": list(range(20)) + [0],
    })
    dms_path = tmp_path / "dms.csv"
    dms.to_csv(dms_path, index=False)

    merged_path = tmp_path / "merged.csv"
    assert cli.main([
        "merge", "--dms", str(dms_path), "--uniprot", ACCESSION,
        "--variant-col", "variant", "--out", str(merged_path), "--offline",
    ]) == 0
    merged = pd.read_csv(merged_path)
    assert (merged["match_status"] == "matched").sum() == 20
    assert (merged["match_status"] != "matched").sum() == 1
    assert (tmp_path / "merged.unmatched.csv").exists()
    assert (tmp_path / "merged.merge_report.json").exists()

    summary_path = tmp_path / "summary.csv"
    assert cli.main([
        "benchmark", "--merged", str(merged_path), "--score-col", "abundance",
        "--n-boot", "50", "--out", str(summary_path),
    ]) == 0
    summary = pd.read_csv(summary_path)
    assert "blosum62_baseline" in set(summary["predictor"])
    assert "spearman" in summary.columns


def test_hardware_flags_are_accepted_and_recorded(workspace):
    tmp_path, cli = workspace
    assert cli.main([
        "run", "-q", "--uniprot", ACCESSION, "--predictors", "blosum62_baseline",
        "--pdb", str(FIXTURES / f"{ACCESSION}_synthetic_helix_TESTONLY.pdb"),
        "--msa-backend", "none", "--offline",
        "--device", "cpu", "--dtype", "float32", "--batch-size", "2",
    ]) == 0
    metadata = json.loads(
        (tmp_path / "outputs" / ACCESSION / "metadata.json").read_text()
    )
    assert metadata["hardware_options"] == {
        "device": "cpu", "dtype": "float32", "batch_size": 2
    }


def test_run_without_an_accession_is_an_error(workspace):
    _tmp_path, cli = workspace
    with pytest.raises(SystemExit):
        cli.main(["run", "--predictors", "blosum62_baseline"])


def test_merge_without_predictions_explains_itself(workspace):
    tmp_path, cli = workspace
    dms = tmp_path / "dms.csv"
    pd.DataFrame({"variant": ["M1A"], "score": [1.0]}).to_csv(dms, index=False)
    with pytest.raises(SystemExit, match="vepbench run"):
        cli.main(["merge", "--dms", str(dms), "--uniprot", ACCESSION,
                  "--variant-col", "variant", "--out", str(tmp_path / "m.csv"),
                  "--offline"])


def test_uniprot_list_file(workspace):
    tmp_path, cli = workspace
    listing = tmp_path / "ids.txt"
    listing.write_text(f"# a comment\n{ACCESSION}\n\n")
    assert cli.main([
        "run", "-q", "--uniprot-list", str(listing),
        "--predictors", "blosum62_baseline",
        "--pdb", str(FIXTURES / f"{ACCESSION}_synthetic_helix_TESTONLY.pdb"),
        "--msa-backend", "none", "--offline",
    ]) == 0
    assert (tmp_path / "outputs" / ACCESSION / "predictions_long.csv").exists()

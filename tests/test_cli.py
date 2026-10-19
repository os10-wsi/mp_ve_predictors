"""Command line behaviour."""

from __future__ import annotations

import importlib
import json
import os
import re
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
    # Seed the structure too: the default feature set includes rSASA and
    # pLDDT, so anything using the defaults needs one.
    shutil.copy(FIXTURES / "AF-P26678-F1-model_v6.pdb",
                config.PATHS.structures / "AF-P26678-F1-model_v6.pdb")

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
    assert "thermompnn" in out


def test_status_lists_environments(capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "esm" in out and "installed" in out


def test_install_env_dry_run(capsys):
    assert main(["install-env", "esm", "--dry-run"]) == 0
    assert "ESM-1v" in capsys.readouterr().out


def test_install_env_accepts_several_names(capsys):
    assert main(["install-env", "esm", "proteinmpnn", "gemme", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 environment(s)" in out
    for name in ("esm:", "proteinmpnn:", "gemme:"):
        assert name in out


def test_install_env_all_still_expands(capsys):
    from vepbench.envs import ENVIRONMENTS

    assert main(["install-env", "all", "--dry-run"]) == 0
    assert f"{len(ENVIRONMENTS)} environment(s)" in capsys.readouterr().out


def test_install_env_does_not_build_anything_twice(capsys):
    """`all` plus a name it already covers is one build, not two."""
    from vepbench.envs import ENVIRONMENTS

    assert main(["install-env", "esm", "all", "--dry-run"]) == 0
    assert f"{len(ENVIRONMENTS)} environment(s)" in capsys.readouterr().out


def test_install_env_rejects_a_typo_before_building(capsys):
    """A bad name in the list must not surface after the earlier downloads."""
    with pytest.raises(SystemExit, match="mtel"):
        main(["install-env", "esm", "mtel", "proteinmpnn", "--dry-run"])
    assert "esm:" not in capsys.readouterr().out


def test_doctor_names_the_script_that_fetches_each_checkpoint(
    bare_checkout, capsys, monkeypatch
):
    """Not a blanket `download_weights_huggingface.sh all`, which cannot
    fetch the ESM checkpoints at all."""
    monkeypatch.setattr("vepbench.envs.EnvSpec.is_installed", lambda self, p: True)
    bare_checkout.main(["doctor", "--predictors", "esm2_3b"])
    out = capsys.readouterr().out
    assert "download_weights_esm.sh esm2-3b" in out
    assert "download_weights_huggingface.sh all" not in out


def test_doctor_json_separates_runnable_from_missing(bare_checkout, capsys):
    """A submit script needs 'nothing can run' and 'something is missing' to
    be different answers; the exit code alone cannot say which."""
    assert bare_checkout.main(
        ["doctor", "--predictors", "blosum62_baseline", "gemme", "--json"]
    ) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] == ["blosum62_baseline"]
    assert "gemme" in report["blocked"]
    assert report["problems"], "a missing environment is still a problem"


def test_doctor_json_is_clean_when_nothing_is_missing(bare_checkout, capsys):
    assert bare_checkout.main(
        ["doctor", "--predictors", "blosum62_baseline", "--json"]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] == ["blosum62_baseline"]
    assert report["blocked"] == {}
    assert report["problems"] == []


def test_clean_lists_categories_and_their_cost(capsys):
    assert main(["clean", "--list"]) == 0
    out = capsys.readouterr().out
    assert "costs to recreate" in out
    assert "never touched: weights" in out


def test_clean_dry_run_removes_nothing(workspace, capsys):
    tmp_path, cli = workspace
    scratch = tmp_path / "data" / "cache" / "work" / "P26678"
    scratch.mkdir(parents=True)
    (scratch / "scores.csv").write_text("x")
    assert cli.main(["clean", "--dry-run"]) == 0
    assert "nothing removed" in capsys.readouterr().out
    assert (scratch / "scores.csv").exists()


def test_clean_rejects_a_protected_directory(workspace, capsys):
    _tmp_path, cli = workspace
    assert cli.main(["clean", "--what", "weights", "--dry-run"]) == 2
    assert "unknown categor" in capsys.readouterr().out


def test_clean_all_still_spares_the_results(workspace, capsys):
    tmp_path, cli = workspace
    results = tmp_path / "outputs" / "P26678"
    results.mkdir(parents=True)
    (results / "predictions_long.csv").write_text("x")
    assert cli.main(["clean", "--what", "all", "--yes"]) == 0
    assert (results / "predictions_long.csv").exists()


def test_shared_flags_work_after_the_subcommand(capsys):
    """`vepbench list -q` must parse as readily as `vepbench -q list`."""
    assert main(["list", "-q"]) == 0


@pytest.fixture
def bare_checkout(tmp_path, monkeypatch):
    """A tree with nothing installed in it.

    `doctor` reads `envs/` under the project root, so without this the result
    would depend on which environments the developer happens to have built.
    """
    monkeypatch.setenv("VEPBENCH_ROOT", str(tmp_path))
    monkeypatch.setenv("VEPBENCH_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("VEPBENCH_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("VEPBENCH_WEIGHTS", str(tmp_path / "weights"))

    from vepbench import config
    importlib.reload(config)
    from vepbench import cli
    importlib.reload(cli)
    yield cli
    importlib.reload(config)
    importlib.reload(cli)


def test_doctor_reports_a_missing_environment(bare_checkout, capsys):
    assert bare_checkout.main(["doctor", "--predictors", "esm2_650m"]) == 1
    out = capsys.readouterr().out
    assert "environment esm" in out
    assert "install-env esm" in out


def test_doctor_is_happy_when_nothing_is_needed(bare_checkout, capsys):
    """The baselines ship ready; doctor must not invent work for them."""
    assert bare_checkout.main(["doctor", "--predictors", "blosum62_baseline"]) == 0
    assert "blosum62_baseline" in capsys.readouterr().out


def test_doctor_lists_passing_checks_only_when_asked(bare_checkout, capsys):
    bare_checkout.main(["doctor", "--predictors", "blosum62_baseline"])
    assert "environment backend" not in capsys.readouterr().out
    bare_checkout.main(["doctor", "--predictors", "blosum62_baseline", "-v"])
    assert "environment backend" in capsys.readouterr().out


def test_doctor_rejects_an_unknown_predictor(bare_checkout, capsys):
    assert bare_checkout.main(["doctor", "--predictors", "blosum62"]) == 2
    assert "unknown predictors" in capsys.readouterr().out


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


def test_prefetch_caches_inputs_then_offline_run_works(workspace, monkeypatch):
    """The cluster pattern: stage inputs where there is a network, run without one."""
    tmp_path, cli = workspace
    # The UniProt entry and AlphaFold model are already in the cache from the
    # fixture, so prefetch should be satisfied without touching the network.
    monkeypatch.setattr(
        "vepbench.uniprot.requests.get",
        lambda *a, **k: pytest.fail("prefetch should not re-fetch a cached entry"),
    )
    assert cli.main([
        "prefetch", "--uniprot", ACCESSION, "--predictors", "blosum62_baseline",
        "--msa-backend", "none",
    ]) == 0
    assert cli.main([
        "run", "-q", "--uniprot", ACCESSION, "--predictors", "blosum62_baseline",
        "--offline", "--msa-backend", "none",
    ]) == 0
    assert (tmp_path / "outputs" / ACCESSION / "predictions_wide.csv").exists()


def test_prefetch_reports_what_it_will_cache(workspace, capsys):
    _tmp_path, cli = workspace
    cli.main(["prefetch", "--uniprot", ACCESSION, "--predictors", "proteinmpnn",
              "--msa-backend", "none"])
    out = capsys.readouterr().out
    assert "sequence" in out
    assert "structure" in out


def test_prefetch_skips_inputs_no_predictor_needs(workspace, capsys):
    """A sequence-only predictor should not drag in a structure."""
    _tmp_path, cli = workspace
    cli.main(["prefetch", "--uniprot", ACCESSION, "--predictors", "blosum62_baseline",
              "--features", "physchem", "--msa-backend", "none"])
    out = capsys.readouterr().out
    assert "structure" not in out.split("===")[0]


def test_prefetch_honours_feature_requirements(workspace, capsys):
    """Asking for a structural feature should stage a structure."""
    _tmp_path, cli = workspace
    cli.main(["prefetch", "--uniprot", ACCESSION, "--predictors", "blosum62_baseline",
              "--features", "rsasa", "--msa-backend", "none"])
    assert "structure" in capsys.readouterr().out


def test_package_runs_without_being_installed():
    """The fallback for clusters whose pip cannot do an editable install.

    The package is pure Python, so running it from the source tree must work
    with nothing installed but its three runtime dependencies.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "vepbench", "list"],
        cwd=root, capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(root)},
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "predictors" in result.stdout


def test_setup_py_shim_exists_and_adds_no_configuration():
    """Old pip needs a setup.py to fall back to; it must stay a shim."""
    from pathlib import Path

    shim = Path(__file__).resolve().parents[1] / "setup.py"
    assert shim.exists(), "setup.py shim is missing; old pip will refuse -e ."
    body = [
        line for line in shim.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # docstring, import and a bare setup() call: nothing that could drift
    # out of step with pyproject.toml
    assert "setup()" in shim.read_text()
    assert "name=" not in shim.read_text(), "configuration belongs in pyproject.toml"
    assert len(body) < 25


REPO = Path(__file__).resolve().parent.parent


def shipped_shell_scripts() -> list[Path]:
    """Every script a user is told to run, excluding built environments."""
    return sorted(
        path for pattern in ("scripts/*.sh", "scripts/runners/*.sh",
                             "examples/*.sh", "envs/*/install.sh")
        for path in REPO.glob(pattern)
    )


def test_there_are_scripts_to_check():
    assert len(shipped_shell_scripts()) > 5


@pytest.mark.parametrize("script", shipped_shell_scripts(), ids=lambda p: p.name)
def test_no_script_assumes_a_bare_python_exists(script):
    """`python` is absent on many Linux systems; `python3` or $PYTHON is not.

    This is a real failure mode rather than a style rule: run_demo.sh died
    with `python: command not found` on a cluster that had only python3.
    """
    offenders = [
        f"{script.name}:{number}: {line.strip()}"
        for number, line in enumerate(script.read_text().splitlines(), start=1)
        # `python` being invoked: followed by a flag, a script or a heredoc.
        # A bare mention in a candidate list guarded by `command -v` is fine.
        if re.search(r"""(?<![\w$/.-])python(?=\s+[-\w./"'$])""", line)
        and not line.lstrip().startswith("#")
    ]
    assert not offenders, "use python3, \"$PYTHON\" or an env interpreter:\n" + \
        "\n".join(offenders)


class TestArraySubmission:
    """A list of accessions should become one task per protein.

    One serial job over 57 proteins wastes the scheduler and loses
    everything after a failure partway through.
    """

    def submit(self, tmp_path, ids, *extra):
        import subprocess

        listing = tmp_path / "ids.txt"
        listing.write_text("".join(f"{i}\n" for i in ids))
        result = subprocess.run(
            ["bash", str(REPO / "scripts" / "submit_job.sh"),
             "--uniprot-list", str(listing), "--predictors", "blosum62_baseline",
             "--dry-run", "--skip-doctor", *extra],
            capture_output=True, text=True, cwd=REPO, check=False,
        )
        assert result.returncode == 0, result.stderr
        # The printed bsub line is shell-quoted so it can be pasted; drop the
        # escapes before matching on the array spec.
        return result.stdout.replace("\\", "")

    def test_a_list_becomes_an_array(self, tmp_path):
        out = self.submit(tmp_path, ["P26678", "P53943", "P04817"])
        assert "[1-3]" in out
        assert "LSB_JOBINDEX" in out

    def test_tasks_are_capped(self, tmp_path):
        out = self.submit(tmp_path, ["P26678", "P53943"], "--max-parallel", "4")
        assert "[1-2]%4" in out

    def test_logs_are_per_task(self, tmp_path):
        out = self.submit(tmp_path, ["P26678", "P53943"])
        assert "%J.%I.out" in out

    def test_one_job_opts_out(self, tmp_path):
        out = self.submit(tmp_path, ["P26678", "P53943"], "--one-job")
        assert "LSB_JOBINDEX" not in out
        assert "--uniprot-list" in out

    def test_a_single_accession_needs_no_array(self, tmp_path):
        out = self.submit(tmp_path, ["P26678"])
        assert "LSB_JOBINDEX" not in out

    def test_an_empty_list_is_refused(self, tmp_path):
        import subprocess

        listing = tmp_path / "empty.txt"
        listing.write_text("\n# nothing but a comment\n\n")
        result = subprocess.run(
            ["bash", str(REPO / "scripts" / "submit_job.sh"),
             "--uniprot-list", str(listing), "--dry-run", "--skip-doctor"],
            capture_output=True, text=True, cwd=REPO, check=False,
        )
        assert result.returncode == 2
        assert "no accessions" in result.stderr


def test_the_demo_resolves_an_interpreter_before_using_one():
    demo = (REPO / "examples" / "run_demo.sh").read_text()
    assert "find_python" in demo
    assert "VEPBENCH_PYTHON" in demo, "allow pointing the demo at a chosen interpreter"


class TestRootDiscovery:
    """vepbench must work from anywhere inside the checkout.

    Taking the working directory as the root resolved everything one level
    too deep when run from a subdirectory: `envs/gemme` became
    `envs/envs/gemme`, every environment read as missing, and the error named
    a path that was never going to exist.
    """

    @pytest.fixture
    def checkout(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VEPBENCH_ROOT", raising=False)
        root = tmp_path / "mp_ve_predictors"
        for marker in ("pyproject.toml",):
            (root / marker).parent.mkdir(parents=True, exist_ok=True)
            (root / marker).write_text("")
        (root / "vepbench").mkdir()
        (root / "envs" / "esm").mkdir(parents=True)
        (root / "scripts" / "runners").mkdir(parents=True)
        return root

    def found_from(self, directory):
        from vepbench.config import find_root

        return find_root(directory)

    def test_the_root_itself(self, checkout):
        assert self.found_from(checkout) == checkout

    @pytest.mark.parametrize(
        "subdir", ["envs", "envs/esm", "scripts", "scripts/runners", "vepbench"]
    )
    def test_any_subdirectory(self, checkout, subdir):
        assert self.found_from(checkout / subdir) == checkout

    def test_envs_is_not_mistaken_for_the_root(self, checkout):
        """envs/ holds envs/<name>/, so a bare envs/ check stops too deep."""
        assert self.found_from(checkout / "envs") == checkout
        assert self.found_from(checkout / "envs") != checkout / "envs"

    def test_the_override_still_wins(self, checkout, monkeypatch, tmp_path):
        monkeypatch.setenv("VEPBENCH_ROOT", str(tmp_path / "elsewhere"))
        assert self.found_from(checkout / "envs") == tmp_path / "elsewhere"

    def test_outside_a_checkout_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VEPBENCH_ROOT", raising=False)
        outside = tmp_path / "unrelated"
        outside.mkdir()
        # The installed package's own checkout, or the directory asked about.
        assert self.found_from(outside) in (outside, Path(__file__).parent.parent)


class TestStatusAcrossProteins:
    """What actually ran, which is not what doctor says is installed."""

    def write(self, outputs, accession, entries):
        import json as _json

        directory = outputs / accession
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "status.json").write_text(
            _json.dumps({"counts": {}, "predictors": entries})
        )

    def test_no_runs_says_so(self, bare_checkout, capsys):
        assert bare_checkout.main(["status", "--all"]) == 0
        assert "no runs yet" in capsys.readouterr().out

    def test_it_tallies_across_proteins(self, bare_checkout, capsys, tmp_path):
        outputs = Path(os.environ["VEPBENCH_OUTPUTS"])
        for accession in ("P00001", "P00002"):
            self.write(outputs, accession, [
                {"predictor": "esm2_650m", "status": "ok"},
                {"predictor": "proteinmpnn", "status": "failed",
                 "message": "No module named 'protein_mpnn_utils'"},
            ])
        assert bare_checkout.main(["status", "--all"]) == 0
        out = capsys.readouterr().out
        assert "2 protein(s)" in out
        assert "protein_mpnn_utils" in out, "the reason must be shown"
        assert "1 predictor(s) have a score for every protein" in out

    def test_a_partial_predictor_is_called_out(self, bare_checkout, capsys):
        outputs = Path(os.environ["VEPBENCH_OUTPUTS"])
        self.write(outputs, "P00001", [{"predictor": "esm2_650m", "status": "ok"}])
        self.write(outputs, "P00002", [{"predictor": "esm2_650m", "status": "failed",
                                        "message": "out of memory"}])
        bare_checkout.main(["status", "--all"])
        out = capsys.readouterr().out
        assert "have some but not all: esm2_650m" in out

    def test_cached_counts_as_having_a_score(self, bare_checkout, capsys):
        """A cached result is a result; it is how passes accumulate."""
        outputs = Path(os.environ["VEPBENCH_OUTPUTS"])
        self.write(outputs, "P00001", [{"predictor": "esm2_650m", "status": "cached"}])
        bare_checkout.main(["status", "--all"])
        assert "1 predictor(s) have a score for every protein" in capsys.readouterr().out

    def test_the_jobs_directory_is_not_a_protein(self, bare_checkout, capsys):
        outputs = Path(os.environ["VEPBENCH_OUTPUTS"])
        (outputs / "jobs").mkdir(parents=True, exist_ok=True)
        self.write(outputs, "P00001", [{"predictor": "esm2_650m", "status": "ok"}])
        bare_checkout.main(["status", "--all"])
        assert "1 protein(s)" in capsys.readouterr().out

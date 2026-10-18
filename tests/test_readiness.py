"""Pre-flight checks: does `doctor` tell the truth about what is in place?

The command exists so that a mistake shows up here rather than four hours
into a queued batch, so the tests are mostly about not lying in either
direction: no false "ready", and no advice that would destroy weights.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

import pytest

from vepbench import VEPBENCH
from vepbench.readiness import (
    ESM_WEIGHT_TARGETS,
    MISSING,
    OK,
    WARNING,
    assess,
    check_binaries,
    check_environments,
    check_legacy_cache,
    check_msa_database,
    check_weights,
    download_command,
    environments_for,
    hf_cache_root,
    hf_catalogue,
    hf_model_cached,
    model_identifiers,
    which_including_bundled,
)


def cache_model(paths, repo_id: str, *, empty: bool = False) -> Path:
    """Fake a Hugging Face download in the cache layout the hub uses."""
    folder = hf_cache_root(paths) / f"models--{repo_id.replace('/', '--')}"
    snapshots = folder / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    if not empty:
        (snapshots / "abc123").mkdir(exist_ok=True)
        (snapshots / "abc123" / "config.json").write_text("{}")
    return folder


def test_cache_lives_with_the_other_weights(paths):
    assert hf_cache_root(paths) == paths.weights / "huggingface" / "hub"


def test_hf_home_wins_when_set(paths, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))
    assert hf_cache_root(paths) == tmp_path / "elsewhere" / "hub"


def test_missing_model_is_not_cached(paths):
    assert not hf_model_cached("facebook/esm2_t33_650M_UR50D", paths=paths)


def test_downloaded_model_is_cached(paths):
    cache_model(paths, "facebook/esm2_t33_650M_UR50D")
    assert hf_model_cached("facebook/esm2_t33_650M_UR50D", paths=paths)


def test_failed_download_does_not_count_as_cached(paths):
    """A directory with no snapshot is an interrupted download, not a model."""
    cache_model(paths, "facebook/esm2_t33_650M_UR50D", empty=True)
    assert not hf_model_cached("facebook/esm2_t33_650M_UR50D", paths=paths)


def test_only_hub_repositories_are_checkable():
    """Bare names such as esmc_600m are fetched by their own library."""
    assert all("/" in repo for name in ("esm2_650m", "saprot", "esmc_600m")
               for repo in model_identifiers(name))


def test_environments_are_scoped_to_the_request():
    """Asking about one predictor must not report on all eighteen envs."""
    assert environments_for(["esm2_650m"]) == ["esm"]
    assert "proteinmpnn" not in environments_for(["esm2_650m", "saprot"])


def test_absent_environment_is_reported_with_its_fix(paths):
    (check,) = check_environments(["esm2_650m"], paths)
    assert check.status == MISSING
    assert check.fix == f"{VEPBENCH} install-env esm"


def test_suggested_commands_name_a_real_interpreter(paths):
    """Plain `python` does not exist on many systems; the advice must run."""
    (check,) = check_environments(["esm2_650m"], paths)
    assert check.fix.startswith(VEPBENCH)
    assert shutil.which(shlex.split(check.fix)[0])


def test_weights_check_follows_the_cache(paths):
    (check,) = check_weights(["esm2_650m"], paths)
    assert check.status == MISSING
    cache_model(paths, "facebook/esm2_t33_650M_UR50D")
    (check,) = check_weights(["esm2_650m"], paths)
    assert check.status == OK


def test_weights_are_not_reported_twice(paths):
    """Two predictors sharing a checkpoint should give one line, not two."""
    names = [n for n in ("saprot", "saprot_dg_masked_marginal")]
    repos = [c.name for c in check_weights(names, paths)]
    assert len(repos) == len(set(repos))


class TestLegacyCache:
    """The old ~/.cache/huggingface is often the only copy of the weights."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        (fake_home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        return fake_home / ".cache" / "huggingface"

    def test_quiet_when_there_is_no_old_cache(self, paths, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "empty"))
        assert check_legacy_cache(paths) == []

    def test_stranded_weights_are_moved_not_deleted(self, paths, home):
        snapshot = home / "hub" / "models--facebook--esm2_t33_650M_UR50D" / "snapshots" / "a"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (check,) = check_legacy_cache(paths)
        assert check.status == WARNING
        assert "facebook/esm2_t33_650M_UR50D" in check.detail
        assert check.fix.startswith("mkdir -p")
        assert "mv " in check.fix
        assert "rm -rf" not in check.fix

    def test_empty_old_cache_may_be_removed(self, paths, home):
        (check,) = check_legacy_cache(paths)
        assert check.fix == f"rm -rf {home}"

    def test_a_copy_in_both_caches_is_reported_as_waste(self, paths, home):
        """The usual state after HF_HOME moved: the same weights twice."""
        repo = "facebook/esm2_t33_650M_UR50D"
        for hub in (home / "hub", hf_cache_root(paths)):
            blob = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "a"
            blob.mkdir(parents=True)
            (blob / "model.bin").write_bytes(b"x" * 2048)
        (check,) = check_legacy_cache(paths)
        assert check.name == "duplicate weights"
        assert check.fix.startswith("rm -rf")
        assert repo in check.detail

    def test_moving_and_deleting_are_reported_separately(self, paths, home):
        """Mixed cache: one model only in the old one, one in both."""
        for repo, hubs in (
            ("facebook/only-here", [home / "hub"]),
            ("facebook/in-both", [home / "hub", hf_cache_root(paths)]),
        ):
            for hub in hubs:
                blob = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "a"
                blob.mkdir(parents=True)
                (blob / "model.bin").write_bytes(b"x" * 1024)
        move, delete = check_legacy_cache(paths)
        assert "facebook/only-here" in move.detail
        assert "mv " in move.fix and "rm -rf" not in move.fix
        assert "facebook/in-both" in delete.detail
        assert "facebook/only-here" not in delete.fix


class TestAssess:
    def test_uninstalled_predictor_is_blocked_with_a_reason(self, paths):
        result = assess(["esm2_650m"], paths=paths)
        assert result.ready == []
        assert "install-env esm" in result.blocked["esm2_650m"]

    def test_installed_without_weights_is_neither_ready_nor_blocked(
        self, paths, monkeypatch
    ):
        """It will run with a network and fail under --offline. Say so."""
        monkeypatch.setattr("vepbench.envs.EnvSpec.is_installed", lambda self, p: True)
        result = assess(["esm2_650m"], paths=paths)
        assert result.ready == []
        assert result.blocked == {}
        assert result.needs_download == {
            "esm2_650m": ["facebook/esm2_t33_650M_UR50D"]
        }

    def test_installed_with_weights_is_ready(self, paths, monkeypatch):
        monkeypatch.setattr("vepbench.envs.EnvSpec.is_installed", lambda self, p: True)
        cache_model(paths, "facebook/esm2_t33_650M_UR50D")
        result = assess(["esm2_650m"], paths=paths)
        assert result.ready == ["esm2_650m"]
        assert result.needs_download == {}

    def test_a_predictor_needing_nothing_is_ready_as_shipped(self, paths):
        """The baselines have no environment and no weights."""
        result = assess(["blosum62_baseline"], paths=paths)
        assert result.ready == ["blosum62_baseline"]

    def test_a_mistyped_name_is_named_not_a_keyerror(self, paths):
        with pytest.raises(ValueError, match="unknown predictors"):
            assess(["blosum62"], paths=paths)

    def test_every_predictor_is_classified_exactly_once(self, paths):
        result = assess(paths=paths)
        from vepbench.predictors import REGISTRY

        covered = set(result.ready) | set(result.blocked) | set(result.needs_download)
        assert covered == set(REGISTRY)
        assert len(covered) == (len(result.ready) + len(result.blocked)
                                + len(result.needs_download))

    def test_ebi_backend_wants_a_contact_address(self, paths, monkeypatch):
        monkeypatch.delenv("VEPBENCH_EMAIL", raising=False)
        result = assess(["blosum62_baseline"], msa_backend="ebi", paths=paths)
        assert any(c.name == "VEPBENCH_EMAIL" and c.status == MISSING
                   for c in result.checks)

    def test_nothing_is_downloaded_or_installed_by_looking(self, paths):
        """`doctor` must be safe to run on a login node, repeatedly."""
        before = sorted(p.relative_to(paths.root) for p in paths.root.rglob("*"))
        assess(["esm2_650m", "saprot"], paths=paths)
        after = sorted(p.relative_to(paths.root) for p in paths.root.rglob("*"))
        assert before == after


class TestLocalAlignmentBackends:
    """jackhmmer and hhblits are chosen by flag, not by predictor."""

    def test_colabfold_needs_no_local_tooling(self):
        assert check_msa_database(["gemme"], "colabfold") == []
        assert not [c for c in check_binaries(["gemme"], "colabfold")
                    if c.name == "binary jackhmmer"]

    def test_jackhmmer_binary_is_checked_when_selected(self):
        (check,) = [c for c in check_binaries(["gemme"], "jackhmmer")
                    if c.name == "binary jackhmmer"]
        assert check.status == MISSING

    def test_the_backend_is_irrelevant_without_an_msa_predictor(self):
        """Asking for jackhmmer while running only ESM-2 needs neither."""
        assert check_msa_database(["esm2_650m"], "jackhmmer") == []
        assert not [c for c in check_binaries(["esm2_650m"], "jackhmmer")
                    if c.name == "binary jackhmmer"]

    def test_absent_database_says_where_to_put_it(self, paths, monkeypatch):
        monkeypatch.delenv("VEPBENCH_MSA_DATABASE", raising=False)
        (check,) = check_msa_database(["gemme"], "jackhmmer")
        assert check.status == MISSING
        assert "uniref90.fasta" in check.detail

    def test_database_from_the_environment_is_accepted(self, monkeypatch, tmp_path):
        database = tmp_path / "uniref90.fasta"
        database.write_text(">x\nMKV\n")
        monkeypatch.setenv("VEPBENCH_MSA_DATABASE", str(database))
        (check,) = check_msa_database(["gemme"], "jackhmmer")
        assert check.status == OK
        assert check.detail == str(database)


def test_dssp_is_checked_only_for_the_feature_that_needs_it():
    assert not [c for c in check_binaries(["esm2_650m"]) if c.name == "binary mkdssp"]
    checks = check_binaries(["esm2_650m"], features=["secondary_structure"])
    assert [c for c in checks if c.name == "binary mkdssp"]


def test_a_missing_binary_demotes_the_predictors_that_need_it(paths, monkeypatch):
    """SaProt cannot run without foldseek, however complete its environment."""
    monkeypatch.setattr("vepbench.envs.EnvSpec.is_installed", lambda self, p: True)
    monkeypatch.setattr("shutil.which", lambda name: None)
    cache_model(paths, "westlake-repl/SaProt_650M_AF2")
    result = assess(["saprot"], paths=paths)
    assert result.ready == []
    assert "foldseek" in result.blocked["saprot"]


def test_an_unset_token_demotes_the_gated_models(paths, monkeypatch):
    monkeypatch.setattr("vepbench.envs.EnvSpec.is_installed", lambda self, p: True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = assess(["esm3_open"], paths=paths)
    assert result.ready == []
    assert "HF_TOKEN" in result.blocked["esm3_open"]


def test_nothing_is_both_ready_and_missing_something(paths, monkeypatch):
    """The report must not contradict itself anywhere in the registry."""
    monkeypatch.setattr("vepbench.envs.EnvSpec.is_installed", lambda self, p: True)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("FOLDX_BINARY", raising=False)
    result = assess(paths=paths)
    gated = {name for check in result.checks if check.status == MISSING
             for name in check.blocks}
    assert not gated & set(result.ready)


class TestBundledBinaries:
    """Tools the install scripts drop in third_party/ need no PATH export.

    Exporting it by hand works in the shell that installed the tool and is
    missing in a batch job submitted from a fresh login, which fails hours
    later inside the predictor.
    """

    @pytest.fixture
    def foldseek(self, paths):
        binary = paths.third_party / "foldseek" / "bin" / "foldseek"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        return binary

    def test_absent_everywhere_is_missing(self, paths, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert which_including_bundled("foldseek", paths) is None

    def test_found_in_third_party_without_being_on_path(
        self, paths, foldseek, monkeypatch
    ):
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert which_including_bundled("foldseek", paths) == str(foldseek)

    def test_path_wins_when_both_exist(self, paths, foldseek, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/foldseek")
        assert which_including_bundled("foldseek", paths) == "/usr/bin/foldseek"

    def test_a_directory_is_not_a_binary(self, paths, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        (paths.third_party / "foldseek" / "bin" / "foldseek").mkdir(parents=True)
        assert which_including_bundled("foldseek", paths) is None

    def test_a_non_executable_file_is_not_a_binary(self, paths, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        binary = paths.third_party / "foldseek" / "bin" / "foldseek"
        binary.parent.mkdir(parents=True)
        binary.write_text("not executable")
        binary.chmod(0o644)
        assert which_including_bundled("foldseek", paths) is None

    def test_runners_get_it_on_path(self, paths, foldseek, monkeypatch):
        """doctor and the runner must agree about where a tool is."""
        import shutil as real_shutil

        from vepbench.envs import runner_environment
        monkeypatch.setattr("vepbench.envs.PATHS", paths)
        env = runner_environment()
        assert real_shutil.which("foldseek", path=env["PATH"]) == str(foldseek)


class TestDownloadCommands:
    """doctor must name the script that actually fetches the checkpoint.

    It used to print `download_weights_huggingface.sh all` for everything,
    including ESM-2 3B, which that script has never heard of. The reader runs
    it, it succeeds having fetched other models, and the gap is still there.
    """

    def repo_for(self, predictor):
        (repo,) = model_identifiers(predictor)
        return repo

    def test_esm_models_point_at_the_esm_script(self):
        command = download_command(self.repo_for("esm2_3b"))
        assert "download_weights_esm.sh esm2-3b" in command
        assert "huggingface" not in command

    def test_the_five_esm1v_replicates_share_one_target(self):
        commands = {
            download_command(f"facebook/esm1v_t33_650M_UR90S_{n}").split("#")[0]
            for n in range(1, 6)
        }
        assert len(commands) == 1
        assert "esm1v" in commands.pop()

    def test_hugging_face_models_name_their_own_target(self):
        command = download_command(self.repo_for("saprot"))
        assert "download_weights_huggingface.sh saprot" in command

    def test_an_unknown_repository_says_so_rather_than_guessing(self):
        assert "no download script covers" in download_command("nobody/nothing")

    def test_the_catalogue_is_read_from_the_script(self, paths):
        """Parsed, not restated, so the two cannot drift apart."""
        catalogue = hf_catalogue()
        assert catalogue, "the catalogue should not parse as empty"
        script = (Path(__file__).parent.parent
                  / "scripts" / "download_weights_huggingface.sh").read_text()
        for repo, (target, _size) in catalogue.items():
            assert repo in script
            assert f'"{target}:{repo}:' in script

    def test_every_esm_target_is_one_the_script_accepts(self):
        script = (Path(__file__).parent.parent
                  / "scripts" / "download_weights_esm.sh").read_text()
        for prefix, target, _size in ESM_WEIGHT_TARGETS:
            assert f"  {target})" in script or f"{target})" in script, target
            assert prefix in script, prefix

    def test_every_checkpoint_any_predictor_needs_is_covered(self):
        """The property that actually matters: no predictor is left stranded."""
        from vepbench.predictors import REGISTRY

        uncovered = [
            repo for name in REGISTRY for repo in model_identifiers(name)
            if "no download script covers" in download_command(repo)
        ]
        assert not uncovered, f"no download command for: {uncovered}"

    def test_every_command_states_a_size(self):
        from vepbench.predictors import REGISTRY

        for name in REGISTRY:
            for repo in model_identifiers(name):
                assert "GB" in download_command(repo), repo

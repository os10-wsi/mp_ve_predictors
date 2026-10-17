"""Pre-flight checks: does `doctor` tell the truth about what is in place?

The command exists so that a mistake shows up here rather than four hours
into a queued batch, so the tests are mostly about not lying in either
direction: no false "ready", and no advice that would destroy weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vepbench.readiness import (
    MISSING,
    OK,
    WARNING,
    assess,
    check_binaries,
    check_environments,
    check_legacy_cache,
    check_msa_database,
    check_weights,
    environments_for,
    hf_cache_root,
    hf_model_cached,
    model_identifiers,
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
    assert check.fix == "python -m vepbench install-env esm"


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

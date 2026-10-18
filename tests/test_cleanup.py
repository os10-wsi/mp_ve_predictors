"""Removing generated files without removing the expensive ones.

The risk in a cleanup command is asymmetric: leaving scratch behind wastes a
little disk, while deleting a cache costs GPU hours and deleting outputs/
costs the run. So most of these tests are about what survives.
"""

from __future__ import annotations

import pytest

from vepbench.cleanup import (
    CATEGORIES,
    DEFAULT_CATEGORIES,
    PROTECTED,
    human,
    remove,
    survey,
)


@pytest.fixture
def bare(tmp_path, monkeypatch):
    """An unseeded tree. The shared `paths` fixture pre-caches P26678."""
    from vepbench.config import Paths

    for var, sub in (("VEPBENCH_DATA", "data"), ("VEPBENCH_OUTPUTS", "outputs"),
                     ("VEPBENCH_WEIGHTS", "weights"),
                     ("VEPBENCH_THIRD_PARTY", "third_party")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    instance = Paths(root=tmp_path)
    instance.ensure()
    return instance


@pytest.fixture
def populated(bare):
    paths = bare
    """A tree with one of everything a run leaves behind."""
    made = {}

    def put(path, text="x" * 32):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    made["work"] = put(paths.cache / "work" / "P26678" / "esm2_650m" / "scores.csv")
    made["jobs"] = put(paths.outputs / "jobs" / "vepbench-P26678-2026.sh")
    made["logs"] = put(paths.outputs / "jobs" / "logs" / "vepbench-P26678.123.out")
    made["pycache"] = put(paths.root / "vepbench" / "__pycache__" / "cli.pyc")
    made["predictions"] = put(paths.cache / "predictions" / "P26678" / "esm2_650m.csv")
    made["inputs"] = put(paths.cache / "msas" / "P26678.a3m")
    made["results"] = put(paths.outputs / "P26678" / "predictions_long.csv")
    # Expensive things that must never be touched.
    made["weights"] = put(paths.weights / "huggingface" / "hub" / "blob")
    made["venv"] = put(paths.envs / "esm" / ".venv" / "bin" / "python")
    made["definition"] = put(paths.envs / "esm" / "requirements.txt")
    made["uv_cache"] = put(paths.uv_cache / "archive-v0" / "pkg" / "f")
    made["third_party"] = put(paths.third_party / "foldseek" / "bin" / "foldseek")
    made["env_pycache"] = put(
        paths.envs / "esm" / ".venv" / "lib" / "__pycache__" / "torch.pyc"
    )
    return made


def clean(names, paths):
    return remove(survey(names, paths=paths))


class TestWhatSurvives:
    def test_defaults_leave_everything_expensive(self, bare, populated):
        clean(list(DEFAULT_CATEGORIES), bare)
        for key in ("predictions", "inputs", "results", "weights", "venv",
                    "definition", "uv_cache", "third_party"):
            assert populated[key].exists(), f"{key} should have survived"

    def test_defaults_remove_the_scratch(self, bare, populated):
        clean(list(DEFAULT_CATEGORIES), bare)
        for key in ("work", "jobs", "logs", "pycache"):
            assert not populated[key].exists(), f"{key} should have gone"

    def test_installed_package_caches_are_left_alone(self, bare, populated):
        """__pycache__ under envs/ belongs to torch, not to this source tree."""
        clean(["pycache"], bare)
        assert populated["env_pycache"].exists()
        assert not populated["pycache"].exists()

    def test_results_survive_every_category_but_their_own(self, bare, populated):
        clean([n for n in CATEGORIES if n != "results"], bare)
        assert populated["results"].exists()

    def test_results_go_only_when_named(self, bare, populated):
        clean(["results"], bare)
        assert not populated["results"].exists()
        assert populated["weights"].exists()

    def test_all_spares_the_built_environments(self, bare, populated):
        """Rebuilding them is an install each; `all` must not mean that."""
        clean([n for n in CATEGORIES if n not in ("results", "venvs")], bare)
        assert populated["venv"].exists()

    def test_venvs_go_only_when_named_and_keep_the_definitions(self, bare, populated):
        clean(["venvs"], bare)
        assert not populated["venv"].exists()
        assert populated["definition"].exists(), "the definition is in git, keep it"
        assert (bare.envs / "esm").is_dir()

    def test_the_uv_cache_is_its_own_category(self, bare, populated):
        clean(["uv-cache"], bare)
        assert not populated["uv_cache"].exists()
        assert populated["venv"].exists()

    def test_no_category_can_reach_a_protected_directory(self):
        assert not set(CATEGORIES) & set(PROTECTED)

    def test_cleaning_the_jobs_keeps_the_directory_usable(self, bare, populated):
        clean(["jobs", "logs"], bare)
        assert not populated["jobs"].exists()
        assert (bare.outputs / "jobs").is_dir()


class TestSurvey:
    def test_it_reports_sizes(self, bare, populated):
        (found,) = survey(["inputs"], paths=bare)
        assert found.count == 1
        assert found.nbytes == 32

    def test_an_empty_tree_is_not_an_error(self, bare):
        assert all(f.count == 0 for f in survey(list(CATEGORIES), paths=bare))

    def test_older_than_excludes_recent_files(self, bare, populated):
        assert survey(["jobs"], paths=bare)[0].count == 1
        assert survey(["jobs"], paths=bare, older_than_days=1)[0].count == 0

    def test_older_than_includes_old_files(self, bare, populated):
        import os
        import time

        old = time.time() - 10 * 86400
        os.utime(populated["jobs"], (old, old))
        assert survey(["jobs"], paths=bare, older_than_days=1)[0].count == 1

    def test_survey_removes_nothing(self, bare, populated):
        survey(list(CATEGORIES), paths=bare)
        assert all(path.exists() for path in populated.values())


@pytest.mark.parametrize("nbytes,expected", [
    (0, "0 B"), (999, "999 B"), (1500, "1.5 kB"),
    (2_500_000, "2.5 MB"), (3_000_000_000, "3.0 GB"),
])
def test_human_sizes(nbytes, expected):
    assert human(nbytes) == expected


class TestInodeAccounting:
    """On a quota'd filesystem the file count runs out before the bytes do."""

    def test_counts_files_not_just_bytes(self, bare, populated):
        """The venv fixture holds bin/python and a lib/__pycache__ entry."""
        (found,) = survey(["venvs"], paths=bare)
        assert found.nbytes == 64
        assert found.inodes == 6, "the .venv dir, two files, and three dirs"

    def test_a_single_file_is_one_inode(self, bare, populated):
        (found,) = survey(["inputs"], paths=bare)
        assert found.inodes == 1


class TestKeepingEnvironments:
    """Swapping one set of environments for another, on a tight quota."""

    def test_named_environments_are_spared(self, bare, populated):
        for name in ("eve", "proteinmpnn"):
            venv = bare.envs / name / ".venv" / "bin"
            venv.mkdir(parents=True)
            (venv / "python").write_text("x")
        remove(survey(["venvs"], paths=bare, keep=["eve"]))
        assert (bare.envs / "eve" / ".venv").is_dir()
        assert not (bare.envs / "proteinmpnn" / ".venv").exists()
        assert not populated["venv"].exists()

    def test_keeping_nothing_removes_everything(self, bare, populated):
        remove(survey(["venvs"], paths=bare, keep=[]))
        assert not populated["venv"].exists()

    def test_keep_does_not_leak_into_other_categories(self, bare, populated):
        """`--keep esm` must not spare an unrelated file called esm."""
        remove(survey(["work", "venvs"], paths=bare, keep=["esm"]))
        assert populated["venv"].exists(), "envs/esm/.venv is the one to keep"
        assert not populated["work"].exists()

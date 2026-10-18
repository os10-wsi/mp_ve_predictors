"""Removing generated files, without removing the expensive ones.

A run leaves four kinds of file behind, and they are not interchangeable.
Scratch and logs are free to recreate. Cached alignments and predictions cost
network time and GPU hours. Results are the point of the exercise. Weights and
environments are tens of gigabytes and hours of installing.

So nothing here takes a blanket "delete everything": each category is named,
priced, and listed before anything is removed.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from vepbench.config import PATHS, Paths


@dataclass
class Category:
    """One kind of generated file, and what deleting it costs."""

    name: str
    description: str
    cost: str
    #: Included in `clean` with no --what. Only things that cost nothing.
    default: bool = False

    def targets(self, paths: Paths) -> list[Path]:
        raise NotImplementedError


@dataclass
class Found:
    category: str
    paths: list[Path] = field(default_factory=list)
    nbytes: int = 0
    #: Files, not just bytes. On a quota'd project filesystem the file count
    #: is usually what runs out first: one torch environment is about 38,000
    #: of them, and the error names whichever small file was unlucky.
    inodes: int = 0

    @property
    def count(self) -> int:
        return len(self.paths)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*")
               if f.is_file() and not f.is_symlink())


def _inodes(path: Path) -> int:
    """Files and directories, which is what a filesystem quota counts."""
    if path.is_file():
        return 1
    return 1 + sum(1 for _ in path.rglob("*"))


def _work(paths: Paths) -> list[Path]:
    """Per-run scratch: variant lists, raw runner output, job JSON."""
    work = paths.cache / "work"
    return sorted(p for p in work.glob("*") if p.is_dir()) if work.is_dir() else []


def _job_scripts(paths: Paths) -> list[Path]:
    """One generated script per submission, kept so you can read what ran."""
    jobs = paths.outputs / "jobs"
    return sorted(jobs.glob("*.sh")) if jobs.is_dir() else []


def _job_logs(paths: Paths) -> list[Path]:
    logs = paths.outputs / "jobs" / "logs"
    return sorted(p for p in logs.glob("*") if p.is_file()) if logs.is_dir() else []


def _pycache(paths: Paths) -> list[Path]:
    # Never below envs/ or third_party/: those belong to installed packages.
    skip = (paths.envs, paths.third_party)
    return sorted(
        p for p in paths.root.rglob("__pycache__")
        if p.is_dir() and not any(s in p.parents or s == p for s in skip)
    )


def _predictions(paths: Paths) -> list[Path]:
    cached = paths.cache / "predictions"
    return sorted(p for p in cached.glob("*") if p.is_dir()) if cached.is_dir() else []


def _inputs(paths: Paths) -> list[Path]:
    found = []
    for sub in ("msas", "structures", "annotations"):
        directory = paths.cache / sub
        if directory.is_dir():
            found += sorted(p for p in directory.glob("*") if p.is_file())
    return found


def _venvs(paths: Paths) -> list[Path]:
    """Built environments, keeping their definitions.

    Separate from the environment definitions, which are version-controlled
    and tiny. Rebuilding one costs an install; keeping one you will not use
    costs about 38,000 files against the quota.
    """
    if not paths.envs.is_dir():
        return []
    return sorted(p / ".venv" for p in paths.envs.glob("*") if (p / ".venv").is_dir())


def _uv_cache(paths: Paths) -> list[Path]:
    """uv's package cache. Removing it makes the next install re-download."""
    return [paths.uv_cache] if paths.uv_cache.is_dir() else []


def _results(paths: Paths) -> list[Path]:
    """The predictions themselves. Never in any default."""
    if not paths.outputs.is_dir():
        return []
    return sorted(p for p in paths.outputs.glob("*") if p.is_dir() and p.name != "jobs")


CATEGORIES: dict[str, tuple[str, str, bool, object]] = {
    "work": ("per-run scratch under data/cache/work",
             "nothing: it is rebuilt by the next run", True, _work),
    "jobs": ("generated job scripts under outputs/jobs",
             "the record of what was submitted", True, _job_scripts),
    "logs": ("LSF stdout and stderr under outputs/jobs/logs",
             "the record of what the jobs printed", True, _job_logs),
    "pycache": ("__pycache__ directories in the source tree",
                "nothing", True, _pycache),
    "predictions": ("cached per-predictor scores under data/cache/predictions",
                    "GPU time: every predictor reruns from scratch", False, _predictions),
    "inputs": ("cached sequences, structures and alignments",
               "network time, and the alignments take minutes each", False, _inputs),
    "results": ("the prediction tables under outputs/<accession>",
                "THE ACTUAL RESULTS of every run", False, _results),
    "venvs": ("built environments under envs/<name>/.venv",
              "an install each, but they hold ~38,000 files apiece", False, _venvs),
    "uv-cache": ("uv's package cache",
                 "re-downloading packages; it is also what lets uv hardlink "
                 "instead of copying into each environment", False, _uv_cache),
}

#: Deleting these is never this command's job. Weights are tens of gigabytes
#: and doctor already prints an exact, checked rm for duplicates. Environment
#: *definitions* are version-controlled and tiny; the built venvs under them
#: are a category of their own, named `venvs`.
PROTECTED = ("weights", "definitions", "third_party", "databases")

DEFAULT_CATEGORIES = tuple(k for k, v in CATEGORIES.items() if v[2])


def describe(name: str) -> tuple[str, str]:
    description, cost, _, _ = CATEGORIES[name]
    return description, cost


def survey(
    names: list[str], paths: Paths = PATHS, older_than_days: int = 0,
    keep: list[str] | None = None,
) -> list[Found]:
    """What would be removed, and how much space it is using.

    ``keep`` spares named environments from the `venvs` category, which is
    how a set of environments is swapped for another on a filesystem that
    cannot hold them all at once.
    """
    cutoff = time.time() - older_than_days * 86400 if older_than_days else None
    spared = set(keep or ())
    results = []
    for name in names:
        _, _, _, finder = CATEGORIES[name]
        paths_found = [
            p for p in finder(paths)
            if (cutoff is None or p.stat().st_mtime < cutoff)
            # envs/<name>/.venv -> <name>
            and not (name == "venvs" and p.parent.name in spared)
        ]
        results.append(Found(
            category=name, paths=paths_found,
            nbytes=sum(_size(p) for p in paths_found),
            inodes=sum(_inodes(p) for p in paths_found),
        ))
    return results


def remove(found: list[Found]) -> int:
    """Delete what the survey found. Returns the number of paths removed."""
    removed = 0
    for entry in found:
        for path in entry.paths:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
    return removed


def human(nbytes: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if nbytes >= scale:
            return f"{nbytes / scale:.1f} {unit}"
    return f"{nbytes} B"

"""Locating third-party repositories that runners drive.

Several predictors are distributed as a research repository with its own entry
point rather than as a package. Those runners call the upstream code instead
of reimplementing it, which keeps the published behaviour intact. This helper
finds the clone and fails with a precise instruction when it is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_path(job: dict, name: str, env_var: str, clone_url: str, install_script: str) -> Path:
    """Return the checkout directory for a third-party repository."""
    candidates = []
    override = os.environ.get(env_var)
    if override:
        candidates.append(Path(override))
    candidates.append(Path(job["third_party_dir"]) / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"{name} is not installed. Expected a checkout at "
        f"{Path(job['third_party_dir']) / name} or the path in ${env_var}.\n"
        f"  Install it with: bash {install_script}\n"
        f"  Upstream: {clone_url}"
    )


def add_to_path(directory: Path) -> None:
    """Put a repository on sys.path so its modules can be imported."""
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)


def weights_file(job: dict, relative: str, download_script: str) -> Path:
    path = Path(job["weights_dir"]) / relative
    if not path.exists():
        raise SystemExit(
            f"missing weights: {path}\n  Download them with: bash {download_script}"
        )
    return path

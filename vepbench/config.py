"""Filesystem layout, caching and run configuration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("VEPBENCH_ROOT", Path.cwd()))


@dataclass
class Paths:
    """Where everything lives. All of it is gitignored except the repo code."""

    root: Path = field(default_factory=lambda: DEFAULT_ROOT)

    @property
    def data(self) -> Path:
        return Path(os.environ.get("VEPBENCH_DATA", self.root / "data"))

    @property
    def outputs(self) -> Path:
        return Path(os.environ.get("VEPBENCH_OUTPUTS", self.root / "outputs"))

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def structures(self) -> Path:
        return self.cache / "structures"

    @property
    def msas(self) -> Path:
        return self.cache / "msas"

    @property
    def annotations(self) -> Path:
        return self.cache / "annotations"

    @property
    def weights(self) -> Path:
        return Path(os.environ.get("VEPBENCH_WEIGHTS", self.data / "weights"))

    @property
    def third_party(self) -> Path:
        return Path(os.environ.get("VEPBENCH_THIRD_PARTY", self.root / "third_party"))

    @property
    def envs(self) -> Path:
        return self.root / "envs"

    def predictions_cache(self, uniprot_id: str, predictor: str) -> Path:
        return self.cache / "predictions" / uniprot_id / f"{predictor}.csv"

    def output_dir(self, uniprot_id: str) -> Path:
        return self.outputs / uniprot_id

    def ensure(self) -> None:
        for path in (self.data, self.outputs, self.cache, self.structures,
                     self.msas, self.annotations, self.weights):
            path.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


@dataclass
class RunConfig:
    """Everything a single ``vepbench run`` invocation needs to know."""

    predictors: list[str]
    include_wt: bool = False
    force: bool = False
    structure_override: Path | None = None
    sequence_override: str | None = None
    msa_override: Path | None = None
    domain: tuple[int, int] | None = None
    features: list[str] = field(default_factory=list)
    msa_backend: str = "ebi"
    msa_database: str | None = None
    offline: bool = False
    device: str | None = None
    dtype: str | None = None
    batch_size: int | None = None
    paths: Paths = field(default_factory=lambda: PATHS)

    def runtime_options(self) -> dict:
        """Hardware options passed through to every subprocess runner."""
        out = {}
        if self.device:
            out["device"] = self.device
        if self.dtype:
            out["dtype"] = self.dtype
        if self.batch_size:
            out["batch_size"] = self.batch_size
        return out


def cache_key(*parts: object) -> str:
    """Stable short hash for cache filenames."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)

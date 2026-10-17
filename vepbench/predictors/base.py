"""Common predictor interface, metadata and registry."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vepbench import VEPBENCH, with_interpreter
from vepbench.config import PATHS, Paths
from vepbench.envs import EnvSpec, get_env, run_in_env
from vepbench.protein import ProteinContext
from vepbench.schema import ScoreConvention, build_long_frame

log = logging.getLogger(__name__)

RUNNER_DIR = Path(__file__).resolve().parents[2] / "scripts" / "runners"


class PredictorError(RuntimeError):
    pass


@dataclass(frozen=True)
class Availability:
    """Whether a predictor can run here, and if not, exactly what is missing."""

    available: bool
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.available


@dataclass(frozen=True)
class PredictorSpec:
    """Static metadata, and the source for PREDICTORS.md."""

    name: str
    display_name: str
    category: str
    convention: ScoreConvention
    citation: str
    repo_url: str
    license: str
    requires: frozenset[str] = frozenset({"sequence"})
    gpu_required: bool = False
    gpu_recommended: bool = False
    approximate_vram_gb: float | None = None
    env: str = ""
    model_version: str = "unspecified"
    trained_on_megascale: bool = False
    trained_on_proteingym: bool = False
    trained_on_clinvar: bool = False
    supervised: bool = False
    notes: str = ""
    install_hint: str = ""

    @property
    def needs_structure(self) -> bool:
        return "structure" in self.requires

    @property
    def needs_msa(self) -> bool:
        return "msa" in self.requires

    def as_dict(self) -> dict:
        return {
            "name": self.name, "display_name": self.display_name,
            "category": self.category, "score_type": self.convention.score_type,
            "higher_is": self.convention.higher_is,
            "convention_verified": self.convention.verified,
            "convention_source": self.convention.source,
            "citation": self.citation, "repo_url": self.repo_url,
            "license": self.license, "requires": sorted(self.requires),
            "gpu_required": self.gpu_required,
            "gpu_recommended": self.gpu_recommended,
            "approximate_vram_gb": self.approximate_vram_gb,
            "env": self.env, "model_version": self.model_version,
            "trained_on_megascale": self.trained_on_megascale,
            "trained_on_proteingym": self.trained_on_proteingym,
            "trained_on_clinvar": self.trained_on_clinvar,
            "supervised": self.supervised, "notes": self.notes,
        }


class Predictor(ABC):
    """Common interface. Subclasses implement ``_predict_raw``.

    ``predict`` wraps the raw output in the shared schema, so no wrapper can
    get harmonization, the column set or the numbering check wrong.
    """

    spec: PredictorSpec

    def __init__(self, spec: PredictorSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def env_spec(self) -> EnvSpec | None:
        return get_env(self.spec.env) if self.spec.env else None

    def availability(self, protein: ProteinContext, *, paths: Paths = PATHS) -> Availability:
        """Check inputs and installation without running anything."""
        for requirement in sorted(self.spec.requires):
            if not protein.has(requirement):
                return Availability(False, f"requires {requirement}, which was not provided")
        env = self.env_spec()
        if env is not None and not env.is_installed(paths):
            hint = with_interpreter(
                self.spec.install_hint or f"python3 -m vepbench install-env {env.name}"
            )
            state = env.state(paths)
            if state == "incomplete":
                return Availability(False, (
                    f"environment {env.name!r} exists but its install did not finish, "
                    f"so some dependencies are missing. Re-run "
                    f"`{VEPBENCH} install-env {env.name}`."
                ))
            if state == "stale":
                return Availability(False, (
                    f"environment {env.name!r} was built from different requirements. "
                    f"Re-run `{VEPBENCH} install-env {env.name}`."
                ))
            return Availability(False, f"environment {env.name!r} is not installed ({hint})")
        return self._extra_availability(protein, paths=paths)

    def _extra_availability(
        self, protein: ProteinContext, *, paths: Paths = PATHS
    ) -> Availability:
        return Availability(True)

    @abstractmethod
    def _predict_raw(self, protein: ProteinContext, variants: pd.DataFrame) -> pd.DataFrame:
        """Return position / wt_aa / mut_aa / score for the requested variants."""

    def model_version(self, protein: ProteinContext) -> str:
        return self.spec.model_version

    def predict(self, protein: ProteinContext, variants: pd.DataFrame) -> pd.DataFrame:
        raw = self._predict_raw(protein, variants)
        if raw is None or len(raw) == 0:
            raise PredictorError(f"{self.name} returned no rows")
        return build_long_frame(
            uniprot_id=protein.uniprot_id, frame=raw, predictor=self.name,
            convention=self.spec.convention, model_version=self.model_version(protein),
        )


class SubprocessPredictor(Predictor):
    """A predictor that runs inside its own environment.

    The contract with a runner script is small and uniform. It is called as
    ``python -I <runner> --job job.json``; the job holds the sequence, optional
    structure and MSA paths, the variant list and any options; and it writes
    ``scores.csv`` with position, wt_aa, mut_aa and score, plus an optional
    ``meta.json`` carrying ``model_version``.
    """

    def __init__(
        self, spec: PredictorSpec, runner_name: str, options: dict | None = None,
        timeout: int | None = None,
    ) -> None:
        super().__init__(spec)
        self.runner_name = runner_name
        self.options = options or {}
        self.timeout = timeout
        self._last_meta: dict = {}

    @property
    def runner_path(self) -> Path:
        return RUNNER_DIR / self.runner_name

    def _extra_availability(
        self, protein: ProteinContext, *, paths: Paths = PATHS
    ) -> Availability:
        if not self.runner_path.exists():
            return Availability(False, f"runner script missing: {self.runner_path}")
        return Availability(True)

    def build_job(self, protein: ProteinContext, variants: pd.DataFrame, paths: Paths) -> dict:
        work_dir = protein.work_dir / self.name
        work_dir.mkdir(parents=True, exist_ok=True)
        variant_path = work_dir / "variants.csv"
        variants.loc[:, ["position", "wt_aa", "mut_aa"]].to_csv(variant_path, index=False)
        # Hardware options from the command line override per-predictor defaults.
        options = {**self.options, **protein.extras.get("runtime_options", {})}
        return {
            "predictor": self.name, "uniprot_id": protein.uniprot_id,
            "sequence": protein.sequence,
            "structure_path": str(protein.structure.path) if protein.structure else None,
            "structure_chain": protein.structure.chain if protein.structure else None,
            "msa_path": str(protein.msa.path) if protein.msa else None,
            "domain": list(protein.domain) if protein.domain else None,
            "variants_csv": str(variant_path),
            "out_csv": str(work_dir / "scores.csv"),
            "out_meta": str(work_dir / "meta.json"),
            "work_dir": str(work_dir),
            "weights_dir": str(paths.weights),
            "third_party_dir": str(paths.third_party),
            "options": options,
        }

    def _predict_raw(self, protein: ProteinContext, variants: pd.DataFrame) -> pd.DataFrame:
        env = self.env_spec()
        if env is None:
            raise PredictorError(f"{self.name} has no environment configured")
        job = self.build_job(protein, variants, PATHS)
        result = run_in_env(env, self.runner_path, job, paths=PATHS, timeout=self.timeout)
        if result.returncode != 0:
            tail = result.stderr.strip().splitlines()[-6:]
            raise PredictorError(
                f"{self.name} runner exited {result.returncode}: " + " | ".join(tail)
            )
        out_csv = Path(job["out_csv"])
        if not out_csv.exists():
            raise PredictorError(f"{self.name} runner produced no output at {out_csv}")
        frame = pd.read_csv(out_csv)
        meta_path = Path(job["out_meta"])
        self._last_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return frame

    def model_version(self, protein: ProteinContext) -> str:
        return self._last_meta.get("model_version", self.spec.model_version)


class UnavailablePredictor(Predictor):
    """A method that is wrapped but cannot run as configured.

    It never invents a score. It reports what is missing, which the runner
    records in status.json.
    """

    def __init__(self, spec: PredictorSpec, reason: str) -> None:
        super().__init__(spec)
        self.reason = reason

    def _extra_availability(
        self, protein: ProteinContext, *, paths: Paths = PATHS
    ) -> Availability:
        return Availability(False, self.reason)

    def _predict_raw(self, protein: ProteinContext, variants: pd.DataFrame) -> pd.DataFrame:
        raise PredictorError(f"{self.name} is unavailable: {self.reason}")


REGISTRY: dict[str, Predictor] = {}


def register(predictor: Predictor) -> Predictor:
    if predictor.name in REGISTRY:
        raise PredictorError(f"duplicate predictor name {predictor.name!r}")
    REGISTRY[predictor.name] = predictor
    return predictor


def get_predictor(name: str) -> Predictor:
    if name not in REGISTRY:
        raise PredictorError(f"unknown predictor {name!r}; try `{VEPBENCH} list`")
    return REGISTRY[name]


def list_predictors() -> list[Predictor]:
    return [REGISTRY[name] for name in sorted(REGISTRY)]


def resolve_names(names: list[str]) -> list[str]:
    """Expand group names such as ``all`` or ``stability``."""
    resolved: list[str] = []
    for name in names:
        if name == "all":
            resolved.extend(sorted(REGISTRY))
        elif name == "sequence_only":
            resolved.extend(sorted(
                n for n, p in REGISTRY.items() if p.spec.requires == frozenset({"sequence"})
            ))
        elif name == "structure_based":
            resolved.extend(sorted(n for n, p in REGISTRY.items() if p.spec.needs_structure))
        elif name == "msa_based":
            resolved.extend(sorted(n for n, p in REGISTRY.items() if p.spec.needs_msa))
        elif name == "stability":
            resolved.extend(sorted(
                n for n, p in REGISTRY.items() if p.spec.category == "stability"
            ))
        elif name == "no_msa":
            resolved.extend(sorted(n for n, p in REGISTRY.items() if not p.spec.needs_msa))
        else:
            resolved.append(name)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in resolved:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered

"""Isolated per-predictor environments.

Many of these tools pin mutually incompatible versions of torch, numpy or
transformers, and one still targets Python 2.7, so they cannot share an
interpreter. Each predictor, or each group of compatible predictors, gets its
own uv-managed virtual environment under ``envs/<name>/.venv``, and the
orchestrator talks to it over a subprocess with a JSON job file. The
orchestrator never imports a predictor's dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from vepbench.config import PATHS, Paths

log = logging.getLogger(__name__)


class EnvError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvSpec:
    """One isolated environment."""

    name: str
    python: str
    description: str
    notes: str = ""
    requires_gpu: bool = False
    extra_repos: tuple[str, ...] = field(default_factory=tuple)

    def directory(self, paths: Paths = PATHS) -> Path:
        return paths.envs / self.name

    def venv(self, paths: Paths = PATHS) -> Path:
        return self.directory(paths) / ".venv"

    def interpreter(self, paths: Paths = PATHS) -> Path:
        return self.venv(paths) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def requirements(self, paths: Paths = PATHS) -> Path:
        return self.directory(paths) / "requirements.txt"

    def install_script(self, paths: Paths = PATHS) -> Path:
        return self.directory(paths) / "install.sh"

    def is_installed(self, paths: Paths = PATHS) -> bool:
        return self.interpreter(paths).exists()


ENVIRONMENTS: dict[str, EnvSpec] = {}


def register_env(spec: EnvSpec) -> EnvSpec:
    ENVIRONMENTS[spec.name] = spec
    return spec


def get_env(name: str) -> EnvSpec:
    if name not in ENVIRONMENTS:
        raise EnvError(f"unknown environment {name!r}; known: {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[name]


# Grouping follows dependency compatibility, not taxonomy.
register_env(EnvSpec(
    name="esm", python="3.11",
    description="ESM-1v and ESM-2 through transformers, ESM-IF1 through fair-esm, and "
                "ESM C and ESM3 through the EvolutionaryScale esm package.",
    notes="ESM3 and ESM C need the EvolutionaryScale licence accepted on Hugging Face "
          "and HF_TOKEN exported. ESM-IF1 weights come from dl.fbaipublicfiles.com.",
))
register_env(EnvSpec(
    name="esm_if1", python="3.11",
    description="ESM-IF1 only. Separated from the main esm environment because it "
                "needs torch-scatter and torch-sparse, whose prebuilt wheels lag "
                "several torch releases behind.",
    notes="Pinned to torch 2.4.1, for which PyG publishes wheels. Weights come from "
          "dl.fbaipublicfiles.com, not Hugging Face.",
))
register_env(EnvSpec(
    name="huggingface", python="3.11",
    description="transformers models scored with masked marginals: SaProt, ProSST, "
                "ProtT5 and Ankh.",
    notes="SaProt also needs foldseek on PATH for the 3Di structural alphabet.",
))
register_env(EnvSpec(
    name="proteinmpnn", python="3.11",
    description="ProteinMPNN and ThermoMPNN, which share the ProteinMPNN feature code.",
    notes="Both repositories are cloned into third_party/ by the install script.",
))
register_env(EnvSpec(
    name="evcouplings", python="3.10",
    description="EVmutation and the plmc couplings model through evcouplings.",
    notes="plmc is built from source by the install script.",
))
register_env(EnvSpec(
    name="eve", python="3.10",
    description="EVE and popEVE, which share the Bayesian VAE code and torch pin.",
    notes="EVE trains a VAE per protein; budget GPU hours per protein, not per variant.",
    requires_gpu=True,
))
register_env(EnvSpec(
    name="deepsequence", python="2.7",
    description="DeepSequence, published against Theano and Python 2.7.",
    notes="Legacy. Use the container in envs/deepsequence/Dockerfile.",
))
register_env(EnvSpec(
    name="tranception", python="3.10",
    description="Tranception and TranceptEVE; TranceptEVE needs EVE scores first.",
    requires_gpu=True,
))
register_env(EnvSpec(
    name="rasp", python="3.10",
    description="RaSP, which pins an older torch and its own cavity model.",
))
register_env(EnvSpec(
    name="gemme", python="3.11",
    description="GEMME, which drives JET2 and needs a Java runtime.",
    notes="Requires Java 8 or newer.",
))
register_env(EnvSpec(
    name="vespa", python="3.11",
    description="VESPA and VESPA-light from the Rost lab.",
))
register_env(EnvSpec(
    name="s3f", python="3.10",
    description="S3F and S3F-MSA, which need torch-geometric and torchdrug.",
    requires_gpu=True,
))
register_env(EnvSpec(
    name="venusrem", python="3.11",
    description="VenusREM retrieval-enhanced scoring.",
))
register_env(EnvSpec(
    name="promep", python="3.10",
    description="ProMEP, which needs DGL and an SE(3)-Transformer build.",
    requires_gpu=True,
))
register_env(EnvSpec(
    name="metl", python="3.11",
    description="METL pretrained models through metl-pretrained.",
))
register_env(EnvSpec(
    name="stability_dg", python="3.11",
    description="SaProt-dG and ESM3-dG from absolute-stability-predictor, which share "
                "LoRA checkpoints and a stability head.",
    notes="The ESM3 half needs the EvolutionaryScale licence.",
))
register_env(EnvSpec(
    name="ddmut", python="3.11",
    description="DDMut client. DDMut is served as a web API rather than as code.",
    notes="Set DDMUT_ENDPOINT for a local deployment; the public server is rate limited.",
))
register_env(EnvSpec(
    name="foldx", python="3.11",
    description="FoldX driver. The FoldX binary is licensed and is not distributed here.",
    notes="Set FOLDX_BINARY, and FOLDX_ROTABASE if your version needs rotabase.txt.",
))


def uv_executable() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise EnvError(
            "uv is not on PATH. Install it from https://docs.astral.sh/uv/, or build the "
            "environments by hand from envs/<name>/requirements.txt."
        )
    return uv


def create_env(spec: EnvSpec, *, paths: Paths = PATHS, force: bool = False) -> Path:
    """Create the virtual environment and install its pinned requirements."""
    directory = spec.directory(paths)
    if not directory.exists():
        raise EnvError(f"no environment definition at {directory}")
    venv = spec.venv(paths)
    if venv.exists() and force:
        shutil.rmtree(venv)
    if not venv.exists():
        log.info("creating environment %s (python %s)", spec.name, spec.python)
        subprocess.run([uv_executable(), "venv", "--python", spec.python, str(venv)],
                       check=True)
    requirements = spec.requirements(paths)
    if requirements.exists():
        subprocess.run(
            [uv_executable(), "pip", "install", "--python", str(spec.interpreter(paths)),
             "-r", str(requirements)],
            check=True,
        )
    install = spec.install_script(paths)
    if install.exists():
        log.info("running %s", install)
        subprocess.run(["bash", str(install)], check=True, cwd=str(directory))
    return venv


@dataclass
class RunnerResult:
    stdout: str
    stderr: str
    returncode: int


def runner_environment() -> dict:
    """Environment variables every runner subprocess inherits.

    ``HF_HUB_DISABLE_XET`` matters: Hugging Face now serves large files through
    its Xet CDN by default, and some networks allow ``huggingface.co`` while
    blocking that CDN. Falling back to the classic LFS endpoint makes weight
    downloads work on those networks and is harmless elsewhere. Set
    ``VEPBENCH_ALLOW_XET=1`` to leave the default alone.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    if os.environ.get("VEPBENCH_ALLOW_XET") != "1":
        env.setdefault("HF_HUB_DISABLE_XET", "1")
    return env


def run_in_env(
    spec: EnvSpec, runner: Path, job: dict, *, paths: Paths = PATHS,
    timeout: int | None = None, allow_host_python: bool = False,
) -> RunnerResult:
    """Execute a runner script inside the predictor's environment.

    The job is passed as a JSON file path rather than on the command line, so
    sequences and variant lists of any size are fine.
    """
    interpreter = spec.interpreter(paths)
    if not interpreter.exists():
        if not allow_host_python:
            raise EnvError(
                f"environment {spec.name!r} is not installed. Run "
                f"`python -m vepbench install-env {spec.name}`."
            )
        interpreter = Path(sys.executable)

    job_path = Path(job["work_dir"]) / f"{runner.stem}.job.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job, indent=2, default=str))

    command = [str(interpreter), "-I", str(runner), "--job", str(job_path)]
    log.info("running %s in env %s", runner.name, spec.name)
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout,
        env=runner_environment(), check=False,
    )
    if completed.returncode != 0:
        log.error("%s failed (exit %d): %s", runner.name, completed.returncode,
                  completed.stderr.strip()[-2000:])
    return RunnerResult(completed.stdout, completed.stderr, completed.returncode)

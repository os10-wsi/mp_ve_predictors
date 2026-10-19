"""Isolated per-predictor environments.

Many of these tools pin mutually incompatible versions of torch, numpy or
transformers, and one still targets Python 2.7, so they cannot share an
interpreter. Each predictor, or each group of compatible predictors, gets its
own uv-managed virtual environment under ``envs/<name>/.venv``, and the
orchestrator talks to it over a subprocess with a JSON job file. The
orchestrator never imports a predictor's dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from vepbench import VEPBENCH
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

    def conda_packages(self, paths: Paths = PATHS) -> Path:
        """Packages that only conda can supply, such as compiled binaries.

        Used only by the conda backend. With uv these have to be installed by
        hand, and the environment's install.sh says so.
        """
        return self.directory(paths) / "conda-packages.txt"

    def install_script(self, paths: Paths = PATHS) -> Path:
        return self.directory(paths) / "install.sh"

    def marker(self, paths: Paths = PATHS) -> Path:
        """Written only after a complete, successful install.

        The interpreter appears in the very first step, before a single
        requirement is installed, so its presence proves nothing. An install
        that dies partway leaves a usable python and a half-filled
        site-packages, which is worse than no environment at all because
        everything downstream believes it is ready.
        """
        return self.venv(paths) / ".vepbench-install-complete"

    def fingerprint(self, paths: Paths = PATHS) -> str:
        """Hash of the files that define this environment's contents."""
        digest = hashlib.sha256()
        for path in (self.requirements(paths), self.conda_packages(paths)):
            digest.update(path.read_bytes() if path.exists() else b"")
        digest.update(self.python.encode())
        return digest.hexdigest()[:16]

    def installation(self, paths: Paths = PATHS) -> dict | None:
        """What the completion marker records, or None if there is not one."""
        marker = self.marker(paths)
        if not marker.exists():
            return None
        try:
            return json.loads(marker.read_text())
        except json.JSONDecodeError:
            return None

    def state(self, paths: Paths = PATHS) -> str:
        """``missing``, ``incomplete``, ``stale`` or ``ok``."""
        if not self.interpreter(paths).exists():
            return "missing"
        record = self.installation(paths)
        if record is None:
            return "incomplete"
        if record.get("fingerprint") != self.fingerprint(paths):
            return "stale"
        return "ok"

    def is_installed(self, paths: Paths = PATHS) -> bool:
        return self.state(paths) == "ok"


ENVIRONMENTS: dict[str, EnvSpec] = {}


def register_env(spec: EnvSpec) -> EnvSpec:
    ENVIRONMENTS[spec.name] = spec
    return spec


def env_profiles() -> dict[str, list[str]]:
    """Environment sets, named after the predictor categories they serve.

    Derived from the registry rather than listed separately, so adding a
    predictor cannot leave a profile quietly out of date. Intended for a
    filesystem whose inode quota holds only a few environments at a time:
    install one profile, run it, swap it for the next. Cached predictions
    survive the swap, so the results accumulate.
    """
    # Deferred: predictors imports this module, so a top-level import here
    # would be circular.
    from vepbench.predictors import REGISTRY  # noqa: PLC0415

    profiles: dict[str, set[str]] = {}
    for predictor in REGISTRY.values():
        if predictor.spec.env:
            profiles.setdefault(predictor.spec.category, set()).add(predictor.spec.env)
    return {name: sorted(envs) for name, envs in sorted(profiles.items())}


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
    name="foldx", python="3.11",
    description="FoldX driver. The FoldX binary is licensed and is not distributed here.",
    notes="Set FOLDX_BINARY, and FOLDX_ROTABASE if your version needs rotabase.txt.",
))


# Environments can be built with uv or with the conda family. Both end up at
# the same path, ``envs/<name>/.venv``, so nothing downstream has to care
# which was used: a conda environment is created with --prefix rather than a
# name, and its interpreter lands at the same place a venv's would.
UV_BACKEND = "uv"
CONDA_BACKENDS = ("micromamba", "mamba", "conda")
ENV_BACKENDS = (UV_BACKEND,) + CONDA_BACKENDS

# Conda channels, in priority order. bioconda carries the structural-biology
# binaries (foldseek, hmmer, reduce) that have no wheels on PyPI.
CONDA_CHANNELS = ("conda-forge", "bioconda")


# Where uv and the conda family put themselves when installed per-user. A
# cluster login node often has these on PATH while a compute or interactive
# node started from a different profile does not, so looking only at PATH
# reports the backend missing on the machine where the work happens.
BACKEND_SEARCH_PATH = (
    "~/.local/bin", "~/.cargo/bin", "~/bin",
    "~/miniforge3/bin", "~/mambaforge/bin", "~/miniconda3/bin", "~/anaconda3/bin",
    "~/micromamba/bin", "/opt/conda/bin",
)


def which_backend(name: str) -> str | None:
    """Find a backend on PATH, or where its installer usually puts it."""
    found = shutil.which(name)
    if found:
        return found
    for directory in BACKEND_SEARCH_PATH:
        candidate = Path(directory).expanduser() / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def resolve_backend(requested: str | None = None) -> str:
    """Pick the tool that builds environments.

    An explicit choice wins, then ``VEPBENCH_ENV_BACKEND``, then whatever is
    on PATH. uv is preferred when present because it is much faster, but conda
    is a first-class alternative and is the better choice on a cluster where
    the compiled dependencies already come from conda.
    """
    requested = requested or os.environ.get("VEPBENCH_ENV_BACKEND")
    if requested:
        if requested not in ENV_BACKENDS:
            raise EnvError(
                f"unknown environment backend {requested!r}; "
                f"choose one of {', '.join(ENV_BACKENDS)}"
            )
        if which_backend(requested) is None:
            raise EnvError(f"{requested} was requested but is not on PATH")
        return requested
    for candidate in (UV_BACKEND, *CONDA_BACKENDS):
        if which_backend(candidate) is not None:
            return candidate
    raise EnvError(
        "no environment backend found. Install uv with "
        "`curl -LsSf https://astral.sh/uv/install.sh | sh`, or use conda, "
        "mamba or micromamba, or build the environments by hand from "
        "envs/<name>/requirements.txt. If one is already installed, it is "
        "neither on PATH nor in " + ", ".join(BACKEND_SEARCH_PATH) + "."
    )


def torch_index_args() -> list[str]:
    """Where to get torch from, when the default build is wrong for the driver.

    PyPI ships one torch wheel per version, built against one CUDA release.
    If it is newer than the driver, torch loads but reports no GPU and
    everything silently runs on CPU, tens of times slower, with only a
    warning buried in the log.

    Set VEPBENCH_TORCH_INDEX to a PyTorch wheel index matching the driver,
    e.g. https://download.pytorch.org/whl/cu126 for any CUDA 12 driver.
    """
    index = os.environ.get("VEPBENCH_TORCH_INDEX")
    return ["--extra-index-url", index] if index else []


def uv_environment(paths: Paths = PATHS) -> dict:
    """Environment for uv, with its cache beside the venvs it fills.

    Without this uv copies rather than hardlinks, which on a quota'd project
    filesystem fails partway through a large install.
    """
    env = dict(os.environ)
    if not env.get("UV_CACHE_DIR"):
        paths.uv_cache.mkdir(parents=True, exist_ok=True)
        env["UV_CACHE_DIR"] = str(paths.uv_cache)
    return env


def _create_with_uv(spec: EnvSpec, paths: Paths, venv: Path) -> None:
    uv = which_backend(UV_BACKEND)
    env = uv_environment(paths)
    if not venv.exists():
        log.info("creating %s with uv (python %s)", spec.name, spec.python)
        subprocess.run([uv, "venv", "--python", spec.python, str(venv)],
                       check=True, env=env)
    requirements = spec.requirements(paths)
    if requirements.exists():
        subprocess.run(
            [uv, "pip", "install", "--python", str(spec.interpreter(paths)),
             *torch_index_args(), "-r", str(requirements)],
            check=True, env=env,
        )
    conda_packages = spec.conda_packages(paths)
    if conda_packages.exists():
        wanted = _read_package_list(conda_packages)
        log.warning(
            "%s also wants these from conda, which uv cannot supply: %s. Install "
            "them yourself, or rebuild this environment with "
            "`--backend conda`.", spec.name, ", ".join(wanted),
        )


def _create_with_conda(spec: EnvSpec, paths: Paths, venv: Path, backend: str) -> None:
    conda = which_backend(backend)
    # --override-channels so only CONDA_CHANNELS are consulted. Without it
    # conda also reads `defaults` from the user's condarc, which lives on
    # repo.anaconda.com: a host many institutions block or do not resolve,
    # since Anaconda's terms changed for organisations. Everything these
    # environments need is on conda-forge and bioconda, which are served
    # from a different host.
    channels: list[str] = ["--override-channels"]
    for channel in CONDA_CHANNELS:
        channels += ["-c", channel]

    if not venv.exists():
        log.info("creating %s with %s (python %s)", spec.name, backend, spec.python)
        subprocess.run(
            [conda, "create", "--yes", "--prefix", str(venv),
             f"python={spec.python}", "pip", *channels],
            check=True,
        )

    conda_packages = spec.conda_packages(paths)
    if conda_packages.exists():
        wanted = _read_package_list(conda_packages)
        if wanted:
            log.info("installing conda packages for %s: %s", spec.name, ", ".join(wanted))
            subprocess.run(
                [conda, "install", "--yes", "--prefix", str(venv), *channels, *wanted],
                check=True,
            )

    requirements = spec.requirements(paths)
    if requirements.exists():
        # pip inside the prefix, so the pins in requirements.txt still apply.
        subprocess.run(
            [str(spec.interpreter(paths)), "-m", "pip", "install", "-r",
             str(requirements)],
            check=True,
        )


def _reinstall(spec: EnvSpec, paths: Paths, backend: str) -> None:
    """Force a reinstall of everything the environment declares.

    A plain install compares the requirements against the installed
    *metadata*, so an interrupted install that left a .dist-info behind
    without its files reports as satisfied and is skipped. That is the usual
    way an environment on a shared filesystem ends up with a package recorded
    as present but not importable. Reinstalling rewrites the files.
    """
    requirements = spec.requirements(paths)
    if not requirements.exists():
        return
    log.warning("%s is missing dependencies after install; reinstalling", spec.name)
    if backend == UV_BACKEND:
        command = [which_backend(UV_BACKEND), "pip", "install", "--reinstall",
                   "--python", str(spec.interpreter(paths)),
                   *torch_index_args(), "-r", str(requirements)]
    else:
        command = [str(spec.interpreter(paths)), "-m", "pip", "install",
                   "--force-reinstall", *torch_index_args(), "-r", str(requirements)]
    subprocess.run(command, check=True, env=uv_environment(paths))


def _read_package_list(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def create_env(
    spec: EnvSpec, *, paths: Paths = PATHS, force: bool = False,
    backend: str | None = None,
) -> Path:
    """Create the environment and install its pinned requirements."""
    directory = spec.directory(paths)
    if not directory.exists():
        # The definition is version-controlled, so a registered environment
        # with no directory means the checkout lost it rather than that the
        # environment does not exist. Say which, and how to get it back.
        raise EnvError(
            f"no environment definition at {directory}. {spec.name!r} is a known "
            f"environment, so this is missing from the checkout rather than "
            f"unsupported. Restore it with `git checkout -- envs/{spec.name}` "
            f"(or `git pull` if you are behind), then check `git status envs/`."
        )
    chosen = resolve_backend(backend)

    venv = spec.venv(paths)
    if venv.exists() and force:
        shutil.rmtree(venv)
    spec.marker(paths).unlink(missing_ok=True)

    if chosen == UV_BACKEND:
        _create_with_uv(spec, paths, venv)
    else:
        _create_with_conda(spec, paths, venv, chosen)

    install = spec.install_script(paths)
    if install.exists():
        log.info("running %s", install)
        subprocess.run(["bash", str(install)], check=True, cwd=str(directory))

    try:
        verify_env(spec, paths=paths)
    except EnvError:
        _reinstall(spec, paths, chosen)
        try:
            verify_env(spec, paths=paths)
        except EnvError as exc:
            raise EnvError(
                f"{exc}\n\nReinstalling did not fix it, so the environment is "
                f"damaged rather than merely incomplete. Rebuild it from scratch:\n"
                f"  {VEPBENCH} install-env {spec.name} --force"
            ) from exc

    marker = spec.marker(paths)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "fingerprint": spec.fingerprint(paths),
        "backend": chosen,
        "python": spec.python,
        "completed": time.time(),
    }, indent=2))
    return venv


# A few imports per environment that prove the install really finished. These
# are deliberately shallow: enough to catch a pip run that died partway,
# cheap enough to run on every build.
VERIFY_IMPORTS: dict[str, tuple[str, ...]] = {
    "esm": ("torch", "transformers", "huggingface_hub", "filelock"),
    "esm_if1": ("torch", "esm"),
    "huggingface": ("torch", "transformers", "huggingface_hub", "filelock"),
    "proteinmpnn": ("torch", "numpy"),
    "evcouplings": ("evcouplings", "numpy"),
    "eve": ("torch", "numpy"),
    "tranception": ("torch", "transformers"),
    "rasp": ("torch", "numpy"),
    "gemme": ("numpy", "Bio"),
    "vespa": ("torch", "transformers"),
    "s3f": ("torch", "torch_geometric"),
    "venusrem": ("torch", "transformers"),
    "promep": ("torch", "dgl"),
    "metl": ("torch", "metl"),
    "stability_dg": ("torch", "transformers", "peft"),
    "foldx": ("pandas",),
}


def verify_env(spec: EnvSpec, *, paths: Paths = PATHS) -> None:
    """Check the environment can import what it is supposed to have.

    Raises EnvError naming the missing modules, so a half-finished install is
    reported at install time rather than hours later inside a runner.
    """
    modules = VERIFY_IMPORTS.get(spec.name)
    if not modules:
        return
    interpreter = spec.interpreter(paths)
    missing = [
        module for module in modules
        if subprocess.run([str(interpreter), "-I", "-c", f"import {module}"],
                          capture_output=True, check=False).returncode != 0
    ]
    if missing:
        raise EnvError(
            f"environment {spec.name!r} is incomplete: cannot import "
            f"{', '.join(missing)}. A package is recorded as installed but its "
            f"files are missing, which is how an interrupted install on a shared "
            f"filesystem usually ends up."
        )


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

    Tools the install scripts put in ``third_party/`` are added to PATH here
    rather than left to the caller to export. Otherwise foldseek works in the
    shell that installed it and is missing in a batch job submitted from a
    fresh login, which fails hours later inside SaProt.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    bundled = [str(d) for d in sorted(PATHS.third_party.glob("*/bin")) if d.is_dir()]
    if bundled:
        env["PATH"] = os.pathsep.join([*bundled, env.get("PATH", "")]).rstrip(os.pathsep)
    if os.environ.get("VEPBENCH_ALLOW_XET") != "1":
        env.setdefault("HF_HUB_DISABLE_XET", "1")
    # Keep model weights off the home filesystem, which on a cluster is
    # usually a small quota that a couple of downloads will fill.
    if not os.environ.get("HF_HOME"):
        PATHS.hf_home.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(PATHS.hf_home)
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
                f"`{VEPBENCH} install-env {spec.name}`."
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

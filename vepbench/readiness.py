"""Pre-flight checks: is everything in place to actually run?

Answers the question you want answered before committing a batch to a
queue, rather than finding out protein by protein. Nothing here downloads or
installs anything; it only looks.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from vepbench.config import PATHS, Paths
from vepbench.envs import ENVIRONMENTS, resolve_backend
from vepbench.msa import DEFAULT_LOCAL_DATABASES, resolve_database
from vepbench.predictors import REGISTRY, resolve_names
from vepbench.protein import ProteinContext
from vepbench.uniprot import UniProtRecord

OK = "ok"
MISSING = "missing"
WARNING = "warning"


@dataclass
class Check:
    """One thing that is either in place or not."""

    name: str
    status: str
    detail: str = ""
    fix: str = ""

    @property
    def symbol(self) -> str:
        return {OK: "ok", WARNING: "warn", MISSING: "MISSING"}[self.status]


@dataclass
class Readiness:
    checks: list[Check] = field(default_factory=list)
    ready: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)
    # Predictors that are installed but whose weights are not in the cache
    # yet. They run fine with a network and fail without one, so they are
    # neither ready nor blocked.
    needs_download: dict[str, list[str]] = field(default_factory=dict)

    @property
    def problems(self) -> list[Check]:
        return [c for c in self.checks if c.status == MISSING]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARNING]


def hf_cache_root(paths: Paths = PATHS) -> Path:
    """Where Hugging Face keeps downloaded models."""
    return Path(os.environ.get("HF_HOME", paths.hf_home)) / "hub"


def hf_model_cached(repo_id: str, paths: Paths = PATHS) -> bool:
    """Is this Hugging Face repository already downloaded?

    Checked by directory name rather than through huggingface_hub, because
    the orchestrator deliberately does not depend on it. A directory with no
    snapshot in it is a failed download, not a cached model.
    """
    folder = hf_cache_root(paths) / f"models--{repo_id.replace('/', '--')}"
    if not folder.is_dir():
        return False
    snapshots = folder / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def model_identifiers(name: str) -> list[str]:
    """Hugging Face repositories a predictor will want, if any."""
    options = getattr(REGISTRY[name], "options", {}) or {}
    identifiers = options.get("models") or (
        [options["model"]] if options.get("model") else []
    )
    # Only Hugging Face repository ids have an owner prefix. Bare names such
    # as esmc_600m are fetched by their own library and cannot be checked
    # this way.
    return [i for i in identifiers if "/" in i]


# External programs some predictors shell out to.
BINARIES = {
    "foldseek": ("saprot, saprot_1p3b, saprot_dg_masked_marginal",
                 "bash scripts/install_foldseek.sh"),
    "java": ("gemme", "install a JRE, or use --backend conda"),
    "jackhmmer": ("--msa-backend jackhmmer", "conda install -c bioconda hmmer"),
    "hhblits": ("--msa-backend hhblits", "conda install -c bioconda hhsuite"),
    "mkdssp": ("--features secondary_structure", "conda install -c bioconda dssp"),
    "docker": ("deepsequence", "deepsequence runs in a container"),
}


def environments_for(names: list[str]) -> list[str]:
    """Only the environments the requested predictors actually need."""
    return sorted({REGISTRY[n].spec.env for n in names if REGISTRY[n].spec.env})


def check_environments(names: list[str], paths: Paths = PATHS) -> list[Check]:
    explanation = {
        "ok": OK,
        "missing": MISSING,
        "incomplete": MISSING,
        "stale": WARNING,
    }
    checks = []
    for name in environments_for(names):
        spec = ENVIRONMENTS[name]
        state = spec.state(paths)
        checks.append(Check(
            name=f"environment {name}",
            status=explanation[state],
            detail={"ok": "ready", "missing": "not installed",
                    "incomplete": "install did not finish",
                    "stale": "requirements changed since it was built"}[state],
            fix="" if state == "ok" else f"python -m vepbench install-env {name}",
        ))
    return checks


def check_weights(names: list[str], paths: Paths = PATHS) -> list[Check]:
    checks = []
    seen: set[str] = set()
    for name in names:
        for repo in model_identifiers(name):
            if repo in seen:
                continue
            seen.add(repo)
            cached = hf_model_cached(repo, paths=paths)
            checks.append(Check(
                name=f"weights {repo}",
                status=OK if cached else MISSING,
                detail="cached" if cached else "not downloaded",
                fix="" if cached else "bash scripts/download_weights_huggingface.sh all",
            ))
    return checks


def check_binaries(
    names: list[str], msa_backend: str = "colabfold", features: list[str] | None = None,
) -> list[Check]:
    wanted = set()
    for name in names:
        spec = REGISTRY[name].spec
        if "saprot" in name:
            wanted.add("foldseek")
        if name == "gemme":
            wanted.add("java")
        if name == "deepsequence":
            wanted.add("docker")
        if spec.env == "foldx":
            wanted.add("foldx")
    # The local alignment backends and DSSP are chosen by flag rather than by
    # predictor, so they have to be asked for separately.
    if msa_backend in ("jackhmmer", "hhblits") and any(
        REGISTRY[n].spec.needs_msa for n in names
    ):
        wanted.add(msa_backend)
    if "secondary_structure" in (features or []):
        wanted.add("mkdssp")
    checks = []
    for binary in sorted(wanted):
        if binary == "foldx":
            path = os.environ.get("FOLDX_BINARY")
            present = bool(path) and Path(path).exists()
            checks.append(Check(
                name="binary foldx", status=OK if present else MISSING,
                detail=path if present else "FOLDX_BINARY not set or not a file",
                fix="" if present else "export FOLDX_BINARY=/path/to/foldx",
            ))
            continue
        found = shutil.which(binary)
        used_by, fix = BINARIES[binary]
        checks.append(Check(
            name=f"binary {binary}", status=OK if found else MISSING,
            detail=found or f"not on PATH, needed by {used_by}",
            fix="" if found else fix,
        ))
    return checks


def check_msa_database(names: list[str], msa_backend: str) -> list[Check]:
    """A local alignment backend is useless without a database to search.

    These are tens to hundreds of gigabytes and are fetched by hand, so an
    absent one is the likeliest reason a jackhmmer run fails late.
    """
    if msa_backend not in ("jackhmmer", "hhblits"):
        return []
    if not any(REGISTRY[n].spec.needs_msa for n in names):
        return []
    database = resolve_database(msa_backend, None)
    if database:
        return [Check(name=f"{msa_backend} database", status=OK, detail=database)]
    expected = ", ".join(DEFAULT_LOCAL_DATABASES.get(msa_backend, ()))
    return [Check(
        name=f"{msa_backend} database", status=MISSING,
        detail=f"no database found; expected one of {expected} under "
               f"{PATHS.databases}",
        fix="put it there, or set VEPBENCH_MSA_DATABASE, or pass --msa-database",
    )]


def check_credentials(names: list[str], msa_backend: str) -> list[Check]:
    checks = []
    gated = [n for n in names if n in ("esm3_open", "esmc_600m", "esm3_dg",
                                       "esm3_dg_masked_marginal")]
    if gated:
        present = bool(os.environ.get("HF_TOKEN"))
        checks.append(Check(
            name="HF_TOKEN", status=OK if present else MISSING,
            detail="set" if present else f"not set, needed by {', '.join(gated)}",
            fix="" if present else (
                "accept the licence at huggingface.co/EvolutionaryScale/"
                "esm3-sm-open-v1 and export HF_TOKEN"
            ),
        ))
    if msa_backend == "ebi":
        present = bool(os.environ.get("VEPBENCH_EMAIL"))
        checks.append(Check(
            name="VEPBENCH_EMAIL", status=OK if present else MISSING,
            detail="set" if present else "the EBI backend requires a contact address",
            fix="" if present else "export VEPBENCH_EMAIL=you@example.org",
        ))
    return checks


def check_storage(paths: Paths = PATHS) -> list[Check]:
    checks = []
    for label, path in (("weights", paths.weights), ("outputs", paths.outputs)):
        target = path
        while not target.exists() and target != target.parent:
            target = target.parent
        usage = shutil.disk_usage(target)
        free_gb = usage.free / 1e9
        checks.append(Check(
            name=f"disk {label}",
            status=OK if free_gb > 20 else WARNING,
            detail=f"{free_gb:.0f} GB free at {path}",
            fix="" if free_gb > 20 else "free space, or move the directory with "
                                        "VEPBENCH_WEIGHTS / VEPBENCH_OUTPUTS",
        ))
    checks += check_legacy_cache(paths)
    return checks


def check_legacy_cache(paths: Paths = PATHS) -> list[Check]:
    """Notice weights left in the Hugging Face default cache.

    Runs set ``HF_HOME`` to the weights directory, so anything still in
    ``~/.cache/huggingface`` is invisible to them and would be downloaded
    again. Moving it is the fix; deleting it is only safe once the models
    are somewhere a run can see, and this is often the only copy.
    """
    home_cache = Path.home() / ".cache" / "huggingface"
    current = hf_cache_root(paths)
    if not home_cache.is_dir() or current == home_cache / "hub":
        return []
    stranded = sorted(
        folder.name.removeprefix("models--").replace("--", "/")
        for folder in (home_cache / "hub").glob("models--*")
        if (folder / "snapshots").is_dir() and any((folder / "snapshots").iterdir())
    )
    if not stranded:
        return [Check(
            name="old HF cache", status=WARNING,
            detail=f"{home_cache} holds no usable models but is still taking up space",
            fix=f"rm -rf {home_cache}",
        )]
    return [Check(
        name="old HF cache", status=WARNING,
        detail=(f"{len(stranded)} model(s) sit in {home_cache}, which runs no longer "
                f"read: {', '.join(stranded)}"),
        fix=(f"mkdir -p {current} && mv {home_cache / 'hub'}/models--* {current}/  "
             f"# move, do not delete: this may be the only copy"),
    )]


def assess(
    names: list[str] | None = None, *, msa_backend: str = "colabfold",
    features: list[str] | None = None, paths: Paths = PATHS,
) -> Readiness:
    """Work out what is ready to run and what is blocking the rest."""
    names = resolve_names(names or ["all"])
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        # A typo here would otherwise surface as a KeyError, which is a poor
        # answer from the command whose job is catching mistakes early.
        raise ValueError(
            f"unknown predictors: {unknown}. Try `python -m vepbench list`."
        )
    result = Readiness()

    try:
        backend = resolve_backend()
        result.checks.append(Check("environment backend", OK, backend))
    except Exception as exc:  # noqa: BLE001
        result.checks.append(Check("environment backend", MISSING, str(exc),
                                   "install uv or conda"))

    result.checks += check_environments(names, paths)
    result.checks += check_weights(names, paths)
    result.checks += check_binaries(names, msa_backend, features)
    result.checks += check_msa_database(names, msa_backend)
    result.checks += check_credentials(names, msa_backend)
    result.checks += check_storage(paths)

    # Ask each predictor whether it could run, given every input available.
    # This is the same check the runner makes, so it cannot drift from it.
    complete = ProteinContext(
        uniprot=UniProtRecord(uniprot_id="PROBE", sequence="MKVLA"),
        work_dir=paths.cache / "probe",
        structure=object(), msa=object(), domain=(1, 5),  # presence is all that matters
    )
    for name in names:
        availability = REGISTRY[name].availability(complete, paths=paths)
        if not availability.available:
            result.blocked[name] = availability.reason
            continue
        absent = [repo for repo in model_identifiers(name)
                  if not hf_model_cached(repo, paths=paths)]
        if absent:
            result.needs_download[name] = absent
        else:
            result.ready.append(name)
    return result

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

from vepbench import VEPBENCH
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
    # Predictors this check gates. A missing check with blocks set demotes
    # them out of `ready`, so the two halves of the report cannot disagree.
    blocks: tuple[str, ...] = ()

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


def which_including_bundled(binary: str, paths: Paths = PATHS) -> str | None:
    """Find a binary on PATH, or in third_party/ where the scripts put it.

    runner_environment() prepends those directories for the subprocesses, so
    checking only PATH here would report a tool missing that a run will find.
    """
    found = shutil.which(binary)
    if found:
        return found
    for directory in sorted(paths.third_party.glob("*/bin")):
        candidate = directory / binary
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


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
            fix="" if state == "ok" else f"{VEPBENCH} install-env {name}",
        ))
    return checks


# The ESM checkpoints are fetched by their own script, because it runs inside
# the `esm` environment rather than the `huggingface` one. Keyed by repository
# prefix so the five ESM-1v replicates share an entry. A test asserts every
# repository here is one download_weights_esm.sh actually knows.
ESM_WEIGHT_TARGETS = (
    ("facebook/esm1v_t33_650M_UR90S_", "esm1v", "12.5 GB"),
    ("facebook/esm2_t33_650M_UR50D", "esm2", "2.5 GB"),
    ("facebook/esm2_t36_3B_UR50D", "esm2-3b", "11 GB"),
)

HF_CATALOGUE_SCRIPT = "scripts/download_weights_huggingface.sh"
ESM_SCRIPT = "scripts/download_weights_esm.sh"


def hf_catalogue(paths: Paths = PATHS) -> dict[str, tuple[str, str]]:
    """Parse the download script's catalogue: repository -> (target, size).

    Read from the script rather than restated here, so the two cannot drift
    into disagreeing about what exists.
    """
    script = paths.root / HF_CATALOGUE_SCRIPT
    entries: dict[str, tuple[str, str]] = {}
    if not script.is_file():
        return entries
    for raw in script.read_text().splitlines():
        line = raw.strip()
        if not (line.startswith('"') and line.endswith('"') and line.count(":") == 3):
            continue
        name, repo, size, _env = line.strip('"').split(":")
        entries[repo] = (name, f"{size} GB")
    return entries


def download_command(repo: str, paths: Paths = PATHS) -> str:
    """The command that actually fetches this checkpoint.

    Naming the wrong script is worse than naming none: the reader runs it, it
    reports success having fetched something else, and the gap is still there.
    """
    for prefix, target, size in ESM_WEIGHT_TARGETS:
        if repo.startswith(prefix):
            return f"bash {ESM_SCRIPT} {target}    # {size}"
    catalogue = hf_catalogue(paths)
    if repo in catalogue:
        target, size = catalogue[repo]
        return f"bash {HF_CATALOGUE_SCRIPT} {target}    # {size}"
    return f"no download script covers {repo}; it is fetched on first use"


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
                fix="" if cached else download_command(repo, paths),
            ))
    return checks


def check_binaries(
    names: list[str], msa_backend: str = "colabfold", features: list[str] | None = None,
) -> list[Check]:
    wanted: dict[str, set[str]] = {}
    for name in names:
        spec = REGISTRY[name].spec
        if "saprot" in name:
            wanted.setdefault("foldseek", set()).add(name)
        if name == "gemme":
            wanted.setdefault("java", set()).add(name)
        if name == "deepsequence":
            wanted.setdefault("docker", set()).add(name)
        if spec.env == "foldx":
            wanted.setdefault("foldx", set()).add(name)
    # The local alignment backends and DSSP are chosen by flag rather than by
    # predictor, so they have to be asked for separately.
    if msa_backend in ("jackhmmer", "hhblits") and any(
        REGISTRY[n].spec.needs_msa for n in names
    ):
        wanted.setdefault(msa_backend, set())
    if "secondary_structure" in (features or []):
        wanted.setdefault("mkdssp", set())

    checks = []
    for binary, used_by in sorted(wanted.items()):
        blocks = tuple(sorted(used_by))
        if binary == "foldx":
            path = os.environ.get("FOLDX_BINARY")
            present = bool(path) and Path(path).exists()
            checks.append(Check(
                name="binary foldx", status=OK if present else MISSING,
                detail=path if present else "FOLDX_BINARY not set or not a file",
                fix="" if present else "export FOLDX_BINARY=/path/to/foldx",
                blocks=() if present else blocks,
            ))
            continue
        found = which_including_bundled(binary)
        described, fix = BINARIES[binary]
        checks.append(Check(
            name=f"binary {binary}", status=OK if found else MISSING,
            detail=found or f"not on PATH, needed by {', '.join(blocks) or described}",
            fix="" if found else fix,
            blocks=() if found else blocks,
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
            blocks=() if present else tuple(sorted(gated)),
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


def _size(nbytes: int) -> str:
    """Bytes as the unit a reader can act on. 0.0 GB tells them nothing."""
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.1f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.0f} MB"
    return f"{nbytes / 1e3:.0f} kB"


def cached_models(hub: Path) -> dict[str, int]:
    """Repository ids with a usable snapshot under this hub, and their size.

    Symlinks are not counted: Hugging Face points snapshot files at shared
    blobs, so following them would count every file twice.
    """
    found: dict[str, int] = {}
    if not hub.is_dir():
        return found
    for folder in sorted(hub.glob("models--*")):
        snapshots = folder / "snapshots"
        if not snapshots.is_dir() or not any(snapshots.iterdir()):
            continue
        size = sum(f.stat().st_size for f in folder.rglob("*")
                   if f.is_file() and not f.is_symlink())
        found[folder.name.removeprefix("models--").replace("--", "/")] = size
    return found


def check_legacy_cache(paths: Paths = PATHS) -> list[Check]:
    """Notice weights left in the Hugging Face default cache.

    Runs set ``HF_HOME`` to the weights directory, so anything still in
    ``~/.cache/huggingface`` is invisible to them. Whether that matters
    depends on what the new cache already holds, so say which: a model held
    only in the old cache has to be moved, while one held in both was paid
    for twice and is safe to delete. Guessing wrong in either direction
    either loses weights or wastes tens of gigabytes.
    """
    home_cache = Path.home() / ".cache" / "huggingface"
    current = hf_cache_root(paths)
    if not home_cache.is_dir() or current == home_cache / "hub":
        return []
    stranded = cached_models(home_cache / "hub")
    if not stranded:
        return [Check(
            name="old HF cache", status=WARNING,
            detail=f"{home_cache} holds no usable models but is still taking up space",
            fix=f"rm -rf {home_cache}",
        )]

    present = set(cached_models(current)) if current.is_dir() else set()
    duplicated = {k: v for k, v in stranded.items() if k in present}
    only_here = {k: v for k, v in stranded.items() if k not in present}

    checks = []
    if only_here:
        checks.append(Check(
            name="old HF cache", status=WARNING,
            detail=(f"{len(only_here)} model(s) ({_size(sum(only_here.values()))}) exist "
                    f"only in {home_cache}, which runs no longer read: "
                    f"{', '.join(sorted(only_here))}"),
            fix=(f"mkdir -p {current} && mv {home_cache / 'hub'}/models--* {current}/  "
                 f"# move, do not delete: this is the only copy"),
        ))
    if duplicated:
        # One `rm -rf` per line would be six absolute paths of 90 characters.
        # Name the directory once and brace-expand the rest.
        folders = sorted(f"models--{name.replace('/', '--')}" for name in duplicated)
        target = (f"{home_cache / 'hub'}/{folders[0]}" if len(folders) == 1
                  else f"{home_cache / 'hub'}/{{{','.join(folders)}}}")
        checks.append(Check(
            name="duplicate weights", status=WARNING,
            detail=(f"{len(duplicated)} model(s) are in both caches, so "
                    f"{_size(sum(duplicated.values()))} in {home_cache} is wasted: "
                    f"{', '.join(sorted(duplicated))}"),
            fix=f"rm -rf {target}  # the cache runs read already has these",
        ))
    return checks


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
            f"unknown predictors: {unknown}. Try `{VEPBENCH} list`."
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
    # A predictor's own availability() cannot see an absent foldseek or an
    # unset HF_TOKEN, so fold those checks back in. Without this the report
    # contradicts itself: ready here, missing three lines further up.
    gated: dict[str, list[str]] = {}
    for check in result.checks:
        if check.status != MISSING:
            continue
        for blocked_name in check.blocks:
            gated.setdefault(blocked_name, []).append(
                f"{check.name} ({check.fix})" if check.fix else check.name
            )

    for name in names:
        availability = REGISTRY[name].availability(complete, paths=paths)
        if not availability.available:
            result.blocked[name] = availability.reason
            continue
        if name in gated:
            result.blocked[name] = "needs " + "; ".join(gated[name])
            continue
        absent = [repo for repo in model_identifiers(name)
                  if not hf_model_cached(repo, paths=paths)]
        if absent:
            result.needs_download[name] = absent
        else:
            result.ready.append(name)
    return result

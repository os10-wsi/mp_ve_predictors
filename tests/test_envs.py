"""Environment construction, with either uv or the conda family."""

from __future__ import annotations

import pytest

from vepbench.envs import (
    CONDA_CHANNELS,
    ENV_BACKENDS,
    ENVIRONMENTS,
    EnvError,
    create_env,
    get_env,
    resolve_backend,
)


def test_known_backends():
    assert ENV_BACKENDS == ("uv", "micromamba", "mamba", "conda")


def test_explicit_backend_must_exist(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(EnvError, match="not on PATH"):
        resolve_backend("conda")


def test_unknown_backend_is_rejected():
    with pytest.raises(EnvError, match="unknown environment backend"):
        resolve_backend("poetry")


def test_environment_variable_selects_the_backend(monkeypatch):
    monkeypatch.setenv("VEPBENCH_ENV_BACKEND", "mamba")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert resolve_backend() == "mamba"


def test_explicit_choice_beats_the_environment_variable(monkeypatch):
    monkeypatch.setenv("VEPBENCH_ENV_BACKEND", "mamba")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert resolve_backend("conda") == "conda"


def test_uv_is_preferred_when_everything_is_available(monkeypatch):
    monkeypatch.delenv("VEPBENCH_ENV_BACKEND", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert resolve_backend() == "uv"


def test_conda_is_used_when_uv_is_absent(monkeypatch):
    monkeypatch.delenv("VEPBENCH_ENV_BACKEND", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None if name == "uv"
                        else f"/usr/bin/{name}")
    assert resolve_backend() == "micromamba"


def test_no_backend_at_all_explains_the_options(monkeypatch):
    monkeypatch.delenv("VEPBENCH_ENV_BACKEND", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(EnvError, match="no environment backend found"):
        resolve_backend()


def test_bioconda_is_a_channel():
    """The structural-biology binaries have no PyPI wheels."""
    assert "bioconda" in CONDA_CHANNELS
    assert "conda-forge" in CONDA_CHANNELS


@pytest.mark.parametrize("name", sorted(ENVIRONMENTS))
def test_every_environment_has_a_requirements_file(name):
    spec = get_env(name)
    assert spec.requirements().exists(), f"{name} has no requirements.txt"


def test_conda_only_packages_are_declared_where_needed():
    """Tools with no PyPI wheel are listed so the conda backend installs them."""
    expected = {
        "rasp": {"reduce", "openmm", "pdbfixer"},
        "gemme": {"openjdk>=11"},
        "huggingface": {"foldseek"},
        "evcouplings": {"hmmer"},
    }
    for name, wanted in expected.items():
        path = get_env(name).conda_packages()
        assert path.exists(), f"{name} should declare conda packages"
        listed = {
            line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        assert listed == wanted


def test_conda_build_uses_a_prefix_so_the_interpreter_path_is_unchanged(
    monkeypatch, tmp_path
):
    """A conda environment must land where a venv would, or nothing finds it."""
    from vepbench.config import Paths

    calls = []
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("subprocess.run",
                        lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})())

    paths = Paths(root=tmp_path)
    directory = paths.envs / "demo"
    directory.mkdir(parents=True)
    (directory / "requirements.txt").write_text("numpy\n")
    (directory / "conda-packages.txt").write_text("foldseek\n")

    spec = type(get_env("esm"))(name="demo", python="3.11", description="test")
    create_env(spec, paths=paths, backend="conda")

    create = next(c for c in calls if "create" in c)
    assert "--prefix" in create
    assert str(paths.envs / "demo" / ".venv") in create
    assert "python=3.11" in create
    # conda packages installed from the right channels
    install = next(c for c in calls if "install" in c and "foldseek" in c)
    assert "bioconda" in install
    # pip requirements go in through the prefix's own interpreter
    pip = next(c for c in calls if "pip" in c and "-r" in c)
    assert str(spec.interpreter(paths)) == pip[0]


def test_uv_build_warns_about_conda_only_packages(monkeypatch, tmp_path, caplog):
    from vepbench.config import Paths

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("subprocess.run",
                        lambda cmd, **kw: type("R", (), {"returncode": 0})())

    paths = Paths(root=tmp_path)
    directory = paths.envs / "demo"
    directory.mkdir(parents=True)
    (directory / "conda-packages.txt").write_text("foldseek\n")

    spec = type(get_env("esm"))(name="demo", python="3.11", description="test")
    with caplog.at_level("WARNING"):
        create_env(spec, paths=paths, backend="uv")
    assert "foldseek" in caplog.text
    assert "--backend conda" in caplog.text


# --- a half-finished install must not look finished ------------------------

def _definition(tmp_path, name="demo", requirements="numpy\n"):
    from vepbench.config import Paths

    paths = Paths(root=tmp_path)
    directory = paths.envs / name
    directory.mkdir(parents=True)
    (directory / "requirements.txt").write_text(requirements)
    return paths, get_env("esm").__class__(name=name, python="3.11", description="t")


def test_an_interpreter_alone_is_not_an_install(tmp_path):
    """uv creates bin/python before installing anything, so it proves nothing."""
    paths, spec = _definition(tmp_path)
    assert spec.state(paths) == "missing"

    interpreter = spec.interpreter(paths)
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    assert spec.state(paths) == "incomplete"
    assert not spec.is_installed(paths)


def test_the_marker_makes_it_complete(tmp_path):
    import json

    paths, spec = _definition(tmp_path)
    interpreter = spec.interpreter(paths)
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    spec.marker(paths).write_text(json.dumps({"fingerprint": spec.fingerprint(paths)}))
    assert spec.state(paths) == "ok"
    assert spec.is_installed(paths)


def test_changed_requirements_make_it_stale(tmp_path):
    import json

    paths, spec = _definition(tmp_path)
    interpreter = spec.interpreter(paths)
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    spec.marker(paths).write_text(json.dumps({"fingerprint": spec.fingerprint(paths)}))
    assert spec.state(paths) == "ok"

    spec.requirements(paths).write_text("numpy\npandas\n")
    assert spec.state(paths) == "stale"
    assert not spec.is_installed(paths)


def test_verification_names_what_is_missing(tmp_path, monkeypatch):
    from vepbench.envs import verify_env

    paths, _spec = _definition(tmp_path)
    spec = get_env("esm").__class__(name="esm", python="3.11", description="t")
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: type("R", (), {"returncode": 0 if "torch" in cmd[-1] else 1})(),
    )
    with pytest.raises(EnvError) as caught:
        verify_env(spec, paths=paths)
    message = str(caught.value)
    assert "filelock" in message            # the module the cluster was missing
    assert "install-env esm" in message     # and how to fix it


def test_every_heavyweight_environment_is_verified():
    """An environment with no import check could fail silently."""
    from vepbench.envs import VERIFY_IMPORTS

    unchecked = sorted(set(ENVIRONMENTS) - set(VERIFY_IMPORTS) - {"deepsequence"})
    assert unchecked == [], f"these have no verification: {unchecked}"


def test_an_incomplete_environment_makes_predictors_unavailable(tmp_path, monkeypatch):
    """The failure the cluster hit: it must be reported, not hit at run time."""
    from vepbench.config import Paths
    from vepbench.predictors import REGISTRY
    from vepbench.protein import ProteinContext
    from vepbench.uniprot import UniProtRecord

    paths = Paths(root=tmp_path)
    spec = get_env("esm")
    interpreter = spec.interpreter(paths)
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()                      # python present, marker absent

    protein = ProteinContext(
        uniprot=UniProtRecord(uniprot_id="P1", sequence="MKVLA"), work_dir=tmp_path
    )
    availability = REGISTRY["esm2_650m"].availability(protein, paths=paths)
    assert not availability.available
    assert "did not finish" in availability.reason
    assert "install-env esm" in availability.reason

"""Compatibility shim for older toolchains.

All real configuration is in pyproject.toml; this file adds nothing to it.

It exists because pip refuses an editable install when the build backend has
no PEP 660 ``build_editable`` hook and there is no setup.py to fall back to.
That happens on clusters where pip or setuptools predates PEP 660
(setuptools 64, August 2022), or where there is no network so pip cannot
fetch the newer setuptools that pyproject.toml asks for and silently uses an
old one already installed.

With this file present, such a pip falls back to the legacy path instead of
failing. If your setuptools is older than 61 it cannot read the metadata in
pyproject.toml either, and the real fix is:

    python -m pip install --upgrade pip setuptools wheel

The package is pure Python, so it also runs with no installation at all:

    cd /path/to/mp_ve_predictors && python -m vepbench list
"""

from setuptools import setup

setup()

"""vepbench: saturation variant-effect prediction across many predictors."""

import shlex as _shlex
import sys as _sys

__version__ = "0.1.0"

# Commands suggested in error messages have to name an interpreter that
# exists. Plain `python` is absent on many Linux systems, and `python3`
# outside an active virtualenv is the wrong one, so point at whichever
# interpreter is running: it is by definition the one that has vepbench.
PYTHON = _shlex.quote(_sys.executable) if _sys.executable else "python3"
VEPBENCH = f"{PYTHON} -m vepbench"

# Several public data sources reject the default user agent of common HTTP
# clients. AlphaFold DB in particular answers 403 to a bare curl or requests
# UA, which looks like a network block but is not one.
USER_AGENT = (
    "vepbench/0.1.0 (https://github.com/os10-wsi/mp_ve_predictors; "
    "membrane-protein variant-effect benchmarking)"
)
HTTP_HEADERS = {"User-Agent": USER_AGENT}


def with_interpreter(text: str) -> str:
    """Point any suggested command at the interpreter that is running.

    Install hints are written as ``python3 -m vepbench ...`` where they are
    declared, which keeps the predictor specs and the generated PREDICTORS.md
    readable. This rewrites them on the way out, so a message points at the
    interpreter that is running, which is the one that has vepbench.
    """
    return text.replace("python3 -m vepbench", VEPBENCH)

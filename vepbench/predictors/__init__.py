"""Predictor wrappers and the registry.

Importing this package registers every predictor; the modules below are
imported for their side effects.
"""

from __future__ import annotations

from vepbench.predictors.base import (  # noqa: F401
    Availability, Predictor, PredictorSpec, REGISTRY, SubprocessPredictor,
    UnavailablePredictor, get_predictor, list_predictors, register, resolve_names,
)
from vepbench.predictors import baseline  # noqa: F401,E402
from vepbench.predictors import esm  # noqa: F401,E402
from vepbench.predictors import inverse_folding  # noqa: F401,E402
from vepbench.predictors import huggingface_plm  # noqa: F401,E402
from vepbench.predictors import evolutionary  # noqa: F401,E402
from vepbench.predictors import stability  # noqa: F401,E402
from vepbench.predictors import newer  # noqa: F401,E402

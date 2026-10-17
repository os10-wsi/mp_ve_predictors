"""Predictor wrappers and the registry.

Importing this package registers every predictor; the modules below are
imported for their side effects.
"""

from __future__ import annotations

from vepbench.predictors import (
    baseline,  # noqa: F401
    esm,  # noqa: F401
    evolutionary,  # noqa: F401
    huggingface_plm,  # noqa: F401
    inverse_folding,  # noqa: F401
    newer,  # noqa: F401
    stability,  # noqa: F401
)
from vepbench.predictors.base import (  # noqa: F401
    REGISTRY,
    Availability,
    Predictor,
    PredictorSpec,
    SubprocessPredictor,
    UnavailablePredictor,
    get_predictor,
    list_predictors,
    register,
    resolve_names,
)

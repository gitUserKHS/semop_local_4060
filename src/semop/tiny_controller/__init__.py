"""Small learned action-ranking controller.

Only NumPy is required for inference. Import ``semop.tiny_controller.training``
explicitly when PyTorch training support is needed.
"""

from .features import (
    CanonicalAction,
    CanonicalProblemGraph,
    CanonicalRelation,
    canonicalize_problem,
    stable_bucket,
)
from .numpy_runtime import NumpyTinyController, TinyControllerConfig

__all__ = [
    "CanonicalAction",
    "CanonicalProblemGraph",
    "CanonicalRelation",
    "NumpyTinyController",
    "TinyControllerConfig",
    "canonicalize_problem",
    "stable_bucket",
]

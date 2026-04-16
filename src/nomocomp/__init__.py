"""
nomocomp — Exact geometric model comparison from local quadratic structure.

Built on the nomogeo observer-geometry kernel.

Replaces AIC/BIC parameter-count penalties with the actual log-determinant
of the local information matrix — the exact fibre-volume correction in the
Gaussian-fibre sector.

Public API:
    GeometricModelComparator  — compare fitted models by geometric score
    extract_information       — recover the observed information matrix
                                from a fitted statsmodels result
    ModelScore                — score container for one fitted model
    ComparisonResult          — full comparison across a model set
"""

from .comparator import GeometricModelComparator, ComparisonResult
from .extraction import extract_information, ModelScore

__version__ = "0.1.0"

__all__ = [
    "GeometricModelComparator",
    "ComparisonResult",
    "extract_information",
    "ModelScore",
    "__version__",
]

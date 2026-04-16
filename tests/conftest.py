"""Shared fixtures for nomocomp tests."""
import sys
from pathlib import Path
import pytest
import numpy as np

# Resolve paths relative to this file:
#   conftest.py  -> nomocomp/tests/conftest.py
#   nomocomp/    -> workspace/nomocomp/
#   observer_geometry/ -> workspace/observer_geometry/
_NOMOCOMP_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _NOMOCOMP_ROOT.parent
_NOMOGEO_SRC = _WORKSPACE / "observer_geometry" / "src"
_NOMOCOMP_SRC = _NOMOCOMP_ROOT / "src"

for p in (_NOMOGEO_SRC, _NOMOCOMP_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


@pytest.fixture
def ols_three_models():
    """Three OLS models: true, tight nuisance, loose nuisance."""
    import statsmodels.api as sm

    rng = np.random.RandomState(42)
    n = 200

    x1 = rng.randn(n)
    z_tight = rng.randn(n) * 10.0
    z_loose = rng.randn(n) * 0.1
    y = 2.0 * x1 + rng.randn(n)

    X_true = sm.add_constant(x1)
    X_tight = sm.add_constant(np.column_stack([x1, z_tight]))
    X_loose = sm.add_constant(np.column_stack([x1, z_loose]))

    return {
        "true": sm.OLS(y, X_true).fit(),
        "tight": sm.OLS(y, X_tight).fit(),
        "loose": sm.OLS(y, X_loose).fit(),
    }


@pytest.fixture
def ols_pair_same_k():
    """Two OLS models with the same k but different information profiles."""
    import statsmodels.api as sm

    rng = np.random.RandomState(123)
    n = 200

    x1 = rng.randn(n)
    z_tight = rng.randn(n) * 50.0
    z_loose = rng.randn(n) * 0.05
    y = 2.0 * x1 + rng.randn(n)

    X_tight = sm.add_constant(np.column_stack([x1, z_tight]))
    X_loose = sm.add_constant(np.column_stack([x1, z_loose]))

    return {
        "tight": sm.OLS(y, X_tight).fit(),
        "loose": sm.OLS(y, X_loose).fit(),
    }

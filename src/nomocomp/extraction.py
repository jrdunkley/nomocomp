"""
Information matrix extraction from fitted statsmodels results.

The observed Fisher information matrix is the negative Hessian of the
log-likelihood at the MLE.  For model comparison via the Laplace
approximation, we need:

    geometric score = -2 logL + log det(I)

where I is the information matrix for the estimated parameters.

Extraction strategy (ordered by preference):
    1. Direct Hessian:  -model.hessian(params, scale=sigma2_mle)
       Available on OLS and most likelihood models.  For OLS with
       scale argument this gives X^T X / sigma^2_MLE exactly.
    2. Inverse covariance:  inv(result.cov_params()) with scale correction.
       More portable across model classes, but requires care about
       what scale factor statsmodels applied.
    3. Numerical Hessian:  finite-difference approximation of the
       log-likelihood second derivatives.  Fallback when analytic
       Hessian is unavailable.

Honest boundary:
    The extracted information matrix is the LOCAL quadratic object at
    the MLE.  It is exact for Gaussian models.  For non-Gaussian models
    it captures the local curvature but not the global shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelScore:
    """Score container for one fitted model.

    Attributes
    ----------
    label : str
        Human-readable model name.
    log_likelihood : float
        Maximised log-likelihood.
    n_params : int
        Number of estimated parameters (for the information matrix).
    nobs : float
        Number of observations.
    information_matrix : ndarray of shape (n_params, n_params)
        Observed Fisher information at the MLE.
    log_det_information : float
        log det(I) — the exact fibre-volume term.
    geometric_score : float
        -2 logL + log det(I) - k log(2pi).  Lower is better.
        This is the Laplace approximation to -2 log p(data|M)
        under a flat prior (TN4 Prop 3.1).
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    extraction_method : str
        How the information matrix was obtained.
    condition_number : float
        Condition number of the information matrix.
    """

    label: str
    log_likelihood: float
    n_params: int
    nobs: float
    information_matrix: np.ndarray
    log_det_information: float
    geometric_score: float
    aic: float
    bic: float
    extraction_method: str
    condition_number: float


def extract_information(
    result: Any,
    label: str = "",
    method: str = "auto",
) -> ModelScore:
    """Extract the information matrix and compute geometric score.

    Parameters
    ----------
    result : statsmodels Results object
        A fitted model result (OLS, WLS, GLS, GLM, or any
        LikelihoodModel result).
    label : str, optional
        Human-readable name for this model.
    method : str, default "auto"
        Extraction method:
        - "auto" : try direct Hessian first, fall back to cov_params.
        - "hessian" : use model.hessian(params, scale=...).
        - "cov_params" : use inv(result.cov_params()).
        - "numerical" : finite-difference Hessian.

    Returns
    -------
    ModelScore
        Score container with all diagnostics.

    Raises
    ------
    ValueError
        If the information matrix cannot be extracted or is not SPD.
    """
    llf = float(result.llf)
    params = np.asarray(result.params, dtype=float)
    k = len(params)
    nobs = float(result.nobs)

    if method == "auto":
        info, used_method = _auto_extract(result, params)
    elif method == "hessian":
        info, used_method = _hessian_extract(result, params), "hessian"
    elif method == "cov_params":
        info, used_method = _cov_params_extract(result), "cov_params"
    elif method == "numerical":
        info, used_method = _numerical_extract(result, params), "numerical"
    else:
        raise ValueError(
            f"Unknown extraction method '{method}'.  "
            f"Use 'auto', 'hessian', 'cov_params', or 'numerical'."
        )

    # Validate SPD
    info = 0.5 * (info + info.T)  # symmetrise
    eigvals = np.linalg.eigvalsh(info)
    if eigvals[0] <= 0:
        raise ValueError(
            f"Information matrix for '{label}' is not positive definite.  "
            f"Smallest eigenvalue: {eigvals[0]:.2e}.  "
            f"The model may not be identified or the Hessian extraction "
            f"may have failed."
        )

    sign, log_det = np.linalg.slogdet(info)
    log_det_info = float(log_det)
    cond = float(eigvals[-1] / eigvals[0])

    # Laplace approximation to -2 log p(data|M) under flat prior:
    #   -2 logL + log det(I) - k log(2pi)
    # The -k log(2pi) term matters for models of different dimension.
    # See TN4 Prop 3.1: (1/2) log det I_n = (k/2) log n + (1/2) log det J_n.
    geo_score = -2.0 * llf + log_det_info - k * np.log(2 * np.pi)
    aic = float(result.aic)
    bic = float(result.bic)

    return ModelScore(
        label=label or f"model_{k}params",
        log_likelihood=llf,
        n_params=k,
        nobs=nobs,
        information_matrix=info,
        log_det_information=log_det_info,
        geometric_score=geo_score,
        aic=aic,
        bic=bic,
        extraction_method=used_method,
        condition_number=cond,
    )


# ── Extraction backends ─────────────────────────────────────────────


def _auto_extract(
    result: Any, params: np.ndarray
) -> tuple[np.ndarray, str]:
    """Try hessian first, then cov_params, then numerical."""
    try:
        info = _hessian_extract(result, params)
        eigvals = np.linalg.eigvalsh(0.5 * (info + info.T))
        if eigvals[0] > 0:
            return info, "hessian"
    except Exception:
        pass

    try:
        info = _cov_params_extract(result)
        eigvals = np.linalg.eigvalsh(0.5 * (info + info.T))
        if eigvals[0] > 0:
            return info, "cov_params"
    except Exception:
        pass

    try:
        info = _numerical_extract(result, params)
        return info, "numerical"
    except Exception:
        pass

    raise ValueError(
        "Could not extract a valid information matrix.  "
        "Tried hessian, cov_params, and numerical methods."
    )


def _hessian_extract(result: Any, params: np.ndarray) -> np.ndarray:
    """Extract via model.hessian() with MLE scale.

    For OLS with scale argument:
        model.hessian(params, scale=sigma2_mle) = -X^T X / sigma2_mle

    For the concentrated likelihood (scale=None), statsmodels OLS
    returns exactly half the true Hessian due to the chain rule
    through sigma^2(beta).  We avoid this by passing the MLE scale
    explicitly when available.
    """
    model = result.model

    # Try with MLE scale (preferred for OLS/GLS)
    if hasattr(result, "ssr") and hasattr(result, "nobs"):
        sigma2_mle = result.ssr / result.nobs
        if sigma2_mle > 0:
            try:
                H = model.hessian(params, scale=sigma2_mle)
                return np.asarray(-H, dtype=float)
            except TypeError:
                pass  # model.hessian doesn't accept scale kwarg

    # Fall back to default hessian (may need scale correction)
    H = model.hessian(params)
    neg_H = np.asarray(-H, dtype=float)

    # For OLS concentrated likelihood: multiply by 2 to recover
    # the true information matrix
    if _is_ols_like(result) and not _hessian_accepts_scale(model, params):
        neg_H = 2.0 * neg_H

    return neg_H


def _cov_params_extract(result: Any) -> np.ndarray:
    """Extract via inv(cov_params()) with scale correction.

    result.cov_params() = scale * inv(X^T X)  for OLS
    where scale = RSS / (n - k).

    The MLE-based information is X^T X / sigma2_MLE where
    sigma2_MLE = RSS / n.

    So inv(cov_params()) = X^T X / scale = X^T X * (n-k) / RSS
    while the MLE info    = X^T X / (RSS/n) = X^T X * n / RSS.

    The ratio is (n-k)/n, so we correct:
        I_MLE = inv(cov_params) * n / (n - k)
    """
    cp = result.cov_params()
    info = np.linalg.inv(cp)

    # Apply scale correction for OLS-like models
    if _is_ols_like(result):
        n = result.nobs
        k = len(result.params)
        correction = n / (n - k)
        info = info * correction

    return np.asarray(info, dtype=float)


def _numerical_extract(
    result: Any, params: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Numerical Hessian of log-likelihood via central finite differences."""
    model = result.model
    k = len(params)
    info = np.zeros((k, k), dtype=float)

    loglike = model.loglike

    for i in range(k):
        ei = np.zeros(k)
        ei[i] = eps
        for j in range(i, k):
            ej = np.zeros(k)
            ej[j] = eps
            fpp = loglike(params + ei + ej)
            fpm = loglike(params + ei - ej)
            fmp = loglike(params - ei + ej)
            fmm = loglike(params - ei - ej)
            info[i, j] = -(fpp - fpm - fmp + fmm) / (4 * eps * eps)
            info[j, i] = info[i, j]

    return info


# ── Helpers ──────────────────────────────────────────────────────────


def _is_ols_like(result: Any) -> bool:
    """Check if the result comes from an OLS-type model."""
    model_class = type(result.model).__name__
    return model_class in ("OLS", "WLS", "GLS", "GLSAR")


def _hessian_accepts_scale(model: Any, params: np.ndarray) -> bool:
    """Check if model.hessian accepts a scale keyword."""
    import inspect

    sig = inspect.signature(model.hessian)
    return "scale" in sig.parameters

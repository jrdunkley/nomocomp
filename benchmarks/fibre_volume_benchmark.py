"""
Fibre-Volume Benchmark: Exact vs Approximate Complexity Penalties
=================================================================

The core claim: AIC/BIC replace log det(I) with 2k or k*log(n).
This benchmark tests whether the exact fibre volume (log det I)
gives more accurate model rankings than the count surrogates.

Design
------
For same-k model pairs, AIC and BIC give IDENTICAL rankings (the
penalty cancels).  Their pairwise ranking reduces to pure log-likelihood.
The geometric score adds the fibre-volume difference, which captures
information-profile structure that AIC/BIC are blind to.

We validate against:
  (a) The exact Bayesian marginal likelihood (analytically known for
      Gaussian models with conjugate priors)
  (b) Out-of-sample prediction error

Success criterion:
  Among same-k model pairs, the geometric pairwise ranking should
  agree with the exact marginal likelihood more often than the
  log-likelihood-only ranking (which is all AIC/BIC can use).

Epistemic status:
  Exact in the Gaussian-fibre sector (fixed-hidden-precision).
  The OLS Hessian is design-determined, not parameter-dependent.

Honest boundary:
  The geometric score with flat-prior Laplace approximation is not
  BIC-consistent.  For truth recovery in the full model set, BIC
  outperforms.  The geometric score's advantage is PAIRWISE accuracy
  among same-k competitors, where AIC/BIC are blind.
"""
from __future__ import annotations

import json
import numpy as np
from scipy import linalg

import _paths  # noqa: F401 — sets up sys.path portably

import statsmodels.api as sm
import nomocomp


def exact_log_marginal_likelihood(
    y: np.ndarray,
    X: np.ndarray,
    sigma2: float,
    tau2: float,
) -> float:
    """Exact Bayesian marginal likelihood for Gaussian OLS.

    Model: y = X beta + eps,  eps ~ N(0, sigma2 I),  beta ~ N(0, tau2 I)
    Marginal: y ~ N(0, sigma2 I + tau2 X X^T)

    Parameters
    ----------
    y : (n,) array
    X : (n, k) design matrix
    sigma2 : noise variance (known)
    tau2 : prior variance for beta (known)

    Returns
    -------
    float : log p(y | model)
    """
    n = len(y)
    # Marginal covariance: Sigma = sigma2 I + tau2 X X^T
    # Use Woodbury: Sigma^{-1} = (1/sigma2) I - (tau2/sigma2^2) X (I + tau2/sigma2 X^T X)^{-1} X^T
    # log det Sigma = log det(sigma2 I + tau2 X X^T)
    #               = n log(sigma2) + log det(I + tau2/sigma2 X^T X)

    k = X.shape[1]
    XtX = X.T @ X
    M = np.eye(k) + (tau2 / sigma2) * XtX  # (k, k)

    log_det_Sigma = n * np.log(sigma2) + np.linalg.slogdet(M)[1]

    # Sigma^{-1} y via Woodbury
    M_inv = np.linalg.inv(M)
    Sigma_inv_y = y / sigma2 - (tau2 / sigma2**2) * X @ M_inv @ (X.T @ y)
    quad = float(y @ Sigma_inv_y)

    return -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det_Sigma - 0.5 * quad


def pairwise_same_k_benchmark(
    n_trials: int = 1000,
    n: int = 200,
    sigma2: float = 1.0,
    tau2: float = 10.0,
    var_tight: float = 100.0,
    var_loose: float = 0.01,
) -> dict:
    """Pairwise comparison of same-k models against exact marginal likelihood.

    Two models, both with k=3:
      M_tight: y ~ x1 + z_tight   (z_tight ~ N(0, var_tight))
      M_loose: y ~ x1 + z_loose   (z_loose ~ N(0, var_loose))

    True DGP: y = 2*x1 + eps, eps ~ N(0, sigma2).
    Neither model is true; both overfit with one irrelevant predictor.
    The question: which overfitting model is less wrong?

    AIC/BIC give identical penalties (same k), so they rank by logL only.
    The geometric score adds the fibre-volume difference.
    We validate against the exact Bayesian marginal likelihood.
    """
    logL_agrees_exact = 0
    geo_agrees_exact = 0
    geo_and_logL_differ = 0
    geo_correct_when_differ = 0
    logL_correct_when_differ = 0

    # Track details
    details = []

    for seed in range(n_trials):
        rng = np.random.RandomState(seed)

        x1 = rng.randn(n)
        z_tight = rng.randn(n) * np.sqrt(var_tight)
        z_loose = rng.randn(n) * np.sqrt(var_loose)
        y = 2.0 * x1 + np.sqrt(sigma2) * rng.randn(n)

        X_tight = sm.add_constant(np.column_stack([x1, z_tight]))
        X_loose = sm.add_constant(np.column_stack([x1, z_loose]))

        # Fit via statsmodels
        r_tight = sm.OLS(y, X_tight).fit()
        r_loose = sm.OLS(y, X_loose).fit()

        # Extract geometric scores
        try:
            comp = nomocomp.GeometricModelComparator()
            result = comp.compare({"tight": r_tight, "loose": r_loose})
        except Exception:
            continue

        s_tight = next(s for s in result.scores if s.label == "tight")
        s_loose = next(s for s in result.scores if s.label == "loose")

        # Log-likelihood ranking (= AIC/BIC ranking for same k)
        logL_prefers = "loose" if r_loose.llf > r_tight.llf else "tight"

        # Geometric ranking
        geo_prefers = "loose" if s_loose.geometric_score < s_tight.geometric_score else "tight"

        # Exact marginal likelihood ranking
        exact_tight = exact_log_marginal_likelihood(y, X_tight, sigma2, tau2)
        exact_loose = exact_log_marginal_likelihood(y, X_loose, sigma2, tau2)
        exact_prefers = "loose" if exact_loose > exact_tight else "tight"

        # Out-of-sample prediction error
        x1_test = rng.randn(1000)
        z_tight_test = rng.randn(1000) * np.sqrt(var_tight)
        z_loose_test = rng.randn(1000) * np.sqrt(var_loose)
        y_test = 2.0 * x1_test + np.sqrt(sigma2) * rng.randn(1000)

        X_tight_test = sm.add_constant(np.column_stack([x1_test, z_tight_test]))
        X_loose_test = sm.add_constant(np.column_stack([x1_test, z_loose_test]))

        mse_tight = np.mean((y_test - X_tight_test @ r_tight.params) ** 2)
        mse_loose = np.mean((y_test - X_loose_test @ r_loose.params) ** 2)
        oos_prefers = "loose" if mse_loose < mse_tight else "tight"

        # Score
        logL_correct = logL_prefers == exact_prefers
        geo_correct = geo_prefers == exact_prefers
        logL_agrees_exact += int(logL_correct)
        geo_agrees_exact += int(geo_correct)

        if geo_prefers != logL_prefers:
            geo_and_logL_differ += 1
            geo_correct_when_differ += int(geo_correct)
            logL_correct_when_differ += int(logL_correct)

        details.append({
            "seed": seed,
            "logL_prefers": logL_prefers,
            "geo_prefers": geo_prefers,
            "exact_prefers": exact_prefers,
            "oos_prefers": oos_prefers,
            "logL_correct": logL_correct,
            "geo_correct": geo_correct,
            "geo_vs_logL_differ": geo_prefers != logL_prefers,
            "log_det_tight": s_tight.log_det_information,
            "log_det_loose": s_loose.log_det_information,
            "fibre_diff": s_tight.log_det_information - s_loose.log_det_information,
            "logL_tight": float(r_tight.llf),
            "logL_loose": float(r_loose.llf),
            "exact_tight": exact_tight,
            "exact_loose": exact_loose,
            "mse_tight": mse_tight,
            "mse_loose": mse_loose,
        })

    n_valid = len(details)

    # OOS validation
    oos_agrees_geo = sum(1 for d in details if d["oos_prefers"] == d["geo_prefers"])
    oos_agrees_logL = sum(1 for d in details if d["oos_prefers"] == d["logL_prefers"])

    return {
        "name": "pairwise_same_k",
        "description": (
            "Same-k pairwise comparison: geometric fibre correction vs "
            "log-likelihood-only (= AIC/BIC for same k).  "
            "Validated against exact Bayesian marginal likelihood."
        ),
        "config": {
            "n_trials": n_valid,
            "n_obs": n,
            "sigma2": sigma2,
            "tau2": tau2,
            "var_tight": var_tight,
            "var_loose": var_loose,
        },
        "exact_agreement_rate": {
            "geometric": geo_agrees_exact / max(1, n_valid),
            "logL_only": logL_agrees_exact / max(1, n_valid),
        },
        "when_criteria_disagree": {
            "n_disagreements": geo_and_logL_differ,
            "geo_correct": geo_correct_when_differ,
            "logL_correct": logL_correct_when_differ,
            "geo_correct_rate": (
                geo_correct_when_differ / max(1, geo_and_logL_differ)
            ),
            "logL_correct_rate": (
                logL_correct_when_differ / max(1, geo_and_logL_differ)
            ),
        },
        "oos_agreement_rate": {
            "geometric": oos_agrees_geo / max(1, n_valid),
            "logL_only": oos_agrees_logL / max(1, n_valid),
        },
        "counts": {
            "geo_agrees_exact": geo_agrees_exact,
            "logL_agrees_exact": logL_agrees_exact,
            "total": n_valid,
        },
    }


def prior_sensitivity_sweep(
    tau2_values: list[float] | None = None,
    n_trials: int = 500,
    n: int = 200,
) -> dict:
    """Sweep the prior width tau2 to check robustness.

    The geometric score (flat-prior Laplace) should track the exact
    marginal likelihood better when tau2 is large (diffuse prior,
    where the Laplace approximation is most accurate).
    """
    if tau2_values is None:
        tau2_values = [0.1, 1.0, 10.0, 100.0, 1000.0]

    sweep = {}
    for tau2 in tau2_values:
        result = pairwise_same_k_benchmark(
            n_trials=n_trials, n=n, tau2=tau2
        )
        sweep[tau2] = {
            "tau2": tau2,
            "geo_exact_agreement": result["exact_agreement_rate"]["geometric"],
            "logL_exact_agreement": result["exact_agreement_rate"]["logL_only"],
            "geo_advantage_when_differ": result["when_criteria_disagree"]["geo_correct_rate"],
            "n_disagreements": result["when_criteria_disagree"]["n_disagreements"],
        }

    return {
        "name": "prior_sensitivity",
        "description": "Exact-agreement rate as a function of prior width tau^2.",
        "sweep": sweep,
    }


def information_contrast_sweep(
    var_ratio_values: list[float] | None = None,
    n_trials: int = 500,
    n: int = 200,
) -> dict:
    """Sweep the variance ratio between tight and loose predictors.

    Larger ratio = more different information profiles = bigger
    difference in fibre volumes = more opportunity for the geometric
    score to outperform.
    """
    if var_ratio_values is None:
        var_ratio_values = [2.0, 10.0, 100.0, 1000.0, 10000.0]

    sweep = {}
    for ratio in var_ratio_values:
        # Keep geometric mean fixed: sqrt(var_tight * var_loose) = 1
        var_tight = np.sqrt(ratio)
        var_loose = 1.0 / np.sqrt(ratio)
        result = pairwise_same_k_benchmark(
            n_trials=n_trials, n=n,
            var_tight=var_tight, var_loose=var_loose,
        )
        sweep[ratio] = {
            "var_ratio": ratio,
            "var_tight": var_tight,
            "var_loose": var_loose,
            "geo_exact_agreement": result["exact_agreement_rate"]["geometric"],
            "logL_exact_agreement": result["exact_agreement_rate"]["logL_only"],
            "geo_advantage_when_differ": result["when_criteria_disagree"]["geo_correct_rate"],
            "n_disagreements": result["when_criteria_disagree"]["n_disagreements"],
            "geo_oos_agreement": result["oos_agreement_rate"]["geometric"],
            "logL_oos_agreement": result["oos_agreement_rate"]["logL_only"],
        }

    return {
        "name": "information_contrast",
        "description": (
            "Exact-agreement rate as a function of the variance ratio "
            "between tight and loose nuisance predictors."
        ),
        "sweep": sweep,
    }


def main():
    print("Fibre-Volume Benchmark: Exact vs Approximate Penalties")
    print("=" * 60)
    print()

    # --- Core benchmark: same-k pairwise ---
    print("Core Benchmark: Same-k pairwise comparison")
    print("-" * 60)
    core = pairwise_same_k_benchmark(n_trials=1000, n=200)

    print(f"  Trials: {core['config']['n_trials']}")
    print(f"  Agreement with exact marginal likelihood:")
    print(f"    Geometric (fibre-corrected): {core['exact_agreement_rate']['geometric']:.1%}")
    print(f"    Log-likelihood only (=AIC/BIC): {core['exact_agreement_rate']['logL_only']:.1%}")
    print()
    print(f"  When geometric and logL disagree ({core['when_criteria_disagree']['n_disagreements']} cases):")
    print(f"    Geometric correct: {core['when_criteria_disagree']['geo_correct_rate']:.1%}")
    print(f"    LogL-only correct: {core['when_criteria_disagree']['logL_correct_rate']:.1%}")
    print()
    print(f"  Agreement with out-of-sample MSE:")
    print(f"    Geometric: {core['oos_agreement_rate']['geometric']:.1%}")
    print(f"    LogL-only: {core['oos_agreement_rate']['logL_only']:.1%}")

    # --- Information contrast sweep ---
    print(f"\n{'='*60}")
    print("Information Contrast Sweep")
    print("-" * 60)
    contrast = information_contrast_sweep(n_trials=500, n=200)
    print(f"  {'ratio':>10s}  {'GEO exact':>10s}  {'logL exact':>10s}  "
          f"{'GEO|differ':>10s}  {'n_differ':>8s}  {'GEO oos':>8s}  {'logL oos':>8s}")
    for ratio, vals in sorted(contrast["sweep"].items()):
        print(
            f"  {vals['var_ratio']:10.0f}  "
            f"{vals['geo_exact_agreement']:10.1%}  "
            f"{vals['logL_exact_agreement']:10.1%}  "
            f"{vals['geo_advantage_when_differ']:10.1%}  "
            f"{vals['n_disagreements']:8d}  "
            f"{vals['geo_oos_agreement']:8.1%}  "
            f"{vals['logL_oos_agreement']:8.1%}"
        )

    # --- Prior sensitivity sweep ---
    print(f"\n{'='*60}")
    print("Prior Sensitivity Sweep")
    print("-" * 60)
    prior = prior_sensitivity_sweep(n_trials=500, n=200)
    print(f"  {'tau2':>10s}  {'GEO exact':>10s}  {'logL exact':>10s}  "
          f"{'GEO|differ':>10s}  {'n_differ':>8s}")
    for tau2, vals in sorted(prior["sweep"].items()):
        print(
            f"  {vals['tau2']:10.1f}  "
            f"{vals['geo_exact_agreement']:10.1%}  "
            f"{vals['logL_exact_agreement']:10.1%}  "
            f"{vals['geo_advantage_when_differ']:10.1%}  "
            f"{vals['n_disagreements']:8d}"
        )

    # Save results
    save_data = {
        "core_benchmark": core,
        "information_contrast": {
            str(k): v for k, v in contrast["sweep"].items()
        },
        "prior_sensitivity": {
            str(k): v for k, v in prior["sweep"].items()
        },
    }

    outpath = str(_paths.BENCHMARKS_DIR / "fibre_volume_results.json")
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()

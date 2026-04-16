"""
Claim Tightening: When does geometric ≠ exact Bayesian ranking?
================================================================

The geometric score uses log det(I) where I = X^T X / sigma2_MLE.
The exact Bayesian marginal likelihood uses log det(I_k + (tau2/sigma2) X^T X).

For same-k models A and B, define:
  geo_prefers_A  iff  -2 logL_A + log det(I_A) < -2 logL_B + log det(I_B)
  exact_prefers_A iff  log p(y|A) > log p(y|B)

These CAN disagree when:
  - The prior term (tau2-dependent) creates a different ranking than
    the flat-prior limit
  - Specifically, if log det(I + tau2/sigma2 * X^T X) is not monotone
    in log det(X^T X / sigma2) across the two models

We test three families:
  1. Substituted predictor (our benchmark): same X except one column swapped
  2. Disjoint predictors: entirely different predictor sets
  3. Correlated predictors: overlapping but not identical sets

For each, we measure how often geo and exact agree as a function of tau2.
"""
from __future__ import annotations

import json
import numpy as np

import _paths  # noqa: F401 — sets up sys.path portably

import statsmodels.api as sm
import nomocomp


def exact_log_marginal(y, X, sigma2, tau2):
    """Exact log p(y|M) for Gaussian OLS with N(0, tau2*I) prior."""
    n, k = X.shape
    M = np.eye(k) + (tau2 / sigma2) * (X.T @ X)
    log_det = n * np.log(sigma2) + np.linalg.slogdet(M)[1]
    M_inv = np.linalg.inv(M)
    Sinv_y = y / sigma2 - (tau2 / sigma2**2) * X @ M_inv @ (X.T @ y)
    quad = float(y @ Sinv_y)
    return -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad


def family_substituted(seed, n=200):
    """Our benchmark family: same X except one column swapped."""
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    z_a = rng.randn(n) * 50.0   # tight
    z_b = rng.randn(n) * 0.05   # loose
    y = 2.0 * x1 + rng.randn(n)
    X_a = sm.add_constant(np.column_stack([x1, z_a]))
    X_b = sm.add_constant(np.column_stack([x1, z_b]))
    return y, X_a, X_b


def family_disjoint(seed, n=200):
    """Entirely different predictor sets (same k)."""
    rng = np.random.RandomState(seed)
    # Model A: two standard predictors
    xa1 = rng.randn(n) * 3.0
    xa2 = rng.randn(n) * 0.5
    # Model B: two different predictors with different info
    xb1 = rng.randn(n) * 0.3
    xb2 = rng.randn(n) * 10.0
    y = rng.randn(n)  # pure noise — neither model is true
    X_a = sm.add_constant(np.column_stack([xa1, xa2]))
    X_b = sm.add_constant(np.column_stack([xb1, xb2]))
    return y, X_a, X_b


def family_correlated(seed, n=200):
    """Overlapping predictor sets with correlations."""
    rng = np.random.RandomState(seed)
    # Shared latent structure
    z = rng.randn(n)
    x_shared = rng.randn(n) + 0.5 * z
    # Model A: shared + high-info unique
    xa_unique = rng.randn(n) * 20.0 + z
    # Model B: shared + low-info unique
    xb_unique = rng.randn(n) * 0.1 + 0.3 * z
    y = 1.5 * x_shared + rng.randn(n)
    X_a = sm.add_constant(np.column_stack([x_shared, xa_unique]))
    X_b = sm.add_constant(np.column_stack([x_shared, xb_unique]))
    return y, X_a, X_b


def family_adversarial(seed, n=200):
    """Adversarial: construct models where logL ranking and det(I) ranking
    are opposed, but the det(I) margin is small relative to the prior term.

    Model A: one medium predictor (var=1)
    Model B: one predictor with var chosen so det(I_B) > det(I_A) but
             the prior-dependent term could reverse it.
    """
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    # Model A: medium-info additional predictor
    za = rng.randn(n) * 1.0
    # Model B: slightly-higher-info additional predictor
    # (close enough that the prior could matter)
    zb = rng.randn(n) * 1.5
    y = 2.0 * x1 + rng.randn(n)
    X_a = sm.add_constant(np.column_stack([x1, za]))
    X_b = sm.add_constant(np.column_stack([x1, zb]))
    return y, X_a, X_b


def test_family(family_fn, family_name, n_trials=500, tau2_values=None):
    """Test geo vs exact agreement for a model family across tau2 values."""
    if tau2_values is None:
        tau2_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    sigma2 = 1.0
    results = {}

    for tau2 in tau2_values:
        n_agree = 0
        n_total = 0

        for seed in range(n_trials):
            try:
                y, X_a, X_b = family_fn(seed)
                r_a = sm.OLS(y, X_a).fit()
                r_b = sm.OLS(y, X_b).fit()

                comp = nomocomp.GeometricModelComparator()
                result = comp.compare({"A": r_a, "B": r_b})
                geo_prefers = result.geometric_ranking[0]

                exact_a = exact_log_marginal(y, X_a, sigma2, tau2)
                exact_b = exact_log_marginal(y, X_b, sigma2, tau2)
                exact_prefers = "A" if exact_a > exact_b else "B"

                n_total += 1
                if geo_prefers == exact_prefers:
                    n_agree += 1
            except Exception:
                pass

        if n_total > 0:
            results[tau2] = {
                "tau2": tau2,
                "agreement_rate": n_agree / n_total,
                "n_agree": n_agree,
                "n_total": n_total,
            }

    return {"family": family_name, "results": results}


def main():
    print("Claim Tightening: Geometric vs Exact Bayesian Agreement")
    print("=" * 60)

    families = [
        (family_substituted, "substituted (our benchmark)"),
        (family_disjoint, "disjoint predictors"),
        (family_correlated, "correlated predictors"),
        (family_adversarial, "adversarial (small contrast)"),
    ]

    all_results = {}

    for family_fn, name in families:
        print(f"\n{name}")
        print("-" * 50)
        result = test_family(family_fn, name, n_trials=500)
        all_results[name] = result

        print(f"  {'tau2':>10s}  {'agreement':>10s}  {'n':>5s}")
        for tau2, vals in sorted(result["results"].items()):
            print(
                f"  {vals['tau2']:10.2f}  "
                f"{vals['agreement_rate']:10.1%}  "
                f"{vals['n_total']:5d}"
            )

    # Mathematical analysis: when does disagreement happen?
    print(f"\n{'='*60}")
    print("Analysis: Disagreement cases in disjoint family")
    print("-" * 50)

    sigma2 = 1.0
    disagreement_cases = []
    for seed in range(500):
        try:
            y, X_a, X_b = family_disjoint(seed)
            r_a = sm.OLS(y, X_a).fit()
            r_b = sm.OLS(y, X_b).fit()

            comp = nomocomp.GeometricModelComparator()
            result = comp.compare({"A": r_a, "B": r_b})
            geo_prefers = result.geometric_ranking[0]

            # Check across several tau2 values
            for tau2 in [0.1, 1.0, 10.0]:
                exact_a = exact_log_marginal(y, X_a, sigma2, tau2)
                exact_b = exact_log_marginal(y, X_b, sigma2, tau2)
                exact_prefers = "A" if exact_a > exact_b else "B"

                if geo_prefers != exact_prefers:
                    s_a = next(s for s in result.scores if s.label == "A")
                    s_b = next(s for s in result.scores if s.label == "B")
                    disagreement_cases.append({
                        "seed": seed,
                        "tau2": tau2,
                        "geo_prefers": geo_prefers,
                        "exact_prefers": exact_prefers,
                        "logL_diff": r_a.llf - r_b.llf,
                        "fibre_diff": s_a.log_det_information - s_b.log_det_information,
                        "exact_diff": exact_a - exact_b,
                    })
        except Exception:
            pass

    if disagreement_cases:
        print(f"  Found {len(disagreement_cases)} disagreement cases")
        for d in disagreement_cases[:10]:
            print(
                f"  seed={d['seed']}, tau2={d['tau2']:.1f}: "
                f"geo→{d['geo_prefers']}, exact→{d['exact_prefers']}, "
                f"logL_diff={d['logL_diff']:.4f}, "
                f"fibre_diff={d['fibre_diff']:.2f}, "
                f"exact_diff={d['exact_diff']:.4f}"
            )
    else:
        print("  No disagreement cases found")

    # Save
    outpath = str(_paths.BENCHMARKS_DIR / "claim_tightening_results.json")
    save_data = {}
    for name, result in all_results.items():
        save_data[name] = {
            str(k): v for k, v in result["results"].items()
        }
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()

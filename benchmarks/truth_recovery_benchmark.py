"""
Truth-Recovery Benchmark: Geometric Score vs AIC/BIC
=====================================================

Simulate from a known Gaussian DGP, fit competing statsmodels OLS models,
and measure how often each criterion recovers the true generating model.

Design
------
True DGP:  y = beta_1 * x_1 + epsilon,  epsilon ~ N(0, 1)

Three candidate models:
  M_true  :  y ~ x_1                    (the truth)
  M_tight :  y ~ x_1 + z_tight          (1 high-Fisher-info nuisance predictor)
  M_loose :  y ~ x_1 + z_loose          (1 low-Fisher-info nuisance predictor)

z_tight has high variance (var >> 1), so its column in X has large norm,
producing a large eigenvalue in X^T X and hence in the information matrix.
z_loose has low variance (var << 1), small norm, small eigenvalue.

Both M_tight and M_loose have the same parameter count (k=3 including
intercept), so AIC and BIC give them IDENTICAL penalties.  The geometric
score distinguishes them via log det(I), which captures the actual
information content of the nuisance predictor.

Success criterion:
  The geometric score should recover M_true more often than AIC/BIC in
  the regime where nuisance-predictor information profiles differ.

Epistemic status:
  Exact in the Gaussian-fibre sector.  The models are all OLS on Gaussian
  data, so the information matrix is the exact local quadratic object.

Honest boundary:
  This benchmark tests the fixed-hidden-precision sector (OLS Hessian is
  design-determined, not parameter-dependent).  Variable-precision
  (GLM, ARIMA) is a separate claim.
"""
from __future__ import annotations

import json
import numpy as np

import _paths  # noqa: F401 — sets up sys.path portably

import statsmodels.api as sm
import nomocomp


def run_single_trial(
    seed: int,
    n: int = 200,
    beta_signal: float = 2.0,
    sigma: float = 1.0,
    var_tight: float = 100.0,
    var_loose: float = 0.01,
) -> dict:
    """Run one trial of the truth-recovery benchmark.

    Returns a dict with the winner under each criterion and diagnostics.
    """
    rng = np.random.RandomState(seed)

    # Signal predictor
    x1 = rng.randn(n)

    # Nuisance predictors with different information profiles
    z_tight = rng.randn(n) * np.sqrt(var_tight)  # high-info
    z_loose = rng.randn(n) * np.sqrt(var_loose)   # low-info

    # True DGP: y = beta*x1 + noise
    y = beta_signal * x1 + sigma * rng.randn(n)

    # Fit three models via statsmodels OLS
    X_true = sm.add_constant(x1)
    X_tight = sm.add_constant(np.column_stack([x1, z_tight]))
    X_loose = sm.add_constant(np.column_stack([x1, z_loose]))

    r_true = sm.OLS(y, X_true).fit()
    r_tight = sm.OLS(y, X_tight).fit()
    r_loose = sm.OLS(y, X_loose).fit()

    # Compare via nomocomp
    comp = nomocomp.GeometricModelComparator()
    result = comp.compare({
        "true": r_true,
        "tight": r_tight,
        "loose": r_loose,
    })

    # Extract winners
    geo_winner = result.geometric_ranking[0]
    aic_winner = result.aic_ranking[0]
    bic_winner = result.bic_ranking[0]

    # Diagnostics
    scores_dict = {s.label: s for s in result.scores}

    return {
        "seed": seed,
        "geo_winner": geo_winner,
        "aic_winner": aic_winner,
        "bic_winner": bic_winner,
        "geo_correct": geo_winner == "true",
        "aic_correct": aic_winner == "true",
        "bic_correct": bic_winner == "true",
        "geo_scores": {s.label: s.geometric_score for s in result.scores},
        "aic_scores": {s.label: s.aic for s in result.scores},
        "bic_scores": {s.label: s.bic for s in result.scores},
        "log_det_info": {s.label: s.log_det_information for s in result.scores},
        "log_likelihood": {s.label: s.log_likelihood for s in result.scores},
        "n_reversals": len(result.ranking_reversals),
        "branch_reversal": result.branch_reversal.reversal if result.branch_reversal else None,
    }


def run_equal_k_benchmark(
    n_trials: int = 500,
    n: int = 200,
    var_tight: float = 100.0,
    var_loose: float = 0.01,
) -> dict:
    """Equal-k benchmark: M_tight and M_loose have the same parameter count.

    AIC/BIC cannot distinguish between them.  The geometric score can.
    """
    results = []
    for seed in range(n_trials):
        try:
            r = run_single_trial(seed, n=n, var_tight=var_tight, var_loose=var_loose)
            results.append(r)
        except Exception as e:
            pass

    n_valid = len(results)
    geo_correct = sum(1 for r in results if r["geo_correct"])
    aic_correct = sum(1 for r in results if r["aic_correct"])
    bic_correct = sum(1 for r in results if r["bic_correct"])

    # When AIC/BIC pick wrong, which wrong model do they pick?
    aic_picks_tight = sum(1 for r in results if r["aic_winner"] == "tight")
    aic_picks_loose = sum(1 for r in results if r["aic_winner"] == "loose")
    geo_picks_tight = sum(1 for r in results if r["geo_winner"] == "tight")
    geo_picks_loose = sum(1 for r in results if r["geo_winner"] == "loose")

    # Count ranking disagreements
    geo_aic_disagree = sum(
        1 for r in results if r["geo_winner"] != r["aic_winner"]
    )
    geo_bic_disagree = sum(
        1 for r in results if r["geo_winner"] != r["bic_winner"]
    )

    return {
        "name": "equal_k_benchmark",
        "description": (
            "M_tight and M_loose have k=3 parameters each.  "
            "AIC/BIC give them identical penalties.  "
            "The geometric score uses log det(I) to distinguish them."
        ),
        "config": {
            "n_trials": n_trials,
            "n_valid": n_valid,
            "n_obs": n,
            "var_tight": var_tight,
            "var_loose": var_loose,
        },
        "truth_recovery_rate": {
            "geometric": geo_correct / max(1, n_valid),
            "aic": aic_correct / max(1, n_valid),
            "bic": bic_correct / max(1, n_valid),
        },
        "wrong_model_picks": {
            "aic_picks_tight": aic_picks_tight,
            "aic_picks_loose": aic_picks_loose,
            "geo_picks_tight": geo_picks_tight,
            "geo_picks_loose": geo_picks_loose,
        },
        "ranking_disagreement_rate": {
            "geo_vs_aic": geo_aic_disagree / max(1, n_valid),
            "geo_vs_bic": geo_bic_disagree / max(1, n_valid),
        },
        "counts": {
            "geo_correct": geo_correct,
            "aic_correct": aic_correct,
            "bic_correct": bic_correct,
            "total": n_valid,
        },
    }


def run_variable_k_benchmark(
    n_trials: int = 500,
    n: int = 200,
) -> dict:
    """Variable-k benchmark: models with different parameter counts AND
    different information profiles.

    Three models:
      M_true  : y ~ x1              (k=2)
      M_fat   : y ~ x1 + z1 + z2   (k=4, two low-info nuisance)
      M_sharp : y ~ x1 + w1         (k=3, one high-info nuisance)

    AIC says M_fat costs 2*4=8, M_sharp costs 2*3=6.
    BIC says M_fat costs 4*log(n), M_sharp costs 3*log(n).
    Geometric score may reveal that M_fat's large parameter count is
    offset by tiny Hessian eigenvalues, while M_sharp's smaller count
    is offset by a huge eigenvalue.
    """
    results = []
    for seed in range(n_trials):
        rng = np.random.RandomState(seed)
        x1 = rng.randn(n)
        z1 = rng.randn(n) * 0.05   # very low info
        z2 = rng.randn(n) * 0.05   # very low info
        w1 = rng.randn(n) * 50.0   # very high info

        y = 2.0 * x1 + rng.randn(n)

        X_true = sm.add_constant(x1)
        X_fat = sm.add_constant(np.column_stack([x1, z1, z2]))
        X_sharp = sm.add_constant(np.column_stack([x1, w1]))

        try:
            r_true = sm.OLS(y, X_true).fit()
            r_fat = sm.OLS(y, X_fat).fit()
            r_sharp = sm.OLS(y, X_sharp).fit()

            comp = nomocomp.GeometricModelComparator()
            result = comp.compare({
                "true": r_true,
                "fat_lowinfo": r_fat,
                "sharp_highinfo": r_sharp,
            })

            geo_winner = result.geometric_ranking[0]
            aic_winner = result.aic_ranking[0]
            bic_winner = result.bic_ranking[0]

            results.append({
                "seed": seed,
                "geo_winner": geo_winner,
                "aic_winner": aic_winner,
                "bic_winner": bic_winner,
                "geo_correct": geo_winner == "true",
                "aic_correct": aic_winner == "true",
                "bic_correct": bic_winner == "true",
                "geo_scores": {s.label: s.geometric_score for s in result.scores},
                "aic_scores": {s.label: s.aic for s in result.scores},
                "log_det_info": {s.label: s.log_det_information for s in result.scores},
            })
        except Exception:
            pass

    n_valid = len(results)
    geo_correct = sum(1 for r in results if r["geo_correct"])
    aic_correct = sum(1 for r in results if r["aic_correct"])
    bic_correct = sum(1 for r in results if r["bic_correct"])

    return {
        "name": "variable_k_benchmark",
        "description": (
            "M_fat (k=4, low-info nuisance) vs M_sharp (k=3, high-info nuisance).  "
            "Tests whether geometric score correctly distinguishes effective "
            "complexity from parameter count."
        ),
        "config": {"n_trials": n_trials, "n_valid": n_valid, "n_obs": n},
        "truth_recovery_rate": {
            "geometric": geo_correct / max(1, n_valid),
            "aic": aic_correct / max(1, n_valid),
            "bic": bic_correct / max(1, n_valid),
        },
        "counts": {
            "geo_correct": geo_correct,
            "aic_correct": aic_correct,
            "bic_correct": bic_correct,
            "total": n_valid,
        },
    }


def run_sample_size_sweep(
    n_values: list[int] | None = None,
    n_trials: int = 200,
    var_tight: float = 100.0,
    var_loose: float = 0.01,
) -> dict:
    """Sweep sample size to show how criteria converge.

    At large n, all criteria should agree (BIC is consistent).
    At small n, the geometric score should separate earlier because
    it uses actual eigenvalues rather than an n-dependent approximation.
    """
    if n_values is None:
        n_values = [30, 50, 100, 200, 500, 1000]

    sweep_results = {}
    for n in n_values:
        results = []
        for seed in range(n_trials):
            try:
                r = run_single_trial(
                    seed, n=n, var_tight=var_tight, var_loose=var_loose
                )
                results.append(r)
            except Exception:
                pass

        n_valid = len(results)
        if n_valid > 0:
            sweep_results[n] = {
                "n_obs": n,
                "n_valid": n_valid,
                "geo_correct_rate": sum(r["geo_correct"] for r in results) / n_valid,
                "aic_correct_rate": sum(r["aic_correct"] for r in results) / n_valid,
                "bic_correct_rate": sum(r["bic_correct"] for r in results) / n_valid,
            }

    return {
        "name": "sample_size_sweep",
        "description": "Truth-recovery rate as a function of sample size.",
        "sweep": sweep_results,
    }


def main():
    print("Truth-Recovery Benchmark: Geometric Score vs AIC/BIC")
    print("=" * 60)
    print()

    # --- Benchmark 1: Equal-k ---
    print("Benchmark 1: Equal parameter count, different information profiles")
    print("-" * 60)
    eq_k = run_equal_k_benchmark(n_trials=500, n=200)
    print(f"  Trials: {eq_k['config']['n_valid']}")
    print(f"  Truth recovery rates:")
    print(f"    Geometric: {eq_k['truth_recovery_rate']['geometric']:.1%}")
    print(f"    AIC:       {eq_k['truth_recovery_rate']['aic']:.1%}")
    print(f"    BIC:       {eq_k['truth_recovery_rate']['bic']:.1%}")
    print(f"  When AIC picks wrong:")
    print(f"    Picks M_tight: {eq_k['wrong_model_picks']['aic_picks_tight']}")
    print(f"    Picks M_loose: {eq_k['wrong_model_picks']['aic_picks_loose']}")
    print(f"  When GEO picks wrong:")
    print(f"    Picks M_tight: {eq_k['wrong_model_picks']['geo_picks_tight']}")
    print(f"    Picks M_loose: {eq_k['wrong_model_picks']['geo_picks_loose']}")
    print(f"  Ranking disagreement rates:")
    print(f"    Geometric vs AIC: {eq_k['ranking_disagreement_rate']['geo_vs_aic']:.1%}")
    print(f"    Geometric vs BIC: {eq_k['ranking_disagreement_rate']['geo_vs_bic']:.1%}")

    # --- Benchmark 2: Variable-k ---
    print(f"\n{'='*60}")
    print("Benchmark 2: Different parameter counts AND information profiles")
    print("-" * 60)
    var_k = run_variable_k_benchmark(n_trials=500, n=200)
    print(f"  Trials: {var_k['config']['n_valid']}")
    print(f"  Truth recovery rates:")
    print(f"    Geometric: {var_k['truth_recovery_rate']['geometric']:.1%}")
    print(f"    AIC:       {var_k['truth_recovery_rate']['aic']:.1%}")
    print(f"    BIC:       {var_k['truth_recovery_rate']['bic']:.1%}")

    # --- Benchmark 3: Sample size sweep ---
    print(f"\n{'='*60}")
    print("Benchmark 3: Sample size sweep")
    print("-" * 60)
    sweep = run_sample_size_sweep(n_trials=200)
    print(f"  {'n':>6s}  {'GEO':>8s}  {'AIC':>8s}  {'BIC':>8s}")
    for n_key, vals in sorted(sweep["sweep"].items()):
        print(
            f"  {vals['n_obs']:6d}  "
            f"{vals['geo_correct_rate']:8.1%}  "
            f"{vals['aic_correct_rate']:8.1%}  "
            f"{vals['bic_correct_rate']:8.1%}"
        )

    # Save results
    save_data = {
        "equal_k": eq_k,
        "variable_k": var_k,
        "sample_size_sweep": {
            k: v for k, v in sweep["sweep"].items()
        },
    }

    outpath = str(_paths.BENCHMARKS_DIR / "truth_recovery_results.json")
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()

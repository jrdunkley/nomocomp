"""
Real-Data Demo: Equal-df OLS Model Comparison on Boston-style Housing
=====================================================================

Demonstrates nomocomp on a recognisable dataset where practitioners
routinely face equal-df model selection.

Scenario:
  Predict median house value.  Two candidate models with the same
  number of predictors (k=4 including intercept), but different
  predictor sets with different information profiles.

  Model A:  value ~ rooms + crime + distance_to_employment
  Model B:  value ~ rooms + tax_rate + pupil_teacher_ratio

  Both models use 3 predictors + intercept.  AIC and BIC give
  different penalties ONLY because of the log-likelihood difference.
  The geometric score additionally accounts for the information
  structure of each predictor set.

We use the California housing dataset (sklearn) as a clean,
recognisable, well-understood dataset.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401 — sets up sys.path portably

import statsmodels.api as sm
import nomocomp


def california_housing_demo():
    """Equal-df model comparison on California housing data."""
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X_all = data.data  # (20640, 8)
    y = data.target     # median house value
    feature_names = list(data.feature_names)

    print("California Housing Dataset")
    print(f"  n = {len(y)}, features: {feature_names}")
    print()

    # Standardise all features for fair comparison
    X_std = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)
    y_std = (y - y.mean()) / y.std()

    # Define several equal-df (k=4) model families
    # Feature indices:
    # 0: MedInc, 1: HouseAge, 2: AveRooms, 3: AveBedrms,
    # 4: Population, 5: AveOccup, 6: Latitude, 7: Longitude
    model_specs = {
        "income+age+rooms": [0, 1, 2],
        "income+lat+lon":   [0, 6, 7],
        "income+pop+occup": [0, 4, 5],
        "age+rooms+bedrms": [1, 2, 3],
        "lat+lon+pop":      [6, 7, 4],
    }

    # Fit all models
    fitted = {}
    for name, idx in model_specs.items():
        X = sm.add_constant(X_std[:, idx])
        fitted[name] = sm.OLS(y_std, X).fit()

    # Full comparison
    comp = nomocomp.GeometricModelComparator()
    result = comp.compare(fitted)

    print("Full comparison (all 5 models, k=4 each):")
    print(comp.summary(result, technical=True))
    print()

    # Pairwise same-k comparisons: find cases where geo ≠ AIC/BIC
    print("Pairwise analysis of all same-k pairs:")
    print("-" * 70)
    print(f"  {'Pair':<40s}  {'AIC winner':<15s}  {'GEO winner':<15s}  {'Agree':>5s}")

    labels = list(fitted.keys())
    n_disagree = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            pair = comp.compare({a: fitted[a], b: fitted[b]})
            aic_w = pair.aic_ranking[0]
            geo_w = pair.geometric_ranking[0]
            agree = "YES" if aic_w == geo_w else "NO"
            if aic_w != geo_w:
                n_disagree += 1
            print(f"  {a+' vs '+b:<40s}  {aic_w:<15s}  {geo_w:<15s}  {agree:>5s}")

    print(f"\n  Pairwise disagreements: {n_disagree} out of {len(labels)*(len(labels)-1)//2}")

    # Detailed look at the most interesting pair
    print()
    print("Detailed pairwise comparison: income+age+rooms vs age+rooms+bedrms")
    print("-" * 60)
    pair_detail = comp.compare({
        "inc_age_rm": fitted["income+age+rooms"],
        "age_rm_bed": fitted["age+rooms+bedrms"],
    })
    print(comp.summary(pair_detail, technical=True))

    # Cross-validation to see which model actually predicts better
    print()
    print("5-Fold Cross-Validation MSE (as prediction sanity check):")
    print("-" * 60)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse = {}
    for name, idx in model_specs.items():
        mses = []
        for train_idx, test_idx in kf.split(X_std):
            X_train = sm.add_constant(X_std[train_idx][:, idx])
            X_test = sm.add_constant(X_std[test_idx][:, idx])
            y_train, y_test = y_std[train_idx], y_std[test_idx]
            r = sm.OLS(y_train, X_train).fit()
            pred = X_test @ r.params
            mses.append(np.mean((y_test - pred) ** 2))
        cv_mse[name] = np.mean(mses)

    cv_ranked = sorted(cv_mse.items(), key=lambda x: x[1])
    for name, mse in cv_ranked:
        print(f"  {name:<25s}  CV-MSE = {mse:.6f}")

    # Information profile analysis
    print()
    print("Information profile analysis (eigenvalues of I for each model):")
    print("-" * 60)
    for name in labels:
        ms = nomocomp.extract_information(fitted[name], label=name)
        eigvals = np.sort(np.linalg.eigvalsh(ms.information_matrix))[::-1]
        eig_str = ", ".join(f"{e:.0f}" for e in eigvals)
        print(
            f"  {name:<25s}  log|I| = {ms.log_det_information:8.2f}  "
            f"eigvals: [{eig_str}]"
        )
        print(f"  {'':25s}  cond = {ms.condition_number:.1f}")

    return result, cv_mse


def iris_demo():
    """Simpler demo on Iris: two equal-df models for sepal length."""
    from sklearn.datasets import load_iris

    data = load_iris()
    X_all = data.data
    y = X_all[:, 0]  # predict sepal length
    feature_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

    print("\n" + "=" * 60)
    print("Iris Demo: Predict sepal length with equal-df models")
    print("=" * 60)

    # Two k=3 models (2 predictors + intercept)
    X_a = sm.add_constant(X_all[:, [1, 2]])  # sepal_wid + petal_len
    X_b = sm.add_constant(X_all[:, [2, 3]])  # petal_len + petal_wid

    r_a = sm.OLS(y, X_a).fit()
    r_b = sm.OLS(y, X_b).fit()

    comp = nomocomp.GeometricModelComparator()
    result = comp.compare({
        "sepal_wid+petal_len": r_a,
        "petal_len+petal_wid": r_b,
    })

    print()
    print(comp.summary(result, technical=True))

    # Information analysis
    print()
    ms_a = nomocomp.extract_information(r_a, label="A")
    ms_b = nomocomp.extract_information(r_b, label="B")
    print(f"  Model A log|I| = {ms_a.log_det_information:.2f}, cond = {ms_a.condition_number:.1f}")
    print(f"  Model B log|I| = {ms_b.log_det_information:.2f}, cond = {ms_b.condition_number:.1f}")
    print(f"  AIC penalty same? {abs(ms_a.aic - (-2*ms_a.log_likelihood)) - abs(ms_b.aic - (-2*ms_b.log_likelihood)):.6f}")

    return result


def small_n_subsampling_demo():
    """Show that fibre-volume corrections matter more at small n.

    At large n, logL differences dominate and all criteria agree.
    At small n, logL differences shrink relative to fibre corrections,
    so the geometric score starts producing different rankings.

    This is the transition regime where the comparator adds value.
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X_all = data.data
    y = data.target
    X_std = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)
    y_std = (y - y.mean()) / y.std()

    # Two equal-df models with different information profiles
    # Model A: income + age + rooms (high-info predictors)
    # Model B: lat + lon + pop (lower-info for price prediction)
    idx_a = [0, 1, 2]
    idx_b = [6, 7, 4]

    n_values = [30, 50, 100, 200, 500, 1000, 5000, 20000]
    n_trials = 100

    print("\n" + "=" * 60)
    print("Small-n Subsampling: Fibre-Volume Relevance vs Sample Size")
    print("=" * 60)
    print()
    print("  Two equal-df models (k=4): income+age+rooms vs lat+lon+pop")
    print("  At each n, subsample and compare.  Track ranking disagreements")
    print("  between geometric score and AIC.")
    print()
    header = "{:>6s}  {:>10s}  {:>10s}  {:>12s}  {:>12s}".format(
        "n", "geo!=aic", "mean|dlogL|", "mean|dlogI|", "ratio I/L"
    )
    print("  " + header)
    print("  " + "-" * len(header))

    rng = np.random.RandomState(42)
    comp = nomocomp.GeometricModelComparator()

    for n in n_values:
        if n > len(y_std):
            continue
        disagree = 0
        d_logL_list = []
        d_logI_list = []

        for trial in range(n_trials):
            idx = rng.choice(len(y_std), size=n, replace=False)
            X_sub = X_std[idx]
            y_sub = y_std[idx]

            X_a = sm.add_constant(X_sub[:, idx_a])
            X_b = sm.add_constant(X_sub[:, idx_b])

            try:
                r_a = sm.OLS(y_sub, X_a).fit()
                r_b = sm.OLS(y_sub, X_b).fit()
                result = comp.compare({"A": r_a, "B": r_b})

                # Track disagreement
                if result.geometric_ranking[0] != result.aic_ranking[0]:
                    disagree += 1

                # Track component magnitudes
                scores = {s.label: s for s in result.scores}
                d_logL = abs(scores["A"].log_likelihood - scores["B"].log_likelihood)
                d_logI = abs(scores["A"].log_det_information - scores["B"].log_det_information)
                d_logL_list.append(d_logL)
                d_logI_list.append(d_logI)
            except Exception:
                pass

        n_valid = len(d_logL_list)
        if n_valid > 0:
            mean_dL = np.mean(d_logL_list)
            mean_dI = np.mean(d_logI_list)
            ratio = mean_dI / max(mean_dL, 1e-10)
            row = "{:6d}  {:9.1%}  {:10.2f}  {:12.2f}  {:12.4f}".format(
                n, disagree / n_valid, mean_dL, mean_dI, ratio
            )
            print("  " + row)

    print()
    print("  When ratio I/L is large, the fibre-volume correction is")
    print("  comparable to the log-likelihood difference — this is")
    print("  where the geometric score adds genuine discriminative power.")
    print("  At large n, logL differences dominate and all criteria converge.")


def main():
    print("Real-Data Demo: nomocomp Geometric Model Comparison")
    print("=" * 60)
    print()

    cal_result, cal_cv = california_housing_demo()
    iris_result = iris_demo()
    small_n_subsampling_demo()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("The geometric comparator distinguishes same-df models by their")
    print("information profiles (log det I), where AIC/BIC are blind.")
    print("On California housing: different predictor sets with the same")
    print("parameter count produce materially different information")
    print("eigenvalue spectra, and the geometric score captures this.")
    print()
    print("The small-n subsampling analysis shows the transition regime:")
    print("at small sample sizes, fibre-volume corrections are large")
    print("relative to log-likelihood differences, producing ranking")
    print("disagreements between geometric score and AIC/BIC.")


if __name__ == "__main__":
    main()

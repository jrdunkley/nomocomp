"""Tests for the geometric model comparator."""
import numpy as np
import pytest
import statsmodels.api as sm
import nomocomp


class TestGeometricModelComparator:
    """Core comparator tests."""

    def test_compare_returns_result(self, ols_three_models):
        comp = nomocomp.GeometricModelComparator()
        result = comp.compare(ols_three_models)
        assert isinstance(result, nomocomp.ComparisonResult)
        assert len(result.scores) == 3
        assert len(result.geometric_ranking) == 3
        assert len(result.aic_ranking) == 3
        assert len(result.bic_ranking) == 3

    def test_rankings_contain_all_labels(self, ols_three_models):
        comp = nomocomp.GeometricModelComparator()
        result = comp.compare(ols_three_models)
        labels = set(ols_three_models.keys())
        assert set(result.geometric_ranking) == labels
        assert set(result.aic_ranking) == labels
        assert set(result.bic_ranking) == labels

    def test_geometric_ranks_tight_worst(self, ols_three_models):
        """High-info nuisance should be penalised most by geometric score."""
        comp = nomocomp.GeometricModelComparator()
        result = comp.compare(ols_three_models)
        # Tight-nuisance should be last in geometric ranking
        assert result.geometric_ranking[-1] == "tight"

    def test_same_k_aic_bic_blind(self, ols_pair_same_k):
        """For same-k models, AIC and BIC rank by logL only."""
        comp = nomocomp.GeometricModelComparator()
        result = comp.compare(ols_pair_same_k)

        s_tight = next(s for s in result.scores if s.label == "tight")
        s_loose = next(s for s in result.scores if s.label == "loose")

        # AIC penalty is the same for both (same k)
        aic_penalty_tight = s_tight.aic - (-2 * s_tight.log_likelihood)
        aic_penalty_loose = s_loose.aic - (-2 * s_loose.log_likelihood)
        assert abs(aic_penalty_tight - aic_penalty_loose) < 1e-6

        # Geometric penalties differ
        geo_penalty_tight = s_tight.geometric_score - (-2 * s_tight.log_likelihood)
        geo_penalty_loose = s_loose.geometric_score - (-2 * s_loose.log_likelihood)
        assert abs(geo_penalty_tight - geo_penalty_loose) > 1.0

    def test_branch_reversal_detection(self, ols_three_models):
        """Branch reversal should be detected when present."""
        comp = nomocomp.GeometricModelComparator()
        result = comp.compare(ols_three_models)
        assert result.branch_reversal is not None
        assert hasattr(result.branch_reversal, "reversal")

    def test_fibre_dominance_diagnostic(self, ols_three_models):
        comp = nomocomp.GeometricModelComparator()
        result = comp.compare(ols_three_models)
        assert result.fibre_dominance is not None

    def test_summary_runs(self, ols_three_models):
        comp = nomocomp.GeometricModelComparator()
        result = comp.compare(ols_three_models)
        s = comp.summary(result)
        assert "Geometric Model Comparison" in s
        s_tech = comp.summary(result, technical=True)
        assert "log|I|" in s_tech

    def test_minimum_two_models(self):
        """Should raise with fewer than 2 models."""
        comp = nomocomp.GeometricModelComparator()
        rng = np.random.RandomState(0)
        X = sm.add_constant(rng.randn(50))
        y = rng.randn(50)
        r = sm.OLS(y, X).fit()
        with pytest.raises(ValueError, match="at least 2"):
            comp.compare({"only_one": r})


class TestSameKExactAgreement:
    """The core mathematical claim: same-k geometric ranking
    matches the exact Bayesian marginal likelihood."""

    def test_geometric_matches_exact_marginal_likelihood(self):
        """For same-k Gaussian OLS, geometric pairwise ranking should
        agree with exact marginal likelihood in nearly all trials."""
        n_agree = 0
        n_total = 100

        for seed in range(n_total):
            rng2 = np.random.RandomState(seed)
            n = 200
            x1 = rng2.randn(n)
            z_tight = rng2.randn(n) * 50.0
            z_loose = rng2.randn(n) * 0.05
            y = 2.0 * x1 + rng2.randn(n)

            X_tight = sm.add_constant(np.column_stack([x1, z_tight]))
            X_loose = sm.add_constant(np.column_stack([x1, z_loose]))

            r_tight = sm.OLS(y, X_tight).fit()
            r_loose = sm.OLS(y, X_loose).fit()

            # Geometric ranking
            comp = nomocomp.GeometricModelComparator()
            result = comp.compare({"tight": r_tight, "loose": r_loose})
            geo_prefers = result.geometric_ranking[0]

            # Exact marginal likelihood (tau2=10, sigma2=1)
            sigma2, tau2 = 1.0, 10.0
            exact_tight = _exact_log_marginal(y, X_tight, sigma2, tau2)
            exact_loose = _exact_log_marginal(y, X_loose, sigma2, tau2)
            exact_prefers = "loose" if exact_loose > exact_tight else "tight"

            if geo_prefers == exact_prefers:
                n_agree += 1

        # Should agree in at least 95% of trials
        assert n_agree / n_total >= 0.95, (
            f"Geometric agreed with exact in only {n_agree}/{n_total} trials"
        )


def _exact_log_marginal(y, X, sigma2, tau2):
    """Exact marginal likelihood for Gaussian OLS with N(0,tau2*I) prior."""
    n, k = X.shape
    M = np.eye(k) + (tau2 / sigma2) * (X.T @ X)
    log_det_Sigma = n * np.log(sigma2) + np.linalg.slogdet(M)[1]
    M_inv = np.linalg.inv(M)
    Sigma_inv_y = y / sigma2 - (tau2 / sigma2**2) * X @ M_inv @ (X.T @ y)
    quad = float(y @ Sigma_inv_y)
    return -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det_Sigma - 0.5 * quad

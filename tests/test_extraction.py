"""Tests for the information matrix extraction layer."""
import numpy as np
import pytest
import statsmodels.api as sm
import nomocomp


class TestExtractInformation:
    """Core extraction tests."""

    def test_returns_model_score(self, ols_three_models):
        ms = nomocomp.extract_information(ols_three_models["true"], label="test")
        assert isinstance(ms, nomocomp.ModelScore)
        assert ms.label == "test"
        assert ms.n_params == 2
        assert ms.nobs == 200

    def test_information_matrix_is_spd(self, ols_three_models):
        for label, result in ols_three_models.items():
            ms = nomocomp.extract_information(result, label=label)
            eigvals = np.linalg.eigvalsh(ms.information_matrix)
            assert eigvals[0] > 0, f"Info matrix not SPD for {label}"

    def test_information_matrix_is_symmetric(self, ols_three_models):
        for label, result in ols_three_models.items():
            ms = nomocomp.extract_information(result, label=label)
            I = ms.information_matrix
            assert np.allclose(I, I.T, atol=1e-10)

    def test_geometric_score_decomposition(self, ols_three_models):
        """geometric_score = -2*logL + log det(I) - k*log(2pi)."""
        import numpy as np
        for label, result in ols_three_models.items():
            ms = nomocomp.extract_information(result, label=label)
            expected = (-2.0 * ms.log_likelihood
                        + ms.log_det_information
                        - ms.n_params * np.log(2 * np.pi))
            assert abs(ms.geometric_score - expected) < 1e-10

    def test_hessian_method(self, ols_three_models):
        """Direct hessian extraction should work for OLS."""
        ms = nomocomp.extract_information(
            ols_three_models["true"], label="h", method="hessian"
        )
        assert ms.extraction_method == "hessian"
        assert ms.information_matrix.shape == (2, 2)

    def test_cov_params_method(self, ols_three_models):
        """cov_params extraction should work as fallback."""
        ms = nomocomp.extract_information(
            ols_three_models["true"], label="c", method="cov_params"
        )
        assert ms.extraction_method == "cov_params"
        assert ms.information_matrix.shape == (2, 2)

    def test_numerical_method(self, ols_three_models):
        """Numerical hessian should approximately match analytic."""
        ms_h = nomocomp.extract_information(
            ols_three_models["true"], label="h", method="hessian"
        )
        ms_n = nomocomp.extract_information(
            ols_three_models["true"], label="n", method="numerical"
        )
        # Numerical should be close to analytic (within ~1%)
        assert np.allclose(
            ms_h.information_matrix, ms_n.information_matrix, rtol=0.02
        )

    def test_information_matches_xtx_over_sigma2(self, ols_three_models):
        """For OLS, I should equal X^T X / sigma2_MLE."""
        result = ols_three_models["true"]
        ms = nomocomp.extract_information(result, label="test", method="hessian")

        X = result.model.exog
        sigma2_mle = result.ssr / result.nobs
        expected_info = X.T @ X / sigma2_mle

        assert np.allclose(ms.information_matrix, expected_info, rtol=1e-8)

    def test_log_det_matches_manual(self, ols_three_models):
        """log det(I) should match manual computation."""
        result = ols_three_models["tight"]
        ms = nomocomp.extract_information(result, label="test")

        manual_log_det = float(np.linalg.slogdet(ms.information_matrix)[1])
        assert abs(ms.log_det_information - manual_log_det) < 1e-10

    def test_different_info_profiles(self, ols_three_models):
        """Tight-nuisance model should have larger log det(I) than loose."""
        ms_tight = nomocomp.extract_information(
            ols_three_models["tight"], label="tight"
        )
        ms_loose = nomocomp.extract_information(
            ols_three_models["loose"], label="loose"
        )
        # Tight has higher-info nuisance -> larger det(I) -> larger log det
        assert ms_tight.log_det_information > ms_loose.log_det_information

    def test_aic_bic_match_statsmodels(self, ols_three_models):
        """AIC and BIC should match statsmodels values."""
        for label, result in ols_three_models.items():
            ms = nomocomp.extract_information(result, label=label)
            assert abs(ms.aic - result.aic) < 1e-6
            assert abs(ms.bic - result.bic) < 1e-6


class TestExtractionEdgeCases:
    """Edge cases and error handling."""

    def test_single_predictor(self):
        """Works with minimal model (intercept only)."""
        rng = np.random.RandomState(0)
        y = rng.randn(50)
        X = sm.add_constant(np.ones(50))  # intercept only won't work
        # Use a real predictor
        X = sm.add_constant(rng.randn(50))
        result = sm.OLS(y, X).fit()
        ms = nomocomp.extract_information(result, label="minimal")
        assert ms.n_params == 2
        assert ms.information_matrix.shape == (2, 2)

    def test_many_predictors(self):
        """Works with many predictors."""
        rng = np.random.RandomState(0)
        n, p = 200, 20
        X = sm.add_constant(rng.randn(n, p))
        y = rng.randn(n)
        result = sm.OLS(y, X).fit()
        ms = nomocomp.extract_information(result, label="many")
        assert ms.n_params == p + 1
        assert ms.information_matrix.shape == (p + 1, p + 1)

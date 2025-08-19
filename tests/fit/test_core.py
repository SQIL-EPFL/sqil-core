import warnings

import numpy as np
import pytest

from sqil_core.fit._core import (
    compute_adjusted_standard_errors,
    compute_aic,
    compute_chi2,
    compute_nrmse,
)


class TestComputeNrmse:
    def test_should_return_zero_for_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        residuals = np.zeros_like(y)
        assert compute_nrmse(residuals, y) == 0.0

    def test_should_return_correct_value_for_real_data(self):
        y = np.array([2.0, 4.0, 6.0, 8.0])
        residuals = np.array([1.0, -1.0, 1.0, -1.0])
        y_span = np.max(y) - np.min(y)
        expected_rmse = np.sqrt(np.mean(residuals**2))
        expected_nrmse = expected_rmse / y_span
        assert np.isclose(compute_nrmse(residuals, y), expected_nrmse)

    def test_should_return_correct_value_for_complex_data(self):
        y = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        residuals = np.array([0.1 + 0.1j, -0.1 - 0.1j, 0.1 + 0.1j])
        n = len(y)
        rmse = np.linalg.norm(residuals) / np.sqrt(n)
        y_abs_span = np.max(np.abs(y)) - np.min(np.abs(y))
        expected_nrmse = rmse / y_abs_span
        assert np.isclose(compute_nrmse(residuals, y), expected_nrmse)

    def test_should_return_nan_for_flat_y_data(self):
        y = np.array([5.0, 5.0, 5.0])
        residuals = np.array([1.0, -1.0, 0.5])
        with pytest.warns(RuntimeWarning):
            result = compute_nrmse(residuals, y)
            assert np.isnan(result)

    def test_should_return_nan_for_zero_residuals_with_flat_y_data(self):
        y = np.array([3.0, 3.0, 3.0])
        residuals = np.array([0.0, 0.0, 0.0])
        with pytest.warns(RuntimeWarning):
            result = compute_nrmse(residuals, y)
            assert np.isnan(result)

    def test_should_warn_on_zero_span(self):
        y = np.array([2.0, 2.0, 2.0])
        residuals = np.array([1.0, -1.0, 0.5])
        with pytest.warns(RuntimeWarning, match="y_data has zero span"):
            compute_nrmse(residuals, y)


class TestComputeAIC:
    def test_should_return_expected_value_for_known_input(self):
        residuals = np.array([1.0, -1.0, 1.0, -1.0])
        n_params = 2
        n = len(residuals)
        rss = np.sum(residuals**2)
        expected_aic = 2 * n_params + n * np.log(rss / n)
        assert np.isclose(compute_aic(residuals, n_params), expected_aic)

    def test_should_return_lower_aic_for_better_fit(self):
        res1 = np.array([1.0, -1.0, 1.0, -1.0])
        res2 = np.array([0.1, -0.1, 0.1, -0.1])
        aic1 = compute_aic(res1, 2)
        aic2 = compute_aic(res2, 2)
        assert aic2 < aic1

    def test_should_penalize_more_parameters(self):
        residuals = np.array([0.5, -0.5, 0.5, -0.5])
        aic_few_params = compute_aic(residuals, 2)
        aic_more_params = compute_aic(residuals, 10)
        assert aic_more_params > aic_few_params

    def test_should_handle_single_residual(self):
        residuals = np.array([0.1])
        n_params = 1
        aic = compute_aic(residuals, n_params)
        assert np.isfinite(aic)

    def test_should_raise_or_warn_on_zero_rss(self):
        residuals = np.zeros(10)
        with pytest.raises(FloatingPointError):
            np.seterr(divide="raise")
            compute_aic(residuals, 1)
        np.seterr(divide="warn")


class TestComputeChi2:
    def test_scalar_sigma_inferred_from_residuals(self):
        residuals = np.array([1.0, -1.0, 1.0])
        chi2 = compute_chi2(residuals, cov_rescaled=False)
        sigma_est = np.std(residuals)
        expected = np.sum((residuals / sigma_est) ** 2)
        assert np.isclose(chi2, expected)

    def test_with_provided_sigma_array(self):
        residuals = np.array([1.0, -2.0, 1.5])
        sigma = np.array([0.5, 0.5, 0.5])
        chi2 = compute_chi2(residuals, sigma=sigma)
        expected = np.sum((residuals / sigma) ** 2)
        assert np.isclose(chi2, expected)

    def test_zero_in_sigma_handled_with_epsilon(self):
        residuals = np.array([1.0, 0.0, -1.0])
        sigma = np.array([1.0, 0.0, 0.5])
        chi2 = compute_chi2(residuals, sigma=sigma)
        assert np.isfinite(chi2)
        assert chi2 > 0

    def test_returns_chi2_and_reduced_chi2(self):
        residuals = np.array([1.0, 2.0, -1.0, 1.5])
        n_params = 2
        chi2, red_chi2 = compute_chi2(residuals, n_params=n_params, cov_rescaled=False)
        dof = len(residuals) - n_params
        assert np.isclose(red_chi2, chi2 / dof)

    def test_nan_reduced_chi2_when_dof_nonpositive(self):
        residuals = np.array([1.0, -1.0])
        n_params = 3  # More params than data points: DOF <= 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chi2, red_chi2 = compute_chi2(residuals, n_params=n_params)
            assert np.isnan(red_chi2)
            assert any("Degrees of freedom" in str(warn.message) for warn in w)

    def test_scalar_sigma_zero_is_safe(self):
        residuals = np.array([1.0, 1.0])
        # cov_rescaled=False forces use of std, which is zero here
        chi2 = compute_chi2(residuals, cov_rescaled=False)
        assert np.isfinite(chi2)

    def test_sigma_scalar_override(self):
        residuals = np.array([1.0, 2.0, 3.0])
        chi2 = compute_chi2(residuals, sigma=2.0)
        expected = np.sum((residuals / 2.0) ** 2)
        assert np.isclose(chi2, expected)

    def test_nan_safe_output_when_all_residuals_zero(self):
        residuals = np.zeros(5)
        chi2 = compute_chi2(residuals, cov_rescaled=False)
        assert np.isfinite(chi2)
        assert chi2 == 0.0


class TestComputeAdjustedStandardErrors:
    def test_basic_usage_with_precomputed_red_chi2(self):
        pcov = np.array([[0.04, 0], [0, 0.09]])
        residuals = np.array([0.1, -0.2, 0.05])
        std_err = compute_adjusted_standard_errors(pcov, residuals, red_chi2=1.0)
        assert np.allclose(std_err, [0.2, 0.3])

    def test_rescales_by_red_chi2(self):
        pcov = np.eye(2)
        residuals = np.array([1.0, -1.0, 0.0])
        std_err = compute_adjusted_standard_errors(pcov, residuals, red_chi2=2.0)
        expected = np.sqrt(np.array([2.0, 2.0]))
        assert np.allclose(std_err, expected)

    def test_computes_red_chi2_if_not_provided(self):
        pcov = np.eye(2)
        residuals = np.array([1.0, 2.0, -1.5, 0.0])
        std_err = compute_adjusted_standard_errors(pcov, residuals, cov_rescaled=False)
        assert std_err.shape == (2,)
        assert np.all(np.isfinite(std_err))

    def test_returns_nan_if_red_chi2_is_nan(self):
        pcov = np.eye(2)
        residuals = np.array([1.0, -1.0])
        red_chi2 = np.nan
        std_err = compute_adjusted_standard_errors(pcov, residuals, red_chi2=red_chi2)
        assert np.all(np.isnan(std_err))

    def test_returns_none_if_pcov_is_none_and_fit_is_not_perfect(self):
        residuals = np.array([1.0, -0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            std_err = compute_adjusted_standard_errors(None, residuals)
            assert std_err is None
            assert any(
                "Covariance matrix could not be estimated" in str(wi.message)
                for wi in w
            )

    def test_warns_on_perfect_fit_with_missing_pcov(self):
        residuals = np.zeros(4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            std_err = compute_adjusted_standard_errors(None, residuals)
            assert std_err is None
            assert any("almost perfect fit" in str(wi.message) for wi in w)

    def test_handles_nan_values_in_rescaled_pcov(self):
        pcov = np.array([[1.0, 0.0], [0.0, 1.0]])
        residuals = np.array([1.0, 1.0])
        std_err = compute_adjusted_standard_errors(pcov, residuals, red_chi2=np.nan)
        assert np.all(np.isnan(std_err))

    def test_cov_rescaled_false_uses_sigma_if_provided(self):
        pcov = np.eye(2)
        residuals = np.array([1.0, -2.0, 0.5])
        sigma = np.array([0.5, 0.5, 0.5])
        std_err = compute_adjusted_standard_errors(
            pcov, residuals, cov_rescaled=False, sigma=sigma
        )
        assert std_err.shape == (2,)
        assert np.all(np.isfinite(std_err))

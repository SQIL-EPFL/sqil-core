from types import SimpleNamespace

import pytest

from sqil_core.fit import (
    FIT_QUALITY_THRESHOLDS,
    FitQuality,
    FitResult,
    evaluate_fit_quality,
    get_best_fit_nrmse_aic,
)


class TestEvaluateFitQuality:

    def test_should_return_correct_quality_for_nrmse(self):
        for threshold, expected_quality in FIT_QUALITY_THRESHOLDS["nrmse"]:
            slightly_below = threshold * 0.99 if threshold != float("inf") else 1e6
            assert evaluate_fit_quality({"nrmse": slightly_below}) == expected_quality

    def test_should_return_correct_quality_for_nmae(self):
        for threshold, expected_quality in FIT_QUALITY_THRESHOLDS["nmae"]:
            slightly_below = threshold * 0.99 if threshold != float("inf") else 1e6
            assert (
                evaluate_fit_quality({"nmae": slightly_below}, recipe="nmae")
                == expected_quality
            )

    def test_should_return_correct_quality_for_red_chi2(self):
        for threshold, expected_quality in FIT_QUALITY_THRESHOLDS["red_chi2"]:
            slightly_below = threshold * 0.99 if threshold != float("inf") else 1e6
            assert (
                evaluate_fit_quality({"red_chi2": slightly_below}, recipe="red_chi2")
                == expected_quality
            )

    def test_should_raise_key_error_if_metric_missing(self):
        with pytest.raises(KeyError):
            evaluate_fit_quality({"nmae": 0.01})  # missing 'nrmse'

    def test_should_raise_if_unknown_recipe(self):
        with pytest.raises(NotImplementedError):
            evaluate_fit_quality({"custom": 0.01}, recipe="custom")


class TestGetBestFitNrmseAic:

    @pytest.fixture
    def quality_equal(self, monkeypatch):
        monkeypatch.setattr("sqil_core.fit._quality.evaluate_fit_quality", lambda m: 1)

    def test_should_return_fit_with_better_nrmse_quality(self, monkeypatch):
        def mock_evaluate_fit_quality(metrics):
            return FitQuality.GOOD if metrics["nrmse"] == 0.5 else FitQuality.ACCEPTABLE

        monkeypatch.setattr(
            "sqil_core.fit._quality.evaluate_fit_quality", mock_evaluate_fit_quality
        )

        fit_a = FitResult(
            params=[1],
            std_err=[0.1],
            fit_output=None,
            metrics={"aic": 100, "nrmse": 0.5},
        )
        fit_b = FitResult(
            params=[1],
            std_err=[0.1],
            fit_output=None,
            metrics={"aic": 90, "nrmse": 7},
        )

        result = get_best_fit_nrmse_aic(fit_a, fit_b)
        assert result == fit_a

    def test_should_raise_if_aic_missing(self, quality_equal):
        fit_a = FitResult(
            params=[1],
            std_err=[0.1],
            fit_output=None,
            metrics={"aic": 100, "nrmse": 0.5},
        )
        fit_b = FitResult(
            params=[1], std_err=[0.1], fit_output=None, metrics={"nrmse": 7}
        )

        with pytest.raises(ValueError):
            get_best_fit_nrmse_aic(fit_a, fit_b)

        with pytest.raises(ValueError):
            get_best_fit_nrmse_aic(fit_b, fit_a)

    def test_should_select_lower_aic_when_equal_nrmse_quality(self, quality_equal):
        fit_a = FitResult(
            params=[1, 2, 3],
            std_err=[0.1, 0.1, 0.1],
            fit_output=None,
            metrics={"aic": 100, "nrmse": 0.5},
        )
        fit_b = FitResult(
            params=[1, 2],
            std_err=[0.1, 0.1],
            fit_output=None,
            metrics={"aic": 95, "nrmse": 0.5},
        )
        result = get_best_fit_nrmse_aic(fit_a, fit_b, aic_rel_tol=0.001)
        assert result == fit_b

    def test_should_prefer_simpler_model_if_aic_within_tol(self, quality_equal):
        fit_a = FitResult(
            params=[1, 2, 3],
            std_err=[0.1] * 3,
            fit_output=None,
            metrics={"aic": 100, "nrmse": 0.5},
        )
        fit_b = FitResult(
            params=[1, 2],
            std_err=[0.1] * 2,
            fit_output=None,
            metrics={"aic": 100.5, "nrmse": 0.5},
        )
        result = get_best_fit_nrmse_aic(fit_a, fit_b, aic_rel_tol=0.01)
        assert result == fit_b

    def test_should_return_first_if_equal_aic_and_complexity(self, quality_equal):
        fit_a = FitResult(
            params=[1, 2],
            std_err=[0.1, 0.1],
            fit_output=None,
            metrics={"aic": 100, "nrmse": 0.5},
        )
        fit_b = FitResult(
            params=[1, 2],
            std_err=[0.1, 0.1],
            fit_output=None,
            metrics={"aic": 100.001, "nrmse": 0.5},
        )
        result = get_best_fit_nrmse_aic(fit_a, fit_b, aic_rel_tol=0.01)
        assert result == fit_a

    def test_should_handle_zero_min_aic_gracefully(self, quality_equal):
        fit_a = FitResult(
            params=[1], std_err=[0.1], fit_output=None, metrics={"aic": 0, "nrmse": 0.5}
        )
        fit_b = FitResult(
            params=[1, 2],
            std_err=[0.1, 0.1],
            fit_output=None,
            metrics={"aic": 0.005, "nrmse": 0.5},
        )
        result = get_best_fit_nrmse_aic(fit_a, fit_b, aic_rel_tol=0.01)
        assert result == fit_a or result == fit_b  # just ensure no crash

import numpy as np
import pytest
from scipy.optimize import curve_fit

from sqil_core.fit import fit_input

TRUE_PARAMS = np.array([1.0, 5.0, 1.0])  # A, x0, fwhm


def lorentzian(x, A, x0, fwhm):
    gamma = fwhm / 2
    return (A * gamma**2) / ((x - x0) ** 2 + gamma**2)


# Create a fixture for generating test data
@pytest.fixture
def fit_data():
    np.random.seed(42)
    x_data = np.linspace(0, 10, 100)
    y_data = lorentzian(x_data, *TRUE_PARAMS) + np.random.normal(0, 0.01, 100)
    return x_data, y_data


@fit_input
def lorentzian_fit(x_data, y_data, guess=None, bounds=None):
    popt, _ = curve_fit(
        lorentzian,
        x_data,
        y_data,
        p0=guess,
        bounds=bounds if bounds else (-np.inf, np.inf),
    )
    return popt  # Return optimized parameters


# Placebo fit function that simply returns guess, bounds, and fixed_params
def fit_function(x_data, y_data, guess=None, bounds=None, fixed_params=None):
    return {"guess": guess, "bounds": bounds, "fixed_params": fixed_params}


# Test class for the decorator
class TestFitInputDecorator:

    @pytest.mark.parametrize(
        "guess, bounds, fixed_params, expected_lower_bounds, expected_upper_bounds",
        [
            (None, None, None, None, None),  # No guess, no bounds
            ([1, 5, 1], None, None, None, None),  # With initial guess, no bounds
            (
                None,
                [(0, 10), (0, 10), (0, 10)],
                None,
                [0, 0, 0],
                [10, 10, 10],
            ),  # With bounds, no guess
            (
                [1, 5, 1],
                [(0, 10), (0, 10), (0, 10)],
                None,
                [0, 0, 0],
                [10, 10, 10],
            ),  # Guess + bounds, no fixed params
            (
                [1, 5, 1],
                [(0, 10), (0, 10), (0, 10)],
                [0],
                [1 - 1e-6, 0, 0],
                [1 + 1e-6, 10, 10],
            ),  # Guess + bounds + fixed params
        ],
    )
    def test_parameter_processing(
        self,
        guess,
        bounds,
        fixed_params,
        expected_lower_bounds,
        expected_upper_bounds,
        fit_data,
    ):
        x_data, y_data = fit_data

        # Applying the decorator to the simple fit function
        @fit_input
        def decorated_fit_function(
            x_data, y_data, guess=None, bounds=None, fixed_params=None
        ):
            return fit_function(x_data, y_data, guess, bounds, fixed_params)

        # Call the decorated function
        result = decorated_fit_function(
            x_data, y_data, guess=guess, bounds=bounds, fixed_params=fixed_params
        )

        # Check bounds processing
        if bounds:
            print(bounds, result["bounds"], expected_lower_bounds)
            assert result["bounds"] is not None
            assert len(result["bounds"]) == 2
            assert np.allclose(result["bounds"][0], expected_lower_bounds)
            assert np.allclose(result["bounds"][1], expected_upper_bounds)
        else:
            assert result["bounds"] is None

    def test_warning_for_unsupported_parameter(self, fit_data):
        x_data, y_data = fit_data

        # Test the decorator with a fit function that doesn't accept guess or bounds
        @fit_input
        def simple_fit(x_data, y_data):
            return {"x_data": x_data, "y_data": y_data}

        with pytest.warns(Warning) as record:
            simple_fit(x_data, y_data, guess=[1, 2], bounds=[(0, 10), (0, 10)])

        assert len(record) == 2
        assert "The fit function doesn't allow any initial guess." in str(
            record[0].message
        )
        assert "The fit function doesn't allow any bounds." in str(record[1].message)

    def test_value_error_for_fixed_params_without_guess(self, fit_data):
        x_data, y_data = fit_data

        # Test the decorator with fixed_params but no guess
        @fit_input
        def simple_fit(x_data, y_data, guess=None, bounds=None, fixed_params=None):
            return {"x_data": x_data, "y_data": y_data}

        with pytest.raises(
            ValueError, match="Using fixed_params requires an initial guess."
        ):
            simple_fit(x_data, y_data, fixed_params=[0])

    @pytest.mark.parametrize(
        "guess, bounds, fixed_params, expected_params",
        [
            (None, None, None, TRUE_PARAMS),  # No guess, no bounds
            ([0, 3, 2], None, None, TRUE_PARAMS),  # With initial guess, no bounds
            (
                None,
                [(-2, 2), (0, 10), (-1, 3)],
                None,
                TRUE_PARAMS,
            ),  # With bounds, no guess
            (
                [0, 3, 2],
                [(-2, 2), (0, 10), (-1, 3)],
                None,
                TRUE_PARAMS,
            ),  # Guess + bounds, no fixed params
        ],
    )
    def test_lorentzian_fit(
        self, guess, bounds, fixed_params, expected_params, fit_data
    ):
        x_data, y_data = fit_data
        fitted_params = lorentzian_fit(
            x_data, y_data, guess=guess, bounds=bounds, fixed_params=fixed_params
        )
        assert np.allclose(fitted_params, expected_params, atol=0.1)

    def test_fit_parameters_remain_fixed(self, fit_data):
        x_data, y_data = fit_data
        guess, bounds, fixed_params = ([0, 3, 2], [(-2, 2), (0, 10), (-1, 3)], [2])
        fitted_params = lorentzian_fit(
            x_data, y_data, guess=guess, bounds=bounds, fixed_params=fixed_params
        )
        assert np.isclose(fitted_params[2], 2, 1e-6)

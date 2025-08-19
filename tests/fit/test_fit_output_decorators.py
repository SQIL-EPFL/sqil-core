import numpy as np
import pytest
import scipy.optimize as spopt
from lmfit import Model, Parameters

from sqil_core.fit import FitResult, fit_output

TRUE_A, TRUE_X0, TRUE_GAMMA = 10, 5, 2

CUSTOM_RESULT = {
    "params": [3, 0, 13, 32],
    "std_err": [0.12, 0.45, 1, 30],
    "metrics": {"red_chi2": 0.65},
    "predict": lambda x: 3 * x + 1,
    "param_names": ["a", "b", "c", "d"],
}


def _lorentzian(x, A, x0, gamma):
    return A * gamma**2 / ((x - x0) ** 2 + gamma**2)


def _check_fit_params(params):
    A, x0, gamma = params
    assert np.isclose(A, TRUE_A, atol=TRUE_A / 10)
    assert np.isclose(x0, TRUE_X0, atol=TRUE_X0 / 10)
    assert np.isclose(gamma, TRUE_GAMMA, atol=TRUE_GAMMA / 10)


def _check_fit_rel_err(errors):
    for err in errors:
        assert err < 0.05


def _check_fit(res):
    assert isinstance(res, FitResult)
    _check_fit_params(res.params)
    _check_fit_rel_err(res.std_err / res.params)


@pytest.fixture
def fit_data():
    # Generate some noisy Lorentzian data
    np.random.seed(22)
    x_data = np.linspace(0, 10, 100)
    y_data = _lorentzian(
        x_data, A=TRUE_A, x0=TRUE_X0, gamma=TRUE_GAMMA
    ) + np.random.normal(0, 0.5, size=x_data.shape)
    return x_data, y_data


class TestFitOutputDecorator:
    def test_should_return_a_FitResult_object(self):
        @fit_output
        def fit():
            return CUSTOM_RESULT

        assert isinstance(fit(), FitResult)

    def test_should_throw_error_for_unknown_fit_output(self):
        for res in [None, 1, "s", [1, 2], ([1, 2], [3, 4])]:

            @fit_output
            def fit(res):
                return res

            with pytest.raises(TypeError):
                assert isinstance(fit(res), FitResult)

    def test_accepts_tuple_with_metadata(self):
        @fit_output
        def fit():
            return CUSTOM_RESULT, {}

        assert isinstance(fit(), FitResult)

    def test_metadata_overrides_result(self):
        metadata = {
            "params": [1, 2, 3],
            "std_err": [1, 2, 3],
            "metrics": {"red_chi2": 1},
            "predict": lambda x: x**2,
            "param_names": ["one", "two", "three"],
        }

        @fit_output
        def fit():
            return CUSTOM_RESULT, metadata

        res = fit()
        assert isinstance(res, FitResult)
        assert res.params == metadata["params"]
        assert res.std_err == metadata["std_err"]
        assert res.metrics == metadata["metrics"]
        assert res.predict == metadata["predict"]
        assert res.param_names == metadata["param_names"]

    def test_metadata_fields_not_present_in_FitResult_go_in_metadata_dict(self):
        metadata = {"other_field": 5}

        @fit_output
        def fit():
            return CUSTOM_RESULT, metadata

        res = fit()
        assert isinstance(res, FitResult)
        assert res.metadata == metadata
        assert res.metadata["other_field"] == 5

    def test_can_pass_functions_in_metadata(self):
        def double_errors(sqil_dict):
            return sqil_dict["std_err"] * 2

        # Define a function in metadata (starts with @)
        # That is evaluated on sqil_dict when all the fit_output
        # information has been computed
        metadata = {"@doubled_errors": double_errors}

        @fit_output
        def fit():
            return CUSTOM_RESULT, metadata

        res = fit()
        assert isinstance(res, FitResult)
        # The filed with @ is removed
        assert res.metadata.get("@doubled_errors", None) is None
        # And is replaced by the computed value
        assert res.metadata["doubled_errors"] == res.std_err * 2

    def test_scipy_curve_fit(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            p0 = [1, 1, 0.1]
            return spopt.curve_fit(_lorentzian, x, y, p0=p0, full_output=True)

        _check_fit(fit(x, y))

    def test_scipy_leastsq(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            def residuals(p, x, y):
                return _lorentzian(x, *p) - y

            p0 = [1, 1, 0.1]
            return spopt.leastsq(residuals, p0, args=(x, y), full_output=True)

        _check_fit(fit(x, y))

    def test_scipy_least_squares(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            def residuals(p):
                return _lorentzian(x, *p) - y

            p0 = [1, 1, 0.1]
            result = spopt.least_squares(residuals, p0)
            return result

        _check_fit(fit(x, y))

    def test_scipy_minimize(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            def objective(p):
                return np.sum((_lorentzian(x, *p) - y) ** 2)

            p0 = [1, 1, 0.1]
            result = spopt.minimize(objective, p0)
            return result

        _check_fit(fit(x, y))

    def test_lmfit(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            model = Model(_lorentzian)
            params = Parameters()
            params.add("A", 1)
            params.add("x0", 1)
            params.add("gamma", 0.1)
            result = model.fit(y, params, x=x)
            return result

        _check_fit(fit(x, y))

    def test_can_pass_param_names_in_metadata(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            p0 = [1, 1, 0.1]
            res = spopt.curve_fit(_lorentzian, x, y, p0=p0, full_output=True)
            return (res, {"param_names": ["a", "z", "k"]})

        res = fit(x, y)
        assert res.param_names == ["a", "z", "k"]

    def test_can_pass_prediction_function_in_metadata_curve_fit(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            p0 = [1, 1, 0.1]
            res = spopt.curve_fit(_lorentzian, x, y, p0=p0, full_output=True)
            return (res, {"predict": _lorentzian})

        res = fit(x, y)
        assert isinstance(res.predict(x), np.ndarray)
        assert len(res.predict(x)) == len(x)

    def test_no_prediction_function(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            p0 = [1, 1, 0.1]
            res = spopt.curve_fit(_lorentzian, x, y, p0=p0, full_output=True)
            return (res, {"predict": _lorentzian})

        res = fit(x, y)
        with pytest.raises(Exception):  # noqa: B017
            assert isinstance(res.predict(x), FitResult)

    def test_predict_from_lmfit(self, fit_data):
        x, y = fit_data

        @fit_output
        def fit(x, y):
            model = Model(_lorentzian)
            params = Parameters()
            params.add("A", 1)
            params.add("x0", 1)
            params.add("gamma", 0.1)
            result = model.fit(y, params, x=x)
            return result

        res = fit(x, y)
        assert isinstance(res.predict(x), np.ndarray)
        assert len(res.predict(x)) == len(x)

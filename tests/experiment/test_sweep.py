import numpy as np
import pytest

from sqil_core.experiment import ExperimentHandler


class MyExperiment(ExperimentHandler):
    """Subclass of ExperimentHandler with override for `qubit_value` for testing."""

    def __init__(self, param_map):
        self.param_map = param_map

    def sequence(self, path, *args, **kwargs):
        pass

    def analyze(self, path, *args, **kwargs):
        pass

    def qubit_value(self, param_id, qu_uid):
        return self.param_map.get((qu_uid, param_id), None)


@pytest.fixture
def handler():
    return MyExperiment({("q0", "freq"): 5.0, ("q1", "bias"): 2.5})


class TestSweepAround:

    def test_should_sweep_linearly_with_n_points(self, handler):
        result = handler.sweep_around(center=5.0, span=1.0, n_points=5)
        expected = np.linspace(4.0, 6.0, 5)
        assert np.allclose(result, expected)

    def test_should_sweep_linearly_with_step(self, handler):
        result = handler.sweep_around(center=5.0, span=1.0, step=0.5)
        expected = np.arange(4.0, 6.0 + 0.25, 0.5)
        assert np.allclose(result, expected)

    def test_should_sweep_asymmetric_span(self, handler):
        result = handler.sweep_around(center=5.0, span=(0.5, 1.5), n_points=5)
        expected = np.linspace(4.5, 6.5, 5)
        assert np.allclose(result, expected)

    def test_should_sweep_logarithmically_with_n_points(self, handler):
        result = handler.sweep_around(center=10.0, span=1.0, n_points=5, scale="log")
        expected = np.logspace(np.log10(9.0), np.log10(11.0), 5)
        assert np.allclose(result, expected)

    def test_should_sweep_logarithmically_with_step(self, handler):
        result = handler.sweep_around(center=10.0, span=1.0, step=0.5, scale="log")
        log_start = np.log10(9.0)
        log_stop = np.log10(11.0)
        num_steps = int(np.floor((log_stop - log_start) / np.log10(1 + 0.5 / 9.0))) + 1
        expected = np.logspace(log_start, log_stop, num=num_steps)
        assert np.allclose(result, expected)

    def test_should_resolve_center_from_qubit_param(self, handler):
        result = handler.sweep_around(center="freq", span=1.0, n_points=3, qu_uid="q0")
        expected = np.linspace(4.0, 6.0, 3)
        assert np.allclose(result, expected)

    def test_should_raise_if_qubit_param_not_found(self, handler):
        with pytest.raises(AttributeError):
            handler.sweep_around(center="missing", span=1.0, n_points=3)

    def test_should_raise_for_invalid_scale(self, handler):
        with pytest.raises(ValueError):
            handler.sweep_around(center=5.0, span=1.0, n_points=3, scale="cubic")

    def test_should_raise_if_both_n_points_and_step_provided(self, handler):
        with pytest.raises(ValueError):
            handler.sweep_around(center=5.0, span=1.0, n_points=10, step=0.1)

    def test_should_raise_if_neither_n_points_nor_step_provided(self, handler):
        with pytest.raises(ValueError):
            handler.sweep_around(center=5.0, span=1.0)

    def test_should_raise_for_non_positive_log_sweep(self, handler):
        with pytest.raises(ValueError):
            handler.sweep_around(center=0.5, span=1.0, n_points=10, scale="log")

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sqil_core.fit import FitResult
from sqil_core.resonator._resonator import (
    S11_reflection,
    S21_hanger,
    S21_transmission,
    quick_fit,
)


class TestQuickFit:
    # Note: the phase_vs_freq fit used by quick_fit works for overly coupled resonators,
    #       which means Re[Q_ext] < Q_int.
    #       We choose Q_int = 2.5e4, |Q_ext| = 0.5e4, <Q_ext = 0.8
    TRUE_PARAMS_DICT = {
        "a": 1.0,
        "alpha": 0.2,
        "tau": 1e-9,
        "Q_tot": 4.2e3,
        "fr": 5e9,
        "Q_ext": 5e3 * np.exp(1j * 0.8),
        "phi_0": 0.8,
    }
    TRUE_PARAMS = list(TRUE_PARAMS_DICT.values())

    # Default values for keyword arguments
    KWARGS = {
        "tau": None,
        "Q_tot": None,
        "fr": None,
        "mag_bg": None,
        "fit_range": None,
        "bias_toward_fr": False,
        "verbose": False,
        "do_plot": False,
    }

    @pytest.fixture
    def mock_data(self):
        freq = np.linspace(4.95e9, 5.05e9, 550)
        np.random.seed(13)
        noise = np.random.normal(0, 3e-3, freq.shape)
        data = {
            "reflection": S11_reflection(freq, *self.TRUE_PARAMS) + noise,
            "hanger": S21_hanger(freq, *self.TRUE_PARAMS) + noise,
            "transmission": S21_transmission(freq, *self.TRUE_PARAMS[:-2]) + noise,
        }
        return freq, data

    @pytest.fixture
    def mock_fit_results(self):
        """Mock return values for the fitting functions used inside quick_fit"""
        (a, alpha, tau, Q_tot, fr, Q_ext, phi0) = self.TRUE_PARAMS

        # Mock for fit_lorentzian
        mock_lorentzian = MagicMock(spec=FitResult)
        mock_lorentzian.params = [1, fr, fr / Q_tot, 0]
        mock_lorentzian.metrics = {"nrmse": 0.35}

        # Mock for fit_skewed_lorentzian
        mock_skewed_lorentzian = MagicMock(spec=FitResult)
        mock_skewed_lorentzian.params = [1.0, 2.0, 3.0, 4.0, fr, Q_tot]

        # Mock for fit_phase_vs_freq
        mock_phase_vs_freq = MagicMock(spec=FitResult)
        mock_phase_vs_freq.params = [0.1, Q_tot, fr]

        # Mock for fit_circle_algebraic
        mock_circle_algebraic = MagicMock(spec=FitResult)
        mock_circle_algebraic.params = [0.2, 0.3, 0.5]

        return {
            "lorentzian": mock_lorentzian,
            "skewed_lorentzian": mock_skewed_lorentzian,
            "phase_vs_freq": mock_phase_vs_freq,
            "circle_algebraic": mock_circle_algebraic,
        }

    def test_output_shape(self, mock_data):
        freq, data = mock_data
        for measurement in ["reflection", "hanger"]:
            result = quick_fit(
                freq, data[measurement], measurement=measurement, **self.KWARGS
            )

            assert len(result) == 7
            a, alpha, tau, Q_tot, fr, Q_ext, phi0 = result

            assert isinstance(a, float)
            assert isinstance(alpha, float)
            assert isinstance(tau, float)
            assert isinstance(Q_tot, float)
            assert isinstance(fr, float)
            assert isinstance(Q_ext, complex)
            assert isinstance(phi0, float)

    def test_can_fit_with_default_parameters(self, mock_data):
        freq, data = mock_data
        for measurement in ["reflection", "hanger"]:
            result = quick_fit(
                freq, data[measurement], measurement=measurement, **self.KWARGS
            )
            assert result == pytest.approx(self.TRUE_PARAMS, rel=0.1)

    def test_passing_tau_should_fix_it(self, mock_data):
        freq, data = mock_data
        # Pass a specifiv value of tau into quick_fit
        custom_tau = 2.1e-9
        kwargs = {**self.KWARGS, "tau": custom_tau}

        for measurement in ["reflection", "hanger"]:
            result = quick_fit(
                freq, data[measurement], measurement=measurement, **kwargs
            )
            a, alpha, tau, *_ = result
            assert tau == custom_tau

    @patch("sqil_core.resonator._resonator.fit_phase_vs_freq")
    def test_passing_Q_tot_or_fr_should_fix_them_until_phase_vs_freq_fit(
        self, mock_fit_phase, mock_data
    ):
        freq, data = mock_data
        true_Q_tot = self.TRUE_PARAMS_DICT["Q_tot"]
        true_fr = self.TRUE_PARAMS_DICT["fr"]

        # Pass specific values for Q_tot and fr into quick_fit
        guess_Q_tot = true_Q_tot * 0.99
        guess_fr = true_fr * 0.999
        kwargs = {**self.KWARGS, "Q_tot": guess_Q_tot, "fr": guess_fr}

        # Make sure freq_vs_phase returns the correct value to avoid errors
        mock_result = MagicMock(spec=FitResult)
        mock_result.params = [0, true_Q_tot, true_fr]
        mock_fit_phase.return_value = mock_result

        for measurement in ["reflection", "hanger"]:
            quick_fit(freq, data[measurement], measurement=measurement, **kwargs)

            # Check if fr and Q_tot were preserved until the fitting function by
            # verifying they were passed correctly to fit_phase_vs_freq
            mock_fit_phase.assert_called()
            call_args, call_kwargs = mock_fit_phase.call_args

            # Check the positional or keyword arguments for fr and Q_tot
            if len(call_args) >= 4:  # If passed as positional args
                assert call_args[3] == guess_fr, (
                    "fr was changed before phase_vs_freq fit"
                )
                assert call_args[2] == guess_Q_tot, (
                    "Q_tot was changed before phase_vs_freq fit"
                )
            else:  # If passed as keyword args
                assert call_kwargs.get("fr") == guess_fr, (
                    "fr was changed before phase_vs_freq fit"
                )
                assert call_kwargs.get("Q_tot") == guess_Q_tot, (
                    "Q_tot was changed before phase_vs_freq fit"
                )

    @patch("sqil_core.resonator._resonator.fit_lorentzian")
    @patch("sqil_core.resonator._resonator.fit_skewed_lorentzian")
    def test_should_estimate_q_tot_and_fr_if_not_provided(
        self, mock_fit_skewed, mock_fit_lorentzian, mock_data, mock_fit_results
    ):
        freq, data = mock_data
        kwargs = {**self.KWARGS, "Q_tot": None, "fr": None}

        mock_fit_skewed.return_value = mock_fit_results["skewed_lorentzian"]
        mock_fit_lorentzian.return_value = mock_fit_results["lorentzian"]

        for measurement in ["reflection", "hanger"]:
            quick_fit(freq, data[measurement], measurement=measurement, **kwargs)
            assert mock_fit_lorentzian.called or mock_fit_skewed.called, (
                f"Neither mock_lorentzian nor mock_fit_skewed was called "
                f"for {measurement}"
            )
            mock_fit_lorentzian.reset_mock()
            mock_fit_skewed.reset_mock()

    @patch("sqil_core.resonator._resonator.estimate_linear_background")
    def test_should_estimate_tau_if_not_provided(self, mock_estimate_bg, mock_data):
        freq, data = mock_data
        kwargs = {**self.KWARGS, "tau": None}

        # Configure the mock to return a specific value
        mock_estimate_bg.return_value = [0, 2 * np.pi * 1e-9]

        for measurement in ["reflection", "hanger"]:
            result = quick_fit(
                freq, data[measurement], measurement=measurement, **kwargs
            )
            # Check if estimate_linear_background was called
            mock_estimate_bg.assert_called_once()
            mock_estimate_bg.reset_mock()
            # Verify tau in the output (should be the estimated value divided by 2π)
            _, _, tau, *_ = result
            assert np.isclose(tau, 1e-9)

    def test_should_thow_error_for_invalid_measurement_type(self, mock_data):
        freq, data = mock_data
        with pytest.raises(Exception) as excinfo:
            quick_fit(freq, data["reflection"], measurement="invalid_type")
        assert "Invalid measurement type" in str(excinfo.value)

# create class TestQuickFit check that the output shape is correct
# try to find a way to check the intermediate value of the parameters fr and Q_tot (maybe hard)
# check for a given precision if the whole function outputs the correct parameters (if they are given, should output same ones except some)
# refer to test_fit_output_decorators.py

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sqil_core.fit import FitResult, fit_circle_algebraic, fit_skewed_lorentzian

# Import the module being tested
from sqil_core.resonator._resonator import S11_reflection, S21_hanger, quick_fit
from sqil_core.utils import estimate_linear_background


class TestQuickFit:

    @pytest.fixture
    def mock_data(self):
        """Generate simple mock data for testing quick_fit"""
        # Create a linear frequency range
        freq = np.linspace(4.9e9, 5.1e9, 500)

        # Create mock complex data simulating a resonator response
        # Center frequency is 5e9 Hz
        fr = 5e9
        Q_tot = 10000
        a = 1.0
        alpha = 0.1
        tau = 1e-9
        phi0 = 0.2
        Q_ext = 15000

        # Create noise-free data for a reflection measurement
        data_reflection = S11_reflection(freq, a, alpha, tau, Q_tot, Q_ext, fr, phi0)

        # Create noise-free data for a hanger measurement
        data_hanger = S21_hanger(freq, a, alpha, tau, Q_tot, Q_ext, fr, phi0)

        # Return both datasets and parameters
        return {
            "freq": freq,
            "data_reflection": data_reflection,
            "data_hanger": data_hanger,
            "params": {
                "a": a,
                "alpha": alpha,
                "tau": tau,
                "Q_tot": Q_tot,
                "Q_ext": Q_ext,
                "fr": fr,
                "phi0": phi0,
            },
        }

    @pytest.fixture
    def mock_fit_results(self):
        """Mock return values for the fitting functions used inside quick_fit"""
        # Mock for fit_skewed_lorentzian
        mock_skewed_lorentzian = MagicMock(spec=FitResult)
        mock_skewed_lorentzian.params = [1.0, 2.0, 3.0, 4.0, 5e9, 10000]

        # Mock for fit_phase_vs_freq
        mock_phase_vs_freq = MagicMock(spec=FitResult)
        mock_phase_vs_freq.params = [0.1, 10000, 5e9]

        # Mock for fit_circle_algebraic
        mock_circle_algebraic = MagicMock(spec=FitResult)
        mock_circle_algebraic.params = [0.2, 0.3, 0.5]

        return {
            "skewed_lorentzian": mock_skewed_lorentzian,
            "phase_vs_freq": mock_phase_vs_freq,
            "circle_algebraic": mock_circle_algebraic,
        }

    def test_quick_fit_return_values(self, mock_data):
        """Test that quick_fit returns the expected number of values"""
        # Test with reflection measurement
        result = quick_fit(
            mock_data["freq"], mock_data["data_reflection"], measurement="reflection"
        )

        # Check if we get 8 return values as expected
        assert len(result) == 8
        a, alpha, tau, Q_tot, Q_ext, fr, phi0, theta0 = result

        # Check types of return values
        assert isinstance(a, float)
        assert isinstance(alpha, float)
        assert isinstance(tau, float)
        assert isinstance(Q_tot, float)
        assert isinstance(Q_ext, complex)  # Q_ext should be complex
        assert isinstance(fr, float)
        assert isinstance(phi0, float)
        assert isinstance(theta0, float)

    def test_quick_fit_with_default_parameters(self, mock_data):
        """Test that quick_fit works with default parameters"""
        # Call with minimal required parameters (letting the function use defaults)
        result = quick_fit(
            mock_data["freq"], mock_data["data_reflection"], measurement="reflection"
        )

        # Check if the function completes and returns proper values
        assert all(
            not np.isnan(val) if isinstance(val, (float, complex)) else True
            for val in result
        )

    def test_quick_fit_with_provided_parameters(self, mock_data):
        """Test that quick_fit respects provided parameters"""
        # Set specific values for optional parameters
        custom_tau = 2e-9
        custom_Q_tot = 12000
        custom_fr = 5.01e9

        # Call with explicit parameters
        result = quick_fit(
            mock_data["freq"],
            mock_data["data_reflection"],
            measurement="reflection",
            tau=custom_tau,
            Q_tot=custom_Q_tot,
            fr=custom_fr,
        )

        # Unpack the results
        a, alpha, tau, Q_tot, Q_ext, fr, phi0, theta0 = result

        # Check if tau maintained the provided value (this parameter should not change)
        assert tau == custom_tau

    @patch("sqil_core.resonator._resonator.fit_phase_vs_freq")
    def test_parameter_preservation_before_fitting(self, mock_fit_phase, mock_data):
        """Test that fr and Q_tot don't change in the first part of the method and only after fitting"""
        # Return a valid FitResult-like object to avoid errors
        mock_result = MagicMock(spec=FitResult)
        mock_result.params = [0, 10000, 5e9]  # theta0, Q_tot, fr
        mock_fit_phase.return_value = mock_result

        # Use existing mock data instead of creating our own
        fr_input = 5e9
        Q_tot_input = 10000

        # Run quick_fit with our input parameters
        quick_fit(
            freq=mock_data["freq"],
            data=mock_data["data_reflection"],
            measurement="reflection",
            fr=fr_input,
            Q_tot=Q_tot_input,
            verbose=False,
            do_plot=False,
        )

        # Check if fr and Q_tot were preserved until the fitting function by verifying
        # they were passed correctly to fit_phase_vs_freq
        mock_fit_phase.assert_called()
        call_args, call_kwargs = mock_fit_phase.call_args

        # Check the positional or keyword arguments for fr and Q_tot
        if len(call_args) >= 4:  # If passed as positional args
            assert (
                call_args[3] == fr_input
            ), "fr was changed before the fitting function"
            assert (
                call_args[2] == Q_tot_input
            ), "Q_tot was changed before the fitting function"
        else:  # If passed as keyword args
            assert (
                call_kwargs.get("fr") == fr_input
            ), "fr was changed before the fitting function"
            assert (
                call_kwargs.get("Q_tot") == Q_tot_input
            ), "Q_tot was changed before the fitting function"

    @patch("sqil_core.resonator._resonator.fit_skewed_lorentzian")
    def test_parameter_estimation_behavior(
        self, mock_fit_skewed, mock_data, mock_fit_results
    ):
        """Test how quick_fit estimates parameters if not provided"""
        # Configure the mock to return our predefined result
        mock_fit_skewed.return_value = mock_fit_results["skewed_lorentzian"]

        # Call without providing Q_tot and fr
        quick_fit(
            mock_data["freq"], mock_data["data_reflection"], measurement="reflection"
        )

        # Verify that fit_skewed_lorentzian was called
        mock_fit_skewed.assert_called_once()

    @patch("sqil_core.resonator._resonator.estimate_linear_background")
    def test_tau_estimation(self, mock_estimate_bg, mock_data):
        """Test that tau is estimated if not provided"""
        # Configure the mock to return a specific value
        mock_estimate_bg.return_value = [0, 2 * np.pi * 1e-9]

        # Call without providing tau
        result = quick_fit(
            mock_data["freq"],
            mock_data["data_reflection"],
            measurement="reflection",
            tau=None,  # explicitly set to None to ensure estimation
        )

        # Check if estimate_linear_background was called
        mock_estimate_bg.assert_called_once()

        # Verify tau in the output (should be the estimated value divided by 2π)
        _, _, tau, *_ = result
        assert np.isclose(tau, 1e-9)

    def test_invalid_measurement_type(self, mock_data):
        """Test that quick_fit raises an exception for invalid measurement types"""
        with pytest.raises(Exception) as excinfo:
            quick_fit(
                mock_data["freq"],
                mock_data["data_reflection"],
                measurement="invalid_type",
            )

        assert "Invalid measurement type" in str(excinfo.value)

    def test_reflection_measurement(self, mock_data):
        """Test closeness of quick_fit with reflection measurement"""
        result = quick_fit(
            mock_data["freq"],
            mock_data["data_reflection"],
            measurement="reflection",
            verbose=False,
            do_plot=False,
        )

        # Basic validation of results
        assert len(result) == 8

        # For a perfect synthetic dataset, the recovered parameters should be close to the input
        a, alpha, tau, Q_tot, Q_ext, fr, phi0, _ = result
        expected_params = mock_data["params"]

        # Check with reasonable tolerances (not exact because of fitting process)
        assert np.isclose(a, expected_params["a"], rtol=0.2)
        assert np.isclose(fr, expected_params["fr"], rtol=0.01)

    def test_hanger_measurement(self, mock_data):
        """Test closeness of quick_fit with hanger measurement"""
        result = quick_fit(
            mock_data["freq"],
            mock_data["data_hanger"],
            measurement="hanger",
            verbose=False,
            do_plot=False,
        )

        # Basic validation of results
        assert len(result) == 8

        # For a perfect synthetic dataset, the recovered parameters should be close to the input
        a, alpha, tau, Q_tot, Q_ext, fr, phi0, _ = result
        expected_params = mock_data["params"]

        # Check with reasonable tolerances (not exact because of fitting process)
        assert np.isclose(a, expected_params["a"], rtol=0.2)
        assert np.isclose(fr, expected_params["fr"], rtol=0.01)

import numpy as np

from sqil_core.fit.fit import *

# class TestComputeStandardErrors:

#     def test_compute_standard_errors_basic(self):
#         x = np.linspace(0, 10, 100)
#         popt = np.array([1.0, 0.5])
#         pcov = np.array([[0.01, 0], [0, 0.02]])
#         residuals = np.random.normal(0, 0.1, size=100)

#         std_errs = compute_standard_errors(x, popt, pcov, residuals)

#         # Check output length
#         assert len(std_errs) == len(popt), "Standard errors length mismatch"

#         # Check non-negative standard errors
#         assert np.all(std_errs >= 0), "Standard errors must be non-negative"

#     def test_compute_standard_errors_zero_residuals(self):
#         x = np.linspace(0, 10, 100)
#         popt = np.array([1.0, 0.5])
#         pcov = np.array([[0.01, 0], [0, 0.02]])
#         residuals = np.zeros(100)

#         std_errs = compute_standard_errors(x, popt, pcov, residuals)

#         assert np.allclose(
#             std_errs, 0
#         ), "Standard errors should be zero for perfect fit"


# class TestFitSkewedLorenzian:

#     def test_fit_skewed_lorentzian_basic(self):
#         f = np.linspace(0.9, 1.1, 500)
#         y_true = 1 / (1 + (2 * 1000 * (f / 1.0 - 1)) ** 2)

#         popt, perc_errs = fit_skewed_lorentzian(f, y_true)

#         # Check parameter count
#         assert len(popt) == 6, "Fitted parameters should have length 6"

#         # Check if fitted resonance frequency is close to true value
#         assert np.isclose(popt[4], 1.0, atol=0.1), "Resonance frequency is not accurate"

#         # Check percentage errors are reasonable
#         assert np.all(perc_errs >= 0), "Percentage errors must be non-negative"

#     def test_fit_skewed_lorentzian_noisy_data(self):
#         np.random.seed(0)
#         f = np.linspace(0.9, 1.1, 500)
#         y_clean = 1 / (1 + (2 * 1000 * (f / 1.0 - 1)) ** 2) + f
#         noise = np.random.normal(0, 0.05, size=f.shape)
#         y_noisy = y_clean + noise

#         popt, _ = fit_skewed_lorentzian(f, y_noisy)

#         assert np.isfinite(
#             popt
#         ).all(), "Fitted parameters should be finite despite noise"

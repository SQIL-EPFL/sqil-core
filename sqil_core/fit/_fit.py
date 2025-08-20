from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy.optimize import curve_fit, fsolve, least_squares, leastsq, minimize

from sqil_core.fit import _models
from sqil_core.utils._utils import fill_gaps, has_at_least_one, make_iterable

from ._core import FitResult, fit_input, fit_output
from ._guess import (
    decaying_exp_guess,
    decaying_oscillations_bounds,
    decaying_oscillations_guess,
    gaussian_bounds,
    gaussian_guess,
    lorentzian_bounds,
    lorentzian_guess,
    many_decaying_oscillations_guess,
    oscillations_bounds,
    oscillations_guess,
)


@fit_input
@fit_output
def fit_lorentzian(
    x_data: np.ndarray,
    y_data: np.ndarray,
    guess: list = None,
    bounds: list[tuple[float]] | tuple = None,
) -> FitResult:
    r"""
    Fits a Lorentzian function to the provided data. The function estimates the
    amplitude (A), center (x0), full width at half maximum (FWHM), and baseline (y0)
    of the Lorentzian function.

    L(x) = A * (|FWHM| / 2) / ((x - x0)^2 + (FWHM^2 / 4)) + y0

    $$L(x) = A \frac{\left| \text{FWHM} \right|}{2} \frac{1}{(x - x_0)^2 + \frac{\text{FWHM}^2}{4}} + y_0$$

    Parameters
    ----------
    x_data : np.ndarray
        The independent variable (e.g., x values of the data).

    y_data : np.ndarray
        The dependent variable (e.g., y values of the data).

    guess : list, optional
        Initial guesses for the fit parameters [A, x0, fwhm, y0]. If not provided,
        defaults are calculated based on the data.

    bounds : list[tuple[float]], optional
        The bounds for the fit parameters in the format [(min, max), ...].
        If not provided, defaults are calculated.

    fixed_params : list[int], optional, default: None
        A list of indices representing parameters in the initial guess that should
        remain unchanged during the fitting process.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`rmse`, root mean squared error).
        - A callable `predict` function for generating fitted responses.
    """  # noqa: E501

    x, y = x_data, y_data

    # Default intial guess if not provided
    if has_at_least_one(guess, None):
        guess = fill_gaps(guess, lorentzian_guess(x_data, y_data))

    # Default bounds if not provided
    if bounds is None:
        bounds = ([None] * len(guess), [None] * len(guess))
    if has_at_least_one(bounds[0], None) or has_at_least_one(bounds[1], None):
        lower, upper = bounds
        lower_guess, upper_guess = lorentzian_bounds(x_data, y_data, guess)
        bounds = (fill_gaps(lower, lower_guess), fill_gaps(upper, upper_guess))

    res = curve_fit(_models.lorentzian, x, y, p0=guess, bounds=bounds, full_output=True)

    return res, {
        "param_names": ["A", "x0", "fwhm", "y0"],
        "predict": _models.lorentzian,
    }


@fit_input
@fit_output
def fit_two_lorentzians_shared_x0(
    x_data_1,
    y_data_1,
    x_data_2,
    y_data_2,
    guess: list = None,
    bounds: list[tuple[float]] | tuple = None,
):
    y_all = np.concatenate([y_data_1, y_data_2])

    if has_at_least_one(guess, None):
        guess_1 = lorentzian_guess(x_data_1, y_data_1)
        guess_2 = lorentzian_guess(x_data_2, y_data_2)
        x01, x02 = guess_1[1], guess_2[1]
        x0 = np.mean([x01, x02])
        guess = fill_gaps(
            guess, np.concatenate([np.delete(guess_1, 1), np.delete(guess_2, 1), [x0]])
        )

    if bounds is None:
        bounds = [[None] * len(guess), [None] * len(guess)]
    if has_at_least_one(bounds[0], None) or has_at_least_one(bounds[1], None):
        lower, upper = bounds
        lower_guess_1, upper_guess_1 = lorentzian_bounds(x_data_1, y_data_1, guess_1)
        lower_guess_2, upper_guess_2 = lorentzian_bounds(x_data_2, y_data_2, guess_2)
        # Combine bounds for 1 and 2
        lower_guess = np.concatenate(
            [
                np.delete(lower_guess_1, 1),
                np.delete(lower_guess_2, 1),
                [np.min([lower_guess_1, lower_guess_2])],
            ]
        )
        upper_guess = np.concatenate(
            [
                np.delete(upper_guess_1, 1),
                np.delete(upper_guess_2, 1),
                [np.max([upper_guess_1, upper_guess_2])],
            ]
        )
        lower = fill_gaps(lower, lower_guess)
        upper = fill_gaps(upper, upper_guess)
        bounds = (lower, upper)

    res = curve_fit(
        lambda _, A1, fwhm1, y01, A2, fwhm2, y02, x0: _models.two_lorentzians_shared_x0(
            x_data_1, x_data_2, A1, fwhm1, y01, A2, fwhm2, y02, x0
        ),
        xdata=np.zeros_like(y_all),  # dummy x, since x1 and x2 are fixed via closure
        ydata=y_all,
        p0=guess,
        # bounds=bounds,
        full_output=True,
    )

    return res, {
        "param_names": ["A1", "fwhm1", "y01", "A2", "fwhm2", "y02", "x0"],
        "predict": _models.two_lorentzians_shared_x0,
        "fit_output_vars": {
            "x_data": np.concatenate([x_data_1, x_data_2]),
            "y_data": y_all,
        },
    }


@fit_input
@fit_output
def fit_gaussian(
    x_data: np.ndarray,
    y_data: np.ndarray,
    guess: list = None,
    bounds: list[tuple[float]] | tuple = None,
) -> FitResult:
    r"""
    Fits a Gaussian function to the provided data. The function estimates the
    amplitude, mean, standard deviation (sigma), and baseline of the Gaussian
    function, and computes the full width at half maximum (FWHM).

    G(x) = A / (|σ| * sqrt(2π)) * exp(- (x - x0)^2 / (2σ^2)) + y0

    $$G(x) = A \frac{1}{\left| \sigma \right| \sqrt{2\pi}} \exp\left( -\frac{(x - x_0)^2}{2\sigma^2} \right) + y_0$$

    Parameters
    ----------
    x_data : np.ndarray
        The independent variable (e.g., x values of the data).

    y_data : np.ndarray
        The dependent variable (e.g., y values of the data).

    guess : list, optional
        Initial guesses for the fit parameters [A, x0, sigma, y0]. If not provided,
        defaults are calculated based on the data.

    bounds : list[tuple[float]], optional
        The bounds for the fit parameters in the format [(min, max), ...].
        If not provided, defaults are calculated.

    fixed_params : list[int], optional, default: None
        A list of indices representing parameters in the initial guess that should
        remain unchanged during the fitting process.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`rmse`, root mean squared error).
        - A callable `predict` function for generating fitted responses.
        - A metadata dictionary containing the FWHM.
    """  # noqa: E501

    x, y = x_data, y_data

    # Default initial guess if not provided
    if has_at_least_one(guess, None):
        guess = fill_gaps(guess, gaussian_guess(x_data, y_data))
    # Default bounds if not provided
    if bounds is None:
        bounds = ([None] * len(guess), [None] * len(guess))
    if has_at_least_one(bounds[0], None) or has_at_least_one(bounds[1], None):
        lower, upper = bounds
        lower_guess, upper_guess = gaussian_bounds(x_data, y_data, guess)
        bounds = (fill_gaps(lower, lower_guess), fill_gaps(upper, upper_guess))

    res = curve_fit(_models.gaussian, x, y, p0=guess, bounds=bounds, full_output=True)

    # Compute FWHM from sigma
    _, _, sigma, _ = res[0]
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    return res, {
        "param_names": ["A", "x0", "sigma", "y0"],
        "predict": _models.gaussian,
        "fwhm": fwhm,
    }


@fit_input
@fit_output
def fit_two_gaussians_shared_x0(
    x_data_1,
    y_data_1,
    x_data_2,
    y_data_2,
    guess: list = None,
    bounds: list[tuple[float]] | tuple = None,
):
    y_all = np.concatenate([y_data_1, y_data_2])

    if has_at_least_one(guess, None):
        guess_1 = gaussian_guess(x_data_1, y_data_1)
        guess_2 = gaussian_guess(x_data_2, y_data_2)
        x01, x02 = guess_1[1], guess_2[1]
        x0 = np.mean([x01, x02])
        guess = fill_gaps(
            guess, np.concatenate([np.delete(guess_1, 1), np.delete(guess_2, 1), [x0]])
        )

    if bounds is None:
        bounds = [[None] * len(guess), [None] * len(guess)]
    if has_at_least_one(bounds[0], None) or has_at_least_one(bounds[1], None):
        lower, upper = bounds
        lower_guess_1, upper_guess_1 = gaussian_bounds(x_data_1, y_data_1, guess_1)
        lower_guess_2, upper_guess_2 = gaussian_bounds(x_data_2, y_data_2, guess_2)
        # Combine bounds for 1 and 2
        lower_guess = np.concatenate(
            [
                np.delete(lower_guess_1, 1),
                np.delete(lower_guess_2, 1),
                [np.min([lower_guess_1, lower_guess_2])],
            ]
        )
        upper_guess = np.concatenate(
            [
                np.delete(upper_guess_1, 1),
                np.delete(upper_guess_2, 1),
                [np.max([upper_guess_1, upper_guess_2])],
            ]
        )
        lower = fill_gaps(lower, lower_guess)
        upper = fill_gaps(upper, upper_guess)
        bounds = (lower, upper)

    res = curve_fit(
        lambda _, A1, fwhm1, y01, A2, fwhm2, y02, x0: _models.two_gaussians_shared_x0(
            x_data_1, x_data_2, A1, fwhm1, y01, A2, fwhm2, y02, x0
        ),
        xdata=np.zeros_like(y_all),  # dummy x, since x1 and x2 are fixed via closure
        ydata=y_all,
        p0=guess,
        # bounds=bounds,
        full_output=True,
    )

    return res, {
        "param_names": ["A1", "fwhm1", "y01", "A2", "fwhm2", "y02", "x0"],
        "predict": _models.two_gaussians_shared_x0,
        "fit_output_vars": {
            "x_data": np.concatenate([x_data_1, x_data_2]),
            "y_data": y_all,
        },
    }


@fit_input
@fit_output
def fit_decaying_exp(
    x_data: np.ndarray,
    y_data: np.ndarray,
    guess: list = None,
    bounds: list[tuple[float]] | tuple = (-np.inf, np.inf),
) -> FitResult:
    r"""
    Fits a decaying exponential function to the provided data. The function estimates
    the amplitude (A), decay time constant (tau), and baseline (y0) of the decaying
    exponential function.

    f(x) = A * exp(-x / τ) + y0

    $$f(x) = A \exp\left( -\frac{x}{\tau} \right) + y_0$$

    Parameters
    ----------
    x_data : np.ndarray
        The independent variable (e.g., x values of the data).

    y_data : np.ndarray
        The dependent variable (e.g., y values of the data).

    guess : list, optional
        Initial guesses for the fit parameters [A, tau, y0]. If not provided,
        defaults are calculated based on the data.

    bounds : list[tuple[float]], optional
        The bounds for the fit parameters in the format [(min, max), ...].
        If not provided, defaults are calculated.

    fixed_params : list[int], optional, default: None
        A list of indices representing parameters in the initial guess that should
        remain unchanged during the fitting process.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`rmse`, root mean squared error).
        - A callable `predict` function for generating fitted responses.
    """
    x, y = x_data, y_data

    # Default intial guess if not provided
    if has_at_least_one(guess, None):
        guess = fill_gaps(guess, decaying_exp_guess(x_data, y_data))

    # Default bounds if not provided
    if bounds is None:
        span_y = np.max(y) - np.min(y)
        c0_min = np.min(y) - 100.0 * span_y
        c0_max = np.max(y) + 100.0 * span_y
        bounds = (
            [-100.0 * span_y, 0.0, c0_min],
            [100.0 * span_y, 100.0 * (np.max(x) - np.min(x)), c0_max],
        )

    res = curve_fit(
        _models.decaying_exp, x, y, p0=guess, bounds=bounds, full_output=True
    )

    return res, {"param_names": ["A", "tau", "y0"], "predict": _models.decaying_exp}


@fit_input
@fit_output
def fit_qubit_relaxation_qp(
    x_data: np.ndarray,
    y_data: np.ndarray,
    guess: list[float] | None = None,
    bounds: list[tuple[float]] | tuple = (-np.inf, np.inf),
    maxfev: int = 10000,
    ftol: float = 1e-11,
) -> FitResult:
    r"""
    Fits a qubit relaxation model with quasiparticle (QP) effects using a
    biexponential decay function. The fitting procedure starts with an initial
    guess derived from a single exponential fit.

    f(x) = A * exp(|nQP| * (exp(-x / T1QP) - 1)) * exp(-x / T1R) + y0

    $$f(x) = A \exp\left( |\text{n}_{\text{QP}}| \left( \exp\left(-\frac{x}{T_{1QP}}\right)
    - 1 \right) \right) \exp\left(-\frac{x}{T_{1R}}\right) + y_0$$

    Parameters
    ----------
    x_data : np.ndarray
        Time data points for the relaxation curve.

    y_data : np.ndarray
        Measured relaxation data.

    guess : list[float], optional
        Initial parameter guesses. If None, a default guess is computed
        using a single exponential fit.

    bounds : tuple[list[float], list[float]], optional
        The bounds for the fit parameters in the format [(min, max), ...].
        If None, reasonable bounds based on the initial guess are applied.

    maxfev : int, optional, default=10000
        Maximum number of function evaluations allowed for the curve fitting.

    ftol : float, optional, default=1e-11
        Relative tolerance for convergence in the least-squares optimization.

    fixed_params : list[int], optional, default: None
        A list of indices representing parameters in the initial guess that should
        remain unchanged during the fitting process.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`rmse`, root mean squared error).
        - A callable `predict` function for generating fitted responses.
    """  # noqa: E501

    # Use a single exponential fit for initial parameter guesses
    from scipy.optimize import curve_fit

    def single_exp(x, a, tau, c):
        return a * np.exp(-x / tau) + c

    single_guess = [y_data[0] - y_data[-1], np.mean(x_data), y_data[-1]]
    single_popt, _ = curve_fit(single_exp, x_data, y_data, p0=single_guess)

    a_guess, T1R_guess, c_guess = single_popt
    T1QP_guess = 0.1 * T1R_guess
    nQP_guess = 1.0

    # Default initial guess
    if guess is None:
        guess = [a_guess * np.exp(1.0), 2.0 * T1R_guess, c_guess, T1QP_guess, nQP_guess]

    # Default parameter bounds
    if bounds is None:
        bounds = (
            [
                -20.0 * np.abs(a_guess),
                1.0e-1 * T1R_guess,
                -10.0 * np.abs(c_guess),
                1.0e-4 * T1R_guess,
                0.0,
            ],
            [
                20.0 * np.abs(a_guess),
                1.0e3 * T1R_guess,
                10.0 * np.abs(c_guess),
                10.0 * T1R_guess,
                1.0e3,
            ],
        )

    res = curve_fit(
        _models.qubit_relaxation_qp,
        x_data,
        y_data,
        p0=guess,
        bounds=bounds,
        maxfev=maxfev,
        ftol=ftol,
        full_output=True,
    )

    return res, {
        "param_names": ["A", "T1R", "y0", "T1QP", "nQP"],
        "predict": _models.qubit_relaxation_qp,
    }


@fit_input
@fit_output
def fit_decaying_oscillations(
    x_data: np.ndarray,
    y_data: np.ndarray,
    guess: list[float] | None = None,
    bounds: list[tuple[float]] | tuple = None,
    num_init: int = 10,
) -> FitResult:
    r"""
    Fits a decaying oscillation model to data. The function estimates key features
    like the oscillation period and phase, and tries multiple initial guesses for
    the optimization process.

    f(x) = A * exp(-x / τ) * cos(2π * (x - φ) / T) + y0

    $$f(x) = A \exp\left( -\frac{x}{\tau} \right) \cos\left( 2\pi \frac{x - \phi}{T} \right) + y_0$$

    Parameters
    ----------
    x_data : np.ndarray
        Independent variable array (e.g., time or frequency).
    y_data : np.ndarray
        Dependent variable array representing the measured signal.
    guess : list[float] or None, optional
        Initial parameter estimates [A, tau, y0, phi, T]. Missing values are
        automatically filled.
    bounds : list[tuple[float]] or tuple, optional
        Lower and upper bounds for parameters during fitting, by default no bounds.
    num_init : int, optional
        Number of phase values to try when guessing, by default 10.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`rmse`, root mean squared error).
        - A callable `predict` function for generating fitted responses.
        - A metadata dictionary containing the pi_time and its standard error.
    """  # noqa: E501
    # Default intial guess if not provided
    if has_at_least_one(guess, None):
        guess = fill_gaps(guess, decaying_oscillations_guess(x_data, y_data, num_init))

    # Default bounds if not provided
    if bounds is None:
        bounds = ([None] * len(guess), [None] * len(guess))
    if has_at_least_one(bounds[0], None) or has_at_least_one(bounds[1], None):
        lower, upper = bounds
        lower_guess, upper_guess = decaying_oscillations_bounds(x_data, y_data, guess)
        bounds = (fill_gaps(lower, lower_guess), fill_gaps(upper, upper_guess))

    A, tau, y0, phi, T = guess
    phi = make_iterable(phi)
    y0 = make_iterable(y0)

    best_fit = None
    best_popt = None
    best_nrmse = np.inf

    @fit_output
    def _curve_fit_osc(x_data, y_data, p0, bounds):
        return curve_fit(
            _models.decaying_oscillations,
            x_data,
            y_data,
            p0,
            bounds=bounds,
            full_output=True,
        )

    # Try multiple initializations
    for phi_guess in phi:
        for offset in y0:
            p0 = [A, tau, offset, phi_guess, T]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_res = _curve_fit_osc(x_data, y_data, p0, bounds)
                if fit_res.metrics["nrmse"] < best_nrmse:
                    best_fit, best_popt = fit_res.output, fit_res.params
                    best_nrmse = fit_res.metrics["nrmse"]
            except Exception:
                if best_fit is None:

                    def _decaying_osc_res(p, x, y):
                        return _models.decaying_oscillations(x, *p) - y

                    result = least_squares(
                        _decaying_osc_res,
                        p0,
                        loss="soft_l1",
                        f_scale=0.1,
                        bounds=bounds,
                        args=(x_data, y_data),
                    )
                    best_fit, best_popt = result, result.x

    if best_fit is None:
        return None

    # Compute pi-time (half-period + phase offset)
    pi_time_raw = 0.5 * best_popt[4] + best_popt[3]
    while pi_time_raw > 0.75 * np.abs(best_popt[4]):
        pi_time_raw -= 0.5 * np.abs(best_popt[4])
    while pi_time_raw < 0.25 * np.abs(best_popt[4]):
        pi_time_raw += 0.5 * np.abs(best_popt[4])

    def _get_pi_time_std_err(sqil_dict):
        if sqil_dict["std_err"] is not None:
            phi_err = sqil_dict["std_err"][3]
            T_err = sqil_dict["std_err"][4]
            if np.isfinite(T_err) and np.isfinite(phi_err):
                return np.sqrt((T_err / 2) ** 2 + phi_err**2)
        return np.nan

    # Metadata dictionary
    metadata = {
        "param_names": ["A", "tau", "y0", "phi", "T"],
        "predict": _models.decaying_oscillations,
        "pi_time": pi_time_raw,
        "@pi_time_std_err": _get_pi_time_std_err,
    }

    return best_fit, metadata


@fit_output
def fit_many_decaying_oscillations(
    x_data: np.ndarray, y_data: np.ndarray, n: int, guess=None
):
    """
    Fits a sum of `n` exponentially decaying oscillations to the given data.

    Each component of the model is of the form:
        A_i * exp(-x / tau_i) * cos(2π * T_i * x + phi_i)

    Parameters
    ----------
    x_data : np.ndarray
        1D array of x-values (e.g., time).
    y_data : np.ndarray
        1D array of y-values (e.g., signal amplitude).
    n : int
        Number of decaying oscillation components to fit.
    guess : list or None, optional
        Optional initial parameter guess. If None, a guess is automatically generated
        using `many_decaying_oscillations_guess`.

    Returns
    -------
    FitResult
    """

    if has_at_least_one(guess, None):
        guess = fill_gaps(guess, many_decaying_oscillations_guess(x_data, y_data, n))

    res = curve_fit(
        _models.many_decaying_oscillations,
        x_data,
        y_data,
        p0=guess,
        # maxfev=10000,
        full_output=True,
    )

    metadata = {
        "param_names": [f"{p}{i}" for i in range(n) for p in ("A", "tau", "phi", "T")]
        + ["y0"],
        "predict": lambda x: _models.many_decaying_oscillations(x, *res[0]),
        "model_name": f"many_decaying_oscillations({n})",
    }

    return res, metadata


@fit_input
@fit_output
def fit_oscillations(
    x_data: np.ndarray,
    y_data: np.ndarray,
    guess: list[float] | None = None,
    bounds: list[tuple[float]] | tuple = None,
    num_init: int = 10,
) -> FitResult:
    r"""
    Fits an oscillation model to data. The function estimates key features
    like the oscillation period and phase, and tries multiple initial guesses for
    the optimization process.

    f(x) = A * cos(2π * (x - φ) / T) + y0

    $$f(x) = A \cos\left( 2\pi \frac{x - \phi}{T} \right) + y_0$$

    Parameters
    ----------
    x_data : np.ndarray
        Independent variable array (e.g., time or frequency).
    y_data : np.ndarray
        Dependent variable array representing the measured signal.
    guess : list[float] or None, optional
        Initial parameter estimates [A, y0, phi, T]. Missing values are automatically
        filled.
    bounds : list[tuple[float]] or tuple, optional
        Lower and upper bounds for parameters during fitting, by default no bounds.
    num_init : int, optional
        Number of phase values to try when guessing, by default 10.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`rmse`, root mean squared error).
        - A callable `predict` function for generating fitted responses.
        - A metadata dictionary containing the pi_time and its standard error.
    """
    # Default intial guess if not provided
    if has_at_least_one(guess, None):
        guess = fill_gaps(guess, oscillations_guess(x_data, y_data, num_init))

    # Default bounds if not provided
    if bounds is None:
        bounds = ([None] * len(guess), [None] * len(guess))
    if has_at_least_one(bounds[0], None) or has_at_least_one(bounds[1], None):
        lower, upper = bounds
        lower_guess, upper_guess = oscillations_bounds(x_data, y_data, guess)
        bounds = (fill_gaps(lower, lower_guess), fill_gaps(upper, upper_guess))

    A, y0, phi, T = guess
    phi = make_iterable(phi)
    y0 = make_iterable(y0)

    best_fit = None
    best_popt = None
    best_nrmse = np.inf

    @fit_output
    def _curve_fit_osc(x_data, y_data, p0, bounds):
        return curve_fit(
            _models.oscillations, x_data, y_data, p0, bounds=bounds, full_output=True
        )

    # Try multiple initializations
    for phi_guess in phi:
        for offset in y0:
            p0 = [A, offset, phi_guess, T]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_res = _curve_fit_osc(x_data, y_data, p0, bounds)
                if fit_res.metrics["nrmse"] < best_nrmse:
                    best_fit, best_popt = fit_res.output, fit_res.params
                    best_nrmse = fit_res.metrics["nrmse"]
            except Exception:
                if best_fit is None:

                    def _oscillations_res(p, x, y):
                        return _models.oscillations(x, *p) - y

                    result = least_squares(
                        _oscillations_res,
                        p0,
                        loss="soft_l1",
                        f_scale=0.1,
                        bounds=bounds,
                        args=(x_data, y_data),
                    )
                    best_fit, best_popt = result, result.x

    if best_fit is None:
        return None

    # Compute pi-time (half-period + phase offset)
    pi_time_raw = 0.5 * best_popt[3] + best_popt[2]
    while pi_time_raw > 0.75 * np.abs(best_popt[3]):
        pi_time_raw -= 0.5 * np.abs(best_popt[3])
    while pi_time_raw < 0.25 * np.abs(best_popt[3]):
        pi_time_raw += 0.5 * np.abs(best_popt[3])

    def _get_pi_time_std_err(sqil_dict):
        if sqil_dict["std_err"] is not None:
            phi_err = sqil_dict["std_err"][2]
            T_err = sqil_dict["std_err"][3]
            if np.isfinite(T_err) and np.isfinite(phi_err):
                return np.sqrt((T_err / 2) ** 2 + phi_err**2)
        return np.nan

    # Metadata dictionary
    metadata = {
        "param_names": ["A", "y0", "phi", "T"],
        "predict": _models.oscillations,
        "pi_time": pi_time_raw,
        "@pi_time_std_err": _get_pi_time_std_err,
    }

    return best_fit, metadata


@fit_output
def fit_circle_algebraic(x_data: np.ndarray, y_data: np.ndarray) -> FitResult:
    """Fits a circle in the xy plane and returns the radius and the position of the
    center.

    Reference: https://arxiv.org/abs/1410.3365
    This function uses an algebraic method to fit a circle to the provided data points.
    The algebraic approach is generally faster and more precise than iterative methods,
    but it can be more sensitive to noise in the data.

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates of the data points.
    y : np.ndarray
        Array of y-coordinates of the data points.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`rmse`, root mean squared error).
        - A callable `predict` function for generating fitted responses.

    Examples
    --------
    >>> fit_result = fit_circle_algebraic(x_data, y_data)
    >>> fit_result.summary()
    """
    z_data = x_data + 1j * y_data

    def calc_moments(z_data):
        xi = z_data.real
        xi_sqr = xi * xi
        yi = z_data.imag
        yi_sqr = yi * yi
        zi = xi_sqr + yi_sqr
        Nd = float(len(xi))
        xi_sum = xi.sum()
        yi_sum = yi.sum()
        zi_sum = zi.sum()
        xiyi_sum = (xi * yi).sum()
        xizi_sum = (xi * zi).sum()
        yizi_sum = (yi * zi).sum()
        return np.array(
            [
                [(zi * zi).sum(), xizi_sum, yizi_sum, zi_sum],
                [xizi_sum, xi_sqr.sum(), xiyi_sum, xi_sum],
                [yizi_sum, xiyi_sum, yi_sqr.sum(), yi_sum],
                [zi_sum, xi_sum, yi_sum, Nd],
            ]
        )

    M = calc_moments(z_data)

    a0 = (
        (
            (M[2][0] * M[3][2] - M[2][2] * M[3][0]) * M[1][1]
            - M[1][2] * M[2][0] * M[3][1]
            - M[1][0] * M[2][1] * M[3][2]
            + M[1][0] * M[2][2] * M[3][1]
            + M[1][2] * M[2][1] * M[3][0]
        )
        * M[0][3]
        + (
            M[0][2] * M[2][3] * M[3][0]
            - M[0][2] * M[2][0] * M[3][3]
            + M[0][0] * M[2][2] * M[3][3]
            - M[0][0] * M[2][3] * M[3][2]
        )
        * M[1][1]
        + (
            M[0][1] * M[1][3] * M[3][0]
            - M[0][1] * M[1][0] * M[3][3]
            - M[0][0] * M[1][3] * M[3][1]
        )
        * M[2][2]
        + (-M[0][1] * M[1][2] * M[2][3] - M[0][2] * M[1][3] * M[2][1]) * M[3][0]
        + (
            (M[2][3] * M[3][1] - M[2][1] * M[3][3]) * M[1][2]
            + M[2][1] * M[3][2] * M[1][3]
        )
        * M[0][0]
        + (
            M[1][0] * M[2][3] * M[3][2]
            + M[2][0] * (M[1][2] * M[3][3] - M[1][3] * M[3][2])
        )
        * M[0][1]
        + (
            (M[2][1] * M[3][3] - M[2][3] * M[3][1]) * M[1][0]
            + M[1][3] * M[2][0] * M[3][1]
        )
        * M[0][2]
    )
    a1 = (
        (
            (M[3][0] - 2.0 * M[2][2]) * M[1][1]
            - M[1][0] * M[3][1]
            + M[2][2] * M[3][0]
            + 2.0 * M[1][2] * M[2][1]
            - M[2][0] * M[3][2]
        )
        * M[0][3]
        + (
            2.0 * M[2][0] * M[3][2]
            - M[0][0] * M[3][3]
            - 2.0 * M[2][2] * M[3][0]
            + 2.0 * M[0][2] * M[2][3]
        )
        * M[1][1]
        + (-M[0][0] * M[3][3] + 2.0 * M[0][1] * M[1][3] + 2.0 * M[1][0] * M[3][1])
        * M[2][2]
        + (-M[0][1] * M[1][3] + 2.0 * M[1][2] * M[2][1] - M[0][2] * M[2][3]) * M[3][0]
        + (M[1][3] * M[3][1] + M[2][3] * M[3][2]) * M[0][0]
        + (M[1][0] * M[3][3] - 2.0 * M[1][2] * M[2][3]) * M[0][1]
        + (M[2][0] * M[3][3] - 2.0 * M[1][3] * M[2][1]) * M[0][2]
        - 2.0 * M[1][2] * M[2][0] * M[3][1]
        - 2.0 * M[1][0] * M[2][1] * M[3][2]
    )
    a2 = (
        (2.0 * M[1][1] - M[3][0] + 2.0 * M[2][2]) * M[0][3]
        + (2.0 * M[3][0] - 4.0 * M[2][2]) * M[1][1]
        - 2.0 * M[2][0] * M[3][2]
        + 2.0 * M[2][2] * M[3][0]
        + M[0][0] * M[3][3]
        + 4.0 * M[1][2] * M[2][1]
        - 2.0 * M[0][1] * M[1][3]
        - 2.0 * M[1][0] * M[3][1]
        - 2.0 * M[0][2] * M[2][3]
    )
    a3 = -2.0 * M[3][0] + 4.0 * M[1][1] + 4.0 * M[2][2] - 2.0 * M[0][3]
    a4 = -4.0

    def func(x):
        return a0 + a1 * x + a2 * x * x + a3 * x * x * x + a4 * x * x * x * x

    def d_func(x):
        return a1 + 2 * a2 * x + 3 * a3 * x * x + 4 * a4 * x * x * x

    x0 = fsolve(func, 0.0, fprime=d_func)

    def solve_eq_sys(val, M):
        # prepare
        M[3][0] = M[3][0] + 2 * val
        M[0][3] = M[0][3] + 2 * val
        M[1][1] = M[1][1] - val
        M[2][2] = M[2][2] - val
        return np.linalg.svd(M)

    U, s, Vt = solve_eq_sys(x0[0], M)

    A_vec = Vt[np.argmin(s), :]

    xc = -A_vec[1] / (2.0 * A_vec[0])
    yc = -A_vec[2] / (2.0 * A_vec[0])
    # the term *sqrt term corrects for the constraint, because it may be altered due to
    # numerical inaccuracies during calculation
    r0 = (
        1.0
        / (2.0 * np.absolute(A_vec[0]))
        * np.sqrt(A_vec[1] * A_vec[1] + A_vec[2] * A_vec[2] - 4.0 * A_vec[0] * A_vec[3])
    )

    std_err = _compute_circle_fit_errors(x_data, y_data, xc, yc, r0)
    return {
        "params": [xc, yc, r0],
        "std_err": std_err,
        "metrics": _compute_circle_fit_metrics(x_data, y_data, xc, yc, r0),
        "predict": lambda theta: (xc + r0 * np.cos(theta), yc + r0 * np.sin(theta)),
        "output": {},
        "param_names": ["xc", "yc", "r0"],
    }


def _compute_circle_fit_errors(x, y, xc, yc, r0):
    """Compute the standard errors for the algebraic circle fit"""
    # Residuals: distance from each point to the fitted circle
    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    residuals = distances - r0

    # Estimate variance of the residuals
    dof = len(x) - 3  # Degrees of freedom: N - number of parameters
    variance = np.sum(residuals**2) / dof

    # Jacobian matrix of residuals with respect to (xc, yc, r0)
    J = np.zeros((len(x), 3))
    J[:, 0] = (xc - x) / distances  # ∂residual/∂xc
    J[:, 1] = (yc - y) / distances  # ∂residual/∂yc
    J[:, 2] = -1  # ∂residual/∂r0

    # Covariance matrix approximation: variance * (JᵗJ)⁻¹
    JTJ_inv = np.linalg.inv(J.T @ J)
    pcov = variance * JTJ_inv

    # Standard errors are the square roots of the diagonal of the covariance matrix
    standard_errors = np.sqrt(np.diag(pcov))

    return standard_errors


def _compute_circle_fit_metrics(x_data, y_data, xc, yc, r0):
    """Computed metrics for the algebraic circle fit"""
    # Compute the distance of each data point to the fitted circle center
    r_data = np.sqrt((x_data - xc) ** 2 + (y_data - yc) ** 2)

    # Compute residuals
    residuals = r_data - r0

    # Calculate R-squared (R²)
    # ssr = np.sum(residuals**2)
    # sst = np.sum((r_data - np.mean(r_data)) ** 2)
    # r2 = 1 - (ssr / sst) if sst > 0 else 0

    # Compute RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    # Return results
    return {"rmse": rmse}


@fit_output
def fit_skewed_lorentzian(x_data: np.ndarray, y_data: np.ndarray):
    r"""
    Fits a skewed Lorentzian model to the given data using least squares optimization.

    This function performs a two-step fitting process to find the best-fitting parameters for a skewed Lorentzian model.
    The first fitting step provides initial estimates for the parameters, and the second step refines those estimates
    using a full model fit.

    L(f) = A1 + A2 * (f - fr) + (A3 + A4 * (f - fr)) / [1 + (2 * Q_tot * ((f / fr) - 1))²]

    $$L(f) = A_1 + A_2 \cdot (f - f_r)+ \frac{A_3 + A_4 \cdot (f - f_r)}{1
    + 4 Q_{\text{tot}}^2 \left( \frac{f - f_r}{f_r} \right)^2}$$

    Parameters
    ----------
    x_data : np.ndarray
        A 1D numpy array containing the x data points for the fit.

    y_data : np.ndarray
        A 1D numpy array containing the y data points for the fit.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`red_chi2`).
        - A callable `predict` function for generating fitted responses.

    Examples
    --------
    >>> fit_result = fit_skewed_lorentzian(x_data, y_data)
    >>> fit_result.summary()
    """  # noqa: E501
    A1a = np.minimum(y_data[0], y_data[-1])
    A3a = -np.max(y_data)
    fra = x_data[np.argmin(y_data)]

    # First fit to get initial estimates for the more complex fit
    def residuals(p, x, y):
        A2, A4, Q_tot = p
        err = y - (
            A1a
            + A2 * (x - fra)
            + (A3a + A4 * (x - fra)) / (1.0 + 4.0 * Q_tot**2 * ((x - fra) / fra) ** 2)
        )
        return err

    p0 = [0.0, 0.0, 1e3]
    p_final, _ = leastsq(residuals, p0, args=(np.array(x_data), np.array(y_data)))
    A2a, A4a, Q_tota = p_final

    # Full parameter fit
    def residuals2(p, x, y):
        A1, A2, A3, A4, fr, Q_tot = p
        err = y - (
            A1
            + A2 * (x - fr)
            + (A3 + A4 * (x - fr)) / (1.0 + 4.0 * Q_tot**2 * ((x - fr) / fr) ** 2)
        )
        return err

    p0 = [A1a, A2a, A3a, A4a, fra, Q_tota]
    popt, pcov, infodict, errmsg, ier = leastsq(
        residuals2, p0, args=(np.array(x_data), np.array(y_data)), full_output=True
    )
    # Since Q_tot is always present as a square it may turn out negative
    popt[-1] = np.abs(popt[-1])

    return (
        (popt, pcov, infodict, errmsg, ier),
        {
            "predict": lambda x: _models.skewed_lorentzian(x, *popt),
            "param_names": ["A1", "A2", "A3", "A4", "fr", "Q_tot"],
        },
    )


def transform_data(
    data: np.ndarray,
    transform_type: str = "optm",
    params: list = None,
    deg: bool = True,
    inv_transform: bool = False,
    full_output: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, Callable]
    | tuple[np.ndarray, Callable, list, np.ndarray]
):
    """
    Transforms complex-valued data using various transformation methods, including
    optimization-based alignment, real/imaginary extraction, amplitude, and phase.
    Works with both 1D and 2D data.

    Parameters
    ----------
    data : np.ndarray
        The complex-valued data to be transformed.

    transform_type : str, optional
        The type of transformation to apply. Options include:
        - 'optm' (default): Optimized translation and rotation.
        - 'trrt': Translation and rotation using provided params.
        - 'real': Extract the real part.
        - 'imag': Extract the imaginary part.
        - 'ampl': Compute the amplitude.
        - 'angl': Compute the phase (in degrees if `deg=True`).

    params : list, optional
        Transformation parameters [x0, y0, phi]. If None and `transform_type='optm'`,
        parameters are estimated automatically.

    deg : bool, optional
        If True, phase transformations return values in degrees (default: True).

    inv_transform : bool, optional
        If true returns transformed data and the function to perform the inverse
        transform.

    full_output : bool, optional
        If True, returns transformed data, the function to perform the inverse
        transform, transformation parameters, and residuals.

    Returns
    -------
    np.ndarray
        The transformed data.

    tuple[np.ndarray, list, np.ndarray] (if `full_output=True`)
        Transformed data, transformation parameters, and residuals.

    Notes
    -----
    - The function applies different transformations based on `transform_type`.
    - If `optm` is selected and `params` is not provided, an optimization routine
      is used to determine the best transformation parameters.

    Example
    -------
    >>> data = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    >>> transformed, params, residuals = transform_data(data, full_output=True)
    >>> print(transformed, params, residuals)
    """

    # Save original shape for reshaping output
    original_shape = data.shape
    flattened_data = data.flatten()

    def transform(data, x0, y0, phi):
        return (data - x0 - 1.0j * y0) * np.exp(1.0j * phi)

    def _inv_transform(data, x0, y0, phi):
        return data * np.exp(-1.0j * phi) + x0 + 1.0j * y0

    def opt_transform(data):
        def transform_err(x):
            return np.sum((transform(data, x[0], x[1], x[2]).imag) ** 2)

        res = minimize(
            fun=transform_err,
            method="Nelder-Mead",
            x0=[
                np.mean(data.real),
                np.mean(data.imag),
                -np.arctan2(np.std(data.imag), np.std(data.real)),
            ],
            options={"maxiter": 1000},
        )

        params = res.x
        transformed_data = transform(data, *params)
        if transformed_data[0] < transformed_data[-1]:
            params[2] += np.pi
        return params

    # Normalize transform_type
    transform_type = str(transform_type).lower()
    if transform_type.startswith(("op", "pr")):
        transform_type = "optm"
    elif transform_type.startswith("translation+rotation"):
        transform_type = "trrt"
    elif transform_type.startswith(("re", "qu")):
        transform_type = "real"
    elif transform_type.startswith(("im", "in")):
        transform_type = "imag"
    elif transform_type.startswith("am"):
        transform_type = "ampl"
    elif transform_type.startswith(("ph", "an")):
        transform_type = "angl"

    # Compute parameters if needed
    if transform_type == "optm" and params is None:
        params = opt_transform(flattened_data)

    # Apply transformation
    if transform_type in ["optm", "trrt"]:
        transformed_flat = transform(flattened_data, *params).real
        residual_flat = transform(flattened_data, *params).imag
    elif transform_type == "real":
        transformed_flat = flattened_data.real
        residual_flat = flattened_data.imag
    elif transform_type == "imag":
        transformed_flat = flattened_data.imag
        residual_flat = flattened_data.real
    elif transform_type == "ampl":
        transformed_flat = np.abs(flattened_data)
        residual_flat = np.unwrap(np.angle(flattened_data))
        if deg:
            residual_flat = np.degrees(residual_flat)
    elif transform_type == "angl":
        transformed_flat = np.unwrap(np.angle(flattened_data))
        residual_flat = np.abs(flattened_data)
        if deg:
            transformed_flat = np.degrees(transformed_flat)

    # Reshape results
    transformed_data = transformed_flat.reshape(original_shape)
    residual = residual_flat.reshape(original_shape)

    # Inverse transformation function
    def inv_transform_fun(d):
        return _inv_transform(d.flatten(), *params).reshape(d.shape)

    if full_output:
        return transformed_data, inv_transform_fun, params, residual
    if inv_transform:
        return transformed_data, inv_transform_fun
    return transformed_data

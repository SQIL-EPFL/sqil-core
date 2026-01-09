from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from scipy.optimize import leastsq, minimize
from tabulate import tabulate

from sqil_core.fit import (
    FitResult,
    fit_circle_algebraic,
    fit_lorentzian,
    fit_output,
    fit_skewed_lorentzian,
    get_best_fit,
)
from sqil_core.utils import estimate_linear_background, format_number
from sqil_core.utils._plot import set_plot_style


@fit_output
def fit_phase_vs_freq_global(
    freq: np.ndarray,
    phase: np.ndarray,
    theta0: float | None = None,
    Q_tot: float | None = None,
    fr: float | None = None,
    disp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits phase response data as a function of frequency using an arctangent model.

    This function models the phase response of a superconducting resonator or circuit
    as a function of frequency. It fits the data using the model:
        θ(f) = θ₀ + 2 * arctan(2 * Q_tot * (1 - f / fr))
    where θ₀ is the phase offset, Q_tot is the total quality factor, and fr is the
    resonant frequency. The fitting is performed using the Nelder-Mead optimization
    method to minimize the sum of squared residuals between the measured and modeled
    phase.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequency data points (in Hz).
    phase : np.ndarray
        Array of measured phase data (in radians).
    theta0 : float, optional
        Initial guess for the phase offset θ₀. If not provided, defaults to the mean
        of `phase`.
    Q_tot : float, optional
        Initial guess for the total quality factor. If not provided, defaults to 0.01.
    fr : float, optional
        Initial guess for the resonant frequency. If not provided, defaults to the mean
        of `freq`.
    disp : bool, optional
        If True, displays optimization progress. Default is True.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`red_chi2`).
        - A callable `predict` function for generating fitted responses.

    Notes
    -----
    - The model assumes the phase response follows the arctangent behavior typical in
      superconducting resonators near resonance.

    Examples
    --------
    >>> freq = np.linspace(5e9, 6e9, 1000)  # Frequency in Hz
    >>> phase = np.random.normal(0, 0.1, size=freq.size)  # Simulated noisy phase data
    >>> popt, perr = fit_phase_vs_freq(freq, phase)
    >>> print("Fitted Parameters (θ₀, Q_tot, fr):", popt)
    >>> print("Percentage Errors:", perr)
    """
    if theta0 is None:
        theta0 = np.mean(phase)
    if Q_tot is None:
        Q_tot = 0.01
    if fr is None:
        fr = np.mean(freq)  # freq[np.argmin(np.abs(phase - np.mean(phase)))]

    def objective(x):
        theta0, Q_tot, fr = x
        model = theta0 + 2 * np.arctan(2 * Q_tot * (1 - freq / fr))
        residuals = phase - model
        return np.square(residuals).sum()

    res = minimize(
        fun=objective,
        x0=[theta0, Q_tot, fr],
        method="Nelder-Mead",
        options={"maxiter": 3000000, "disp": disp},
    )

    return res, {
        "predict": lambda f: theta0 + 2 * np.arctan(2 * Q_tot * (1 - f / fr)),
        "param_names": ["θ₀", "Q_tot", "fr"],
    }


@fit_output
def fit_phase_vs_freq(freq, phase, theta0, Q_tot, fr):
    """
    Fits the phase response of a superconducting resonator using an arctangent model.

    Reference: https://arxiv.org/abs/1410.3365
    This function models the phase response as:
        φ(f) = θ₀ + 2 * arctan(2 * Q_tot * (1 - f / f_r))

    where:
        - φ(f) is the measured phase response (in radians),
        - θ₀ is the phase offset,
        - Q_tot is the total (loaded) quality factor,
        - f_r is the resonant frequency.

    The fitting is performed using a stepwise least-squares optimization to accurately
    estimate the parameters θ₀, Q_tot, and f_r from experimental data.

    Parameters
    ----------
    freq : array-like
        Frequency data (in Hz) at which the phase response was measured.
    phase : array-like
        Unwrapped phase response data (in radians) corresponding to `freq`.
    theta0 : float
        Initial guess for the phase offset θ₀ (in radians).
    Q_tot : float
        Initial guess for the total (loaded) quality factor Q_tot.
    fr : float
        Initial guess for the resonant frequency f_r (in Hz).

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`red_chi2`).
        - A callable `predict` function for generating fitted responses.

    Notes
    -----
    - The fitting is performed in multiple stages for improved stability:
        1. Optimize θ₀ and f_r (fixing Q_tot).
        2. Optimize Q_tot and f_r (fixing θ₀).
        3. Optimize f_r alone.
        4. Optimize Q_tot alone.
        5. Joint optimization of θ₀, Q_tot, and f_r.
    - This stepwise optimization handles parameter coupling and improves convergence.

    Example
    -------
    >>> fitted_params, percent_errors = fit_phase_vs_freq(freq, phase, 0.0, 1000, 5e9)
    >>> print(f"Fitted Parameters: θ₀ = {fitted_params[0]}, Q_tot = {fitted_params[1]}, f_r = {fitted_params[2]}")
    >>> print(f"Percentage Errors: θ₀ = {percent_errors[0]}%, Q_tot = {percent_errors[1]}%, f_r = {percent_errors[2]}%")
    """  # noqa: E501
    # Unwrap the phase of the complex data to avoid discontinuities
    phase = np.unwrap(phase)

    # Define the distance function to handle phase wrapping
    def dist(x):
        np.absolute(x, x)
        c = (x > np.pi).astype(int)
        return x + c * (-2.0 * x + 2.0 * np.pi)

    # Step 1: Optimize θ₀ and fr with Q_tot fixed
    def residuals_1(p, x, y, Q_tot):
        theta0, fr = p
        err = dist(y - (theta0 + 2.0 * np.arctan(2.0 * Q_tot * (1.0 - x / fr))))
        return err

    p0 = [theta0, fr]
    p_final = leastsq(
        lambda a, b, c: residuals_1(a, b, c, Q_tot), p0, args=(freq, phase)
    )
    theta0, fr = p_final[0]

    # Step 2: Optimize Q_tot and fr with θ₀ fixed
    def residuals_2(p, x, y, theta0):
        Q_tot, fr = p
        err = dist(y - (theta0 + 2.0 * np.arctan(2.0 * Q_tot * (1.0 - x / fr))))
        return err

    p0 = [Q_tot, fr]
    p_final = leastsq(
        lambda a, b, c: residuals_2(a, b, c, theta0), p0, args=(freq, phase)
    )
    Q_tot, fr = p_final[0]

    # Step 3: Optimize fr alone
    def residuals_3(p, x, y, theta0, Q_tot):
        fr = p
        err = dist(y - (theta0 + 2.0 * np.arctan(2.0 * Q_tot * (1.0 - x / fr))))
        return err

    p0 = fr
    p_final = leastsq(
        lambda a, b, c: residuals_3(a, b, c, theta0, Q_tot), p0, args=(freq, phase)
    )
    fr = float(p_final[0].item())

    # Step 4: Optimize Q_tot alone
    def residuals_4(p, x, y, theta0, fr):
        Q_tot = p
        err = dist(y - (theta0 + 2.0 * np.arctan(2.0 * Q_tot * (1.0 - x / fr))))
        return err

    p0 = Q_tot
    p_final = leastsq(
        lambda a, b, c: residuals_4(a, b, c, theta0, fr), p0, args=(freq, phase)
    )
    Q_tot = float(p_final[0].item())

    # Step 5: Joint optimization of θ₀, Q_tot, and fr
    def residuals_5(p, x, y):
        theta0, Q_tot, fr = p
        err = dist(y - (theta0 + 2.0 * np.arctan(2.0 * Q_tot * (1.0 - x / fr))))
        return err

    p0 = [theta0, Q_tot, fr]
    final_result = leastsq(residuals_5, p0, args=(freq, phase), full_output=True)

    return (
        final_result,
        {
            "predict": lambda f: theta0 + 2 * np.arctan(2 * Q_tot * (1 - f / fr)),
            "param_names": ["θ₀", "Q_tot", "fr"],
        },
    )


def S11_reflection(
    freq: np.ndarray,
    a: float,
    alpha: float,
    tau: float,
    Q_tot: float,
    fr: float,
    Q_ext: float,
    phi: float,
    mag_bg: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculates the S11 reflection coefficient for a superconducting resonator with an optional magnitude background.

    This function models the S11 reflection parameter, representing how much of an
    incident signal is reflected by a resonator. It includes both the resonator's
    frequency-dependent response and an optional magnitude background correction,
    providing a more accurate fit for experimental data.

    The S11 reflection is computed as:
        S11(f) = env(f) * resonator(f)
    where:
        - env(f) = a * mag_bg(f) * exp(i * α) * exp(2πi * (f - f₀) * τ)
          models the environmental response, including amplitude scaling, phase shifts,
          time delays, and optional frequency-dependent magnitude background.
        - resonator(f) = 1 - [2 * Q_tot / |Q_ext|] * exp(i * φ) / [1 + 2i * Q_tot * (f / fr - 1)]
          models the resonator's frequency response.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequency points (in Hz) at which to evaluate the S11 parameter.
    a : float
        Amplitude scaling factor for the environmental response.
    alpha : float
        Phase offset (in radians) for the environmental response.
    tau : float
        Time delay (in seconds) representing the signal path delay.
    Q_tot : float
        Total quality factor of the resonator (includes internal and external losses).
    fr : float
        Resonant frequency of the resonator (in Hz).
    Q_ext : float
        External quality factor, representing coupling losses to external circuitry.
    phi : float
        Additional phase shift (in radians) in the resonator response.
    mag_bg : np.ndarray or None, optional
        Frequency-dependent magnitude background correction. If provided, it should be
        an array of the same shape as `freq`. Defaults to 1 (no correction).

    Returns
    -------
    S11 : np.ndarray
        Complex array representing the S11 reflection coefficient across the input
        frequencies.

    Notes
    -----
    - Passing mag_bg = np.nan has the same effect of passing mag_bg = None

    Examples
    --------
    >>> freq = np.linspace(4.9e9, 5.1e9, 500)  # Frequency sweep around 5 GHz
    >>> mag_bg = freq**2 + 3 * freq  # Example magnitude background
    >>> S11 = S11_reflection(freq, a=1.0, alpha=0.0, tau=1e-9,
    ...                      Q_tot=5000, fr=5e9, Q_ext=10000, phi=0.0, mag_bg=mag_bg)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(freq, 20 * np.log10(np.abs(S11)))  # Plot magnitude in dB
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.ylabel("S11 Magnitude (dB)")
    >>> plt.title("S11 Reflection Coefficient with Magnitude Background")
    >>> plt.show()
    """  # noqa: E501
    if mag_bg is None or np.isscalar(mag_bg) and np.isnan(mag_bg):
        mag_bg = 1

    env = a * mag_bg * np.exp(1j * alpha) * np.exp(2j * np.pi * (freq - freq[0]) * tau)
    resonator = 1 - (2 * Q_tot / np.abs(Q_ext)) * np.exp(1j * phi) / (
        1 + 2j * Q_tot * (freq / fr - 1)
    )
    return env * resonator


def S21_hanger(
    freq: np.ndarray,
    a: float,
    alpha: float,
    tau: float,
    Q_tot: float,
    fr: float,
    Q_ext: float,
    phi: float,
    mag_bg: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculates the S21 transmission coefficient using the hanger resonator model with an optional magnitude background.

    This function models the S21 transmission parameter, which describes how much of an
    incident signal is transmitted through a superconducting resonator. The model combines
    the resonator's frequency-dependent response with an environmental background response
    and an optional magnitude background correction to more accurately reflect experimental data.

    The S21 transmission is computed as:
        S21(f) = env(f) * resonator(f)
    where:
        - env(f) = a * mag_bg(f) * exp(i * α) * exp(2πi * (f - f₀) * τ)
          models the environmental response, accounting for amplitude scaling, phase shifts,
          and signal path delays.
        - resonator(f) = 1 - [Q_tot / |Q_ext|] * exp(i * φ) / [1 + 2i * Q_tot * (f / fr - 1)]
          models the frequency response of the hanger-type resonator.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequency points (in Hz) at which to evaluate the S21 parameter.
    a : float
        Amplitude scaling factor for the environmental response.
    alpha : float
        Phase offset (in radians) for the environmental response.
    tau : float
        Time delay (in seconds) representing the signal path delay.
    Q_tot : float
        Total quality factor of the resonator (includes internal and external losses).
    fr : float
        Resonant frequency of the resonator (in Hz).
    Q_ext : float
        External quality factor, representing coupling losses to external circuitry.
    phi : float
        Additional phase shift (in radians) in the resonator response.
    mag_bg : np.ndarray or None, optional
        Frequency-dependent magnitude background correction. If provided, it should be
        an array of the same shape as `freq`. Defaults to 1 (no correction).

    Returns
    -------
    S21 : np.ndarray
        Complex array representing the S21 transmission coefficient across the input
        frequencies.

    Notes
    -----
    - Passing mag_bg = np.nan has the same effect of passing mag_bg = None

    Examples
    --------
    >>> freq = np.linspace(4.9e9, 5.1e9, 500)  # Frequency sweep around 5 GHz
    >>> mag_bg = freq**2 + 3 * freq  # Example magnitude background
    >>> S21 = S21_hanger(freq, a=1.0, alpha=0.0, tau=1e-9,
    ...                  Q_tot=5000, fr=5e9, Q_ext=10000, phi=0.0, mag_bg=mag_bg)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(freq, 20 * np.log10(np.abs(S21)))  # Plot magnitude in dB
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.ylabel("S21 Magnitude (dB)")
    >>> plt.title("S21 Transmission Coefficient with Magnitude Background")
    >>> plt.show()
    """  # noqa: E501
    if mag_bg is None or np.isscalar(mag_bg) and np.isnan(mag_bg):
        mag_bg = 1

    env = a * mag_bg * np.exp(1j * alpha) * np.exp(2j * np.pi * (freq - freq[0]) * tau)
    resonator = 1 - (Q_tot / np.abs(Q_ext)) * np.exp(1j * phi) / (
        1 + 2j * Q_tot * (freq / fr - 1)
    )
    return env * resonator


def S21_transmission(
    freq: np.ndarray,
    a: float,
    alpha: float,
    tau: float,
    Q_tot: float,
    fr: float,
    mag_bg: float | None = None,
) -> np.ndarray:
    """
    Computes the complex S21 transmission for a single-pole resonator model.

    This model describes the transmission response of a resonator. The total response
    includes both the resonator and a complex background envelope with a possible linear
    phase delay.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array (in Hz) over which the S21 transmission is evaluated.
    a : float
        Amplitude scaling factor of the background envelope.
    alpha : float
        Phase offset of the background envelope (in radians).
    tau : float
        Time delay in the background response (in seconds).
    Q_tot : float
        Total quality factor of the resonator.
    fr : float
        Resonant frequency of the resonator (in Hz).
    mag_bg : float or None, optional
        Optional background magnitude scaling. If `None` or `NaN`, it defaults to 1.

    Returns
    -------
    np.ndarray
        Complex-valued S21 transmission array over the specified frequency range.
    """

    if mag_bg is None or np.isscalar(mag_bg) and np.isnan(mag_bg):
        mag_bg = 1

    env = a * np.exp(1j * alpha) * np.exp(2j * np.pi * (freq - freq[0]) * tau)
    resonator = 1 / (1 + 2j * Q_tot * (freq / fr - 1))
    return env * resonator


def S11_reflection_mesh(freq, a, alpha, tau, Q_tot, Q_ext, fr, phi):
    """
    Vectorized S11 reflection function.

    Parameters
    ----------
    freq : array, shape (N,)
        Frequency points.
    a, alpha, tau, Q_tot, Q_ext, fr, phi : scalar or array
        Parameters of the S11 model.

    Returns
    -------
    S11 : array
        Complex reflection coefficient. Shape is (M1, M2, ..., N)
        where M1, M2, ... are the broadcasted shapes of the parameters.
    """
    # Ensure freq is at least 2D for broadcasting (1, N)
    freq = np.atleast_1d(freq)  # (N,)

    # Ensure all parameters are at least 1D arrays for broadcasting
    a = np.atleast_1d(a)  # (M1,)
    alpha = np.atleast_1d(alpha)  # (M2,)
    tau = np.atleast_1d(tau)  # (M3,)
    Q_tot = np.atleast_1d(Q_tot)  # (M4,)
    Q_ext = np.atleast_1d(Q_ext)  # (M5,)
    fr = np.atleast_1d(fr)  # (M6,)
    phi = np.atleast_1d(phi)  # (M7,)

    # Reshape frequency to (1, 1, ..., 1, N) for proper broadcasting
    # This makes sure freq has shape (1, 1, ..., N)
    freq = freq[np.newaxis, ...]

    # Calculate the envelope part
    env = (
        a[..., np.newaxis]
        * np.exp(1j * alpha[..., np.newaxis])
        * np.exp(2j * np.pi * (freq - freq[..., 0:1]) * tau[..., np.newaxis])
    )

    # Calculate the resonator part
    resonator = 1 - (
        2 * Q_tot[..., np.newaxis] / np.abs(Q_ext[..., np.newaxis])
    ) * np.exp(1j * phi[..., np.newaxis]) / (
        1 + 2j * Q_tot[..., np.newaxis] * (freq / fr[..., np.newaxis] - 1)
    )

    return env * resonator


def linmag_fit(freq: np.ndarray, data: np.ndarray) -> FitResult:
    """
    Fits the magnitude squared of complex data to a Lorentzian profile.

    This function computes the normalized magnitude of the input complex data and fits
    its squared value to a Lorentzian function to characterize resonance features.
    If the initial Lorentzian fit quality is poor (based on NRMSE), it attempts a fit
    using a skewed Lorentzian model and returns the better fit.

    Parameters
    ----------
    freq : np.ndarray
        Frequency values corresponding to the data points.
    data : np.ndarray
        Complex-valued data to be fitted.

    Returns
    -------
    FitResult
        The best fit result from either the Lorentzian or skewed Lorentzian fit,
        selected based on fit quality.
    """

    linmag = np.abs(data)
    norm_linmag = linmag / np.max(linmag)
    # Lorentzian fit
    fit_res = fit_lorentzian(freq, norm_linmag**2)
    # If the lorentzian fit is bad, try a skewed lorentzian
    if not fit_res.is_acceptable("nrmse"):
        fit_res_skewed = fit_skewed_lorentzian(freq, norm_linmag**2)
        fit_res = get_best_fit(fit_res, fit_res_skewed)

    return fit_res


def quick_fit(
    freq: np.ndarray,
    data: np.ndarray,
    measurement: Literal["reflection", "hanger", "transmission"],
    tau: float | None = None,
    Q_tot: float | None = None,
    fr: float | None = None,
    mag_bg: np.ndarray | None = None,
    fit_range: float | None = None,
    bias_toward_fr: bool = False,
    verbose: bool = False,
    do_plot: bool = False,
) -> tuple[float, float, float, complex, float, float, float]:
    """
    Extracts resonator parameters from complex S-parameter data using circle fitting for reflection or hanger measurements.

    This function analyzes complex-valued resonator data by fitting a circle in the complex plane and
    refining key resonator parameters. It estimates or refines the total quality factor (Q_tot),
    resonance frequency (fr). For reflection and hanger it also estimates the external quality
    factor (Q_ext), while correcting for impedance mismatch.

    Parameters
    ----------
    freq : np.ndarray
        Frequency data.
    data : np.ndarray
        Complex-valued S-parameter data (e.g., S11 for reflection or S21 for hanger configuration).
    measurement : {'reflection', 'hanger'}
        Type of measurement setup. Use 'reflection' for S11 or 'hanger' for S21.
    tau : float, optional
        Initial guess for the cable delay IN RADIANS. If you are passing a value obtained from a linear fit
        divide it by 2pi. If None, it is estimated from a linear fit.
    Q_tot : float, optional
        Initial guess for the total quality factor. If None, it is estimated from a skewed Lorentzian fit.
    fr : float, optional
        Initial guess for the resonance frequency. If None, it is estimated from a skewed Lorentzian fit.
    mag_bg : np.ndarray, optional
        Magnitude background correction data. Defaults to NaN if not provided.
    fit_range: float, optional
        The number x that defines the range [fr-x, fr+x] in which the fitting should be performed. The estimation of
        cable delay and amplitude background will be perfomed on the full data. Defaults to `3 * fr / Q_tot`.
    bias_toward_fr : bool, optional
        If true performs circle fits using the 50% of points closest to the resonance. Defaults is False.
    verbose : bool, optional
        If True, detailed fitting results and progress are printed. Default is False.
    do_plot : bool, optional
        If True, plots the fitted circle and off-resonant point for visualization. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - a (float): Amplitude scaling factor from the off-resonant point.
        - alpha (float): Phase offset from the off-resonant point (in radians).
        - tau (float): Estimated cable delay (in radians).
        - Q_tot (float): Total quality factor.
        - fr (float): Resonance frequency.
        - Q_ext (complex): External quality factor, accounting for impedance mismatch.
        - phi0 (float): Phase shift due to impedance mismatch (in radians).

    Notes
    -----
    - If `tau` is not provided it is estimated by fitting a line through the last 5% of phase points.
    - If `Q_tot` or `fr` is not provided, they are estimated by fitting a skewed Lorentzian model.
    - Visualization helps assess the quality of intermediate steps.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100) + 1j * np.random.rand(100)
    >>> a, alpha, Q_tot, Q_ext, fr, phi0, theta0 = quick_fit(data, measurement='reflection', verbose=True, do_plot=True)
    >>> print(f"Resonance Frequency: {fr} Hz, Q_tot: {Q_tot}, Q_ext: {Q_ext}")
    """  # noqa: E501
    # Sanitize inputs
    if (
        measurement != "reflection"
        and measurement != "hanger"
        and measurement != "transmission"
    ):
        raise Exception(
            f"Invalid measurement type {measurement}. "
            "Must be either 'reflection', 'hanger' or 'transmission'"
        )
    if mag_bg is None:
        mag_bg = np.nan

    # Define amplitude and phase
    linmag = np.abs(data)
    phase = np.unwrap(np.angle(data))

    # Inital estimate for Q_tot and fr by fitting a lorentzian on
    # the squared manitude data
    if (Q_tot is None) or (fr is None):
        if verbose:
            print("* Lorentzian estimation of fr and Q_tot")
        norm_linmag = linmag / np.max(linmag)
        fit_res = fit_lorentzian(freq, norm_linmag**2)
        _, efr, fwhm, _ = fit_res.params
        eQ_tot = efr / fwhm
        # If the lorentzian fit is bad, try a skewed lorentzian
        if fit_res.metrics["nrmse"] > 0.5:
            fit_res_skewed = fit_skewed_lorentzian(freq, norm_linmag**2)
            (A1, A2, A3, A4, efr, eQ_tot) = fit_res.params
            delta_aic = fit_res["aic"] - fit_res_skewed["aic"]
            fit_res = fit_res_skewed if delta_aic >= 0 else fit_res

        # Assign only parameters for which no initial guess was provided
        Q_tot = Q_tot or eQ_tot
        fr = fr or efr
        if verbose:
            print(fit_res.model_name)
            fit_res.summary()
            if fr != efr:
                print(f" -> Still considering fr = {fr}\n")
            elif Q_tot != eQ_tot:
                print(f" -> Still considering Q_tot = {Q_tot}\n")

    # Initial estimate for tau by fitting a line through the last 5% of phase points
    if tau is None:
        if verbose:
            print("* Linear estimation of cable delay from the last 5% of phase points")
        [_, tau] = estimate_linear_background(
            freq, phase, points_cut=0.05, cut_from_back=True
        )
        tau /= 2 * np.pi
        if verbose:
            print(f" -> tau [rad]: {tau}\n")

    # Remove cable delay
    phase1 = phase - 2 * np.pi * tau * (freq - freq[0])
    data1 = linmag * np.exp(1j * phase1)

    # Cut data around the estimated resonance frequency
    if fit_range is None:
        fit_range = 3 * fr / Q_tot
    mask = (freq > fr - fit_range) & (freq < fr + fit_range)
    freq, linmag, phase, data = freq[mask], linmag[mask], phase[mask], data[mask]
    phase1, data1 = phase1[mask], data1[mask]
    if not np.isscalar(mag_bg):
        mag_bg = mag_bg[mask]

    # Move cirle to center
    if bias_toward_fr:
        fr_idx = np.abs(freq - fr).argmin()
        idx_range = int(len(freq) / 4)
        re, im = np.real(data1), np.imag(data1)
        fit_res = fit_circle_algebraic(
            re[fr_idx - idx_range : fr_idx + idx_range],
            im[fr_idx - idx_range : fr_idx + idx_range],
        )
    else:
        fit_res = fit_circle_algebraic(np.real(data1), np.imag(data1))
    (xc, yc, r0) = fit_res.params
    data3 = data1 - xc - 1j * yc
    phase3 = np.unwrap(np.angle(data3))

    # Fit phase vs frequency
    if verbose:
        print("* Phase vs frequency fit")
    fit_res = fit_phase_vs_freq(freq, phase3, theta0=0, Q_tot=Q_tot, fr=fr)
    (theta0, Q_tot, fr) = fit_res.params
    if verbose:
        fit_res.summary()

    # Find the off-resonant point
    p_offres = (xc + 1j * yc) + r0 * np.exp(1j * (theta0 + np.pi))
    a = np.abs(p_offres)
    alpha = np.angle(p_offres)
    # Rescale data
    linmag5 = linmag / a
    phase5 = phase1 - alpha
    data5 = linmag5 * np.exp(1j * phase5)

    Q_ext = None
    phi0 = None
    if measurement == "reflection" or measurement == "hanger":
        # Find impedence mismatch
        if bias_toward_fr:
            fr_idx = np.abs(freq - fr).argmin()
            re, im = np.real(data5), np.imag(data5)
            fit_res = fit_circle_algebraic(
                re[fr_idx - idx_range : fr_idx + idx_range],
                im[fr_idx - idx_range : fr_idx + idx_range],
            )
        else:
            fit_res = fit_circle_algebraic(np.real(data5), np.imag(data5))
        (xc6, yc6, r06) = fit_res.params
        phi0 = -np.arcsin(yc6 / r06)

        # Q_ext and Q_int
        if measurement == "reflection":
            Q_ext = Q_tot / (r06 * np.exp(-1j * phi0))
        elif measurement == "hanger":
            Q_ext = Q_tot / (2 * r06 * np.exp(-1j * phi0))

    # Refine phase offset and amplitude scaling
    if measurement == "reflection":
        res6 = S11_reflection(freq, a, alpha, tau, Q_tot, Q_ext, fr, phi0, mag_bg / a)
    elif measurement == "hanger":
        res6 = S21_hanger(freq, a, alpha, tau, Q_tot, Q_ext, fr, phi0, mag_bg / a)
    elif measurement == "transmission":
        res6 = S21_transmission(freq, a, alpha, tau, Q_tot, fr)
        a *= (np.max(linmag) - np.min(linmag)) / (
            np.max(np.abs(res6)) - np.min(np.abs(res6))
        )
        alpha += phase[0] - np.unwrap(np.angle(res6))[0]

    # Plot small summary
    if do_plot:
        v = np.linspace(0, 2 * np.pi, 100)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(
            np.real(data1),
            np.imag(data1),
            "o",
            label="Data cut (without cable delay)",
            zorder=1,
        )
        ax.plot(xc + r0 * np.cos(v), yc + r0 * np.sin(v), label="Circle fit", zorder=2)
        ax.scatter(
            np.real(p_offres),
            np.imag(p_offres),
            color="tab:cyan",
            label="Off-resonant point",
            zorder=3,
            s=120,
            marker="*",
        )
        ax.scatter(xc, yc, color="tab:red")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.axis("equal")
        ax.margins(x=0.25, y=0.25)
        ax.grid(True)
        ax.set_title("Data fit and off-resonant point")
        ax.legend()
        # Show
        fig.tight_layout()
        plt.show()

    return a, alpha, tau, Q_tot, fr, Q_ext, phi0


@fit_output
def full_fit(
    freq, data, measurement, a, alpha, tau, Q_tot, fr, Q_ext=1, phi0=0, mag_bg=None
) -> FitResult:
    """
    Performs a full fit of the measured resonator data using a selected model
    (either reflection or hanger-type measurement). The fitting is handled
    using the lmfit Model framework.

    IMPORTANT: This fitting function should only be used to refine already
    good initial guesses!

    Parameters
    ----------
    freq : np.ndarray
        A 1D array of frequency values in Hz.

    data : np.ndarray
        A 1D array of complex-valued measured resonator data.

    measurement : str
        Type of measurement. Should be either:
        - `"reflection"`: Uses the `S11_reflection` model.
        - `"hanger"`: Uses the `S21_hanger` model.

    a : float
        Amplitude scaling factor.

    alpha : float
        Phase offset parameter.

    tau : float
        Cable delay or propagation time.

    Q_tot : float
        Total quality factor of the resonator.

    fr : float
        Resonant frequency.

    Q_ext : float
        External quality factor (coupling quality factor).
        Only for reflection and hanger.

    phi0 : float
        Phase offset at resonance. Only for relfection and hanger.

    mag_bg : np.ndarray, optional
        A 1D array representing the magnitude background response, if available.
        Default is None.

    Returns
    -------
    FitResult
        A `FitResult` object containing:
        - Fitted parameters (`params`).
        - Standard errors (`std_err`).
        - Goodness-of-fit metrics (`red_chi2`).
        - A callable `predict` function for generating fitted responses.

    Example
    -------
    >>> freq = np.linspace(1e9, 2e9, 1000)  # Example frequency range
    >>> data = np.exp(1j * freq * 2 * np.pi / 1e9)  # Example complex response
    >>> fit_result = full_fit(freq, data, "reflection", 1, 0, 0, 1e4, 2e4, 1.5e9, 0)
    >>> fit_result.summary()
    """
    model_name = None

    if measurement == "reflection":

        def S11_reflection_fixed(freq, a, alpha, tau, Q_tot, fr, Q_ext_mag, phi):
            return S11_reflection(
                freq, a, alpha, tau, Q_tot, fr, Q_ext_mag, phi, mag_bg
            )

        model = Model(S11_reflection_fixed)
        params = model.make_params(
            a=a, alpha=alpha, tau=tau, Q_tot=Q_tot, fr=fr, Q_ext_mag=Q_ext, phi=phi0
        )
        model_name = "S11_reflection"

    elif measurement == "hanger":

        def S21_hanger_fixed(freq, a, alpha, tau, Q_tot, fr, Q_ext_mag, phi):
            return S21_hanger(freq, a, alpha, tau, Q_tot, fr, Q_ext_mag, phi, mag_bg)

        model = Model(S21_hanger_fixed)
        params = model.make_params(
            a=a, alpha=alpha, tau=tau, Q_tot=Q_tot, fr=fr, Q_ext_mag=Q_ext, phi=phi0
        )
        model_name = "S21_hanger"

    elif measurement == "transmission":

        def S21_transmission_fixed(freq, a, alpha, tau, Q_tot, fr):
            return S21_transmission(freq, a, alpha, tau, Q_tot, fr, mag_bg)

        model = Model(S21_transmission_fixed)
        params = model.make_params(a=a, alpha=alpha, tau=tau, Q_tot=Q_tot, fr=fr)
        model_name = "S21_transmission"

    res = model.fit(data, params, freq=freq)
    return res, {"model_name": model_name}


def plot_resonator(
    freq, data, x_fit=None, y_fit=None, mag_bg: np.ndarray | None = None, title=""
):
    """
    Plots the resonator response in three different representations:
    - Complex plane (Re vs. Im)
    - Magnitude response (Amplitude vs. Frequency)
    - Phase response (Phase vs. Frequency)

    Parameters
    ----------
    freq : np.ndarray
        A 1D array representing the frequency values.

    data : np.ndarray
        A 1D array of complex-valued data points corresponding to the resonator response

    fit : np.ndarray, optional
        A 1D array of complex values representing the fitted model response.
        Default is None.

    mag_bg : np.ndarray, optional
        A 1D array representing the background magnitude response, if available.
        Default is None.

    title : str, optional
        The title of the plot. Default is an empty string.
    """

    set_plot_style(plt)

    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 2)
    ms = plt.rcParams.get("lines.markersize")

    # Subplot on the left (full height, first column)
    ax1 = fig.add_subplot(gs[:, 0])  # Left side spans both rows
    ax1.plot(np.real(data), np.imag(data), "o", color="tab:blue", ms=ms + 1)
    if y_fit is not None:
        ax1.plot(np.real(y_fit), np.imag(y_fit), color="tab:red")
    ax1.set_aspect("equal")
    ax1.set_xlabel("In-phase")
    ax1.set_ylabel("Quadrature")
    ax1.grid(True)

    # Subplot on the top-right (first row, second column)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(freq, np.abs(data), "o", color="tab:blue", ms=ms - 1)
    if y_fit is not None:
        ax2.plot(x_fit, np.abs(y_fit), color="tab:red")
    if (mag_bg is not None) and (not np.isnan(mag_bg).any()):
        ax2.plot(freq, mag_bg, "-.", color="tab:green")
    ax2.set_ylabel("Magnitude [V]")
    ax2.grid(True)

    # Subplot on the bottom-right (second row, second column)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(freq, np.unwrap(np.angle(data)), "o", color="tab:blue", ms=ms - 1)
    if y_fit is not None:
        ax3.plot(x_fit, np.unwrap(np.angle(y_fit)), color="tab:red")
    ax3.set_ylabel("Phase [rad]")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.grid(True)

    fig.suptitle(title)
    fig.tight_layout()

    return fig, (ax1, ax2, ax3)


def print_resonator_params(fit_params, measurement):
    table_data = []

    if measurement == "reflection" or measurement == "hanger":
        a, alpha, tau, Q_tot, fr, Q_ext_mag, phi0 = fit_params
        Q_ext = Q_ext_mag * np.exp(1j * phi0)
        Q_int = compute_Q_int(Q_tot, Q_ext_mag, phi0)
        kappa_ext = fr / np.real(Q_ext)
        kappa_int = fr / Q_int
        kappa_tot = fr / Q_tot

        table_data.append(["fr", f"{format_number(fr, 6, unit='Hz', latex=False)}"])
        table_data.append(["Re[Q_ext]", f"{np.real(Q_ext):.0f}"])
        table_data.append(["Q_int", f"{Q_int:.0f}"])
        table_data.append(["Q_tot", f"{Q_tot:.0f}"])

        table_data.append(
            ["kappa_ext", format_number(kappa_ext, 4, unit="Hz", latex=False)]
        )
        table_data.append(
            ["kappa_int", format_number(kappa_int, 4, unit="Hz", latex=False)]
        )
        table_data.append(
            ["kappa_tot", format_number(kappa_tot, 4, unit="Hz", latex=False)]
        )
    elif measurement == "transmission":
        a, alpha, tau, Q_tot, fr = fit_params
        kappa_tot = fr / Q_tot
        table_data.append(["fr", f"{format_number(fr, 6, unit='Hz', latex=False)}"])
        table_data.append(["Q_tot", f"{Q_tot:.0f}"])
        table_data.append(
            ["kappa_tot", format_number(kappa_tot, 4, unit="Hz", latex=False)]
        )

    print(tabulate(table_data, headers=["Param", "Value"], tablefmt="github"))


def compute_Q_int(Q_tot, Q_ext_mag, Q_ext_phase):
    """Compute Q_internal given Q_total and the manitude and phase of Q_external."""
    return 1 / (1 / Q_tot - np.real(1 / (Q_ext_mag * np.exp(-1j * Q_ext_phase))))

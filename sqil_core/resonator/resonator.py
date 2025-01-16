import numpy as np
from scipy.optimize import leastsq, minimize

from sqil_core.fit import compute_standard_errors, compute_standard_errors_minimize


def fit_phase_vs_freq_global(
    freq: np.ndarray,
    phase: np.ndarray,
    theta0: float | None = None,
    Q_tot: float | None = None,
    fr: float | None = None,
    disp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits phase response data as a function of frequency using an arctangent model.

    This function models the phase response of a superconducting resonator or circuit
    as a function of frequency. It fits the data using the model:
        θ(f) = θ₀ + 2 * arctan(2 * Q_tot * (1 - f / fr))
    where θ₀ is the phase offset, Q_tot is the total quality factor, and fr is the
    resonant frequency. The fitting is performed using the Nelder-Mead optimization
    method to minimize the sum of squared residuals between the measured and modeled phase.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequency data points (in Hz).
    phase : np.ndarray
        Array of measured phase data (in radians).
    theta0 : float, optional
        Initial guess for the phase offset θ₀. If not provided, defaults to the mean of `phase`.
    Q_tot : float, optional
        Initial guess for the total quality factor. If not provided, defaults to 0.01.
    fr : float, optional
        Initial guess for the resonant frequency. If not provided, defaults to the mean of `freq`.
    disp : bool, optional
        If True, displays optimization progress. Default is True.

    Returns
    -------
    res.x : np.ndarray
        Optimized parameters `[θ₀, Q_tot, fr]` after fitting.
    perc_errors : np.ndarray
        Percentage errors for each optimized parameter, providing uncertainty estimates.

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

    std_errors, perc_errors = compute_standard_errors_minimize(freq, res, objective)

    return res.x, perc_errors


def fit_phase_vs_freq(freq, phase, theta0, Q_tot, fr):
    """
    Fits the phase response of a superconducting resonator using an arctangent model.

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
    p_final : np.ndarray
        Optimized parameters [θ₀, Q_tot, f_r].
    perc_errs : np.ndarray
        Percentage errors of the fitted parameters [θ₀_error%, Q_tot_error%, f_r_error%].

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
    """
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
    fr = float(p_final[0])

    # Step 4: Optimize Q_tot alone
    def residuals_4(p, x, y, theta0, fr):
        Q_tot = p
        err = dist(y - (theta0 + 2.0 * np.arctan(2.0 * Q_tot * (1.0 - x / fr))))
        return err

    p0 = Q_tot
    p_final = leastsq(
        lambda a, b, c: residuals_4(a, b, c, theta0, fr), p0, args=(freq, phase)
    )
    Q_tot = float(p_final[0])

    # Step 5: Joint optimization of θ₀, Q_tot, and fr
    def residuals_5(p, x, y):
        theta0, Q_tot, fr = p
        err = dist(y - (theta0 + 2.0 * np.arctan(2.0 * Q_tot * (1.0 - x / fr))))
        return err

    p0 = [theta0, Q_tot, fr]
    p_final, pcov, infodict, errmsg, ier = leastsq(
        residuals_5, p0, args=(freq, phase), full_output=True
    )

    # Compute standard and percentage errors
    std_errors, perc_errors = compute_standard_errors(
        freq, p_final, pcov, infodict["fvec"]
    )

    return p_final, perc_errors


def S11_reflection(
    freq: np.ndarray,
    a: float,
    alpha: float,
    tau: float,
    Q_tot: float,
    Q_ext: float,
    fr: float,
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
    Q_ext : float
        External quality factor, representing coupling losses to external circuitry.
    fr : float
        Resonant frequency of the resonator (in Hz).
    phi : float
        Additional phase shift (in radians) in the resonator response.
    mag_bg : np.ndarray or None, optional
        Frequency-dependent magnitude background correction. If provided, it should be
        an array of the same shape as `freq`. Defaults to 1 (no correction).

    Returns
    -------
    S11 : np.ndarray
        Complex array representing the S11 reflection coefficient across the input frequencies.

    Examples
    --------
    >>> freq = np.linspace(4.9e9, 5.1e9, 500)  # Frequency sweep around 5 GHz
    >>> mag_bg = freq**2 + 3 * freq  # Example magnitude background
    >>> S11 = S11_reflection(freq, a=1.0, alpha=0.0, tau=1e-9,
    ...                      Q_tot=5000, Q_ext=10000, fr=5e9, phi=0.0, mag_bg=mag_bg)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(freq, 20 * np.log10(np.abs(S11)))  # Plot magnitude in dB
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.ylabel("S11 Magnitude (dB)")
    >>> plt.title("S11 Reflection Coefficient with Magnitude Background")
    >>> plt.show()
    """
    if mag_bg is None:
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
    Q_ext: float,
    fr: float,
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
    Q_ext : float
        External quality factor, representing coupling losses to external circuitry.
    fr : float
        Resonant frequency of the resonator (in Hz).
    phi : float
        Additional phase shift (in radians) in the resonator response.
    mag_bg : np.ndarray or None, optional
        Frequency-dependent magnitude background correction. If provided, it should be
        an array of the same shape as `freq`. Defaults to 1 (no correction).

    Returns
    -------
    S21 : np.ndarray
        Complex array representing the S21 transmission coefficient across the input frequencies.

    Examples
    --------
    >>> freq = np.linspace(4.9e9, 5.1e9, 500)  # Frequency sweep around 5 GHz
    >>> mag_bg = freq**2 + 3 * freq  # Example magnitude background
    >>> S21 = S21_hanger(freq, a=1.0, alpha=0.0, tau=1e-9,
    ...                  Q_tot=5000, Q_ext=10000, fr=5e9, phi=0.0, mag_bg=mag_bg)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(freq, 20 * np.log10(np.abs(S21)))  # Plot magnitude in dB
    >>> plt.xlabel("Frequency (Hz)")
    >>> plt.ylabel("S21 Magnitude (dB)")
    >>> plt.title("S21 Transmission Coefficient with Magnitude Background")
    >>> plt.show()
    """
    if mag_bg is None:
        mag_bg = 1
    env = a * mag_bg * np.exp(1j * alpha) * np.exp(2j * np.pi * (freq - freq[0]) * tau)
    resonator = 1 - (Q_tot / np.abs(Q_ext)) * np.exp(1j * phi) / (
        1 + 2j * Q_tot * (freq / fr - 1)
    )
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
        Complex reflection coefficient. Shape is (M1, M2, ..., N) where M1, M2, ... are the broadcasted shapes of the parameters.
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

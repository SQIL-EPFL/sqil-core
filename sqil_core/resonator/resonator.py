import numpy as np
from scipy.optimize import minimize

from sqil_core.fit import compute_standard_errors_minimize


def fit_phase_vs_freq_bounded(
    freq: np.ndarray,
    phase: np.ndarray,
    theta_0: float | None = None,
    Q_tot: float | None = None,
    f_r: float | None = None,
    bound_theta_0: tuple[float, float] | None = None,
    bound_Q_tot: tuple[float, float] | None = None,
    bound_f_r: tuple[float, float] | None = None,
):
    # Set initial guesses if not provided
    if theta_0 is None:
        theta_0 = np.mean(phase)
    if Q_tot is None:
        Q_tot = 0.01
    if f_r is None:
        f_r = np.mean(freq)  # freq[np.argmin(np.abs(phase - np.mean(phase)))]

    # Set initial bounds if not provided
    if bound_theta_0 is None:
        bound_theta_0 = (0, 2 * np.pi)
    if bound_Q_tot is None:
        bound_Q_tot = (0, 1e9)
    if bound_f_r is None:
        bound_f_r = (freq[0], freq[-1])

    bounds = [bound_theta_0, bound_Q_tot, bound_f_r]

    def objective(x):
        theta_0, Q_tot, f_r = x
        model = theta_0 + 2 * np.arctan(2 * Q_tot * (1 - freq / f_r))
        residuals = phase - model
        return np.square(residuals).sum()

    res = minimize(
        fun=objective, x0=[theta_0, Q_tot, f_r], method="Powell", bounds=bounds
    )

    std_errors, perc_errors = compute_standard_errors_minimize(freq, res, objective)

    return res.x, perc_errors


def fit_phase_vs_freq(
    freq: np.ndarray,
    phase: np.ndarray,
    theta_0: float | None = None,
    Q_tot: float | None = None,
    f_r: float | None = None,
    bound_theta_0: tuple[float, float] | None = None,
    bound_Q_tot: tuple[float, float] | None = None,
    bound_f_r: tuple[float, float] | None = None,
):
    # Set initial guesses if not provided
    if theta_0 is None:
        theta_0 = np.mean(phase)
    if Q_tot is None:
        Q_tot = 0.01
    if f_r is None:
        f_r = np.mean(freq)  # freq[np.argmin(np.abs(phase - np.mean(phase)))]

    # Set initial bounds if not provided
    if bound_theta_0 is None:
        bound_theta_0 = (0, 2 * np.pi)
    if bound_Q_tot is None:
        bound_Q_tot = (0, 1e9)
    if bound_f_r is None:
        bound_f_r = (freq[0], freq[-1])

    bounds = [bound_theta_0, bound_Q_tot, bound_f_r]

    def objective(x):
        theta_0, Q_tot, f_r = x
        model = theta_0 + 2 * np.arctan(2 * Q_tot * (1 - freq / f_r))
        residuals = phase - model
        return np.square(residuals).sum()

    res = minimize(fun=objective, x0=[theta_0, Q_tot, f_r], method="Nelder-Mead")

    std_errors, perc_errors = compute_standard_errors_minimize(freq, res, objective)

    return res.x, perc_errors


def S11_reflection(freq, a, alpha, tau, Q_tot, Q_ext, f_r, phi):
    env = a * np.exp(1j * alpha) * np.exp(2j * np.pi * (freq - freq[0]) * tau)
    resonator = 1 - (2 * Q_tot / np.abs(Q_ext)) * np.exp(1j * phi) / (
        1 + 2j * Q_tot * (freq / f_r - 1)
    )
    return env * resonator


def S21_hanger(freq, a, alpha, tau, Q_tot, Q_ext, f_r, phi, mag_bg=1):
    Deltaf = freq - freq[0]
    env = a * (mag_bg) * np.exp(1j * alpha) * np.exp(2j * np.pi * Deltaf * tau)
    resonator = 1 - (Q_tot / np.abs(Q_ext)) * np.exp(1j * phi) / (
        1 + 2j * Q_tot * (freq / f_r - 1)
    )
    return env * resonator


def S11_reflection_mesh(freq, a, alpha, tau, Q_tot, Q_ext, f_r, phi):
    """
    Vectorized S11 reflection function.

    Parameters
    ----------
    freq : array, shape (N,)
        Frequency points.
    a, alpha, tau, Q_tot, Q_ext, f_r, phi : scalar or array
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
    f_r = np.atleast_1d(f_r)  # (M6,)
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
        1 + 2j * Q_tot[..., np.newaxis] * (freq / f_r[..., np.newaxis] - 1)
    )

    return env * resonator

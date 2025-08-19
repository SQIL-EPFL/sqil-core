import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import hilbert

from sqil_core.utils import compute_fft, get_peaks


def estimate_peak(
    x_data: np.ndarray, y_data: np.ndarray
) -> tuple[float, float, float, float, bool]:
    """
    Estimates the key properties of a peak or dip in 1D data.

    This function analyzes a one-dimensional dataset to identify whether the dominant
    feature is a peak or dip and then estimates the following parameters:
    - The position of the peak/dip (x0)
    - The full width at half maximum (FWHM)
    - The peak/dip height
    - The baseline value (y0)
    - A flag indicating if it is a peak (True) or a dip (False)

    Parameters
    ----------
    x_data : np.ndarray
        Array of x-values.
    y_data : np.ndarray
        Array of y-values corresponding to `x_data`.

    Returns
    -------
    x0 : float
        The x-position of the peak or dip.
    fwhm : float
        Estimated full width at half maximum.
    peak_height : float
        Height (or depth) of the peak or dip relative to the baseline.
    y0 : float
        Baseline level from which the peak/dip is measured.
    is_peak : bool
        True if the feature is a peak; False if it is a dip.

    Notes
    -----
    - The function uses the median of `y_data` to determine whether the dominant
      feature is a peak or a dip.
    - FWHM is estimated using the positions where the signal crosses the half-max level.
    - If fewer than two crossings are found, a fallback FWHM is estimated as 1/10th
      of the x-range.
    """

    x, y = x_data, y_data
    y_median = np.median(y)
    y_max, y_min = np.max(y), np.min(y)

    # Determine if it's a peak or dip
    if y_max - y_median >= y_median - y_min:
        idx = np.argmax(y)
        is_peak = True
        y0 = y_min
        peak_height = y_max - y0
    else:
        idx = np.argmin(y)
        is_peak = False
        y0 = y_max
        peak_height = y0 - y_min

    x0 = x[idx]

    # Estimate FWHM using half-max crossings
    half_max = y0 + (peak_height / 2.0 if is_peak else -peak_height / 2.0)
    crossings = np.where(np.diff(np.sign(y - half_max)))[0]
    if len(crossings) >= 2:
        fwhm = np.abs(x[crossings[-1]] - x[crossings[0]])
    else:
        fwhm = (x[-1] - x[0]) / 10.0

    return x0, fwhm, peak_height, y0, is_peak


def lorentzian_guess(x_data, y_data):
    """Guess lorentzian fit parameters."""
    x0, fwhm, peak_height, y0, is_peak = estimate_peak(x_data, y_data)

    # Compute A from peak height = 2A / FWHM
    A = (peak_height * fwhm) / 2.0
    if not is_peak:
        A = -A

    guess = [A, x0, fwhm, y0]
    return guess


def lorentzian_bounds(x_data, y_data, guess):
    """Guess lorentzian fit bounds."""
    x, y = x_data, y_data
    A, *_ = guess

    x_span = np.max(x) - np.min(x)
    A_abs = np.abs(A) if A != 0 else 1.0
    fwhm_min = (x[1] - x[0]) if len(x) > 1 else x_span / 10

    bounds = (
        [-10 * A_abs, np.min(x) - 0.1 * x_span, fwhm_min, np.min(y) - 0.5 * A_abs],
        [+10 * A_abs, np.max(x) + 0.1 * x_span, x_span, np.max(y) + 0.5 * A_abs],
    )
    return bounds


def gaussian_guess(x_data, y_data):
    """Guess gaussian fit parameters."""
    x0, fwhm, peak_height, y0, is_peak = estimate_peak(x_data, y_data)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to σ

    A = peak_height * sigma * np.sqrt(2 * np.pi)
    if not is_peak:
        A = -A

    guess = [A, x0, sigma, y0]
    return guess


def gaussian_bounds(x_data, y_data, guess):
    """Guess gaussian fit bounds."""
    x, y = x_data, y_data
    A, *_ = guess

    x_span = np.max(x) - np.min(x)
    sigma_min = (x[1] - x[0]) / 10 if len(x) > 1 else x_span / 100
    sigma_max = x_span
    A_abs = np.abs(A)

    bounds = (
        [-10 * A_abs, np.min(x) - 0.1 * x_span, sigma_min, np.min(y) - 0.5 * A_abs],
        [10 * A_abs, np.max(x) + 0.1 * x_span, sigma_max, np.max(y) + 0.5 * A_abs],
    )
    return bounds


def oscillations_guess(x_data, y_data, num_init=10):
    """Generate robust initial guesses for oscillation parameters."""
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    dx = np.mean(np.diff(x_data))

    # Amplitude guess (robust against outliers)
    A = (np.percentile(y_data, 95) - np.percentile(y_data, 5)) / 2

    # Offset guess (tail median + mean)
    y0_tail = np.median(y_data[-max(5, len(y_data) // 10) :])
    y0_mean = np.mean(y_data)
    y0_candidates = [y0_tail, y0_mean]

    # FFT-based T (period)
    y_demeaned = y_data - np.mean(y_data)
    freqs = rfftfreq(len(x_data), d=dx)
    spectrum = np.abs(rfft(y_demeaned))
    peak_idx = np.argmax(spectrum[1:]) + 1  # Ignore DC
    freq_peak = freqs[peak_idx]
    T = 1 / freq_peak if freq_peak > 0 else np.ptp(x_data)  # fallback to range

    # Phase estimate from cross-correlation
    cos_wave = np.cos(2 * np.pi * x_data / T)
    lag = np.argmax(np.correlate(y_demeaned, cos_wave, mode="full")) - len(x_data) + 1
    phi_base = x_data[0] + lag * dx
    phi_candidates = np.linspace(phi_base - T, phi_base + T, num_init)
    phi_candidates = np.mod(phi_candidates, T)

    return [A, y0_candidates, phi_candidates, T]


def oscillations_bounds(x_data, y_data, guess):
    """Generate realistic bounds for oscillation parameters."""
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    A, y0, phi, T = guess

    # Add small offset to ensure bounds don't collaps
    eps = 1e-12

    A_min = 0.1 * A - eps
    A_max = 10 * A

    y0_min = np.min(y_data) - eps
    y0_max = np.max(y_data)

    phi_min = 0.0 - eps
    phi_max = T  # reasonable 1-period wrap

    T_min = 0.1 * T - eps
    T_max = 10 * T

    lower = [A_min, y0_min, phi_min, T_min]
    upper = [A_max, y0_max, phi_max, T_max]
    return (lower, upper)


def decaying_oscillations_guess(x_data, y_data, num_init=10):
    """Generate robust initial guesses for decaying oscillation parameters."""
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # Oscillations params
    A, y0_candidates, phi_candidates, T = oscillations_guess(x_data, y_data, num_init)

    # Decay time (tau) from log-envelope
    try:
        y_demeaned = y_data - np.mean(y_data)
        envelope = np.abs(hilbert(y_demeaned))
        log_env = np.log(np.clip(envelope, 1e-10, None))
        slope, _ = np.polyfit(x_data, log_env, 1)
        tau = -1 / slope if slope < 0 else np.ptp(x_data)
    except Exception:
        tau = np.ptp(x_data)

    # Rough estimate of y0 with a local min or mean of last N points
    N_tail = max(3, int(0.1 * len(y_data)))
    tail_mean = np.mean(y_data[-N_tail:])
    y0_decay = min(np.min(y_data), tail_mean)
    y0_candidates.append(y0_decay)

    return [A, tau, y0_candidates, phi_candidates, T]


def decaying_oscillations_bounds(x_data, y_data, guess):
    """Generate realistic bounds for decaying oscillation parameters."""
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    A, tau, y0, phi, T = guess
    lower, upper = oscillations_bounds(x_data, y_data, [A, y0, phi, T])

    tau_min = 0.01 * tau
    tau_max = 10 * tau

    lower.insert(1, tau_min)
    upper.insert(1, tau_max)
    return (lower, upper)


def many_decaying_oscillations_guess(x_data, y_data, n):
    offset = np.mean(y_data)
    y_centered = y_data - offset

    freqs, fft_mag = compute_fft(x_data, y_centered)
    peak_freqs, peak_mags = get_peaks(freqs, fft_mag)

    if len(peak_freqs) < n:
        raise ValueError(
            f"Not enough frequency peaks found to initialize {n} oscillations."
        )

    guess = []
    signal_duration = x_data[-1] - x_data[0]

    for i in range(n):
        A = peak_mags[i]
        tau = signal_duration / (2 + i)  # Increasing τ for later oscillations
        phi = 0.0  # Can be refined
        T = peak_freqs[i]
        guess.extend([A, tau, phi, T])

    guess.append(offset)
    return guess


def decaying_exp_guess(x_data: np.ndarray, y_data: np.ndarray) -> list[float]:
    """
    Robust initial guess for decaying exponential even if the full decay isn't captured.
    """
    x = np.asarray(x_data)
    y = np.asarray(y_data)

    # Rough estimate of y0 with a local min or mean of last N points
    N_tail = max(3, int(0.1 * len(y)))
    tail_mean = np.mean(y[-N_tail:])
    y0 = min(np.min(y), tail_mean)

    # Amplitude
    A = y[0] - y0
    A = np.clip(A, 1e-12, None)

    # Ensure sign consistency
    if np.abs(np.max(y) - y0) > np.abs(A):
        A = np.max(y) - y0

    # Estimate tau using log-linear fit of the first ~30% of data
    N_fit = max(5, int(0.3 * len(x)))
    y_fit = y[:N_fit] - y0
    mask = y_fit > 0  # log() only valid on positive values

    if np.count_nonzero(mask) > 1:
        x_fit = x[:N_fit][mask]
        log_y = np.log(y_fit[mask])
        slope, intercept = np.polyfit(x_fit, log_y, 1)
        tau = -1 / slope if slope < 0 else (x[-1] - x[0]) / 3
    else:
        tau = (x[-1] - x[0]) / 3

    tau = max(tau, np.finfo(float).eps)

    return [A, tau, y0]

import numpy as np


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

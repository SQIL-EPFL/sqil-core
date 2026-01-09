import numpy as np
from scipy.signal import argrelextrema, find_peaks


def remove_offset(data: np.ndarray, avg: int = 3) -> np.ndarray:
    """Removes the initial offset from a data matrix or vector by subtracting
    the average of the first `avg` points. After applying this function,
    the first point of each column of the data will be shifted to (about) 0.

    Parameters
    ----------
    data : np.ndarray
        Input data, either a 1D vector or a 2D matrix
    avg : int, optional
        The number of initial points to average when calculating
        the offset, by default 3

    Returns
    -------
    np.ndarray
       The input data with the offset removed
    """
    is1D = len(data.shape) == 1
    if is1D:
        return data - np.mean(data[0:avg])
    return data - np.mean(data[:, 0:avg], axis=1).reshape(data.shape[0], 1)


def estimate_linear_background(
    x: np.ndarray,
    data: np.ndarray,
    points_cut: float = 0.1,
    cut_from_back: bool = False,
) -> list:
    """
    Estimates the linear background for a given data set by fitting a linear model to a subset of the data.

    This function performs a linear regression to estimate the background (offset and slope) from the
    given data by selecting a portion of the data as specified by the `points_cut` parameter. The linear
    fit is applied to either the first or last `points_cut` fraction of the data, depending on the `cut_from_back`
    flag. The estimated background is returned as the coefficients of the linear fit.

    Parameters
    ----------
    x : np.ndarray
        The independent variable data.
    data : np.ndarray
        The dependent variable data, which can be 1D or 2D (e.g., multiple measurements or data points).
    points_cut : float, optional
        The fraction of the data to be considered for the linear fit. Default is 0.1 (10% of the data).
    cut_from_back : bool, optional
        Whether to use the last `points_cut` fraction of the data (True) or the first fraction (False).
        Default is False.

    Returns
    -------
    list
        The coefficients of the linear fit: a list with two elements, where the first is the offset (intercept)
        and the second is the slope.

    Notes
    -----
    - If `data` is 2D, the fit is performed on each column of the data separately.
    - The function assumes that `x` and `data` have compatible shapes.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> data = 3 * x + 2 + np.random.normal(0, 1, size=(100,))
    >>> coefficients = estimate_linear_background(x, data, points_cut=0.2)
    >>> print("Estimated coefficients:", coefficients)
    """  # noqa: E501
    is1D = len(data.shape) == 1
    points = data.shape[0] if is1D else data.shape[1]
    cut = int(points * points_cut)

    # Consider just the cut points
    if not cut_from_back:
        x_data = x[0:cut] if is1D else x[:, 0:cut]
        y_data = data[0:cut] if is1D else data[:, 0:cut]
    else:
        x_data = x[-cut:] if is1D else x[:, -cut:]
        y_data = data[-cut:] if is1D else data[:, -cut:]

    ones_column = np.ones_like(x_data[0, :]) if not is1D else np.ones_like(x_data)
    X = np.vstack([ones_column, x_data[0, :] if not is1D else x_data]).T
    # Linear fit
    coefficients, residuals, _, _ = np.linalg.lstsq(
        X, y_data if is1D else y_data.T, rcond=None
    )

    return coefficients.T


def remove_linear_background(
    x: np.ndarray, data: np.ndarray, points_cut=0.1
) -> np.ndarray:
    """Removes a linear background from the input data (e.g. the phase background
    of a spectroscopy).


    Parameters
    ----------
    data : np.ndarray
        Input data. Can be a 1D vector or a 2D matrix.

    Returns
    -------
    np.ndarray
        The input data with the linear background removed. The shape of the
        returned array matches the input `data`.
    """
    coefficients = estimate_linear_background(x, data, points_cut)

    # Remove background over the whole array
    is1D = len(data.shape) == 1
    ones_column = np.ones_like(x[0, :]) if not is1D else np.ones_like(x)
    X = np.vstack([ones_column, x[0, :] if not is1D else x]).T
    return data - (X @ coefficients.T).T


def linear_interpolation(
    x: float | np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> float | np.ndarray:
    """
    Performs linear interpolation to estimate the value of y at a given x.

    This function computes the interpolated y-value for a given x using two known
    points (x1, y1) and (x2, y2) on a straight line. It supports both scalar and array
    inputs for x, enabling vectorized operations.

    Parameters
    ----------
    x : float or np.ndarray
        The x-coordinate(s) at which to interpolate.
    x1 : float
        The x-coordinate of the first known point.
    y1 : float
        The y-coordinate of the first known point.
    x2 : float
        The x-coordinate of the second known point.
    y2 : float
        The y-coordinate of the second known point.

    Returns
    -------
    float or np.ndarray
        The interpolated y-value(s) at x.

    Notes
    -----
    - If x1 and x2 are the same, the function returns y1 to prevent division by zero.
    - Assumes that x lies between x1 and x2 for meaningful interpolation.

    Examples
    --------
    >>> linear_interpolation(3, 2, 4, 6, 8)
    5.0
    >>> x_vals = np.array([3, 4, 5])
    >>> linear_interpolation(x_vals, 2, 4, 6, 8)
    array([5., 6., 7.])
    """
    if x1 == x2:
        return y1
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def line_between_2_points(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """
    Computes the equation of a line passing through two points.

    Given two points (x1, y1) and (x2, y2), this function returns the y-intercept and
    slope of the line connecting them. If x1 and x2 are the same, the function returns
    y1 as the intercept and a slope of 0 to avoid division by zero.

    Parameters
    ----------
    x1 : float
        The x-coordinate of the first point.
    y1 : float
        The y-coordinate of the first point.
    x2 : float
        The x-coordinate of the second point.
    y2 : float
        The y-coordinate of the second point.

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - The y-intercept (float), which is y1.
        - The slope (float) of the line passing through the points.

    Notes
    -----
    - If x1 and x2 are the same, the function assumes a vertical line and returns a
    slope of 0.
    - The returned y-intercept is based on y1 for consistency in edge cases.

    Examples
    --------
    >>> line_between_2_points(1, 2, 3, 4)
    (2, 1.0)
    >>> line_between_2_points(2, 5, 2, 10)
    (5, 0)
    """
    if x1 == x2:
        return np.inf, y1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def soft_normalize(data: np.ndarray) -> np.ndarray:
    """
    Apply soft normalization to a 1D or 2D array with optional NaNs.

    This function performs z-score normalization followed by a smooth
    non-linear compression using a hyperbolic tangent (tanh) function.
    It is designed to reduce the effect of outliers while preserving
    the dynamic range of typical data values. The result is rescaled to [0, 1].

    For 2D arrays, normalization is done row-wise, but compression is
    based on a global threshold across all non-NaN entries.

    Parameters
    ----------
    data : np.ndarray
        Input data, must be a 1D or 2D NumPy array. Can contain NaNs.

    Returns
    -------
    np.ndarray
        Normalized data, same shape as input, with values scaled to [0, 1].
        NaNs are preserved.

    Raises
    ------
    ValueError
        If `data` is not 1D or 2D.

    Notes
    -----
    - Z-score normalization is done using nanmean and nanstd.
    - Outliers are compressed using a tanh centered at a scaled threshold.
    - Output values are guaranteed to be in [0, 1] range, except NaNs.
    - Rows with zero standard deviation are flattened to 0.5.
    """

    if data.ndim not in [1, 2]:
        raise ValueError("Input must be 1D or 2D")

    data = np.array(data, dtype=np.float64)
    nan_mask = np.isnan(data)

    if data.ndim == 1:
        mean = np.nanmean(data)
        std = np.nanstd(data)
        std = 1.0 if std == 0 else std
        abs_z = np.abs((data - mean) / std)
    else:
        mean = np.nanmean(data, axis=1, keepdims=True)
        std = np.nanstd(data, axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        abs_z = np.abs((data - mean) / std)

    # Flatten over all values for global thresholding
    flat_abs_z = abs_z[~nan_mask]
    if flat_abs_z.size == 0:
        return np.full_like(data, 0.5)

    threshold = 4.0 * np.mean(flat_abs_z)
    alpha = 1.0 / (4.0 * np.std(flat_abs_z)) if np.std(flat_abs_z) != 0 else 1.0

    compressed = np.tanh(alpha * (abs_z - threshold))

    # Rescale to [0, 1]
    compressed[nan_mask] = np.nan
    min_val = np.nanmin(compressed)
    max_val = np.nanmax(compressed)
    if max_val == min_val:
        rescaled = np.full_like(compressed, 0.5)
    else:
        rescaled = (compressed - min_val) / (max_val - min_val)

    rescaled[nan_mask] = np.nan
    return rescaled


def find_closest_index(arr, target):
    """
    Find the index of the element in `arr` closest to the `target` value.
    """

    return np.abs(arr - target).argmin()


def compute_snr_peaked(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x0: float,
    fwhm: float,
    noise_region_factor: float = 2.5,
    min_points: int = 20,
) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) for a peaked function (e.g., Lorentzian, Gaussian)
    based on the provided fit parameters. The SNR is calculated by comparing the signal strength
    at the peak (x0) with the noise level estimated from a region outside the peak.

    Parameters
    ----------
    x_data : np.ndarray
        Array of x values (independent variable), typically representing frequency or position.

    y_data : np.ndarray
        Array of y values (dependent variable), representing the measured values (e.g., intensity, amplitude).

    x0 : float
        The location of the peak (center of the distribution), often the resonance frequency or peak position.

    fwhm : float
        The Full Width at Half Maximum (FWHM) of the peak. This defines the width of the peak and helps determine
        the region for noise estimation.

    noise_region_factor : float, optional, default=2.5
        The factor used to define the width of the noise region as a multiple of the FWHM. The noise region is
        considered outside the interval `(x0 - noise_region_factor * fwhm, x0 + noise_region_factor * fwhm)`.

    min_points : int, optional, default=20
        The minimum number of data points required in the noise region to estimate the noise level. If the number
        of points in the noise region is smaller than this threshold, a warning is issued.

    Returns
    -------
    float
        The computed Signal-to-Noise Ratio (SNR), which is the ratio of the signal strength at `x0` to the
        standard deviation of the noise. If the noise standard deviation is zero, the SNR is set to infinity.

    Notes
    -----
    - The function assumes that the signal has a clear peak at `x0` and that the surrounding data represents noise.
    - If the noise region contains fewer than `min_points` data points, a warning is raised suggesting the adjustment of `noise_region_factor`.

    Example
    -------
    >>> x_data = np.linspace(-10, 10, 1000)
    >>> y_data = np.exp(-(x_data**2))  # Example Gaussian
    >>> x0 = 0
    >>> fwhm = 2.0
    >>> snr = compute_snr_peaked(x_data, y_data, x0, fwhm)
    >>> print(snr)
    """  # noqa: E501

    # Signal strength at x0
    signal = y_data[np.argmin(np.abs(x_data - x0))]

    # Define noise region (outside noise_region_factor * FWHM)
    noise_mask = (x_data < (x0 - noise_region_factor * fwhm)) | (
        x_data > (x0 + noise_region_factor * fwhm)
    )
    noise_data = y_data[noise_mask]

    # Check if there are enough data points for noise estimation
    if len(noise_data) < min_points:
        Warning(
            f"Only {len(noise_data)} points found in the noise region. "
            "Consider reducing noise_region_factor."
        )

    # Compute noise standard deviation
    noise_std = np.std(noise_data)

    # Compute SNR
    snr = signal / noise_std if noise_std > 0 else np.inf  # Avoid division by zero

    return snr


def find_first_minima_idx(data):
    """
    Find the index of the first local minimum in a 1D array.

    Parameters
    ----------
    data : array-like
        1D sequence of numerical values.

    Returns
    -------
    int or None
        Index of the first local minimum, or None if no local minimum is found.

    Notes
    -----
    A local minimum is defined as a point that is smaller than its immediate neighbors.
    Uses `scipy.signal.argrelextrema` to detect local minima.

    Examples
    --------
    >>> data = [3, 2, 4, 1, 5]
    >>> find_first_minima_idx(data)
    1
    """
    data = np.array(data)
    minima_indices = argrelextrema(data, np.less)[0]

    # Check boundaries for minima (optional)
    if data.size < 2:
        return None

    if len(minima_indices) > 0:
        return minima_indices[0]

    return None


def compute_fft(x_data, y_data):
    """
    Computes the Fast Fourier Transform (FFT) of a signal and returns the positive
    frequency spectrum.

    Parameters
    ----------
    x_data : np.ndarray
        Time or independent variable array, assumed to be uniformly spaced.
    y_data : np.ndarray
        Signal data corresponding to `x_data`. Can be real or complex.

    Returns
    -------
    positive_freqs : np.ndarray
        Array of positive frequency components corresponding to the FFT.
    fft_magnitude : np.ndarray
        Magnitude of the FFT at the positive frequencies.

    Notes
    -----
    - The signal is centered by subtracting its mean before computing the FFT, which
    removes the DC component.
    - Only the positive frequency half of the FFT spectrum is returned, assuming
    a real-valued input signal.
    - If `y_data` is complex, returned FFT values still reflect magnitude only.
    - The input `x_data` must be uniformly spaced for the frequency axis to be accurate.

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> y = np.sin(2 * np.pi * 50 * t)
    >>> freqs, spectrum = compute_fft(t, y)
    """

    # Subtract DC offset to focus on oscillations
    y_data_centered = y_data - np.mean(y_data)

    # Calculate time step (assumes uniform spacing)
    dt = x_data[1] - x_data[0]
    N = len(x_data)

    # Compute FFT and frequency axis
    fft_vals = np.fft.fft(y_data_centered)
    freqs = np.fft.fftfreq(N, dt)

    # Take only positive frequencies
    positive_freqs = freqs[: N // 2]
    fft_magnitude = np.abs(fft_vals[: N // 2])

    return positive_freqs, fft_magnitude


def get_peaks(x_data, y_data, prominence: float | None = None, sort=True):
    """
    Detects and returns peaks in a 1D signal based on prominence.

    Parameters
    ----------
    x_data : np.ndarray
        1D array of x-values corresponding to `y_data` (e.g., frequency or time axis).
    y_data : np.ndarray
        1D array of y-values representing the signal in which to find peaks.
    prominence : float or None, optional
        Minimum prominence of peaks to detect. If None, defaults to 5% of the maximum
        value in `y_data`.
    sort : bool, optional
        If True, peaks are sorted in descending order of magnitude. Default is True.

    Returns
    -------
    peak_freqs : np.ndarray
        x-values at which peaks occur.
    peak_magnitudes : np.ndarray
        y-values (magnitudes) at the detected peaks.

    Notes
    -----
    - Uses `scipy.signal.find_peaks` for detection.

    Examples
    --------
    >>> x = np.linspace(0, 10, 1000)
    >>> y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(1000)
    >>> freqs, mags = get_peaks(x, np.abs(y))
    >>> print(freqs[:3], mags[:3])  # Show top 3 peak locations and magnitudes
    """

    if prominence is None:
        prominence = 0.05 * np.max(y_data)

    # Find peaks
    peaks, properties = find_peaks(y_data, prominence=prominence)

    # Get the corresponding frequencies and magnitudes
    peak_freqs = x_data[peaks]
    peak_magnitudes = y_data[peaks]

    if sort:
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        peak_freqs = peak_freqs[sorted_indices]
        peak_magnitudes = peak_magnitudes[sorted_indices]

    return peak_freqs, peak_magnitudes


def amplitude_to_power_dBm(amplitude, offset_dBm=10):
    """Converts amplitude to power and adds an offset (SHFQC range)

    Parameters
    ----------
    amplitude : float.
        Amplitude
    offset_dBm : float (optional)
        Power offset added to the amplitude (SHFQC range). By default 10.

    Returns
    -------
    float
        Power [dBm]
    """
    return offset_dBm + 20 * np.log10(amplitude)


def mask_outliers(data: np.ndarray | list, threshold: float = 3.5) -> np.ndarray:
    """
    Detect outliers by comparing each value to the median of the data and measuring
    its deviation using the Median Absolute Deviation (MAD), which is minimally
    affectedy by extreme values.

    For each data point x:
        modified_z = 0.6745 * |x - median| / MAD

    Any point with modified_z >= threshold is replaced with np.nan.

    Parameters
    ----------
    data : array-like
        Input data, 1D.
    threshold : float
        Modified Z-score threshold for detecting outliers (default 3.5).

    Returns
    -------
    np.ndarray
        Array where outliers are replaced with np.nan.
    """
    data = np.asarray(data, dtype=float)

    med = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - med))

    if mad == 0:
        return data.copy()

    modified_z = 0.6745 * (data - med) / mad
    modified_z = np.abs(modified_z)

    return np.where(modified_z < threshold, data, np.nan)

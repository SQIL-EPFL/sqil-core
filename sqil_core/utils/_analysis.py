import numpy as np


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
    """
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

    This function computes the interpolated y-value for a given x using two known points (x1, y1) and (x2, y2)
    on a straight line. It supports both scalar and array inputs for x, enabling vectorized operations.

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

    Given two points (x1, y1) and (x2, y2), this function returns the y-intercept and slope of the line
    connecting them. If x1 and x2 are the same, the function returns y1 as the intercept and a slope of 0
    to avoid division by zero.

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
    - If x1 and x2 are the same, the function assumes a vertical line and returns a slope of 0.
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


def compute_snr_peaked(
    x_data, y_data, x0, fwhm, noise_region_factor=2.5, min_points=20
):
    """
    Computes the Signal-to-Noise Ratio (SNR) using the fit parameters of a peaked function,
    e.g. Lorentzian, Gaussian.

    Parameters:
    - x_data: 1D numpy array of x values (e.g., frequency).
    - y_data: 1D numpy array of y values (e.g., signal intensity).
    - x0: Peak position on the x axis.
    - fwhm: Full-width at half-maximum.
    - noise_region_factor: How far from x0 (in multiples of FWHM) to measure noise (default: 2.5).
    - min_points: Minimum required data points in the noise region (default: 20).

    Returns:
    - snr: The computed signal-to-noise ratio.
    """

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
            f"Only {len(noise_data)} points found in the noise region. Consider reducing noise_region_factor."
        )

    # Compute noise standard deviation
    noise_std = np.std(noise_data)

    # Compute SNR
    snr = signal / noise_std if noise_std > 0 else np.inf  # Avoid division by zero

    return snr

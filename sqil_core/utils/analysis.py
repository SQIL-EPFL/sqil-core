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


def estimate_linear_background(x: np.ndarray, data: np.ndarray, points_cut=0.1) -> list:
    is1D = len(data.shape) == 1
    points = data.shape[0] if is1D else data.shape[1]
    cut = int(points * points_cut)

    # Consider just the cut points
    x_data = x[0:cut] if is1D else x[0:cut, :]
    X = np.vstack([np.ones_like(x_data), x_data]).T
    y_data = data[0:cut] if is1D else data[0:cut, :]

    # Linear fit
    coefficients, residuals, _, _ = np.linalg.lstsq(
        X, y_data if is1D else y_data.T, rcond=None
    )

    return coefficients


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
    X = np.vstack([np.ones_like(x), x]).T
    return data - (X @ coefficients).T

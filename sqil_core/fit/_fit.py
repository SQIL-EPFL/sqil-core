import numpy as np
import scipy.optimize as spopt

import sqil_core.fit._models as _models

from ._core import FitResult, fit_output


@fit_output
def fit_circle_algebraic(x_data: np.ndarray, y_data: np.ndarray) -> FitResult:
    """Fits a circle in the xy plane and returns the radius and the position of the center.

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

    x0 = spopt.fsolve(func, 0.0, fprime=d_func)

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
    # the term *sqrt term corrects for the constraint, because it may be altered due to numerical inaccuracies during calculation
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
    ssr = np.sum(residuals**2)
    sst = np.sum((r_data - np.mean(r_data)) ** 2)
    r2 = 1 - (ssr / sst) if sst > 0 else 0

    # Compute RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    # Return results
    return {"rmse": rmse}


@fit_output
def fit_skewed_lorentzian(x_data: np.ndarray, y_data: np.ndarray):
    """
    Fits a skewed Lorentzian model to the given data using least squares optimization.

    This function performs a two-step fitting process to find the best-fitting parameters for a skewed Lorentzian model.
    The first fitting step provides initial estimates for the parameters, and the second step refines those estimates
    using a full model fit.

    The skewed Lorentzian model is defined as:
        A1 + A2 * (x - fra) + (A3 + A4 * (x - fra)) / (1.0 + 4.0 * Q_tot^2 * ((x - fra) / fra) ^ 2)

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
    """
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
    p_final, _ = spopt.leastsq(residuals, p0, args=(np.array(x_data), np.array(y_data)))
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
    popt, pcov, infodict, errmsg, ier = spopt.leastsq(
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

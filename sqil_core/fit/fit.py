import warnings

import numpy as np
import scipy.optimize as spopt


def fit_circle_algebraic(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
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
    tuple[float, float, float]
        A tuple containing the fitted circle parameters:
        - xc (float): x-coordinate of the circle's center.
        - yc (float): y-coordinate of the circle's center.
        - r0 (float): Radius of the circle.

    Examples
    --------
    >>> xc, yc, r0 = fit_circle_algebraic(x_data, y_data)
    """
    z_data = x + 1j * y

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

    return xc, yc, r0


def fit_skewed_lorentzian(x: np.ndarray, y: np.ndarray):
    """
    Fits a skewed Lorentzian function to the provided data.

    This function performs a two-step least-squares optimization to fit a skewed
    Lorentzian model to the input data. The skewed Lorentzian model is useful for
    modeling asymmetric resonance peaks, often observed in physical and spectroscopic data.

    Parameters
    ----------
    x : np.ndarray
        Array of independent variable data (e.g., frequency).
    y : np.ndarray
        Array of dependent variable data (e.g., amplitude) corresponding to `x`.

    Returns
    -------
    popt : np.ndarray
        Array of optimized parameters for the skewed Lorentzian fit:
        - A1 (float): Baseline offset.
        - A2 (float): Linear slope adjustment.
        - A3 (float): Amplitude of the Lorentzian peak.
        - A4 (float): Skewness factor adjusting asymmetry.
        - fr (float): Resonance frequency (peak position).
        - Q_tot (float): Total (or loaded) quality factor (controls peak sharpness).

    perc_errs : np.ndarray
        Array of percentage errors for each fitted parameter, calculated from
        the standard errors of the fit.

    Notes
    -----
    - The fitting process uses `scipy.optimize.leastsq` for non-linear least squares fitting.
    - The quality factor `Q_tot` is always returned as a positive value.
    - Initial parameter estimates are obtained through a simplified first fit.

    Examples
    --------
    >>> import numpy as np
    >>> x_data = np.linspace(0.9, 1.1, 1000)
    >>> y_data = 1 / (1 + 4 * (1000**2) * ((x_data - 1.0) / 1.0)**2)  # Ideal Lorentzian
    >>> params, perc_errs = fit_skewed_lorentzian(x_data, y_data)
    >>> (A1, A2, A3, A4, fr, Q_tot) = params
    >>> print("Fitted Parameters:", params)
    >>> print("Percentage Errors:", perc_errs)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x_data, y_data)
    >>> # Import skewed_lorenzian_fun from sqil-core to plot the result
    >>> plt.plot(x_data, skewed_lorenzian_fun(x_data, *params))
    >>> plt.show()
    """

    A1a = np.minimum(y[0], y[-1])
    A3a = -np.max(y)
    fra = x[np.argmin(y)]

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
    p_final, _ = spopt.leastsq(residuals, p0, args=(np.array(x), np.array(y)))
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
        residuals2, p0, args=(np.array(x), np.array(y)), full_output=True
    )
    # Since Q_tot is always present as a square it may turn out negative
    popt[-1] = np.abs(popt[-1])

    # Calculate percentage errors
    std_errs, perc_errs = compute_standard_errors(x, popt, pcov, infodict["fvec"])

    return popt, perc_errs


def skewed_lorentzian_fun(
    f: np.ndarray, A1: float, A2: float, A3: float, A4: float, fr: float, Q_tot: float
) -> np.ndarray:
    """
    Computes the skewed Lorentzian function.

    This function models asymmetric resonance peaks using a skewed Lorentzian
    function, which is commonly used in spectroscopy and resonator analysis to account
    for both peak sharpness and asymmetry.

    The function is defined as:
        L(f) = A1 + A2 * (f - fr) + (A3 + A4 * (f - fr)) / [1 + (2 * Q_tot * ((f / fr) - 1))²]

    Parameters
    ----------
    f : np.ndarray
        Array of frequency or independent variable values.
    A1 : float
        Baseline offset of the curve.
    A2 : float
        Linear slope adjustment, accounting for background trends.
    A3 : float
        Amplitude of the Lorentzian peak.
    A4 : float
        Skewness factor that adjusts the asymmetry of the peak.
    fr : float
        Resonance frequency or the peak position.
    Q_tot : float
        Total (or loaded) quality factor controlling the sharpness and width of the resonance peak.

    Returns
    -------
    np.ndarray
        The computed skewed Lorentzian values corresponding to each input `f`.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.linspace(0.9, 1.1, 500)
    >>> y = skewed_lorentzian_fun(f, A1=0.0, A2=0.0, A3=1.0, A4=0.0, fr=1.0, Q_tot=1000)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, y)
    >>> plt.show()
    """
    return (
        A1
        + A2 * (f - fr)
        + (A3 + A4 * (f - fr)) / (1 + (2 * Q_tot * (f / fr - 1)) ** 2)
    )


def compute_standard_errors(
    x: np.ndarray, popt: np.ndarray, pcov: np.ndarray, residuals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the standard errors and percentage errors of fitted parameters from the covariance matrix.

    This function calculates the standard errors for each parameter obtained from a fitting procedure.
    It uses the covariance matrix and residuals to account for data variance and the quality of the fit.
    The covariance matrix is rescaled by the reduced chi-squared value to provide more accurate error
    estimates. Additionally, the percentage errors are computed based on the standard errors relative
    to the fitted parameters.

    Parameters
    ----------
    x : np.ndarray
        The independent variable data used in the fitting process.
    popt : np.ndarray
        The optimized parameters obtained from the curve fitting.
    pcov : np.ndarray
        The covariance matrix returned by the fitting routine (e.g., scipy.optimize.leastsq).
    residuals : np.ndarray
        The residuals between the fitted model and the observed data (i.e., y - y_fit).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - standard_errors: The standard errors corresponding to each fitted parameter in popt.
        - percentage_errors: The percentage errors for each parameter in popt.

    Notes
    -----
    - The standard errors are derived by rescaling the covariance matrix with the reduced chi-squared
      value to account for model accuracy.
    - Degrees of freedom (DoF) are computed as N - p, where N is the number of data points and p is the
      number of fitted parameters.
    - Assumes that the residuals are normally distributed.
    - The percentage errors are calculated as the ratio of standard errors to the absolute value of the
      fitted parameters, multiplied by 100.

    Examples
    --------
    >>> import numpy as np
    >>> popt = [1.0, 0.5]
    >>> pcov = np.array([[0.04, 0.01], [0.01, 0.02]])
    >>> residuals = np.array([0.1, -0.2, 0.05, -0.1])
    >>> x = np.linspace(0, 1, 4)
    >>> std_errs, perc_errs = compute_standard_errors(x, popt, pcov, residuals)
    >>> print("Standard Errors:", std_errs)
    >>> print("Percentage Errors:", perc_errs)
    """
    if pcov is None:
        if np.allclose(residuals, 0, atol=1e-10):
            warnings.warn(
                "Covariance matrix could not be estimated due to an almost perfect fit. "
                "Standard errors are undefined but may not be necessary in this case."
            )
        else:
            warnings.warn(
                "Covariance matrix could not be estimated. This could be due to poor model fit "
                "or numerical instability. Review the data or model configuration."
            )
        return np.full_like(popt, np.nan, dtype=float)

    # Calculate reduced chi-squared
    dof = len(x) - len(popt)  # degrees of freedom (N - p)
    if dof <= 0:
        warnings.warn(
            "Degrees of freedom (dof) is non-positive. This may indicate overfitting or insufficient data."
        )
        chi2_red = np.nan  # Invalid value for chi-squared if dof <= 0
    else:
        chi2_red = np.sum(residuals**2) / dof

    # Rescale the covariance matrix
    if np.isnan(chi2_red):
        pcov_rescaled = np.nan
    else:
        pcov_rescaled = pcov * chi2_red

    # Calculate standard errors for each parameter
    if np.any(np.isnan(pcov_rescaled)):
        standard_errors = np.full_like(popt, np.nan, dtype=float)
    else:
        standard_errors = np.sqrt(np.diag(pcov_rescaled))

    # Calculate percentage errors for each parameter
    if np.any(popt == 0):
        percentage_errors = np.full_like(popt, np.nan, dtype=float)
    else:
        percentage_errors = (standard_errors / np.abs(popt)) * 100

    return standard_errors, percentage_errors


def compute_standard_errors_minimize(x, res, objective, epsilon=1e-8):
    """
    Computes standard and percentage errors from the optimization result.

    Parameters
    ----------
    res : OptimizeResult
        The result object from `scipy.optimize.minimize`.
    objective : callable
        The objective function used in the optimization.
    epsilon : float
        Step size for numerical gradient approximation.

    Returns
    -------
    std_errors : np.ndarray
        Standard errors of the fitted parameters.
    perc_errors : np.ndarray
        Percentage errors of the fitted parameters.
    """
    popt = res.x
    n_params = len(popt)

    # Approximate the Jacobian using finite differences
    jacobian = spopt.approx_fprime(popt, objective, epsilon)

    # Degrees of freedom (N - p)
    dof = len(x) - n_params
    chi2_red = 2 * res.fun / dof  # Approximate reduced chi-squared

    # Approximate covariance matrix
    try:
        cov_matrix = (
            np.linalg.inv(jacobian[:, np.newaxis] @ jacobian[np.newaxis, :]) * chi2_red
        )
        std_errors = np.sqrt(np.diag(cov_matrix))
        perc_errors = (std_errors / np.abs(popt)) * 100
    except np.linalg.LinAlgError:
        std_errors = np.full_like(popt, np.nan)
        perc_errors = np.full_like(popt, np.nan)
        print(
            "Warning: Covariance matrix could not be estimated. Fit may be degenerate or ill-conditioned."
        )

    return std_errors, perc_errors

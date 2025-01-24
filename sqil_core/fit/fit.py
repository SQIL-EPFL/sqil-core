import inspect
import warnings

import lmfit
import numpy as np
import scipy.optimize as spopt

from sqil_core.utils import print_fit_metrics, print_fit_params

from .models import skewed_lorentzian


def fit_input(fit_func):
    """Decorator to handle optional fitting inputs like guess, bounds, and fixed parameters."""

    def wrapper(
        x_data,
        y_data,
        guess=None,
        bounds=None,
        fixed_params=None,
        fixed_bound_factor=1e-6,
        **kwargs,
    ):
        # Inspect function to check if it requires guess and bounds
        func_params = inspect.signature(fit_func).parameters

        # Process guess if the function accepts it
        if "guess" in func_params:
            num_params = len(guess) if guess is not None else 0
            guess = np.full(num_params, None) if guess is None else np.array(guess)
        else:
            guess = None

        # Process bounds if the function accepts it
        if "bounds" in func_params:
            if guess is not None:
                num_params = len(guess)
                bounds = [(None, None)] * num_params if bounds is None else bounds
                processed_bounds = np.array(
                    [(-np.inf, np.inf) if b is None else b for b in bounds]
                )
                lower_bounds, upper_bounds = (
                    processed_bounds[:, 0],
                    processed_bounds[:, 1],
                )

                # Fix parameters by setting a very tight bound
                if fixed_params is not None:
                    for idx in fixed_params:
                        tolerance = (
                            abs(guess[idx]) * fixed_bound_factor
                            if guess[idx] != 0
                            else fixed_bound_factor
                        )
                        lower_bounds[idx] = guess[idx] - tolerance
                        upper_bounds[idx] = guess[idx] + tolerance
            else:
                lower_bounds, upper_bounds = None, None
        else:
            lower_bounds, upper_bounds = None, None

        # Prepare arguments dynamically
        fit_args = {"x_data": x_data, "y_data": y_data, **kwargs}

        if guess is not None and "guess" in func_params:
            fit_args["guess"] = guess
        if bounds is not None and "bounds" in func_params:
            fit_args["bounds"] = (lower_bounds, upper_bounds)

        # Call the wrapped function with preprocessed inputs
        return fit_func(**fit_args)

    return wrapper


def fit_output(fit_func):
    """Decorator to standardize the output of fitting functions."""

    def wrapper(*args, **kwargs):
        # Perform the fit
        fit_result = fit_func(*args, **kwargs)

        x_data, y_data = args[:2]  # Assume x and y data are the first 2 params
        fit_function = None
        param_names = None
        # Get output type and extract parameters
        if isinstance(fit_result, tuple):
            raw_fit_output = fit_result[0]
            fit_function = fit_result[1] if len(fit_result) > 1 else None
            param_names = fit_result[2] if len(fit_result) > 2 else None
        else:
            raw_fit_output = fit_result

        # Standardized processing of different optimizer outputs
        metrics = None
        # Check if the result is a custom dictionary output
        if isinstance(raw_fit_output, dict):
            params = raw_fit_output.get("params", [])
            std_err = raw_fit_output.get("std_err", None)
            fit_function = fit_function or raw_fit_output.get("predict", None)
            param_names = raw_fit_output.get("param_names", None)
            metrics = raw_fit_output.get("metrics", None)

        # Check if the result is a tuple (likely scipy.optimize output)
        elif isinstance(raw_fit_output, tuple):
            # It's a scipy result, assume it's in the form (popt, pcov, ...)
            popt, pcov = raw_fit_output[0], raw_fit_output[1]

            if len(raw_fit_output) > 2:
                infodict = raw_fit_output[2]
                residuals = infodict["fvec"]
            else:
                y_fit = fit_function(x_data)
                residuals = y_data - y_fit

            params = np.array(popt)
            std_err = compute_standard_errors(x_data, popt, pcov, residuals)

        # Check if the result comes from lmfit
        elif isinstance(raw_fit_output, lmfit.model.ModelResult):
            params = np.array(list(raw_fit_output.params.valuesdict().values()))
            std_err = np.array(
                [
                    param.stderr if param.stderr is not None else np.nan
                    for param in raw_fit_output.params.values()
                ]
            )
            if not fit_function:
                # Determine the independent variable name used in the fit
                independent_var = (
                    raw_fit_output.userkws.keys()
                    & raw_fit_output.model.independent_vars
                )
                independent_var = (
                    independent_var.pop()
                    if independent_var
                    else raw_fit_output.model.independent_vars[0]
                )
                fit_function = lambda x: raw_fit_output.eval(**{independent_var: x})
            param_names = param_names or list(raw_fit_output.params.keys())

        else:
            raise TypeError("Unknown fit result type")

        # Compute fit metrics if not provided
        if metrics is None:
            y_fit = fit_function(x_data)
            metrics = compute_fit_metrics(y_data, y_fit)

        # Return params and a standardized FitResult object
        return params, FitResult(
            params, std_err, metrics, fit_function, raw_fit_output, param_names
        )

    return wrapper


class FitResult:
    def __init__(self, params, std_err, metrics, predict, fit_output, param_names=None):
        self.params = params  # Dictionary of fitted parameters
        self.std_err = std_err  # Dictionary of parameter standard errors
        self.metrics = metrics  # Dictionary of fit quality metrics
        self.predict = predict  # Prediction function with optimized parameters
        self.output = fit_output  # Raw optimizer output
        self.param_names = param_names or list(range(len(params)))

    def __repr__(self):
        return (
            f"FitResult(\n"
            f"  params={self.params},\n"
            f"  std_err={self.std_err},\n"
            f"  metrics={self.metrics}\n)"
        )

    def summary(self):
        """Prints a detailed summary of the fit results."""
        print_fit_metrics(self.metrics)
        print_fit_params(
            self.param_names,
            self.params,
            self.std_err,
            self.std_err / self.params * 100,
        )


def compute_fit_metrics(y_true, y_fit):
    y_range = np.max(y_true) - np.min(y_true)
    residuals = y_true - y_fit

    # R²
    # Quantifies how much of the variance in the data is explained by the fitted model
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Root mean squared error (RMSE)
    # Measures the average deviation between the data points and the fit
    # Penalizes large errors
    rmse = np.sqrt(np.mean(residuals**2))
    # Normalized
    nrmse = rmse / y_range if y_range != 0 else 0

    # Chi² (assuming residual standard deviation as uncertainty)
    chi2 = np.sum((residuals / np.std(y_true)) ** 2) if np.std(y_true) > 0 else 0

    return {"r2": r2, "rmse": rmse, "nrmse": nrmse, "chi2": chi2}


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
        The residuals between the fitted model and the observed data (i.e., y_data - y_fit).

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

    return standard_errors


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
    if dof <= 0:
        warnings.warn(
            "Degrees of freedom (dof) is non-positive. This may indicate overfitting or insufficient data."
        )
        chi2_red = np.nan
    else:
        chi2_red = 2 * res.fun / dof  # Approximate reduced chi-squared

    # Approximate covariance matrix
    try:
        cov_matrix = (
            np.linalg.inv(jacobian[:, np.newaxis] @ jacobian[np.newaxis, :]) * chi2_red
        )
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        std_errors = np.full_like(popt, np.nan)
        print(
            "Warning: Covariance matrix could not be estimated. Fit may be degenerate or ill-conditioned."
        )

    return std_errors


@fit_output
def fit_circle_algebraic(
    x_data: np.ndarray, y_data: np.ndarray
) -> tuple[float, float, float]:
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
        (popt, pcov, infodict),
        lambda x: skewed_lorentzian(x, *popt),
        ["A1", "A2", "A3", "A4", "fr", "Q_tot"],
    )

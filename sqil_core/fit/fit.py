import inspect
import warnings

import lmfit
import numpy as np
import scipy.optimize as spopt
from lmfit.model import ModelResult

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

        # Extract information from function arguments
        x_data, y_data = get_xy_data_from_fit_args(*args, **kwargs)
        sigma = kwargs.get("sigma", None)
        has_sigma = isinstance(sigma, (list, np.ndarray))

        sqil_dict = {
            "params": [],
            "std_err": None,
            "metrics": None,
            "predict": None,
            "output": None,
            "param_names": None,
        }
        metadata = {}
        formatted = None

        # Check if the fit output is a tuple and separate it into raw_fit_ouput and metadata
        if (
            isinstance(fit_result, tuple)
            and (len(fit_result) == 2)
            and isinstance(fit_result[1], dict)
        ):
            raw_fit_output, metadata = fit_result
        else:
            raw_fit_output = fit_result
        sqil_dict["output"] = raw_fit_output

        # Format the raw_fit_output into a standardized dict
        # Scipy tuple (curve_fit, leastsq)
        if is_scipy_tuple(raw_fit_output):
            formatted = format_scipy_tuple(raw_fit_output, has_sigma=has_sigma)

        # Scipy least squares
        elif is_scipy_least_squares(raw_fit_output):
            formatted = format_scipy_least_squares(raw_fit_output, has_sigma=has_sigma)

        # Scipy minimize
        elif is_scipy_minimize(raw_fit_output):
            residuals = None
            predict = metadata.get("predict", None)
            if predict and callable(predict):
                residuals = y_data - metadata["predict"](x_data)
            formatted = format_scipy_minimize(
                raw_fit_output, residuals=residuals, has_sigma=has_sigma
            )

        # lmfit
        elif is_lmfit(raw_fit_output):
            formatted = format_lmfit(raw_fit_output)

        # Custom fit output
        elif isinstance(raw_fit_output, dict):
            formatted = raw_fit_output

        else:
            raise TypeError(
                "Couldn't recognize the output.\n"
                + "Are you using scipy? Did you forget to set `full_output=True` in your fit method?"
            )

        # Update sqil_dict with the formatted fit_output
        if formatted is not None:
            sqil_dict.update(formatted)

        # Add/override fileds using metadata
        sqil_dict.update(metadata)

        # Assign the optimized parameters to the prediction function
        if sqil_dict["predict"] is not None:
            params = sqil_dict["params"]
            predict = sqil_dict["predict"]
            n_inputs = _count_function_parameters(predict)
            if n_inputs == 1 + len(params):
                sqil_dict["predict"] = lambda x: predict(x, *params)

        return FitResult(
            params=sqil_dict.get("params", []),
            std_err=sqil_dict.get("std_err", None),
            fit_output=raw_fit_output,
            metrics=sqil_dict.get("metrics", None),
            predict=sqil_dict.get("predict", None),
            param_names=sqil_dict.get("param_names", None),
        )

    return wrapper


def _count_function_parameters(func):
    sig = inspect.signature(func)
    return len(
        [
            param
            for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        ]
    )


class FitResult:
    def __init__(
        self, params, std_err, fit_output, metrics=None, predict=None, param_names=None
    ):
        self.params = params  # Dictionary of fitted parameters
        self.std_err = std_err  # Dictionary of parameter standard errors
        self.output = fit_output  # Raw optimizer output
        self.metrics = metrics  # Dictionary of fit quality metrics
        self.predict = (
            predict or self._no_prediction
        )  # Fit function with optimized parameters
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

    def _no_prediction(self):
        raise Exception("No predition function available")


def compute_adjusted_standard_errors(
    pcov: np.ndarray,
    residuals: np.ndarray,
    red_chi2=None,
    includes_sigma=True,
    sigma=None,
) -> np.ndarray:
    """`sigma` should only be used in case the optimization doesn't include the experimental error but the experimetal errors are known"""
    # Check for invalid covariance
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
        return None

    # Calculate reduced chi-squared
    n_params = len(np.diag(pcov))
    if red_chi2 is None:
        _, red_chi2 = compute_chi2(
            residuals, n_params, includes_sigma=includes_sigma, sigma=sigma
        )

    # Rescale the covariance matrix
    if np.isnan(red_chi2):
        pcov_rescaled = np.nan
    else:
        pcov_rescaled = pcov * red_chi2

    # Calculate standard errors for each parameter
    if np.any(np.isnan(pcov_rescaled)):
        standard_errors = np.full(n_params, np.nan, dtype=float)
    else:
        standard_errors = np.sqrt(np.diag(pcov_rescaled))

    return standard_errors


def compute_chi2(
    residuals, n_params=None, includes_sigma=True, sigma: np.ndarray = None
):
    """`sigma` should only be used in case the optimization doesn't include the experimental error but the experimetal errors are known"""
    # If the optimization does not account for th experimental sigma,
    # approximate it with the std of the residuals
    S = 1 if includes_sigma else np.std(residuals)
    # If the experimental error is provided, use that instead
    if sigma is not None:
        S = sigma

    # Replace 0 elements of S with the machine epsilon to avoid divisions by 0
    if not np.isscalar(S):
        S_safe = np.where(S == 0, np.finfo(float).eps, S)
    else:
        S_safe = np.finfo(float).eps if S == 0 else S

    # Compute chi squared
    chi2 = np.sum((residuals / S_safe) ** 2)
    # If number of parameters is not provided return just chi2
    if n_params is None:
        return chi2

    # Reduced chi squared
    dof = len(residuals) - n_params  # degrees of freedom (N - p)
    if dof <= 0:
        warnings.warn(
            "Degrees of freedom (dof) is non-positive. This may indicate overfitting or insufficient data."
        )
        red_chi2 = np.nan
    else:
        red_chi2 = np.sum(residuals**2) / dof

    return chi2, red_chi2


def compute_fit_metrics(residuals, y_data, sigma: np.ndarray = None):
    y_range = np.max(y_data) - np.min(y_data)

    # R²
    # Quantifies how much of the variance in the data is explained by the fitted model
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Root mean squared error (RMSE)
    # Measures the average deviation between the data points and the fit
    # Penalizes large errors
    rmse = np.sqrt(np.mean(residuals**2))
    # Normalized
    nrmse = rmse / y_range if y_range != 0 else 0

    # Chi² (assuming residual standard deviation as uncertainty)
    sigma = sigma or np.std(residuals)
    chi2 = np.sum((residuals / sigma) ** 2) if sigma > 0 else 0

    return {"r2": r2, "rmse": rmse, "nrmse": nrmse, "chi2": chi2}


def get_xy_data_from_fit_args(*args, **kwargs):
    # Possible keyword names for x and y data
    x_keys = ["x_data", "xdata", "x"]
    y_keys = ["y_data", "ydata", "y"]

    # Validate if an object is a 1D vector
    def is_valid_vector(obj):
        return isinstance(obj, (list, np.ndarray)) and np.ndim(obj) == 1

    x_data, y_data = None, None

    # Look for x_data in keyword arguments
    for key in x_keys:
        if key in kwargs and is_valid_vector(kwargs[key]):
            x_data = kwargs[key]
            break
    # Look for y_data in keyword arguments
    for key in y_keys:
        if key in kwargs and is_valid_vector(kwargs[key]):
            y_data = kwargs[key]
            break

    # If both parameters were found, return them
    if (x_data is not None) and (y_data is not None):
        return x_data, y_data

    # If the args have only 1 entry
    if len(args) == 1 and is_valid_vector(args[0]):
        if y_data is not None:
            x_data = args[0]
        else:
            y_data = args[0]

    # If x and y were not found, try finding the first two consecutive vectors in args
    if x_data is None or y_data is None:
        # Check pairs of consecutive elements
        for i in range(len(args) - 1):
            if is_valid_vector(args[i]) and is_valid_vector(args[i + 1]):
                x_data, y_data = args[i], args[i + 1]
                break

    return x_data, y_data


def format_scipy_tuple(result, has_sigma=False):
    if not isinstance(result, tuple):
        raise TypeError("Fit result must be a tuple")

    std_err = None
    popt, pcov, infodict = None, None, None

    # Extract output parameters
    length = len(result)
    popt = result[0]
    pcov = result[1] if length > 1 else None
    infodict = result[2] if length > 2 else None

    if infodict is not None:
        residuals = infodict["fvec"]
        _, red_chi2 = compute_chi2(
            residuals, n_params=len(popt), includes_sigma=has_sigma
        )
        if pcov is not None:
            std_err = compute_adjusted_standard_errors(
                pcov, residuals, includes_sigma=has_sigma, red_chi2=red_chi2
            )

    return {"params": popt, "std_err": std_err, "metrics": {"red_chi2": red_chi2}}


def format_scipy_least_squares(result, has_sigma=False):
    params = result.x
    residuals = result.fun
    cov = np.linalg.inv(result.jac.T @ result.jac)
    _, red_chi2 = compute_chi2(
        residuals, n_params=len(params), includes_sigma=has_sigma
    )
    std_err = compute_adjusted_standard_errors(
        cov, residuals, includes_sigma=has_sigma, red_chi2=red_chi2
    )

    return {"params": params, "std_err": std_err, "metrics": {"red_chi2": red_chi2}}


def format_scipy_minimize(result, residuals=None, has_sigma=False):
    """Precise estimation of std_err requires a function to compute residuals"""
    params = result.x
    cov = get_covariance_from_scipy_optimize_result(result)
    metrics = None

    if residuals is None:
        std_err = np.sqrt(np.abs(result.hess_inv.diagonal()))
    else:
        std_err = compute_adjusted_standard_errors(
            cov, residuals, includes_sigma=has_sigma
        )
        _, red_chi2 = compute_chi2(
            residuals, n_params=len(params), includes_sigma=has_sigma
        )
        metrics = {"red_chi2": red_chi2}

    return {"params": params, "std_err": std_err, "metrics": metrics}


def format_lmfit(result: ModelResult):
    """lmfit std errors are already rescaled by the reduced chi"""
    params = np.array([param.value for param in result.params.values()])
    param_names = list(result.params.keys())
    std_err = np.array(
        [
            param.stderr if param.stderr is not None else np.nan
            for param in result.params.values()
        ]
    )
    # Determine the independent variable name used in the fit
    independent_var = result.userkws.keys() & result.model.independent_vars
    independent_var = (
        independent_var.pop() if independent_var else result.model.independent_vars[0]
    )
    fit_function = lambda x: result.eval(**{independent_var: x})

    return {
        "params": params,
        "std_err": std_err,
        "metrics": {"red_chi2": result.redchi},
        "predict": fit_function,
        "param_names": param_names,
    }


def get_covariance_from_scipy_optimize_result(
    result: spopt.OptimizeResult,
) -> np.ndarray:
    if hasattr(result, "hess_inv"):
        hess_inv = result.hess_inv

        # Handle different types of hess_inv
        if isinstance(hess_inv, np.ndarray):
            return hess_inv
        elif hasattr(hess_inv, "todense"):
            return hess_inv.todense()

    if hasattr(result, "hess") and result.hess is not None:
        try:
            return np.linalg.inv(result.hess)
        except np.linalg.LinAlgError:
            pass  # Hessian is singular, cannot compute covariance

    return None


def is_scipy_tuple(result):
    if isinstance(result, tuple):
        if len(result) < 3:
            raise TypeError(
                "Fit result is a tuple, but couldn't recognize it.\n"
                + "Are you using scipy? Did you forget to set `full_output=True` in your fit method?"
            )
        popt = result[0]
        cov_ish = result[1]
        infodict = result[2]
        keys_to_check = ["fvec"]

        if cov_ish is not None:
            cov_check = isinstance(cov_ish, np.ndarray) and cov_ish.ndim == 2
        else:
            cov_check = True
        return (
            isinstance(popt, np.ndarray)
            and cov_check
            and (all(key in infodict for key in keys_to_check))
        )
    return False


def is_scipy_minimize(result):
    return (
        isinstance(result, spopt.OptimizeResult)
        and hasattr(result, "fun")
        and np.isscalar(result.fun)
        and hasattr(result, "jac")
    )


def is_scipy_least_squares(result):
    return (
        isinstance(result, spopt.OptimizeResult)
        and hasattr(result, "cost")
        and hasattr(result, "fun")
        and hasattr(result, "jac")
    )


def is_lmfit(result):
    return isinstance(result, ModelResult)


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

import inspect

import numpy as np
import scipy.optimize as spopt
from lmfit.model import ModelResult

from sqil_core.utils import print_fit_metrics as _print_fit_metrics
from sqil_core.utils import print_fit_params as _print_fit_params
from sqil_core.utils._utils import _count_function_parameters


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
        _print_fit_metrics(self.metrics)
        _print_fit_params(
            self.param_names,
            self.params,
            self.std_err,
            self.std_err / self.params * 100,
        )

    def _no_prediction(self):
        raise Exception("No predition function available")


def fit_output(fit_func):
    """Decorator to standardize the output of fitting functions."""

    def wrapper(*args, **kwargs):
        # Perform the fit
        fit_result = fit_func(*args, **kwargs)

        # Extract information from function arguments
        x_data, y_data = _get_xy_data_from_fit_args(*args, **kwargs)
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
        if _is_scipy_tuple(raw_fit_output):
            formatted = _format_scipy_tuple(raw_fit_output, has_sigma=has_sigma)

        # Scipy least squares
        elif _is_scipy_least_squares(raw_fit_output):
            formatted = _format_scipy_least_squares(raw_fit_output, has_sigma=has_sigma)

        # Scipy minimize
        elif _is_scipy_minimize(raw_fit_output):
            residuals = None
            predict = metadata.get("predict", None)
            if predict and callable(predict):
                residuals = y_data - metadata["predict"](x_data)
            formatted = _format_scipy_minimize(
                raw_fit_output, residuals=residuals, has_sigma=has_sigma
            )

        # lmfit
        elif _is_lmfit(raw_fit_output):
            formatted = _format_lmfit(raw_fit_output)

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


def _is_scipy_tuple(result):
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


def _is_scipy_minimize(result):
    return (
        isinstance(result, spopt.OptimizeResult)
        and hasattr(result, "fun")
        and np.isscalar(result.fun)
        and hasattr(result, "jac")
    )


def _is_scipy_least_squares(result):
    return (
        isinstance(result, spopt.OptimizeResult)
        and hasattr(result, "cost")
        and hasattr(result, "fun")
        and hasattr(result, "jac")
    )


def _is_lmfit(result):
    return isinstance(result, ModelResult)


def _format_scipy_tuple(result, has_sigma=False):
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


def _format_scipy_least_squares(result, has_sigma=False):
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


def _format_scipy_minimize(result, residuals=None, has_sigma=False):
    """Precise estimation of std_err requires a function to compute residuals"""
    params = result.x
    cov = _get_covariance_from_scipy_optimize_result(result)
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


def _format_lmfit(result: ModelResult):
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


def _get_xy_data_from_fit_args(*args, **kwargs):
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


def _get_covariance_from_scipy_optimize_result(
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


# TODO: rethink approach
def _fit_input(fit_func):
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

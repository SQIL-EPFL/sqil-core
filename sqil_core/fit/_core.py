import inspect

import numpy as np
import scipy.optimize as spopt
from lmfit.model import ModelResult

from sqil_core.utils import print_fit_metrics as _print_fit_metrics
from sqil_core.utils import print_fit_params as _print_fit_params
from sqil_core.utils._utils import _count_function_parameters


class FitResult:
    """
    Stores the result of a fitting procedure.

    This class encapsulates the fitted parameters, their standard errors, optimizer output,
    and fit quality metrics. It also provides functionality for summarizing the results and
    making predictions using the fitted model.

    Parameters
    ----------
    params : dict
        Array of fitted parameters.
    std_err : dict
        Array of standard errors of the fitted parameters.
    fit_output : any
        Raw output from the optimization routine.
    metrics : dict, optional
        Dictionary of fit quality metrics (e.g., R-squared, reduced chi-squared).
    predict : callable, optional
        Function of x that returns predictions based on the fitted parameters.
        If not provided, an exception will be raised when calling it.
    param_names : list, optional
        List of parameter names, defaulting to a range based on the number of parameters.

    Methods
    -------
    summary()
        Prints a detailed summary of the fit results, including parameter values,
        standard errors, and fit quality metrics.
    _no_prediction()
        Raises an exception when no prediction function is available.
    """

    def __init__(
        self, params, std_err, fit_output, metrics=None, predict=None, param_names=None
    ):
        self.params = params
        self.std_err = std_err
        self.output = fit_output
        self.metrics = metrics
        self.predict = predict or self._no_prediction
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
    """
    Decorator to standardize the output of fitting functions.

    This decorator processes the raw output of various fitting libraries
    (such as SciPy's curve_fit, least_squares leastsq, and minimize, as well as lmfit)
    and converts it into a unified `FitResult` object. It extracts
    optimized parameters, their standard errors, fit quality metrics,
    and a prediction function.

    Parameters
    ----------
    fit_func : Callable
        A function that performs fitting and returns raw fit output,
        possibly along with metadata.

    Returns
    -------
    Callable
        A wrapped function that returns a `FitResult` object containing:
        - `params` : list
            Optimized parameter values.
        - `std_err` : list or None
            Standard errors of the fitted parameters.
        - `metrics` : dict or None
            Dictionary of fit quality metrics (e.g., reduced chi-squared).
        - `predict` : Callable or None
            A function that predicts values using the optimized parameters.
        - `output` : object
            The raw optimizer output from the fitting process.
        - `param_names` : list or None
            Names of the fitted parameters.

    Raises
    ------
    TypeError
        If the fitting function's output format is not recognized.

    Notes
    -----
    - If the fit function returns a tuple `(raw_output, metadata)`,
      the metadata is extracted and applied to enhance the fit results.
      In case of any conflicts, the metadata overrides the computed values.

    Examples
    --------
    >>> @fit_output
    ... def my_fitting_function(x, y):
    ...     return some_raw_fit_output
    ...
    >>> fit_result = my_fitting_function(x_data, y_data)
    >>> print(fit_result.params)
    """

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
                residuals = y_data - metadata["predict"](x_data, *raw_fit_output.x)
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
    cov_rescaled=True,
    sigma=None,
) -> np.ndarray:
    """
    Compute adjusted standard errors for fitted parameters.

    This function adjusts the covariance matrix based on the reduced chi-squared
    value and calculates the standard errors for each parameter. It accounts for
    cases where the covariance matrix is not available or the fit is nearly perfect.

    Parameters
    ----------
    pcov : np.ndarray
        Covariance matrix of the fitted parameters, typically obtained from an
        optimization routine.
    residuals : np.ndarray
        Residuals of the fit, defined as the difference between observed and
        model-predicted values.
    red_chi2 : float, optional
        Precomputed reduced chi-squared value. If `None`, it is computed from
        `residuals` and `sigma`.
    cov_rescaled : bool, default=True
        Whether the fitting process already rescales the covariance matrix with
        the reduced chi-squared.
    sigma : np.ndarray, optional
        Experimental uncertainties. Only used if `cov_rescaled=False` AND
        known experimental errors are available.

    Returns
    -------
    np.ndarray
        Standard errors for each fitted parameter. If the covariance matrix is
        undefined, returns `None`.

    Warnings
    --------
    - If the covariance matrix is not available (`pcov is None`), the function
      issues a warning about possible numerical instability or a near-perfect fit.
    - If the reduced chi-squared value is `NaN`, the function returns `NaN` for
      all standard errors.

    Notes
    -----
    - The covariance matrix is scaled by the reduced chi-squared value to adjust
      for under- or overestimation of uncertainties.
    - If `red_chi2` is not provided, it is computed internally using the residuals.
    - If a near-perfect fit is detected (all residuals close to zero), the function
      warns that standard errors may not be necessary.

    Examples
    --------
    >>> pcov = np.array([[0.04, 0.01], [0.01, 0.09]])
    >>> residuals = np.array([0.1, -0.2, 0.15])
    >>> compute_adjusted_standard_errors(pcov, residuals)
    array([0.2, 0.3])
    """
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
            residuals, n_params, cov_rescaled=cov_rescaled, sigma=sigma
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


def compute_chi2(residuals, n_params=None, cov_rescaled=True, sigma: np.ndarray = None):
    """
    Compute the chi-squared (χ²) and reduced chi-squared (χ²_red) statistics.

    This function calculates the chi-squared value based on residuals and an
    estimated or provided uncertainty (`sigma`). If the number of model parameters
    (`n_params`) is specified, it also computes the reduced chi-squared.

    Parameters
    ----------
    residuals : np.ndarray
        The difference between observed and model-predicted values.
    n_params : int, optional
        Number of fitted parameters. If provided, the function also computes
        the reduced chi-squared (χ²_red).
    cov_rescaled : bool, default=True
        Whether the covariance matrix has been already rescaled by the fit method.
        If `True`, the function assumes proper uncertainty scaling. Otherwise,
        it estimates uncertainty from the standard deviation of the residuals.
    sigma : np.ndarray, optional
        Experimental uncertainties. Should only be used when the fitting process
        does not account for experimental errors AND known uncertainties are available.

    Returns
    -------
    chi2 : float
        The chi-squared statistic (χ²), which measures the goodness of fit.
    red_chi2 : float (if `n_params` is provided)
        The reduced chi-squared statistic (χ²_red), computed as χ² divided by
        the degrees of freedom (N - p). If `n_params` is `None`, only χ² is returned.

    Warnings
    --------
    - If the degrees of freedom (N - p) is non-positive, a warning is issued,
      and χ²_red is set to NaN. This may indicate overfitting or an insufficient
      number of data points.
    - If any uncertainty value in `sigma` is zero, it is replaced with machine epsilon
      to prevent division by zero.

    Notes
    -----
    - If `sigma` is not provided and `cov_rescaled=False`, the function estimates
      the uncertainty using the standard deviation of residuals.
    - The reduced chi-squared value (χ²_red) should ideally be close to 1 for a good fit.
      Values significantly greater than 1 indicate underfitting, while values much less
      than 1 suggest overfitting.

    Examples
    --------
    >>> residuals = np.array([0.1, -0.2, 0.15, -0.05])
    >>> compute_chi2(residuals, n_params=2)
    (0.085, 0.0425)  # Example output
    """
    # If the optimization does not account for th experimental sigma,
    # approximate it with the std of the residuals
    S = 1 if cov_rescaled else np.std(residuals)
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
    """
    Check whether the given result follows the expected structure of a SciPy optimization tuple.
    """
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
    """
    Check whether the given result follows the expected structure of a SciPy minimize.
    """
    return (
        isinstance(result, spopt.OptimizeResult)
        and hasattr(result, "fun")
        and np.isscalar(result.fun)
        and hasattr(result, "jac")
    )


def _is_scipy_least_squares(result):
    """
    Check whether the given result follows the expected structure of a SciPy least_squares.
    """
    return (
        isinstance(result, spopt.OptimizeResult)
        and hasattr(result, "cost")
        and hasattr(result, "fun")
        and hasattr(result, "jac")
    )


def _is_lmfit(result):
    """
    Check whether the given result follows the expected structure of a lmfit fit.
    """
    return isinstance(result, ModelResult)


def _format_scipy_tuple(result, has_sigma=False):
    """
    Formats the output of a SciPy fitting function into a standardized dictionary.

    This function takes the tuple returned by SciPy optimization functions (e.g., `curve_fit`, `leastsq`)
    and extracts relevant fitting parameters, standard errors, and reduced chi-squared values. It ensures
    the result is structured consistently for further processing.

    Parameters
    ----------
    result : tuple
        A tuple containing the fitting results from a SciPy function. Expected structure:
        - `result[0]`: `popt` (optimized parameters, NumPy array)
        - `result[1]`: `pcov` (covariance matrix, NumPy array or None)
        - `result[2]`: `infodict` (dictionary containing residuals, required for error computation)

    has_sigma : bool, optional
        Indicates whether the fitting procedure considered experimental errors (`sigma`).
        If `True`, the covariance matrix (`pcov`) does not need rescaling.

    Returns
    -------
    dict
        A dictionary containing:
        - `"params"`: The optimized parameters (`popt`).
        - `"std_err"`: The standard errors computed from the covariance matrix (`pcov`).
        - `"metrics"`: A dictionary containing the reduced chi-squared (`red_chi2`).
    """
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
            residuals, n_params=len(popt), cov_rescaled=has_sigma
        )
        if pcov is not None:
            std_err = compute_adjusted_standard_errors(
                pcov, residuals, cov_rescaled=has_sigma, red_chi2=red_chi2
            )

    return {"params": popt, "std_err": std_err, "metrics": {"red_chi2": red_chi2}}


def _format_scipy_least_squares(result, has_sigma=False):
    """
    Formats the output of a SciPy least-squares optimization into a standardized dictionary.

    This function processes the result of a SciPy least-squares fitting function (e.g., `scipy.optimize.least_squares`)
    and structures the fitting parameters, standard errors, and reduced chi-squared values for consistent downstream use.

    Parameters
    ----------
    result : `scipy.optimize.OptimizeResult`
        The result of a least-squares optimization (e.g., from `scipy.optimize.least_squares`).
        It must contain the following fields:
        - `result.x`: Optimized parameters (NumPy array)
        - `result.fun`: Residuals (array of differences between the observed and fitted data)
        - `result.jac`: Jacobian matrix (used to estimate covariance)

    has_sigma : bool, optional
        Indicates whether the fitting procedure considered experimental errors (`sigma`).
        If `True`, the covariance matrix does not need rescaling.

    Returns
    -------
    dict
        A dictionary containing:
        - `"params"`: Optimized parameters (`result.x`).
        - `"std_err"`: Standard errors computed from the covariance matrix and residuals.
        - `"metrics"`: A dictionary containing the reduced chi-squared (`red_chi2`).
    """
    params = result.x
    residuals = result.fun
    cov = np.linalg.inv(result.jac.T @ result.jac)
    _, red_chi2 = compute_chi2(residuals, n_params=len(params), cov_rescaled=has_sigma)
    std_err = compute_adjusted_standard_errors(
        cov, residuals, cov_rescaled=has_sigma, red_chi2=red_chi2
    )

    return {"params": params, "std_err": std_err, "metrics": {"red_chi2": red_chi2}}


def _format_scipy_minimize(result, residuals=None, has_sigma=False):
    """
    Formats the output of a SciPy minimize optimization into a standardized dictionary.

    This function processes the result of a SciPy minimization optimization (e.g., `scipy.optimize.minimize`)
    and structures the fitting parameters, standard errors, and reduced chi-squared values for consistent downstream use.

    Parameters
    ----------
    result : `scipy.optimize.OptimizeResult`
        The result of a minimization optimization (e.g., from `scipy.optimize.minimize`).
        It must contain the following fields:
        - `result.x`: Optimized parameters (NumPy array).
        - `result.hess_inv`: Inverse Hessian matrix used to estimate the covariance.

    residuals : array-like, optional
        The residuals (differences between observed data and fitted model).
        If not provided, standard errors will be computed based on the inverse Hessian matrix.

    has_sigma : bool, optional
        Indicates whether the fitting procedure considered experimental errors (`sigma`).
        If `True`, the covariance matrix does not need rescaling.

    Returns
    -------
    dict
        A dictionary containing:
        - `"params"`: Optimized parameters (`result.x`).
        - `"std_err"`: Standard errors computed either from the Hessian matrix or based on the residuals.
        - `"metrics"`: A dictionary containing the reduced chi-squared (`red_chi2`), if residuals are provided.
    """
    params = result.x
    cov = _get_covariance_from_scipy_optimize_result(result)
    metrics = None

    if residuals is None:
        std_err = np.sqrt(np.abs(result.hess_inv.diagonal()))
    else:
        std_err = compute_adjusted_standard_errors(
            cov, residuals, cov_rescaled=has_sigma
        )
        _, red_chi2 = compute_chi2(
            residuals, n_params=len(params), cov_rescaled=has_sigma
        )
        metrics = {"red_chi2": red_chi2}

    return {"params": params, "std_err": std_err, "metrics": metrics}


def _format_lmfit(result: ModelResult):
    """
    Formats the output of an lmfit model fitting result into a standardized dictionary.

    This function processes the result of an lmfit model fitting (e.g., from `lmfit.Model.fit`) and
    structures the fitting parameters, their standard errors, reduced chi-squared, and a prediction function.

    Parameters
    ----------
    result : `lmfit.ModelResult`
        The result of an lmfit model fitting procedure. It must contain the following fields:
        - `result.params`: A dictionary of fitted parameters and their values.
        - `result.redchi`: The reduced chi-squared value.
        - `result.eval`: A method to evaluate the fitted model using independent variable values.
        - `result.userkws`: Dictionary of user-supplied keywords that includes the independent variable.
        - `result.model.independent_vars`: List of independent variable names in the model.

    Returns
    -------
    dict
        A dictionary containing:
        - `"params"`: Optimized parameters (as a NumPy array).
        - `"std_err"`: Standard errors of the parameters.
        - `"metrics"`: A dictionary containing the reduced chi-squared (`red_chi2`).
        - `"predict"`: A function that predicts the model's output given an input (using optimized parameters).
        - `"param_names"`: List of parameter names.

    Notes
    -----
    - lmfit already rescales standard errors by the reduced chi-squared, so no further adjustments are made.
    - The independent variable name used in the fit is determined from `result.userkws` and `result.model.independent_vars`.
    - The function creates a prediction function (`predict`) from the fitted model.
    """
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


def _get_covariance_from_scipy_optimize_result(
    result: spopt.OptimizeResult,
) -> np.ndarray:
    """
    Extracts the covariance matrix (or an approximation) from a scipy optimization result.

    This function attempts to retrieve the covariance matrix of the fitted parameters from the
    result object returned by a scipy optimization method. It first checks for the presence of
    the inverse Hessian (`hess_inv`), which is used to estimate the covariance. If it's not available,
    the function attempts to compute the covariance using the Hessian matrix (`hess`).

    Parameters
    ----------
    result : `scipy.optimize.OptimizeResult`
        The result object returned by a scipy optimization function, such as `scipy.optimize.minimize` or `scipy.optimize.curve_fit`.
        This object contains the optimization results, including the Hessian or its inverse.

    Returns
    -------
    np.ndarray or None
        The covariance matrix of the optimized parameters, or `None` if it cannot be computed.
        If the inverse Hessian (`hess_inv`) is available, it will be returned directly.
        If the Hessian matrix (`hess`) is available and not singular, its inverse will be computed and returned.
        If neither is available, the function returns `None`.

    Notes
    -----
    - If the Hessian matrix (`hess`) is singular or nearly singular, the covariance matrix cannot be computed.
    - In some cases, the inverse Hessian (`hess_inv`) is directly available and provides the covariance without further computation.
    """
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


def _get_xy_data_from_fit_args(*args, **kwargs):
    """
    Extracts x and y data from the given arguments and keyword arguments.

    This helper function retrieves the x and y data (1D vectors) from the function's arguments or keyword arguments.
    The function checks for common keyword names like "x_data", "xdata", "x", "y_data", "ydata", and "y", and returns
    the corresponding data. If no keyword arguments are found, it attempts to extract the first two consecutive 1D
    vectors from the positional arguments.

    Parameters
    ----------
    *args : variable length argument list
        The positional arguments passed to the function, potentially containing the x and y data.

    **kwargs : keyword arguments
        The keyword arguments passed to the function, potentially containing keys such as "x_data", "x", "y_data", or "y".

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        A tuple containing the x data and y data as 1D numpy arrays or lists. If no valid data is found, returns (None, None).

    Raises
    ------
    ValueError
        If both x and y data cannot be found in the input arguments.

    Notes
    -----
    - The function looks for the x and y data in the keyword arguments first, in the order of x_keys and y_keys.
    - If both x and y data are not found in keyword arguments, the function will look for the first two consecutive
      1D vectors in the positional arguments.
    - If the data cannot be found, the function will return (None, None).
    - The function validates that the extracted x and y data are 1D vectors (either lists or numpy arrays).
    """
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

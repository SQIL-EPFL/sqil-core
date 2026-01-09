import inspect
import warnings
from functools import wraps

import numpy as np
import scipy.optimize as spopt
from lmfit.model import ModelResult

from sqil_core.fit._quality import FitQuality, evaluate_fit_quality, format_fit_metrics
from sqil_core.utils._formatter import format_fit_params
from sqil_core.utils._utils import _count_function_parameters


class FitResult:
    """
    Stores the result of a fitting procedure.

    This class encapsulates the fitted parameters, their standard errors, optimizer
    output, and fit quality metrics. It also provides functionality for summarizing
    the results and making predictions using the fitted model.

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
        List of parameter names, defaulting to a range based on the number of parameters
    model_name : str, optional
        Name of the model used to fit the data.
    metadata : dict, optional
        Additional information that can be passed in the fit result.

    Methods
    -------
    summary()
        Prints a detailed summary of the fit results, including parameter values,
        standard errors, and fit quality metrics.
    _no_prediction()
        Raises an exception when no prediction function is available.
    """

    def __init__(
        self,
        params,
        std_err,
        fit_output,
        metrics=None,
        predict=None,
        param_names=None,
        model_name=None,
        metadata=None,
    ):
        self.params = params
        self.std_err = std_err
        self.output = fit_output
        self.metrics = metrics or {}
        self.predict = predict or self._no_prediction
        self.param_names = param_names or list(range(len(params)))
        self.model_name = model_name
        self.metadata = metadata or {}

        self.params_by_name = dict(zip(self.param_names, self.params, strict=False))

    def __repr__(self):
        return (
            f"FitResult(\n"
            f"  params={self.params},\n"
            f"  std_err={self.std_err},\n"
            f"  metrics={self.metrics}\n)"
        )

    def summary(self, no_print=False):
        """Prints a detailed summary of the fit results."""
        s = format_fit_metrics(self.metrics) + "\n"
        s += format_fit_params(
            self.param_names,
            self.params,
            self.std_err,
            np.array(self.std_err) / self.params * 100,
        )
        if not no_print:
            print(s)
        return s

    def quality(self, recipe="nrmse"):
        return evaluate_fit_quality(self.metrics, recipe)

    def is_acceptable(self, recipe="nrmse", threshold=FitQuality.ACCEPTABLE):
        return self.quality(recipe) >= threshold

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
        - `metadata` : dict
            A dictionary containing extra information. Advanced uses include passing
            functions that get evaluated after fit result has been processed.
            See the documentation, Notebooks/The fit_output decorator

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

    @wraps(fit_func)
    def wrapper(*args, **kwargs):
        # Perform the fit
        fit_result = fit_func(*args, **kwargs)

        # Extract information from function arguments
        x_data, y_data = _get_xy_data_from_fit_args(*args, **kwargs)
        sigma = kwargs.get("sigma", None)
        has_sigma = isinstance(sigma, (list, np.ndarray))

        # Initilize variables
        sqil_keys = ["params", "std_err", "metrics", "predict", "output", "param_names"]
        sqil_dict = {key: None for key in sqil_keys}
        metadata = {}
        formatted = None
        # Set the default parameters to an empty array instead of None
        sqil_dict["params"] = []

        # Check if the fit output is a tuple and separate it into raw_fit_ouput
        # and metadata
        if (
            isinstance(fit_result, tuple)
            and (len(fit_result) == 2)
            and isinstance(fit_result[1], dict)
        ):
            raw_fit_output, metadata = fit_result
        else:
            raw_fit_output = fit_result
        sqil_dict["output"] = raw_fit_output

        # Check if there are variables to override in metadata before continuing
        if "fit_output_vars" in metadata:
            overrides = metadata["fit_output_vars"]
            x_data = overrides.get("x_data", x_data)
            y_data = overrides.get("y_data", y_data)
            del metadata["fit_output_vars"]

        # Format the raw_fit_output into a standardized dict
        if raw_fit_output is None:
            raise TypeError("Fit didn't coverge, result is None")
        # Scipy tuple (curve_fit, leastsq)
        if _is_scipy_tuple(raw_fit_output):
            formatted = _format_scipy_tuple(raw_fit_output, y_data, has_sigma=has_sigma)

        # Scipy least squares
        elif _is_scipy_least_squares(raw_fit_output):
            formatted = _format_scipy_least_squares(
                raw_fit_output, y_data, has_sigma=has_sigma
            )

        # Scipy minimize
        elif _is_scipy_minimize(raw_fit_output):
            residuals = None
            predict = metadata.get("predict", None)
            if (x_data is not None) and (predict is not None) and callable(predict):
                residuals = y_data - metadata["predict"](x_data, *raw_fit_output.x)
            formatted = _format_scipy_minimize(
                raw_fit_output, residuals=residuals, y_data=y_data, has_sigma=has_sigma
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
                + "Are you using scipy? Did you forget to set `full_output=True` "
                "in your fit method?"
            )

        # Update sqil_dict with the formatted fit_output
        if formatted is not None:
            sqil_dict.update(formatted)

        # Add/override fileds using metadata
        sqil_dict.update(metadata)

        # Process metadata
        metadata = _process_metadata(metadata, sqil_dict)
        # Remove fields already present in sqil_dict from metadata
        filtered_metadata = {k: v for k, v in metadata.items() if k not in sqil_keys}

        # Assign the optimized parameters to the prediction function
        model_name = metadata.get("model_name", None)
        if sqil_dict["predict"] is not None:
            if model_name is None:
                model_name = sqil_dict["predict"].__name__
            params = sqil_dict["params"]
            predict = sqil_dict["predict"]
            n_inputs = _count_function_parameters(predict)
            if n_inputs == 1 + len(params):
                sqil_dict["predict"] = lambda x: predict(x, *params)

        return FitResult(
            params=sqil_dict.get("params", []),
            std_err=sqil_dict.get("std_err"),
            fit_output=raw_fit_output,
            metrics=sqil_dict.get("metrics", {}),
            predict=sqil_dict.get("predict"),
            param_names=sqil_dict.get("param_names"),
            model_name=model_name,
            metadata=filtered_metadata,
        )

    return wrapper


# TODO: make a function that handles the bounds for lmfit.
# Such a function will take the bounds as they are returned by @fit_input, i.e lower and
# upper bounds and it will iterated through the lmfit parameters to apply the bounds.
def fit_input(fit_func):
    """
    Decorator to handle optional fitting inputs like initial guesses, bounds, and fixed parameters
    for a fitting function.

    - `guess` : list or np.ndarray, optional, default=None
        The initial guess for the fit. If None it's not passed to the fit function.
    - `bounds` : list or np.ndarray, optional, default=(-np.inf, np.inf)
        The bounds on the fit parameters in the form [(min, max), (min, max), ...].
    - `fixed_params` : list or np.ndarray, optional, default=None
        Indices of the parameters that must remain fixed during the optimization.
        For example fitting `f(x, a, b)`, if we want to fix the value of `a` we would pass
        `fit_f(guess=[a_guess, b_guess], fixed_params=[0])`
    - `fixed_bound_factor` : float, optional, default=1e-6
        The relative tolerance allowed for parameters that must remain fixed (`fixed_params`).

    IMPORTANT: This decorator requires the x and y input vectors to be named `x_data` and `y_data`.
        The initial guess must be called `guess` and the bounds `bounds`.

    Parameters
    ----------
    fit_func : callable
        The fitting function to be decorated. This function should accept `x_data` and `y_data` as
        mandatory parameters and may optionally accept `guess` and `bounds` (plus any other additional
        parameter).

    Returns
    -------
    callable
        A wrapper function that processes the input arguments and then calls the original fitting
        function with the preprocessed inputs. This function also handles warnings if unsupported
        parameters are passed to the fit function.

    Notes
    -----
    - The parameters in `guess`, `bounds` and `fixed_params` must be in the same order as in the
      modeled function definition.
    - The decorator can fix certain parameters by narrowing their bounds based on an initial guess
      and a specified `fixed_bound_factor`.
    - The decorator processes bounds by setting them as `(-np.inf, np.inf)` if they are not specified (`None`).

    Examples
    -------
    >>> @fit_input
    ... def my_fit_func(x_data, y_data, guess=None, bounds=None, fixed_params=None):
    ...     # Perform fitting...
    ...     return fit_result
    >>> x_data = np.linspace(0, 10, 100)
    >>> y_data = np.sin(x_data) + np.random.normal(0, 0.1, 100)
    >>> result = my_fit_func(x_data, y_data, guess=[1, 1], bounds=[(0, 5), (-np.inf, np.inf)])
    """  # noqa: E501

    @wraps(fit_func)
    def wrapper(
        *params,
        guess=None,
        bounds=None,
        fixed_params=None,
        fixed_bound_factor=1e-6,
        sigma=None,
        **kwargs,
    ):
        # Inspect function to check if it requires guess and bounds
        func_params = inspect.signature(fit_func).parameters

        # Check if the user passed parameters that are not supported by the fit fun
        if (guess is not None) and ("guess" not in func_params):
            warnings.warn(
                "The fit function doesn't allow any initial guess.", stacklevel=2
            )
        if (bounds is not None) and ("bounds" not in func_params):
            warnings.warn("The fit function doesn't allow any bounds.", stacklevel=2)
        if (fixed_params is not None) and (guess is None):
            raise ValueError("Using fixed_params requires an initial guess.")

        # Process bounds if the function accepts it
        if (bounds is not None) and ("bounds" in func_params):
            processed_bounds = np.array(
                [(-np.inf, np.inf) if b is None else b for b in bounds],
                dtype=np.float64,
            )
            lower_bounds, upper_bounds = (
                processed_bounds[:, 0],
                processed_bounds[:, 1],
            )
        else:
            lower_bounds, upper_bounds = None, None

        # Fix parameters by setting a very tight bound
        if (fixed_params is not None) and (guess is not None):
            if bounds is None:
                lower_bounds = -np.inf * np.ones(len(guess))
                upper_bounds = np.inf * np.ones(len(guess))
            for idx in fixed_params:
                tolerance = (
                    abs(guess[idx]) * fixed_bound_factor
                    if guess[idx] != 0
                    else fixed_bound_factor
                )
                lower_bounds[idx] = guess[idx] - tolerance
                upper_bounds[idx] = guess[idx] + tolerance

        # Prepare arguments dynamically
        fit_args = {**kwargs}

        if guess is not None and "guess" in func_params:
            fit_args["guess"] = guess
        if (
            (bounds is not None) or (fixed_params is not None)
        ) and "bounds" in func_params:
            fit_args["bounds"] = (lower_bounds, upper_bounds)

        # Call the wrapped function with preprocessed inputs
        fit_args = {**kwargs, **fit_args}
        return fit_func(*params, **fit_args)

    return wrapper


def _process_metadata(metadata: dict, sqil_dict: dict):
    """Process metadata by computing values that cannot be calculated before having
    the sqil_dict. For example use the standard errors to compute a different metric.

    Treats items whose key starts with @ as functions that take sqil_dict as input.
    So it evaluates them and renames the key removing the @.
    """
    res = metadata.copy()
    for key, value in metadata.items():
        if key.startswith("@"):
            res[key[1:]] = value(sqil_dict)
            del res[key]
    return res


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
                "Covariance matrix could not be estimated due to an almost perfect fit."
                " Standard errors are undefined but may not be necessary in this case.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Covariance matrix could not be estimated. This could be due to poor"
                "model fit or numerical instability. "
                "Review the data or model configuration.",
                stacklevel=2,
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
    - The reduced chi-squared value (χ²_red) should ideally be close to 1 for a good fit
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
            "Degrees of freedom (dof) is non-positive. "
            "This may indicate overfitting or insufficient data.",
            stacklevel=3,
        )
        red_chi2 = np.nan
    else:
        red_chi2 = chi2 / dof

    return chi2, red_chi2


def compute_aic(residuals: np.ndarray, n_params: int) -> float:
    """
    Computes the Akaike Information Criterion (AIC) for a given model fit.

    The AIC is a metric used to compare the relative quality of statistical models
    for a given dataset. It balances model fit with complexity, penalizing models
    with more parameters to prevent overfitting.

    Interpretation: The AIC has no maeaning on its own, only the difference between
    the AIC of model1 and the one of model2.
    ΔAIC = AIC_1 - AIC_2
    If ΔAIC > 10 -> model 2 fits much better.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals between the observed data and model predictions.
    n_params : int
        Number of free parameters in the fitted model.

    Returns
    -------
    float
        The Akaike Information Criterion value.
    """

    n = len(residuals)
    rss = np.sum(residuals**2)
    return 2 * n_params + n * np.log(rss / n)


def compute_nrmse(residuals: np.ndarray, y_data: np.ndarray) -> float:
    """
    Computes the Normalized Root Mean Squared Error (NRMSE) of a model fit.

    Lower is better.

    The NRMSE is a scale-independent metric that quantifies the average magnitude
    of residual errors normalized by the range of the observed data. It is useful
    for comparing the fit quality across different datasets or models.

    For complex data it's computed using the L2 norm and the span of the magnitude.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals between the observed data and model predictions.
    y_data : np.ndarray
        The original observed data used in the model fitting.

    Returns
    -------
    float
        The normalized root mean squared error (NRMSE).
    """
    n = len(residuals)
    if np.iscomplexobj(y_data):
        y_abs_span = np.max(np.abs(y_data)) - np.min(np.abs(y_data))
        if y_abs_span == 0:
            warnings.warn(
                "y_data has zero span in magnitude. NRMSE is undefined.",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.nan
        rmse = np.linalg.norm(residuals) / np.sqrt(n)
        nrmse = rmse / y_abs_span
    else:
        y_span = np.max(y_data) - np.min(y_data)
        if y_span == 0:
            warnings.warn(
                "y_data has zero span. NRMSE is undefined.",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.nan
        rss = np.sum(residuals**2)
        nrmse = np.sqrt(rss / n) / y_span

    return nrmse


def _is_scipy_tuple(result):
    """
    Check whether the given result follows the expected structure of a SciPy
    optimization tuple.
    """
    if isinstance(result, tuple):
        if len(result) < 3:
            raise TypeError(
                "Fit result is a tuple, but couldn't recognize it.\n"
                + "Are you using scipy? Did you forget to set `full_output=True` "
                "in your fit method?"
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
    Check whether the given result follows the expected structure of a SciPy
    least_squares.
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


def _format_scipy_tuple(result, y_data=None, has_sigma=False):
    """
    Formats the output of a SciPy fitting function into a standardized dictionary.

    This function takes the tuple returned by SciPy optimization functions
    (e.g., `curve_fit`, `leastsq`) and extracts relevant fitting parameters,
    standard errors, and reduced chi-squared values. It ensures the result is
    structured consistently for further processing.

    Parameters
    ----------
    result : tuple
        A tuple containing the fitting results from a SciPy function. Expected
        structure:
        - `result[0]`: `popt` (optimized parameters, NumPy array)
        - `result[1]`: `pcov` (covariance matrix, NumPy array or None)
        - `result[2]`: `infodict` (dictionary containing residuals, required
          for error computation)

    y_data : bool, optional
        The y data that has been fit. Used to compute some fit metrics.

    has_sigma : bool, optional
        Indicates whether the fitting procedure considered experimental errors
        (`sigma`). If `True`, the covariance matrix (`pcov`) does not need
        rescaling.

    Returns
    -------
    dict
        A dictionary containing:
        - `"params"`: The optimized parameters (`popt`).
        - `"std_err"`: The standard errors computed from the covariance matrix
          (`pcov`).
        - `"metrics"`: A dictionary containing the reduced chi-squared
          (`red_chi2`).
    """

    if not isinstance(result, tuple):
        raise TypeError("Fit result must be a tuple")

    popt, pcov, infodict = None, None, None
    std_err = None
    metrics = {}

    # Extract output parameters
    length = len(result)
    popt = result[0]
    pcov = result[1] if length > 1 else None
    infodict = result[2] if length > 2 else None

    if infodict is not None:
        residuals = infodict["fvec"]
        # Reduced chi squared
        _, red_chi2 = compute_chi2(
            residuals, n_params=len(popt), cov_rescaled=has_sigma
        )
        # AIC
        aic = compute_aic(residuals, len(popt))
        # NRMSE
        if y_data is not None:
            nrmse = compute_nrmse(residuals, y_data)
            metrics.update({"nrmse": nrmse})
        metrics.update({"red_chi2": red_chi2, "aic": aic})
        # Standard error
        if pcov is not None:
            std_err = compute_adjusted_standard_errors(
                pcov, residuals, cov_rescaled=has_sigma, red_chi2=red_chi2
            )
    return {"params": popt, "std_err": std_err, "metrics": metrics}


def _format_scipy_least_squares(result, y_data=None, has_sigma=False):
    """
    Formats the output of a SciPy least-squares optimization into a standardized
    dictionary.

    This function processes the result of a SciPy least-squares fitting function
    (e.g., `scipy.optimize.least_squares`) and structures the fitting parameters,
    standard errors, and reduced chi-squared values for consistent downstream use.

    Parameters
    ----------
    result : `scipy.optimize.OptimizeResult`
        The result of a least-squares optimization (e.g., from
        `scipy.optimize.least_squares`). It must contain the following fields:
        - `result.x`: Optimized parameters (NumPy array)
        - `result.fun`: Residuals (array of differences between the observed and
          fitted data)
        - `result.jac`: Jacobian matrix (used to estimate covariance)

    y_data : bool, optional
        The y data that has been fit. Used to compute some fit metrics.

    has_sigma : bool, optional
        Indicates whether the fitting procedure considered experimental errors
        (`sigma`). If `True`, the covariance matrix does not need rescaling.

    Returns
    -------
    dict
        A dictionary containing:
        - `"params"`: Optimized parameters (`result.x`).
        - `"std_err"`: Standard errors computed from the covariance matrix and
          residuals.
        - `"metrics"`: A dictionary containing the reduced chi-squared
          (`red_chi2`).
    """
    metrics = {}

    params = result.x
    residuals = result.fun
    cov = np.linalg.inv(result.jac.T @ result.jac)

    _, red_chi2 = compute_chi2(residuals, n_params=len(params), cov_rescaled=has_sigma)
    aic = compute_aic(residuals, len(params))
    if y_data is not None:
        nrmse = compute_nrmse(residuals, y_data)
        metrics.update({"nrmse": nrmse})
    metrics.update({"red_chi2": red_chi2, "aic": aic})

    std_err = compute_adjusted_standard_errors(
        cov, residuals, cov_rescaled=has_sigma, red_chi2=red_chi2
    )

    return {"params": params, "std_err": std_err, "metrics": metrics}


def _format_scipy_minimize(result, residuals=None, y_data=None, has_sigma=False):
    """
    Formats the output of a SciPy minimize optimization into a standardized
    dictionary.

    This function processes the result of a SciPy minimization optimization
    (e.g., `scipy.optimize.minimize`) and structures the fitting parameters,
    standard errors, and reduced chi-squared values for consistent downstream
    use.

    Parameters
    ----------
    result : `scipy.optimize.OptimizeResult`
        The result of a minimization optimization (e.g., from
        `scipy.optimize.minimize`). It must contain the following fields:
        - `result.x`: Optimized parameters (NumPy array).
        - `result.hess_inv`: Inverse Hessian matrix used to estimate the
          covariance.

    residuals : array-like, optional
        The residuals (differences between observed data and fitted model). If
        not provided, standard errors will be computed based on the inverse
        Hessian matrix.

    y_data : bool, optional
        The y data that has been fit. Used to compute some fit metrics.

    has_sigma : bool, optional
        Indicates whether the fitting procedure considered experimental errors
        (`sigma`). If `True`, the covariance matrix does not need rescaling.

    Returns
    -------
    dict
        A dictionary containing:
        - `"params"`: Optimized parameters (`result.x`).
        - `"std_err"`: Standard errors computed either from the Hessian matrix
          or based on the residuals.
        - `"metrics"`: A dictionary containing the reduced chi-squared
          (`red_chi2`), if residuals are provided.
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
        aic = compute_aic(residuals, len(params))
        if y_data is not None:
            nrmse = compute_nrmse(residuals, y_data)
            metrics.update({"nrmse": nrmse})
        metrics.update({"red_chi2": red_chi2, "aic": aic})

    return {"params": params, "std_err": std_err, "metrics": metrics}


def _format_lmfit(result: ModelResult):
    """
    Formats the output of an lmfit model fitting result into a standardized
    dictionary.

    This function processes the result of an lmfit model fitting (e.g., from
    `lmfit.Model.fit`) and structures the fitting parameters, their standard
    errors, reduced chi-squared, and a prediction function.

    Parameters
    ----------
    result : `lmfit.ModelResult`
        The result of an lmfit model fitting procedure. It must contain the
        following fields:
        - `result.params`: A dictionary of fitted parameters and their values.
        - `result.redchi`: The reduced chi-squared value.
        - `result.eval`: A method to evaluate the fitted model using independent
          variable values.
        - `result.userkws`: Dictionary of user-supplied keywords that includes
          the independent variable.
        - `result.model.independent_vars`: List of independent variable names in
          the model.

    Returns
    -------
    dict
        A dictionary containing:
        - `"params"`: Optimized parameters (as a NumPy array).
        - `"std_err"`: Standard errors of the parameters.
        - `"metrics"`: A dictionary containing the reduced chi-squared
          (`red_chi2`).
        - `"predict"`: A function that predicts the model's output given an
          input (using optimized parameters).
        - `"param_names"`: List of parameter names.

    Notes
    -----
    - lmfit already rescales standard errors by the reduced chi-squared, so no
      further adjustments are made.
    - The independent variable name used in the fit is determined from
      `result.userkws` and `result.model.independent_vars`.
    - The function creates a prediction function (`predict`) from the fitted
      model.
    """

    params = np.array([param.value for param in result.params.values()])
    param_names = list(result.params.keys())
    std_err = np.array(
        [
            param.stderr if param.stderr is not None else np.nan
            for param in result.params.values()
        ]
    )

    aic = compute_aic(result.residual, len(params))
    nrmse = compute_nrmse(result.residual, result.data)
    metrics = {"red_chi2": result.redchi, "aic": aic, "nrmse": nrmse}

    # Determine the independent variable name used in the fit
    independent_var = result.userkws.keys() & result.model.independent_vars
    independent_var = (
        independent_var.pop() if independent_var else result.model.independent_vars[0]
    )

    def fit_function(x):
        return result.eval(**{independent_var: x})

    return {
        "params": params,
        "std_err": std_err,
        "metrics": metrics,
        "predict": fit_function,
        "param_names": param_names,
    }


def _get_covariance_from_scipy_optimize_result(
    result: spopt.OptimizeResult,
) -> np.ndarray:
    """
    Extracts the covariance matrix (or an approximation) from a scipy
    optimization result.

    This function attempts to retrieve the covariance matrix of the fitted
    parameters from the result object returned by a scipy optimization method.
    It first checks for the presence of the inverse Hessian (`hess_inv`), which
    is used to estimate the covariance. If it's not available, the function
    attempts to compute the covariance using the Hessian matrix (`hess`).

    Parameters
    ----------
    result : `scipy.optimize.OptimizeResult`
        The result object returned by a scipy optimization function, such as
        `scipy.optimize.minimize` or `scipy.optimize.curve_fit`. This object
        contains the optimization results, including the Hessian or its inverse.

    Returns
    -------
    np.ndarray or None
        The covariance matrix of the optimized parameters, or `None` if it
        cannot be computed. If the inverse Hessian (`hess_inv`) is available,
        it will be returned directly. If the Hessian matrix (`hess`) is
        available and not singular, its inverse will be computed and returned.
        If neither is available, the function returns `None`.

    Notes
    -----
    - If the Hessian matrix (`hess`) is singular or nearly singular, the
      covariance matrix cannot be computed.
    - In some cases, the inverse Hessian (`hess_inv`) is directly available and
      provides the covariance without further computation.
    """

    if hasattr(result, "hess_inv"):
        hess_inv = result.hess_inv

        # Handle different types of hess_inv
        if isinstance(hess_inv, np.ndarray):
            return hess_inv
        if hasattr(hess_inv, "todense"):
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

    This helper function retrieves the x and y data (1D vectors) from the
    function's arguments or keyword arguments. The function checks for common
    keyword names like "x_data", "xdata", "x", "y_data", "ydata", and "y", and
    returns the corresponding data. If no keyword arguments are found, it
    attempts to extract the first two consecutive 1D vectors from the positional
    arguments.

    Parameters
    ----------
    *args : variable length argument list
        The positional arguments passed to the function, potentially containing
        the x and y data.

    **kwargs : keyword arguments
        The keyword arguments passed to the function, potentially containing
        keys such as "x_data", "x", "y_data", or "y".

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        A tuple containing the x data and y data as 1D numpy arrays or lists.
        If no valid data is found, returns (None, None).

    Raises
    ------
    ValueError
        If both x and y data cannot be found in the input arguments.

    Notes
    -----
    - The function looks for the x and y data in the keyword arguments first, in
      the order of x_keys and y_keys.
    - If both x and y data are not found in keyword arguments, the function will
      look for the first two consecutive 1D vectors in the positional arguments.
    - If the data cannot be found, the function will return (None, None).
    - The function validates that the extracted x and y data are 1D vectors
      (either lists or numpy arrays).
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

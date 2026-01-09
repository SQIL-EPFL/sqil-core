from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Literal

import numpy as np
from tabulate import tabulate

if TYPE_CHECKING:
    from sqil_core.fit._core import FitResult


class FitQuality(IntEnum):
    BAD = 0
    ACCEPTABLE = 1
    GOOD = 2
    GREAT = 3

    def __str__(self):
        return self.name


FIT_QUALITY_THRESHOLDS = {
    "nrmse": [
        (0.01, FitQuality.GREAT),
        (0.03, FitQuality.GOOD),
        (0.08, FitQuality.ACCEPTABLE),
        (np.inf, FitQuality.BAD),
    ],
    "nmae": [
        (0.01, FitQuality.GREAT),
        (0.03, FitQuality.GOOD),
        (0.08, FitQuality.ACCEPTABLE),
        (np.inf, FitQuality.BAD),
    ],
    "red_chi2": [
        (0.5, FitQuality.ACCEPTABLE),
        (0.9, FitQuality.GOOD),
        (1.1, FitQuality.GREAT),
        (2.0, FitQuality.GOOD),
        (5.0, FitQuality.ACCEPTABLE),
        (np.inf, FitQuality.BAD),
    ],
}


def evaluate_fit_quality(fit_metrics: dict, recipe: str = "nrmse") -> FitQuality:
    """
    Evaluates the quality category of a fit based on a specified metric recipe.

    This function maps a numeric fit metric (e.g., NRMSE or AIC) to a qualitative
    fit quality category (GREAT, GOOD, ACCEPTABLE, BAD) using predefined thresholds.
    These thresholds are stored in the `FIT_QUALITY_THRESHOLDS` dictionary and must be
    provided for each supported recipe.

    Parameters
    ----------
    fit_metrics : dict
        Dictionary containing computed metrics from a fit. Must include the key
        specified by `recipe`.
    recipe : str, optional
        The name of the metric to evaluate quality against. Default is "nrmse".

    Returns
    -------
    FitQuality
        A qualitative classification of the fit (GREAT, GOOD, ACCEPTABLE, BAD),
        represented by an enum or constant defined in `FitQuality`.
    """

    value = fit_metrics.get(recipe)
    if value is None:
        raise KeyError(
            f"The metrics provided aren't sufficient to use recipe '{recipe}'"
        )

    thresholds = FIT_QUALITY_THRESHOLDS.get(recipe)
    if thresholds is None:
        raise NotImplementedError(
            f"No fit quality threshold available for '{recipe}'."
            + " You can add them to 'FIT_QUALITY_THRESHOLDS'"
        )

    for threshold, quality in thresholds:
        if value <= threshold:
            return quality

    return FitQuality.BAD


def get_best_fit(
    fit_res_a: FitResult,
    fit_res_b: FitResult,
    recipe: Literal["nrmse_aic"] = "nrmse_aic",
):
    """
    Selects the better fit result according to a specified selection recipe.

    This function acts as a dispatcher to choose between two fit results using a
    predefined comparison strategy.

    Supported recipies:
        - "nrmse_aic": uses NRMSE as primary metric and adjusts it with AIC if the
            NRMSE are in the same quality category.

    Parameters
    ----------
    fit_res_a : FitResult
        The first fit result object containing metrics and parameters.
    fit_res_b : FitResult
        The second fit result object containing metrics and parameters.
    recipe : Literal["nrmse_aic"], optional
        The name of the comparison strategy to use.

    Returns
    -------
    FitResult
        The selected fit result, based on the comparison strategy.

    Examples
    --------
    >>> best_fit = get_best_fit(fit1, fit2)
    >>> print("Best-fit parameters:", best_fit.params)
    """

    if recipe == "nrmse_aic":
        return get_best_fit_nrmse_aic(fit_res_a, fit_res_b)
    raise NotImplementedError(f"Recipe {recipe} does not exist")


def get_best_fit_nrmse_aic(
    fit_res_a: FitResult, fit_res_b: FitResult, aic_rel_tol: float = 0.01
):
    """
    Selects the better fit result based on NRMSE quality and AIC with complexity penalty

    This function compares two fit results by first evaluating the normalized root
    mean squared error (NRMSE) using a quality categorization scheme. If the fits
    differ in NRMSE quality, the one with better quality is selected. If the
    qualities are equal, the function compares the Akaike Information Criterion (AIC),
    using a relative tolerance to determine statistical equivalence. When AIC values
    are within tolerance, the simpler model (with fewer parameters) is preferred.

    Parameters
    ----------
    fit_res_a : FitResult
        The first FitResult object.
    fit_res_b : FitResult
        The second FitResult object.
    aic_rel_tol : float, optional
        The relative tolerance for AIC comparison. If the relative difference in AIC
        is below this threshold, models are considered equally good, and complexity
        (number of parameters) is used as a tiebreaker. Default is 0.01.

    Returns
    -------
    FitResult
        The preferred fit result based on NRMSE category, AIC, and model simplicity.

    Notes
    -----
    - If models are statistically equivalent in AIC and have the same complexity,
      the first result is returned for consistency.
    - If the minimum AIC is zero, relative delta AIC is replaced by its absolute counter
      part, but still using the aic_rel_tol as tolerance.

    Examples
    --------
    >>> best_fit = get_best_fit_nrmse_aic(fit1, fit2)
    >>> print("Selected model parameters:", best_fit.params)
    """

    quality_a = evaluate_fit_quality(fit_res_a.metrics)
    quality_b = evaluate_fit_quality(fit_res_b.metrics)

    # If NMRSE qualities are not in the same category, return the best one
    if quality_a != quality_b:
        return fit_res_a if quality_a > quality_b else fit_res_b
    aic_a = fit_res_a.metrics.get("aic")
    aic_b = fit_res_b.metrics.get("aic")

    # Use AIC to penalize fit complexity
    if aic_a is None or aic_b is None:
        raise ValueError("Missing AIC value in one of the fits")
    delta = abs(aic_a - aic_b)
    min_aic = abs(min(aic_a, aic_b))
    rel_delta = delta / min_aic if min_aic != 0 else delta
    if rel_delta < aic_rel_tol:
        # Within tolerance: consider them equivalent, return simpler (fewer params)
        len_a, len_b = len(fit_res_a.params), len(fit_res_b.params)
        if len_a != len_b:
            return fit_res_a if len_a < len_b else fit_res_b
        # Otherwise: arbitrary but consistent
        return fit_res_a
    # Outside tolerance: pick the one with lower AIC
    return fit_res_a if aic_a < aic_b else fit_res_b


def format_fit_metrics(fit_metrics: dict, keys: list[str] | None = None) -> str:
    """
    Formats and summarizes selected fit metrics with qualitative evaluations.

    This function generates a human-readable table that reports selected fit metrics
    (e.g., reduced χ², R², NRMSE) alongside their numerical values and qualitative
    quality assessments. Quality categories are determined using `evaluate_fit_quality`.

    Parameters
    ----------
    fit_metrics : dict
        Dictionary of fit metrics to display. Should contain values for keys like
        "red_chi2", "r2", "nrmse", etc.
    keys : list of str, optional
        Subset of metric keys to include in the output. If None, all available keys
        in `fit_metrics` are considered.

    Returns
    -------
    str
        A plain-text table summarizing the selected metrics with their values and
        associated quality labels.

    Notes
    -----
    - Complex-valued R² metrics are skipped.
    - Keys are optionally renamed for output formatting
    (e.g., "red_chi2" → "reduced χ²").

    Examples
    --------
    >>> metrics = {"red_chi2": 1.2, "r2": 0.97, "nrmse": 0.05}
    >>> print(format_fit_metrics(metrics))
    reduced χ²   1.200e+00   GOOD
    R²           9.700e-01   GOOD
    nrmse        5.000e-02   GOOD
    """

    table_data = []

    if keys is None:
        keys = fit_metrics.keys() if fit_metrics else []

    # Print fit quality parameters
    for key in keys:
        value = fit_metrics[key]
        quality = ""
        # Evaluate reduced Chi-squared
        if key == "red_chi2":
            key = "reduced χ²"
            quality = evaluate_fit_quality(fit_metrics, "red_chi2")
        # Evaluate R-squared
        elif key == "r2":
            # Skip if complex
            if isinstance(value, complex):
                continue
            key = "R²"
            quality = evaluate_fit_quality(fit_metrics, "r2")
        # Normalized root mean square error NRMSE
        # Normalized mean absolute error NMAE and
        elif (key == "nrmse") or (key == "nmae"):
            quality = evaluate_fit_quality(fit_metrics, key)
        else:
            continue

        quality_label = str(quality)

        table_data.append([key, f"{value:.3e}", quality_label])
    return tabulate(table_data, tablefmt="plain")

from enum import IntEnum

import numpy as np
from tabulate import tabulate


class FitQuality(IntEnum):
    BAD = 0
    ACCEPTABLE = 1
    GOOD = 2
    GREAT = 3


FIT_QUALITY_LABELS = {
    FitQuality.GREAT: "GREAT",
    FitQuality.GOOD: "GOOD",
    FitQuality.ACCEPTABLE: "ACCEPTABLE",
    FitQuality.BAD: "BAD",
}


FIT_QUALITY_THRESHOLDS = {
    "nrmse": [
        (0.01, FitQuality.GREAT),
        (0.03, FitQuality.GOOD),
        (0.1, FitQuality.ACCEPTABLE),
        (np.inf, FitQuality.BAD),
    ],
    "nmae": [
        (0.01, FitQuality.GREAT),
        (0.03, FitQuality.GOOD),
        (0.1, FitQuality.ACCEPTABLE),
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


def evaluate_fit_quality(fit_metrics: dict, recipe="nrmse") -> FitQuality:
    value = fit_metrics.get(recipe, np.nan)
    if np.isnan(value):
        return FitQuality.BAD

    thresholds = FIT_QUALITY_THRESHOLDS.get(recipe)
    if thresholds is None:
        raise ValueError(f"No fit quality recipe named '{recipe}'")

    for threshold, quality in thresholds:
        if value <= threshold:
            return quality

    return FitQuality.BAD


def format_fit_metrics(fit_metrics, keys: list[str] | None = None):
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

        quality_label = FIT_QUALITY_LABELS.get(quality, "UNKNOWN")

        table_data.append([key, f"{value:.3e}", quality_label])
    return tabulate(table_data, tablefmt="plain")

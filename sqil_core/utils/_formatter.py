from decimal import ROUND_DOWN, Decimal

import numpy as np
from scipy.stats import norm
from tabulate import tabulate

from ._const import _EXP_UNIT_MAP, _PARAM_METADATA


def _cut_to_significant_digits(number, n):
    """Cut a number to n significant digits."""
    if number == 0:
        return 0  # Zero has no significant digits
    d = Decimal(str(number))
    shift = d.adjusted()  # Get the exponent of the number
    rounded = d.scaleb(-shift).quantize(
        Decimal("1e-{0}".format(n - 1)), rounding=ROUND_DOWN
    )
    return float(rounded.scaleb(shift))


def format_number(
    num: float | np.ndarray, precision: int = 3, unit: str = "", latex: bool = True
) -> str:
    """Format a number (or an array of numbers) in a nice way for printing.

    Parameters
    ----------
    num : float | np.ndarray
        Input number (or array). Should not be rescaled,
        e.g. input values in Hz, NOT GHz
    precision : int
        The number of digits of the output number. Must be >= 3.
    unit : str, optional
        Unit of measurement, by default ''
    latex : bool, optional
        Include Latex syntax, by default True

    Returns
    -------
    str
        Formatted number
    """
    # Handle arrays
    if isinstance(num, (list, np.ndarray)):
        return [format_number(n, precision, unit, latex) for n in num]

    # Return if not a number
    if not isinstance(num, (int, float, complex)):
        return num

    # Format number
    exp_form = f"{num:.12e}"
    base, exponent = exp_form.split("e")
    # Make exponent a multiple of 3
    base = float(base) * 10 ** (int(exponent) % 3)
    exponent = (int(exponent) // 3) * 3
    # Apply precision to the base
    if precision < 3:
        precision = 3
    base_precise = _cut_to_significant_digits(
        base, precision + 1
    )  # np.round(base, precision - (int(exponent) % 3))
    base_precise = np.round(
        base_precise, precision - len(str(base_precise).split(".")[0])
    )
    if int(base_precise) == float(base_precise):
        base_precise = int(base_precise)

    # Build string
    if unit:
        res = f"{base_precise}{'~' if latex else ' '}{_EXP_UNIT_MAP[exponent]}{unit}"
    else:
        res = f"{base_precise}" + (f" x 10^{{{exponent}}}" if exponent != 0 else "")
    return f"${res}$" if latex else res


def get_name_and_unit(param_id: str) -> str:
    """Get the name and unit of measurement of a prameter, e.g. Frequency [GHz].

    Parameters
    ----------
    param : str
        Parameter ID, as defined in the param_dict.json file.

    Returns
    -------
    str
        Name and [unit]
    """
    meta = _PARAM_METADATA[param_id]
    scale = meta["scale"] if "scale" in meta else 1
    exponent = -(int(f"{scale:.0e}".split("e")[1]) // 3) * 3
    return f"{meta['name']} [{_EXP_UNIT_MAP[exponent]}{meta['unit']}]"


def print_fit_params(param_names, params, std_errs=None, perc_errs=None):
    matrix = [param_names, params]

    headers = ["Param", "Fitted value"]
    if std_errs is not None:
        headers.append("STD error")
        std_errs = [f"{n:.3e}" for n in std_errs]
        matrix.append(std_errs)
    if perc_errs is not None:
        headers.append("% Error")
        perc_errs = [f"{n:.2f}" for n in perc_errs]
        matrix.append(perc_errs)

    matrix = np.array(matrix)
    data = [matrix[:, i] for i in range(len(params))]

    table = tabulate(data, headers=headers, tablefmt="github")
    print(table + "\n")


def print_fit_metrics(fit_quality, keys: list[str] | None = None):
    if keys is None:
        keys = fit_quality.keys() if fit_quality else []

    # Print fit quality parameters
    for key in keys:
        value = fit_quality[key]
        quality = ""
        # Evaluate reduced Chi-squared
        if key == "red_chi2":
            key = "reduced χ²"
            if value <= 0.5:
                quality = "GREAT (or overfitting)"
            elif (value > 0.9) and (value <= 1.1):
                quality = "GREAT"
            elif (value > 0.5) and (value <= 2):
                quality = "GOOD"
            elif (value > 2) and (value <= 5):
                quality = "MEDIUM"
            elif value > 5:
                quality = "BAD"
        # Evaluate R-squared
        elif key == "r2":
            # Skip if complex
            if isinstance(value, complex):
                continue
            key = "R²"
            if value < 0:
                quality = "BAD - a horizontal line would be better"
            elif value > 0.97:
                quality = "GREAT"
            elif value > 0.95:
                quality = "GOOD"
            elif value > 0.80:
                quality = "MEDIUM"
            else:
                quality = "BAD"
        # Normalized mean absolute error NMAE and
        # normalized root mean square error NRMSE
        elif (key == "nmae") or (key == "nrmse"):
            if value < 0.1:
                quality = "GREAT"
            elif value < 0.2:
                quality = "GOOD"
            else:
                quality = "BAD"

        # Print result
        print(f"{key}\t{value:.3e}\t{quality}")


def _sigma_for_confidence(confidence_level: float) -> float:
    """
    Calculates the sigma multiplier (z-score) for a given confidence level.

    Parameters
    ----------
    confidence_level : float
        The desired confidence level (e.g., 0.95 for 95%, 0.99 for 99%).

    Returns
    -------
    float
        The sigma multiplier to use for the confidence interval.
    """
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1 (exclusive).")

    alpha = 1 - confidence_level
    sigma_multiplier = norm.ppf(1 - alpha / 2)

    return sigma_multiplier

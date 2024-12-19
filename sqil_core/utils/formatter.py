from decimal import ROUND_DOWN, Decimal

import numpy as np

from .const import EXP_UNIT_MAP, PARAM_METADATA
from .read import read_json


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
        return [format_number(n, unit, latex) for n in num]

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
        res = f"{base_precise}{'~' if latex else ' '}{EXP_UNIT_MAP[exponent]}{unit}"
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
    meta = PARAM_METADATA[param_id]
    scale = meta["scale"] if "scale" in meta else 1
    exponent = -(int(f"{scale:.0e}".split("e")[1]) // 3) * 3
    return f"{meta['name']} [{EXP_UNIT_MAP[exponent]}{meta['unit']}]"


def get_x_id_by_plot_dim(exp_id: str, plot_dim: str, sweep_param: str | None) -> str:
    if exp_id == "CW_onetone":
        if plot_dim == "1":
            return sweep_param or "ro_freq"
        return "ro_freq"


def build_title(title: str, path: str, params: list[str]) -> str:
    """Build a plot title that includes the values of given parameters found in
    the params_dict.json file, e.g. One tone with I = 0.5 mA.

    Parameters
    ----------
    title : str
        Title of the plot to which the parameters will be appended.

    path: str
        Path to the param_dict.json file.

    params : List[str]
        List of keys of parameters in the param_dict.json file.

    Returns
    -------
    str
        The original title followed by parameter values.
    """
    dic = read_json(f"{path}/param_dict.json")
    title += " with "
    for idx, param in enumerate(params):
        if not (param in PARAM_METADATA.keys()) or not (param in dic):
            title += f"{param} = ? & "
            continue
        meta = PARAM_METADATA[param]
        value = format_number(dic[param], meta["unit"])
        title += f"${meta['symbol']} =${value} & "
        if idx % 2 == 0 and idx != 0:
            title += "\n"
    return title[0:-3]

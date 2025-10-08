import json
from decimal import ROUND_DOWN, Decimal

import attrs
import numpy as np
from scipy.stats import norm
from tabulate import tabulate

from ._const import _EXP_UNIT_MAP, PARAM_METADATA


def _cut_to_significant_digits(number, n):
    """Cut a number to n significant digits."""
    if number == 0:
        return 0  # Zero has no significant digits
    d = Decimal(str(number))
    shift = d.adjusted()  # Get the exponent of the number
    rounded = d.scaleb(-shift).quantize(Decimal(f"1e-{n - 1}"), rounding=ROUND_DOWN)
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
    exp_name = _EXP_UNIT_MAP.get(exponent, None)
    # If the exponent value is not mapped to a name
    if exp_name is None:
        return f"{exp_form} {unit}" if unit else exp_form
    # Apply precision to the base
    precision = max(precision, 3)
    base_precise = _cut_to_significant_digits(base, precision + 1)
    base_precise = np.round(
        base_precise, precision - len(str(base_precise).split(".")[0])
    )
    if int(base_precise) == float(base_precise):
        base_precise = int(base_precise)

    # Build string
    if unit:
        res = f"{base_precise}{'~' if latex else ' '}{exp_name}{unit}"
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
    return f"{meta['name']} [{_EXP_UNIT_MAP[exponent]}{meta['unit']}]"


def format_fit_params(param_names, params, std_errs=None, perc_errs=None):
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
    return table + "\n"


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


class ParamInfo:
    """Parameter information for items of param_dict

    Attributes:
        id (str): QPU key
        value (any): the value of the parameter
        name (str): full name of the parameter (e.g. Readout frequency)
        symbol (str): symbol of the parameter in Latex notation (e.g. f_{RO})
        unit (str): base unit of measurement (e.g. Hz)
        scale (int): the scale that should be generally applied to raw data
        (e.g. 1e-9 to take raw Hz to GHz)
    """

    def __init__(self, id, value=None, metadata=None):
        self.id = id
        self.value = value

        if metadata is not None:
            meta = metadata
        elif id in PARAM_METADATA:
            meta = PARAM_METADATA[id]
        else:
            meta = {}

        self.name = meta.get("name", None)
        self.symbol = meta.get("symbol", id)
        self.unit = meta.get("unit", "")
        self.scale = meta.get("scale", 1)
        self.precision = meta.get("precision", 3)

        if self.name is None:
            self.name = self.id[0].upper() + self.id[1:].replace("_", " ")

    def to_dict(self):
        """Convert ParamInfo to a dictionary."""
        return {
            "id": self.id,
            "value": self.value,
            "name": self.name,
            "symbol": self.symbol,
            "unit": self.unit,
            "scale": self.scale,
            "precision": self.precision,
        }

    @property
    def name_and_unit(self, latex=True):
        unit = f"[{self.rescaled_unit}]" if self.unit or self.scale != 1 else ""
        if unit == "":
            return self.name
        return self.name + rf" ${unit}$" if latex else rf" {unit}"

    @property
    def rescaled_unit(self):
        exponent = -(int(f"{self.scale:.0e}".split("e")[1]) // 3) * 3
        exp_name = _EXP_UNIT_MAP.get(exponent, "")
        unit = f"{exp_name}{self.unit}"
        return unit

    @property
    def symbol_and_value(self, latex=True):
        sym = f"${self.symbol}$" if latex else self.symbol
        equal = "$=$" if latex else " = "
        val = format_number(self.value, self.precision, self.unit, latex=latex)
        return f"{sym}{equal}{val}"

    def __str__(self):
        """Return a JSON-formatted string of the object."""
        return json.dumps(self.to_dict())

    def __eq__(self, other):
        if isinstance(other, ParamInfo):
            return (self.id == other.id) & (self.value == other.value)
        if isinstance(other, (int, float, complex, str)):
            return self.value == other
        return False

    def __bool__(self):
        return bool(self.id)


ParamDict = dict[str, ParamInfo]


def param_info_from_schema(key, metadata) -> ParamInfo:
    metadata_id = metadata.get("param_id")
    if metadata_id is not None:
        return ParamInfo(metadata_id)
    return ParamInfo(key, metadata=metadata)


def enrich_qubit_params(qubit) -> ParamDict:
    qubit_params = attrs.asdict(qubit.parameters)
    res = {}
    for key, value in qubit_params.items():
        res[key] = ParamInfo(key, value)
    return res


def get_relevant_exp_parameters(
    qubit_params: ParamDict, exp_param_ids: list, sweep_ids: list, only_keys=True
):
    # If current is not null, add it to relevant parameters
    current_info = qubit_params.get("current", None)
    if current_info is not None and current_info.value is not None:
        exp_param_ids = [*exp_param_ids, "current"]

    # Filter out sweeps
    no_sweeps = [id for id in exp_param_ids if id not in sweep_ids]

    # Filter special cases
    parms_to_exclude = []
    # No external LO frequency => external Lo info is irrelevant
    if ("readout_external_lo_frequency" in exp_param_ids) and (
        not qubit_params.get("readout_external_lo_frequency").value
    ):
        parms_to_exclude = [
            "readout_external_lo_frequency",
            "readout_external_lo_power",
        ]

    filtered = [id for id in no_sweeps if id not in parms_to_exclude]
    result = {key: value for key, value in qubit_params.items() if key in filtered}

    return list(result.keys()) if only_keys else result

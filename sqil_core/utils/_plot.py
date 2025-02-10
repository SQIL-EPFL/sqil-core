import numpy as np

from ._const import _PARAM_METADATA
from ._formatter import format_number
from ._read import read_json


def set_plot_style(plt):
    style = {
        "font.size": 20,
        "xtick.labelsize": 18,  # X-axis tick labels
        "ytick.labelsize": 18,  # Y-axis tick labels
        "lines.linewidth": 2.5,  # Line width
        "lines.marker": "o",
        "lines.markersize": 7,  # Marker size
        "lines.markeredgewidth": 1.5,  # Marker line width
        "lines.markerfacecolor": "none",
        "axes.grid": True,
        "grid.linestyle": "--",
        "xtick.major.size": 8,
        "xtick.major.width": 1.5,
        "ytick.major.size": 8,
        "ytick.major.width": 1.5,
        "figure.figsize": (20, 7),
    }
    return plt.rcParams.update(style)


def reset_plot_style(plt):
    return plt.rcParams.update(plt.rcParamsDefault)


def get_x_id_by_plot_dim(exp_id: str, plot_dim: str, sweep_param_id: str | None) -> str:
    """Returns the param_id of the parameter that should be used as the x-axis."""
    if exp_id == "CW_onetone" or exp_id == "pulsed_onetone":
        if plot_dim == "1":
            return sweep_param_id or "ro_freq"
        return "ro_freq"
    elif exp_id == "CW_twotone" or exp_id == "pulsed_twotone":
        if plot_dim == "1":
            return sweep_param_id or "qu_freq"
        return "qu_freq"


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
        if not (param in _PARAM_METADATA.keys()) or not (param in dic):
            title += f"{param} = ? & "
            continue
        meta = _PARAM_METADATA[param]
        value = format_number(dic[param], 3, meta["unit"])
        title += f"${meta['symbol']} =${value} & "
        if idx % 2 == 0 and idx != 0:
            title += "\n"
    return title[0:-3]


def guess_plot_dimension(
    f: np.ndarray, sweep: np.ndarray | list = [], threshold_2D=10
) -> tuple[list["1", "1.5", "2"] | np.ndarray]:
    """Guess if the plot should be a 1D line, a collection of 1D lines (1.5D),
    or a 2D color plot.

    Parameters
    ----------
    f : np.ndarray
        Main variable, usually frequency
    sweep : Union[np.ndarray, List], optional
        Sweep variable, by default []
    threshold_2D : int, optional
        Threshold of sweeping parameters after which the data is considered, by default 10

    Returns
    -------
    Tuple[Union['1', '1.5', '2'], np.ndarray]
        The plot dimension ('1', '1.5' or '2') and the vector that should be used as the x
        axis in the plot.
    """
    if len(sweep) > threshold_2D:
        return "2"
    elif len(f.shape) == 2 and len(sweep.shape) == 1:
        return "1.5"
    else:
        return "1"

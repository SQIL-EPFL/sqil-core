import matplotlib.pyplot as plt
import numpy as np

from ._const import PARAM_METADATA
from ._formatter import format_number, param_info_from_schema
from ._read import extract_h5_data, map_data_dict, read_json


def set_plot_style(plt):
    """Sets the matplotlib plotting style to a SQIL curated one."""
    style = {
        "font.size": 20,
        "xtick.labelsize": 18,  # X-axis tick labels
        "ytick.labelsize": 18,  # Y-axis tick labels
        "lines.linewidth": 2.5,  # Line width
        # "lines.marker": "o",
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
    reset_plot_style(plt)
    return plt.rcParams.update(style)


def reset_plot_style(plt):
    """Resets the matplotlib plotting style to its default value."""
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
        if not (param in PARAM_METADATA.keys()) or not (param in dic):
            title += f"{param} = ? & "
            continue
        meta = PARAM_METADATA[param]
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


def plot_mag_phase(path=None, datadict=None):
    if path is None and datadict is None:
        raise Exception("At least one of `path` and `datadict` must be specified.")

    if path is not None:
        datadict = extract_h5_data(path, schema=True)

    # Get schema and map data
    schema = datadict.get("schema")
    x_data, y_data, sweeps, datadict_map = map_data_dict(datadict)

    # Get metadata on x_data and y_data
    x_info = param_info_from_schema(
        datadict_map["x_data"], schema[datadict_map["x_data"]]
    )
    y_info = param_info_from_schema(
        datadict_map["y_data"], schema[datadict_map["y_data"]]
    )
    # Rescale data
    x_data_scaled = x_data * x_info.scale
    y_data_scaled = y_data * y_info.scale

    set_plot_style(plt)

    if len(sweeps) == 0:  # 1D plot
        fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

        x_data_scaled = x_data * x_info.scale
        y_data_scaled = y_data * y_info.scale

        axs[0].plot(x_data_scaled, np.abs(y_data_scaled), "o")
        axs[0].set_ylabel(
            "Magnitude" + f" [{y_info.rescaled_unit}]" if y_info.unit else ""
        )
        axs[0].tick_params(labelbottom=True)
        axs[0].xaxis.set_tick_params(
            which="both", labelbottom=True
        )  # Redundant for safety

        axs[1].plot(x_data_scaled, np.unwrap(np.angle(y_data_scaled)), "o")
        axs[1].set_xlabel(x_info.name_and_unit)
        axs[1].set_ylabel("Phase [rad]")
    else:  # 2D plot
        fig, axs = plt.subplots(1, 2, figsize=(24, 12), sharex=True, sharey=True)

        sweep_key = datadict_map["sweeps"][0]
        sweep0_info = param_info_from_schema(sweep_key, schema[sweep_key])
        sweep0_scaled = sweeps[0] * sweep0_info.scale

        print(sweep0_info)

        c0 = axs[0].pcolormesh(
            x_data_scaled, sweep0_scaled, np.abs(y_data_scaled), shading="auto"
        )
        fig.colorbar(c0, ax=axs[0])
        axs[0].set_title(
            "Magnitude" + f" [{y_info.rescaled_unit}]" if y_info.unit else ""
        )
        axs[0].set_xlabel(x_info.name_and_unit)
        axs[0].set_ylabel(sweep0_info.name_and_unit)

        c1 = axs[1].pcolormesh(
            x_data_scaled,
            sweep0_scaled,
            np.unwrap(np.angle(y_data_scaled)),
            shading="auto",
        )
        fig.colorbar(c1, ax=axs[1])
        axs[1].set_title("Phase [rad]")
        axs[1].set_xlabel(x_info.name_and_unit)
        axs[1].tick_params(labelleft=True)
        axs[1].xaxis.set_tick_params(
            which="both", labelleft=True
        )  # Redundant for safety

    fig.tight_layout()
    return fig, axs

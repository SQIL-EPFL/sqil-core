from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from sqil_core.fit import transform_data

from ._analysis import (
    amplitude_to_power_dBm,
    remove_linear_background,
    remove_offset,
    soft_normalize,
)
from ._const import PARAM_METADATA
from ._formatter import ParamInfo, format_number, get_relevant_exp_parameters
from ._read import get_data_and_info, read_json

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from sqil_core.fit._core import FitResult
    from sqil_core.utils import ParamDict


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
    if exp_id == "CW_twotone" or exp_id == "pulsed_twotone":
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
        if param not in PARAM_METADATA.keys() or param not in dic:
            title += f"{param} = ? & "
            continue
        meta = PARAM_METADATA[param]
        value = format_number(dic[param], 3, meta["unit"])
        title += f"${meta['symbol']} =${value} & "
        if idx % 2 == 0 and idx != 0:
            title += "\n"
    return title[0:-3]


def guess_plot_dimension(
    f: np.ndarray, sweep: np.ndarray | list = None, threshold_2D=10
) -> tuple[list[1, 1.5, 2] | np.ndarray]:
    """Guess if the plot should be a 1D line, a collection of 1D lines (1.5D),
    or a 2D color plot.

    Parameters
    ----------
    f : np.ndarray
        Main variable, usually frequency
    sweep : Union[np.ndarray, List], optional
        Sweep variable, by default []
    threshold_2D : int, optional
        Threshold of sweeping parameters after which the data is considered,
        by default 10

    Returns
    -------
    Tuple[Union['1', '1.5', '2'], np.ndarray]
        The plot dimension ('1', '1.5' or '2') and the vector that should be used as the
        x-axis in the plot.
    """
    if sweep is None:
        sweep = []

    if len(sweep) > threshold_2D:
        return "2"
    if len(f.shape) == 2 and len(sweep.shape) == 1:
        return "1.5"
    return "1"


def finalize_plot(
    fig: Figure | None,
    title: str,
    qu_id: str,
    fit_res: FitResult = None,
    qubit_params: ParamDict = None,
    updated_params: dict = None,
    sweep_info=None,
    relevant_params=None,
):
    """
    Annotates a matplotlib figure with experiment parameters, fit quality, and title.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to annotate.
    title : str
        Title text to use for the plot.
    fit_res : FitResult, optional
        Fit result object containing model name and quality summary.
    qubit_params : ParamDict, optional
        Dictionary of experimental qubit parameters, indexed by parameter ID.
    updated_params : dict, optional
        Dictionary of updated parameters (e.g., from fitting), where keys are param IDs
        and values are numeric or symbolic parameter values.
    sweep_info : dict, optional
        Information about sweep parameters (e.g., their IDs and labels).
    relevant_params : list, optional
        List of parameter IDs considered relevant for display under "Experiment".
    """
    if fig is None:
        return

    if qubit_params is None:
        qubit_params = {}
    if updated_params is None:
        updated_params = {}
    if sweep_info is None:
        sweep_info = {}
    if relevant_params is None:
        relevant_params = []

    # Make a summary of relevant experimental parameters
    exp_params_keys = get_relevant_exp_parameters(
        qubit_params, relevant_params, [info.id for info in sweep_info]
    )
    params_str = ",   ".join(
        [qubit_params[id].symbol_and_value for id in exp_params_keys]
    )
    # Make a summary of the updated qubit parameters
    updated_params_info = {k: ParamInfo(k, v) for k, v in updated_params.items()}
    update_params_str = ",   ".join(
        [updated_params_info[id].symbol_and_value for id in updated_params_info]
    )

    # Find appropriate y_position to print text
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig_height_inches = bbox.height
    if fig_height_inches < 8:
        y_pos = -0.05
    elif fig_height_inches < 10:
        y_pos = -0.03
    elif fig_height_inches < 13:
        y_pos = -0.02
    else:
        y_pos = -0.01

    # Add text to the plot
    if not title.endswith(qu_id):
        title += f" @ {qu_id}"
    fig.suptitle(f"{title}\n" + update_params_str)
    if fit_res:
        fig.text(0.02, y_pos, f"Model: {fit_res.model_name} - {fit_res.quality()}")
    if params_str:
        fig.text(0.4, y_pos, "Experiment:   " + params_str, ha="left")


def plot_mag_phase(path=None, datadict=None, raw=False, transpose=False, plot=None):
    """
    Plot the magnitude and phase of complex measurement data from an db path or
    in-memorydictionary.

    This function generates either a 1D or 2D plot of the magnitude and phase of complex
    data, depending on the presence of sweep parameters. It supports normalization and
    background subtraction.

    Parameters
    ----------
    path : str or None, optional
        Path to the folder containing measurement data. Required if `datadict` is not
        provided.
    datadict : dict or None, optional
        Pre-loaded data dictionary with schema, typically extracted using
        `extract_h5_data`.
        Required if `path` is not provided.
    raw : bool, default False
        If True, skip normalization and background subtraction for 2D plots. Useful for
        viewing raw data.
    transpose: bool, default False
        Transposes the plot, swapping x and y axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    axs : matplotlib.axes.Axes or ndarray of Axes
        The Axes object(s) used for the subplot(s).

    Raises
    ------
    Exception
        If neither `path` nor `datadict` is provided.

    Notes
    -----
    - Axes and units are automatically inferred from the schema in the dataset.
    """

    all_data, all_info, _ = get_data_and_info(path=path, datadict=datadict)
    x_data, y_data, sweeps = all_data
    x_info, y_info, sweep_info = all_info

    # Rescale data
    x_data_scaled = x_data * x_info.scale
    y_data_scaled = y_data * y_info.scale
    y_unit = f" [{y_info.rescaled_unit}]" if y_info.unit else ""

    set_plot_style(plt)

    if len(sweeps) == 0:  # 1D plot
        if plot is None:
            fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        else:
            fig, axs = plot

        axs[0].plot(x_data_scaled, np.abs(y_data_scaled), "o")
        axs[0].set_ylabel("Magnitude" + y_unit)
        axs[0].tick_params(labelbottom=True)
        axs[0].xaxis.set_tick_params(
            which="both", labelbottom=True
        )  # Redundant for safety

        axs[1].plot(x_data_scaled, np.unwrap(np.angle(y_data_scaled)), "o")
        axs[1].set_xlabel(x_info.name_and_unit)
        axs[1].set_ylabel("Phase [rad]")
    else:  # 2D plot
        if plot is not None:
            fig, axs = plot
        elif not transpose:
            fig, axs = plt.subplots(1, 2, figsize=(24, 12), sharex=True, sharey=True)
        else:
            fig, axs = plt.subplots(2, 1, figsize=(20, 16), sharex=True, sharey=True)

        # Process mag and phase
        mag, phase = np.abs(y_data), np.unwrap(np.angle(y_data))
        if not raw:
            mag = soft_normalize(remove_offset(mag))
            flat_phase = remove_linear_background(x_data, phase, points_cut=1)
            phase = soft_normalize(flat_phase)
        # Load sweep parameter
        sweep0_info = sweep_info[0]
        sweep0_scaled = sweeps[0] * sweep0_info.scale

        if transpose:
            x_data_scaled = x_data_scaled[0, :]
            x_data_scaled, sweep0_scaled = sweep0_scaled, x_data_scaled
            mag, phase = mag.T, phase.T
            x_info, sweep0_info = sweep0_info, x_info

        c0 = axs[0].pcolormesh(
            x_data_scaled, sweep0_scaled, mag, shading="auto", cmap="PuBu"
        )
        if raw:
            fig.colorbar(c0, ax=axs[0])
            axs[0].set_title("Magnitude" + y_unit)
        else:
            axs[0].set_title("Magnitude (normalized)")
        if not transpose:
            axs[0].set_xlabel(x_info.name_and_unit)
        axs[0].set_ylabel(sweep0_info.name_and_unit)
        if transpose:
            axs[0].tick_params(labelbottom=True)

        c1 = axs[1].pcolormesh(
            x_data_scaled, sweep0_scaled, phase, shading="auto", cmap="PuBu"
        )
        if raw:
            fig.colorbar(c1, ax=axs[1])
            axs[1].set_title("Phase [rad]")
        else:
            axs[1].set_title("Phase (normalized)")
        axs[1].set_xlabel(x_info.name_and_unit)
        if transpose:
            axs[1].set_ylabel(sweep0_info.name_and_unit)
        axs[1].tick_params(labelleft=True)
        axs[1].xaxis.set_tick_params(
            which="both", labelleft=True
        )  # Redundant for safety

    fig.tight_layout()
    return fig, axs


def add_power_axis(ax, power_offset=10, n_ticks=6):
    # Get ticks
    amp_ticks = np.linspace(*ax.get_ylim(), num=n_ticks)
    power_ticks = amplitude_to_power_dBm(amp_ticks, power_offset)
    # Add power axis
    ax_pow = ax.twinx()
    ax_pow.set_ylim(ax.get_ylim())
    # Set new ticks
    ax.set_yticks(amp_ticks)
    ax_pow.set_yticks(amp_ticks)
    ax_pow.set_yticklabels([f"{p:.1f}" for p in power_ticks])
    ax_pow.set_ylabel("Power [dBm]")
    ax_pow.grid(False)


def plot_projection_IQ(path=None, datadict=None, proj_data=None, full_output=False):
    """
    Plots the real projection of complex I/Q data versus the x-axis and the full IQ
    plane.

    Parameters
    ----------
    path : str, optional
        Path to the HDF5 file containing the data. Required if `datadict` is not
        provided.
    datadict : dict, optional
        Pre-loaded data dictionary with schema, typically extracted using
        `extract_h5_data`.
        Required if `path` is not provided.
    proj_data : np.ndarray, optional
        Precomputed projected data (real part of transformed complex values).
        If not provided, it will be computed using `transform_data`.
    full_output : bool, default False
        Whether to return projected data and the inverse transformation function.

    Returns
    -------
    res : tuple
        If `full_output` is False:
            (fig, [ax_proj, ax_iq])
        If `full_output` is True:
            (fig, [ax_proj, ax_iq], proj_data, inv)
        - `fig`: matplotlib Figure object.
        - `ax_proj`: Axis for projection vs x-axis.
        - `ax_iq`: Axis for I/Q scatter plot.
        - `proj_data`: The real projection of the complex I/Q data.
        - `inv`: The inverse transformation function used during projection.

    Notes
    -----
    This function supports only 1D datasets. If sweep dimensions are detected, no plot
    is created.
    The projection is performed using a data transformation routine (e.g., PCA or
    rotation).
    """

    all_data, all_info, _ = get_data_and_info(path=path, datadict=datadict)
    x_data, y_data, sweeps = all_data
    x_info, y_info, sweep_info = all_info

    # Get y_unit
    y_unit = f" [{y_info.rescaled_unit}]" if y_info.unit else ""

    fig = None
    set_plot_style(plt)

    if len(sweeps) == 0:
        # Project data
        if proj_data is None:
            proj_data, inv = transform_data(y_data, inv_transform=True)

        set_plot_style(plt)
        fig = plt.figure(figsize=(20, 7), constrained_layout=True)
        gs = GridSpec(nrows=1, ncols=10, figure=fig, wspace=0.2)

        # Plot the projection
        ax_proj = fig.add_subplot(gs[:, :6])  # 6/10 width
        ax_proj.plot(x_data * x_info.scale, proj_data.real * y_info.scale, "o")
        ax_proj.set_xlabel(x_info.name_and_unit)
        ax_proj.set_ylabel("Projected" + y_unit)

        # Plot IQ data
        ax_iq = fig.add_subplot(gs[:, 6:])  # 4/10 width
        ax_iq.scatter(0, 0, marker="+", color="black", s=150)
        ax_iq.plot(y_data.real * y_info.scale, y_data.imag * y_info.scale, "o")
        ax_iq.set_xlabel("In-Phase" + y_unit)
        ax_iq.set_ylabel("Quadrature" + y_unit)
        ax_iq.set_aspect(aspect="equal", adjustable="datalim")

    if full_output:
        res = (fig, [ax_proj, ax_iq], proj_data, inv)
    else:
        res = (fig, [ax_proj, ax_iq])
    return res


def plot_IQ_ellipse(
    data: np.ndarray,
    ax: Axes,
    color: str | None = None,
    label: str | None = None,
    center_kwargs: dict = {},
    ellipse_kwargs: dict = {},
    conf: float = 0.99,
) -> Axes:
    """
    Plot a confidence ellipse for complex IQ data on a given matplotlib axis.

    This function computes a robust center (using the median) and a covariance-based
    confidence ellipse for complex-valued IQ samples. The ellipse corresponds to a
    specified confidence level of a 2D Gaussian distribution.

    The ellipse axes and orientation are obtained via principal component analysis
    (PCA) of the covariance matrix of the IQ data.

    Parameters
    ----------
    data : np.ndarray
        Complex-valued IQ samples.
    ax : matplotlib.axes.Axes
        Matplotlib axis on which the center point and confidence ellipse
        will be drawn.
    color : str, optional
        Color used for both the center marker and the ellipse outline.
        If None, matplotlib chooses the default color cycle.
    label : str, optional
        Label associated with the center marker.
    center_kwargs : dict, optional
        Additional keyword arguments passed to `ax.plot` when drawing the
        center point. These are merged with the default center styling.
    ellipse_kwargs : dict, optional
        Additional keyword arguments passed to `matplotlib.patches.Ellipse`
        to control the appearance of the confidence ellipse.
    conf : float, optional
        Confidence level of the ellipse (default is 0.99), interpreted as the
        cumulative probability of a 2D Gaussian distribution.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plotted center point and confidence ellipse added.

    Notes
    -----
    - The center is computed using the median rather than the mean to reduce
      sensitivity to outliers.
    - Ellipse scaling is based on the chi-square distribution with two degrees
      of freedom, which is appropriate for 2D Gaussian statistics.
    """

    default_center_kw = {"marker": "o", "color": color, "label": label}
    default_ellipse_kw = {"edgecolor": color, "facecolor": "none", "lw": 2}
    ellipse_kwargs = {**default_ellipse_kw, **ellipse_kwargs}
    center_kwargs = {**default_center_kw, **center_kwargs}

    # Data matrix
    X = np.column_stack((data.real, data.imag))

    # Get center and covariance
    mu = np.median(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # PCA
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # chi-square quantile for 2D Gaussian
    scale = chi2.ppf(conf, df=2)

    width = 2 * np.sqrt(scale * eigvals[0])
    height = 2 * np.sqrt(scale * eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Plot
    center_pt, *_ = ax.plot(mu[0], mu[1], **center_kwargs)

    if "edgecolor" not in ellipse_kwargs:
        ellipse_kwargs["edgecolor"] = center_pt.get_color()

    ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, **ellipse_kwargs)
    ax.add_patch(ellipse)

    return ax

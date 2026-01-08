from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import mpld3
import numpy as np

from sqil_core.utils import (
    extract_h5_data,
    get_measurement_id,
    is_multi_qubit_datadict,
    read_qpu,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from sqil_core.fit import FitResult


class AnalysisResult:
    """
    Container for storing and managing results from a measurement analysis.

    Attributes
    ----------
    output : dict
        Dictionary of generic outputs that don't fit into other categories
    updated_params : dict[str, dict]
        Dictionary containing the updated parameters for each qubit.
    figures : dict[str, matplotlib.figure.Figure]
        Dictionary of matplotlib figures.
    fits : dict[str, FitResult]
        Dictionary of fit results.
    extra_data : dict[str, np.ndarray]
        Dictionary of auxiliary computed arrays (e.g., processed IQ data, FFT results).

    Methods
    -------
    add_exp_info_to_figures(dir_path)
        Annotates each figure with experiment ID and cooldown name from directory path.
    save_output(dir_path)
        Saves the generic output dictionary as a JSON file.
    save_figures(dir_path)
        Saves all figures as PNG and interactive HTML using mpld3.
    aggregate_fit_summaries()
        Aggregates human-readable summaries from all fit results.
    save_fits(dir_path)
        Saves aggregated fit summaries to a markdown file.
    save_extra_data(dir_path)
        Stores auxiliary numerical data in an HDF5 file.
    save_all(dir_path)
        Runs all save methods and annotates figures with experimental metadata.
    update(new_anal_res)
        Merges another `AnalysisResult` instance into the current one.
    """

    output: dict = {}
    updated_params: dict[str, dict] = {}
    figures: dict[str, Figure] = {}
    fits: dict[str, FitResult] = {}
    extra_data: dict[str, np.ndarray] = {}
    data_path: str = ""

    def __init__(
        self,
        data_path: str = None,
        output: dict = None,
        updated_params: dict[str, dict] = None,
        figures: dict[str, Figure] = None,
        fits: dict[str, FitResult] = None,
        extra_data: dict[str, np.ndarray] = None,
    ):
        self.data_path = data_path or ""
        self.output = output or {}
        self.updated_params = updated_params or {}
        self.figures = figures or {}
        self.fits = fits or {}
        self.extra_data = extra_data or {}

    def add_exp_info_to_figures(self, dir_path: str):
        if not self.figures:
            return
        id = get_measurement_id(dir_path)
        cooldown_name = Path(dir_path).parts[-3]
        for _, fig in self.figures.items():
            # Add dummy text to infer font size
            dummy_text = fig.text(0, 0, "dummy", visible=False)
            font_size = dummy_text.get_fontsize()
            dummy_text.remove()
            fig.text(
                0.98,
                0.98,
                f"{cooldown_name}\n{id} | {dir_path[-16:]}",
                ha="right",
                va="top",
                color="gray",
                fontsize=font_size * 0.8,
            )

    def save_output(self, dir_path: str):
        with open(os.path.join(dir_path, "output.json"), "w") as f:
            json.dump(self.output, f)

    def save_figures(self, dir_path: str):
        """Saves figures both as png and interactive html."""
        for key, fig in self.figures.items():
            path = os.path.join(dir_path, key)
            fig.savefig(os.path.join(f"{path}.png"), bbox_inches="tight", dpi=300)
            html = mpld3.fig_to_html(fig)
            with open(f"{path}.html", "w") as f:
                f.write(html)

    def aggregate_fit_summaries(self):
        """Aggreate all the fit summaries and include model name."""
        result = ""
        for key, fit in self.fits.items():
            summary = fit.summary(no_print=True)
            result += f"{key}\nModel: {fit.model_name}\n{summary}\n"
        return result

    def save_fits(self, dir_path: str):
        if not self.fits:
            return
        with open(os.path.join(dir_path, "fit.mono.md"), "w", encoding="utf-8") as f:
            f.write(self.aggregate_fit_summaries())

    def save_extra_data(self, dir_path: str):
        if not self.extra_data:
            return
        with h5py.File(os.path.join(dir_path, "extra.ddh5"), "a") as f:
            grp = f.require_group("data")
            for key, array in self.extra_data.items():
                # Overwrite if already exists
                if key in grp:
                    del grp[key]
                grp.create_dataset(key, data=array)

    def save_all(self, dir_path: str):
        self.save_output(dir_path)
        self.add_exp_info_to_figures(dir_path)
        self.save_figures(dir_path)
        self.save_fits(dir_path)
        self.save_extra_data(dir_path)

    def update(self, new_anal_res: AnalysisResult):
        """Updates all the fields of the current analysis result."""
        self.output.update(new_anal_res.output)
        self.figures.update(new_anal_res.figures)
        self.fits.update(new_anal_res.fits)
        self.extra_data.update(
            new_anal_res.extra_data
        )  # TODO: check how to handle this

        for qu_id, params in new_anal_res.updated_params.items():
            self.add_params(params, qu_id)

    def _add_entries_to_attr(self, attr: str, new_values: dict, parent_key=None):
        dic: dict = getattr(self, attr)
        if parent_key:
            if parent_key not in dic:
                dic[parent_key] = {}
            return dic[parent_key].update(new_values)
        return dic.update(new_values)

    def add_output(self, new_result: dict, qu_id: str):
        return self._add_entries_to_attr("output", new_result, qu_id)

    def add_params(self, new_params: dict, qu_id: str):
        """Add updated parameters for the specified qubit."""
        return self._add_entries_to_attr("updated_params", new_params, qu_id)

    def add_figure(self, new_figure: Figure, name: str, qu_id: str):
        """Add a figure for the specified qubit."""
        if not name.startswith(qu_id):
            name = f"{qu_id}_{name}"
        return self._add_entries_to_attr("figures", {name: new_figure})

    def add_fit(self, new_fit: FitResult, name: str, qu_id: str):
        """Add a fit for the specified qubit."""
        if not name.startswith(qu_id):
            name = f"{qu_id} - {name}"
        return self._add_entries_to_attr("fits", {name: new_fit})

    def add_extra_data(self, new_data: np.ndarray | list, name: str, qu_id: str):
        """Add extra data for the specified qubit."""
        if not name.startswith(qu_id):
            name = f"{qu_id}/{name}"
        return self._add_entries_to_attr("extra_data", {name: new_data})

    def get_fit(self, name: str, qu_id: str):
        return self.fits.get(f"{qu_id} - {name}")

    def get_figure(self, name: str, qu_id: str):
        return self.fits.get(f"{qu_id}_{name}")


def multi_qubit_handler(single_qubit_handler):
    """Transforms a function able to analyze single qubit data, into a function
    that analyzes multiple qubits."""

    @wraps(single_qubit_handler)
    def wrapper(
        *args,
        path=None,
        datadict=None,
        qpu=None,
        qu_id=None,
        anal_res_tot: AnalysisResult | None = None,
        **kwargs,
    ):
        anal_res_tot = anal_res_tot or AnalysisResult(data_path=path)
        fun_kwargs = locals().copy()
        fun_kwargs.update(fun_kwargs.pop("kwargs", {}))

        # Extract the full_datadict, which can be either single or multi qubit
        if path is not None and datadict is None:
            full_datadict = extract_h5_data(path, get_metadata=True)
        elif datadict is not None:
            full_datadict = datadict
        else:
            raise ValueError("At least one of `path` or `datadict` must be not be None")

        # Extract the qpu
        if qpu is None and path is not None:
            qpu = read_qpu(path, "qpu_old.json")
            fun_kwargs["qpu"] = qpu

        # Check if full_datadict is multi qubit
        is_multi_qubit = is_multi_qubit_datadict(full_datadict)

        if is_multi_qubit:
            if qu_id is None:  # no qu_id is specified => process all qubits
                db_metadata = full_datadict.get("metadata", {})
                db_schema = db_metadata.get("schema")
                for qid in full_datadict.keys():
                    fun_kwargs["qu_id"] = qid
                    fun_kwargs["datadict"] = {
                        **full_datadict[qid],
                        "metadata": {"schema": db_schema},
                    }
                    anal_res_qu = single_qubit_handler(*args, **fun_kwargs)
                    anal_res_tot.update(anal_res_qu)
                    return anal_res_tot
            else:  # qu_id is specified => process single qubit
                datadict = full_datadict[qu_id]
        else:  # full_datadict is NOT multi qubit => it's already single qubit
            datadict = full_datadict

        fun_kwargs["datadict"] = datadict
        return single_qubit_handler(*args, **fun_kwargs)

    return wrapper

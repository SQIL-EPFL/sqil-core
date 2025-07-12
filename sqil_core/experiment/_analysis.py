from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import mpld3
import numpy as np

from sqil_core.utils import get_measurement_id

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from sqil_core.fit import FitResult


class AnalysisResult:
    updated_params: dict[str, dict] = {}
    figures: dict[str, Figure] = {}
    fits: dict[str, FitResult] = {}
    extra_data: dict[str, np.ndarray] = {}

    def __init__(
        self,
        updated_params: dict[str, dict] = {},
        figures: dict[str, Figure] = {},
        fits: dict[str, FitResult] = {},
        extra_data: dict[str, np.ndarray] = {},
    ):
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
        self.add_exp_info_to_figures(dir_path)
        self.save_figures(dir_path)
        self.save_fits(dir_path)
        self.save_extra_data(dir_path)

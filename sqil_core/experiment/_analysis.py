from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import mpld3

from sqil_core.utils import get_measurement_id

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from sqil_core.fit import FitResult


class AnalysisResult:
    updated_params: dict[str, dict] = {}
    figures: dict[str, Figure] = {}
    fits: dict[str, FitResult] = {}

    def __init__(
        self,
        updated_params: dict = {},
        figures: dict = {},
        fits: dict = {},
    ):
        self.updated_params = updated_params or {}
        self.figures = figures or {}
        self.fits = fits or {}

    def add_exp_info_to_figures(self, dir_path: str):
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

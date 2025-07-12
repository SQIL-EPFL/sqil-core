from __future__ import annotations

import os
from typing import TYPE_CHECKING

import mpld3

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

    def save_figures(self, dir_path: str):
        """Saves figures both as png and interactive html."""
        for key, fig in self.figures.items():
            path = os.path.join(dir_path, key)
            fig.savefig(os.path.join(f"{path}.png"))
            html = mpld3.fig_to_html(fig)
            with open(f"{path}.html", "w") as f:
                f.write(html)

    def aggregate_fit_summaries(self):
        result = ""
        for key, fit in self.fits.items():
            summary = fit.summary(no_print=True)
            result += f"{key}\nModel: {fit.model_name}\n{summary}\n"
        return result

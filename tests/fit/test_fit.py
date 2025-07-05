import numpy as np
import pytest

from sqil_core.fit._fit import fit_lorentzian
from sqil_core.fit._models import lorentzian


class TestFitLorentzian:

    def setup_method(self):
        self.x = np.linspace(-10, 10, 1000)

    def _generate_data(self, A, x0, fwhm, y0, noise_std=0):
        y_clean = lorentzian(self.x, A, x0, fwhm, y0)
        noise = np.random.normal(0, noise_std, size=self.x.shape)
        return y_clean + noise

    def test_perfect_data(self):
        A, x0, fwhm, y0 = 5, 2, 1.5, 0.5
        y = self._generate_data(A, x0, fwhm, y0)
        res = fit_lorentzian(self.x, y)

        # params could be dict or list, so we normalize access
        params = (
            res.params
            if isinstance(res.params, dict)
            else dict(zip(res.param_names, res.params))
        )
        for true_val, pname in zip([A, x0, fwhm, y0], ["A", "x0", "fwhm", "y0"]):
            assert np.isclose(params[pname], true_val, rtol=0.01)

    @pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.1, 0.2])
    def test_noisy_data(self, noise_std):
        A, x0, fwhm, y0 = 3, -3, 2, 1
        y = self._generate_data(A, x0, fwhm, y0, noise_std=noise_std)
        res = fit_lorentzian(self.x, y)

        params = (
            res.params
            if isinstance(res.params, dict)
            else dict(zip(res.param_names, res.params))
        )
        tol = 0.2 if noise_std <= 0.1 else 0.4
        for true_val, pname in zip([A, x0, fwhm, y0], ["A", "x0", "fwhm", "y0"]):
            assert np.isclose(params[pname], true_val, rtol=tol)

        # nrmse check in metrics dict
        assert res.metrics.get("nrmse", 0) >= noise_std * 0.05

    def test_with_default_guess_and_bounds(self):
        A, x0, fwhm, y0 = 7, 3, 1, 2
        y = self._generate_data(A, x0, fwhm, y0, noise_std=0.05)
        res = fit_lorentzian(self.x, y)

        params = (
            res.params
            if isinstance(res.params, dict)
            else dict(zip(res.param_names, res.params))
        )
        for true_val, pname in zip([A, x0, fwhm, y0], ["A", "x0", "fwhm", "y0"]):
            assert np.isclose(params[pname], true_val, rtol=0.1)

    def test_edge_case_small_fwhm(self):
        A, x0, fwhm, y0 = 4, 1, 0.05, 0
        y = self._generate_data(A, x0, fwhm, y0, noise_std=0.02)
        res = fit_lorentzian(self.x, y)

        params = (
            res.params
            if isinstance(res.params, dict)
            else dict(zip(res.param_names, res.params))
        )
        assert np.isclose(params["fwhm"], fwhm, rtol=0.3)

    def test_edge_case_large_fwhm(self):
        A, x0, fwhm, y0 = 2, -2, 10, 1
        y = self._generate_data(A, x0, fwhm, y0, noise_std=0.05)
        res = fit_lorentzian(self.x, y)

        params = (
            res.params
            if isinstance(res.params, dict)
            else dict(zip(res.param_names, res.params))
        )
        assert np.isclose(params["fwhm"], fwhm, rtol=0.1)

    def test_flat_data(self):
        y = np.ones_like(self.x) * 5

        with pytest.warns(RuntimeWarning):
            res = fit_lorentzian(self.x, y)

            params = (
                res.params
                if isinstance(res.params, dict)
                else dict(zip(res.param_names, res.params))
            )

    def test_noisy_data_with_outliers(self):
        rng = np.random.default_rng(seed=42)
        true_params = [5, 2, 1, 0.6]
        y = lorentzian(self.x, *true_params)
        noise = rng.normal(0, 0.05, size=self.x.shape)
        y_noisy = y + noise
        outlier_indices = rng.choice(len(self.x), size=10, replace=False)
        y_noisy[outlier_indices] += rng.normal(3, 1, size=10)

        res = fit_lorentzian(self.x, y_noisy)

        params = (
            res.params
            if isinstance(res.params, dict)
            else dict(zip(res.param_names, res.params))
        )
        for true_val, pname in zip(true_params, ["A", "x0", "fwhm", "y0"]):
            assert np.isclose(params[pname], true_val, rtol=0.5)

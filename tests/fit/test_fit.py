import numpy as np
import pytest

from sqil_core.fit._fit import fit_lorentzian, fit_oscillations
from sqil_core.fit._models import lorentzian, oscillations


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


class TestFitOscillations:
    def setup_method(self):
        self.x = np.linspace(0, 10, 1000)

    def _generate_data(self, x, A, T, phi, y0, noise_std=0, outliers=False, seed=42):
        rng = np.random.default_rng(seed)
        y = oscillations(x, A, y0, phi, T)
        if noise_std > 0:
            y += rng.normal(0, noise_std, size=x.shape)
        if outliers:
            idx = rng.choice(len(x), size=5, replace=False)
            y[idx] += rng.normal(5, 2, size=5)
        return y

    def test_perfect_cosine_fit(self):
        A, T, phi, y0 = 3, 2, 0.5, 1
        y = self._generate_data(self.x, A, T, phi, y0)
        res = fit_oscillations(self.x, y)

        params = dict(zip(res.param_names, res.params))
        assert np.isclose(params["A"], A, rtol=0.01)
        assert np.isclose(params["T"], T, rtol=0.01)
        assert np.isclose(params["phi"], phi, rtol=0.01)
        assert np.isclose(params["y0"], y0, rtol=0.01)

    @pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.1, 0.2, 0.5, 1])
    def test_noisy_cosine_fit(self, noise_std):
        A, T, phi, y0 = 2, 4, 1.0, 0.5
        y = self._generate_data(self.x, A, T, phi, y0, noise_std=noise_std)
        res = fit_oscillations(self.x, y)

        params = dict(zip(res.param_names, res.params))
        tol = 0.1 if noise_std <= 0.1 else 0.2
        assert np.isclose(params["A"], A, rtol=tol)
        assert np.isclose(params["T"], T, rtol=tol)
        assert np.isclose(params["phi"] % params["T"], phi % params["T"], rtol=tol)
        assert np.isclose(params["y0"], y0, rtol=tol)
        assert res.metrics["nrmse"] < 2 * noise_std

    def test_short_signal(self):
        x = np.linspace(0, 3, 40)
        A, T, phi, y0 = 1.5, 1, 0.25, 0
        y = self._generate_data(x, A, T, phi, y0)
        res = fit_oscillations(x, y)

        params = dict(zip(res.param_names, res.params))
        assert np.isclose(params["T"], T, rtol=0.15)
        assert np.isclose(params["A"], A, rtol=0.2)

    def test_low_amplitude(self):
        A, T, phi, y0 = 0.05, 1.5, 0.3, 1
        y = self._generate_data(self.x, A, T, phi, y0, noise_std=0.01)
        res = fit_oscillations(self.x, y)

        params = dict(zip(res.param_names, res.params))
        assert np.isclose(params["A"], A, rtol=0.5)
        assert np.isclose(params["y0"], y0, atol=0.1)

    def test_with_default_guess_and_bounds(self):
        A, T, phi, y0 = 4, 3.5, 1.0, 2
        y = self._generate_data(self.x, A, T, phi, y0, noise_std=0.05)
        res = fit_oscillations(self.x, y, guess=None, bounds=None)

        params = dict(zip(res.param_names, res.params))
        assert np.isclose(params["T"], T, rtol=0.1)
        assert np.isclose(params["A"], A, rtol=0.1)
        assert np.isclose(params["y0"], y0, rtol=0.1)

    def test_flat_signal(self):
        y = np.ones_like(self.x) * 5
        with pytest.raises(TypeError):
            res = fit_oscillations(self.x, y)

    def test_outlier_robustness(self):
        A, T, phi, y0 = 3, 2.5, 0.4, 0.8
        y = self._generate_data(self.x, A, T, phi, y0, noise_std=0.05, outliers=True)
        res = fit_oscillations(self.x, y)

        params = dict(zip(res.param_names, res.params))
        assert np.isclose(params["T"], T, rtol=0.2)
        assert np.isclose(params["phi"], phi, rtol=0.2)
        assert np.isclose(params["y0"], y0, rtol=0.3)

    def test_pi_time_in_metadata(self):
        A, T, phi, y0 = 2, 4, 1.0, 0.5
        y = self._generate_data(self.x, A, T, phi, y0)
        res = fit_oscillations(self.x, y)

        pi_time = res.metadata["pi_time"]
        expected_pi_time = 0.5 * T + phi
        expected_pi_time = expected_pi_time % T  # Should be folded within range
        assert np.isclose(pi_time % T, expected_pi_time, atol=0.1)

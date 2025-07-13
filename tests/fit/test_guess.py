import numpy as np
import pytest

from sqil_core.fit._guess import estimate_peak, oscillations_guess
from sqil_core.fit._models import gaussian, lorentzian, oscillations, skewed_lorentzian


class TestEstimatePeak:

    def test_peak_detection_gaussian(self):
        x = np.linspace(0, 10, 200)
        y = gaussian(x, A=10, x0=5, sigma=0.5, y0=1)
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert is_peak is True
        assert np.isclose(x0, 5, atol=0.2)
        assert baseline == pytest.approx(np.min(y))
        assert height > 0
        assert fwhm > 0

    def test_dip_detection_inverted_lorentzian(self):
        x = np.linspace(0, 10, 200)
        # Generate a dip by inverting lorentzian and adding offset
        y = 2 - lorentzian(x, A=5, x0=5, fwhm=0.8, y0=0)
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert is_peak is False
        assert np.isclose(x0, 5, atol=0.2)
        assert baseline == pytest.approx(np.max(y))
        assert height > 0
        assert fwhm > 0

    def test_skewed_lorentzian_peak(self):
        x = np.linspace(4, 6, 300)
        y = skewed_lorentzian(x, A1=1, A2=0, A3=3, A4=1, fr=5, Q_tot=50)
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert is_peak is True
        assert np.isclose(x0, 5, atol=0.1)
        assert height > 0
        assert baseline <= np.min(y)
        assert fwhm > 0

    def test_fwhm_fallback_flat(self):
        x = np.linspace(0, 1, 10)
        y = np.ones_like(x) * 3.0  # flat baseline, no peak/dip
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert fwhm == pytest.approx(0.1, rel=1e-2)  # fallback to 1/10th of range
        assert height == 0
        assert isinstance(is_peak, bool)

    def test_should_select_the_most_dominant_if_multiple_peaks_gaussian(self):
        x = np.linspace(0, 10, 500)
        y = gaussian(x, A=8, x0=3, sigma=0.3, y0=0) + 0.5 * gaussian(
            x, A=6, x0=7, sigma=0.5, y0=0
        )
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert is_peak is True
        assert np.isclose(x0, 3, atol=0.2)
        assert height > 0
        assert fwhm > 0

    def test_peak_at_edge_gaussian(self):
        x = np.linspace(0, 10, 100)
        y = gaussian(x, A=10, x0=0, sigma=0.5, y0=0)
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert is_peak is True
        assert np.isclose(x0, x[0], atol=0.1)
        assert height > 0

    def test_dip_at_edge_lorentzian(self):
        x = np.linspace(0, 10, 100)
        y = 2 - lorentzian(x, A=8, x0=10, fwhm=0.6, y0=0)
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert is_peak is False
        assert np.isclose(x0, x[-1], atol=0.1)
        assert height > 0

    def test_non_uniform_x_gaussian(self):
        x = np.sort(np.random.rand(150) * 10)
        y = gaussian(x, A=7, x0=5, sigma=0.7, y0=1)
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert is_peak is True
        assert height > 0
        assert fwhm > 0

    def test_small_dataset_two_points(self):
        x = np.array([0, 1])
        y = np.array([1, 2])
        x0, fwhm, height, baseline, is_peak = estimate_peak(x, y)
        assert fwhm == pytest.approx((x[-1] - x[0]) / 10)
        assert isinstance(is_peak, bool)


class TestOscillationsGuess:

    def test_pure_cosine(self):
        x = np.linspace(0, 10, 500)
        y = oscillations(x, A=3, T=2, phi=0.5, y0=1)
        A, y0_candidates, phi_candidates, T = oscillations_guess(x, y)

        assert pytest.approx(A, rel=0.15) == 3
        assert any(pytest.approx(y0, rel=0.1) == 1 for y0 in y0_candidates)
        assert pytest.approx(T, rel=0.1) == 2
        assert all(0 <= phi < T for phi in phi_candidates)

    def test_offset_cosine_with_noise(self):
        rng = np.random.default_rng(123)
        x = np.linspace(0, 20, 800)
        noise = rng.normal(0, 0.2, size=x.shape)
        y = oscillations(x, A=4, T=5, phi=1.5, y0=2.5) + noise

        A, y0_candidates, phi_candidates, T = oscillations_guess(x, y)

        assert A > 2.5  # should remain robust despite noise
        assert any(abs(y0 - 2.5) < 0.5 for y0 in y0_candidates)
        assert pytest.approx(T, rel=0.15) == 5
        assert all(0 <= phi < T for phi in phi_candidates)

    def test_low_amplitude_signal(self):
        x = np.linspace(0, 6, 300)
        y = oscillations(x, A=0.3, T=1.5, phi=0.2, y0=0.1)

        A, y0_candidates, phi_candidates, T = oscillations_guess(x, y)

        assert pytest.approx(A, rel=0.3) == 0.3
        assert any(abs(y0 - 0.1) < 0.2 for y0 in y0_candidates)
        assert pytest.approx(T, rel=0.2) == 1.5
        assert all(0 <= phi < T for phi in phi_candidates)

    def test_constant_signal(self):
        x = np.linspace(0, 10, 100)
        y = np.ones_like(x) * 1.0

        A, y0_candidates, phi_candidates, T = oscillations_guess(x, y)

        assert A == pytest.approx(0.0)
        assert all(y0 == pytest.approx(1.0) for y0 in y0_candidates)
        assert T > 0  # fallback value based on range
        assert all(0 <= phi < T for phi in phi_candidates)

    def test_short_signal(self):
        x = np.linspace(0, 2, 20)
        y = oscillations(x, A=2, T=1, phi=0.25, y0=0.5)

        A, y0_candidates, phi_candidates, T = oscillations_guess(x, y)

        assert pytest.approx(A, rel=0.2) == 2
        assert any(abs(y0 - 0.5) < 0.3 for y0 in y0_candidates)
        assert pytest.approx(T, rel=0.2) == 1
        assert all(0 <= phi < T for phi in phi_candidates)

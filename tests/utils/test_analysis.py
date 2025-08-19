import numpy as np
import pytest

from sqil_core.utils import (
    compute_snr_peaked,
    estimate_linear_background,
    find_first_minima_idx,
    line_between_2_points,
    linear_interpolation,
    remove_linear_background,
    soft_normalize,
)


class TestEstimateLinearBackground:
    @pytest.fixture
    def synthetic_data(self):
        """Fixture to generate synthetic data for tests."""
        x = np.linspace(0, 10, 100)
        np.random.seed(53)
        # 1D data
        data_1d = 2 * x + 1 + np.random.normal(0, 0.01, size=(100,))
        # 2D data with 3 measurements
        x_2d = np.vstack([x for _ in range(3)])
        data_2d = np.vstack(
            [2 * x + 1 + np.random.normal(0, 0.01, size=(100,)) for _ in range(3)]
        )
        return x, data_1d, x_2d, data_2d

    def test_1d_no_cut(self, synthetic_data):
        x, data, _, _ = synthetic_data
        coefficients = estimate_linear_background(
            x, data, points_cut=1, cut_from_back=False
        )

        assert np.isclose(coefficients[0], 1, atol=0.15)
        assert np.isclose(coefficients[1], 2, atol=0.15)
        assert len(coefficients) == 2

    def test_1d_with_cut(self, synthetic_data):
        x, data, _, _ = synthetic_data
        coefficients = estimate_linear_background(
            x, data, points_cut=0.1, cut_from_back=False
        )

        assert np.isclose(coefficients[0], 1, atol=0.15)
        assert np.isclose(coefficients[1], 2, atol=0.15)
        assert len(coefficients) == 2

    def test_1d_from_back(self, synthetic_data):
        x, data, _, _ = synthetic_data
        # Make a data array that is 0 at the beginning to check that the regresssion
        # is actually happening from the back
        data[0:50] = 0
        coefficients = estimate_linear_background(x, data, 0.1, cut_from_back=True)

        assert np.isclose(coefficients[0], 1, atol=0.15)
        assert np.isclose(coefficients[1], 2, atol=0.15)
        assert len(coefficients) == 2

    def test_1d_from_front(self, synthetic_data):
        x, data, _, _ = synthetic_data
        # Make a data array that is 0 at the end to check that the regresssion
        # is actually happening from the front
        data[50:] = 0
        coefficients = estimate_linear_background(x, data, 0.1, cut_from_back=False)

        assert np.isclose(coefficients[0], 1, atol=0.15)
        assert np.isclose(coefficients[1], 2, atol=0.15)
        assert len(coefficients) == 2

    def test_2d_data_no_cut(self, synthetic_data):
        _, _, x, data = synthetic_data

        coefficients = estimate_linear_background(
            x, data, points_cut=1, cut_from_back=False
        )
        for coef in coefficients:
            assert np.isclose(coef[0], 1, atol=0.15)
            assert np.isclose(coef[1], 2, atol=0.15)
        assert coefficients.shape == (3, 2)

    def test_2d_data_with_cut(self, synthetic_data):
        _, _, x, data = synthetic_data

        coefficients = estimate_linear_background(
            x, data, points_cut=0.1, cut_from_back=False
        )
        for coef in coefficients:
            assert np.isclose(coef[0], 1, atol=0.15)
            assert np.isclose(coef[1], 2, atol=0.15)
        assert coefficients.shape == (3, 2)

    def test_2d_from_back(self, synthetic_data):
        _, _, x, data = synthetic_data
        data[:, :50] = 0

        coefficients = estimate_linear_background(
            x, data, points_cut=0.1, cut_from_back=True
        )
        print(coefficients)
        for coef in coefficients:
            assert np.isclose(coef[0], 1, atol=0.15)
            assert np.isclose(coef[1], 2, atol=0.15)
        assert coefficients.shape == (3, 2)

    def test_2d_from_front(self, synthetic_data):
        _, _, x, data = synthetic_data
        data[:, 50:] = 0

        coefficients = estimate_linear_background(
            x, data, points_cut=0.1, cut_from_back=False
        )
        for coef in coefficients:
            assert np.isclose(coef[0], 1, atol=0.15)
            assert np.isclose(coef[1], 2, atol=0.15)

        assert coefficients.shape == (3, 2)

    def test_points_cut_too_large(self, synthetic_data):
        x, data, _, _ = synthetic_data
        coefficients = estimate_linear_background(
            x, data, points_cut=1.5, cut_from_back=False
        )

        # Check if the coefficients are still computed
        assert len(coefficients) > 0


class TestRemoveLinearBackground:
    @pytest.fixture
    def synthetic_data(self):
        """Fixture to generate synthetic data for tests."""
        x = np.linspace(0, 10, 100)
        np.random.seed(42)
        # 1D data with linear background
        data_1d = 3 * x + 2 + np.random.normal(0, 0.01, size=(100,))
        # 2D data with 3 measurements
        x_2d = np.vstack([x for _ in range(3)])
        data_2d = np.vstack(
            [3 * x + 2 + np.random.normal(0, 0.01, size=(100,)) for _ in range(3)]
        )
        return x, data_1d, x_2d, data_2d

    def test_1d_background_removal(self, synthetic_data):
        x, data, _, _ = synthetic_data
        data_corrected = remove_linear_background(x, data, points_cut=0.1)

        coefficients = estimate_linear_background(x, data_corrected, points_cut=1)
        assert np.isclose(coefficients[0], 0, atol=0.1)
        assert np.isclose(coefficients[1], 0, atol=0.1)
        assert data_corrected.shape == data.shape

    def test_2d_background_removal(self, synthetic_data):
        _, _, x, data = synthetic_data
        data_corrected = remove_linear_background(x, data, points_cut=0.1)

        coefficients = estimate_linear_background(x, data_corrected, points_cut=1)
        for coef in coefficients:
            assert np.isclose(coef[0], 0, atol=0.1)
            assert np.isclose(coef[1], 0, atol=0.1)
        assert data_corrected.shape == data.shape

    def test_output_shape(self, synthetic_data):
        x, data_1d, x_2d, data_2d = synthetic_data
        corrected_1d = remove_linear_background(x, data_1d, points_cut=0.1)
        corrected_2d = remove_linear_background(x_2d, data_2d, points_cut=0.1)

        assert corrected_1d.shape == data_1d.shape
        assert corrected_2d.shape == data_2d.shape


class TestLinearInterpolation:
    def test_scalar_interpolation(self):
        result = linear_interpolation(3, 2, 4, 6, 8)
        assert np.isclose(result, 5.0)

    def test_array_interpolation(self):
        x_vals = np.array([3, 4, 5])
        expected = np.array([5.0, 6.0, 7.0])
        result = linear_interpolation(x_vals, 2, 4, 6, 8)
        assert np.allclose(result, expected)

    def test_x_equal_x1_x2(self):
        result = linear_interpolation(5, 3, 7, 3, 10)  # x1 == x2
        assert result == 7  # Should return y1

    def test_x_below_x1(self):
        result = linear_interpolation(1, 2, 4, 6, 8)
        expected = 3.0  # Extrapolated value
        assert np.isclose(result, expected)

    def test_x_above_x2(self):
        result = linear_interpolation(7, 2, 4, 6, 8)
        expected = 9.0  # Extrapolated value
        assert np.isclose(result, expected)

    def test_negative_values(self):
        result = linear_interpolation(-2, -5, -10, 5, 10)
        expected = -4.0
        assert np.isclose(result, expected)

    def test_x1_greater_than_x2(self):
        result = linear_interpolation(3, 6, 8, 2, 4)
        assert np.isclose(result, 5.0)

    def test_large_values(self):
        result = linear_interpolation(1e6, 0, 0, 2e6, 2e6)
        expected = 1e6
        assert np.isclose(result, expected)


class TestLineBetween2Points:
    def test_standard_case(self):
        slope, intercept = line_between_2_points(1, 2, 3, 4)
        assert intercept == 1.0
        assert slope == 1.0

    def test_horizontal_line(self):
        slope, intercept = line_between_2_points(1, 5, 3, 5)
        assert intercept == 5.0
        assert slope == 0.0

    def test_vertical_line(self):
        slope, intercept = line_between_2_points(2, 5, 2, 10)
        print(slope, intercept)
        assert intercept == 5
        assert slope == np.inf  # Function handles vertical case with slope = inf

    def test_negative_slope(self):
        slope, intercept = line_between_2_points(1, 5, 3, 1)
        assert intercept == 7.0
        assert slope == -2.0

    def test_reverse_order(self):
        slope1, intercept1 = line_between_2_points(1, 2, 3, 4)
        slope2, intercept2 = line_between_2_points(3, 4, 1, 2)
        assert intercept1 == intercept2
        assert slope1 == slope2

    def test_large_values(self):
        slope, intercept = line_between_2_points(1e6, 2e6, 2e6, 4e6)
        assert intercept == 0.0
        assert slope == 2.0

    def test_fractional_values(self):
        slope, intercept = line_between_2_points(0.5, 1.5, 2.5, 5.5)
        assert intercept == 0.5
        assert slope == 2.0


class TestSoftNormalize:
    def test_shapes_are_unchanged(self):
        assert soft_normalize(np.random.rand(100)).shape == (100,)
        assert soft_normalize(np.random.rand(10, 50)).shape == (10, 50)

    def test_does_not_change_nans(self):
        arr = np.array([1.0, 2.0, np.nan, 3.0])
        out = soft_normalize(arr)
        assert np.isnan(out[2])  # NaN stays NaN

    def test_soft_normalize_constant_input(self):
        arr = np.ones(100)
        out = soft_normalize(arr)
        assert np.allclose(out, 0.5)

    def test_soft_normalize_range(self):
        np.random.seed(7)
        arr = np.random.randn(1000)
        out = soft_normalize(arr)
        assert 0.0 <= np.nanmin(out) <= 1.0
        assert 0.0 <= np.nanmax(out) <= 1.0


class TestComputeSNRPeaked:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(13)
        x_data = np.linspace(0, 10, 1000)
        # Create a Gaussian-like peak centered at x0 = 5 with FWHM ~ 1
        peak = np.exp(-((x_data - 5) ** 2) / (2 * (0.5**2)))
        noise = np.random.normal(0, 0.08, x_data.shape)  # Gaussian noise
        y_data = peak + noise
        return x_data, y_data

    def test_snr_computation(self, synthetic_data):
        x_data, y_data = synthetic_data
        snr = compute_snr_peaked(x_data, y_data, x0=5, fwhm=1, noise_region_factor=2.5)
        assert snr > 10  # Expect a reasonably high SNR for a clear peak

    def test_snr_low_signal(self, synthetic_data):
        x_data, y_data = synthetic_data
        peak = 0.1 * np.exp(-((x_data - 5) ** 2) / (2 * (0.5**2)))
        noise = np.random.normal(0, 0.08, x_data.shape)  # Gaussian noise
        y_data = peak + noise
        snr = compute_snr_peaked(x_data, y_data, x0=5, fwhm=1, noise_region_factor=2.5)
        assert snr < 5  # Expect a lower SNR due to weak signal

    def test_snr_high_noise(self, synthetic_data):
        x_data, y_data = synthetic_data
        noise = np.random.normal(0, 1, x_data.shape)  # Stronger noise
        y_data += noise
        snr = compute_snr_peaked(x_data, y_data, x0=5, fwhm=1, noise_region_factor=2.5)
        assert snr < 2  # Expect a very low SNR due to high noise

    def test_snr_no_noise(self, synthetic_data):
        x_data, y_data = synthetic_data
        y_data = np.exp(-((x_data - 5) ** 2) / (2 * (0.5**2)))  # No noise
        snr = compute_snr_peaked(x_data, y_data, x0=5, fwhm=1, noise_region_factor=2.5)
        assert snr > 100  # Expect an extremely high SNR

    def test_snr_noise_region_too_small(self, synthetic_data):
        x_data, y_data = synthetic_data
        with pytest.warns(Warning):
            snr = compute_snr_peaked(
                x_data, y_data, x0=5, fwhm=1, noise_region_factor=10
            )
        assert snr > 0


class TestFindFirstMinimaIdx:
    @pytest.fixture
    def simple_data(self):
        return np.array([3, 2, 4, 1, 5])

    @pytest.fixture
    def data_with_no_minima(self):
        return np.array([1, 2, 3, 4, 5])

    @pytest.fixture
    def data_with_multiple_minima(self):
        return np.array([5, 1, 3, 0, 2, 1])

    @pytest.fixture
    def data_with_minimum_at_start(self):
        return np.array([0, 1, 2, 3, 4])

    @pytest.fixture
    def data_with_minimum_at_end(self):
        return np.array([4, 3, 2, 1, 0])

    def test_should_return_first_local_minimum_index(self, simple_data):
        idx = find_first_minima_idx(simple_data)
        assert idx == 1  # first local min at index 1

    def test_should_return_none_when_no_local_minimum(self, data_with_no_minima):
        idx = find_first_minima_idx(data_with_no_minima)
        assert idx is None

    def test_should_return_first_of_multiple_local_minima(
        self, data_with_multiple_minima
    ):
        idx = find_first_minima_idx(data_with_multiple_minima)
        assert idx == 1  # first local min at index 1 (value=1)

    def test_should_handle_empty_array(self):
        idx = find_first_minima_idx(np.array([]))
        assert idx is None

    def test_should_return_none_for_single_element_array(self):
        idx = find_first_minima_idx(np.array([42]))
        assert idx is None

    def test_should_return_none_for_constant_array(self):
        idx = find_first_minima_idx(np.array([2, 2, 2, 2]))
        assert idx is None

    def test_should_return_none_for_two_elements_no_minimum(self):
        idx = find_first_minima_idx(np.array([2, 3]))
        assert idx is None

    def test_should_return_none_for_two_elements_equal(self):
        idx = find_first_minima_idx(np.array([2, 2]))
        assert idx is None

    # def test_should_return_minimum_in_middle_of_plateau(self):
    #     data = np.array([5, 3, 3, 3, 4])
    #     idx = find_first_minima_idx

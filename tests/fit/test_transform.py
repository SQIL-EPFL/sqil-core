import numpy as np
import pytest

from sqil_core.fit import transform_data


class TestTransformData:
    @pytest.fixture
    def complex_data(self):
        return np.array([1 + 1j, 2 + 2j, 3 + 3j])

    @pytest.fixture
    def multidim_complex_data(self):
        # A 2x3 complex array
        return np.array([[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]])

    @pytest.fixture
    def transformed_with_full_output(self, complex_data):
        return transform_data(complex_data, transform_type="optm", full_output=True)

    def test_should_apply_real_transform(self, complex_data):
        transformed = transform_data(complex_data, transform_type="real")
        assert np.allclose(transformed, complex_data.real)

    def test_should_apply_imag_transform(self, complex_data):
        transformed = transform_data(complex_data, transform_type="imag")
        assert np.allclose(transformed, complex_data.imag)

    def test_should_apply_amplitude_transform(self, complex_data):
        transformed = transform_data(complex_data, transform_type="ampl")
        expected = np.abs(complex_data)
        assert np.allclose(transformed, expected)

    def test_should_apply_angle_transform_in_degrees(self, complex_data):
        transformed = transform_data(complex_data, transform_type="angl", deg=True)
        expected = np.degrees(np.unwrap(np.angle(complex_data)))
        assert np.allclose(transformed, expected)

    def test_should_apply_angle_transform_in_radians(self, complex_data):
        transformed = transform_data(complex_data, transform_type="angl", deg=False)
        expected = np.unwrap(np.angle(complex_data))
        assert np.allclose(transformed, expected)

    def test_should_apply_optimized_transform(self, complex_data):
        transformed = transform_data(complex_data, transform_type="optm")
        assert transformed.shape == complex_data.shape
        assert isinstance(transformed, np.ndarray)

    def test_should_apply_transform_with_provided_params(self, complex_data):
        params = [1, 1, np.pi / 4]
        transformed = transform_data(complex_data, transform_type="trrt", params=params)
        expected = ((complex_data - 1 - 1j) * np.exp(1j * (np.pi / 4))).real
        assert np.allclose(transformed, expected)

    def test_should_return_inverse_transform_function(self, complex_data):
        transformed, inv_fun = transform_data(
            complex_data, transform_type="optm", inv_transform=True
        )
        assert callable(inv_fun)
        recovered = inv_fun(transformed + 0j)
        assert recovered.shape == complex_data.shape

    def test_should_return_full_output(self, transformed_with_full_output):
        transformed, inv_fun, params, residuals = transformed_with_full_output
        assert isinstance(transformed, np.ndarray)
        assert callable(inv_fun)
        assert isinstance(params, (list, np.ndarray))
        assert isinstance(residuals, np.ndarray)

    def test_should_handle_alias_transform_types(self, complex_data):
        assert np.allclose(
            transform_data(complex_data, transform_type="optimize"),
            transform_data(complex_data, transform_type="optm"),
        )
        assert np.allclose(
            transform_data(complex_data, transform_type="realpart"),
            transform_data(complex_data, transform_type="real"),
        )
        assert np.allclose(
            transform_data(complex_data, transform_type="imaginary"),
            transform_data(complex_data, transform_type="imag"),
        )
        assert np.allclose(
            transform_data(complex_data, transform_type="amplitude"),
            transform_data(complex_data, transform_type="ampl"),
        )
        assert np.allclose(
            transform_data(complex_data, transform_type="angle"),
            transform_data(complex_data, transform_type="angl"),
        )

    def test_should_apply_translation_rotation_transform_and_return_correct_shapes(
        self, complex_data
    ):
        params = [1, -1, 0.5]
        transformed, inv_fun, *_ = transform_data(
            complex_data,
            transform_type="trrt",
            params=params,
            inv_transform=True,
            full_output=True,
        )
        assert transformed.shape == complex_data.shape
        assert isinstance(inv_fun, type(lambda x: x))

    def test_should_preserve_shape_after_real_transform(self, multidim_complex_data):
        transformed = transform_data(multidim_complex_data, transform_type="real")
        assert transformed.shape == multidim_complex_data.shape
        assert np.allclose(transformed, multidim_complex_data.real)

    def test_should_preserve_shape_after_imag_transform(self, multidim_complex_data):
        transformed = transform_data(multidim_complex_data, transform_type="imag")
        assert transformed.shape == multidim_complex_data.shape
        assert np.allclose(transformed, multidim_complex_data.imag)

    def test_should_preserve_shape_after_ampl_transform(self, multidim_complex_data):
        transformed = transform_data(multidim_complex_data, transform_type="ampl")
        assert transformed.shape == multidim_complex_data.shape
        expected = np.abs(multidim_complex_data)
        assert np.allclose(transformed, expected)

    def test_should_preserve_shape_after_angl_transform(self, multidim_complex_data):
        transformed = transform_data(
            multidim_complex_data, transform_type="angl", deg=True
        )
        expected = np.degrees(
            np.unwrap(np.angle(multidim_complex_data.flatten()))
        ).reshape(multidim_complex_data.shape)
        assert transformed.shape == multidim_complex_data.shape
        assert np.allclose(transformed, expected)

    def test_should_preserve_shape_after_optm_transform(self, multidim_complex_data):
        transformed = transform_data(multidim_complex_data, transform_type="optm")
        assert transformed.shape == multidim_complex_data.shape

    def test_inverse_transform_should_support_different_shapes(
        self, multidim_complex_data
    ):
        transformed, inv_fun = transform_data(
            multidim_complex_data, transform_type="optm", inv_transform=True
        )
        # Create new data with a different shape
        extended_input = np.ones((4, 5), dtype=complex)
        recovered = inv_fun(extended_input)
        assert recovered.shape == extended_input.shape

    def test_full_output_should_preserve_shapes(self, multidim_complex_data):
        transformed, inv_fun, params, residuals = transform_data(
            multidim_complex_data, transform_type="optm", full_output=True
        )
        assert transformed.shape == multidim_complex_data.shape
        assert residuals.shape == multidim_complex_data.shape
        assert callable(inv_fun)
        assert len(params) == 3  # [x0, y0, phi]

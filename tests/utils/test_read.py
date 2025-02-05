import json
import os
import tempfile
from pathlib import Path

import pytest

from sqil_core.utils._read import *


class TestExtractH5Data:

    test_data = {
        "array1": [1, 2, 3, 4],
        "array2": [5.5, 6.6, 7.7, 8.8],
        "array3": [1 + 1j, 1 + 2j, 1 + 3j],
    }

    @pytest.fixture(scope="function")
    def temp_h5_file(self, tmp_path):
        """Fixture that writes the HDF5 file once for all tests."""
        # Path to the temporary HDF5 file
        temp_file_path = tmp_path / "data.ddh5"

        # Open the file using h5py to write data
        with h5py.File(temp_file_path, "w") as file:
            # Create a group 'data' inside the HDF5 file
            data_group = file.create_group("data")

            # Write the datasets inside the 'data' group
            for key, value in self.test_data.items():
                data_group.create_dataset(key, data=value)

        # Yield the file path for the tests to use
        yield str(temp_file_path)

    def test_read_all_data(self, temp_h5_file):
        data = extract_h5_data(temp_h5_file, None)
        for key, value in self.test_data.items():
            assert np.array_equal(data[key], np.array(value))

    def test_extract_only_one_array(self, temp_h5_file):
        (arr1,) = extract_h5_data(temp_h5_file, ["array1"])
        assert np.array_equal(arr1, np.array(self.test_data["array1"]))

    def test_extract_many_arrays(self, temp_h5_file):
        arr1, arr2, arr3 = extract_h5_data(temp_h5_file, ["array1", "array2", "array3"])
        assert np.array_equal(arr1, np.array(self.test_data["array1"]))
        assert np.array_equal(arr2, np.array(self.test_data["array2"]))
        assert np.array_equal(arr3, np.array(self.test_data["array3"]))

    def test_if_invalid_key_return_empty(self, temp_h5_file):
        arr1, test1, test2, test3 = extract_h5_data(
            temp_h5_file, ["array1", "", True, None]
        )
        assert np.array_equal(arr1, np.array(self.test_data["array1"]))
        assert np.array_equal(test1, np.array([]))
        assert np.array_equal(test2, np.array([]))
        assert np.array_equal(test3, np.array([]))

    def test_if_the_key_is_not_present_return_empty(self, temp_h5_file):
        arr1, test1 = extract_h5_data(temp_h5_file, ["array1", "test1"])
        assert np.array_equal(arr1, np.array(self.test_data["array1"]))
        assert np.array_equal(test1, np.array([]))

    def test_if_keys_are_empty_or_invalid_return_dict(self, temp_h5_file):
        # None
        data = extract_h5_data(temp_h5_file, None)
        for key, value in self.test_data.items():
            assert np.array_equal(data[key], np.array(value))
        # Empty
        data = extract_h5_data(temp_h5_file, [])
        for key, value in self.test_data.items():
            assert np.array_equal(data[key], np.array(value))

    def test_passing_the_path_to_a_folder_opens_data_ddh5(self, temp_h5_file):
        folder_path = os.path.dirname(temp_h5_file)
        self.test_read_all_data(folder_path)


def test_read_json(tmp_path):
    """Test the read_json function using pytest's tmp_path fixture."""
    test_data = {"name": "Eren Yaeger", "age": 19, "head": 0}

    # Create the path for the temporary file
    temp_file_path = tmp_path / "test.json"

    # Write test JSON data to the temp file
    with open(temp_file_path, "w") as temp_file:
        json.dump(test_data, temp_file)

    # Call the function to read the file and store the result
    result = read_json(temp_file_path)

    # Assert that the data read matches the data written
    assert result == test_data, f"Expected {test_data}, but got {result}"


class TestReadParamDict:

    test_json_dict = {
        "ro_freq": 7556200000.0,
        "qu_power": "sweeping",
        "current": 0.001,
        "CW_onetone": {"ro_freq_start": 7526200000.0, "sweep": False},
        "CW_twotone": {"qu_freq_step": 250000.0},
        "vna_avg": 4,
        "sweep_start": -70,
        "sweep_list": [-70.0, -5.0],
    }

    expected = {
        "ro_freq": ParamInfo("ro_freq", 7556200000.0),
        "qu_power": ParamInfo("qu_power", "sweeping"),
        "current": ParamInfo("current", 1e-3),
        "CW_onetone": {
            "ro_freq_start": ParamInfo("ro_freq_start", 7526200000.0),
            "sweep": ParamInfo("sweep", False),
        },
        "CW_twotone": {
            "qu_freq_step": ParamInfo("qu_freq_step", 250000.0),
        },
        "vna_avg": ParamInfo("vna_avg", 4),
        "sweep_start": ParamInfo("sweep_start", -70),
        "sweep_list": ParamInfo("sweep_list", [-70.0, -5.0]),
    }

    expected_str = {
        "ro_freq": '{"id": "ro_freq", "value": 7556200000.0, "name": "Readout frequency", "symbol": "f_{RO}", "unit": "Hz", "scale": 1e-09}',
        "vna_avg": '{"id": "vna_avg", "value": 4, "name": "VNA averages", "symbol": "avg_{VNA}", "unit": "", "scale": 1}',
    }

    @pytest.fixture(scope="function")
    def temp_json_file(self, tmp_path):
        """Fixture that writes the param_dict file once for all tests."""
        # Path to the temporary param_dict.json file
        temp_file_path = tmp_path / "param_dict.json"

        # Write test data to temporary json file
        with open(temp_file_path, "w") as temp_file:
            json.dump(self.test_json_dict, temp_file)

        # Yield the file path for the tests to use
        yield str(temp_file_path)

    # Convert ParamInfo objects to dictionaries for comparison
    def param_info_to_dict(self, obj):
        if isinstance(obj, ParamInfo):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {key: self.param_info_to_dict(value) for key, value in obj.items()}
        return obj

    def test_reads_param_dict_into_a_dict_of_objects_with_metadata(
        self, temp_json_file
    ):
        res = read_param_dict(temp_json_file)
        for param in res:
            assert (
                res[param] == self.expected[param]
            ), f"got {str(res[param])} but should be {str(self.expected[param])}"

    def test_check_with_hard_coded_string_form(self, temp_json_file):
        res = read_param_dict(temp_json_file)
        for param in self.expected_str:
            assert str(res[param]) == self.expected_str[param]

    def test_should_only_need_the_path_to_the_directory(self, temp_json_file):
        folder_path = os.path.dirname(temp_json_file)
        res = read_param_dict(folder_path)
        for param in res:
            assert (
                res[param] == self.expected[param]
            ), f"got {str(res[param])} but should be {str(self.expected[param])}"

    def test_should_return_default_values_if_param_is_unknown(self, temp_json_file):
        test_data = {"test_id": 51}
        folder_path = Path(os.path.dirname(temp_json_file))
        temp_file_path = folder_path / "test.json"
        # Write test JSON data to the temp file
        with open(temp_file_path, "w") as temp_file:
            json.dump(test_data, temp_file)

        res = read_param_dict(temp_file_path)
        assert (
            str(res["test_id"])
            == '{"id": "test_id", "value": 51, "name": "test_id", "symbol": "test_id", "unit": "", "scale": 1}'
        )

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

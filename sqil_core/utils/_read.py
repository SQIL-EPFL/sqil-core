from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

import h5py
import numpy as np
import yaml
from laboneq import serializers

from sqil_core.utils._formatter import param_info_from_schema

from ._const import _EXP_UNIT_MAP, PARAM_METADATA

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU


# TODO: add tests for schema
def extract_h5_data(
    path: str, keys: list[str] | None = None, schema=False
) -> dict | tuple[np.ndarray, ...]:
    """Extract data at the given keys from an HDF5 file. If no keys are
    given (None) returns the data field of the object.

    Parameters
    ----------
    path : str
        path to the HDF5 file or a folder in which is contained a data.ddh5 file
    keys : None or List, optional
        list of keys to extract from file['data'], by default None

    Returns
    -------
    Dict or Tuple[np.ndarray, ...]
        The full data dictionary if keys = None.
        The tuple with the requested keys otherwise.

    Example
    -------
        Extract the data object from the dataset:
        >>> data = extract_h5_data(path)
        Extracting only 'amp' and 'phase' from the dataset:
        >>> amp, phase = extract_h5_data(path, ['amp', 'phase'])
        Extracting only 'phase':
        >>> phase, = extract_h5_data(path, ['phase'])
    """
    # If the path is to a folder open /data.ddh5
    if os.path.isdir(path):
        path = os.path.join(path, "data.ddh5")

    with h5py.File(path, "r") as h5file:
        data = h5file["data"]
        data_keys = data.keys()

        db_schema = None
        if schema:
            db_schema = json.loads(data.attrs.get("__schema__"))

        # Extract only the requested keys
        if bool(keys) and (len(keys) > 0):
            res = []
            for key in keys:
                key = str(key)
                if (not bool(key)) | (key not in data_keys):
                    res.append([])
                    continue
                res.append(np.array(data[key][:]))
            if not schema and len(res) == 1:
                return res[0]
            return tuple(res) if not schema else (*tuple(res), db_schema)
        # Extract the whole data dictionary
        h5_dict = _h5_to_dict(data)
        return h5_dict if not schema else {**h5_dict, "schema": db_schema}
    #


def _h5_to_dict(obj) -> dict:
    """Convert h5 data into a dictionary"""
    data_dict = {}
    for key in obj.keys():
        item = obj[key]
        if isinstance(item, h5py.Dataset):
            data_dict[key] = item[:]
        elif isinstance(item, h5py.Group):
            data_dict[key] = extract_h5_data(item)
    return data_dict


def map_data_dict(data_dict: dict):
    """
    Maps experimental data to standardized arrays using a provided schema.

    This function interprets the structure of a measurement data dictionary
    (obtained using extract_h5_data) by extracting relevant data fields according
    to roles specified in the database schema. It returns the x-axis values, y-axis data,
    any additional sweep parameters, and a mapping of keys used for each role.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing measurement data and an associated 'schema' key
        that defines the role of each field (e.g., "x-axis", "data", "axis").

    Returns
    -------
    x_data : np.ndarray
        Array containing the x-axis values.
    y_data : np.ndarray
        Array containing the y-axis (measured) data.
    sweeps : list[np.ndarray]
        List of additional swept parameter arrays (if any).
    key_map : dict
        Dictionary with keys `"x_data"`, `"y_data"`, and `"sweeps"` indicating
        the corresponding keys used in the original `data_dict`.

    Notes
    -----
    - If the schema is missing, the function prints a warning and returns empty arrays.
    - Each item in the schema must be a dictionary with a `"role"` key.

    Examples
    --------
    >>> x, y, sweeps, mapping = map_data_dict(experiment_data)
    >>> print(f"x-axis data from key: {mapping['x_data']}")
    """

    schema = data_dict.get("schema", None)
    if schema is None:
        print(
            "Cannot automatically read data: no database schema was provided by the experiment."
        )

    x_data, y_data, sweeps = np.array([]), np.array([]), []
    key_map = {"x_data": "", "y_data": "", "sweeps": []}

    for key, value in schema.items():
        if type(value) is not dict:
            continue
        role = value.get("role", None)
        if role == "data":
            key_map["y_data"] = key
            y_data = data_dict[key]
        elif role == "x-axis":
            key_map["x_data"] = key
            x_data = data_dict[key]
        elif role == "axis":
            key_map["sweeps"].append(key)
            sweeps.append(data_dict[key])

    return x_data, y_data, sweeps, key_map


def extract_mapped_data(path: str):
    """
    Loads measurement data from an HDF5 file and maps it into x_data, y_data and sweeps.
    The map and the database schema on which it relies are also returned.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file containing experimental data and schema definitions.

    Returns
    -------
    x_data : np.ndarray
        Array of x-axis values extracted according to the schema.
    y_data : np.ndarray
        Array of measured data values (y-axis).
    sweeps : list[np.ndarray]
        List of arrays for any additional swept parameters defined in the schema.
    datadict_map : dict
        Mapping of keys used for `"x_data"`, `"y_data"`, and `"sweeps"` in the original file.
    schema : dict
        The schema used to interpret the data structure and field roles.

    Notes
    -----
    - This function expects the file to contain a top-level "schema" key that defines the
      role of each dataset (e.g., "data", "x-axis", "axis").
    - Uses `extract_h5_data` and `map_data_dict` internally for loading and interpretation.

    Examples
    --------
    >>> x, y, sweeps, datadict_map, schema = extract_mapped_data(path)
    """

    datadict = extract_h5_data(path, schema=True)
    schema = datadict.get("schema")
    x_data, y_data, sweeps, datadict_map = map_data_dict(datadict)
    return x_data, y_data, sweeps, datadict_map, schema


def get_data_and_info(path=None, datadict=None):
    if path is None and datadict is None:
        raise Exception("At least one of `path` and `datadict` must be specified.")

    if path is not None:
        datadict = extract_h5_data(path, schema=True)

    # Get schema and map data
    schema = datadict.get("schema")
    x_data, y_data, sweeps, datadict_map = map_data_dict(datadict)

    # Get metadata on x_data and y_data
    x_info = param_info_from_schema(
        datadict_map["x_data"], schema[datadict_map["x_data"]]
    )
    y_info = param_info_from_schema(
        datadict_map["y_data"], schema[datadict_map["y_data"]]
    )

    sweep_info = []
    for sweep_key in datadict_map["sweeps"]:
        sweep_info.append(param_info_from_schema(sweep_key, schema[sweep_key]))

    return (x_data, y_data, sweeps), (x_info, y_info, sweep_info), datadict


def read_json(path: str) -> dict:
    """Reads a json file and returns the data as a dictionary."""
    with open(path) as f:
        dictionary = json.load(f)
    return dictionary


def read_yaml(path: str) -> dict:
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def read_qpu(dir_path: str, filename: str) -> QPU:
    """Reads QPU file stored in dir_path/filename using laboneq serializers."""
    qpu = serializers.load(os.path.join(dir_path, filename))
    return qpu


def get_measurement_id(path):
    return os.path.basename(path)[0:5]


def copy_folder(src: str, dst: str):
    # Ensure destination exists
    os.makedirs(dst, exist_ok=True)

    # Copy files recursively
    for root, dirs, files in os.walk(src):
        for dir_name in dirs:
            os.makedirs(
                os.path.join(dst, os.path.relpath(os.path.join(root, dir_name), src)),
                exist_ok=True,
            )
        for file_name in files:
            shutil.copy2(
                os.path.join(root, file_name),
                os.path.join(dst, os.path.relpath(os.path.join(root, file_name), src)),
            )

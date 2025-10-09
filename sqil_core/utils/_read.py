from __future__ import annotations

import json
import os
import re
import shutil
from typing import TYPE_CHECKING

import h5py
import numpy as np
import yaml
from laboneq import serializers

from sqil_core.utils._formatter import param_info_from_schema

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU


# TODO: add tests for schema
def extract_h5_data(
    path: str, keys: list[str] | None = None, get_metadata=False
) -> dict | tuple[np.ndarray, ...]:
    """Extract data at the given keys from an HDF5 file. If no keys are
    given (None) returns the data field of the object.

    Parameters
    ----------
    path : str
        path to the HDF5 file or a folder in which is contained a data.ddh5 file
    keys : None or List, optional
        list of keys to extract from file['data'], by default None
    get_metadata : bool, optional
        whether or not to extract also metadata, like database schema and qubit IDs,
        by default None.

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

        metadata = {}
        if get_metadata:
            metadata["schema"] = json.loads(data.attrs.get("__schema__", "null"))
            metadata["qu_ids"] = json.loads(data.attrs.get("__qu_ids__", "null"))
            metadata["params"] = json.loads(data.attrs.get("__params__", "null"))

        # Extract only the requested keys
        if bool(keys) and (len(keys) > 0):
            res = []
            for key in keys:
                key = str(key)
                if (not bool(key)) | (key not in data_keys):
                    res.append([])
                    continue
                res.append(np.array(data[key][:]))
            if not get_metadata and len(res) == 1:
                return res[0]
            return tuple(res) if not get_metadata else (*tuple(res), metadata)
        # Extract the whole data dictionary
        h5_dict = _h5_to_dict(data)
        return h5_dict if not get_metadata else {**h5_dict, "metadata": metadata}


def _h5_to_dict(obj) -> dict:
    """Recursively convert h5py Group or Dataset to nested dict."""
    result = {}
    for key, item in obj.items():
        if isinstance(item, h5py.Dataset):
            result[key] = item[()]
        elif isinstance(item, h5py.Group):
            result[key] = _h5_to_dict(item)  # recursive step
    return result


def map_datadict(datadict: dict):
    metadata = datadict.get("metadata", {})
    schema = metadata.get("schema")
    qu_ids = metadata.get("qu_ids")
    if schema is None:
        print(
            "Cannot automatically read data: no database schema was "
            "provided by the experiment."
        )

    # Handle data with only one unspecified qubit
    if qu_ids is None:
        qu_ids = [None]

    result = {}
    for qu_id in qu_ids:
        qu_datadict = datadict[qu_id] if qu_id is not None else datadict
        x_data, y_data, sweeps = np.array([]), np.array([]), []
        key_map = {"x_data": "", "y_data": "", "sweeps": []}

        for key, value in schema.items():
            if type(value) is not dict:
                continue
            role = value.get("role", None)
            if role == "data":
                key_map["y_data"] = key
                y_data = qu_datadict[key]
            elif role == "x-axis":
                key_map["x_data"] = key
                x_data = qu_datadict[key]
            elif role == "axis":
                key_map["sweeps"].append(key)
                sweeps.append(qu_datadict[key])

        result[qu_id] = x_data, y_data, sweeps, key_map

    # Handle data with only one unspecified qubit
    if list(result.keys()) == [None]:
        return result[None]

    return result


def get_data_and_info(path=None, datadict=None):
    if path is None and datadict is None:
        raise Exception("At least one of `path` and `datadict` must be specified.")

    if datadict is None and path is not None:
        datadict = extract_h5_data(path, get_metadata=True)

    # Get schema and map data
    metadata = datadict.get("metadata", {})
    schema = metadata.get("schema")
    qu_ids = metadata.get("qu_ids")

    mapped_data = map_datadict(datadict)

    # Handle data with only one unspecified qubit
    if qu_ids is None:
        qu_ids = [None]
        dic = {None: mapped_data}
        mapped_data = dic

    data_res, info_res, dict_res = {}, {}, {}
    for qu_id in qu_ids:
        if qu_id is not None:
            qu_datadict = {**datadict[qu_id], "metadata": {"schema": schema}}
        else:
            qu_datadict = datadict

        x_data, y_data, sweeps, datadict_map = mapped_data[qu_id]

        # Get metadata on x_data and y_data
        y_info = param_info_from_schema(
            datadict_map["y_data"], schema[datadict_map["y_data"]]
        )
        x_info = None
        if datadict_map["x_data"]:
            x_info = param_info_from_schema(
                datadict_map["x_data"], schema[datadict_map["x_data"]]
            )

        sweep_info = []
        for sweep_key in datadict_map["sweeps"]:
            sweep_info.append(param_info_from_schema(sweep_key, schema[sweep_key]))

        data_res[qu_id] = (x_data, y_data, sweeps)
        info_res[qu_id] = (x_info, y_info, sweep_info)
        dict_res[qu_id] = qu_datadict

    # Handle data with only one unspecified qubit
    if list(data_res.keys()) == [None]:
        return data_res[None], info_res[None], dict_res[None]
    return data_res, info_res, dict_res


def is_multi_qubit_datadict(datadict):
    # Check that all the keys are "q<NUMBER>", e.g. "q0", "q1", ...
    pattern = re.compile(r"^q\d+$|metadata")
    all_keys_match = all(pattern.match(key) for key in datadict)
    return all_keys_match


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

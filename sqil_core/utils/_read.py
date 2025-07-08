import json
import os
import shutil

import h5py
import numpy as np
import yaml

from ._const import _EXP_UNIT_MAP, PARAM_METADATA


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

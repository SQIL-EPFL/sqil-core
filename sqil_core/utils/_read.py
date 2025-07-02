import json
import os
import shutil

import h5py
import numpy as np
import yaml

from ._const import _EXP_UNIT_MAP, _PARAM_METADATA


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


class ParamInfo:
    """Parameter information for items of param_dict

    Attributes:
        id (str): param_dict key
        value (any): the value of the parameter
        name (str): full name of the parameter (e.g. Readout frequency)
        symbol (str): symbol of the parameter in Latex notation (e.g. f_{RO})
        unit (str): base unit of measurement (e.g. Hz)
        scale (int): the scale that should be generally applied to raw data (e.g. 1e-9 to take raw Hz to GHz)
    """

    def __init__(self, id, value):
        self.id = id
        self.value = value
        if id in _PARAM_METADATA:
            meta = _PARAM_METADATA[id]
        else:
            meta = {}
        self.name = meta["name"] if "name" in meta else id
        self.symbol = meta["symbol"] if "symbol" in meta else id
        self.unit = meta["unit"] if "unit" in meta else ""
        self.scale = meta["scale"] if "scale" in meta else 1

    def get_name_and_unit(self):
        res = self.name
        if self.unit != "":
            exponent = -(int(f"{self.scale:.0e}".split("e")[1]) // 3) * 3
            unit = f" [{_EXP_UNIT_MAP[exponent]}{self.unit}]"
            res += unit
        return res

    def to_dict(self):
        """Convert ParamInfo to a dictionary."""
        return {
            "id": self.id,
            "value": self.value,
            "name": self.name,
            "symbol": self.symbol,
            "unit": self.unit,
            "scale": self.scale,
        }

    def __str__(self):
        """Return a JSON-formatted string of the object."""
        return json.dumps(self.to_dict())

    def __eq__(self, other):
        if isinstance(other, ParamInfo):
            return (self.id == other.id) & (self.value == other.value)
        if isinstance(other, (int, float, complex, str)):
            return self.value == other
        return False

    def __bool__(self):
        return bool(self.id)


ParamDict = dict[str, ParamInfo | dict[str, ParamInfo]]


def _enrich_param_dict(param_dict: dict) -> ParamDict:
    """Add metadata to param_dict entries."""
    res = {}
    for key, value in param_dict.items():
        if isinstance(value, dict):
            # Recursive step for nested dictionaries
            res[key] = _enrich_param_dict(value)
        else:
            res[key] = ParamInfo(key, value)
    return res


def read_param_dict(path: str) -> ParamDict:
    """Read param_dict and include additional information for each entry.

    Parameters
    ----------
    path : str
        Path to the file or a folder in which is contained a param_dict.json file

    Returns
    -------
    ParamDict
        The param_dict with additional metadata
    """
    # If the path is to a folder open /param_dict.json
    if os.path.isdir(path):
        path = os.path.join(path, "param_dict.json")
    return _enrich_param_dict(read_json(path))


def get_sweep_param(path: str, exp_id: str):
    params = read_param_dict(path)
    sweep_id = params[exp_id]["sweep"].value
    if sweep_id:
        return params[sweep_id]
    return ParamInfo("", "")


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

import json
import os

import h5py
import numpy as np

from .const import PARAM_METADATA


def extract_h5_data(
    path: str, keys: list[str] | None = None
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
        # Extract only the requested keys
        if bool(keys) and (len(keys) > 0):
            res = []
            for key in keys:
                key = str(key)
                if (not bool(key)) | (key not in data_keys):
                    res.append([])
                    continue
                res.append(np.array(data[key][:]))
            return tuple(res)
        # Extract the whole data dictionary
        return _h5_to_dict(data)


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
        if id in PARAM_METADATA:
            meta = PARAM_METADATA[id]
        else:
            meta = {}
        self.name = meta["name"] if "name" in meta else id
        self.symbol = meta["symbol"] if "symbol" in meta else id
        self.unit = meta["unit"] if "unit" in meta else ""
        self.scale = meta["scale"] if "scale" in meta else 1

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

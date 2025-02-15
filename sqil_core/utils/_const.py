import numpy as np

_EXP_UNIT_MAP = {
    -15: "p",
    -12: "f",
    -9: "n",
    -6: r"\mu",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
}

_PARAM_METADATA = {
    "current": {"name": "Current", "symbol": "I", "unit": "A", "scale": 1e3},
    "ro_freq": {
        "name": "Readout frequency",
        "symbol": "f_{RO}",
        "unit": "Hz",
        "scale": 1e-9,
    },
    "ro_power": {
        "name": "Readout power",
        "symbol": "P_{RO}",
        "unit": "dBm",
        "scale": 1,
    },
    "qu_freq": {
        "name": "Qubit frequency",
        "symbol": "f_q",
        "unit": "Hz",
        "scale": 1e-9,
    },
    "qu_power": {"name": "Qubit power", "symbol": "P_q", "unit": "dBm", "scale": 1},
    "vna_bw": {"name": "VNA bandwidth", "symbol": "BW_{VNA}", "unit": "Hz", "scale": 1},
    "vna_avg": {"name": "VNA averages", "symbol": "avg_{VNA}", "unit": "", "scale": 1},
    "index": {"name": "Index", "symbol": "idx", "unit": "", "scale": 1},
}

ONE_TONE_PARAMS = np.array(
    ["current", "ro_power", "vna_bw", "vna_avg", "qu_power", "qu_freq"]
)

TWO_TONE_PARAMS = np.array(
    ["ro_freq", "ro_power", "current", "vna_bw", "vna_avg", "qu_power"]
)

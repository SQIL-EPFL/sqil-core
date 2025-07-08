import numpy as np

_EXP_UNIT_MAP = {
    -15: "p",
    -12: "f",
    -9: "n",
    -6: r"\mu ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
}

PARAM_METADATA = {
    "readout_resonator_frequency": {
        "name": "Readout frequency",
        "symbol": "f_{RO}",
        "unit": "Hz",
        "scale": 1e-9,
    },
    "readout_range_out": {
        "name": "Readout power offset",
        "symbol": "P_0^{RO}",
        "unit": "dBm",
        "scale": 1,
    },
    "readout_amplitude": {
        "name": "Readout amplitude",
        "symbol": "P_{amp}^{RO}",
        "unit": "",
        "scale": 1,
    },
    "readout_length": {
        "name": "Readout length",
        "symbol": "T_{RO}",
        "unit": "s",
        "scale": 1e6,
    },
    "readout_lo_frequency": {
        "name": "Internal readout LO frequency",
        "symbol": "f_{LO-int}^{RO}",
        "unit": "Hz",
        "scale": 1e-9,
    },
    "external_lo_frequency": {
        "name": "External LO frequency",
        "symbol": "f_{LO}^{Ext}",
        "unit": "Hz",
        "scale": 1e-9,
    },
    "external_lo_power": {
        "name": "External LO power",
        "symbol": "P_{LO}^{Ext}",
        "unit": "dBm",
        "scale": 1,
    },
    "readout_kappa_tot": {"symbol": r"\kappa_{tot}", "unit": "Hz", "scale": "MHz"},
}

ONE_TONE_PARAMS = np.array(
    [
        "readout_amplitude",
        "readout_length",
        "external_lo_frequency",
        "external_lo_power",
    ]
)

TWO_TONE_PARAMS = np.array(
    ["ro_freq", "ro_power", "current", "vna_bw", "vna_avg", "qu_power"]
)

import numpy as np

_EXP_UNIT_MAP = {
    -18: "a",
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
    18: "E",
}

PARAM_METADATA = {
    # General
    "frequency": {
        "name": "Frequency",
        "symbol": "f",
        "unit": "Hz",
        "scale": 1e-9,
        "precision": 5,
    },
    "current": {"name": "Current", "symbol": "I", "unit": "A", "scale": 1e3},
    "temperature": {
        "name": "Temperature",
        "symbol": "T",
        "unit": "K",
        "scale": 1e3,
    },
    # QPU parameters
    "readout_resonator_frequency": {
        "name": "Readout frequency",
        "symbol": "f_{RO}",
        "unit": "Hz",
        "scale": 1e-9,
        "precision": 5,
    },
    "readout_range_out": {
        "name": "Readout power offset",
        "symbol": "P_0^{RO}",
        "unit": "dBm",
        "scale": 1,
    },
    "readout_amplitude": {
        "name": "Readout amplitude",
        "symbol": "A_{RO}",
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
    "readout_external_lo_frequency": {
        "name": "External LO frequency",
        "symbol": "f_{LO}^{Ext}",
        "unit": "Hz",
        "scale": 1e-9,
    },
    "readout_external_lo_power": {
        "name": "External LO power",
        "symbol": "P_{LO}^{Ext}",
        "unit": "dBm",
        "scale": 1,
    },
    "readout_kappa_tot": {"symbol": r"\kappa_{tot}", "unit": "Hz", "scale": 1e-6},
    "ge_chi_shift": {
        "name": "Dispersive shift",
        "symbol": r"\chi_{ge}",
        "unit": "Hz",
        "scale": 1e-6,
    },
    "resonance_frequency_ge": {
        "name": "Qubit frequency",
        "symbol": "f_{ge}",
        "unit": "Hz",
        "scale": 1e-9,
        "precision": 5,
    },
    "resonance_frequency_ef": {
        "name": "Qubit frequency",
        "symbol": "f_{ef}",
        "unit": "Hz",
        "scale": 1e-9,
        "precision": 5,
    },
    "spectroscopy_amplitude": {
        "name": "Spectroscopy amplitude",
        "symbol": "A_{sp}",
        "unit": "",
        "scale": 1,
    },
    "ge_drive_amplitude_pi": {
        "name": "Drive amplitude pi ge",
        "symbol": r"A_{\pi}^{ge}",
        "unit": "",
        "scale": 1,
    },
    "ge_drive_length": {
        "name": "Drive length ge",
        "symbol": r"T_{\pi}^{ge}",
        "unit": "s",
        "scale": 1e9,
    },
    "ge_T1": {"name": "T1", "symbol": "T_1", "unit": "s", "scale": 1e6},
    "ge_T2": {"name": "T2", "symbol": "T_2", "unit": "s", "scale": 1e6},
    "ge_T2_star": {"name": "T2*", "symbol": "T_2^*", "unit": "s", "scale": 1e6},
    "reset_delay_length": {
        "name": "Reset delay",
        "symbol": "T_{reset}",
        "unit": "s",
        "scale": 1e6,
    },
    "ef_drive_amplitude_pi": {
        "name": "Drive amplitude pi ef",
        "symbol": r"A_{\pi}^{ef}",
        "unit": "",
        "scale": 1,
    },
    "ef_drive_length": {
        "name": "Drive length ef",
        "symbol": r"T_{\pi}^{ef}",
        "unit": "s",
        "scale": 1e9,
    },
    "ef_T1": {"name": "T1", "symbol": "T_1", "unit": "s", "scale": 1e6},
    "ef_T2": {"name": "T2", "symbol": "T_2", "unit": "s", "scale": 1e6},
    "ef_T2_star": {"name": "T2*", "symbol": "T_2^*", "unit": "s", "scale": 1e6},
    "aux_reset_delay_length": {
        "name": "Aux reset delay",
        "symbol": "T_{reset}^{AUX}",
        "unit": "s",
        "scale": 1e6,
    },
    "aux_drive_length": {
        "name": "Aux drive length",
        "symbol": "T_{AUX}",
        "unit": "s",
        "scale": 1,
    },
    "aux_drive_amplitude": {
        "name": "Aux drive amplitude",
        "symbol": "A_{AUX}",
        "unit": "",
        "scale": 1,
    },
    "aux_frequency": {
        "name": "Aux frequency",
        "symbol": "f_{AUX}",
        "unit": "Hz",
        "scale": 1e-9,
    },
    # CW
    "readout_power": {
        "name": "Readout power",
        "symbol": "P_{RO}",
        "unit": "dBm",
        "scale": 1,
    },
    "drive_power": {
        "name": "Drive power",
        "symbol": "P_{d}",
        "unit": "dBm",
        "scale": 1,
    },
    "readout_acquire_bandwith": {
        "name": "Readout bandwidth",
        "symbol": "BW_{RO}",
        "unit": "",
        "scale": 1,
    },
    "readout_acquire_averages": {
        "name": "Readout averages",
        "symbol": "AVG_{RO}",
        "unit": "",
        "scale": 1,
    },
}

ONE_TONE_PARAMS = np.array(
    [
        "readout_amplitude",
        "readout_length",
        "readout_external_lo_frequency",
        "readout_external_lo_power",
    ]
)

TWO_TONE_PARAMS = np.array(["spectroscopy_amplitude"])

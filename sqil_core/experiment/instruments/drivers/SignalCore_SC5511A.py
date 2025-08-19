"""
Created on Fri Mar 19 14:11:31 2021

@author: Chao Zhou

A simple driver for SignalCore SC5511A to be used with QCoDes, transferred from the one written by Erick Brindock
"""

import ctypes
import logging
from typing import Any

# from helpers.customized_drivers.sc5511a import * #is this actually required?
from qcodes import Instrument
from qcodes.utils.validators import Enum, Numbers

# from Hatlab_QCoDes_Drivers import DLLPATH
# import Hatlab_QCoDes_Drivers

# ChainedSigCores = yaml.safe_load(open(Hatlab_QCoDes_Drivers.__path__[0]+"\\..\\DeviceInfo\\ChainedSigCores.yaml"))
DLLPATH = r"C:\Program Files\SignalCore\SC5511A\api\c\x64"


def search_5511(max_connection=20):
    """
    param max_connection: maximum number of connected device
    """
    dll_5511 = ctypes.CDLL(DLLPATH + "//sc5511a.dll")
    sn_buffers = [
        ctypes.create_string_buffer(8) for i in range(max_connection)
    ]  # buffer to store list of serial numbers
    pointers = (ctypes.c_char_p * max_connection)(
        *map(ctypes.addressof, sn_buffers)
    )  # pointer list for the buffers
    n_5511 = dll_5511.sc5511a_search_devices(ctypes.byref(pointers))
    pointer_list = []
    if n_5511 == 0:
        print("No SC5511A 20GHz-SigCire is connected")
    else:
        print("Following SC5511A 20GHz-SigCores are connected:")
        for i in range(n_5511):
            print(pointers[i])
            pointer_list.append(pointers[i].decode("utf-8"))
    return pointer_list


class Device_rf_params_t(ctypes.Structure):
    _fields_ = [
        ("rf1_freq", ctypes.c_ulonglong),
        ("start_freq", ctypes.c_ulonglong),
        ("stop_freq", ctypes.c_ulonglong),
        ("step_freq", ctypes.c_ulonglong),
        ("sweep_dwell_time", ctypes.c_uint),
        ("sweep_cycles", ctypes.c_uint),
        ("buffer_time", ctypes.c_uint),
        ("rf_level", ctypes.c_float),
        ("rf2_freq", ctypes.c_short),
    ]


class Device_temperature_t(ctypes.Structure):
    _fields_ = [("device_temp", ctypes.c_float)]


class Operate_status_t(ctypes.Structure):
    _fields_ = [
        ("rf1_lock_mode", ctypes.c_ubyte),
        ("rf1_loop_gain", ctypes.c_ubyte),
        ("device_access", ctypes.c_ubyte),
        ("rf2_standby", ctypes.c_ubyte),
        ("rf1_standby", ctypes.c_ubyte),
        ("auto_pwr_disable", ctypes.c_ubyte),
        ("alc_mode", ctypes.c_ubyte),
        ("rf1_out_enable", ctypes.c_ubyte),
        ("ext_ref_lock_enable", ctypes.c_ubyte),
        ("ext_ref_detect", ctypes.c_ubyte),
        ("ref_out_select", ctypes.c_ubyte),
        ("list_mode_running", ctypes.c_ubyte),
        ("rf1_mode", ctypes.c_ubyte),
        ("harmonic_ss", ctypes.c_ubyte),
        ("over_temp", ctypes.c_ubyte),
    ]


class Pll_status_t(ctypes.Structure):
    _fields_ = [
        ("sum_pll_ld", ctypes.c_ubyte),
        ("crs_pll_ld", ctypes.c_ubyte),
        ("fine_pll_ld", ctypes.c_ubyte),
        ("crs_ref_pll_ld", ctypes.c_ubyte),
        ("crs_aux_pll_ld", ctypes.c_ubyte),
        ("ref_100_pll_ld", ctypes.c_ubyte),
        ("ref_10_pll_ld", ctypes.c_ubyte),
        ("rf2_pll_ld", ctypes.c_ubyte),
    ]


class List_mode_t(ctypes.Structure):
    _fields_ = [
        ("sss_mode", ctypes.c_ubyte),
        ("sweep_dir", ctypes.c_ubyte),
        ("tri_waveform", ctypes.c_ubyte),
        ("hw_trigger", ctypes.c_ubyte),
        ("step_on_hw_trig", ctypes.c_ubyte),
        ("return_to_start", ctypes.c_ubyte),
        ("trig_out_enable", ctypes.c_ubyte),
        ("trig_out_on_cycle", ctypes.c_ubyte),
    ]


class Device_status_t(ctypes.Structure):
    _fields_ = [
        ("list_mode", List_mode_t),
        ("operate_status_t", Operate_status_t),
        ("pll_status_t", Pll_status_t),
    ]


class Device_info_t(ctypes.Structure):
    _fields_ = [
        ("serial_number", ctypes.c_uint32),
        ("hardware_revision", ctypes.c_float),
        ("firmware_revision", ctypes.c_float),
        ("manufacture_date", ctypes.c_uint32),
    ]


# End of Structures------------------------------------------------------------
class SignalCore_SC5511A(Instrument):
    def __init__(
        self, name: str, serial_number: str, dll=None, debug=False, **kwargs: Any
    ):
        super().__init__(name, **kwargs)
        logging.info(
            __name__
            + f" : Initializing instrument SignalCore generator {serial_number}"
        )
        if dll is not None:
            self._dll = dll
        else:
            self._dll = ctypes.CDLL(DLLPATH + "//sc5511a.dll")

        if debug:
            print(self._dll)

        self._dll.sc5511a_open_device.restype = ctypes.c_uint64
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(
                ctypes.c_char_p(bytes(serial_number, "utf-8"))
            )
        )
        self._serial_number = ctypes.c_char_p(bytes(serial_number, "utf-8"))
        self._rf_params = Device_rf_params_t(0, 0, 0, 0, 0, 0, 0, 0, 0)
        self._status = Operate_status_t(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self._open = False
        self._temperature = Device_temperature_t(0)

        self._pll_status = Pll_status_t()
        self._list_mode = List_mode_t()
        self._device_status = Device_status_t(
            self._list_mode, self._status, self._pll_status
        )
        if debug:
            print(serial_number, self._handle)
            self._dll.sc5511a_get_device_status(
                self._handle, ctypes.byref(self._device_status)
            )
            status = self._device_status.operate_status_t.rf1_out_enable
            print("check status", status)

        self._dll.sc5511a_close_device(self._handle)
        self._device_info = Device_info_t(0, 0, 0, 0)
        self.get_idn()
        self.do_set_auto_level_disable(
            0
        )  # setting this to 1 will lead to unstable output power

        # self.do_set_ref_out_freq(10)
        # if serial_number in ChainedSigCores["Modified_Ref_In"]:
        # self.do_set_ref_out_freq(100)
        # elif serial_number in ChainedSigCores["Origional_Ref_In"]:
        # self.do_set_ref_out_freq(10)

        self.add_parameter(
            "power",
            label="power",
            get_cmd=self.do_get_power,
            get_parser=float,
            set_cmd=self.do_set_power,
            set_parser=float,
            unit="dBm",
            vals=Numbers(min_value=-144, max_value=19),
        )

        self.add_parameter(
            "output_status",
            label="output_status",
            get_cmd=self.do_get_output_status,
            get_parser=int,
            set_cmd=self.do_set_output_status,
            set_parser=int,
            vals=Numbers(min_value=0, max_value=1),
        )

        self.add_parameter(
            "frequency",
            label="frequency",
            get_cmd=self.do_get_frequency,
            get_parser=float,
            set_cmd=self.do_set_frequency,
            set_parser=float,
            unit="Hz",
            vals=Numbers(min_value=0, max_value=20e9),
        )

        self.add_parameter(
            "reference_source",
            label="reference_source",
            get_cmd=self.do_get_reference_source,
            get_parser=int,
            set_cmd=self.do_set_reference_source,
            set_parser=int,
            vals=Numbers(min_value=0, max_value=1),
        )

        self.add_parameter(
            "auto_level_disable",
            label="0 = power is leveled on frequency change",
            get_cmd=self.do_get_auto_level_disable,
            get_parser=int,
            set_cmd=self.do_set_auto_level_disable,
            set_parser=int,
            vals=Numbers(min_value=0, max_value=1),
        )

        self.add_parameter(
            "ref_out_freq",
            label="reference out frequency in MHz",
            get_cmd=self.do_get_ref_out_freq,
            get_parser=int,
            set_cmd=self.do_set_ref_out_freq,
            set_parser=int,
            vals=Enum(10, 100),
        )

        self.add_parameter(
            "temperature",
            label="temperature",
            get_cmd=self.do_get_device_temp,
            get_parser=float,
            unit="C",
            vals=Numbers(min_value=0, max_value=200),
        )

        if self._device_status.operate_status_t.ext_ref_lock_enable == 0:
            self.do_set_reference_source(1)

    def set_open(self, open):
        if open and not self._open:
            self._handle = ctypes.c_void_p(
                self._dll.sc5511a_open_device(self._serial_number)
            )
            self._open = True
        elif not open and self._open:
            self._dll.sc5511a_close_device(self._handle)
            self._open = False
        return True

    def close(self):
        self.set_open(0)

    def get_device_status(self):
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_device_status(
            self._handle, ctypes.byref(self._device_status)
        )
        self._dll.sc5511a_close_device(self._handle)
        return self._device_status

    def do_set_output_status(self, enable):
        """
        Turns the output of RF1 on or off.
            Input:
                enable (int) = OFF = 0 ; ON = 1
        """
        logging.info(__name__ + " : Setting output to %s" % enable)
        c_enable = ctypes.c_ubyte(enable)
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        completed = self._dll.sc5511a_set_output(self._handle, c_enable)
        self._dll.sc5511a_close_device(self._handle)
        return completed

    def do_get_output_status(self):
        """
        Reads the output status of RF1
            Output:
                status (int) : OFF = 0 ; ON = 1
        """
        logging.info(__name__ + " : Getting output")
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_device_status(
            self._handle, ctypes.byref(self._device_status)
        )
        status = self._device_status.operate_status_t.rf1_out_enable
        self._dll.sc5511a_close_device(self._handle)
        return status

    def do_set_frequency(self, frequency):
        """
        Sets RF1 frequency. Valid between 100MHz and 20GHz
            Args:
                frequency (int) = frequency in Hz
        """
        c_freq = ctypes.c_ulonglong(int(frequency))
        logging.info(__name__ + " : Setting frequency to %s" % frequency)
        close = False
        if not self._open:
            self._handle = ctypes.c_void_p(
                self._dll.sc5511a_open_device(self._serial_number)
            )
            close = True
        if_set = self._dll.sc5511a_set_freq(self._handle, c_freq)
        if close:
            self._dll.sc5511a_close_device(self._handle)
        return if_set

    def do_get_frequency(self):
        logging.info(__name__ + " : Getting frequency")
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_rf_parameters(self._handle, ctypes.byref(self._rf_params))
        frequency = self._rf_params.rf1_freq
        self._dll.sc5511a_close_device(self._handle)
        return frequency

    def do_set_reference_source(self, lock_to_external):
        logging.info(__name__ + " : Setting reference source to %s" % lock_to_external)
        ref_out_freq = self.do_get_ref_out_freq()
        ref_out_sel = ctypes.c_ubyte(0 if ref_out_freq == 10 else 1)
        lock = ctypes.c_ubyte(lock_to_external)
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        # source = self._dll.sc5511a_set_clock_reference(self._handle, ref_out_sel, lock) #does not work for firmware > 3.6
        source = self._dll.sc5511a_set_clock_reference(
            self._handle, ref_out_sel, ref_out_sel, lock, lock
        )
        self._dll.sc5511a_close_device(self._handle)
        return source

    def do_set_standby(self, enable=False):
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        error_code = self._dll.sc5511a_set_standby(self._handle, enable)
        self._dll.sc5511a_close_device(self._handle)
        return error_code, enable

    def do_get_reference_source(self):
        logging.info(__name__ + " : Getting reference source")
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_device_status(
            self._handle, ctypes.byref(self._device_status)
        )
        enabled = (
            self._device_status.operate_status_t.ext_ref_lock_enable
            and self._device_status.operate_status_t.ext_ref_detect
        )
        self._dll.sc5511a_close_device(self._handle)
        return enabled

    def do_set_ref_out_freq(self, freq):
        logging.info(__name__ + " : Setting reference out freq to %s" % freq)
        if freq == 10:
            ref_out_sel = 0
        elif freq == 100:
            ref_out_sel = 1

        ref_out_sel = ctypes.c_ubyte(ref_out_sel)
        lock = ctypes.c_ubyte(self.do_get_reference_source())
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        source = self._dll.sc5511a_set_clock_reference(self._handle, ref_out_sel, lock)
        self._dll.sc5511a_close_device(self._handle)
        return source

    def do_get_ref_out_freq(self):
        logging.info(__name__ + " : Getting reference source")
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_device_status(
            self._handle, ctypes.byref(self._device_status)
        )
        ref_out_sel = self._device_status.operate_status_t.ref_out_select
        freq = 100 if ref_out_sel else 10
        self._dll.sc5511a_close_device(self._handle)
        return freq

    def do_set_power(self, power):
        logging.info(__name__ + " : Setting power to %s" % power)
        c_power = ctypes.c_float(power)
        close = False
        if not self._open:
            self._handle = ctypes.c_void_p(
                self._dll.sc5511a_open_device(self._serial_number)
            )
            close = True
        completed = self._dll.sc5511a_set_level(self._handle, c_power)
        if close:
            self._dll.sc5511a_close_device(self._handle)
        return completed

    def do_get_power(self):
        logging.info(__name__ + " : Getting Power")
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_rf_parameters(self._handle, ctypes.byref(self._rf_params))
        rf_level = self._rf_params.rf_level
        self._dll.sc5511a_close_device(self._handle)
        return rf_level

    def get_ext_ref_lock():
        # prints whether or not an external reference is detected
        error_code, OPERATE = SC1.get_operate_status()  # noqa: F821
        if OPERATE["ext_ref_detect"] == 0:
            print("There is no external reference detected")
        if OPERATE["ext_ref_detect"] == 1:
            print("An external reference has been detected")
        _error_handler(error_code)  # noqa: F821
        return OPERATE["ext_ref_detect"]

    def do_set_auto_level_disable(self, enable):
        logging.info(__name__ + " : Settingalc auto to %s" % enable)
        if enable == 1:
            enable = 0
        elif enable == 0:
            enable = 1
        c_enable = ctypes.c_ubyte(enable)
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        completed = self._dll.sc5511a_set_auto_level_disable(self._handle, c_enable)
        self._dll.sc5511a_close_device(self._handle)
        return completed

    def do_get_auto_level_disable(self):
        logging.info(__name__ + " : Getting alc auto status")
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_device_status(
            self._handle, ctypes.byref(self._device_status)
        )
        enabled = self._device_status.operate_status_t.auto_pwr_disable
        self._dll.sc5511a_close_device(self._handle)
        if enabled == 1:
            enabled = 0
        elif enabled == 0:
            enabled = 1
        return enabled

    def do_get_device_temp(self):
        logging.info(__name__ + " : Getting device temperature")
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_temperature(self._handle, ctypes.byref(self._temperature))
        device_temp = self._temperature.device_temp
        self._dll.sc5511a_close_device(self._handle)
        return device_temp

    def get_idn(self) -> dict[str, str | None]:
        logging.info(__name__ + " : Getting device info")
        self._handle = ctypes.c_void_p(
            self._dll.sc5511a_open_device(self._serial_number)
        )
        self._dll.sc5511a_get_device_info(self._handle, ctypes.byref(self._device_info))
        device_info = self._device_info
        self._dll.sc5511a_close_device(self._handle)

        def date_decode(date_int: int):
            date_str = f"{date_int:032b}"
            yr = f"20{int(date_str[:8], 2)}"
            month = f"{int(date_str[16:24], 2)}"
            day = f"{int(date_str[8:16], 2)}"
            return f"{month}/{day}/{yr}"

        IDN: dict[str, str | None] = {
            "vendor": "SignalCore",
            "model": "SC5511A",
            "serial_number": self._serial_number.value.decode("utf-8"),
            "firmware_revision": device_info.firmware_revision,
            "hardware_revision": device_info.hardware_revision,
            "manufacture_date": date_decode(device_info.manufacture_date),
        }
        return IDN


if __name__ == "__main__":
    SC3 = SignalCore_SC5511A("SC3", "10003C69")

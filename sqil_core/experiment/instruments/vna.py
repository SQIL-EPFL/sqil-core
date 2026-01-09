from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from numpy import exp
from qcodes.instrument import find_or_create_instrument

from sqil_core.config_log import logger
from sqil_core.experiment.instruments import Instrument

from .drivers.RohdeSchwarzZNB import RohdeSchwarzZNBBase, ZNBChannel

if TYPE_CHECKING:
    from numpy import ndarray


class VNA(Instrument, ABC):
    s_param: Literal["S11", "S12", "S21", "S22"] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        s_param = self.config.get("s_param", None)
        if s_param is None:
            raise KeyError(
                f"Missing s_param for {self.name}. "
                "Please specify either S11, S12, S21, or S22 in your setup file."
            )
        self.s_param = s_param

    def _default_on_before_experiment(self, *args, sender=None, **kwargs):
        bandwidth = self.get_variable("bandwidth", sender)
        averages = self.get_variable("averages", sender)
        if bandwidth:
            self.set_bandwidth(bandwidth)
        if averages:
            self.set_averages(averages)
        self.turn_on()

    def _default_on_before_sequence(self, *args, sender=None, **kwargs):
        freq_range = self.get_variable("frequency_range", sender)
        power = self.get_variable("power", sender)
        if freq_range:
            if not hasattr(freq_range, "__len__"):
                raise ValueError(
                    f"Invalid frequency range for {self.name}. "
                    f"Expected a tuple (start, stop, n_points), but got {freq_range}."
                )
            self.set_frequency_range(*freq_range)
        if power:
            self.set_power(power)

    def _default_on_after_experiment(self, *args, sender=None, **kwargs):
        self.turn_off()

    @abstractmethod
    def set_frequency_range(self, start, stop, n_points) -> None:
        pass

    @abstractmethod
    def set_power(self, value) -> None:
        pass

    @abstractmethod
    def set_bandwidth(self, value) -> None:
        pass

    @abstractmethod
    def set_averages(self, value) -> None:
        pass

    @abstractmethod
    def get_IQ_data(self) -> ndarray:
        pass

    def turn_on(self) -> None:
        logger.debug(f"Turning on {self.name}")
        self.instrument.turn_on()

    def turn_off(self) -> None:
        logger.debug(f"Turning off {self.name}")
        self.instrument.turn_off()


class SqilRohdeSchwarzZNA26(VNA):
    """
    Frequency:
        [10 MHz, 26.5 GHz]
    Power:
        [-150 dBm, 100 dBm]
    """

    def _default_connect(self, *args, **kwargs):
        logger.debug(f"Connecting to {self.name} ({self.model})")
        return find_or_create_instrument(
            RohdeSchwarzZNBBase,
            self.name,
            self.address,
            init_s_params=False,
            reset_channels=False,
        )

    def _default_disconnect(self, *args, **kwargs):
        logger.debug(f"Disconnecting from {self.name} ({self.model})")
        self.device.close()

    def _default_setup(self, *args, **kwargs):
        logger.debug(f"Setting up {self.name}")
        self.turn_off()
        logger.debug(" -> Turned off")

        chan = ZNBChannel(
            self.device,
            name="CHMEAS",
            channel=1,
            vna_parameter=self.s_param,
            existing_trace_to_bind_to="Trc1",
        )
        self.device.channels.append(chan)
        logger.debug(" -> Added channel CHMEAS")

        self.device.cont_meas_on()
        self.device.display_single_window()

        self.set_power(-60)
        logger.debug(" -> Power = -60 dBm")

    def set_frequency_range(self, start, stop, n_points) -> None:
        self.device.channels.CHMEAS.start(start)
        self.device.channels.CHMEAS.stop(stop)
        self.device.channels.CHMEAS.npts(n_points)

    def set_power(self, value) -> None:
        self.device.channels.CHMEAS.power(value)

    def set_bandwidth(self, value) -> None:
        self.device.channels.CHMEAS.bandwidth(value)

    def set_averages(self, value) -> None:
        self.device.channels.CHMEAS.avg(value)

    def get_IQ_data(self):
        mag_dB, phase = self.device.channels.CHMEAS.trace_db_phase.get()
        data = 10 ** (mag_dB / 20) * exp(1j * phase)
        return data

    def turn_on(self) -> None:
        self.device.rf_on()

    def turn_off(self) -> None:
        self.device.rf_off()


def vna_factory(id, config) -> VNA:
    device_class = None
    model = config.get("model", "")
    if model == "RohdeSchwarzZNA26":
        device_class = SqilRohdeSchwarzZNA26
    else:
        raise ValueError(f"Unsupported model: {model}")

    return device_class(id, config)

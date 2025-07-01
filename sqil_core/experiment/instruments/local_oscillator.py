from abc import ABC, abstractmethod

import yaml
from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzSGS100A
from qcodes_contrib_drivers.drivers.SignalCore.SignalCore import SC5521A

from sqil_core.config_log import logger
from sqil_core.experiment.instruments import Instrument
from sqil_core.utils._formatter import format_number

from .drivers.SignalCore_SC5511A import SignalCore_SC5511A


class LocalOscillatorBase(Instrument, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def set_frequency(self, value) -> None:
        """Set the frequency of the local oscillator."""
        pass

    @abstractmethod
    def set_power(self, value) -> None:
        """Set the power of the local oscillator."""
        pass

    @abstractmethod
    def turn_on(self) -> None:
        """Turn the local oscillator on."""
        pass

    @abstractmethod
    def turn_off(self) -> None:
        """Turn the local oscillator off."""
        pass


class SqilRohdeSchwarzSGS100A(LocalOscillatorBase):
    """
    Frequency:
        [1 MHz, 20 GHz], resolution 0.001 Hz
    Power:
        [-120 dB, 25 dB], resolution 0.01 dB
    """

    def _default_connect(self, *args, **kwargs):
        logger.info(f"Connecting to {self.name} ({self.model})")
        return RohdeSchwarzSGS100A(self.name, self.address)

    def _default_disconnect(self, *args, **kwargs):
        logger.info(f"Disconnecting from {self.name} ({self.model})")
        self.turn_off()

    def _default_setup(self, *args, **kwargs):
        logger.info(f"Setting up {self.name}")
        self.turn_off()
        self.set_power(-60)

    def set_frequency(self, value) -> None:
        self.device.frequency(value)

    def set_power(self, value) -> None:
        self.device.power(value)

    def turn_on(self) -> None:
        self.device.on()

    def turn_off(self) -> None:
        self.device.off()


class SqilSignalCoreSC5511A(LocalOscillatorBase):
    """
    PORT 1 specifications
    Frequency:
        [100 MHz, 20 GHz], resolution 1 Hz
    Power:
        @ freq < 18 GHz: [-20 dBm, 15 dBm], resolution 0.01 dBm
        @ freq > 18 GHz: [-20 dBm, 10 dBm], resolution 0.01 dBm
    """

    def _default_connect(self, *args, **kwargs):
        logger.info(f"Connecting to {self.name} ({self.model})")
        return SignalCore_SC5511A(self.name, self.address)

    def _default_disconnect(self, *args, **kwargs):
        logger.info(f"Disconnecting from {self.name} ({self.model})")
        self.turn_off()

    def _default_setup(self, *args, **kwargs):
        logger.info(f"Setting up {self.name}")
        self.turn_off()
        self.set_power(-40)
        self.device.do_set_reference_source(1)  # to enable phase locking
        self.device.do_set_standby(True)  # update PLL locking
        self.device.do_set_standby(False)

    def set_frequency(self, value) -> None:
        self.device.do_set_ref_out_freq(value)

    def set_power(self, value) -> None:
        self.device.power(value)

    def turn_on(self) -> None:
        self.device.do_set_output_status(1)

    def turn_off(self) -> None:
        self.device.do_set_output_status(0)


class SqilSignalCoreSC5521A(LocalOscillatorBase):
    """
    Frequency:
        [160 MHz, 40 GHz], resolution 1 Hz
    Power:
        @ freq < 30 GHz: [-10 dBm, 15 dBm], resolution 0.1 dBm
        @ freq < 35 GHz: [-10 dBm, 10 dBm], resolution 0.1 dBm
        @ freq > 35 GHz: [-10 dBm, 3 dBm], resolution 0.1 dBm
    """

    def _default_connect(self, *args, **kwargs):
        logger.info(f"Connecting to {self.name} ({self.model})")
        return SC5521A(self.name)

    def _default_disconnect(self, *args, **kwargs):
        logger.info(f"Disconnecting from {self.name} ({self.model})")
        self.turn_off()

    def _default_setup(self, *args, **kwargs):
        logger.info(f"Setting up {self.name}")
        self.turn_off()
        self.set_power(-40)

    def set_frequency(self, value) -> None:
        self.device.clock_frequency(value)

    def set_power(self, value) -> None:
        self.device.power(value)

    def turn_on(self) -> None:
        self.device.status("on")

    def turn_off(self) -> None:
        self.device.status("off")


class LocalOscillator(LocalOscillatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model = kwargs.get("config", {}).get("model", "")
        if model == "RohdeSchwarzSGS100A":
            self._device_class = SqilRohdeSchwarzSGS100A
        elif model == "SignalCore_SC5511A":
            self._device_class = SqilSignalCoreSC5511A
        elif model == "SC5521A":
            self._device_class = SqilSignalCoreSC5521A
        else:
            raise ValueError(f"Unsupported model: {model}")

        self.instrument = self._device_class(self.id, self.config)

    def _default_connect(self, *args, **kwargs):
        pass

    def _default_disconnect(self, *args, **kwargs):
        pass

    def _default_setup(self, *args, **kwargs):
        pass

    def _default_on_before_experiment(self, *args, sender=None, **kwargs):
        self.turn_on()

    def _default_on_before_sequence(self, *args, sender=None, **kwargs):
        freq = self.get_variable("frequency", sender)
        power = self.get_variable("power", sender)
        if freq:
            self.set_frequency(freq)
        if power:
            self.set_power(power)

    def _default_on_after_experiment(self, *args, sender=None, **kwargs):
        self.turn_off()

    def set_frequency(self, value) -> None:
        logger.info(
            f"Setting frequency to {format_number(value, 5, unit="Hz", latex=False)} for {self.name}"
        )
        self.instrument.set_frequency(value)

    def set_power(self, value) -> None:
        logger.info(
            f"Setting power to {format_number(value, 4, unit="dB", latex=False)} for {self.name}"
        )
        self.instrument.set_power(value)

    def turn_on(self) -> None:
        logger.info(f"Turning on {self.name}")
        self.instrument.turn_on()

    def turn_off(self) -> None:
        logger.info(f"Turning off {self.name}")
        self.instrument.turn_off()

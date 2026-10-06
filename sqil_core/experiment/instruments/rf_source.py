from abc import ABC, abstractmethod

from qcodes.instrument import find_or_create_instrument
from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzSGS100A

from sqil_core.config_log import logger
from sqil_core.experiment.instruments import Instrument

from .drivers.SignalCore_SC5511A import SignalCore_SC5511A


class RfSource(Instrument, ABC):
    def _default_on_before_sequence(self, *args, sender=None, **kwargs):
        freq = self.get_variable("frequency", sender)
        power = self.get_variable("power", sender)
        if freq:
            self.set_frequency(freq)
        if power:
            self.set_power(power)

    def _default_on_after_experiment(self, *args, sender=None, **kwargs):
        self.turn_off()

    @abstractmethod
    def set_frequency(self, value) -> None:
        pass

    @abstractmethod
    def set_power(self, value) -> None:
        pass

    def turn_on(self) -> None:
        logger.debug(f"Turning on {self.name}")
        self.instrument.turn_on()

    def turn_off(self) -> None:
        logger.debug(f"Turning off {self.name}")
        self.instrument.turn_off()


class SqilRohdeSchwarzSGS100A(RfSource):
    """
    Frequency:
        [1 MHz, 20 GHz], resolution 0.001 Hz
    Power:
        [-120 dB, 25 dBm], resolution 0.01 dB
    """

    def _default_connect(self, *args, **kwargs):
        logger.debug(f"Connecting to {self.name} ({self.model})")
        return find_or_create_instrument(RohdeSchwarzSGS100A, self.name, self.address)

    def _default_disconnect(self, *args, **kwargs):
        logger.debug(f"Disconnecting from {self.name} ({self.model})")
        self.device.close()

    def _default_setup(self, *args, **kwargs):
        logger.debug(f"Setting up {self.name}")
        self.turn_off()
        logger.debug(" -> Turned off")
        self.set_power(-60)
        logger.debug(" -> Power = -60 dBm")

    def set_frequency(self, value) -> None:
        self.device.frequency(value)

    def set_power(self, value) -> None:
        self.device.power(value)

    def turn_on(self) -> None:
        self.device.on()

    def turn_off(self) -> None:
        self.device.off()


class SqilSignalCoreSC5511A(RfSource):
    """
    PORT 1 specifications
    Frequency:
        [100 MHz, 20 GHz], resolution 1 Hz
    Power:
        @ freq < 18 GHz: [-20 dBm, 15 dBm], resolution 0.01 dBm
        @ freq > 18 GHz: [-20 dBm, 10 dBm], resolution 0.01 dBm
    """

    def _default_connect(self, *args, **kwargs):
        logger.debug(f"Connecting to {self.name} ({self.model})")
        return find_or_create_instrument(SignalCore_SC5511A, self.name, self.address)

    def _default_disconnect(self, *args, **kwargs):
        logger.debug(f"Disconnecting from {self.name} ({self.model})")
        self.turn_off()

    def _default_setup(self, *args, **kwargs):
        logger.debug(f"Setting up {self.name}")
        self.turn_off()
        logger.debug(" -> Turned off")
        self.set_power(-40)
        logger.debug(" -> Power = -40 dBm")
        # Phase locking loop
        self.device.do_set_reference_source(1)  # to enable phase locking
        logger.debug(" -> Activated phase locking loop")
        # Flip standby to update the PLL flag in the GUI
        self.device.do_set_standby(True)
        self.device.do_set_standby(False)
        logger.debug(" -> Updated GUI")

    def set_frequency(self, value) -> None:
        self.device.do_set_frequency(value)

    def set_power(self, value) -> None:
        self.device.power(value)

    def turn_on(self) -> None:
        self.device.do_set_output_status(1)

    def turn_off(self) -> None:
        self.device.do_set_output_status(0)


def rf_source_factory(id, config) -> RfSource:
    device_class = None
    model = config.get("model", "")
    if model == "RohdeSchwarzSGS100A":
        device_class = SqilRohdeSchwarzSGS100A
    elif model == "SignalCoreSC5511A":
        device_class = SqilSignalCoreSC5511A
    else:
        raise ValueError(f"Unsupported model: {model}")

    return device_class(id, config)

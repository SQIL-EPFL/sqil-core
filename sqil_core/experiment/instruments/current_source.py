from abc import ABC, abstractmethod

from qcodes.instrument import find_or_create_instrument

from sqil_core.config_log import logger
from sqil_core.experiment.instruments import Instrument
from sqil_core.experiment.instruments.drivers.Yokogawa_GS200 import YokogawaGS200
from sqil_core.utils._formatter import format_number


class CurrentSource(Instrument, ABC):
    ramp_step: float | None = None
    ramp_step_delay: float | None = None

    _default_ramp_step = 1e-6
    _default_ramp_step_delay = 8e-3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get additional parameters
        step = self.config.get("ramp_step", None)
        delay = self.config.get("ramp_step_delay", None)
        if step is None:
            logger.warning(
                f"No ramp step found for {self.name}, using default value of "
                f"{self._default_ramp_step}"
            )
        self.ramp_step = step or self._default_ramp_step
        if delay is None:
            logger.warning(
                f"No ramp step delay found for {self.name}, using default value of "
                f"{self._default_ramp_step_delay}"
            )
        self.ramp_step_delay = delay or self._default_ramp_step_delay

    def _default_on_before_experiment(self, *args, sender=None, **kwargs):
        # self.turn_on()
        pass

    def _default_on_before_sequence(self, *args, sender=None, **kwargs):
        # self.instrument = self._device_class(self.id, self.config)
        self.connect()
        # self.setup()
        current = self.get_variable("current", sender)
        if current is not None:
            self.ramp_current(current)
        self.disconnect()

    def _default_on_after_experiment(self, *args, sender=None, **kwargs):
        pass

    @abstractmethod
    def ramp_current(self, value, step=None, step_delay=None) -> None:
        pass

    def turn_on(self) -> None:
        logger.debug(f"Turning on {self.name}")
        self.instrument.turn_on()

    def turn_off(self) -> None:
        logger.debug(f"Turning off {self.name}")
        self.instrument.turn_off()

    def _wrap_step_and_delay(self, step, step_delay):
        step = step or self.ramp_step
        step_delay = step_delay or self.ramp_step_delay
        if step is None or step_delay is None:
            raise ValueError(
                f"Missing ramp_step ({step}) or ramp_step_delay ({step_delay}) "
                f"for {self.name}."
            )
        return step, step_delay


class SqilYokogawaGS200(CurrentSource):
    def _default_connect(self, *args, **kwargs):
        logger.debug(f"Connecting to {self.name} ({self.model})")
        return find_or_create_instrument(YokogawaGS200, self.name, self.address)

    def _default_disconnect(self, *args, **kwargs):
        logger.debug(f"Disconnecting from {self.name} ({self.model})")
        self.device.close()

    def _default_setup(self, *args, **kwargs):
        logger.debug(f"Setting up {self.name}")
        # Set current mode
        if self.device.source_mode() != "CURR":
            self.turn_off()
            logger.debug(" -> Source mode: current")
            self.device.source_mode("CURR")
        # Voltage limit
        v_lim = self.config.get("voltage_limit", 1)
        logger.debug(f" -> Voltage limit {v_lim} V")
        self.device.voltage_limit(v_lim)
        # Current range
        i_range = self.config.get("current_range", 1e-3)
        logger.debug(f" -> Current range {i_range} A")
        self.device.current_range(i_range)
        # Set current to 0, turn on and disconnect
        # self.ramp_current(0)
        self.turn_on()
        self.disconnect()

    def ramp_current(self, value, step=None, step_delay=None) -> None:
        step, step_delay = self._wrap_step_and_delay(step, step_delay)
        pretty_curr = format_number(value, 5, unit="A", latex=False)
        logger.debug(f"Ramping current to {pretty_curr} on {self.name}")
        self.device.ramp_current(value, step, step_delay)

    def turn_on(self) -> None:
        self.device.on()

    def turn_off(self) -> None:
        self.device.off()


def current_source_factory(id, config) -> CurrentSource:
    device_class = None
    model = config.get("model", "")
    if model == "Yokogawa_GS200":
        device_class = SqilYokogawaGS200
    else:
        raise ValueError(f"Unsupported model: {model}")

    return device_class(id, config)

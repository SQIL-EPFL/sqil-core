from abc import ABC, abstractmethod

import yaml
from qcodes.instrument_drivers.yokogawa import YokogawaGS200

from sqil_core.config_log import logger
from sqil_core.experiment.instruments import Instrument
from sqil_core.utils._formatter import format_number


class CurrentSourceBase(Instrument, ABC):
    ramp_step: float | None = None
    ramp_step_delay: float | None = None

    _default_ramp_step = 1e-6
    _default_ramp_step_delay = 8e-3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def ramp_current(self, value) -> None:
        """Ramp the current to the desired value."""
        pass


class SqilYokogawaGS200(CurrentSourceBase):
    def _default_connect(self, *args, **kwargs):
        logger.info(f"Connecting to {self.name} ({self.model})")
        return YokogawaGS200(self.name, self.address)

    def _default_disconnect(self, *args, **kwargs):
        logger.info(f"Disconnecting from {self.name} ({self.model})")
        self.device.close()

    def _default_setup(self, *args, **kwargs):
        logger.info(f"Setting up {self.name}")
        v_lim = self.config.get("voltage_limit", 5)
        i_range = self.config.get("current_range", 1e-3)
        logger.info(f" -> Voltage limit {v_lim} V")
        logger.info(f" -> Current range {i_range} A")
        self.device.voltage_limit(v_lim)
        self.device.current_range(i_range)

    def ramp_current(self, value, step, step_delay) -> None:
        self.device.ramp_current(value, step, step_delay)

    def turn_on(self) -> None:
        self.device.on()

    def turn_off(self) -> None:
        self.device.off()


class CurrentSoruce(CurrentSourceBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model = kwargs.get("config", {}).get("model", "")
        if model == "Yokogawa_GS200":
            self._device_class = SqilYokogawaGS200
        else:
            raise ValueError(f"Unsupported model: {model}")

        # Get additional parameters
        step = self.config.get("ramp_step", None)
        delay = self.config.get("ramp_step_dealy", None)
        if step is None:
            logger.warning(
                f"No ramp step found for {self.name}, using default value of {self._default_ramp_step}"
            )
        self.ramp_step = step or self._default_ramp_step
        if delay is None:
            logger.warning(
                f"No ramp step delay found for {self.name}, using default value of {self._default_ramp_step_delay}"
            )
        self.ramp_step_delay = delay or self._default_ramp_step_delay

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
        # self.instrument = self._device_class(self.id, self.config)
        self.connect()
        self.setup()
        current = self.get_variable("current", sender)
        if current is not None:
            self.ramp_current(current)
        self.disconnect()

    def _default_on_after_experiment(self, *args, sender=None, **kwargs):
        pass

    def ramp_current(self, value, step=None, step_delay=None) -> None:
        step = step or self.ramp_step
        step_delay = step_delay or self.ramp_step_delay
        if step is None or step_delay is None:
            raise ValueError(
                f"Missing ramp_step ({step}) or ramp_step_delay ({step_delay}) for {self.name}."
            )
        logger.info(
            f"Ramping current to {format_number(value, 5, unit="A", latex=False)} on {self.name}"
        )
        self.instrument.ramp_current(value, step, step_delay)

    def turn_on(self) -> None:
        logger.info(f"Turning on {self.name}")
        self.instrument.turn_on()

    def turn_off(self) -> None:
        logger.info(f"Turning off {self.name}")
        self.instrument.turn_off()

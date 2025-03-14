import yaml
from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzSGS100A
from qcodes_contrib_drivers.drivers.SignalCore.SignalCore import SC5521A

from sqil_core.config_log import logger
from sqil_core.experiment.lo_event_handler import lo_event_handlers

from .drivers.SignalCore_SC5511A import SignalCore_SC5511A


class LocalOscillator:
    # Class to unify the APIs of each local oscillator

    def __init__(self, instrument_id, config=None, config_path="setup.yaml"):
        """
        Initialize a local oscillator by ID from configuration.

        Args:
            instrument_id: ID of the instrument in the config file
            config: Configuration dictionary containing instrument settings.
                   Takes precedence over config_path.
            config_path: Path to the YAML configuration file
        """
        if config is None and config_path is not None:
            # Load configuration from file
            with open(config_path, "r") as file:
                file_config = yaml.safe_load(file)

            if (
                "instruments" not in file_config
                or instrument_id not in file_config["instruments"]
            ):
                raise ValueError(
                    f"Instrument '{instrument_id}' not found in config file"
                )

            config = file_config["instruments"][instrument_id]
        elif config is None:
            raise ValueError("Either config or config_path must be provided")

        self.id = instrument_id
        self.config = config

        if "name" not in config:
            raise ValueError(f"Missing 'name' for instrument '{instrument_id}'")
        self.name = config["name"]

        if "model" not in config:
            raise ValueError(f"Missing 'model' for instrument '{instrument_id}'")
        self.model = config["model"]

        if self.model != "SC5521A" and "address" not in config:
            raise ValueError(f"Missing 'address' for instrument '{instrument_id}'")

        self.address = config.get("address")  # May be None for SC5521A
        self.type = config.get("type")
        self.device = None

        # Connect automatically
        self.connect()

        lo_event_handlers.register_local_oscillator(self)

    def connect(self):
        logger.info(f"Connecting to {self.name} at {self.address}")
        if self.model == "RohdeSchwarzSGS100A":
            self.device = RohdeSchwarzSGS100A(self.name, self.address)

        elif self.model == "SignalCore_SC5511A":
            self.device = SignalCore_SC5511A(self.name, self.address)

        elif self.model == "SC5521A":
            self.device = SC5521A(self.name)

        else:
            raise ValueError(f"Unsupported instrument type: {self.model}")
        logger.info(f"Successfully connected to {self.name}")
        logger.debug("-> done")

    def setup(self, frequency=10):
        """
        Apply instrument-specific setup
        """
        logger.info(f"Setting up {self.name}")
        if self.model == "RohdeSchwarzSGS100A":
            self.device.status(False)
            self.device.power(-60)  # for safety

        elif self.model == "SignalCore_SC5511A":
            self.device.power(-40)  # for safety
            self.device.do_set_output_status(0)
            self.device.do_set_ref_out_freq(frequency)
            self.device.do_set_reference_source(1)  # to enable phase locking
            self.device.do_set_standby(True)  # update PLL locking
            self.device.do_set_standby(False)

        elif self.model == "SC5521A":
            self.device.status("off")
            self.device.power(-10)  # for safety
            self.device.clock_frequency(frequency)
        logger.debug("-> done")

    def power(self, value):
        logger.info(f"Changing {self.name} power to {value}")
        self.device.power(value)
        logger.debug("-> done")

    def frequency(self, value):
        logger.info(f"Changing {self.name} frequency to {value}")
        self.device.frequency(value)
        logger.debug("-> done")

    def on(self):
        logger.info(f"Turning {self.name} on")
        if self.model == "RohdeSchwarzSGS100A":
            self.device.on()
        elif self.model == "SignalCore_SC5511A":
            self.device.do_set_output_status(1)
        elif self.model == "SC5521A":
            self.device.status("on")
        logger.debug("-> done")

    def off(self):
        logger.info(f"Turning {self.name} off")
        if self.model == "RohdeSchwarzSGS100A":
            self.device.off()
        elif self.model == "SignalCore_SC5511A":
            self.device.do_set_output_status(0)
        elif self.model == "SC5521A":
            self.device.status("off")
        logger.debug("-> done")

    # these methods are for readability on call
    def unregister(self):
        lo_event_handlers.unregister_local_oscillator(self)

    def register(self):
        lo_event_handlers.register_local_oscillator(self)

    @classmethod
    def disable_all_auto_control(cls):
        lo_event_handlers.disable_auto_control()

    @classmethod
    def enable_all_auto_control(cls):
        lo_event_handlers.enable_auto_control()

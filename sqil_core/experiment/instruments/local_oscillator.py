import yaml
from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzSGS100A
from qcodes_contrib_drivers.drivers.SignalCore.SignalCore import SC5521A

from .drivers.SignalCore_SC5511A_locking_fix import *


class LocalOscillator:
    # Class to unify the APIs of each local oscillator

    def __init__(self, instrument_id, config_path="setup.yaml"):
        """
        Initialize a local oscillator by ID from configuration.

        Args:
            instrument_id: ID of the instrument in the config file
            config_path: Path to the YAML configuration file
        """
        # Load configuration
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if "instruments" not in config or instrument_id not in config["instruments"]:
            raise ValueError(f"Instrument '{instrument_id}' not found in config")

        self.id = instrument_id
        self.config = config["instruments"][instrument_id]
        self.name = self.config["name"]
        self.address = self.config["address"]
        self.model = self.config["model"]
        self.type = self.config["type"]
        self.device = None

        # Connect automatically
        self.connect()

    def connect(self):
        if self.model == "RohdeSchwarzSGS100A":
            self.device = RohdeSchwarzSGS100A(self.name, self.address)

        elif self.model == "SignalCore_SC5511A":
            self.device = SignalCore_SC5511A(self.name, self.address)

        elif self.model == "SC5521A":
            self.device = SC5521A(self.name)

        else:
            raise ValueError(f"Unsupported instrument type: {self.model}")

    def setup(self, frequency=10, turn_on=True):
        """
        Apply instrument-specific setup
        """
        if self.model == "RohdeSchwarzSGS100A":
            self.device.status(False)
            self.device.power(-60)  # for safety
            if turn_on:
                self.device.status(True)

        elif self.model == "SignalCore_SC5511A":
            self.device.power(-40)  # for safety
            self.device.do_set_output_status(0)
            self.device.do_set_ref_out_freq(frequency)
            self.device.do_set_reference_source(1)  # to enable phase locking
            self.device.do_set_standby(True)  # update PLL locking
            self.device.do_set_standby(False)
            if turn_on:
                self.device.do_set_output_status(1)

        elif self.model == "SC5521A":
            self.device.status("off")
            self.device.power(-10)  # for safety
            self.device.clock_frequency(frequency)
            if turn_on:
                self.device.status("on")

    def power(self, value):
        self.device.power(value)

    def frequency(self, value):
        self.device.frequency(value)

    def on(self):
        if self.model == "RohdeSchwarzSGS100A":
            self.device.on()
        elif self.model == "SignalCore_SC5511A":
            self.device.do_set_output_status(1)
        elif self.model == "SC5521A":
            self.device.status("on")

    def off(self):
        if self.model == "RohdeSchwarzSGS100A":
            self.device.off()
        elif self.model == "SignalCore_SC5511A":
            self.device.do_set_output_status(0)
        elif self.model == "SC5521A":
            self.device.status("off")

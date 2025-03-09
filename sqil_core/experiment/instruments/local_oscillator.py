import yaml


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
        self.type = self.config["type"]
        self.device = None

        # Connect automatically
        self._connect()

    def _connect(self):
        if self.type == "RohdeSchwarzSGS100A":
            from rohdeschwarz_sgs100a import RohdeSchwarzSGS100A

            self.device = RohdeSchwarzSGS100A(self.name, self.address)

        elif self.type == "SignalCore_SC5511A":
            from signalcore import SignalCore_SC5511A

            self.device = SignalCore_SC5511A(self.name, self.address)

        else:
            raise ValueError(f"Unsupported instrument type: {self.type}")

    def setup(self):
        """
        Apply instrument-specific setup
        """
        if self.type == "RohdeSchwarzSGS100A":
            # From image 1
            self.device.status(False)
            self.device.power(-60)  # for safety
            self.device.status(True)

        elif self.type == "SignalCore_SC5511A":
            # From image 1
            self.device.power(-40)  # for safety
            self.device.do_set_output_status(0)
            self.device.do_set_ref_out_freq(10)
            self.device.do_set_reference_source(1)  # to enable phase locking
            self.device.do_set_standby(True)  # update PLL locking
            self.device.do_set_standby(False)
            self.device.do_set_output_status(1)

    def power(self, value):
        self.device.power(value)

    def frequency(self, value):
        self.device.frequency(value)

    def on(self):
        if self.type == "RohdeSchwarzSGS100A":
            self.device.on()
        elif self.type == "SignalCore_SC5511A":
            self.device.do_set_output_status(1)

    def off(self):
        if self.type == "RohdeSchwarzSGS100A":
            self.device.off()
        elif self.type == "SignalCore_SC5511A":
            self.device.do_set_output_status(0)

    def close(self):
        if hasattr(self.device, "close"):
            self.device.close()

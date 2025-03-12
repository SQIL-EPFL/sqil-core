from experiment.instruments.local_oscillator import LocalOscillator

from sqil_core.config_log import logger
from sqil_core.measurement._events import after_experiment, before_experiment
from sqil_core.utils._read import read_yaml


class Instruments:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)


class Experiment:
    instruments: Instruments | None = None
    __setup_path = ""

    _instrument_factories = {
        "LO": LocalOscillator,
    }

    def __init__(
        self, params: dict = {}, param_dict_path: str = "", setup_path: str = ""
    ):
        self.__setup_path = setup_path

        # Read setup file
        if not setup_path:
            config = read_yaml("config.yaml")
            setup_path = read_yaml(config["setup_path"])
        setup = read_yaml(setup_path)

        # Load/connect instruments
        instrument_dict = setup.get("instruments", None)
        self._connect_instruments(instrument_dict)
        # Handle custom setup of instruments
        self.setup_instruments()

    def _connect_instruments(self, instrument_dict: dict | None):
        if not instrument_dict:
            logger.warning(
                f"Unable to find any instrument in {self.__setup_path}"
                + "Do you have an `instruments` entry in your setup file?"
            )
            self.instruments = Instruments({})
            return

        instance_dict = {}
        for instrument_id, config in instrument_dict.items():
            instrument_type = config.get("type")
            instrument_factory = self._instrument_factories.get(instrument_type)

            if not instrument_factory:
                logger.warning(
                    f"Unknown instrument type '{instrument_type}' for '{instrument_id}'. "
                    f"Available types: {list(self._instrument_factories.keys())}"
                )
                continue

            try:
                instance = instrument_factory(instrument_id, config=config)
                instance_dict[instrument_id] = instance
                logger.info(
                    f"Successfully connected to {config.get('name', instrument_id)}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to connect to {config.get('name', instrument_id)}: {str(e)}"
                )
        self.instruments = Instruments(instance_dict)

    def setup_instruments(self):
        """Custom instrument setup defined by the user"""
        pass

    def sequence(self):
        """Experimental sequence defined by the user"""
        pass

    def run(self):
        before_experiment.send()
        self.sequence()
        after_experiment.send()

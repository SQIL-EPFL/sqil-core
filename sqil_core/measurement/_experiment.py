from sqil_core.config_log import logger
from sqil_core.utils._read import read_yaml


class Instruments:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)


class Experiment:
    instruments: Instruments | None = None
    __setup_path = ""

    def __init__(
        self, params: dict = {}, param_dict_path: str = "", setup_path: str = ""
    ):
        self.__setup_path = setup_path

        # Read setup file
        if not setup_path:
            config = read_yaml("config.yaml")
            setup_path = read_yaml(config["setup_path"])
        setup = read_yaml(setup_path)

        instrument_dict = setup.get("instruments", None)
        self._connect_instruments(instrument_dict)

    def _connect_instruments(self, instrument_dict: dict | None):
        if not instrument_dict:
            logger.warning(
                f"Unable to find any instrument in {self.__setup_path}"
                + "Do you have an `instruments` entry in your setup file?"
            )
            self.instruments = Instruments({})
            return

        # TODO: create the right instance for each instrument and map it by name
        # in instance dict. Currently (and maybe ever) we have only 1 type, LocalOscillator
        instance_dict = {}
        self.instruments = Instruments(instance_dict)

    def setup_instruments(self):
        pass

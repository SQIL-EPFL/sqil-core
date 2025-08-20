from laboneq.dsl.quantum import QPU
from laboneq.simple import DeviceSetup

from sqil_core.experiment.instruments import Instrument


class ZI_Instrument(Instrument):
    _descriptor = ""
    _generate_setup = None
    _generate_qpu = None

    def __init__(self, id, config):
        super().__init__(id, config)
        self._descriptor = config.get("descriptor", "")

        self._generate_setup = config.get("generate_setup", None)
        if not self._generate_setup:
            raise NotImplementedError(
                "get_setup is not implemented in your setup file.\n"
                + "You should define it as part of the zi section of your instruments "
                "dictionary.\n" + "instruments['zi']['generate_setup']"
            )

        self._generate_qpu = config.get("generate_qpu", None)
        if not self._generate_qpu:
            raise NotImplementedError(
                "get_qpu is not implemented in your setup file.\n"
                + "You should define it as part of the zi section of your instruments "
                "dictionary.\n" + "instruments['zi']['generate_qpu']"
            )

    def generate_setup(self, *params, **kwargs) -> DeviceSetup:
        return self._generate_setup(*params, **kwargs)

    def generate_qpu(self, *params, **kwargs) -> QPU:
        return self._generate_qpu(*params, **kwargs)

    def _default_connect(self):
        pass
        # setup = self.config.get("setup_obj", None)
        # if setup is not None:
        #     self._session = Session(setup)
        #     return self.session
        # raise "Zuirch instruments needs a 'setup_obj' field in your setup file"

    def _default_setup(self):
        pass

    def _default_disconnect(self):
        pass

    @property
    def descriptor(self):
        """LaboneQ descriptor (read-only) - deprecated."""
        return self._descriptor

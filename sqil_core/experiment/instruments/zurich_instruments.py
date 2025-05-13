from laboneq.simple import Session

from sqil_core.experiment.instruments import Instrument


class ZI_Instrument(Instrument):
    _descriptor = ""
    _get_qpu = lambda: None

    def __init__(self, id, config):
        super().__init__(id, config)
        self._descriptor = config.get("descriptor", "")
        self._get_qpu = config.get("get_qpu", None)
        if not self._get_qpu:
            raise NotImplementedError(
                "get_qpu is not implemented in your setup file.\n"
                + "You should define it as part of the zi section of your instruments dictionary.\n"
                + "instruments['zi']['get_qpu']"
            )

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
        """LaboneQ descriptor (read-only)."""
        return self._descriptor

    @property
    def get_qpu(self):
        """Get QPU (read-only)."""
        return self._get_qpu

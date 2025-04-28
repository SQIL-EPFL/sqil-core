from laboneq.simple import Session

from sqil_core.experiment.instruments import Instrument


class ZI_Instrument(Instrument):

    def _default_connect(self):
        setup = self.config.get("setup_obj", None)
        if setup is not None:
            return Session(setup)

        raise "Zuirch instruments needs a 'setup_obj' field in your setup file"

    def _default_setup(self):
        pass

    def _default_disconnect(self):
        pass

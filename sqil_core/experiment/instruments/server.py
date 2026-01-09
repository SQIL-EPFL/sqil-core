import os
import sys
from typing import cast

import Pyro5.api as pyro
import Pyro5.errors as pyro_errors

from sqil_core.config_log import logger
from sqil_core.experiment.instruments import Instrument
from sqil_core.experiment.instruments.current_source import current_source_factory
from sqil_core.experiment.instruments.local_oscillator import LocalOscillator
from sqil_core.experiment.instruments.rf_source import rf_source_factory
from sqil_core.experiment.instruments.vna import vna_factory
from sqil_core.experiment.instruments.zurich_instruments import ZI_Instrument
from sqil_core.utils._read import read_yaml
from sqil_core.utils._utils import _extract_variables_from_module, _hash_file

_instrument_classes = {
    "LO": LocalOscillator,
    "ZI": ZI_Instrument,
    "CS": current_source_factory,
    "RF": rf_source_factory,
    "VNA": vna_factory,
}


@pyro.expose
class InstrumentServer:
    """
    Instruments server. Configures the instruments once and distributes instrument
    instances to other modules. Providing the path of the setup file is not required,
    but it allows to detect changes in the setup file that require the server to restart
    The server is available at PYRO:SERVER@localhost:9090.
    """

    NAME = "SERVER"
    PORT = 9090
    URI = f"PYRO:{NAME}@localhost:{PORT}"

    def __init__(self, instrument_dict: dict, setup_path="") -> None:
        logger.info("Starting server")
        self._instruments = connect_instruments(instrument_dict)
        self._daemon = pyro.Daemon(port=InstrumentServer.PORT)
        self._services: dict[str, pyro.URI] = {}

        self._setup_path = os.path.abspath(setup_path)
        self._setup_hash = None
        if setup_path:
            self._setup_hash = _hash_file(self._setup_path)
            logger.info(f"Setup path: {self._setup_path}")
            logger.info(f"Setup hash: {self.setup_hash!s}")

    def serve(self) -> None:
        self._expose()
        for id, instrument in self._instruments.items():
            uri = self._daemon.register(instrument, objectId=instrument.id)
            self._services[id] = uri
            logger.info(f"Registered {instrument = } with daemon at {uri = }.")
        self._daemon.register(self, objectId=InstrumentServer.NAME)
        with self._daemon:
            logger.info("Local server setup complete! Now listening for requests...")
            self._daemon.requestLoop()

    def _expose(self) -> None:
        classes = {instrument.__class__ for _, instrument in self._instruments.items()}
        # classes |= {Session}
        for cls in classes:
            pyro.expose(cls)
            logger.info(f"Exposed class {cls} to Pyro5.")

    def shutdown(self) -> None:
        logger.info("Shutting down the local server...")
        self._disconnect_all()
        with self._daemon:
            self._daemon.shutdown()
        logger.info("Local server shutdown complete!")

    def _disconnect_all(self) -> None:
        logger.info("Disonnecting instruments...")
        for _, instrument in self._instruments.items():
            instrument.disconnect()

    @property
    def services(self) -> dict[str, pyro.URI]:
        return {**self._services}

    @property
    def setup_hash(self) -> str | None:
        return self._setup_hash

    @property
    def setup_path(self) -> str:
        return self._setup_path


def link_instrument_server() -> tuple[pyro.Proxy, dict[str, pyro.Proxy]]:
    """Link to the instruments server."""
    server = pyro.Proxy(InstrumentServer.URI)

    if server.setup_path:
        current_hash = _hash_file(server.setup_path)
        print(server.setup_hash, current_hash)
        if current_hash != server.setup_hash:
            message = (
                f"Changes detected in the setup file {server.setup_path}. "
                + "Please restart the server to apply the changes."
            )
            logger.error(message)
            raise Exception(message)

    try:
        services = cast(dict[str, pyro.URI], server.services)
    except pyro_errors.CommunicationError as err:
        logger.error(f"Local server not found at {InstrumentServer.URI}")
        raise err from None
    else:
        instruments = {id: pyro.Proxy(uri) for id, uri in services.items()}
        return (server, instruments)


def unlink_instrument_server(server: pyro.Proxy, **instruments: pyro.Proxy) -> None:
    """Unlink from the instruments server."""
    server._pyroRelease()
    for _, instrument in instruments.items():
        instrument._pyroRelease()


def start_instrument_server(setup_path: str = ""):
    """Start a new instruments server using the provided setup file.
    If the path to the setup file is not provided it will be guessedby readig
    ./config.yaml.
    """
    if not setup_path:
        config = read_yaml("config.yaml")
        setup_path = config.get("setup_path", "setup.py")
    setup = _extract_variables_from_module("setup", setup_path)

    instrument_dict = setup.get("instruments", None)
    if not instrument_dict:
        logger.warning(
            f"Unable to find any instruments in {setup_path}"
            + "Do you have an `instruments` entry in your setup file?"
        )
        return None

    server = InstrumentServer(instrument_dict, setup_path=setup_path)
    server.serve()

    return server


def connect_instruments(instrument_dict: dict | None) -> dict[str, Instrument]:
    if not instrument_dict:
        return {}

    instance_dict = {}
    for instrument_id, config in instrument_dict.items():
        instrument_type = config.get("type")
        instrument_factory = _instrument_classes.get(instrument_type)

        if not instrument_factory:
            logger.warning(
                f"Unknown instrument type '{instrument_type}' for '{instrument_id}'. "
                f"Available types: {list(_instrument_classes.keys())}"
            )
            continue

        try:
            instance = instrument_factory(instrument_id, config=config)
            instance_dict[instrument_id] = instance
            logger.debug(
                f"Successfully connected to {config.get('name', instrument_id)}"
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to {config.get('name', instrument_id)}: {e!s}"
            )
            sys.exit(-1)
    return instance_dict

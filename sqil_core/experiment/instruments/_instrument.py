from abc import ABC, abstractmethod

from sqil_core.experiment.helpers._function_override_handler import (
    FunctionOverrideHandler,
)


class Instrument(FunctionOverrideHandler, ABC):
    """
    Base class for instruments with configurable behavior.

    Supports overriding `connect`, `setup`, and `disconnect` methods
    via a configuration dictionary.
    """

    def __init__(self, id: str, config: dict):
        """
        Initializes the instrument with an ID and configuration.

        If `connect`, `setup`, or `disconnect` are provided in `config`,
        they override the default implementations.
        """
        super().__init__()

        self._id = id
        self._type = config.get("type", "")
        self._model = config.get("model", "")
        self._name = config.get("name", "")
        self._address = config.get("address", "")
        self._config = config

        self._default_functions = {
            "connect": self._default_connect,
            "setup": self._default_setup,
            "disconnect": self._default_disconnect,
        }
        self._functions = self._default_functions.copy()

        # Override functions if provided in config
        for method_name in self._default_functions:
            if method := config.get(method_name):
                self.override_function(method_name, method)

        self._default_functions = self._functions.copy()
        self.connect()  # Auto-connect on instantiation

    def connect(self, *args, **kwargs):
        """Calls the overridden or default `connect` method."""
        return self.call("connect", *args, **kwargs)

    @abstractmethod
    def _default_connect(self, *args, **kwargs):
        """Default `connect` implementation (must be overridden)."""
        pass

    def setup(self, *args, **kwargs):
        """Calls the overridden or default `setup` method."""
        return self.call("setup", *args, **kwargs)

    @abstractmethod
    def _default_setup(self, *args, **kwargs):
        """Default `setup` implementation (must be overridden)."""
        pass

    def disconnect(self, *args, **kwargs):
        """Calls the overridden or default `disconnect` method."""
        return self.call("disconnect", *args, **kwargs)

    @abstractmethod
    def _default_disconnect(self, *args, **kwargs):
        """Default `disconnect` implementation (must be overridden)."""
        pass

    @property
    def id(self):
        """Instrument ID (read-only)."""
        return self._id

    @property
    def type(self):
        """Instrument type (read-only)."""
        return self._type

    @property
    def model(self):
        """Instrument model (read-only)."""
        return self._model

    @property
    def name(self):
        """Instrument name (read-only)."""
        return self._name

    @property
    def address(self):
        """Instrument address (read-only)."""
        return self._address

    @property
    def config(self):
        """Instrument configuration dictionary (read-only)."""
        return self._config

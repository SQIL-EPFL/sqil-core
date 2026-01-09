from abc import ABC, abstractmethod

import Pyro5.server

from sqil_core.experiment._events import (
    after_experiment,
    after_sequence,
    before_experiment,
    before_sequence,
)
from sqil_core.experiment.helpers._function_override_handler import (
    FunctionOverrideHandler,
)


@Pyro5.server.expose
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
        self._variables = config.get("variables", {})
        self._config = config
        self._device = None

        self._default_functions = {
            "connect": self._default_connect,
            "setup": self._default_setup,
            "disconnect": self._default_disconnect,
            "on_before_experiment": self._default_on_before_experiment,
            "on_after_experiment": self._default_on_after_experiment,
            "on_before_sequence": self._default_on_before_sequence,
            "on_after_sequence": self._default_on_after_sequence,
        }
        self._functions = self._default_functions.copy()

        # Override functions if provided in config
        for method_name in self._default_functions:
            if method := config.get(method_name):
                self.override_function(method_name, method)

        self._default_functions = self._functions.copy()
        self._device = self.connect()  # Auto-connect on instantiation

        # Subscribe to events
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        before_experiment.connect(
            lambda *a, **kw: self.call("on_before_experiment", *a, **kw), weak=False
        )
        before_sequence.connect(
            lambda *a, **kw: self.call("on_before_sequence", *a, **kw), weak=False
        )
        after_sequence.connect(
            lambda *a, **kw: self.call("on_after_sequence", *a, **kw), weak=False
        )
        after_experiment.connect(
            lambda *a, **kw: self.call("on_after_experiment", *a, **kw), weak=False
        )

    def connect(self, *args, **kwargs):
        """Calls the overridden or default `connect` method."""
        self._device = self.call("connect", *args, **kwargs)
        return self._device

    @abstractmethod
    def _default_connect(self, *args, **kwargs):
        """Default `connect` implementation (must be overridden)."""

    def setup(self, *args, **kwargs):
        """Calls the overridden or default `setup` method."""
        return self.call("setup", *args, **kwargs)

    @abstractmethod
    def _default_setup(self, *args, **kwargs):
        """Default `setup` implementation (must be overridden)."""

    def disconnect(self, *args, **kwargs):
        """Calls the overridden or default `disconnect` method."""
        return self.call("disconnect", *args, **kwargs)

    @abstractmethod
    def _default_disconnect(self, *args, **kwargs):
        pass

    def on_before_experiment(self, *args, **kwargs):
        """Calls the overridden or default `on_before_experiment` method."""
        return self.call("on_before_experiment", *args, **kwargs)

    def _default_on_before_experiment(self, *args, **kwargs):
        pass

    def on_before_sequence(self, *args, **kwargs):
        """Calls the overridden or default `on_before_sequence` method."""
        return self.call("on_before_sequence", *args, **kwargs)

    def _default_on_before_sequence(self, *args, **kwargs):
        pass

    def on_after_experiment(self, *args, **kwargs):
        """Calls the overridden or default `on_after_experiment` method."""
        return self.call("on_after_experiment", *args, **kwargs)

    def _default_on_after_experiment(self, *args, **kwargs):
        pass

    def on_after_sequence(self, *args, **kwargs):
        """Calls the overridden or default `on_after_sequence` method."""
        return self.call("on_after_sequence", *args, **kwargs)

    def _default_on_after_sequence(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        """
        Dynamically expose all attributes to Pyro server.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def get_variable(self, key, *args, **kwargs):
        var = self._variables.get(key, None)
        if callable(var):
            var = var(*args, **kwargs)
        return var

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
    def variables(self):
        """Instrument variables (read-only)."""
        return self._variables

    @property
    def config(self):
        """Instrument configuration dictionary (read-only)."""
        return self._config

    @property
    def device(self):
        """Raw instrument instance (read-only)."""
        return self._device

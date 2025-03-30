import pytest

from sqil_core.experiment import Instrument


class GoogooMeter(Instrument):
    def _default_connect(self, *args, **kwargs):
        return "Class default connect"

    def _default_setup(self, *args, **kwargs):
        return "Class default setup"

    def _default_disconnect(self, *args, **kwargs):
        return "Class default disconnect"

    def set_goo(self):
        return "Class default goo"


def custom_setup(self):
    return "Custom default setup"


def more_custom_setup(self):
    return "More custom setup"


CONFIG = {
    "type": "googoo",
    "model": "XYZ123",
    "name": "goo123",
    "address": "192.168.1.1",
}


class TestInstrument:

    def test_should_instantiate_instrument_with_correct_properties(self):
        instrument = GoogooMeter(id="1", config=CONFIG)

        assert instrument.id == "1"
        assert instrument.type == "googoo"
        assert instrument.model == "XYZ123"
        assert instrument.name == "goo123"
        assert instrument.address == "192.168.1.1"
        assert instrument.config == CONFIG

    def test_properties_are_readonly(self):
        instrument = GoogooMeter(id="1", config=CONFIG)
        with pytest.raises(AttributeError):
            instrument.id = "3"
        with pytest.raises(AttributeError):
            instrument.type = "idk"
        with pytest.raises(AttributeError):
            instrument.model = "12"
        with pytest.raises(AttributeError):
            instrument.name = "jacob"
        with pytest.raises(AttributeError):
            instrument.address = "lausanne"
        with pytest.raises(AttributeError):
            instrument.config = {"id": 7}

    def test_should_override_functions_from_config(self):
        control = GoogooMeter(id="0", config=CONFIG)
        assert control.setup() == "Class default setup"

        config = {**CONFIG, "setup": custom_setup}
        instrument = GoogooMeter(id="1", config=config)
        assert instrument.setup() == "Custom default setup"

    def test_should_override_from_config_all_allowed_functions(self):
        """The functions that can be overridden are: connect, setup, and disconnect."""
        control = GoogooMeter(id="0", config=CONFIG)
        assert control.setup() == "Class default setup"

        config = {
            **CONFIG,
            "setup": custom_setup,
            "connect": lambda self: "Custom default connect",
            "disconnect": lambda self: "Custom default disconnect",
        }
        instrument = GoogooMeter(id="1", config=config)
        assert instrument.connect() == "Custom default connect"
        assert instrument.setup() == "Custom default setup"
        assert instrument.disconnect() == "Custom default disconnect"

    def test_should_not_override_not_allowed_functions(self):
        """The functions that can be overridden are: connect, setup, and disconnect."""
        control = GoogooMeter(id="0", config=CONFIG)
        assert control.setup() == "Class default setup"

        config = {
            **CONFIG,
            "set_goo": lambda self: "Custom default goo",
        }
        instrument = GoogooMeter(id="1", config=config)
        assert instrument.set_goo() == "Class default goo"

    def test_should_allow_overriding_functions(self):
        control = GoogooMeter(id="0", config={**CONFIG, "setup": custom_setup})
        assert control.setup() == "Custom default setup"

        instrument = GoogooMeter(id="1", config={**CONFIG, "setup": custom_setup})
        instrument.override_function("setup", more_custom_setup)
        assert instrument.setup() == "More custom setup"

    def test_should_restore_function_from_config(self):
        config = {**CONFIG, "setup": custom_setup}
        instrument = GoogooMeter(id="1", config=config)

        instrument.override_function("setup", more_custom_setup)
        assert instrument.setup() == "More custom setup"
        instrument.restore_function("setup")
        assert instrument.setup() == "Custom default setup"

    def test_should_restore_function_to_class_default_if_not_included_in_config(self):
        instrument = GoogooMeter(id="1", config=CONFIG)

        instrument.override_function("setup", more_custom_setup)
        assert instrument.setup() == "More custom setup"
        instrument.restore_function("setup")
        assert instrument.setup() == "Class default setup"


# Running the tests
pytest.main()

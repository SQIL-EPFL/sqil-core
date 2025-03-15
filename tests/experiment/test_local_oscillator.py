import unittest
from unittest.mock import MagicMock, mock_open, patch

import yaml

from sqil_core.experiment.instruments.local_oscillator import LocalOscillator


class TestLocalOscillator(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "name": "test_lo",
            "model": "RohdeSchwarzSGS100A",
            "address": "TCPIP0::1.2.3.4::inst0::INSTR",
            "type": "LO",
        }

        self.logger_patch = patch("sqil_core.config_log.logger")
        self.event_handler_patch = patch(
            "sqil_core.experiment.lo_event_handler.lo_event_handlers"
        )
        self.rohde_patch = patch(
            "qcodes.instrument_drivers.rohde_schwarz.RohdeSchwarzSGS100A"
        )
        self.sc5511a_patch = patch(
            "experiment.instruments.drivers.SignalCore_SC5511A.SignalCore_SC5511A"
        )
        self.sc5521a_patch = patch(
            "qcodes_contrib_drivers.drivers.SignalCore.SignalCore.SC5521A"
        )

        self.mock_logger = self.logger_patch.start()
        self.mock_event_handler = self.event_handler_patch.start()
        self.mock_rohde = self.rohde_patch.start()
        self.mock_sc5511a = self.sc5511a_patch.start()
        self.mock_sc5521a = self.sc5521a_patch.start()

        self.mock_device = MagicMock()
        self.mock_rohde.return_value = self.mock_device
        self.mock_sc5511a.return_value = self.mock_device
        self.mock_sc5521a.return_value = self.mock_device

    def tearDown(self):
        self.logger_patch.stop()
        self.event_handler_patch.stop()
        self.rohde_patch.stop()
        self.sc5511a_patch.stop()
        self.sc5521a_patch.stop()

    def test_init_should_use_explicit_config_when_provided(self):
        lo = LocalOscillator("test_id", config=self.mock_config, config_path=None)

        self.assertEqual(lo.id, "test_id")
        self.assertEqual(lo.name, "test_lo")
        self.assertEqual(lo.model, "RohdeSchwarzSGS100A")
        self.assertEqual(lo.address, "TCPIP0::1.2.3.4::inst0::INSTR")
        self.assertEqual(lo.type, "LO")

        self.mock_rohde.assert_called_once_with(
            "test_lo", "TCPIP0::1.2.3.4::inst0::INSTR"
        )

        self.mock_event_handler.register_local_oscillator.assert_called_once_with(lo)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=yaml.dump(
            {
                "instruments": {
                    "test_id": {
                        "name": "yaml_lo",
                        "model": "RohdeSchwarzSGS100A",
                        "address": "TCPIP0::5.6.7.8::inst0::INSTR",
                        "type": "LO",
                    }
                }
            }
        ),
    )
    def test_init_should_load_config_from_yaml_when_config_not_provided(
        self, mock_file
    ):
        lo = LocalOscillator("test_id", config=None, config_path="fake_path.yaml")

        self.assertEqual(lo.name, "yaml_lo")
        self.assertEqual(lo.address, "TCPIP0::5.6.7.8::inst0::INSTR")

        mock_file.assert_called_once_with("fake_path.yaml", "r")

    def test_init_should_raise_value_error_when_missing_name(self):
        config_without_name = self.mock_config.copy()
        del config_without_name["name"]

        with self.assertRaises(ValueError) as context:
            LocalOscillator("test_id", config=config_without_name, config_path=None)

        self.assertIn("Missing 'name'", str(context.exception))

    def test_init_should_raise_value_error_when_missing_model(self):
        config_without_model = self.mock_config.copy()
        del config_without_model["model"]

        with self.assertRaises(ValueError) as context:
            LocalOscillator("test_id", config=config_without_model, config_path=None)

        self.assertIn("Missing 'model'", str(context.exception))

    def test_init_should_raise_value_error_when_missing_address_for_non_sc5521a(self):
        config_without_address = self.mock_config.copy()
        del config_without_address["address"]

        with self.assertRaises(ValueError) as context:
            LocalOscillator("test_id", config=config_without_address, config_path=None)

        self.assertIn("Missing 'address'", str(context.exception))

    def test_connect_should_instantiate_correct_driver_for_rohdeschwarz(self):
        lo = LocalOscillator("test_id", config=self.mock_config, config_path=None)

        self.mock_rohde.reset_mock()

        lo.connect()

        self.mock_rohde.assert_called_once_with(
            "test_lo", "TCPIP0::1.2.3.4::inst0::INSTR"
        )

    def test_connect_should_instantiate_correct_driver_for_signalcore_sc5511a(self):
        sc5511a_config = self.mock_config.copy()
        sc5511a_config["model"] = "SignalCore_SC5511A"

        lo = LocalOscillator("test_id", config=sc5511a_config, config_path=None)

        self.mock_sc5511a.reset_mock()

        lo.connect()

        self.mock_sc5511a.assert_called_once_with(
            "test_lo", "TCPIP0::1.2.3.4::inst0::INSTR"
        )

    def test_connect_should_instantiate_correct_driver_for_sc5521a(self):
        sc5521a_config = self.mock_config.copy()
        sc5521a_config["model"] = "SC5521A"
        sc5521a_config["address"] = None

        lo = LocalOscillator("test_id", config=sc5521a_config, config_path=None)

        self.mock_sc5521a.reset_mock()

        lo.connect()

        self.mock_sc5521a.assert_called_once_with("test_lo")

    def test_connect_should_raise_value_error_for_unsupported_model(self):
        invalid_model_config = self.mock_config.copy()
        invalid_model_config["model"] = "UnsupportedModel"

        lo = LocalOscillator("test_id", config=invalid_model_config, config_path=None)

        with self.assertRaises(ValueError) as context:
            lo.connect()

        self.assertIn("Unsupported instrument type", str(context.exception))

    def test_setup_should_configure_rohdeschwarz_correctly(self):
        lo = LocalOscillator("test_id", config=self.mock_config, config_path=None)

        lo.setup(frequency=10)

        self.mock_device.status.assert_called_once_with(False)
        self.mock_device.power.assert_called_once_with(-60)

    def test_setup_should_configure_signalcore_sc5511a_correctly(self):
        sc5511a_config = self.mock_config.copy()
        sc5511a_config["model"] = "SignalCore_SC5511A"

        lo = LocalOscillator("test_id", config=sc5511a_config, config_path=None)

        lo.setup(frequency=20)

        self.mock_device.power.assert_called_once_with(-40)
        self.mock_device.do_set_output_status.assert_called_once_with(0)
        self.mock_device.do_set_ref_out_freq.assert_called_once_with(20)
        self.mock_device.do_set_reference_source.assert_called_once_with(1)
        self.mock_device.do_set_standby.assert_any_call(True)
        self.mock_device.do_set_standby.assert_any_call(False)

    def test_setup_should_configure_sc5521a_correctly(self):
        sc5521a_config = self.mock_config.copy()
        sc5521a_config["model"] = "SC5521A"
        sc5521a_config["address"] = None

        lo = LocalOscillator("test_id", config=sc5521a_config, config_path=None)

        lo.setup(frequency=30)

        self.mock_device.status.assert_called_once_with("off")
        self.mock_device.power.assert_called_once_with(-10)
        self.mock_device.clock_frequency.assert_called_once_with(30)

    def test_power_should_pass_value_to_device(self):
        lo = LocalOscillator("test_id", config=self.mock_config, config_path=None)

        lo.power(-25)

        self.mock_device.power.assert_called_with(-25)

    def test_frequency_should_pass_value_to_device(self):
        lo = LocalOscillator("test_id", config=self.mock_config, config_path=None)

        lo.frequency(12.5e9)

        self.mock_device.frequency.assert_called_with(12.5e9)

    def test_on_should_call_correct_method_for_each_instrument_type(self):
        test_cases = [
            {
                "model": "RohdeSchwarzSGS100A",
                "address": "TCPIP0::1.2.3.4::inst0::INSTR",
                "expected_method": "on",
                "expected_args": (),
                "expected_kwargs": {},
            },
            {
                "model": "SignalCore_SC5511A",
                "address": "10003ABC",
                "expected_method": "do_set_output_status",
                "expected_args": (1,),
                "expected_kwargs": {},
            },
            {
                "model": "SC5521A",
                "address": None,
                "expected_method": "status",
                "expected_args": ("on",),
                "expected_kwargs": {},
            },
        ]

        for test_case in test_cases:
            config = self.mock_config.copy()
            config["model"] = test_case["model"]
            if test_case["address"] is None:
                if "address" in config:
                    del config["address"]
            else:
                config["address"] = test_case["address"]

            self.mock_rohde_device.reset_mock()
            self.mock_sc5511a_device.reset_mock()
            self.mock_sc5521a_device.reset_mock()
            self.mock_device.reset_mock()

            lo = LocalOscillator("test_id", config=config, config_path=None)
            lo.on()

            expected_method = getattr(self.mock_device, test_case["expected_method"])

            expected_method.assert_called_once_with(
                *test_case["expected_args"], **test_case["expected_kwargs"]
            )

    def test_off_should_call_correct_method_for_each_instrument_type(self):
        test_cases = [
            {
                "model": "RohdeSchwarzSGS100A",
                "address": "TCPIP0::1.2.3.4::inst0::INSTR",
                "expected_method": "off",
                "expected_args": (),
                "expected_kwargs": {},
            },
            {
                "model": "SignalCore_SC5511A",
                "address": "10003ABC",
                "expected_method": "do_set_output_status",
                "expected_args": (0,),
                "expected_kwargs": {},
            },
            {
                "model": "SC5521A",
                "address": None,
                "expected_method": "status",
                "expected_args": ("off",),
                "expected_kwargs": {},
            },
        ]

        for test_case in test_cases:
            config = self.mock_config.copy()
            config["model"] = test_case["model"]
            if test_case["address"] is None:
                if "address" in config:
                    del config["address"]
            else:
                config["address"] = test_case["address"]

            self.mock_rohde_device.reset_mock()
            self.mock_sc5511a_device.reset_mock()
            self.mock_sc5521a_device.reset_mock()
            self.mock_device.reset_mock()

            lo = LocalOscillator("test_id", config=config, config_path=None)
            lo.off()

            expected_method = getattr(self.mock_device, test_case["expected_method"])

            expected_method.assert_called_once_with(
                *test_case["expected_args"], **test_case["expected_kwargs"]
            )

    def test_register_unregister_should_call_event_handler_methods(self):
        lo = LocalOscillator("test_id", config=self.mock_config, config_path=None)

        self.mock_event_handler.reset_mock()

        lo.unregister()
        self.mock_event_handler.unregister_local_oscillator.assert_called_once_with(lo)

        lo.register()
        self.mock_event_handler.register_local_oscillator.assert_called_once_with(lo)

    def test_class_methods_should_call_event_handler_class_methods(self):
        LocalOscillator.disable_all_auto_control()
        self.mock_event_handler.disable_auto_control.assert_called_once()

        self.mock_event_handler.reset_mock()

        LocalOscillator.enable_all_auto_control()
        self.mock_event_handler.enable_auto_control.assert_called_once()


if __name__ == "__main__":
    unittest.main()

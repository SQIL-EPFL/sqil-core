import unittest
from unittest.mock import MagicMock, patch

from sqil_core.experiment._events import after_experiment, before_experiment
from sqil_core.experiment._experiment import Experiment
from sqil_core.experiment.instruments.local_oscillator import LocalOscillator
from sqil_core.experiment.lo_event_handler import lo_event_handlers
from sqil_core.experiment.setup_registry import setup_registry


class TestExperimentIntegration(unittest.TestCase):
    def setUp(self):
        self.logger_patch = patch("sqil_core.config_log.logger")
        self.read_yaml_patch = patch("sqil_core.experiment._experiment.read_yaml")
        self.rohde_schwarz_patch = patch(
            "sqil_core.experiment.instruments.local_oscillator.RohdeSchwarzSGS100A"
        )
        self.signalcore_sc5511a_patch = patch(
            "sqil_core.experiment.instruments.local_oscillator.SignalCore_SC5511A"
        )
        self.signalcore_sc5521a_patch = patch(
            "sqil_core.experiment.instruments.local_oscillator.SC5521A"
        )

        self.mock_logger = self.logger_patch.start()
        self.mock_read_yaml = self.read_yaml_patch.start()

        self.mock_rohde_schwarz = self.rohde_schwarz_patch.start()
        self.mock_signalcore_sc5511a = self.signalcore_sc5511a_patch.start()
        self.mock_signalcore_sc5521a = self.signalcore_sc5521a_patch.start()

        self.mock_rohde_device = MagicMock()
        self.mock_rohde_schwarz.return_value = self.mock_rohde_device

        self.mock_sc5511a_device = MagicMock()
        self.mock_signalcore_sc5511a.return_value = self.mock_sc5511a_device

        self.mock_sc5521a_device = MagicMock()
        self.mock_signalcore_sc5521a.return_value = self.mock_sc5521a_device

        self.mock_setup_data = {
            "instruments": {
                "rohde": {
                    "name": "rohde_lo",
                    "model": "RohdeSchwarzSGS100A",
                    "address": "TCPIP0::1.2.3.4::inst0::INSTR",
                    "type": "LO",
                },
                "sc5511a": {
                    "name": "sc5511a_lo",
                    "model": "SignalCore_SC5511A",
                    "address": "10003ABC",
                    "type": "LO",
                },
                "sc5521a": {
                    "name": "sc5521a_lo",
                    "model": "SC5521A",
                    "type": "LO",
                },
            }
        }
        self.mock_read_yaml.return_value = self.mock_setup_data

        lo_event_handlers.local_oscillators = []
        lo_event_handlers.auto_control_enabled = True

        setup_registry.setup_functions = {}

    def tearDown(self):
        self.logger_patch.stop()
        self.read_yaml_patch.stop()
        self.rohde_schwarz_patch.stop()
        self.signalcore_sc5511a_patch.stop()
        self.signalcore_sc5521a_patch.stop()

    def test_experiment_should_auto_control_oscillators_during_run(self):
        experiment = Experiment(setup_path="test_setup.yaml")

        self.mock_rohde_device.reset_mock()
        self.mock_sc5511a_device.reset_mock()
        self.mock_sc5521a_device.reset_mock()

        # different kind of mock
        # the event handler expects a LO instance,
        # while the LO constructor expects a low-level instance (with the driver etc)
        mock_rohde_lo = MagicMock()
        mock_rohde_lo.name = "rohde_lo"
        mock_sc5511a_lo = MagicMock()
        mock_sc5511a_lo.name = "sc5511a_lo"
        mock_sc5521a_lo = MagicMock()
        mock_sc5521a_lo.name = "sc5521a_lo"

        lo_event_handlers.register_local_oscillator(mock_rohde_lo)
        lo_event_handlers.register_local_oscillator(mock_sc5511a_lo)
        lo_event_handlers.register_local_oscillator(mock_sc5521a_lo)

        experiment.run()

        mock_rohde_lo.on.assert_called_once()
        mock_sc5511a_lo.on.assert_called_once()
        mock_sc5521a_lo.on.assert_called_once()

        mock_rohde_lo.off.assert_called_once()
        mock_sc5511a_lo.off.assert_called_once()
        mock_sc5521a_lo.off.assert_called_once()

    def test_experiment_should_not_auto_control_when_disabled(self):
        experiment = Experiment(setup_path="test_setup.yaml")

        LocalOscillator.disable_all_auto_control()

        self.mock_rohde_device.reset_mock()
        self.mock_sc5511a_device.reset_mock()
        self.mock_sc5521a_device.reset_mock()

        experiment.run()

        self.mock_rohde_device.on.assert_not_called()
        self.mock_rohde_device.off.assert_not_called()
        self.mock_sc5511a_device.do_set_output_status.assert_not_called()
        self.mock_sc5521a_device.status.assert_not_called()

        LocalOscillator.enable_all_auto_control()

    def test_experiment_should_respect_custom_setup(self):
        def custom_rohde_setup(lo):
            lo.frequency(5e9)
            lo.power(-30)

        def custom_sc5511a_setup(lo):
            lo.setup(frequency=100)
            lo.power(-20)

        setup_registry.register_setup("rohde", custom_rohde_setup)
        setup_registry.register_setup("sc5511a", custom_sc5511a_setup)

        Experiment(setup_path="test_setup.yaml")

        self.mock_rohde_device.frequency.assert_called_with(5e9)
        self.mock_rohde_device.power.assert_called_with(-30)

        self.mock_sc5511a_device.do_set_ref_out_freq.assert_called_with(100)
        self.mock_sc5511a_device.power.assert_called_with(-20)

        self.mock_sc5521a_device.status.assert_called_with("off")
        self.mock_sc5521a_device.power.assert_called_with(-10)

    def test_experiment_should_handle_mixed_lo_control(self):
        experiment = Experiment(setup_path="test_setup.yaml")

        experiment.instruments.rohde.unregister()

        self.mock_rohde_device.reset_mock()
        self.mock_sc5511a_device.reset_mock()
        self.mock_sc5521a_device.reset_mock()

        experiment.run()

        self.mock_rohde_device.on.assert_not_called()
        self.mock_rohde_device.off.assert_not_called()

        self.mock_sc5511a_device.do_set_output_status.assert_any_call(1)
        self.mock_sc5511a_device.do_set_output_status.assert_any_call(0)

        self.mock_sc5521a_device.status.assert_any_call("on")
        self.mock_sc5521a_device.status.assert_any_call("off")

    def test_experiment_should_handle_lo_exceptions_gracefully(self):
        experiment = Experiment(setup_path="test_setup.yaml")

        self.mock_rohde_device.on.side_effect = Exception("Cannot turn on")
        self.mock_sc5511a_device.do_set_output_status.side_effect = [
            None,
            Exception("Cannot turn off"),
        ]

        with self.assertLogs("sqil_logger", level="ERROR") as cm:
            experiment.run()

        # direct log output, no mock -> more robust
        log_output = "\n".join(cm.output)

        self.assertIn("Failed to turn on rohde_lo: Cannot turn on", log_output)
        self.assertIn("Failed to turn off sc5511a_lo: Cannot turn off", log_output)


class CustomExperiment(Experiment):
    def sequence(self):
        self.step_count = 0

        self.step_count += 1
        self.instruments.rohde.frequency(10e9)

        self.step_count += 1
        self.instruments.rohde.power(-15)

        self.step_count += 1


class TestCustomExperiment(unittest.TestCase):
    def setUp(self):
        self.logger_patch = patch("sqil_core.config_log.logger")
        self.read_yaml_patch = patch("sqil_core.experiment._experiment.read_yaml")
        self.rohde_schwarz_patch = patch(
            "sqil_core.experiment.instruments.local_oscillator.RohdeSchwarzSGS100A"
        )

        self.mock_logger = self.logger_patch.start()
        self.mock_read_yaml = self.read_yaml_patch.start()
        self.mock_rohde_schwarz = self.rohde_schwarz_patch.start()

        self.mock_rohde_device = MagicMock()
        self.mock_rohde_schwarz.return_value = self.mock_rohde_device

        self.mock_setup_data = {
            "instruments": {
                "rohde": {
                    "name": "rohde_lo",
                    "model": "RohdeSchwarzSGS100A",
                    "address": "TCPIP0::1.2.3.4::inst0::INSTR",
                    "type": "LO",
                }
            }
        }
        self.mock_read_yaml.return_value = self.mock_setup_data

    def tearDown(self):
        self.logger_patch.stop()
        self.read_yaml_patch.stop()
        self.rohde_schwarz_patch.stop()

    def test_custom_experiment_should_execute_sequence_with_event_handlers(self):
        before_handler = MagicMock()
        after_handler = MagicMock()

        before_experiment.connect(before_handler)
        after_experiment.connect(after_handler)

        try:
            experiment = CustomExperiment(setup_path="test_setup.yaml")
            experiment.run()

            self.assertEqual(experiment.step_count, 3)

            self.mock_rohde_device.frequency.assert_called_with(10e9)
            self.mock_rohde_device.power.assert_called_with(-15)

            before_handler.assert_called_once()
            after_handler.assert_called_once()

        finally:
            before_experiment.disconnect(before_handler)
            after_experiment.disconnect(after_handler)


if __name__ == "__main__":
    unittest.main()

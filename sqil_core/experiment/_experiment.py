from __future__ import annotations

from abc import ABC, abstractmethod

from laboneq import workflow
from laboneq.dsl.quantum import TransmonParameters
from laboneq.dsl.quantum.qpu import QPU
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.dsl.session import Session
from laboneq.simple import DeviceSetup
from laboneq.simple import Experiment as LaboneQExperiment
from laboneq.workflow.tasks import compile_experiment, run_experiment
from laboneq_applications.analysis.resonator_spectroscopy import analysis_workflow
from laboneq_applications.experiments.options import TuneUpWorkflowOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_modify,
    update_qubits,
)
from numpy.typing import ArrayLike
from qcodes import Instrument as QCodesInstrument

from sqil_core.config_log import logger
from sqil_core.experiment._events import after_experiment, before_experiment
from sqil_core.experiment.instruments.local_oscillator import LocalOscillator
from sqil_core.experiment.instruments.server import (
    connect_instruments,
    link_instrument_server,
)
from sqil_core.experiment.setup_registry import setup_registry
from sqil_core.utils._read import read_yaml
from sqil_core.utils._utils import _extract_variables_from_module


class Instruments:
    def __init__(self, data):
        self._instruments = data
        for key, value in data.items():
            setattr(self, key, value)

    def __iter__(self):
        """Allow iteration directly over instrument instances."""
        return iter(self._instruments.values())


class ExperimentHandler(ABC):
    instruments: Instruments | None = None
    zi_setup: DeviceSetup
    zi_session: Session
    qpu: QPU

    def __init__(
        self,
        params: dict = {},
        param_dict_path: str = "",
        setup_path: str = "",
        server=False,
    ):
        if server:
            server, instrument_instances = link_instrument_server()
        else:
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
            instrument_instances = connect_instruments(instrument_dict)

        # Create Zurich Instruments session
        zi = instrument_instances.get("zi", None)
        if zi is not None:
            self.zi_setup = DeviceSetup.from_descriptor(zi.descriptor, zi.address)
            self.zi_session = Session(self.zi_setup)
            self.zi_session.connect()
            if zi.get_qpu is not None:
                self.qpu = zi.get_qpu(self.zi_setup)

        self.instruments = Instruments(instrument_instances)

    # TODO: move to server
    def _setup_instruments(self):
        """Default setup for all instruments with support for custom setups"""
        logger.info("Setting up instruments")
        if not hasattr(self, "instruments"):
            logger.warning("No instruments to set up")
            return

        for instrument in self.instruments:
            # for robustness against future modifications
            if not hasattr(instrument, "setup"):
                continue

            try:
                instrument_name = getattr(instrument, "name", instrument.id)

                if setup_registry.has_custom_setup(instrument.id):
                    logger.info(
                        f"Applying registered custom setup to {instrument_name}"
                    )
                    setup_registry.apply_setup(instrument.id, instrument)
                else:
                    logger.info(f"Running default setup to {instrument_name}")
                    instrument.setup()
            except Exception as e:
                logger.error(
                    f"Error during setup of {getattr(instrument, 'name', instrument.id)}: {str(e)}"
                )

    @abstractmethod
    def sequence(self, *params, **kwargs):
        """Experimental sequence defined by the user"""
        pass

    @abstractmethod
    def analyze(self, raw_data, *params, **kwargs):
        pass

    def run(self, *params, **kwargs):
        before_experiment.send()

        seq = self.sequence(*params, **kwargs)
        is_laboneq_exp = type(seq) == LaboneQExperiment
        result = None

        if is_laboneq_exp:
            compiled_exp = compile_experiment(self.zi_session, seq)
            result = run_experiment(self.zi_session, compiled_exp)
        else:
            result = seq

        after_experiment.send()

        QCodesInstrument.close_all()

        self.analyze(result, *params, **kwargs)

        return result

    def laboneq_workflow_runner(
        session: Session, qpu: QPU, qubit: QuantumElement, name: str | None = None
    ):
        @workflow.workflow(name="workflow_runner")
        def laboneq_workflow_runner(
            options: TuneUpWorkflowOptions | None = None,
        ) -> None:
            qubit = temporary_modify(qubit, temporary_parameters)

            exp = create_experiment(
                qpu,
                qubit,
                frequencies=frequencies,
            )
            compiled_exp = compile_experiment(session, exp)
            result = run_experiment(session, compiled_exp)
            return result
            # with workflow.if_(options.do_analysis):
            #     analysis_results = analysis_workflow(result, qubit, frequencies)
            #     qubit_parameters = analysis_results.output
            #     with workflow.if_(options.update):
            #         update_qubits(qpu, qubit_parameters["new_parameter_values"])
            # workflow.return_(result)

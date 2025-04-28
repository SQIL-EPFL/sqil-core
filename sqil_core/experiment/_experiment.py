from __future__ import annotations

from abc import ABC, abstractmethod

from laboneq import workflow
from laboneq.dsl.quantum import TransmonParameters
from laboneq.dsl.quantum.qpu import QPU
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.dsl.session import Session
from laboneq.simple import Experiment as LaboneQExperiment
from laboneq.workflow.tasks import compile_experiment, run_experiment
from laboneq_applications.analysis.resonator_spectroscopy import analysis_workflow
from laboneq_applications.experiments.options import TuneUpWorkflowOptions
from laboneq_applications.tasks.parameter_updating import (
    temporary_modify,
    update_qubits,
)
from numpy.typing import ArrayLike

from sqil_core.config_log import logger
from sqil_core.experiment._events import after_experiment, before_experiment
from sqil_core.experiment.instruments.local_oscillator import LocalOscillator
from sqil_core.experiment.instruments.server import link_instrument_server
from sqil_core.experiment.setup_registry import setup_registry


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

    def __init__(
        self, params: dict = {}, param_dict_path: str = "", setup_path: str = ""
    ):
        # Get instruments from the server
        server, instrument_instances = link_instrument_server()
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
    def sequence(self):
        """Experimental sequence defined by the user"""
        pass

    def run(self):
        before_experiment.send()

        seq = self.sequence()
        is_laboneq_exp = type(seq, LaboneQExperiment)
        result = None

        if is_laboneq_exp:
            session = self.instruments.zi
            compiled_exp = compile_experiment(session, seq)
            result = run_experiment(session, compiled_exp)
            pass
        else:
            result = seq

        after_experiment.send()

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

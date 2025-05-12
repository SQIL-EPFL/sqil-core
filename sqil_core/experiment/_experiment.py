from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
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
from sqil_core.experiment.data.plottr import DataDict, DDH5Writer
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
    setup: dict
    instruments: Instruments | None = None

    zi_setup: DeviceSetup
    zi_session: Session
    qpu: QPU

    db_schema: dict = None

    def __init__(
        self,
        params: dict = {},
        param_dict_path: str = "",
        setup_path: str = "",
        server=False,
    ):
        # Read setup file
        if not setup_path:
            config = read_yaml("config.yaml")
            setup_path = config.get("setup_path", "setup.py")
        self.setup = _extract_variables_from_module("setup", setup_path)

        # Get instruments through the server or connect locally
        if server:
            server, instrument_instances = link_instrument_server()
        else:
            instrument_dict = self.setup.get("instruments", None)
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
        db_type = self.setup.get("storage", {}).get("db_type", "")
        if db_type == "plottr":
            return self.run_with_plottr(*params, **kwargs)

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

        # Close and delete QCodes instances to avoid connection issues in following experiments
        QCodesInstrument.close_all()
        for instrument in self.instruments:
            del instrument

        # Run analysis script
        self.analyze(result, *params, **kwargs)

        return result

    def run_with_plottr(self, *params, **kwargs):
        before_experiment.send()

        sweep = kwargs.get("sweep", None)
        if sweep is not None:
            # TODO: dynamically add unit
            self.db_schema.update(
                **{"sweep": {"type": "axis", "name": sweep.keys()[0]}}
            )

        params_map, _ = map_inputs(self.sequence)
        datadict = build_plottr_dict(self.db_schema)
        db_path = self.setup["storage"]["db_path"]
        db_path_local = self.setup["storage"]["db_path_local"]

        # TODO: dynamically assign self.exp_name to class name if not provided
        data_to_save = {}
        with DDH5Writer(datadict, db_path_local, name=self.exp_name) as writer:
            filepath_parent = writer.filepath.parent
            path = str(filepath_parent)
            last_two_parts = path.split(os.sep)[-2:]
            new_path = os.path.join(db_path, *last_two_parts)
            writer.save_text("directry_path.md", new_path)

            seq = self.sequence(*params, **kwargs)
            is_laboneq_exp = type(seq) == LaboneQExperiment
            result = None

            if is_laboneq_exp:
                compiled_exp = compile_experiment(self.zi_session, seq)
                result = run_experiment(self.zi_session, compiled_exp)
                qu_idx_by_uid = [qubit.uid for qubit in self.qpu.qubits]
                raw_data = result[qu_idx_by_uid[params[0]]].result.data
                data_to_save["data"] = raw_data
            else:
                result = seq
                data_to_save["data"] = result

            # Add parameters to saved data
            datadict_keys = datadict.keys()
            for key, value in params_map.items():
                if key in datadict_keys:
                    data_to_save[key] = params[value]
            # Save data using plottr
            writer.add_data(**data_to_save)

            # Add sweep to datadict
            if sweep is not None:
                writer.add_data(**{"sweep": sweep})

            after_experiment.send()

        # Close and delete QCodes instances to avoid connection issues in following experiments
        QCodesInstrument.close_all()
        for instrument in self.instruments:
            del instrument

        # Run analysis script
        self.analyze(result, *params, **kwargs)

        return result


def build_plottr_dict(db_schema):
    """Create a DataDict object from the given schema."""
    axes = []
    db = {}

    data_key = "data"
    data_unit = ""

    for key, value in db_schema.items():
        if value.get("type") == "axis":
            unit = value.get("unit", "")
            db[key] = dict(unit=unit)
            axes.append(key)
        elif value.get("type") == "data":
            data_key = key
            data_unit = value.get("unit", "")
    db[data_key] = dict(axes=axes, unit=data_unit)

    datadict = DataDict(**db)
    datadict.add_meta("schema", db_schema)

    return datadict


import inspect


def map_inputs(func):
    """Extracts parameter names and keyword arguments from a function signature."""
    sig = inspect.signature(func)
    params = {}
    kwargs = []

    for index, (name, param) in enumerate(sig.parameters.items()):
        if param.default == inspect.Parameter.empty:
            # Positional or required argument
            params[name] = index
        else:
            # Keyword argument
            kwargs.append(name)

    return params, kwargs

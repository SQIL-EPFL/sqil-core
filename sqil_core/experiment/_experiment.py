from __future__ import annotations

import copy
import itertools
import json
import os
from abc import ABC, abstractmethod
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from laboneq import serializers, workflow
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
from sqil_core.experiment._events import (
    after_experiment,
    after_sequence,
    before_experiment,
    before_sequence,
    clear_signal,
)
from sqil_core.experiment.data.plottr import DataDict, DDH5Writer
from sqil_core.experiment.instruments.local_oscillator import LocalOscillator
from sqil_core.experiment.instruments.server import (
    connect_instruments,
    link_instrument_server,
)
from sqil_core.experiment.instruments.zurich_instruments import ZI_Instrument

# from sqil_core.experiment.setup_registry import setup_registry
from sqil_core.utils._read import copy_folder, read_yaml
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
            # Reset event listeners
            clear_signal(before_experiment)
            clear_signal(before_sequence)
            clear_signal(after_sequence)
            clear_signal(after_experiment)
        instrument_instances = connect_instruments(instrument_dict)

        # Create Zurich Instruments session
        zi = cast(ZI_Instrument, instrument_instances.get("zi", None))
        if zi is not None:
            self.zi_setup = zi.generate_setup()
            # self.zi_setup = DeviceSetup.from_descriptor(zi.descriptor, zi.address)
            self.zi_session = Session(self.zi_setup)
            self.zi_session.connect()
            self._load_qpu(zi.generate_qpu)

        self.instruments = Instruments(instrument_instances)
        self._setup_instruments()

    def _load_qpu(self, generate_qpu: Callable):
        qpu_filename = self.setup["storage"].get("qpu_filename", "qpu.json")
        db_path_local = self.setup["storage"]["db_path_local"]
        try:
            self.qpu = serializers.load(os.path.join(db_path_local, qpu_filename))
        except FileNotFoundError:
            logger.warning(
                f"Cannot find QPU file name {qpu_filename} in {db_path_local}"
            )
            logger.warning(f" -> Creating a new QPU file")
            self.qpu = generate_qpu(self.zi_setup)
            os.makedirs(db_path_local, exist_ok=True)
            serializers.save(
                self.qpu,
                os.path.join(db_path_local, qpu_filename),
            )

    # Move to server
    def _setup_instruments(self):
        """Default setup for all instruments with support for custom setups"""
        logger.info("Setting up instruments")
        if not hasattr(self, "instruments"):
            logger.warning("No instruments to set up")
            return

        for instrument in self.instruments:
            if not hasattr(instrument, "setup"):
                continue
            instrument.setup()

    @abstractmethod
    def sequence(self, *args, **kwargs):
        """Experimental sequence defined by the user"""
        pass

    @abstractmethod
    def analyze(self, path, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        try:
            db_type = self.setup.get("storage", {}).get("db_type", "")

            if db_type == "plottr":
                return self.run_with_plottr(*args, **kwargs)
            else:
                return self.run_raw(*args, **kwargs)

        finally:
            # Close and delete QCodes instances to avoid connection issues in following experiments
            QCodesInstrument.close_all()
            for instrument in self.instruments:
                del instrument

    def run_with_plottr(self, *args, **kwargs):
        logger.info("Before exp")
        before_experiment.send(sender=self)

        # Map input parameters index to their name
        params_map, _ = map_inputs(self.sequence)

        # Get information on sweeps
        sweeps: dict = kwargs.get("sweeps", None)
        sweep_keys = []
        sweep_grid = []
        sweep_schema = {}
        if sweeps is not None:
            # Name of the parameters to sweep
            sweep_keys = list(sweeps.keys())
            # Create a mesh grid of all the sweep parameters
            sweep_grid = list(itertools.product(*sweeps.values()))
            # Add sweeps to the database schema
            for i, key in enumerate(sweep_keys):
                # TODO: dynamically add unit
                sweep_schema[f"sweep{i}"] = {"type": "axis", "name": key}

        # Create the plotter datadict (database) using the inferred schema
        db_schema = {**self.db_schema, **sweep_schema}
        datadict = build_plottr_dict(db_schema)
        # Get local and server storage folders
        db_path = self.setup["storage"]["db_path"]
        db_path_local = self.setup["storage"]["db_path_local"]

        # TODO: dynamically assign self.exp_name to class name if not provided
        with DDH5Writer(datadict, db_path_local, name=self.exp_name) as writer:
            # Get the path to the folder where the data will be stored
            storage_path = get_plottr_path(writer, db_path)
            storage_path_local = get_plottr_path(writer, db_path_local)
            # Save helper files
            writer.save_text("directry_path.md", storage_path)
            # Save backup qpu
            old_qubits = self.qpu.copy_quantum_elements()
            serializers.save(self.qpu, os.path.join(storage_path_local, "qpu_old.json"))

            # TODO: for index sweep don't recompile laboneq
            for sweep_values in sweep_grid or [None]:
                data_to_save = {}

                # Run/create the experiment. Creates it for laboneq, runs it otherwise
                seq = self.sequence(*args, **kwargs)
                # Detect if the sequence created a laboneq experiment
                is_laboneq_exp = type(seq) == LaboneQExperiment

                if is_laboneq_exp:
                    qu_indices = kwargs.get("qu_idx", [0])
                    if type(qu_indices) == int:
                        qu_indices = [qu_indices]
                    used_qubits = [self.qpu.quantum_elements[i] for i in qu_indices]
                    qu_idx_by_uid = [qubit.uid for qubit in self.qpu.quantum_elements]
                    # TODO: save and re-apply old qubit params
                    # Reset to the first value of every sweep,
                    # then override current sweep value for all qubits
                    for qubit in used_qubits:
                        tmp = dict(zip(sweep_keys, sweep_values or []))
                        qubit.update(**tmp)
                    # Create the experiment (required to update params)
                    seq = self.sequence(*args, **kwargs)
                    compiled_exp = compile_experiment(self.zi_session, seq)
                    # pulse_sheet(self.zi_setup, compiled_exp, self.exp_name)
                    before_sequence.send(sender=self)
                    result = run_experiment(self.zi_session, compiled_exp)
                    after_sequence.send(sender=self)
                    # TODO: handle multiple qubits. Maybe different datadicts?
                    raw_data = result[qu_idx_by_uid[qu_indices[0]]].result.data
                    data_to_save["data"] = raw_data
                    result = raw_data
                else:
                    # TODO: handle results for different instrumets
                    data_to_save["data"] = seq

                # Add parameters to the data to save
                datadict_keys = datadict.keys()
                for key, value in params_map.items():
                    if key in datadict_keys:
                        data_to_save[key] = args[value]
                # Add sweeps to the data to save
                if sweeps is not None:
                    for i, key in enumerate(sweep_keys):
                        data_to_save[f"sweep{i}"] = sweep_values[i]

                # Save data using plottr
                writer.add_data(**data_to_save)

            after_experiment.send()

        # Reset the qpu to its previous state
        self.qpu.quantum_operations.detach_qpu()
        self.qpu = QPU(old_qubits, self.qpu.quantum_operations)

        # Run analysis script
        try:
            anal_res = self.analyze(storage_path_local, *args, **kwargs)
            # writer.save_text("analysis.md", anal_res)
            plt.show()
        except Exception as e:
            logger.error(f"Error while analyzing the data {e}")

        # TODO: update qpu
        serializers.save(self.qpu, os.path.join(storage_path_local, "qpu_new.json"))

        # Copy the local folder to the server
        copy_folder(storage_path_local, storage_path)

    def run_raw(self, *args, **kwargs):
        before_experiment.send(sender=self)

        seq = self.sequence(*args, **kwargs)
        is_laboneq_exp = type(seq) == LaboneQExperiment
        result = None

        if is_laboneq_exp:
            compiled_exp = compile_experiment(self.zi_session, seq)
            result = run_experiment(self.zi_session, compiled_exp)
        else:
            result = seq

        after_experiment.send(sender=self)

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

    datadict.add_meta("schema", json.dumps(db_schema))

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


def get_plottr_path(writer: DDH5Writer, root_path):
    filepath_parent = writer.filepath.parent
    path = str(filepath_parent)
    last_two_parts = path.split(os.sep)[-2:]
    return os.path.join(root_path, *last_two_parts)


from laboneq.simple import OutputSimulator


def pulse_sheet(device_setup, compiled_exp, name):
    start = 0
    end = 0.15e-6
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    # Get physical channel references via the logical signals
    drive_iq_port = device_setup.logical_signal_by_uid("q0/drive").physical_channel
    measure_iq_port = device_setup.logical_signal_by_uid("q0/measure").physical_channel
    acquire_port = device_setup.logical_signal_by_uid("q0/acquire").physical_channel

    # Get waveform snippets from the simulation
    simulation = OutputSimulator(compiled_exp)

    drive_snippet = simulation.get_snippet(
        drive_iq_port, start=start, output_length=end
    )

    measure_snippet = simulation.get_snippet(
        measure_iq_port, start=start, output_length=end
    )

    acquire_snippet = simulation.get_snippet(
        acquire_port, start=start, output_length=end
    )

    fig = plt.figure(figsize=(15, 5))
    plt.plot(
        drive_snippet.time * 1e6,
        drive_snippet.wave.real,
        color=colors[0],
        label="Qubit I",
    )
    plt.fill_between(
        drive_snippet.time * 1e6, drive_snippet.wave.real, color=colors[0], alpha=0.6
    )
    plt.plot(
        drive_snippet.time * 1e6,
        drive_snippet.wave.imag,
        color=colors[1],
        label="Qubit Q",
    )
    plt.fill_between(
        drive_snippet.time * 1e6, drive_snippet.wave.imag, color=colors[1], alpha=0.6
    )

    plt.plot(
        measure_snippet.time * 1e6,
        measure_snippet.wave.real,
        color=colors[2],
        label="Readout I",
    )
    plt.fill_between(
        measure_snippet.time * 1e6,
        measure_snippet.wave.real,
        color=colors[2],
        alpha=0.6,
    )
    plt.plot(
        measure_snippet.time * 1e6,
        measure_snippet.wave.imag,
        color=colors[3],
        label="Readout Q",
    )
    plt.fill_between(
        measure_snippet.time * 1e6,
        measure_snippet.wave.imag,
        color=colors[3],
        alpha=0.6,
    )
    plt.plot(
        acquire_snippet.time * 1e6,
        acquire_snippet.wave.real,
        color=colors[4],
        label="acquire start",
    )

    plt.legend()
    plt.xlabel(r"Time($\mu s$)")
    plt.ylabel("Amplitude")
    plt.title(name)
    plt.show()

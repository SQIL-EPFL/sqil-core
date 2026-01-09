from __future__ import annotations

import itertools
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import attrs
import matplotlib.pyplot as plt
import numpy as np
from laboneq import serializers
from laboneq.dsl.quantum.qpu import QPU
from laboneq.dsl.session import Session
from laboneq.simple import DeviceSetup
from laboneq.simple import Experiment as LaboneQExperiment
from laboneq.simple import show_pulse_sheet
from laboneq.workflow.tasks import compile_experiment, run_experiment
from qcodes import Instrument as QCodesInstrument
from tqdm.auto import tqdm

from sqil_core.config_log import logger
from sqil_core.experiment._analysis import AnalysisResult
from sqil_core.experiment._events import (
    after_experiment,
    after_sequence,
    before_experiment,
    before_sequence,
    clear_signal,
)
from sqil_core.experiment.data.plottr import DataDict, DDH5Writer
from sqil_core.experiment.helpers._labone_wrappers import w_save
from sqil_core.experiment.instruments.server import (
    connect_instruments,
    link_instrument_server,
)
from sqil_core.experiment.instruments.zurich_instruments import ZI_Instrument

# from sqil_core.experiment.setup_registry import setup_registry
from sqil_core.utils._read import copy_folder, read_yaml
from sqil_core.utils._utils import (
    _extract_variables_from_module,
    flatten_dict,
    make_iterable,
    unflatten_dict,
)


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
    is_zi_exp: bool | None = None
    save_zi_result: bool = False

    db_schema: dict = None
    _run_args: tuple[list, dict] = ([], {})

    def __init__(
        self,
        setup_path: str = "",
        emulation=False,
        server=False,
        is_zi_exp=None,
        no_instruments=False,
        qpu=None,
    ):
        self.emulation = emulation
        if self.emulation:
            logger.warning("Using emulation")

        # Read setup file
        config = {}
        if not setup_path:
            config = read_yaml("config.yaml")
            setup_path = config.get("setup_path", "setup.py")
        self.setup = _extract_variables_from_module("setup", setup_path)

        self.qpu = qpu

        # Set log level
        log_level = config.get("log_level", logging.DEBUG)
        logger.setLevel(log_level)

        if no_instruments == True:
            # Load QPU
            if self.qpu is None:
                generate_qpu = self.setup.get("generate_qpu")
                generate_qpu_args = []
                self._load_qpu(generate_qpu, generate_qpu_args)
            self.is_zi_exp = False
            self.instruments = Instruments({})
            return

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

        # Subscribe experiment (not instruments) to events
        before_experiment.connect(self.on_before_experiment, weak=False)
        after_experiment.connect(self.on_after_experiment, weak=False)

        # Connect to instruments
        instrument_instances = connect_instruments(instrument_dict)

        # Get the generate QPU function
        generate_qpu = self.setup.get("generate_qpu")
        generate_qpu_args = []

        # Create Zurich Instruments session
        zi = cast(ZI_Instrument, instrument_instances.get("zi", None))
        if zi is not None:
            self.zi_setup = zi.generate_setup()
            self.zi_session = Session(self.zi_setup, log_level=logging.WARN)
            self.zi_session.connect(do_emulation=self.emulation)
            generate_qpu_args = [self.zi_setup]

        if is_zi_exp is None:
            self.is_zi_exp = zi is not None

        # Load QPU
        if self.qpu is None:
            self._load_qpu(generate_qpu, generate_qpu_args)

        self.instruments = Instruments(instrument_instances)
        self._setup_instruments()

    def _load_qpu(self, generate_qpu: Callable, generate_qpu_args: list):
        qpu_filename = self.setup["storage"].get("qpu_filename", "qpu.json")
        db_path_local = self.setup["storage"]["db_path_local"]
        try:
            self.qpu = serializers.load(os.path.join(db_path_local, qpu_filename))
        except FileNotFoundError:
            logger.warning(
                f"Cannot find QPU file name {qpu_filename} in {db_path_local}"
            )
            logger.warning(" -> Creating a new QPU file")
            self.qpu = generate_qpu(*generate_qpu_args)
            os.makedirs(db_path_local, exist_ok=True)
            w_save(self.qpu, os.path.join(db_path_local, qpu_filename))

    # Move to server
    def _setup_instruments(self):
        """Default setup for all instruments with support for custom setups"""
        logger.debug("Setting up instruments")
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

    @abstractmethod
    def analyze(self, path, *args, **kwargs):
        pass

    def on_before_experiment(self, *args, **kwargs):
        pass

    def on_after_experiment(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        try:
            db_type = self.setup.get("storage", {}).get("db_type", "")

            if db_type == "plottr":
                return self.run_with_plottr(*args, **kwargs)
            return self.run_raw(*args, **kwargs)

        finally:
            # Close and delete QCodes instances to avoid connection issues in
            # following experiments
            QCodesInstrument.close_all()
            for instrument in self.instruments:
                try:
                    instrument.disconnect()
                finally:
                    del instrument

    def run_with_plottr(self, *args, qu_ids=None, pulse_sheet=False, **kwargs):
        # Sanitize inputs
        if qu_ids is None:
            qu_ids = ["q0"]
        qu_ids = make_iterable(qu_ids)
        run_kwargs = {**kwargs, "qu_ids": qu_ids, "pulse_sheet": pulse_sheet}
        self._run_args = (args, run_kwargs)

        # Before experiment
        logger.debug("Before exp")
        before_experiment.send(sender=self)

        # Map input parameters index to their name
        params_map, _ = map_inputs(self.sequence)
        qubit_order = {qu_id: idx for idx, qu_id in enumerate(qu_ids)}

        # Get information on sweeps
        sweeps: dict = kwargs.get("sweeps", None)
        sweep_keys, sweep_grid, sweep_len, sweep_schema = [], {}, 0, {}
        if sweeps is not None:
            sweep_keys, sweep_grid, sweep_len, sweep_schema = parse_sweeps(
                sweeps, qu_ids
            )
            # Update experiment name
            self.exp_name = "_vs_".join([self.exp_name] + sweep_keys)

        # Create the plotter datadict (database) using the inferred schema
        db_schema = {**self.db_schema, **sweep_schema}
        datadict = build_plottr_dict(db_schema, qu_ids=qu_ids)
        data_keys = [
            key for key, _ in datadict.data_items() if key not in datadict.axes()
        ]
        # Get local and server storage folders
        db_path = self.setup["storage"]["db_path"]
        db_path_local = self.setup["storage"]["db_path_local"]

        # TODO: dynamically assign self.exp_name to class name if not provided
        compiled_exp = None
        with DDH5Writer(datadict, db_path_local, name=self.exp_name) as writer:
            # Get the path to the folder where the data will be stored
            storage_path = get_plottr_path(writer, db_path)
            storage_path_local = get_plottr_path(writer, db_path_local)
            # Save helper files
            writer.save_text("paths.md", f"{storage_path_local}\n{storage_path}")
            # Save backup qpu
            old_qubits = self.qpu.copy_quantum_elements()
            serializers.save(self.qpu, os.path.join(storage_path_local, "qpu_old.json"))

            bar = tqdm(range(sweep_len), desc="Sweep") if sweep_len else [None]
            for sweep_idx in bar:
                data_to_save = {qu_id: {} for qu_id in qu_ids}

                # Reset to the first value of every sweep,
                # then override current sweep value for all qubits
                if sweep_idx is not None:
                    for qu_id in qu_ids:
                        sweep_values = sweep_grid[qu_id][sweep_idx]
                        tmp = dict(zip(sweep_keys, sweep_values, strict=False))
                        self.qpu[qu_id].update(**tmp)

                # Run/create the experiment. Creates it for laboneq, runs it otherwise
                if self.is_zi_exp:
                    # Create the experiment (required to update params)
                    if compiled_exp is None or sweep_keys not in ["index", "current"]:
                        seq = self.sequence(*args, **run_kwargs)
                        compiled_exp = compile_experiment(self.zi_session, seq)
                        logger.info(
                            f"*** ZI estimated runtime: {compiled_exp.estimated_runtime:.2f} s ***"
                        )
                        if pulse_sheet:
                            end_time = (
                                pulse_sheet
                                if type(pulse_sheet) in (int, float)
                                else None
                            )
                            create_pulse_sheet(
                                self.zi_setup,
                                compiled_exp,
                                self.exp_name,
                                end=end_time,
                                qu_ids=qu_ids,
                            )
                            show_pulse_sheet(
                                f"{storage_path_local}/pulsesheet",
                                compiled_exp,
                                interactive=False,
                            )
                    before_sequence.send(sender=self)
                    result = run_experiment(self.zi_session, compiled_exp)
                    after_sequence.send(sender=self)

                    if self.save_zi_result:
                        serializers.save(
                            result, os.path.join(storage_path_local, "zi_result.json")
                        )

                    for data_key in data_keys:
                        data_key_corrected = data_key
                        split_key = np.array(data_key.split("/"))
                        if data_key not in result:
                            # Experiment has no explicit handle handle
                            if split_key[-1] == "data" and data_key not in result:
                                data_key_corrected = f"{split_key[0]}/result"
                            # Only cal traces are returned - used for IQ blobs
                            elif "cal_trace" in result.data[qu_ids[0]]:
                                data_key_corrected = f"{split_key[0]}/cal_trace/{'/'.join(split_key[1:])}"
                        else:
                            data_key_corrected = f"{data_key}/result"
                        data_to_save[data_key] = result.get_data(data_key_corrected)
                else:
                    before_sequence.send(sender=self)
                    # TODO: multiple qubit support
                    seq_res = self.sequence(*args, **run_kwargs)
                    if type(seq_res) is dict:
                        for qu_id in qu_ids:
                            for key, value in seq_res.items():
                                data_to_save[f"{qu_id}/{key}"] = value
                    else:
                        data_to_save["q0/data"] = seq_res
                    for p_name, p_idx in params_map.items():
                        if p_name in datadict.keys():
                            data_to_save[f"q0/{p_name}"] = args[p_idx]
                    after_sequence.send(sender=self)

                # Add parameters to the data to save
                nested_datadict = unflatten_dict(datadict)
                nested_data_to_save = unflatten_dict(data_to_save)
                for qu_id in qu_ids:
                    datadict_keys = nested_datadict[qu_id].keys()
                    for p_name, p_idx in params_map.items():
                        if p_name in datadict_keys:
                            nested_data_to_save[qu_id][p_name] = args[p_idx][
                                qubit_order[qu_id]
                            ]
                # Add parameters that are not axis
                non_axis_params = {}
                for p_name, db_info in db_schema.items():
                    if db_info.get("role") == "param":
                        p_idx = params_map.get(p_name)
                        non_axis_params[p_name] = args[p_idx]
                    datadict.add_meta("params", json.dumps(non_axis_params))
                # Add sweeps to the data to save
                if sweeps is not None:
                    for qu_id in qu_ids:
                        for i, _ in enumerate(sweep_keys):
                            sweep_value = sweep_grid[qu_id][sweep_idx][i]
                            nested_data_to_save[qu_id][f"sweep{i}"] = sweep_value

                # Save data using plottr
                writer.add_data(**flatten_dict(nested_data_to_save))

            after_experiment.send()

        # Reset the qpu to its previous state
        self.qpu.quantum_operations.detach_qpu()
        self.qpu = QPU(old_qubits, self.qpu.quantum_operations)

        # Run analysis script
        anal_res = None
        try:
            anal_res = self.analyze(storage_path_local, *args, **run_kwargs)
            if type(anal_res) is AnalysisResult:
                anal_res = cast(AnalysisResult, anal_res)
                anal_res.save_all(storage_path_local)
                # Update QPU
                if kwargs.get("update_params", True):
                    for qu_id in anal_res.updated_params.keys():
                        qubit = self.qpu[qu_id]
                        qubit.update(**anal_res.updated_params[qu_id])
                # writer.save_text("analysis.md", anal_res)
                plt.show()
        except Exception as e:
            logger.error(f"Error while analyzing the data {e}")

        w_save(self.qpu, os.path.join(storage_path_local, "qpu_new.json"))
        qpu_filename = self.setup["storage"].get("qpu_filename", "qpu.json")
        w_save(self.qpu, os.path.join(db_path_local, qpu_filename))

        # Copy the local folder to the server
        copy_folder(storage_path_local, storage_path)

        return anal_res

    def run_raw(self, *args, **kwargs):
        before_experiment.send(sender=self)

        seq = self.sequence(*args, **kwargs)
        self.is_zi_exp = type(seq) is LaboneQExperiment
        result = None

        if self.is_zi_exp:
            compiled_exp = compile_experiment(self.zi_session, seq)
            result = run_experiment(self.zi_session, compiled_exp)
        else:
            result = seq

        after_experiment.send(sender=self)

        return result

    def custom_plottr(self, logic, db_schema, qu_ids=["q0"]):
        # Create the plotter datadict (database)
        datadict = build_plottr_dict(db_schema, qu_ids=qu_ids)
        # Get local and server storage folders
        db_path = self.setup["storage"]["db_path"]
        db_path_local = self.setup["storage"]["db_path_local"]
        with DDH5Writer(datadict, db_path_local, name=self.exp_name) as writer:
            # Get the path to the folder where the data will be stored
            storage_path = get_plottr_path(writer, db_path)
            storage_path_local = get_plottr_path(writer, db_path_local)
            # Save helper files
            writer.save_text("paths.md", f"{storage_path_local}\n{storage_path}")

            # Run custom logic
            logic(datadict)

            # Copy the local folder to the server
            copy_folder(storage_path_local, storage_path)

    def sweep_around(
        self,
        center: str | float,
        span: float | tuple[float, float],
        n_points: int = None,
        step: float = None,
        scale: str = "linear",
        qu_id="q0",
    ):
        """
        Generates a sweep of values around a specified center, either numerically or by
        referencing a qubit parameter.

        Parameters
        ----------
        center : str or float
            Center of the sweep. If a string, it's interpreted as the name of a qubit
            parameter and resolved via `qubit_value`. If a float, used directly.
        span : float or tuple of float
            If a float, sweep will extend symmetrically by `span` on both sides of
            `center`.
            If a tuple `(left, right)`, creates an asymmetric sweep: `center - left` to
            `center + right`.
        n_points : int, optional
            Number of points in the sweep. Specify exactly one of `n_points` or `step`.
        step : float, optional
            Step size in the sweep. Specify exactly one of `n_points` or `step`.
        scale : {'linear', 'log'}, default 'linear'
            Whether to generate the sweep on a linear or logarithmic scale.
            For logarithmic sweeps, all generated values must be > 0.
        qu_id : str, default "q0"
            Qubit identifier used to resolve `center` if it is a parameter name.

        Returns
        -------
        np.ndarray
            Array of sweep values.

        Raises
        ------
        AttributeError
            If `center` is a string and not found in the qubit's parameter set.
        ValueError
            If scale is not one of 'linear' or 'log'.
            If a log-scale sweep is requested with non-positive start/stop values.
            If both or neither of `n_points` and `step` are provided.

        Notes
        -----
        - For log scale and `step`-based sweeps, the step is interpreted in
            multiplicative terms, and an approximate number of points is derived.
        - Sweep boundaries are inclusive when using `step`, thanks to the `+ step / 2`
            adjustment.
        """

        if isinstance(center, str):
            value = self.qubit_value(param_id=center, qu_id=qu_id)
            if value is None:
                raise AttributeError(
                    f"No attribute {center} in qubit {qu_id} parameters."
                )
            center = value

        # Handle symmetric or asymmetric span
        if isinstance(span, tuple):
            left, right = span
        else:
            left = right = span

        start = center - left
        stop = center + right

        if scale not in ("linear", "log"):
            raise ValueError("scale must be 'linear' or 'log'")

        if start <= 0 or stop <= 0:
            if scale == "log":
                raise ValueError("Logarithmic sweep requires all values > 0")

        if (n_points is None) == (step is None):
            raise ValueError("Specify exactly one of 'n_points' or 'step'")

        if scale == "linear":
            if step is not None:
                return np.arange(start, stop + step / 2, step)
            return np.linspace(start, stop, n_points)

        if step is not None:
            # Compute approximate number of points from step in log space
            log_start = np.log10(start)
            log_stop = np.log10(stop)
            num_steps = (
                int(np.floor((log_stop - log_start) / np.log10(1 + step / start))) + 1
            )
            return np.logspace(log_start, log_stop, num=num_steps)
        return np.logspace(np.log10(start), np.log10(stop), n_points)

    def qubit_value(self, param_id, qu_id="q0"):
        """Get a qubit parameter value from the QPU."""
        params = self.qpu[qu_id].parameters
        return attrs.asdict(params).get(param_id)

    @property
    def run_args(self) -> tuple[list, dict]:
        """Returns args and kwargs used to run the experiment."""
        return self._run_args


def parse_sweeps(sweeps, qu_ids):
    sweep_keys = []
    sweep_grid = {}
    sweep_schema = {}

    # Name of the parameters to sweep
    sweep_keys = list(sweeps.keys())

    # Create database schema for sweeps
    for i, key in enumerate(sweep_keys):
        sweep_schema[f"sweep{i}"] = {"role": "axis", "param_id": key}

    # Build sweep_map of shape { sweep_key: { qu_id: values, ... }, ...}
    sweep_map = {}
    for key, value in sweeps.items():
        if type(value) is dict:
            # Check if there is a sweep for each qubit used
            if not set(qu_ids).issubset(value.keys()):
                raise KeyError(
                    f"Sweep qubit ids ({value.keys()} do not match the qubits "
                    f"currently in use ({qu_ids})"
                )
            # Build sweep by qubit dict
            sweep_map[key] = {qu_id: value[qu_id] for qu_id in qu_ids}
            # Check if all the sweeps have the same length
            if len(set([len(v) for v in sweep_map[key].values()])) != 1:
                raise ValueError(
                    f"The sweep values of `{key}` must have the same length for each "
                    "qubit."
                )
        else:
            sweep_map[key] = {qu_id: value for qu_id in qu_ids}

    # Create a mesh grid of all the sweep parameters
    for qu_id in qu_ids:
        values = [s[qu_id] for s in sweep_map.values()]
        sweep_grid[qu_id] = list(itertools.product(*values))

    # Get the length of the sweep by checking the first qubit,
    # since all sweeps must have the same length
    sweep_len = len(sweep_grid[qu_ids[0]])

    return sweep_keys, sweep_grid, sweep_len, sweep_schema


def build_plottr_dict(db_schema, qu_ids):
    """Create a DataDict object from the given schema."""
    axes = {qu_id: [] for qu_id in qu_ids}
    db = {qu_id: {} for qu_id in qu_ids}

    data_keys = []
    data_unit = ""

    for qu_id in qu_ids:
        for key, value in db_schema.items():
            if value.get("role") in ("axis", "x-axis"):
                unit = value.get("unit", "")
                db[qu_id][key] = dict(unit=unit)
                axes[qu_id].append(f"{qu_id}/{key}")
            elif value.get("role") == "data":
                data_keys.append(key)
                data_unit = value.get("unit", "")
        for data_key in data_keys:
            db[qu_id][data_key] = dict(axes=axes[qu_id], unit=data_unit)

    datadict = DataDict(**flatten_dict(db))

    datadict.add_meta("schema", json.dumps(db_schema))
    datadict.add_meta("qu_ids", json.dumps(qu_ids))

    return datadict


import inspect  # noqa: E402


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


from laboneq.simple import OutputSimulator  # noqa: E402


def create_pulse_sheet(device_setup, compiled_exp, name, end=10e-6, qu_ids=None):
    if qu_ids is None:
        qu_ids = ["q0"]

    start = 0
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    fig, axs = plt.subplots(2, 1, figsize=(20, 5))
    for idx, qu_id in enumerate(qu_ids):
        # Get physical channel references via the logical signals
        drive_iq_port = device_setup.logical_signal_by_uid("q0/drive").physical_channel
        measure_iq_port = device_setup.logical_signal_by_uid(
            "q0/measure"
        ).physical_channel
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

        axs[idx].plot(
            drive_snippet.time * 1e6,
            drive_snippet.wave.real,
            color=colors[0],
            label="Qubit I",
        )
        axs[idx].fill_between(
            drive_snippet.time * 1e6,
            drive_snippet.wave.real,
            color=colors[0],
            alpha=0.6,
        )
        axs[idx].plot(
            drive_snippet.time * 1e6,
            drive_snippet.wave.imag,
            color=colors[1],
            label="Qubit Q",
        )
        axs[idx].fill_between(
            drive_snippet.time * 1e6,
            drive_snippet.wave.imag,
            color=colors[1],
            alpha=0.6,
        )

        axs[idx].plot(
            measure_snippet.time * 1e6,
            measure_snippet.wave.real,
            color=colors[2],
            label="Readout I",
        )
        axs[idx].fill_between(
            measure_snippet.time * 1e6,
            measure_snippet.wave.real,
            color=colors[2],
            alpha=0.6,
        )
        axs[idx].plot(
            measure_snippet.time * 1e6,
            measure_snippet.wave.imag,
            color=colors[3],
            label="Readout Q",
        )
        axs[idx].fill_between(
            measure_snippet.time * 1e6,
            measure_snippet.wave.imag,
            color=colors[3],
            alpha=0.6,
        )
        axs[idx].plot(
            acquire_snippet.time * 1e6,
            acquire_snippet.wave.real,
            color=colors[4],
            label="acquire start",
        )

        axs[idx].legend()
        axs[idx].set_xlabel(r"Time($\mu s$)")
        axs[idx].set_ylabel("Amplitude")
        axs[idx].set_title(f"{name} - {qu_id}")

    plt.show()

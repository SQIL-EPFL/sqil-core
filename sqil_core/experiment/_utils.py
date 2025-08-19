from sqil_core.config_log import logger
from sqil_core.experiment import ExperimentHandler


def bind_instrument_qubit(qubit_param: str, qu_uid="q0"):
    def bind(exp: ExperimentHandler):
        try:
            qubit = exp.qpu[qu_uid]
            return getattr(qubit.parameters, qubit_param)
        except Exception:
            logger.error(
                f"Error binding instrument to qubit parameter `{qubit_param}`"
                + " (for qubit {qu_uid})"
            )

    return lambda exp: bind(exp)

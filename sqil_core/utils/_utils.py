import hashlib
import importlib.util
import inspect
import sys

from sqil_core.config_log import logger


def _count_function_parameters(func):
    sig = inspect.signature(func)
    return len(
        [
            param
            for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        ]
    )


def _extract_variables_from_module(module_name, path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get all variables and their values
        variables = {
            name: value
            for name, value in vars(module).items()
            if not name.startswith("__")
        }
        return variables

    except Exception as e:
        logger.error(f"Error while extracting variables from {path}: {str(e)}")

    return {}


def _hash_file(path):
    """Generate a hash for the file using SHA256."""
    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as file:
            for byte_block in iter(lambda: file.read(4096), b""):
                sha256_hash.update(byte_block)
    except Exception as e:
        logger.error(f"Unable to hash file '{path}': {str(e)}")
        return None
    return sha256_hash.hexdigest()

import hashlib
import importlib.util
import inspect
import sys
from collections import defaultdict
from collections.abc import Iterable

from sqil_core.config_log import logger


def fill_gaps(primary_list: list, fallback_list: list) -> list:
    """
    Fills gaps in the primary list using values from the fallback list.

    This function iterates through two lists, and for each pair of elements,
    it fills in the gaps where the element in the primary list is `None`
    with the corresponding element from the fallback list. If the element
    in the primary list is not `None`, it is kept as-is.

    Parameters
    ----------
    primary_list : list
        A list of values where some elements may be `None`, which will be replaced by
        values from `fallback_list`.
    fallback_list : list
        A list of values used to fill gaps in `primary_list`.

    Returns
    -------
    result : list
        A new list where `None` values in `primary_list` are replaced by corresponding
        values from `fallback_list`.

    Examples
    --------
    >>> primary_list = [1, None, 3, None, 5]
    >>> fallback_list = [10, 20, 30, 40, 50]
    >>> fill_gaps(primary_list, fallback_list)
    [1, 20, 3, 40, 5]
    """
    if (fallback_list is None) or (len(fallback_list) == 0):
        return primary_list

    if primary_list is None:
        return fallback_list

    if len(primary_list) == 0:
        primary_list = []

    result = primary_list
    fallback_list = fallback_list[0 : len(primary_list)]
    for i in range(len(fallback_list)):
        if result[i] is None:
            result[i] = fallback_list[i]

    return result


def make_iterable(obj) -> Iterable:
    """
    Ensures that the given object is an iterable.

    If the input object is already an iterable (excluding strings), it is returned
    as-is. Otherwise, it is wrapped in a list to make it iterable.

    Parameters
    ----------
    obj : Any
        The object to be converted into an iterable.

    Returns
    -------
    iterable : Iterable
        An iterable version of the input object. If the input is not already an
        iterable, it is returned as a single-element list.

    Examples
    --------
    >>> make_iterable(42)
    [42]

    >>> make_iterable([1, 2, 3])
    [1, 2, 3]

    >>> make_iterable("hello")
    ["hello"]  # Strings are not treated as iterables in this function
    """
    if isinstance(obj, str):
        return [obj]
    return obj if isinstance(obj, Iterable) else [obj]


def has_at_least_one(lst: list, value) -> bool:
    """
    Checks whether a given value appears at least once in a list.
    If the object passed is not iterable, it is converted to an interable,
    e.g. if lst = 5, the function transform lst = [lst].

    Parameters
    ----------
    lst : list
        The list to search.
    value : Any
        The value to look for in the list. If `None`, the function checks for the
        presence of `None` using identity comparison.

    Returns
    -------
    bool
        True if the value appears at least once in the list; False otherwise.
    """
    lst = make_iterable(lst)

    if value is None:
        return any(x is None for x in lst)
    else:
        return any(x == value for x in lst)


def flatten_dict(dic):
    return {
        f"{parent_key}/{key}": val
        for parent_key, nested in dic.items()
        for key, val in nested.items()
    }


def unflatten_dict(flat: dict) -> dict:
    nested = defaultdict(dict)
    for key, value in flat.items():
        if "/" in key:
            parent_key, field = key.split("/", 1)
            nested[parent_key][field] = value
    return dict(nested)


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

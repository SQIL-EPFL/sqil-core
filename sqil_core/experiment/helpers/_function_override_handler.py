from contextlib import contextmanager
from types import MethodType


class FunctionOverrideHandler:
    """
    A base class that allows functions to be overridden, restored, and temporarily
    replaced.

    Attributes
    ----------
    _default_functions : dict
        A dictionary storing the default functions of the object.
    _functions : dict
        A dictionary storing the current functions, which may include overridden
        versions.
    """

    def __init__(self):
        """
        Initializes the handler with empty dictionaries for default and overridden
        functions.
        """
        self._default_functions = {}
        self._functions = {}

    def override_function(self, func_name, new_func):
        """
        Overrides a function with a new implementation.

        Parameters
        ----------
        func_name : str
            The name of the function to override.
        new_func : function
            The new function implementation.
        """
        if func_name in self._functions:
            self._functions[func_name] = MethodType(new_func, self)
        else:
            raise AttributeError(f"Function '{func_name}' not found in the object.")

    def restore_function(self, func_name):
        """
        Restores a function to its default implementation.

        Parameters
        ----------
        func_name : str
            The name of the function to restore.
        """
        if func_name in self._default_functions:
            self._functions[func_name] = self._default_functions[func_name]
        else:
            raise AttributeError(
                f"Default for function '{func_name}' not found in the object."
            )

    @contextmanager
    def temporary_override(self, func_name, temp_func):
        """
        Temporarily overrides a function within a context manager.

        Parameters
        ----------
        func_name : str
            The name of the function to override temporarily.
        temp_func : function
            The temporary function implementation.

        Yields
        ------
        None
        """
        if func_name not in self._functions:
            raise AttributeError(f"Function '{func_name}' not found in the object.")

        original_func = self._functions[func_name]
        try:
            self._functions[func_name] = MethodType(temp_func, self)
            yield
        finally:
            self._functions[func_name] = original_func  # Restore the original

    def restore_all_functions(self):
        """
        Restores all overridden functions to their default implementations.
        """
        self._functions = self._default_functions.copy()

    def call(self, func_name, *args, **kwargs):
        """
        Calls a function by its name, passing any provided arguments.

        Parameters
        ----------
        func_name : str
            The name of the function to call.
        *args : tuple
            Positional arguments to pass to the function.
        **kwargs : dict
            Keyword arguments to pass to the function.

        Returns
        -------
        Any
            The return value of the called function.
        """
        if func_name in self._functions:
            return self._functions[func_name](*args, **kwargs)
        raise AttributeError(f"Function '{func_name}' not found in Instrument.")

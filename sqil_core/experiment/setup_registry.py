# from sqil_core.config_log import logger


# class SetupRegistry:
#     """
#     Registry for custom instrument setup functions.
#     Allows registering setup functions from anywhere in the code.
#     """

#     def __init__(self):
#         self.setup_functions = {}  # {instrument_id: setup_function}

#     def register_setup(self, instrument_id, setup_function):
#         """
#         Register a custom setup function for a specific instrument ID.

#         Args:
#             instrument_id: ID of the instrument (matches the ID in setup.yaml)
#             setup_function: Function that takes an instrument instance as its argument
#         """
#         self.setup_functions[instrument_id] = setup_function
#         logger.info(f"Registered custom setup for instrument '{instrument_id}'")

#     def unregister_setup(self, instrument_id):
#         """Remove custom setup for an instrument"""
#         if instrument_id in self.setup_functions:
#             del self.setup_functions[instrument_id]
#             logger.info(f"Unregistered custom setup for instrument '{instrument_id}'")

#     def has_custom_setup(self, instrument_id):
#         """Check if an instrument has custom setup registered"""
#         return instrument_id in self.setup_functions

#     def apply_setup(self, instrument_id, instrument):
#         """Apply custom setup if available, otherwise return False"""
#         if instrument_id in self.setup_functions:
#             try:
#                 self.setup_functions[instrument_id](instrument)
#                 logger.info(f"Applied custom setup for '{instrument_id}'")
#                 return True
#             except Exception as e:
#                 logger.error(f"Error in custom setup for '{instrument_id}': {str(e)}")
#                 return False
#         return False


# # Singleton instance
# setup_registry = SetupRegistry()

# from sqil_core.config_log import logger

# from ._events import after_experiment, before_experiment


# class EventHandlers:
#     """
#     Event handler class for LocalOscillator events.
#     This class manages automatic LO power control during experiments
#     with the ability to override the default behavior.
#     """

#     def __init__(self):
#         self.auto_control_enabled = True
#         self.local_oscillators = []

#         before_experiment.connect(self.handle_before_experiment)
#         after_experiment.connect(self.handle_after_experiment)

#     def register_local_oscillator(self, lo):
#         """Register a LocalOscillator instance to be controlled by the event handlers"""
#         if lo not in self.local_oscillators:
#             self.local_oscillators.append(lo)
#             logger.info(f"Registered {lo.name} for automatic experiment control")

#     def unregister_local_oscillator(self, lo):
#         """Unregister a LocalOscillator from automatic control"""
#         if lo in self.local_oscillators:
#             self.local_oscillators.remove(lo)
#             logger.info(f"Unregistered {lo.name} from automatic experiment control")

#     def enable_auto_control(self):
#         """Enable automatic control of local oscillators during experiments"""
#         self.auto_control_enabled = True
#         logger.info("Enabled automatic LO control for experiments")

#     def disable_auto_control(self):
#         """Disable automatic control of local oscillators during experiments"""
#         self.auto_control_enabled = False
#         logger.info("Disabled automatic LO control for experiments")

#     # turn on all registered LOs on start event
#     def handle_before_experiment(self, sender):
#         if not self.auto_control_enabled:
#             logger.info(
#                 "Automatic LO control is disabled - skipping pre-experiment activation"
#             )
#             return

#         logger.info("Turning on local oscillators before experiment")
#         for lo in self.local_oscillators:
#             try:
#                 lo.on()
#                 logger.info(f"Successfully turned on {lo.name}")
#             except Exception as e:
#                 logger.error(f"Failed to turn on {lo.name}: {str(e)}")

#     # turn off all registered LOs on stop event
#     def handle_after_experiment(self, sender):
#         if not self.auto_control_enabled:
#             logger.info(
#                 "Automatic LO control is disabled - skipping post-experiment deactivation"
#             )
#             return

#         logger.info("Turning off local oscillators after experiment")
#         for lo in self.local_oscillators:
#             try:
#                 lo.off()
#                 logger.info(f"Successfully turned off {lo.name}")
#             except Exception as e:
#                 logger.error(f"Failed to turn off {lo.name}: {str(e)}")


# # Create a singleton instance to use in the LO class
# lo_event_handlers = EventHandlers()

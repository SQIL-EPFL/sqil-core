from sqil_core.experiment.setup_registry import setup_registry


# these are example setup methods
def custom_sgsa_setup(lo):
    """Custom setup for the SGSA local oscillator"""
    lo.frequency(12.3e9)
    lo.power(-25)


def custom_sc_setup(lo):
    """Custom setup for SignalCore"""
    lo.setup(frequency=50)  # Call original setup with custom params
    lo.power(-15)


# it registers the custom setup for all experiments bc it is at a general level

# setup_registry.register_setup('sgsa', custom_sgsa_setup)
# setup_registry.register_setup('SC5511A', custom_sc_setup)

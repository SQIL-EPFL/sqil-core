from blinker import NamedSignal

before_experiment = NamedSignal("before_experiment")
after_experiment = NamedSignal("after_experiment")

before_sequence = NamedSignal("before_sequence")
after_sequence = NamedSignal("after_sequence")


def clear_signal(signal):
    """Removes all listeners (receivers) from an event (signal)."""
    receivers = list(signal.receivers.values())
    for receiver in receivers:
        signal.disconnect(receiver)


def one_time_listener(signal, func):
    """Listens for an event only once."""

    def wrapper(*args, **kwargs):
        signal.disconnect(wrapper)
        return func(*args, **kwargs)

    signal.connect(wrapper, weak=False)
    return wrapper

from blinker import NamedSignal

before_experiment = NamedSignal("before_experiment")
after_experiment = NamedSignal("after_experiment")

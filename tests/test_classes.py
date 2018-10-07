import numpy as np
from generation import make_stimulus
from data_classes import *

# %%  Mocking up some data
frequency = 15
number_of_neurons = 30
stimulus_duration = 1000

# Simulating neurons with a given fixed frequency
fixed_event_times = np.sort(np.random.rand(number_of_neurons, frequency) * stimulus_duration, 1)
# Simulating neurons with a given average frequency
expected_in_duration = frequency * stimulus_duration / 1000
nonfixed_event_times = np.array(
    [
        np.sort(np.random.rand(  # Generate some random number of spikes according to given frequency
            int(np.abs((np.random.randn() * expected_in_duration / 6 + expected_in_duration)))
        ) * stimulus_duration)  # Multiply for scaling
        for i in range(number_of_neurons)  # Generate specified number of neurons
    ]
)
# %% Testing neuron
neuron_fixed_times = fixed_event_times[0]
neuron_nonfixed_times = nonfixed_event_times[0]
n_fixed = Neuron(
    events=neuron_fixed_times,
    frequency_generated=frequency,
    stimulus_duration=stimulus_duration,
)
n_nonfixed = Neuron(
    events=neuron_nonfixed_times,
    frequency_generated=frequency,
    stimulus_duration=stimulus_duration,
)
# %% Testing Stimulus
neuron_list_fixed = [Neuron(row, frequency, stimulus_duration) for row in fixed_event_times]
neuron_list_nonfixed = [Neuron(row, frequency, stimulus_duration) for row in nonfixed_event_times]

stim_fixed = Stimulus(neuron_list_fixed, 15, 1000)
stim_non_fixed = Stimulus(neuron_list_nonfixed, 15, 1000)



import numpy as np
from generation import make_stimulus
from data_classes import *

# %%  Mocking up some data
frequency = 15
number_of_neurons = 30
stimulus_duration = 1000
number_of_stimuli = 15

# Simulating neurons with a given fixed frequency
fixed_event_times = np.sort(np.random.rand(number_of_stimuli, number_of_neurons, frequency) * stimulus_duration, 2)
fixed_event_times_other = np.sort(np.random.rand(number_of_stimuli, number_of_neurons, frequency) * stimulus_duration,
                                  2)
# Simulating neurons with a given average frequency
expected_in_duration = frequency * stimulus_duration / 1000
nonfixed_event_times = np.array(
    [
        [np.sort(np.random.rand(  # Generate some random number of spikes according to given frequency
            int(np.abs((np.random.randn() * expected_in_duration / 6 + expected_in_duration)))
        ) * stimulus_duration)  # Multiply for scaling
         for _ in range(number_of_neurons)]  # Generate specified number of neurons
        for _ in range(number_of_stimuli)
    ]
)
nonfixed_event_times_other = np.array(
    [
        [np.sort(np.random.rand(  # Generate some random number of spikes according to given frequency
            int(np.abs((np.random.randn() * expected_in_duration / 6 + expected_in_duration)))
        ) * stimulus_duration)  # Multiply for scaling
         for _ in range(number_of_neurons)]  # Generate specified number of neurons
        for _ in range(number_of_stimuli)
    ]
)
# %% Testing neuron
neuron_fixed_times = fixed_event_times[0][0]
neuron_nonfixed_times = nonfixed_event_times[0][0]
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
neuron_list_fixed = [Neuron(row, stimulus_duration, frequency) for row in fixed_event_times[0]]
neuron_list_nonfixed = [Neuron(row, stimulus_duration, frequency ) for row in nonfixed_event_times[0]]

stim_fixed = Stimulus(neuron_list_fixed, stimulus_duration, frequency)
stim_nonfixed = Stimulus(neuron_list_nonfixed, stimulus_duration, frequency)

stim_fresh_fixed = Stimulus.make(frequency, number_of_neurons, stimulus_duration, exact_frequency=True)
stim_fresh_nonfixed = Stimulus.make(frequency, number_of_neurons, stimulus_duration)
# %% Testing stimuli set
stimuli_list_fixed = [[Neuron(row, stimulus_duration, frequency) for row in fixed_event_times[i]] for i in
                      range(number_of_stimuli)]
stimuli_list_fixed_other = [[Neuron(row, stimulus_duration, frequency) for row in fixed_event_times_other[i]] for i in
                            range(number_of_stimuli)]

stimuli_list_nonfixed = [[Neuron(row, stimulus_duration, frequency) for row in nonfixed_event_times[i]] for i in
                         range(number_of_stimuli)]
stimuli_list_nonfixed_other = [[Neuron(row, stimulus_duration, frequency) for row in nonfixed_event_times_other[i]] for
                               i in range(number_of_stimuli)]

stimset_fixed = StimuliSet(stimuli=stimuli_list_fixed, labels=[*[0] * len(stimuli_list_fixed)],
                            frequency_generated=frequency, stimulus_duration=stimulus_duration, event_type='spikes')
stimset_nonfixed = StimuliSet(stimuli=stimuli_list_nonfixed, labels=[*[0] * len(stimuli_list_nonfixed)],
                               frequency_generated=frequency, stimulus_duration=stimulus_duration, event_type='spikes')
stimset_fixed_other = StimuliSet(stimuli=stimuli_list_fixed_other, labels=[*[1] * len(stimuli_list_fixed_other)],
                                  frequency_generated=frequency, stimulus_duration=stimulus_duration,
                                  event_type='spikes')
stimset_nonfixed_other = StimuliSet(stimuli=stimuli_list_nonfixed_other, labels=[*[1] * len(stimuli_list_nonfixed_other)],
                                     frequency_generated=frequency, stimulus_duration=stimulus_duration,
                                     event_type='spikes')

stimset_combined_fixed = stimset_fixed + stimset_fixed_other
stimset_combined_nonfixed = stimset_nonfixed + stimset_nonfixed_other
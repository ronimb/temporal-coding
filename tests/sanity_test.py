from generation.conversion import convert_stimulus, convert_stimuli_set
from generation.transform import fixed_release, stochastic_release, forward_shift, symmetric_interval_shift
from generation import make_stimulus, make_set_from_specs, make_set_from_stimuli
import numpy as np
# %% Global parameters
# For stimuli
stimulus_duration = 500
frequency = 50
number_of_neurons = 30

# For transformations
num_transformed = 100

release_duration = 5
number_of_vesicles=20

# %% Testing single stimulus generation
params_single_stim_nonfixed = dict(
    frequency=frequency,
    number_of_neurons=number_of_neurons,
    stimulus_duration=stimulus_duration
)

params_single_stim_fixed = dict(
    frequency=frequency,
    number_of_neurons=number_of_neurons,
    stimulus_duration=stimulus_duration,
    exact_frequency=True
)

single_stim_fixed = make_stimulus(**params_single_stim_fixed)
single_stim_nonfixed = make_stimulus(**params_single_stim_nonfixed)
# %% Testing stimuliset generation
params_forward_shift = dict(
    stimulus_duration=stimulus_duration,
    max_temporal_shift=5,
    num_transformed=num_transformed
)
params_symmetric_interval = dict(
    stimulus_duration=stimulus_duration,
    interval=(1, 3),
    num_transformed=num_transformed
)
params_fixed_release = dict(
    release_duration=release_duration,
    number_of_vesicles=number_of_vesicles,
    stimulus_duration=stimulus_duration,

)

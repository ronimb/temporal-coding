from generation import transform
from generation.make_stimuli_set import make_set_from_specs
from tools import calc_stimuli_distance
import numpy as np
from generation.conversion import convert_stimuli_set
import matplotlib.pyplot as plt
import seaborn as sns

# %%
creation_params = dict(
    frequency=15,
    number_of_neurons=30,
    stimulus_duration=500,
)

set_transform_function = transform.stochastic_release
set_transform_params = dict(
    release_duration=5,
    number_of_vesicles=20,
    stimulus_duration=creation_params['stimulus_duration'],
    release_probability=1,
    num_transformed=50
)

origin_transform_function = transform.symmetric_interval_shift
origin_transform_params = dict(
    stimulus_duration=creation_params['stimulus_duration'],
    interval=(1, 3)
)
# %% Generating test stimuli set
stimuli_set = make_set_from_specs(**creation_params, set_size=200,
                                   set_transform_function=set_transform_function,
                                   set_transform_params=set_transform_params,
                                   origin_transform_function=origin_transform_function,
                                   origin_transform_params=origin_transform_params)
label_a_inds = stimuli_set.stimuli[stimuli_set.labels == 0]
label_b_inds = stimuli_set.stimuli[stimuli_set.labels == 1]
d = np.array(
    [calc_stimuli_distance(a, b, stimulus_duration=creation_params['stimulus_duration'])
     for a, b in zip(label_a_inds, label_b_inds)])

# %% Testing conversion
func_converted = convert_stimuli_set(stimuli_set)
method_converted = stimuli_set.convert_for_tempotron()


from generation import make_stimulus
from generation import transform
from generation.make_stimuli_set import make_set_from_specs, make_set_from_stimuli
from tools import calc_stimuli_distance
import numpy as np
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
    interval=(3, 7)
)
# %%

stimuli_set1 = make_set_from_specs(**creation_params, set_size=200,
                                   set_transform_function=set_transform_function,
                                   set_transform_params=set_transform_params)
s1a = stimuli_set1['stimulus'][stimuli_set1['label'] == 0]
s1b = stimuli_set1['stimulus'][stimuli_set1['label'] == 1]
d1 = np.array([calc_stimuli_distance(a, b, stimulus_duration=creation_params['stimulus_duration']) for a,b in zip(s1a, s1b)])

stimuli_set2 = make_set_from_specs(**creation_params, set_size=200,
                                   set_transform_function=set_transform_function,
                                   set_transform_params=set_transform_params,
                                   origin_transform_function=origin_transform_function,
                                   origin_transform_params=origin_transform_params)
s2a = stimuli_set2['stimulus'][stimuli_set2['label'] == 0]
s2b = stimuli_set2['stimulus'][stimuli_set2['label'] == 1]
d2 = np.array([calc_stimuli_distance(a, b, stimulus_duration=creation_params['stimulus_duration']) for a,b in zip(s2a, s2b)])

stim_a = make_stimulus(**creation_params)
stim_b = make_stimulus(**creation_params)

stimuli_set3 = make_set_from_stimuli((stim_a, stim_b), 
                                     stimulus_duration=creation_params['stimulus_duration'],
                                     set_size=200, 
                                     set_transform_function=set_transform_function,
                                     set_transform_params=set_transform_params)
s3a = stimuli_set3['stimulus'][stimuli_set3['label'] == 0]
s3b = stimuli_set3['stimulus'][stimuli_set3['label'] == 1]
d3 = np.array([calc_stimuli_distance(a, b, stimulus_duration=creation_params['stimulus_duration']) for a,b in zip(s3a, s3b)])

all_ds = -np.log10([d1, d2, d3])

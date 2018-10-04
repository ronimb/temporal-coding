from generation.make_stimuli_set import make_set_from_specs
from generation.conversion import  convert_stimuli_set
from generation import transform
# %%
set_size = 200

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
stimuli_set = make_set_from_specs(**creation_params, set_size=set_size,
                                   set_transform_function=set_transform_function,
                                   set_transform_params=set_transform_params)

convert_stimuli_set(stimuli_set)
# %%

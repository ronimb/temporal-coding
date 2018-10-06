from classification import Tempotron
from generation import transform, make_set_from_specs
from generation.conversion import convert_stimuli_set
from copy import copy
from Experiment import Experiment
# %%
model_params = dict(
    tau=2,  # Voltage time decay constant
    threshold=0.005  # Threshold for firing, firing will result in a "1" classification
)

# Model training parameters
model_training_params = dict(
    repetitions=10,  # Number of training batches from training set to train with
    batch_size=50,  # Size of each training batch
    learning_rate=1e-3,  # Learning rate for the training stage
    training_set_size=100,  # Size of the set used for training, rest of the  set is used for testing
)

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

converted_stimuli_set = copy(stimuli_set)
convert_stimuli_set(converted_stimuli_set)
# %%
T = Tempotron(50,**model_params, stimulus_duration=stimuli_set.stimulus_duration)
# %%
e1 = Experiment(stimuli_set=stimuli_set,
               model=T)
e2 = Experiment(stimuli_set=stimuli_set,
                model=model_params)
e3 = Experiment(stimuli_set=converted_stimuli_set,
                model=T)
e4 = Experiment(stimuli_set=converted_stimuli_set,
                model=model_params)

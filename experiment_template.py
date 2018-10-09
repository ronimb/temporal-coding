# TODO: Write documentation for experiment_template
"""

"""
from generation import transform
from Experiment import Experiment
from tools import save_obj

# %% Parameter specification
# System parameters
save_location = '/home/ron/OneDrive/Documents/Masters/Parnas/temporal-coding/Results/'
save_name = 'test_experiment'
# Base experiment parameters
number_of_repetitions = 2  # Number of times to repeat the experiment with the exact same conditions
set_size = 200  # The size of the set(s) to generate for this experiment
fraction_training = 0.5 # Fraction of samples to be used in training, the rest go to testing

# Machine learning model parameters
model_params = dict(
    tau=2,  # Voltage time decay constant
    threshold=0.005  # Threshold for firing, firing will result in a "1" classification
)

# Model training parameters
training_params = dict(
    training_repetitions=3,  # Number of training batches from training set to train with
    batch_size=10,  # Size of each training batch
    learning_rate=1e-3,  # Learning rate for the training stage
    fraction_training=fraction_training,  # Fraction of set to be used for training
)
# These are the basic parameters for the spiking neuron stimuli from which the experiment originates
# Changes to this might be required for different modes of generation (e.g. comparing different frequencies)
stimuli_creation_params = dict(
    frequency=50,
    number_of_neurons=5,
    stimulus_duration=100,
    set_size=set_size
)

# These parameters control how similar the basic two original stimuli will be, if random generation is desired,
# Assign each of these with None
origin_transform_function = transform.symmetric_interval_shift  # The function to use for transfrom stimulus_a to stimulus_b
origin_transform_params = dict(  # The parameters with which to execute the specified transformation function
    stimulus_duration=stimuli_creation_params['stimulus_duration'],
    interval=(1, 3)
)

# These parameters set the function that will be used to generate the transformed version of both stimuli
set_transform_function = transform.stochastic_release  # The function to use when generating each transformed stimulus
set_transform_params = dict(  # The parameters with which to execute the specified transformation function
    release_duration=5,
    number_of_vesicles=20,
    stimulus_duration=stimuli_creation_params['stimulus_duration'],
    release_probability=1,
)

# %%
experiment = Experiment(
    stimuli_creation_params=stimuli_creation_params,
    model=model_params,
    training_params=training_params,
    origin_transform_function=origin_transform_function,
    origin_transform_params=origin_transform_params,
    set_transform_function=set_transform_function,
    set_transform_params=set_transform_params,
    repetitions=number_of_repetitions
)
experiment.run()
experiment.save(save_location, save_name)
# TODO: Write documentation for experiment_template
"""

"""
from generation import make_set_from_specs
from generation import transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools import check_folder, gen_datestr

# %% Parameter specification
# System parameters
stimuli_save_location = ''  # Location at which to save stimuli_sets generated for current experiment, LEAVE EMPTY FOR NO-SAVING
results_save_location = ''  # Location at which the results file will be saved, include filename

# Base experiment parameters
number_of_repetitions = 30  # Number of times to repeat the experiment with the exact same conditions
set_size = 200  # The size of the set(s) to generate for this experiment

training_set_size = 100  # Size of the set used for training
test_set_size = 100  # Size of the set used for testing
number_of_training_repetitions = 10  # Number of training batches from training set to train with
training_batch_size = 50  # Size of each training batch
learning_rate = 1e-3 # Learning rate for the training stage

# These are the basic parameters for the spiking neuron stimuli from which the experiment originates
# Changes to this might be required for different modes of generation (e.g. comparing different frequencies)
creation_params = dict(
    frequency=15,
    number_of_neurons=30,
    stimulus_duration=500,
)

# These parameters control how similar the basic two original stimuli will be, if random generation is desired,
# Assign each of these with None
origin_transform_function = transform.symmetric_interval_shift  # The function to use for transfrom stimulus_a to stimulus_b
origin_transform_params = dict(  # The parameters with which to execute the specified transformation function
    stimulus_duration=creation_params['stimulus_duration'],
    interval=(3, 5)
)

# These parameters set the function that will be used to generate the transformed version of both stimuli
set_transform_function = transform.stochastic_release  # The function to use when generating each transformed stimulus
set_transform_params = dict(  # The parameters with which to execute the specified transformation function
    release_duration=5,
    number_of_vesicles=20,
    stimulus_duration=creation_params['stimulus_duration'],
    release_probability=1,
    num_transformed=50
)

# %%

# Create pandas array to contain number_of_repetitions items with pre, post, diff and distance
# Iterate number_of_repetitions times
#   Create stimuli set
#   TODO: Change the following behaviour, wrap the object in a training function
#   Create tempotron object
#   Create classification network
#   Calculate accuracy  pre accuracy over entire set
#   Divide set into training and test_sets (use indexes to conserve memory)
#   Train with training set
#   Calculate post accuracy over test_set
#   Calculate diff
# Do summary analytics and some plotting with the resulting array

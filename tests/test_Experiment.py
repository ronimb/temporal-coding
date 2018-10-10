"""

"""
from generation import transform
from Experiment import Experiment
from tools import check_folder, save_obj, load_obj
import os
# %% Parameter specification
# System parameters
save_folder = 'exp_save'
experiment_name = 'exptest'  # This can be left empty
# Base experiment parameters
number_of_repetitions = 3  # Number of times to repeat the experiment with the exact same conditions
set_size = 100  # The size of the set(s) to generate for this experiment
fraction_training = 0.5 # Fraction of samples to be used in training, the rest go to testing

# Machine learning model parameters
model_params = dict(
    tau=2,  # Voltage time decay constant
    threshold=0.05  # Threshold for firing, firing will result in a "1" classification
)

# Model training parameters
training_params = dict(
    training_repetitions=5,  # Number of training batches from training set to train with
    batch_size=15,  # Size of each training batch
    learning_rate=1e-3,  # Learning rate for the training stage
    fraction_training=fraction_training,  # Fraction of set to be used for training
)
# These are the basic parameters for the spiking neuron stimuli from which the experiment originates
# Changes to this might be required for different modes of generation (e.g. comparing different frequencies)
stimuli_creation_params = dict(
    frequency=15,
    number_of_neurons=30,
    stimulus_duration=500,
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

# Handle empty experiment name
if experiment_name:
    experiment_name = experiment_name + '_'
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
# %% Test integrated saving
# Run experiment
experiment.run()
# Save experiment
experiment.save(save_folder, experiment_name)
#  test script saving
with open(__file__, 'r') as file:
    this_file = file.read()
script_save_location = os.path.join(save_folder, f'{experiment_name}experiment_template.py')
with open(script_save_location, 'w') as template_file:
    template_file.write(this_file)
    template_file.close()

# %% test saving
#
# tempotron_params_location = os.path.join(save_folder, f'params{experiment_name}.tempotron')
# experiment_params_location = os.path.join(save_folder, f'params{experiment_name}.experiment')
# t = experiment.model
# # %% saving tempotron
# # saving tempotron main parameters
# mdl_params_dict = t.__dict__
#
# mdl_save_params = ['number_of_neurons', 'tau', 'threshold', 'stimulus_duration', 'weights', 'eqs']
# mdl_params_savedict = {key: mdl_params_dict[key] for key in mdl_save_params}
#
#
#
# mdl_networks = mdl_params_dict['networks']
# # Excluding the plotting network
# if 'plot' in mdl_networks:
#     mdl_networks.pop('plot')
# network_sizes = {name: network['number_of_stimuli'] for name, network in mdl_networks.items()}
# mdl_params_savedict['network_sizes'] = network_sizes
# save_obj(mdl_params_savedict, tempotron_params_location)
# # %% saving experiment parameters
# exp_params_dict = experiment.__dict__
#
# exp_save_params = ['stimuli_creation_params', 'training_params',
#                    'origin_transform_function', 'origin_transform_params',
#                    'set_transform_function', 'set_transform_params', 'repetitions',
#                    'stimuli_sets', 'rep_times', 'results']
#
# exp_params_savedict = {key: exp_params_dict[key] for key in exp_save_params}
#
# save_obj(exp_params_savedict, experiment_params_location)
# # %% test loading
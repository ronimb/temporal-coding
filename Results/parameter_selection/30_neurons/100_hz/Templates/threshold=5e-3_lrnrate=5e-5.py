# TODO: Write documentation for experiment_template
"""

"""
from generation import transform
from Experiment import Experiment
from os import path
from tools import sec_to_time, gen_datestr
from time import time
from tools import check_folder

# Record start time for this experiment
start_time = time()
start_date = gen_datestr()
print(f"---- Started experiment : {start_date}")
# %% Specific parameters for parameter selection, easier to control from here
threshold = 5e-3
learning_rate = 5e-5
# %% Parameter specification
# Set location of root report (results) folder
# report_folder = '/home/ron/OneDrive/Documents/Masters/Parnas/temporal-coding/Results/' # For laptop
report_folder = '/home/ronimber/PycharmProjects/temporal-coding/Results/' # For Google-Compute-Engine
# Optional: Set ordered sub-folders according to condition keywords
condition_folders = ('parameter_selection', '30_neurons', '100_hz', 'interval_(1-3)')  # If empty, saves at report folder

# Determine name for save folder
save_folder = path.join(report_folder, *condition_folders)
# Make sure the folder exists, create with all subfolders if it does not
check_folder(save_folder)

# Optional: Set custom name for the experiment, useful when running several experiments with some shared conditions,
# All experiment related files will have this name prepended to them with an underscore
experiment_name = f'thresh={threshold}_lrnRate={learning_rate}'  # This can be left empty to save without experiment name

# Base experiment parameters
number_of_repetitions = 10  # Number of times to repeat the experiment with the exact same conditions
set_size = 200  # The size of the set(s) to generate for this experiment
fraction_training = 0.5  # Fraction of samples to be used in training, the rest go to testing
stimulus_duration = 500 # Maximal duration of the stimulus

# Machine learning model parameters
model_params = dict(
    tau=2,  # Voltage time decay constant
    threshold=threshold  # Threshold for firing, firing will result in a "1" classification
)

# Model training parameters
training_params = dict(
    training_repetitions=15,  # Number of training batches from training set to train with
    batch_size=50,  # Size of each training batch
    learning_rate=learning_rate,  # Learning rate for the training stage
    fraction_training=fraction_training,  # Fraction of set to be used for training
)
# These are the basic parameters for the spiking neuron stimuli from which the experiment originates
# Changes to this might be required for different modes of generation (e.g. comparing different frequencies)
stimuli_creation_params = dict(
    frequency=100,
    number_of_neurons=30,
    stimulus_duration=stimulus_duration,
    set_size=set_size
)

# These parameters control the degree of similarity between the two original stimuli
# Essentialy, one is a transformed version of the other by some function with some specified parameters
origin_transform_function = transform.symmetric_interval_shift  # The function to use for transfrom stimulus_a to stimulus_b
origin_transform_params = dict(  # The parameters with which to execute the specified transformation function
    stimulus_duration=stimulus_duration,
    interval=(1, 3)
)

# These parameters determine what transformation will be applied to the original stimuli to generate
# the StimuliSet for the experiment (consisting of set_size/2 transformed samples of each)
set_transform_function = transform.stochastic_release  # The function to use when generating each transformed stimulus
set_transform_params = dict(  # The parameters with which to execute the specified transformation function
    release_duration=5,
    number_of_vesicles=20,
    stimulus_duration=stimulus_duration,
    release_probability=1,
)

# Handle empty experiment name
if experiment_name:
    experiment_name = experiment_name + '_'
# %% Running and controlling the experiment
# Set up the experiment
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
# Run the experiment
experiment.run()

# Saving all experiment data
experiment.save(save_folder, experiment_name)

# Save backup copy of this file
with open(__file__, 'r') as file:
    this_file = file.read()

with open(path.join(save_folder, f'{experiment_name}experiment_template.py'), 'w') as template_file:
    template_file.write(this_file)
    template_file.close()

# %% Report finish
end_time = time()
runtime = sec_to_time(end_time - start_time)
end_date = gen_datestr()
print(f"EXPERIMENT FINISHED:\n\tStarted: {start_date}\n\tEnded: {end_date}\n\tTook: {runtime}")

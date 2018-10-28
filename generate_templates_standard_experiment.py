import string
import os
from tools import dict_product, check_folder

# %%
template_file = '/home/ron/OneDrive/Documents/Masters/Parnas/temporal-coding/Templates/standard_experiment.template'

# %% define conditions for template generation
# Set the main report folder where templates will be saved
# report_folder = '/home/ron/OneDrive/Documents/Masters/Parnas/temporal-coding/results/experiments'  # Ron-Laptop
report_folder = '/home/ronimber/PycharmProjects/temporal-coding/results/experiments' # Google Compute Engine

experiment_set_name = ''  # Optional name for experiment set, if provided will create as a folder in the reports folder

# Define conditions for which to generate the templates, subfolders will be created by number_of_neurons and frequency
main_conditions = dict(
    number_of_neurons=(30,),
    frequency=(15, 50, 100),
    interval=((1, 3), (3, 5), (5, 7), (7, 9)),
    release_probability=(0.25, 0.5, 0.75, 1),
    release_duration=(3, 6, 9)
)

# Define shared conditions, this generally doesn't need to be touched
shared_conditions = dict(
    number_of_repetitions=30,  # Number of repetitions from each experimental condition
    # Model parameters (note that threshold is separated from tau due to the differences between stimuli sizes)
    tau=2,  # Time decay constant for tempotron
    # Set creation parameters
    set_size=200,  # Total number of transformed stimuli in set, half from each origin
    stimulus_duration=500,  # Maximal duration of the stimulus
    # Set transformation parameters
    number_of_vesicles=20,  # Number of vesicles released in response to each spike
    # Training parameters
    training_steps=30,  # Number of training repetitions
    batch_size=50,  # Number of stimuli to use for each training repetitions
    fraction_training=0.5,  # # Fraction of set to be used for training
    # The following two parameters may be set manually, however if left empty (i.e '' or None)
    # Pre determined optimized parameters will be used
    threshold=None,  # Threshold for firing of tempotron model
    learning_rate=None,  # Learning rate for the training stage
)

number_of_neuron_params = {
    30: {'threshold': 0.025, 'learning_rate': 5e-5},
    50: {'threshold': None, 'learning_rate': None},  # TBD
    100: {'threshold': None, 'learning_rate': None}  # TBD
}
# %%
save_folder = os.path.join(report_folder, experiment_set_name)
# Read template file
with open(template_file, 'r') as f:
    template = string.Template(f.read())

conditions_list = dict_product(main_conditions)  # Creating list with dictionary for each condition
for condition in conditions_list:
    condition_params = shared_conditions.copy()  # Grabbing the shared conditios
    condition_params.update(condition)  # Adding current conditions
    # Setting the threshold and learning rate
    condition_params['threshold'] = number_of_neuron_params[condition['number_of_neurons']]['threshold']
    condition_params['learning_rate'] = number_of_neuron_params[condition['number_of_neurons']]['learning_rate']

    # Determine location and name of generated template file
    condition_report_folder = os.path.join(save_folder,
                                        f"{condition['number_of_neurons']}_neurons",
                                        f"{condition['frequency']}_hz")
    check_folder(condition_report_folder)
    condition_name = f"interval={condition_params['interval']}_relprob={condition_params['release_probability']}_relduration={condition_params['release_duration']}_"
    # Adding to params dictionary
    condition_params['report_folder'] = condition_report_folder
    condition_params['condition_name'] = condition_name
    target_file = os.path.join(condition_report_folder, f'{condition_name}experiment_template.py')

    # Creating template text
    target_contents = template.substitute(condition_params)

    # Saving template
    with open(target_file, 'w') as file:
        file.write(target_contents)

# target_contents = template.substitute(substitutions)
# with open(target, 'w') as f:
#     f.write(target_contents)

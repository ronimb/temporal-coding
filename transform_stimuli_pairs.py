"""
This script operates on folders containing .npy files where each file contains numerous stimuli pairs
from a given number_of_neurons x frequency x shift_interval condition.

The folder structure from which data is loaded is as follows:

{Mon}_{Day_of_month}/source_pairs/{number_of_neurons}_neurons/
These folders contain the files for all conditions generated with a specific number of neurons,
files in this folder follow the following format:
{frequency}Hz_interval={interval_low}-{interval_high}ms.npy

For example:
A file generated on september 9th with 150 neurons firing at 15Hz and in which one member of the pair is shifted by ±(3 - 5) milliseconds
with respect to the other will be located at:

Sep_09/source_pairs/150/15Hz_interval=3-5ms.npy
"""

from generation.transform import temporal_shift, fixed_release, stochastic_release
import os
import numpy as np
from tools import check_folder, gen_datestr
from itertools import product

# %%
load_folder = '/mnt/disks/data/Oct_01/source_pairs'
save_folder = '/mnt/disks/data/Oct_01/transformed'

# Conditions of stimuli in the folders to be loaded from
frequencies = [15, 50, 100]
number_of_neurons = [30, 150, 500]
stimulus_duration = 500  # Units: ms
temporal_shift_intervals = ((3, 5), (3, 7),
                            (3, 10), (3, 15))

# Transformation parameters
num_transformed = 100  # Number of transformed stimuli from each stimulus
transformation_function = stochastic_release  # Function to use for transforming each stimulus, choose from the existing functions in the generation.trasnform module

# Parameters to use for all transformations
fixed_transformation_params = {'number_of_vesicles': 20,
                               'max_stimulus_duration': stimulus_duration,
                               'num_transformed': num_transformed,
                               }  # Make sure these parameters are the correct ones for the chosen function

# Parameters used to create different transformation conditions
conditional_transformation_params = {'release_probability': (0.25, 0.5, 0.75, 1),
                                     'release_duration': (3, 6, 9)}

# Re-organizing the parameters for use in function
condition_list = list((dict(zip(conditional_transformation_params, x))
                       for x in product(*conditional_transformation_params.values())))

# %%
for num_neur in number_of_neurons:
    print(f'Started working on transform stimuli with {num_neur} neurons - {gen_datestr()}')
    stimuli_folder = os.path.join(load_folder, f'{num_neur}_neurons')  # Folder from which to actually load the stimuli
    for freq in frequencies:
        print(f'\t{freq}Hz stimuli - {gen_datestr()}')
        # Create folder for saving transformed stimuli
        frequency_folder = os.path.join(save_folder, f'{num_neur}_neurons',
                                        f'{freq}Hz')  # Folder for saving transformed stimuli of current num_nuer x frequency combination
        check_folder(frequency_folder)
        for interval in temporal_shift_intervals:
            print(f'\t\tinterval = {interval} - {gen_datestr()}')
            # Load the file containing all the pairs in the current condition
            all_pairs_file_location = os.path.join(stimuli_folder,
                                                   f'{freq}Hz_interval={interval[0]}-{interval[1]}ms.npy')
            all_pairs = np.load(all_pairs_file_location)
            for i, pair in enumerate(all_pairs):
                # Now, from each pair of stimuli we create for each member of the pair num_transformed transformed
                # Stimuli by applying the transformation_function. we do this for each of the conditions specified
                # By the transformation parameters in conditional_transformation_params

                for condition_params in condition_list:
                    # For each condition, we need to create a dictionary of parameters for the transformation function
                    current_params = fixed_transformation_params.copy()  # First copy the fixed params
                    current_params.update(condition_params)  # Add the parameters of the current condition
                    # Create transformed versions of each stimulus
                    transformed_a = transformation_function(stimulus=pair['stimulus_a'], **current_params)
                    transformed_b = transformation_function(stimulus=pair['stimulus_b'], **current_params)
                    # Creating single array of stimuli from both pair members
                    all_transformed_stimuli = np.array(np.zeros(num_transformed * 2),
                                                       dtype={'names': ('stimulus', 'label'),
                                                              'formats': (object, bool)})
                    all_transformed_stimuli['stimulus'] = [*transformed_a, *transformed_b]
                    all_transformed_stimuli['label'] = [*[0] * num_transformed, *[1] * num_transformed]

                    # Save transformed stimuli
                    save_file_name = f"interval={interval[0]}-{interval[1]}ms_{''.join([f'({key}={val})' for key, val in condition_params.items()])}_#{i}.npy"
                    save_file_location = os.path.join(frequency_folder, save_file_name)
                    with open(save_file_location, 'wb') as file:
                        np.save(file, all_transformed_stimuli)

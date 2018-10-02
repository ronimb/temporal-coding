'''
This script is used to generate pairs of stimuli where one of the stimuli
is a temporally shifted version of the other.
The difference between the stimuli is defined by how much
the spikes in the original stimulus are allowed to shift, for
this purpose a symmetric interval shift is used, allowing
each spike to move forward or backward in time within a specified interval.
'''
# %%
from generation import make_stimulus
from generation.transform import symmetric_interval_shift
from tools import calc_stimuli_distance, check_folder, gen_datestr
import numpy as np
import os
import datetime

# %%
# Set up file saving
target_folder = '/mnt/disks/data'  # Set up where the sets will be saved
check_folder(target_folder)

# Stimuli generation parameters
frequencies = [15, 50, 100]
number_of_neurons = [30, 150, 500]
duration = 500  # Units: ms

# Pair generation parameters
number_of_pairs = 100  # Number of stimulus pairs for each interval condition
# Intervals to use for temporal shifting
temporal_shift_intervals = [[3, 5], [3, 7],
                            [3, 10], [3, 15]]  # Units: Seconds

# %%
current_date = gen_datestr(with_time=False)
for num_neur in number_of_neurons:
    start_time = gen_datestr()
    print(f'Creating stimuli with {num_neur} neurons -  - started: {start_time}')

    current_folder = os.path.join(target_folder, current_date, 'source_pairs',
                                  f'{num_neur}_neurons')
    check_folder(current_folder)
    for freq in frequencies:
        print(f'\tNow working on stimuli of {freq}Hz')
        for interval in temporal_shift_intervals:
            print(f'\t\tCreating stimuli pairs by shifting ±{interval}ms')
            # Creating placeholders for all stimuli in current condition
            orig_stimuli = []
            shifted_stimuli = []
            distances = []
            for i in range(number_of_pairs):
                # Generate original stimulus
                orig_stimulus = make_stimulus(frequency=freq, num_neurons=num_neur, stimulus_duration=duration)
                # Generate shifted stimulus
                shifted_stimulus = symmetric_interval_shift(stimulus=orig_stimulus, stimulus_duration=duration,
                                                            interval=interval, num_transformed=1)
                # Calculate distance metric between two stimuli for reference
                distance = calc_stimuli_distance(stimulus_a=orig_stimulus,
                                                 stimulus_b=shifted_stimulus,
                                                 duration=duration)
                # Append data to placeholder lists
                orig_stimuli.append(orig_stimulus)
                shifted_stimuli.append(shifted_stimulus)
                distances.append(distance)
            # Create structured numpy array of all stimuli in current condition
            pairs = np.array(np.zeros(number_of_pairs),
                             dtype={'names': ('stimulus_a', 'stimulus_b', 'distance'),
                                    'formats': (object, object, float)})
            pairs['stimulus_a'] = orig_stimuli
            pairs['stimulus_b'] = shifted_stimuli
            pairs['distance'] = distances
            # Save stimuli from current condition
            condition_file_name = f'{freq}Hz_interval={int(interval[0])}-{int(interval[1])}ms.npy'
            condition_file_location = os.path.join(current_folder, condition_file_name)
            with open(condition_file_location, 'wb') as file:
                np.save(file, pairs)
    end_time = gen_datestr()
    print(f'Finished creating stimuli with {num_neur} neurons - done: {end_time}\n')

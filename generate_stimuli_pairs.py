'''
This file is used to generate pairs of stimuli where one of the stimuli
is a temporally shifted version of the other.
The difference between the stimuli is defined by how much
the spikes in the original stimulus are allowed to shift, for
this purpose a symmetric interval shift is used, allowing
each spike to move forward or backward in time within a specified interval.
'''
# %%
from generation import make_stimulus
from generation.transform import symmetric_interval_shift
from tools import calc_stimuli_distance
import numpy as np
import pyspike as spk
import os
import pickle
import time
import datetime

# %%
# Set up file saving
target_folder = '/mnt/disks/data'  # Set up where the sets will be saved
if not(os.path.exists(target_folder)):
    oldmask = os.umask(0)
    os.mkdir(target_folder, 0o777)
    os.umask(oldmask)

# Stimuli generation parameters
frequencies = [15, 50, 100]
number_of_neurons = [30, 150, 500]
duration = 0.5  # Units: Seconds

# Pair generation parameters
number_of_pairs = 30  # Number of stimulus pairs for each interval condition
# Intervals to use for temporal shifting
temporal_shift_intervals = [[3e-3, 5e-3], [3e-3, 7e-3],
                            [3e-3, 10e-3], [3e-3, 15e-3]] # Units: Seconds

# %%
for interval in temporal_shift_intervals:
    start_time = datetime.datetime.now().strftime('%d/%m %H:%M:%S')
    print(f'Creating stimuli for interval ±{np.multiply(interval, 1000)}ms - started: {start_time}')
    for num_neur in number_of_neurons:
        print(f'\tNow working on stimuli with {num_neur} neurons')
        for freq in frequencies:
            print(f'\t\tNow working on stimuli of {freq}Hz')
            # Creating placeholders for all stimuli in current condition
            orig_stimuli = []
            shifted_stimuli = []
            distances = []
            for i in range(number_of_pairs):
                # Generate original stimulus
                orig_stimulus = make_stimulus(frequency=freq, duration=duration,
                                              num_neurons=num_neur)
                # Generate shifted stimulus
                shifted_stimulus = symmetric_interval_shift(stimulus=orig_stimulus, duration=duration,
                                                            interval=interval, num_shifted=1)
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
    end_time = datetime.datetime.now().strftime('%d/%m %H:%M:%S')
    print(f'Finished with interval ±{np.multiply(interval, 1000)}ms - done: {end_time}')

from make_test_samples import gen_with_vesicle_release, convert_multi_samples
from itertools import product
import pandas as pd
import numpy as np
import datetime
import pickle
import time
import os
from collections import namedtuple
import warnings

warnings.filterwarnings("ignore")  # Used to ignore brian deprecation warnings
# %% Define the conditions
multi_params = dict(num_neurons=[30, 100],
                    rates=[150],
                    spans=[3, 6, 9])
mode = 1
num_ves = 20
duration = 500

num_sets = 30  # Number of sets to generate from each condition

set_sizes = 100  # Shared between conditions

conditions = product(*multi_params.values())
num_conditions = np.prod([len(items) for items in multi_params.values()])
# %% set up data saving model
# data_folder = 'D:\Data'
data_folder = '/mnt/disks/data'
Condition = namedtuple('Condition', ['num_neur', 'rate', 'span'])

date = datetime.datetime.now()
datestr = f'{date.day:02}_{date.month:02}_{date.year-2000}'

main_folder = f"{data_folder}/{datestr}_vesrel"
if not(os.path.exists(main_folder)):
    oldmask = os.umask(0)
    os.mkdir(main_folder, 0o777)
    os.umask(oldmask)
    param_str = f"""Conditions: {multi_params}
    
    Params:
    mode={mode}
    num_ves={num_ves}
    num_sets={num_sets}
    set_size={set_sizes}"""


    with open(f"{main_folder}/Samples_Info.txt", 'w') as handle:
        handle.write(param_str)
# %% Sample generation
for i, (num_neur, rate, span) in enumerate(conditions):
    start_time = time.time()
    print(
        f'Now generating {num_neur} neurons at {rate}Hz with {span}ms release span  -  Condition #{i+1}/{num_conditions}')
    # multi_samples[(num_neur, rate, span)] = []
    descriptor = Condition(num_neur, rate, span)
    folder_name = '_'.join([f'{descriptor._fields[i]}={descriptor[i]}' for i in range(len(descriptor))])
    current_folder = rf'{main_folder}/{folder_name}'
    if not(os.path.exists(current_folder)):
        oldmask = os.umask(0)
        os.makedirs(current_folder, 0o777)
        os.umask(oldmask)
    # condition_samples = pd.Series(np.zeros(num_sets), name=descriptor, dtype=object)
    for j in range(num_sets):
        print(f'Sample #{j:>2}', end='\r')
        beta_params = dict(span=span, mode=mode)
        data = gen_with_vesicle_release(
            rate=rate,
            num_neur=num_neur,
            duration_ms=duration,
            beta_params=beta_params,
            num_ves=num_ves,
            set1_size=set_sizes,
            set2_size=set_sizes)
        data = {key: data for key, data in zip(['data', 'labels'],convert_multi_samples(data))}
        with open(f"{current_folder}/set{j}.pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # condition_samples[j] = data
    # Save data
    print('  Saving...', end='')
    run_time = time.time() - start_time
    time_str = timestr = 'took {:0>2.0f}:{:.2f} (min:sec)'.format(*divmod(run_time, 60))
    print(f'Done!  | {time_str}')



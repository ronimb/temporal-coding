from make_test_samples import make_test_samples
from itertools import product
import pandas as pd
import numpy as np
import datetime
import pickle
# %% Define the conditions
shift_sizes = [5]
rates = [50]
num_neurons = [30, 500]

num_sets = 30  # Number of sets to generate from each condition

set_sizes = 100 # Shared between conditions

conditions = product(num_neurons, rates, shift_sizes)
n_conds = len(shift_sizes) * len(rates) * len(num_neurons)
# %% Sample generation
multi_samples = dict()
for i, (num_neur, rate, shift) in enumerate(conditions):
    print(f'Now generating {num_neur} neurons at {rate}Hz with {shift}ms temporal shift  -  Condition #{i+1}/{n_conds}')
    multi_samples[(num_neur, rate, shift)] = []
    for _ in range(num_sets):
        multi_samples[(num_neur, rate, shift)].append(make_test_samples(
        rate=rate,
        duration_sec=1,
        num_neur=num_neur ,
        shift_size=shift,
        set1_size=set_sizes,
        set2_size=set_sizes))
# %% Sample verification
# verfdict = {
#     key: [multi_samples[key]['data'][i][j].shape for i,j
#           in product(
#               range(set_sizes*2),
#               range(key[0]))]
#           for key in multi_samples.keys()
#           }
# obs_mean = pd.Series({key: np.mean(verfdict[key]) for key in verfdict.keys()})
# %% Save data
date = datetime.datetime.now()
datestr = f'{date.day:02}{date.month:02}{date.year-2000}'
with open(rf"D:\Data\Samples_{datestr}.pickle", 'wb') as handle:
    pickle.dump(multi_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

param_str = f'''num_neurons={num_neurons}
rates={rates}
shift_sizes={shift_sizes}

num_sets={num_sets}'''

with open(rf"D:\Data\Samples_{datestr}.txt", 'w') as handle:
    handle.write(param_str)
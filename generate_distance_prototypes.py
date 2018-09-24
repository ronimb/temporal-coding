import numpy as np
from numba import jit, prange
from multiprocessing import Pool
import pickle
from time import time, strftime, gmtime, ctime
import os
import datetime
import brian2 as b2
b2.BrianLogger.suppress_name('method_choice')
import pyspike as spk
import matplotlib.pyplot as plt
from make_test_samples import make_spk, find_match_dists, multi_shift, calc_distance
# %%
def find_for_neuron(neuron, distance, shift_params, eps=1e-3):
    shifted = multi_shift(neuron, **shift_params)
    distances_from_neuron = calc_distance(shifted, neuron)['vector']
    matching_ind = find_match_dists(distances_from_neuron, distance, eps=eps)
    if matching_ind:
        res = (shifted[matching_ind], distances_from_neuron[matching_ind])
    else:
        res = None, None
    return res
# %% Manually determined conditions for distance generation
num_shifts = 50
conds = {(15, 0.05): {'shifts': np.arange(15), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (15, 0.1): {'shifts': np.arange(5,35,5), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (15, 0.2): {'shifts': np.arange(20, 120, 10), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (15, 0.3): {'shifts': [150, 200, 300]},
         (50, 0.05): {'shifts': np.arange(10), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (50, 0.1): {'shifts': np.arange(10), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (50, 0.2): {'shifts': np.arange(5, 35, 5), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (50, 0.3): {'shifts': [150, 200, 300]},
         (100, 0.05): {'shifts': np.arange(5), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (100, 0.1): {'shifts': np.arange(3, 11), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (100, 0.2): {'shifts': np.arange(5, 15), 'frac_shifts': np.linspace(0.1, 1, 10)},
         (100, 0.3): {'shifts': [150, 200, 300]}
        }
for cond in conds.values():
    cond.update({'n': num_shifts})
# %%
distances = [0.05, 0.1, 0.2, 0.3]
frequencies = [15, 50 ,100]
num_neurons = [30, 150, 500]
num_samples = 30
duration_ms = 500
data_folder = '/mnt/disks/data'
main_folder = f"{data_folder}/{datetime.datetime.now().strftime('%d_%m')}_samples/"
if not(os.path.exists(main_folder)):
    oldmask = os.umask(0)
    os.mkdir(main_folder, 0o777)
    os.umask(oldmask)
# %%
total_neurons = np.sum(num_neurons) * num_samples

for freq in frequencies:
    freq_start = time()
    print(f'Working on {freq}Hz, started on {ctime()}')
    freq_folder = f'{main_folder}/{freq}hz/'
    if not (os.path.exists(freq_folder)):
        oldmask = os.umask(0)
        os.mkdir(freq_folder, 0o777)
        os.umask(oldmask)
    for dist in distances:
        dist_start = time()
        print(f'\t distance={dist}, started on {ctime()}')
        # Set conditions for each distance frequency combination
        # For each neuron in the set, generate a small sample of shifted versions and pick the closest one
        neurons = make_spk(freq, duration_ms, total_neurons, exact_freq=False)
        print('\t\t  ', end="", flush=True)
        neur_times = []
        pairs = np.zeros(total_neurons, dtype={'names': ('a', 'b', 'distance'),
                                               'formats': (object, object, float)})
        params = conds[(freq, dist)]
        found = 0
        i = 0
        while found < total_neurons:
            i += 1
            if not(neurons):
                neurons = make_spk(freq, duration_ms, total_neurons, exact_freq=False)
            neuron = neurons.pop()
            print(f'\b\b\b\b\b{freq}Hz, distance={dist} : Scanning neuron #{i:000002}')
            neuron_start = time()
            matching = None
            matching, real_dist = find_for_neuron(neuron, dist, params)
            if matching:
                pairs[found]['a'] = neuron
                pairs[found]['b'] = matching
                pairs[found]['distance'] = real_dist
                found += 1
                print(f'Found {found}/{total_neurons} Neurons so far ({found/total_neurons:.2%})')
            neur_times.append(time() - neuron_start)

        # neur_times = np.zeros(total_neurons)
        # pair_num = 0
        # i = 0
        # while neurons:
        #     i+=1
        #     neuron = neurons.pop()
        #     print(f'\b\b\b\b\bNeuron #{i:00002}', end="", flush=True)
        #     neuron_start = time()
        #     params = conds[(freq, dist)]
        #     matching = None
        #     counter = 0
        #     while not(matching):
        #         if counter >= 25:
        #             neuron = make_spk(freq, duration_ms, 10, uniform_freq=False)
        #             counter = 0
        #             print('Counter exceeded')
        #         matching, real_dist = find_for_neuron(neuron, dist, params)
        #         counter += 1
        #     pairs[pair_num]['a'] = neuron
        #     pairs[pair_num]['b'] = matching
        #     pairs[pair_num]['distance'] = real_dist
        #     pair_num += 1
        #     neur_times[i-1] = time() - neuron_start
        print(f'Begin dividing into samples, started on {ctime()}')
        division_time = time()
        loc = 0
        for num_neur in num_neurons:
            all_samples = np.zeros(num_samples, dtype={'names': ('a', 'b', 'distance'),
                                               'formats': (object, object, float)})
            print(f'Making samples of {num_neur} neurons')
            for samp in range(num_samples):
                all_samples[samp]['a'] = pairs[loc:loc+num_neur]['a']
                all_samples[samp]['b'] = pairs[loc:loc+num_neur]['b']
                all_samples[samp]['distance'] = pairs[loc:loc+num_neur]['distance'].mean()
                loc += num_neur
            print('Saving')

            with open(f'{freq_folder}{num_neur}Neurons_distance={dist}_{num_samples}Samples.npy', 'wb') \
                    as file:
                np.save(file, all_samples)

        print(f"\t {dist} distance took {strftime('%M:%S', (gmtime(time()-dist_start)))}")
        print(f"Mean time per neuron is {strftime('%M:%S', gmtime(np.mean(neur_times)))}")
    print(f"{freq}hz took {strftime('%M:%S', gmtime(time()-freq_start))}")


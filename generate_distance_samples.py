import numpy as np
import pandas as pd
import pickle
import time
import datetime
import brian2 as b2
b2.BrianLogger.suppress_name('method_choice')
import pyspike as spk

import matplotlib.pyplot as plt
# %%
def plot_sample(sample):
    for i, train in enumerate(sample):
        n_spikes = train.shape[0]
        ax = plt.scatter(train, n_spikes * [i], marker='|', c='blue')
        plt.yticks([])
    return ax

def gen_single_set(n, r, duration=1 * b2.units.second):
    neuron = b2.NeuronGroup(n, "rate : Hz", threshold='rand()<rate*dt')
    neuron.rate = r * b2.units.Hz
    spikes = b2.SpikeMonitor(neuron, record=True)
    b2.run(duration)
    trains = [train / (1 * b2.units.ms) for train in spikes.spike_trains().values()]
    return np.array(trains)

def determine_sample_distance(sample_a, sample_b, duration=1000):
    distances = []
    for tr_a, tr_b in zip(sample_a, sample_b):
        tr_a = spk.SpikeTrain(tr_a, (0, duration))
        tr_b = spk.SpikeTrain(tr_b, (0, duration))
        distances.append(spk.spike_distance(tr_a, tr_b))
    return np.mean(distances)


def spikes_to_ves(sample, span, mode=1, num_ves=20):
    ves_array = []
    for train in sample:
        num_spikes = train.shape[0]
        ves_offsets = release_fun(span, mode, (num_ves, num_spikes))
        ves_times = (train + ves_offsets).flatten()
        ves_times.sort()
        ves_array.append(ves_times)
    return np.array(ves_array)
# %%




import numpy as np
from numba import jit, prange
from multiprocessing import Pool
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
# %%
distances = [0.05, 0.1, 0.2, 0.3]
frequencies = [15, 50 ,100]
num_neurons = [30, 150, 500]
num_prototypes = 30
# %%

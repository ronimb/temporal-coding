import numpy as np
import pickle
import time
import brian2 as b2
from brian2.units import ms
import sys
import matplotlib.pyplot as plt

from make_test_samples import gen_with_vesicle_release
# %%
def sec_to_str(sec):
    m, s = divmod(sec, 60)
    return f'{m:0>2.0f}:{s:0>2.2f}'

def convert_sample(sample):
    inds = []
    times = []
    counts = []
    for i, neuron in enumerate(sample):
        neuron = np.trunc(neuron * 10) / 10
        time, count = np.unique(neuron, return_counts=True)
        inds.extend([i] * time.shape[0])
        times.extend(time)
        counts.extend(count)
    return np.array([inds, times, counts])

def convert_all_samples(samples):
    labels = samples['labels']
    num_neurons = samples['data'][0].shape[0]
    num_samples = samples['data'].shape[0]

    inds = []
    times = []
    counts = []
    for i, sample in enumerate(samples['data']):
        current = convert_sample(sample)
        inds.extend(current[0] + (num_neurons * i))
        times.extend(current[1])
        counts.extend(current[2])
    inds = np.array(inds)
    times = np.array(times)
    counts = np.array(counts)
    return (inds, times, counts), labels
# %%
# loc = r'C:\Users\ron\OneDrive\Documents\Masters\Parnas\temporal-coding\Data\n30_r15_vesrel_testset.pickle'
# loc = sys.argv[1]
# with open(loc, 'rb') as file:
    # samples = pickle.load(file)
t = time.time()
samples = gen_with_vesicle_release(rate=100,
                                       num_neur=500,
                                       span=5,
                                       mode=1,
                                       num_ves=20)
print(f'Sample generation took {sec_to_str(time.time()-t)}')
num_neurons = samples['data'][0].shape[0]
num_samples = samples['data'].shape[0]

t = time.time()
(inds, times, _counts), labels = convert_all_samples(samples)

print(f'Conversion took {sec_to_str(time.time()-t)}')
# %%
t = time.time()
print('Setting up network')
duration = 500
tau = 2*ms
count = np.zeros((duration*10, num_samples*num_neurons), int)
count[(times * 10).astype(int), inds.astype(int)] = _counts
eqs = 'dv/dt = -v / tau: 1'
target = b2.NeuronGroup(num_samples, eqs, threshold='v>0.005', reset='v=0')
driving = b2.SpikeGeneratorGroup(num_samples*num_neurons, inds, times*ms)
counts = b2.TimedArray(count, dt=b2.defaultclock.dt)
synapses = b2.Synapses(driving, target, 'w: 1', on_pre='v+=w*counts(t, i)')
i = np.arange(num_samples*num_neurons)
j = np.array([[i]*num_neurons for i in range(num_samples)]).flatten()
synapses.connect(j=j, i=i)
synapses.w = np.random.normal(0, 1e-3, num_neurons*num_samples)

s = b2.SpikeMonitor(target, record=True)
print(f'Finished network setup, took {sec_to_str(time.time()-t)}')
# %%
print('Eval #1')
t = time.time()
b2.run(duration * ms)
t = time.time() - t
print(f'Finished first evaluation, took {sec_to_str(t)}')

print('Eval #2')
t = time.time()
b2.run(duration * ms)
t = time.time() - t
print(f'Finished second evaluation, took {sec_to_str(t)}')


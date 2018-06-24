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
    converted_samples = np.zeros(len(inds),
                                 dtype={'names': ('inds', 'times', 'counts'),
                                        'formats': (int, float, int)})
    converted_samples['inds'] = inds
    converted_samples['times'] = times
    converted_samples['counts'] = counts
    return converted_samples, labels
# %%
# loc = r'C:\Users\ron\OneDrive\Documents\Masters\Parnas\temporal-coding\Data\n30_r15_vesrel_testset.pickle'
# loc = sys.argv[1]
# with open(loc, 'rb') as file:
    # samples = pickle.load(file)
t = time.time()
orig_samples = gen_with_vesicle_release(rate=100,
                                       num_neur=500,
                                       span=5,
                                       mode=1,
                                       num_ves=20)
print(f'Sample generation took {sec_to_str(time.time()-t)}')
num_neurons = orig_samples['data'][0].shape[0]
num_samples = orig_samples['data'].shape[0]

t = time.time()
samples, labels = convert_all_samples(orig_samples)
print(f'Conversion took {sec_to_str(time.time()-t)}')
# %%
t = time.time()
print('Setting up network')
duration = 500
tau = 2*ms
count = np.zeros((duration*10, num_samples*num_neurons), int)
count[(samples['times'] * 10).astype(int), samples['inds'].astype(int)] = samples['counts']
eqs = 'dv/dt = -v / tau : 1'
target = b2.NeuronGroup(num_samples, eqs, threshold='v>0.005', reset='v=0')
driving = b2.SpikeGeneratorGroup(num_samples*num_neurons, samples['inds'], samples['times']*ms)
counts = b2.TimedArray(count, dt=b2.defaultclock.dt)
synapses = b2.Synapses(driving, target, 'w: 1', on_pre='v+=w*counts(t, i)')
i = np.arange(num_samples*num_neurons)
j = np.array([[i]*num_neurons for i in range(num_samples)]).flatten()
synapses.connect(j=j, i=i)
synapses.w = np.random.normal(0, 1e-3, num_neurons*num_samples)

s = b2.SpikeMonitor(target, record=True)
print(f'Finished network setup, took {sec_to_str(time.time()-t)}')
# %%
def return_subset(batch_size, samples, labels):
    selected_inds = np.sort(np.random.choice(range(num_samples), batch_size, replace=False))
    batch_inds = (selected_inds * num_neurons + np.arange(num_neurons)[:, np.newaxis]).flatten()
    ind_locs = np.in1d(samples['inds'], batch_inds)
    subset = np.zeros(ind_locs.sum(), dtype={'names': ('inds', 'times', 'counts'),
                                        'formats': (int, float, int)})
    subset['inds'] = samples['inds'][ind_locs]

    samp_map = {v: i for i, v in enumerate(selected_inds)}
    subset['inds'], neur = np.divmod(subset['inds'], num_neurons)
    u, inv = np.unique(subset['inds'], return_inverse=True)
    subset['inds'] = np.array([samp_map[x] for x in u])[inv] * num_neurons + neur
    subset['times'] = samples['times'][ind_locs]
    subset['counts'] = samples['counts'][ind_locs]

    return subset, labels[selected_inds]
# %%
b2.start_scope()
batch_size = 50
num_reps = 100
duration = 500

subset, labels_sub = return_subset(batch_size, samples, labels)
selected_inds = np.random.choice(range(num_samples), batch_size, replace=False)

# First classify
weights = np.random.normal(0, 1e-3, batch_size*num_neurons)
tr_1_count = np.zeros((duration * 10, batch_size * num_neurons), int)
tr_1_count[(subset['times'] * 10).astype(int), subset['inds']] =  subset['counts']
tr_1_target = b2.NeuronGroup(batch_size, eqs, threshold='v>0.005', reset='v=0')
tr_1_driving = b2.SpikeGeneratorGroup(batch_size*num_neurons, subset['inds'], subset['times']*ms)
tr_1_counts = b2.TimedArray(tr_1_count, dt=b2.defaultclock.dt)
tr_1_synapses = b2.Synapses(tr_1_driving, tr_1_target, 'w: 1', on_pre='v+=w*tr_1_counts(t, i)')
tr_1_i = np.arange(batch_size*num_neurons)
tr_1_j = np.array([[i]*num_neurons for i in range(batch_size)]).flatten()
tr_1_synapses.connect(j=tr_1_j, i=tr_1_i)
tr_1_synapses.w = weights

tr_1_s = b2.SpikeMonitor(tr_1_target, record=True)
tr_1_m = b2.StateMonitor(tr_1_target, 'v', record=True)
b2.run(duration * ms)

decisions = [tr_1_s.spike_trains()[i].shape[0] != 0 for i in range(batch_size)]
v_max_times = np.argmax(tr_1_m.v, 1)

v_max_t = v_max_times.max()
# Train
b2.start_scope()
tr_2_target = b2.NeuronGroup(batch_size, eqs)
tr_2_driving = b2.SpikeGeneratorGroup(batch_size*num_neurons, subset['inds'], subset['times']*ms)
tr_2_counts = b2.TimedArray(tr_1_count, dt=b2.defaultclock.dt)
tr_2_synapses = b2.Synapses(tr_2_driving, tr_2_target, 'w: 1', on_pre='v+=1*tr_2_counts(t, i)')
tr_2_synapses.connect(j=tr_1_j, i=tr_1_i)
tr_2_synapses.w = weights

tr_2_m = b2.StateMonitor(tr_2_target, 'v', record=True)
b2.run(v_max_t * ms)
# %%
x = []
print('Eval #1')
t = time.time()
b2.run(duration * ms)
x.append(s.spike_trains())
t = time.time() - t
print(f'Finished first evaluation, took {sec_to_str(t)}')

print('Eval #2')
t = time.time()
b2.run(duration * ms)
x.append(s.spike_trains())
t = time.time() - t
print(f'Finished second evaluation, took {sec_to_str(t)}')


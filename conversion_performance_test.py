# %%
from numba import prange, jit, njit
import numpy as np
from multiprocessing import Pool
from make_test_samples import gen_with_vesicle_release
import time

# %%
def sec_to_str(sec):
    m, s = divmod(sec, 60)
    return f'{m:0>2.0f}:{s:0>2.2f}'


# %%
@jit(parallel=True)
def tst(s):
    inds = []
    times = []
    counts = []
    num_events = []
    for i in prange(s.shape[0]):
        neuron = np.trunc(s[i] *10) / 10
        time, count = np.unique(neuron, return_counts=True)
        num_events.append(time.shape[0])
        inds.extend([i] * time.shape[0])
        times.extend(time)
        counts.extend(count)
    return np.array([inds, times, counts]), num_events

# %%
def cn(n):
    neuron = np.trunc(n * 10) / 10
    time, count = np.unique(neuron, return_counts=True)
    inds = [0] * time.shape[0]
    return (inds, time, count)

# %%
def ca(a):
    p = Pool(12)
    labels = a['labels']
    res = p.map(tst, a['data'])
    ts = np.hstack([[x[0][0] + num_neurons * i, x[0][1], x[0][2]] for i, x in enumerate(res)])
    num_evs = np.hstack([x[1] for x in res])
    p.close()
    p.join()
    converted_samples = np.zeros(ts.shape[1],
                                 dtype={'names': ('inds', 'times', 'counts'),
                                        'formats': (int, float, int)})
    converted_samples['inds'] = ts[0]
    converted_samples['times'] = ts[1]
    converted_samples['counts'] = ts[2]
    return converted_samples, labels

def ca2(a):
    p = Pool(12)
    labels = a['labels']
    res = p.map(tst, a['data'])
    ts = np.hstack([[x[0] + num_neurons * i, x[1], x[2]] for i, x in enumerate(res)])
    p.close()
    p.join()
    return ts, labels
# %%
# s is a a single sample
def tst2(s):
    tw = np.array([np.hstack([[i] * x.shape[0] for i, x in enumerate(s)]),
               np.trunc(np.hstack(s) * 10) / 10])
    (inds, times), counts = np.unique(tw, axis=1, return_counts=True)
    return np.array([inds, times, counts])

# Testing for subset selection
def subset(num_samples, samples):
    inds = np.random.choice(range(samples['data'].shape[0]), size=num_samples, replace=False).astype(int)
    sub_dict = {'data': samples['data'][inds], 'labels': samples['labels'][inds]}
    return ca(sub_dict)

# %%
def return_subset_orig(batch_size, samples, labels):
    selected_inds = np.sort(np.random.choice(range(num_samples), batch_size, replace=False))
    batch_inds = (selected_inds * num_neurons + np.arange(num_neurons)[:, np.newaxis])
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

def return_subset_numev(batch_size, samples, labels, numev):
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
t = time.time()
batch_size = 50
beta_params = dict(
    span=5,
    mode=1
)
orig_samples = gen_with_vesicle_release(rate=100,
                                        num_neur=500,
                                        beta_params,
                                        num_ves=20)
print(f'Sample generation took {sec_to_str(time.time()-t)}')

num_neurons = orig_samples['data'][0].shape[0]
num_samples = orig_samples['data'].shape[0]
t = time.time()
samples_, _= ca(orig_samples)
print(f'Conversion took {sec_to_str(time.time()-t)}')
# %%
from cython_test.return_subset import return_subset
import pickle
import time
with open('/mnt/disks/data/flat.pickle', 'rb') as f:
    samples = pickle.load(f)
with open('/mnt/disks/data/flatlabs.pickle', 'rb') as f:
    labels = pickle.load(f)
print('Done loading')
def sec_to_str(sec):
    m, s = divmod(sec, 60)
    return f'{m:0>2.0f}:{s:0>2.2f}'
t = time.time()
subset, sub_labels = return_subset(50, samples, labels.astype(int), 200, 500)
print(sec_to_str(time.time()-t))



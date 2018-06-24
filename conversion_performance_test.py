# %%
from numba import prange, jit, njit
import numpy as np
from multiprocessing import Pool
@jit(parallel=True)
def tst(s):
    inds = []
    times = []
    counts = []
    for i in prange(s.shape[0]):
        neuron = np.trunc(s[i] *10) / 10
        time, count = np.unique(neuron, return_counts=True)
        inds.extend([i] * time.shape[0])
        times.extend(time)
        counts.extend(count)
    return np.array([inds, times, counts])

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
    ts = np.hstack([[x[0] + 500 * i, x[1], x[2]] for i, x in enumerate(res)])
    p.close()
    p.join()
    converted_samples = np.zeros(ts.shape[1],
                                 dtype={'names': ('inds', 'times', 'counts'),
                                        'formats': (int, float, int)})
    converted_samples['inds'] = ts[0]
    converted_samples['times'] = ts[1]
    converted_samples['counts'] = ts[2]
    return converted_samples, labels

# %%
# s is a a single sample
def tst2(s):
    tw = np.array([np.hstack([[i] * x.shape[0] for i, x in enumerate(s)]),
               np.trunc(np.hstack(s) * 10) / 10])
    (inds, times), counts = np.unique(tw, axis=1, return_counts=True)
    return np.array([inds, times, counts])


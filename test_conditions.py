import pandas as pd
from itertools import product
import pickle
import numpy as np
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import re
from Tempotron_Brian import Tempotron
# %%
learning_rates = np.array([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
thresholds = np.sort(np.multiply([[2.5], [5.0], [7.5]], [[1e-4, 1e-3, 1e-2, 1e-1]]).flatten())
num_vars = learning_rates.shape[0] * thresholds.shape[0]
reps = 5
pre = np.zeros([learning_rates.shape[0], thresholds.shape[0], reps])
post = pre.copy()

folder = '/mnt/disks/data/18_07_samples/vesrel'
conditions = dict(num_neur=[30, 150, 500],
                  freq=[15, 50, 100],
                  span=6,
                  distance=0.3)
# %%
def acc_for_cond(conds):
    n = conds['num_neur']
    f = conds['freq']
    print(f"starting Num_Neur={n} | Freq = {f}")
    cond_str = f"num_neur={n}_rate={f}_distance={conds['distance']}_span={conds['span']}"
    cond_folder = os.path.join(folder, cond_str)
    files = os.listdir(cond_folder)[:reps]
    for i, file in enumerate(files):
        with open(os.path.join(cond_folder, file), 'rb') as file:
            samples = pickle.load(file)
            labels = samples[1]
            samples = samples[0]
        for j, thresh in enumerate(thresholds):
            T = Tempotron(conds['num_neur'], tau=2, threshold=thresh)
            T.make_classification_network(num_samples=200, name='test')
            for k, lr in enumerate(learning_rates):
                var_num = thresholds.shape[0] * j + k
                print(f"\tNum_Neur={n} | Freq = {f} --> Sample #{i+1}/{len(files)}  |  Condition #{var_num + 1} / {num_vars}")
                pre[k, j, i] = T.accuracy('test', samples, labels).mean()
                T.train(samples, labels, num_reps=30, learning_rate=lr)
                post[k, j, i] =  T.accuracy('test', samples, labels).mean()
        print(f"\tNum_Neur={n} | Freq = {f} --> Sample #{i+1}/{len(files)} ----> DONE")
    diff = post - pre
    return {'pre': pre,
            'post': post,
            'diff': diff}
# %%
cond_combs = np.array(list(product(conditions['num_neur'], conditions['freq'])))
cond_list = [{'num_neur': n, 'freq': f,
              'span': conditions['span'],
              'distance': conditions['distance']}
             for n,f in cond_combs]
res = np.zeros(len(cond_list), dtype={'names': ('num_neur', 'freq', 'counts'),
                                             'formats': (int, int, object)})
res['num_neur'] = cond_combs[:, 0]
res['freq'] = cond_combs[:, 1]
with mp.Pool(8) as P:
    res['counts'] = P.map(acc_for_cond, cond_list)
# %%
np.save('/mnt/disks/data/accuracy.npy', res)


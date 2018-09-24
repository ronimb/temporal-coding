import pandas as pd
from itertools import product
import pickle
import numpy as np
import os
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import re
import datetime as dt
from Tempotron_Brian import Tempotron
# %%
learning_rates = np.array([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
thresholds = np.sort(np.multiply([[2.5], [5.0], [7.5]], [[1e-4, 1e-3, 1e-2, 1e-1]]).flatten())
num_vars = learning_rates.shape[0] * thresholds.shape[0]

num_samples = 5
train_reps = 30

folder = '/mnt/disks/data/18_07_samples/vesrel'
sample_conditions = dict(num_neur=[30, 150, 500],
                         freq=[15, 50, 100],
                         span=6,
                         distance=0.3)
res_file = 'preliminary_results.csv'
headers = ['num_neur', 'freq', 'samp_num', 'distance', 'span', 'threshold', 'learning_rate', 'pre', 'post', 'diff']
if not os.path.isfile(res_file):
    with open(res_file, 'a') as f:
        f.write(str(headers).replace("'", '').strip('[]').replace(' ','') + '\n')
time_fmt = lambda t: time.strftime('%M:%S', time.gmtime(time.time() - t))
# %%
def acc_for_samp(samp_conds):
    cond_start = time.time()
    pre = np.zeros([learning_rates.shape[0], thresholds.shape[0], num_samples])
    post = pre.copy()
    n = samp_conds['num_neur']
    f = samp_conds['freq']
    print(f"starting Num_Neur={n} | Freq = {f}hz")
    cond_str = f"num_neur={n}_rate={f}_distance={samp_conds['distance']}_span={samp_conds['span']}"
    cond_folder = os.path.join(folder, cond_str)
    files = os.listdir(cond_folder)[:num_samples]
    for i, file in enumerate(files):
        file_start = time.time()
        with open(os.path.join(cond_folder, file), 'rb') as file:
            samples = pickle.load(file)
            labels = samples[1]
            samples = samples[0]
        for j, thresh in enumerate(thresholds):
            T = Tempotron(samp_conds['num_neur'], tau=2, threshold=thresh)
            for k, lr in enumerate(learning_rates):
                train_start = time.time()
                T.reset()
                T.make_classification_network(num_samples=200, name='test')
                var_num = thresholds.shape[0] * j + k
                print(f"\t\tNum_Neur={n} | Freq = {f} --> Sample #{i+1}/{len(files)}  |  Condition #{var_num + 1} / {num_vars}")
                pre[k, j, i] = T.accuracy('test', samples, labels).mean()
                T.train(samples, labels, num_reps=train_reps, learning_rate=lr)
                post[k, j, i] =  T.accuracy('test', samples, labels).mean()
                print(
                    f"\t\tNum_Neur={n} | Freq = {f} --> Sample #{i+1}/{len(files)}  |  Condition #{var_num + 1} / {num_vars} ----> Done! took {time_fmt(train_start)}")
    diff = post - pre
    return {'pre': pre,
            'post': post,
            'diff': diff}

def acc_for_file(file_conds):
    t = time.time()
    print(
        f"Started working on a sample with {file_conds['num_neur']} Neurons firing at {file_conds['freq']}hz at a distance of {file_conds['distance']} with a span of {file_conds['span']}, training with a threshold of {file_conds['threshold']} and a learning rate of {file_conds['learning_rate']}")
    file_str = f"num_neur={file_conds['num_neur']}_rate={file_conds['freq']}_distance={file_conds['distance']}_span={file_conds['span']}/set{file_conds['samp_num']}.pickle"
    file_loc = os.path.join(folder, file_str)
    with open(file_loc, 'rb') as file:
        samples = pickle.load(file)
        labels = samples[1]
        samples = samples[0]
    T = Tempotron(file_conds['num_neur'], tau=2, threshold=file_conds['threshold'])
    T.make_classification_network(num_samples=200, name='test')
    pre = T.accuracy('test', samples, labels).mean()
    T.train(samples, labels, num_reps=train_reps, learning_rate=file_conds['learning_rate'])
    post = T.accuracy('test', samples, labels).mean()
    diff = post - pre
    return pd.Series([pre, post, diff], ['pre', 'post', 'diff'],
                     name=(file_conds['num_neur'], file_conds['freq'], file_conds['distance'], file_conds['span']))
    print(
        f"Finished working on a sample with {file_conds['num_neur']} Neurons firing at {file_conds['freq']}hz at a distance of {file_conds['distance']} with a span of {file_conds['span']}, took {time_fmt(t)}")

def acc_for_file_writer(file_conds):
    t = time.time()
    curr_time = dt.datetime.now(dt.timezone(dt.timedelta(hours=3)))
    print(
        f"{curr_time.strftime('%d/%m/%y - %H:%M')} -- Started working on a sample with {file_conds['num_neur']} Neurons firing at {file_conds['freq']}hz at a distance of {file_conds['distance']} with a span of {file_conds['span']}, training with a threshold of {file_conds['threshold']} and a learning rate of {file_conds['learning_rate']}")
    file_str = f"num_neur={file_conds['num_neur']}_rate={file_conds['freq']}_distance={file_conds['distance']}_span={file_conds['span']}/set{file_conds['samp_num']}.pickle"
    file_loc = os.path.join(folder, file_str)
    with open(file_loc, 'rb') as file:
        samples = pickle.load(file)
        labels = samples[1]
        samples = samples[0]
    T = Tempotron(file_conds['num_neur'], tau=2, threshold=file_conds['threshold'])
    T.make_classification_network(num_samples=200, name='test')
    pre = T.accuracy('test', samples, labels).mean()
    T.train(samples, labels, num_reps=train_reps, learning_rate=file_conds['learning_rate'])
    post = T.accuracy('test', samples, labels).mean()
    diff = post - pre
    line_data = np.array([
        file_conds['num_neur'],
        file_conds['freq'],
        file_conds['samp_num'],
        file_conds['distance'],
        file_conds['span'],
        file_conds['threshold'],
        file_conds['learning_rate'],
        pre,
        post,
        diff
    ])
    line_str = np.array2string(line_data, separator=',', precision=10, floatmode='maxprec').\
        strip('[]').replace('\n','').replace(' ', '')
    with open(res_file, 'a') as file:
        file.write(line_str)
        file.write('\n')

# %%
def check_existing(file_loc, cond_list):
    cond_df = pd.DataFrame(cond_list)
    existing = pd.read_csv(file_loc).iloc[:, 0:7][cond_df.columns]
    existing_inds = []
    for _, row in existing.iterrows():
        matching = (cond_df.values == row.values).all(1)
        if matching.any():
            ind = np.where(matching)[0][0]
            existing_inds.append(ind)
    print(f'Found {len(existing_inds)} / {cond_df.shape[0]} in condition list, removing')
    new_cond_list = [cond_list[i] for i in range(len(cond_list)) if i not in existing_inds]
    return new_cond_list

samp_cond_combs = np.array(list(product(sample_conditions['num_neur'], sample_conditions['freq'])))
all_cond_combs = (list(product(learning_rates, thresholds, range(num_samples), samp_cond_combs )))
file_cond_list = [{'num_neur': n, 'freq': f, 'threshold': thresh, 'learning_rate': lr, 'samp_num': samp_num,
              'span': sample_conditions['span'],
              'distance': sample_conditions['distance']}
                  for lr, thresh, samp_num, (n,f) in all_cond_combs]
print(len(file_cond_list))
file_cond_list = check_existing(res_file, file_cond_list)
print(len(file_cond_list))
# samp_cond_list = [{'num_neur': n, 'freq': f,
#               'span': sample_conditions['span'],
#               'distance': sample_conditions['distance']}
#                   for n,f in samp_cond_combs]
# %%
# res = np.zeros(len(samp_cond_list), dtype={'names': ('num_neur', 'freq', 'accuracy'),
#                                              'formats': (int, int, object)})
# res['num_neur'] = samp_cond_combs[:, 0]
# res['freq'] = samp_cond_combs[:, 1]
with mp.Pool(14) as P:
    res = P.map(acc_for_file_writer, file_cond_list)

# with open(os.path.join('/mnt/disks/data/','acc_test.pickle'), 'wb') as file:
    # pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)

import numpy as np
import pandas as pd
import pickle
import time
import datetime

import warnings
warnings.filterwarnings("ignore")  # Used to ignore brian deprecation warnings

from Tempotron import Tempotron

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# %%
samples_path = r"D:\Data\Samples_270518.pickle"
with open(samples_path, 'rb') as file:
    multi_samples = pickle.load(file)
# %%
tau = 2  # Decay constant for exponentoial decay, Milliseconds
threshold = 0.005  # Threshold to cross for firing
learning_rate = 1e-4  # Parameter controlling size of weight change for wrong trials

train_reps = [2,5,10,20]

length_to_use = 500 # MS
num_neur = (30, 500)
rate, shift = 50, 5
num_sets = 10
num_conds = 2 * len(train_reps)
# %%
df_ind = pd.MultiIndex.from_product((num_neur, train_reps))
acc_before = pd.DataFrame(np.zeros((num_conds, num_sets)), index=df_ind)
acc_after = acc_before.copy()

rep_times = acc_before.copy()
run_times = acc_before.copy()
# %%
for num_neur, _, _ in multi_samples:
    print(f'{num_neur} neurons in set')
    for reps in train_reps:
        print(f'\t {reps} training repetitions')
        for i in range(num_sets):
            print(f'\tSample #{i:>2}', end='')
            start_time = time.time()
            samples = multi_samples[num_neur, rate, shift][i]['data']
            labels = multi_samples[num_neur, rate, shift][i]['labels']
            T = Tempotron(num_neurons=num_neur, tau=tau, threshold=threshold)  # Set up model for current repetition
            acc_before.loc[num_neur, reps][i] = T.accuracy(samples, labels)  # Evaluate accuracy - before
            for _ in range(reps):
                T.train(samples, labels, learning_rate=learning_rate)
            acc_after.loc[num_neur, reps][i] = T.accuracy(samples, labels)
            curr_run_time = time.time() - start_time
            run_times.loc[num_neur, reps][i] = curr_run_time
            timestr = '  | took {:0>2.0f}:{:.2f} (min:sec)'.format(*divmod(curr_run_time, 60))
            print(timestr)
acc_diff = acc_after - acc_before
# %%
date = datetime.datetime.now()
datestr = f'{date.day:02}{date.month:02}{date.year-2000}'

acc_before.to_csv(f'Data/iternum_acc_before_{datestr}.csv')
acc_after.to_csv(f'Data/iternum_acc_after_{datestr}.csv')
acc_diff.to_csv(f'Data/iternum_acc_diff_{datestr}.csv')
run_times.to_csv(f'Data/iternum_runtimes_{datestr}.csv')
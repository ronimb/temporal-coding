from Old.Tempotron import Tempotron
import pandas as pd
import pickle
import numpy as np
import time
import datetime
# %% loading sample data
samples_path = r"D:\Data\Samples_230518.pickle"
with open(samples_path, 'rb') as file:
    multi_samples = pickle.load(file)
# Dictionary key order is (num_neur, rate, shift)
# %% Global tempotron params
tau = 2  # Decay constant for exponentoial decay, Milliseconds
threshold = 0.005  # Threshold to cross for firing
learning_rate = 1e-4  # Parameter controlling size of weight change for wrong trials



train_reps = 5  # Number of iterations over the entire set during training
eval_reps = 30  # Number of full repetitions in each condition (Accuracy-Before -> Training -> Accuracy after)

# %% Preparing dataframes
df_ind = pd.MultiIndex.from_tuples(multi_samples.keys(), names=('num_neurons', 'rate', 'shift'))
num_conds = df_ind.shape[0]
acc_before = pd.DataFrame(np.zeros((num_conds, eval_reps)), index=df_ind)
acc_after = acc_before.copy()

rep_times = acc_before.copy()
# %% Main code

# Loop over conditions
for i, (num_neur, rate, shift) in enumerate(df_ind):
    samples = multi_samples[num_neur, rate, shift]['data']
    labels = multi_samples[num_neur, rate, shift]['labels']
    print(f"Working on {num_neur} neurons with {rate}hz and {shift}ms temporal shift, #{i+1}/{num_conds}")
    # Repeat *eval_reps* times for each condition
    for eval_rep in range(eval_reps):
        rep_start_time = time.time()  # Keeping track of execution times
        T = Tempotron(num_neurons=num_neur, tau=tau, threshold=threshold) # Set up model for current repetition
        acc_before.loc[num_neur, rate, shift][eval_rep] = T.accuracy(samples, labels)  # Evaluate accuracy - before
        # Train over the entire set *train_reps* times
        for _ in range(train_reps):
            T.train(samples, labels, learning_rate=learning_rate)
        acc_after.loc[num_neur, rate, shift][eval_rep] = T.accuracy(samples, labels)  # Evaluate accuracy - after
        rep_times.loc[num_neur, rate, shift][eval_rep] = time.time() - rep_start_time
acc_diff = acc_after - acc_before
# %% Saving data
date = datetime.datetime.now()
datestr = f'{date.day:02}{date.month:02}{date.year-2000}'

acc_before.to_csv(f'Data/acc_before_{datestr}.csv')
acc_after.to_csv(f'Data/acc_after_{datestr}.csv')
acc_diff.to_csv(f'Data/acc_diff_{datestr}.csv')
rep_times.to_csv(f'Data/runtimes_{datestr}.csv')


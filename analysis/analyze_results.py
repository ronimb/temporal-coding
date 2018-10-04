import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# %% Set metadata
# Columns in the imported data table
columns = [
    'num_neur',         # Number of neurons in the stimulus
    'freq',             # Average firing frequency of neurons in stimulus
    'samp_num',         # Sample index
    'distance',         # Distance between pair of neurons
    'span',             # Time span of the synaptic function used to produce vesicles
    'threshold',        # Threshold for firing (a "1" classification)
    'learning_rate',    # Model learning rate
    'pre',              # Discriminability before training
    'post',             # Discriminability after training
    'diff'              # Difference betweeen the two above
]

# Parameters dictionary with all possible values used in the model
parameters = dict(
    num_neur=[30, 150, 500],
    freq=[15, 50, 100],
    distance=[0.05, 0.1, 0.2, 0.3],
    span=[3, 6, 9],
    # The following two parameters were only used for evaluating optimal learning parameters
    threshold=np.sort
        (
        np.multiply(
            [[2.5], [5.0], [7.5]], [[1e-4, 1e-3, 1e-2, 1e-1]]).
            flatten()
    ),
    learning_rate=np.array([1e-6, 5e-6, 1e-5,
                            5e-5, 1e-4, 5e-4,
                            1e-3, 5e-3]))

# Conditions to summarize for
# The model hyperparameters
eval_learning_conditions = ['threshold', 'learning_rate']
# These conditions are those used when evaluating threshold and learning_rate, for which span=6 and distance=0.3 always
eval_summary_conditions = ['num_neur', 'freq']
# These are the actual conditions of interests
summary_conditions = ['num_neur', 'freq', 'distance', 'span']

# %% Load file and treat categories
file_name = 'Results/preliminary_results_lowdist.csv'
all_data = pd.read_csv(file_name)

categorical_columns = columns[:-3] # Columns containing categorical data from category list
for col in categorical_columns:
    all_data[col] = all_data[col].astype('category')
# Exclude the sample index number from actual categorical analyses
categorical_columns.remove('samp_num')
condition_samp_count = all_data.groupby(["num_neur", "freq", "threshold", "learning_rate"]).count()['samp_num']
# %% Clean the data
grouped = all_data.groupby(categorical_columns)
category_means = grouped.mean()
# Dump zero differences
zero_diff_indexes = np.isclose(category_means['diff'],0)
category_means = category_means[~zero_diff_indexes]
# Dump trials with erratic pre-training discrimination
pre_matching_indexes = np.isclose(category_means['pre'], 0.5,
                                 atol=5e-3)
category_means = category_means[pre_matching_indexes]
# %% Take only the upper %10 percent within each condition
condition_combs = product(*[parameters[condition] for condition in eval_summary_conditions])
top_means = pd.DataFrame()
for cnd_values in condition_combs:
    # ToDO: Fix this to work with all sets of conditions using the names somehow
    data = category_means.loc[cnd_values]
    high_diff_indexes = data['diff'] >= np.percentile(data['diff'], 90) # Select top 10 percents
    high_diff_data = data[high_diff_indexes]
    high_diff_data['num_samps'] = 0 # Variable used to track number of samples used in evaluating the mean
    high_diff_data.index = high_diff_data.index.droplevel(['distance', 'span']) # This only needs to happen now with the evaluation

    for (threshold, learning_rate) in high_diff_data.index:
        # TODO: Generalize this to work with all sets of conditions
        count = condition_samp_count[cnd_values[0], cnd_values[1], threshold, learning_rate]
        high_diff_data.loc[(threshold, learning_rate), 'num_samps'] = count

    for val, name in zip(cnd_values, eval_summary_conditions):
        high_diff_data = pd.concat([high_diff_data], keys=[val], names=[name])

    top_means = top_means.append(high_diff_data)

# %%
top_means.to_csv('cleaned_lowdist_highdiffs.csv')
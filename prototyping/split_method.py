import numpy as np
import pandas as pd
from generation import StimuliSet
# %%

num_neur = 30
num_stims = np.random.randint(175, 225)
freq = 50
duration = 500

frac_stim__label_a = np.random.uniform(0.25, 0.75)
fraction_split = np.random.uniform(0.25, 0.75)

act_freq = freq * (duration / 1000)
stdev = int(act_freq / 6)

rand_stimuli = np.array([  # Stimulus
    [np.random.uniform(500, size=np.random.randint(act_freq - stdev, act_freq + stdev)) for _ in range(num_neur)]
    # Neuron
    for _ in range(num_stims)])
rand_labels = np.random.choice([0, 1], size=num_stims, p=[frac_stim__label_a, 1 - frac_stim__label_a])


s = StimuliSet(
    stimuli=rand_stimuli,
    labels=rand_labels,
    stimulus_duration=duration
)
a, b = s.split(0.37)
# %% Old...
# num_stimuli_set_1 = np.round(num_stims * fraction_split).astype(int)
# num_stimuli_set_2 = num_stims - num_stimuli_set_1
# stimuli_counts = [num_stimuli_set_1, num_stimuli_set_2]
#
# unique_labels, original_label_counts = np.unique(rand_labels, return_counts=True)
# num_label_a = original_label_counts[0]
# num_label_b = original_label_counts[1]
# df = pd.DataFrame(np.zeros((2, 2)), columns=['label_a', 'label_b'], index=['set1', 'set2'])
# df['label_a'] = np.multiply(num_label_a, stimuli_counts) / num_stims
# df['label_b'] = np.multiply(num_label_b, stimuli_counts) / num_stims
# df = df.round().astype(int)
#
# label_a_indexes = np.where(rand_labels == unique_labels[0])[0]
# label_b_indexes = np.where(rand_labels == unique_labels[1])[0]
#
# label_a_to_stimuliset_1 = np.random.choice(label_a_indexes, df.label_a.set1)
# label_a_to_stimuliset_2 = np.setdiff1d(label_a_indexes, label_a_to_stimuliset_1)
#
# # Assemble sets
# stimuli_set_a_indices = np.concatenate([label_a_to_stimuliset_1, label_b_to_stimuliset_1])
# stimuli_set_b_indices = np.concatenate([label_a_to_stimuliset_2, label_b_to_stimuliset_2])
# # %%
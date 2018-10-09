import numpy as np
import pandas as pd

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
num_stim_label_a = num_stims - np.sum(rand_labels)
num_stim_label_b = np.sum(rand_labels)

label_alloc = [num_stim_label_a, num_stim_label_b]
num_set_1 = np.round(num_stims * fraction_split)
num_set_2 = num_stims - num_set_1

df = pd.DataFrame(np.zeros((2, 2)), columns=['label_a', 'label_b'], index=['set1', 'set2'])

df['label_a'] = np.multiply(num_stim_label_a, [num_set_1, num_set_2]) / num_stims
df['label_b'] = np.multiply(num_stim_label_b, [num_set_1, num_set_2]) / num_stims
df = df.round()
df['sum'] = df.sum(1)
df.loc['sum'] = df.sum()

print(f'with {num_stims} stimuli obtained the following table: ')
print(df.head())
print(label_alloc)
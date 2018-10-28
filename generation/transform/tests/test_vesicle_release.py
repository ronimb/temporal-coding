from generation import make_stimulus
from generation.transform import fixed_release, stochastic_release
from itertools import product
import numpy as np
# %%
frequencies = [15, 50, 100]
test_stimuli = {'exact': {freq: [] for freq in frequencies},
                'average': {freq: [] for freq in frequencies}}
num_neurons = 50
num_stimuli = 10
duration = 500
for freq in frequencies:
    for i in range(num_stimuli):
        test_stimuli['average'][freq].append(
            make_stimulus(freq, num_neurons, duration)
        )
        test_stimuli['exact'][freq].append(
            make_stimulus(freq, num_neurons, duration, exact_frequency=True)
        )
# %%
def test_stochastic_release():
    durations = [3, 5, 7, 9, 12]
    ves_nums = [5, 20, 100]
    rel_probs = [0.1, 0.25, 0.5, 0.75, 1]
    rel_conditions = product(durations, ves_nums, rel_probs)
    for (release_duration, num_ves, rel_prob) in rel_conditions:
        for stimulus_type, all_freqs in test_stimuli.items():
            for frequency, stimuli in all_freqs.items():
                for stimulus in stimuli:
                    transformed = stochastic_release(stimulus, duration, release_duration, num_ves, rel_prob, 10)
# %%
if __name__ == '__main__':
    test_stochastic_release()

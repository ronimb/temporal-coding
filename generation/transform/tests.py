from generation.transform.temporal_shift import forward_shift, symmetric_interval_shift
from generation import make_stimulus
import numpy as np

# %%
frequencies = [15, 50, 100]
test_stimuli = {'exact': {freq: [] for freq in frequencies},
                'average': {freq: [] for freq in frequencies}}
num_neurons = 50
num_stimuli = 10
duration = .5
for freq in frequencies:
    for i in range(num_stimuli):
        test_stimuli['average'][freq].append(
            make_stimulus(freq, .5, num_neurons)
        )
        test_stimuli['exact'][freq].append(
            make_stimulus(freq, .5, num_neurons, exact_frequency=True)
        )


def test_forward_shift():
    shifts = [5, 10, 50, 100]
    num_shifted = 30
    for shift in shifts:
        for stimulus_type, all_freqs in test_stimuli.items():
            for frequency, stimuli in all_freqs.items():
                for stimulus in stimuli:
                    # Generate shifted samples in current condition
                    shifted = forward_shift(stimulus, duration,
                                            max_temporal_shift=shift / 1000,
                                            num_shifted=num_shifted)
                    # Test for spikes exceeding max duration
                    exceeding_duration = [any([(neuron > (duration * 1000)).any() for neuron in shifted_stimulus]) for
                                          shifted_stimulus in shifted]
                    if any(exceeding_duration):
                        raise Exception(f'Some neurons exceed maximal duration when shift={shift}')
                    difference = shifted - stimulus
                    exceeding_shift = [any([(diff > shift).any() for diff in shifted_diff])
                                       for shifted_diff in difference]
                    if any(exceeding_shift):
                        raise Exception(f'Some neurons exceed maximal shift when shift={shift}')
    print('forward shifting passed testing successfully')


def test_symmetric_shift():
    intervals = [[0, 5], [0, 10], [0, 100], [5, 10], [5, 100], [25, 50]]
    num_shifted = 30
    for interval in intervals:
        for stimulus_type, all_freqs in test_stimuli.items():
            for frequency, stimuli in all_freqs.items():
                for stimulus in stimuli:
                    # Generate shifted samples in current condition
                    shifted = symmetric_interval_shift(stimulus, duration,
                                                       interval=np.divide(interval, 1000),
                                                       num_shifted=num_shifted)
                    # Test for spikes exceeding max duration
                    exceeding_duration = [any([(neuron > (duration * 1000)).any() for neuron in shifted_stimulus]) for
                                          shifted_stimulus in shifted]
                    if any(exceeding_duration):
                        raise Exception(f'Some neurons exceed maximal duration when shift={shift}')
                    # Test for spikes under t=0
                    before_start = [any([(neuron < 0).any() for neuron in shifted_stimulus]) for
                                    shifted_stimulus in shifted]
                    if any(before_start):
                        raise Exception(f'Some neurons spike before t=0 with interval={interval}')
                    difference = shifted - stimulus
                    exceeding_shift = [any([((np.abs(diff) > interval[1]) | (np.abs(diff) < interval[0]))
                                           .any() for diff in shifted_diff])
                                       for shifted_diff in difference]
                    if any(exceeding_shift):
                        raise Exception(f'Some neurons shifted outside of interval with interval={interval}')
    print('interval symmetric shifting passed testing successfully')


if __name__ == '__main__':
    test_forward_shift()
    test_symmetric_shift()

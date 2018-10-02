"""
This file contains functions used to manipulate "stimuli_sets";
These "stimuli setts" are typically a collection of transformed versions of two originak stimuli with the
some "distance based" relationship to one another(e.g. stimulus b is some short temporal variation of stimulus a and the stimuli in
the collection are labelled, transformed versions of either stimulus a or stimulus b).
"""
# %%
import numpy as np
from multiprocessing import Pool


# %%
def shuffle_set(stimuli_set):
    """
    This function is used to take a stimuli)set of stimuli shuffle their order in the list to
    randomize the order obtained by serial generation.

    in other words, nothing but a wrapper for np.random.shuffle

    """
    np.random.shuffle(stimuli_set)


def combine_and_label(set_a: np.array, set_b: np.array, shuffle: bool = True) -> np.array:
    """
     Combines transformed versions of two original stimuli in set_a and set_b into
     A labelled stimuli_set and returns it
    :param set_a: A Collection of transformed versions from stimulus_a
    :param set_b: A Collection of transformed versions from stimulus_b
    :param shuffle: Whether to shuffle the order of the items in the returned stimuli_set
    :return:
    """

    # Unpacking size parameters
    a_size = set_a.shape[0]
    b_size = set_b.shape[0]
    # Calculate total size
    total_size = a_size + b_size
    # Combine stimuli
    combined_stimuli = [*set_a, *set_b]
    # Create label vector
    labels = [*[0] * a_size, *[1] * b_size]
    # Create combined array
    stimuli_set = np.array(
        np.zeros(total_size),
        dtype=dict(
            names=('stimulus', 'label'),
            formats=(object, bool))
    )
    stimuli_set['stimulus'] = combined_stimuli
    stimuli_set['label'] = labels
    if shuffle:
        shuffle_set(stimuli_set)
    return stimuli_set

def convert_set(stimuli_set, number_of_neurons, stimulus_duration):
    @jit(parallel=True)
    def _convert_stimulus(stimulus):
        inds = []
        times = []
        counts = []
        num_events = []
        for i in prange(stimulus.shape[0]):
            neuron = np.trunc(stimulus[i] * 10) / 10
            time, count = np.unique(neuron, return_counts=True)
            num_events.append(time.shape[0])
            inds.extend([i] * time.shape[0])
            times.extend(time)
            counts.extend(count)
        return np.array([inds, times, counts])


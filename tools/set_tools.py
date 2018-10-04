"""
This file contains functions used to manipulate "stimuli_sets";
These "stimuli setts" are typically a collection of transformed versions of two originak stimuli with the
some "distance based" relationship to one another(e.g. stimulus b is some short temporal variation of stimulus a and the stimuli in
the collection are labelled, transformed versions of either stimulus a or stimulus b).
"""
# %%
import numpy as np
from multiprocessing import Pool
from generation.set_classes import StimuliSet


# %%
def shuffle_set(stimuli_set):
    """
    This function is used to take a stimuli)set of stimuli shuffle their order in the list to
    randomize the order obtained by serial generation.
    """
    num_stimuli = len(stimuli_set)
    rand_inds = np.random.choice(range(num_stimuli), replace=False, size=num_stimuli).astype(int)
    stimuli_set.labels = stimuli_set.labels[rand_inds]
    stimuli_set.stimuli = stimuli_set.stimuli[rand_inds]


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
    combined_stimuli = np.array([*set_a, *set_b])
    # Create label vector
    labels = np.array([*[0] * a_size, *[1] * b_size])
    # Create combined array
    stimuli_set = StimuliSet(
        stimuli=combined_stimuli,
        labels=labels,
        converted=False
    )
    if shuffle:
        shuffle_set(stimuli_set)
    return stimuli_set

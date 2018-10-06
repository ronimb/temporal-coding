"""
This file contains functions used to manipulate "stimuli_sets";
These "stimuli setts" are typically a collection of transformed versions of two originak stimuli with the
some "distance based" relationship to one another(e.g. stimulus b is some short temporal variation of stimulus a and the stimuli in
the collection are labelled, transformed versions of either stimulus a or stimulus b).
"""
# %%
import numpy as np
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


def combine_sets(transformed_from_a: np.array,
                 transformed_from_b: np.array, shuffle: bool = True) -> StimuliSet:
    """
     Combines transformed versions of two original stimuli in set_a and set_b into
     A labelled stimuli_set and returns it
    :param transformed_from_a: A StimuliSet of transformed versions from stimulus_a
    :param transformed_from_b: A StimuliSet of transformed versions from stimulus_b
    :param shuffle: Whether to shuffle the order of the items in the returned stimuli_set
    :return: stimuli_set:  A StimuliSet object with the following fields:
                         stimuli -  A collection of stimuli in the following format:
                                    Normal:     A numpy object array where each item is a stimulus as an array of
                                                neurons and their respective event times
                         labels -   Label for each stimulus in the set according to its origin
                                    (from one of two possible original stimuli)
                         converted - Wheter the stimuli are converted or not, in this case False
     NOTE: Other fields for stimuli_set are added when generating using make_stimuli_set
    """

    # Unpacking size parameters
    a_size = len(transformed_from_a)
    b_size = len(transformed_from_b)
    # Calculate total size
    total_size = a_size + b_size
    # Combine stimuli
    combined_stimuli = np.array([*transformed_from_a.stimuli,
                                 *transformed_from_b.stimuli])
    # Create label vector
    labels = np.array([*[0] * a_size, *[1] * b_size])
    # Create combined array
    stimuli_set = StimuliSet(
        stimuli=combined_stimuli,
        labels=labels,
        converted=False,
        stimulus_duration=transformed_from_a.stimulus_duration
    )
    if shuffle:
        shuffle_set(stimuli_set)
    return stimuli_set


def load_set(file_location):
    pass


def split_train_test(stimuli_set, training_set_size):
    num_stimuli = len(stimuli_set)
    all_indexes = np.arange(num_stimuli)
    indexes_of_training_stimuli = np.random.choice(all_indexes, size=training_set_size, replace=False)
    if stimuli_set.converted:
        test_set = None
        training_set = None
    else:
        test_set = None
        training_set = None
    return test_set, training_set


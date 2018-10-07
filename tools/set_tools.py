"""
This file contains functions used to manipulate "stimuli_sets";
These "stimuli setts" are typically a collection of transformed versions of two originak stimuli with the
some "distance based" relationship to one another(e.g. stimulus b is some short temporal variation of stimulus a and the stimuli in
the collection are labelled, transformed versions of either stimulus a or stimulus b).
"""
# %%
import numpy as np

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


def load_set(file_location):
    pass


#ToDo: Finish split_train_test
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


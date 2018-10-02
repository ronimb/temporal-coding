"""
This is used to convert a stimulus to the format with which brian can work for indexing
Event times
"""
import numpy as np
from numba import jit, prange


# %%
@jit(parallel=True)
def convert_stimulus(stimulus: object) -> object:
    """

    :param stimulus: Numpy array where each element is a single neurons spike time, specified in milliseconds
    :return: converted_stimulus: A structured numpy array with the following fields: index, time, count
                                 where each item represents events taking place at a 'time' from neuron #'index' in the
                                 set, and with 'count' representing the number of events that took place at that junction.
                                 count is required to account for numerous vesicles released in short time intervals
    """
    # create placeholder lists
    indexes = []
    times = []
    counts = []
    for i in prange(stimulus.shape[0]):
        neuron = np.trunc(stimulus[i] * 10) / 10
        event_times, counts = np.unique(neuron, return_counts=True)
        indexes.extend([i] * event_times.shape[0])
        times.extend(*event_times)
        counts.extend(*counts)

    converted_stimulus = np.array(
        np.zeros(len(times)),
        dtype=dict(
            names=('index', 'time', 'count'),
            formats=(int, float, int)
        ))
    converted_stimulus['index'] = indexes
    converted_stimulus['time'] = times
    converted_stimulus['count'] = counts
    return converted_stimulus

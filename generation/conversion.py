"""
This is used to convert a stimulus to the format with which brian can work for indexing
Event times
"""
import numpy as np
from numba import jit, prange
from multiprocessing import Pool
from generation.set_classes import StimuliSet


# %%
@jit(parallel=True)
def convert_stimulus(stimulus: np.array) -> np.array:
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
        event_times, event_counts = np.unique(neuron, return_counts=True)
        indexes.extend([i] * event_times.shape[0])
        times.extend(event_times)
        counts.extend(event_counts)

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


def convert_stimuli_set(stimuli_set: StimuliSet,
                        pool_size: int = 8):
    """
    neuron index consider both neuron number and stimulus number
    :param stimuli_set:  A StimuliSet object with the following fields:
                         stimuli -  A collection of stimuli in the following format:
                                    Normal:     A numpy object array where each item is a stimulus as an array of
                                                neurons and their respective event times
                         labels -   Label for each stimulus in the set according to its origin
                                    (from one of two possible original stimuli)
                         original_stimuli - tuple containing both original stimuli as numpy arrays of neurons and their
                                            corresponding event times (spikes or vesicle releases)
                         original_stimuli_distance - The average spike-distance metric between neurons in the two stimuli
                         converted - Wheter the stimuli are converted or not, in this case will be set to True
                         stimulus_duration - copy of the stimulus stimulus_duration above

    :param pool_size: number of cores to use for multiprocessing

    this function will convert the stimuli field to the following format:
    Converted:  A structured numpy array with the following fields: index, time, count
                                    where each item represents events taking place at a 'time',
                                    originating from neuron #'index' in the set,
                                    and with 'count' representing the number of events that took place at that junction.
                                    count is required to account for numerous vesicles released in short time intervals
    and set the converted flag as True
    """
    # Extract number of neurons
    number_of_neurons = stimuli_set.stimuli[0].shape[0]

    with Pool(pool_size) as p:
        res = p.map(convert_stimulus, stimuli_set.stimuli)
        p.close()
        p.join()
    ts = np.hstack(
        [
            [x['index'] + number_of_neurons * i, x['time'], x['count']]
            for i, x in enumerate(res)]
    )
    converted_samples = np.zeros(ts.shape[1],
                                 dtype={'names': ('index', 'time', 'count'),
                                        'formats': (int, float, int)})
    converted_samples['index'] = ts[0]
    converted_samples['time'] = ts[1]
    converted_samples['count'] = ts[2]
    stimuli_set.stimuli = converted_samples
    stimuli_set.converted = True

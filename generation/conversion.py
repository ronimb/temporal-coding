"""
This is used to convert a stimulus to the format with which brian can work for indexing
Event times
"""
import numpy as np
from numba import jit, prange
from multiprocessing import Pool


# %%
@jit(parallel=True)
def convert_stimulus(stimulus: np.array) -> np.array:
    """

    :param stimulus: Numpy array where each element is a single neurons spike time, specified in milliseconds
    :return: converted_stimulus: A structured numpy array with the following fields: index, time, count
                                 where each item represents events taking place at a 'time' from neuron #'index' in the
                                 set, and with 'count' representing the number of events that took place at that junction.
                      from generation.make_stimuli_set import make_set_from_specs
from generation import transform
from classification import evaluate, batch_train
from classification import Tempotron
from time import time
# %%
set_size = 200

stimuli_creation_params = dict(
    frequency=15,
    number_of_neurons=30,
    stimulus_duration=500,
)

set_transform_function = transform.stochastic_release
set_transform_params = dict(
    release_duration=5,
    number_of_vesicles=20,
    stimulus_duration=stimuli_creation_params['stimulus_duration'],
    release_probability=1,
    num_transformed=50
)

origin_transform_function = transform.symmetric_interval_shift
origin_transform_params = dict(
    stimulus_duration=stimuli_creation_params['stimulus_duration'],
    interval=(3, 7)
)

tempotron_params = dict(
    number_of_neurons=stimuli_creation_params['number_of_neurons'],
    tau=2,
    threshold=0.05,
    stimulus_duration=stimuli_creation_params['stimulus_duration']
)

training_params = dict(
    learning_rate=1e-3,
    batch_size=50,
    training_steps=15,
)
# %%
stimuli_set = make_set_from_specs(**stimuli_creation_params, set_size=set_size,
                                   set_transform_function=set_transform_function,
                                   set_transform_params=set_transform_params)

T = Tempotron(**tempotron_params)

t1 = time.time()
pre = evaluate(stimuli_set, tempotron=T)
batch_train(stimuli_set, tempotron=T, **training_params)
post = evaluate(stimuli_set, tempotron=T)
d1 = time.time() - t1
# %%
stimuli_set._make_tempotron_converted()

T = Tempotron(**tempotron_params)

t2 = time.time()
pre = evaluate(stimuli_set, tempotron=T)
batch_train(stimuli_set, tempotron=T, **training_params)
post = evaluate(stimuli_set, tempotron=T)
d2 = time.time() - t1

print(d1)
print(d2)           count is required to account for numerous vesicles released in short time intervals
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


def convert_stimuli_set(stimuli_set: object,
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
    return converted_samples

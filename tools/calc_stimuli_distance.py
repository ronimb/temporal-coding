import pyspike as spk
import numpy as np
def calc_stimuli_distance(stimulus_a: np.array, stimulus_b: np.array,
                          stimulus_duration: float) -> object:
    """
    This function computes the average distance between neurons in two stimuli
    using the spike-distance metric  (see: http://www.scholarpedia.org/article/SPIKE-distance)

    :param stimulus_a: numpy array where each element is a single neurons spike times, specified in milliseconds
    :param stimulus_b: numpy array where each element is a single neurons spike times, specified in milliseconds
    :param stimulus_duration: Maximal stimulus_duration of the stimulus, units: Sec
    """
    # Verify stimuli are comparable
    if stimulus_a.size != stimulus_b.size:
        raise Exception('Stimuli must consist of same number of neurons')

    distances = [] # Placeholder for distances between each pair of neurons
    for neuron_a, neuron_b in zip(stimulus_a, stimulus_b):
        # Converting to pyspike SpikeTrain object for calculation
        neuron_a = spk.SpikeTrain(neuron_a, edges=[0, stimulus_duration * 1000])
        neuron_b = spk.SpikeTrain(neuron_b, edges=[0, stimulus_duration * 1000])
        # Compute distance
        distance = spk.spike_distance(neuron_a, neuron_b)
        distances.append(distance)
    mean_distance = np.mean(distance)
    return mean_distance
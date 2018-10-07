import pyspike as spk
import numpy as np


def calc_stimuli_distance(stimulus_a: np.array, stimulus_b: np.array) -> object:
    """
    This function computes the average distance between neurons in two stimuli
    using the spike-distance metric  (see: http://www.scholarpedia.org/article/SPIKE-distance)

    :param stimulus_a: A stimulus object
    :param stimulus_b: Another stimulus objects
    """
    # Verify stimuli are comparable
    if stimulus_a.size != stimulus_b.size:
        raise Exception('Stimuli must consist of same number of neurons')
    elif stimulus_a.stimulus_duration != stimulus_b.stimulus_duration:
        raise Exception('Stimuli must be of equal duration')

    distances = []  # Placeholder for distances between each pair of neurons
    for neuron_a, neuron_b in zip(stimulus_a.neurons, stimulus_b.neurons):
        # Converting to pyspike SpikeTrain object for calculation
        neuron_a = spk.SpikeTrain(neuron_a.events, edges=[0, neuron_a.stimulus_duration])
        neuron_b = spk.SpikeTrain(neuron_b.events, edges=[0, neuron_b.stimulus_duration])
        # Compute distance
        distance = spk.spike_distance(neuron_a, neuron_b)
        distances.append(distance)
    mean_distance = np.mean(distance)
    return mean_distance

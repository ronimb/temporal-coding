# A stimulus is a collection of all the afferent neurons converging on the target neuron at the synapse
import numpy as np
from data_classes import Neuron, Stimulus


def _bool_poisson(frequency: int, num_neurons: int, stimulus_duration: float, dt: float = 1e-5) -> np.array:
    """
    Intended for internal use only, generates collection of poisson neurons and returns
    a boolean array of spikes according to the specified stimulus_duration and dt.

    :param frequency: The average firing frequency of each neuron in the sample, units: Hz
    :param num_neurons: Number of neurons in the stimulus, units: Integer
    :param stimulus_duration: Maximal stimulus_duration of the stimulus, units: ms
    :param dt: Simulation time step, units: Sec
    """
    duration_sec = stimulus_duration / 1000
    num_bins = np.round(duration_sec / dt).astype(int)  # Number of time bins in the simulation

    # Generate uniform random values between 0-1 in each time bin for each neuron
    random_vals = np.random.rand(num_neurons, num_bins)

    # Insert spike wherever the random values is smaller then frequency * dt
    bool_spikes = random_vals <= (frequency * dt)  # As boolean array

    return bool_spikes


## REFRACTORY PERIOD NOT YET IMPLENETED!
def make_stimulus(frequency: int, number_of_neurons: int, stimulus_duration: float, refractory_period: float = 2e-3,
                  dt: float = 1e-5, exact_frequency=False) -> np.array:
    """
    Used to create a stimulus which consists of multiple neurons all firing with the same frequency.
    may be used to generate with either an average frequency or an exact frequency.

    :param frequency: The average firing frequency of each neuron in the sample, units: Hz
    :param number_of_neurons: Number of neruons in the stimulus, units: Integer
    :param stimulus_duration: Maximal stimulus_duration of the stimulus, units: ms
    :param refractory_period: Length of minimal period between two spikes, units: Sec  CURRENTLY NOT IMPLEMENTED
    :param dt: Simulation time step, units: Sec
    :param exact_frequency: whether all neurons fire with the same exact frequency, or the same average frequency

    :rtype: np.array
    :return: stimulus: numpy array where each element is a single neurons spike times, specified in milliseconds
    """

    def _return_exact(bool_stimulus: np.array) -> np.array:
        """
        Used for filtering the stimulus for neurons firing only at the exact frequency
        :param bool_stimulus: stimulus to be filtered
        :return: exact: filtered boolean stimulus
        """
        duration_sec = stimulus_duration / 1000
        # Count spikes in each neuron
        spike_count = bool_stimulus.sum(1)
        # Find neurons firing at the correct frequency
        correct_count = spike_count == np.round((frequency * duration_sec))  # Rounding to handle edge cases
        # Keep only those neurons firing at the correct frequency
        exact = bool_stimulus[correct_count]
        return exact

    # Generate the stimulus in boolean form
    spikes_bool = _bool_poisson(frequency, number_of_neurons, stimulus_duration, dt)

    # Check that each neuron spikes at least once and re-generate otherwise
    num_spikes = spikes_bool.sum(1)
    zero_spikes = num_spikes == 0
    while zero_spikes.any():
        num_zero_spikes = zero_spikes.sum()  # Count  number of neurons with no spikes
        new_neurons = _bool_poisson(frequency, num_zero_spikes, stimulus_duration)  # Generate new neurons
        spikes_bool[zero_spikes] = new_neurons
        # Check again
        num_spikes = spikes_bool.sum(1)
        zero_spikes = num_spikes == 0

    # Handle exact frequency requirement
    if exact_frequency:
        # Filter out neurons not firing at the exact frequency
        spikes_bool = _return_exact(spikes_bool)
        # Generate new neurons until we have the desired number of neurons firing at the exact frequency
        while spikes_bool.shape[0] < number_of_neurons:
            # Generate new neurons
            new_neurons_bool = _bool_poisson(frequency, number_of_neurons * 2, stimulus_duration, dt)
            # Filter these new neurons
            new_neurons_bool = _return_exact(new_neurons_bool)
            # Add these correct neurons to the stimulus
            spikes_bool = np.append(spikes_bool, new_neurons_bool, 0)
        # Making sure we have precisely the desired number of neurons and no more
        spikes_bool = spikes_bool[0:number_of_neurons, :]

    # Transforming the boolean stimulus to an array in which each object represents a neuron and its spike times
    neuron_index, firing_indexes = np.where(spikes_bool)  # Find indexes of neurons and indexes of spikes
    times = firing_indexes * dt * 1000  # Transform firing time indexes to seconds
    # Create the stimulus object
    stimulus = np.array([times[neuron_index == i] for i in
                         range(number_of_neurons)])  # Create the array, a numpy object array is used for indexing reasons

    return stimulus

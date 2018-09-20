# A stimulus is a collection of all the afferent neurons converging on the target neuron at the synapse
import numpy as np


def _bool_poisson(frequency: int, duration: float, num_neurons: int, dt: float = 1e-5) -> np.array:
    """
    Intended for internal use only, generates collection of poisson neurons and returns
    a boolean array of spikes according to the specified duration and dt.

    :param frequency: The average firing frequency of each neuron in the sample, units: Hz
    :param duration: Maximal duration of the stimulus, units: Sec
    :param num_neurons: Number of neruons in the stimulus, units: Integer
    :param dt: Simulation time step, units: Sec
    """

    num_bins = np.round(duration / dt).astype(int)  # Number of time bins in the simulation

    # Generate uniform random values between 0-1 in each time bin for each neuron
    random_vals = np.random.rand(num_neurons, num_bins)

    # Insert spike wherever the random values is smaller then frequency * dt
    bool_spikes = random_vals <= (frequency * dt)  # As boolean array

    return bool_spikes


## REFRACTORY PERIOD NOT YET IMPLENETED!
def make_stimulus(frequency: int, duration: float, num_neurons: int,
                  refractory_period: float = 2e-3, dt: float = 1e-5,
                  exact_frequency=False) -> np.array:
    """
    Used to create a stimulus which consists of multiple neurons all firing with the same frequency.
    may be used to generate with either an average frequency or an exact frequency.

    :param frequency: The average firing frequency of each neuron in the sample, units: Hz
    :param duration: Maximal duration of the stimulus, units: Sec
    :param num_neurons: Number of neruons in the stimulus, units: Integer
    :param refractory_period: Length of minimal period between two spikes, units: Sec  CURRENTLY NOT IMPLEMENTED
    :param dt: Simulation time step, units: Sec
    :param exact_frequency: whether all spikes to use average frequency or exact frequency for all neurons

    :rtype: np.array
    :return: stimulus: numpy  array where each element is a single neurons spike times, specified in milliseconds
    """
    
    def _return_exact(bool_stimulus: np.array) -> np.array:
        """
        Used for filtering the stimulus for neurons firing only at the exact frequency
        :param bool_stimulus: stimulus to be filtered
        :return: exact: filtered boolean stimulus
        """
        # Count spikes in each neuron
        spike_count = bool_stimulus.sum(1)
        # Find neurons firing at the correct frequency
        correct_count = spike_count == np.round((frequency * duration))  # Rounding to handle edge cases
        # Keep only those neurons firing at the correct frequency
        exact = bool_stimulus[correct_count]
        return exact
        
    # Generate the stimulus in boolean form
    spikes_bool = _bool_poisson(frequency, duration, num_neurons, dt)
    
    # Handle exact frequency requirement
    if exact_frequency:
        # Filter out neurons not firing at the exact frequency
        spikes_bool = _return_exact(spikes_bool)
        # Generate new neurons until we have the desired number of neurons firing at the exact frequency
        while spikes_bool.shape[0] < num_neurons:
            # Generate new neurons
            new_neurons_bool = _bool_poisson(frequency, duration, num_neurons * 2, dt)
            # Filter these new neurons
            new_neurons_bool = _return_exact(new_neurons_bool)
            # Add these correct neurons to the stimulus
            spikes_bool = np.append(spikes_bool, new_neurons_bool, 0)
        # Making sure we have precisely the desired number of neurons and no more
        spikes_bool = spikes_bool[0:num_neurons, :]

    # Transforming the boolean stimulus to an array in which each object represents a neuron and its spike times
    neuron_index, firing_indexes = np.where(spikes_bool)  # Find indexes of neurons and indexes of spikes
    times = firing_indexes * dt * 1000  # Transform firing time indexes to seconds
    stimulus = np.array([times[neuron_index == i] for i in range(num_neurons)]) # Create the array, a numpy object array is used for indexing reasons

    return stimulus

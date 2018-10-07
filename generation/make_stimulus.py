# A stimulus is a collection of all the afferent neurons converging on the target neuron at the synapse
from data_classes import Stimulus

#TODO: get rid of this file once all tests have been rereferenced to the Stimulus object
def make_stimulus(frequency: int, number_of_neurons: int, stimulus_duration: float, refractory_period: float = 2e-3,
                  dt: float = 1e-5, exact_frequency=False) -> Stimulus:
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
    :return: stimulus: a Stimulus object
    """

    return Stimulus.make(frequency, number_of_neurons, stimulus_duration, refractory_period, dt, exact_frequency)

# A stimulus is a collection of all the afferent neurons converging on the target neuron at the synapse
import brian2 as b2
from brian2.units import ms, Hz
import numpy as np


## REFRACTORY PERIOD NOT YET IMPLENETED!
def make_poisson(frequency: int, duration: float, num_neurons: int,
                 refractory_period: float = 2e-3, dt: float = 1e-5) -> object:
    """


    :param frequency: The average firing frequency of each neuron in the sample, units: Hz
    :param duration: Duration of the stimulus, units: Sec
    :param num_neurons: Number of neruons in the stimulus
    :param refractory_period: Length of minimal period between two spikes, units: Sec  CURRENTLY NOT IMPLEMENTED
    :param dt: Simulation time step, units: Sec

    :rtype: np.array
    :return: stimulus: numpy object array where each element is a single neurons spike times
    """

    num_bins = np.round(duration / dt).astype(int)  # Number of time bins in the simulation

    # Generate uniform random values between 0-1 in each time bin for each neuron
    random_vals = np.random.rand(num_neurons, num_bins)

    # Insert spike wherever the random values is smaller then frequency * dt
    spike_inds = random_vals <= (frequency * dt)  # As boolean array
    inds, times = np.where(spike_inds)  # Convert to list of arrays: [neuron_indexes, spike_times]
    times = times * dt  # Transform index numbers to seconds
    stimulus = [times[inds == i] for i in range(num_neurons)]
    # Remove spikes that violate the refractory period - AWAITING IMPLEMENTATION
    # if refractory_period:
    #     for i in range(num_neurons):
    #         spikes = spike_inds[1][spike_inds[0] == i]  # Isolate spikes from current neuron
    #         diffs = np.diff(spikes)  # Distance between each two successive spikes in the neuron
    #         violations = diffs <= refractory_period
    #         good = np.where(~violations)[0]
    return np.array(stimulus)

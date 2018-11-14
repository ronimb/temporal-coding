"""
This file contains all the functions required for transforming a stimulus
by transforming spikes into released vesicles
"""
# %%
import numpy as np
from scipy.stats import beta
import sympy as sym
import attr


# %%
def get_beta_params(beta_mode: float, beta_span: float = 15,
                    mode_centrality: float = 3, decay_velocity: float = 2,
                    return_params=False) -> tuple:
    """
    This function is used to derive the beta distribution A and B parameters
    for the customized heavy tailed beta_distribution

    :param beta_span: Maximal stimulus_duration of the heavy tailed beta distribution
    :param beta_mode: The mode of the distribution
    :param mode_centrality: Parameter controlling how tight the probability distribution is around the mode - Higher values will result in more values closer to the mode
    :param decay_velocity: Parameter controlling how fast the probability decays after the mode, higher values will lead to less extreme values
    """
    # Calculate parameters for beta distribution
    beta_a = 1 + (mode_centrality / beta_span) * (decay_velocity - 1)
    beta_b = (beta_span * (beta_a - 1) - beta_mode * (beta_a - 2)) / beta_mode

    # Create scipy distribution object to represent beta distribution
    beta_dist = beta(a=beta_a, b=beta_b, scale=beta_span)

    if return_params:
        return beta_dist, beta_a, beta_b
    else:
        return beta_dist



def make_beta_bins(release_duration, beta_dist, dt_ms=0.001, time_bins=[]):
    """
    This function is used to generate the possible vesicle release times, along with
    the probablities of release for each possible release time
    """

    # Create time bins
    if not any(time_bins):  # Create time bins
        bin_borders = np.arange(0 + dt_ms, release_duration + dt_ms, dt_ms)  # Create border values in specified range
        time_bins = bin_borders[:-1] + (dt_ms / 2)  # Find centers between each each two borders
    else:
        bin_borders = np.array([*np.r_[time_bins - (dt_ms / 2)], time_bins[-1] + (dt_ms / 2)])

    # Calculate probability for each possible time value
    probabilities = beta_dist.cdf(bin_borders[1:]) - beta_dist.cdf(bin_borders[:-1])

    # Retain only bins within the possible release stimulus_duration
    possible_inds = time_bins <= release_duration
    time_bins = time_bins[possible_inds]
    probabilities = probabilities[possible_inds]

    return time_bins, probabilities



def fixed_release(stimulus: np.array, stimulus_duration: float, release_duration: float, number_of_vesicles: int,
                  release_probability: float = 1, distribution_mode: int = 1, mode_centrality: int = 3,
                  decay_velocity: int = 2, num_transformed: int = 1) -> np.array:
    """
    This function takes a stimulus, and creates transformed versions of it by generating vesicle release processes
    at each spikes occurence in each of the stimulus' neurons.
    Returns a collection of transformed versions of the stimulus, where in each one the times correspond to times of vesicle release

    :param stimulus: A numpy array representing the stimulus in which each line (or object) is a neuron with spike times given in ms
    :param release_duration: Maximal stimulus_duration of vesicle release process, Units : ms
    :param number_of_vesicles: Precise number of vesicles release for each spike
    :param stimulus_duration: maximal stimulus_duration of stimulus, Units : ms
    :param release_probability: Probability of release for each vesicle, the fraction of vesicles released for each spike
    :param distribution_mode: The mode of the heavy-tailed beta distribution used to determine release times
    :param mode_centrality: Parameter controlling how tight the probability distribution is around the mode - Higher values will result in more values closer to the mode
    :param decay_velocity: Parameter controlling how fast the probability decays after the mode, higher values will lead to less extreme release times
    :param num_transformed: Number of transformed version of the original stimulus to generate

    :return: all_transformed_stimuli: array where each object is a transformed stimulus
    """
    # Calculate parameters for beta distribution
    beta_a, beta_b, _ = get_beta_params(beta_span=release_duration, beta_mode=distribution_mode,
                                        mode_centrality=mode_centrality, decay_velocity=decay_velocity)

    def _release_for_neuron(neuron: np.array) -> np.array:
        """
        Uses a customized heavy tailed beta distribution to generate vesicle release times
        for all spikes from a single neuron
        :param neuron: An array with the spike times of a single neuron
        :return: transformed_neuron: An array with vesicle release times derived from neurons spikes
        """

        # Generate release time offsets
        number_of_spikes = neuron.shape[0]
        vesicle_time_offsets = np.random.beta(beta_a, beta_b,
                                              size=(number_of_vesicles, number_of_spikes)) * release_duration
        # Transform neuron by adding vesicle time offsets to spike times
        transformed_neuron = (neuron + vesicle_time_offsets).flatten()
        # Remove spikes that exceed the maximal stimulus_duration
        transformed_neuron = transformed_neuron[transformed_neuron < stimulus_duration]
        # Handle release probability by excluding part of the vesicles
        if release_probability != 1:
            # Generate uniform 0-1 random values for each vesicle
            random_values = np.random.rand(transformed_neuron.shape[0])
            # Remove part of the vesicles
            vesicles_to_keep = random_values <= release_probability
            transformed_neuron = transformed_neuron[vesicles_to_keep]
        # Sort the vesicle release times
        transformed_neuron.sort()

        return transformed_neuron

    def _transform_stimulus():
        """
        Transform all neurons in a stimulus by generating vesicle release times for each spike in each neuron
        :return: transformed_neurons: array with all transformed neurons from the original stimulus
        """
        transformed_neurons = []  # Placeholder for all neurons in stimulus
        for neuron in stimulus:
            # Generate the transformed neuron
            transformed_neuron = _release_for_neuron(neuron)
            # Add to the stimulus placeholder
            transformed_neurons.append(transformed_neuron)
        transformed_neurons = np.array(transformed_neurons)
        return transformed_neurons

    all_transformed_stimuli = []  # Placeholder for transformed stimuli
    for i in range(num_transformed):
        # Generate transformed stimulus
        transformed_stimulus = _transform_stimulus()
        # Add to array of transformed stimuli
        all_transformed_stimuli.append(transformed_stimulus)
    if num_transformed == 1:
        all_transformed_stimuli = np.array(all_transformed_stimuli)[0]
    else:
        all_transformed_stimuli = np.array(all_transformed_stimuli)
    return all_transformed_stimuli


def stochastic_release(stimulus: np.array, stimulus_duration: float, release_duration: float, number_of_vesicles: int,
                       release_probability: float = 1, distribution_mode: float = 1, mode_centrality: float = 3,
                       decay_velocity: float = 2, distribution_span: float = 15, expected_duration: float = 9,
                       num_transformed: int = 1) -> np.array:
    """
    This function is used to generate transformed versions of a stimulus by releasing vesicles in a stochastic manner
    in response to the spikes in the original stimulus.
    The number of vesicles released is stochastic, and as such the number of vesicles specified in this functions
    input variables reflects the expected rather than the precise number of vesicles which will be released.

    :param stimulus: A numpy array representing the stimulus in which each line (or object) is a neuron with spike times given in ms
    :param release_duration: Maximal stimulus_duration of vesicle release process, Units : ms
    :param number_of_vesicles: Expected number of vesicles release for each spike
    :param stimulus_duration: maximal stimulus_duration of stimulus, Units : ms
    :param release_probability: Probability of release for each vesicle, the fraction of vesicles released for each spike
    :param distribution_mode: The mode of the heavy-tailed beta distribution used to determine release times
    :param mode_centrality: Parameter controlling how tight the probability distribution is around the mode - Higher values will result in more values closer to the mode
    :param decay_velocity: Parameter controlling how fast the probability decays after the mode, higher values will lead to less extreme release times
    :param distribution_span: Parameter controlling the length of the actual distribution, actual release times are from a clipped interval, MUST BE LARGER THAN RELEASE DURATION!
    :param expected_duration: The stimulus_duration of time for which the expected number of vesicles released equals number_of_vesicles, MUST BET LARGER THAN RELEASE_DURATION, AND SMALLER THAN DISRIBUTION SPAN
    :param num_transformed: Number of transformed version of the original stimulus to generate

    :return: all_transformed_stimuli: array where each object is a transformed stimulus
    """

    # Calculate parameters for beta distribution and get representation of distribution
    beta_dist = get_beta_params(beta_mode=distribution_mode, beta_span=distribution_span,
                                mode_centrality=mode_centrality, decay_velocity=decay_velocity)
    # Calculate probability scaling factor to set expected number of vesicles at the expected stimulus_duration
    probability_scaling_factor = number_of_vesicles * beta_dist.cdf(expected_duration)

    # Generate vesicle release time bin centers and their probabilities
    possible_release_times, release_times_probabilities = make_beta_bins(release_duration, beta_dist)

    def _release_for_neuron(neuron: np.array) -> np.array:
        """
        Uses a customized heavy tailed beta distribution to stochastically generate vesicle release times
        for all spikes from a single neuron
        :param neuron: An array with the spike times of a single neuron
        :return: transformed_neuron: An array with vesicle release times derived from neurons spikes
        """

        number_of_spikes = neuron.shape[0]
        # Generate uniform random values 0-1 for each possible release time, for each spike
        random_values = np.random.rand(number_of_spikes, possible_release_times.shape[0])
        # Determine which bins fired using the time bin probabilities and the scaling factormas
        release_inds = (random_values / probability_scaling_factor) <= release_times_probabilities
        # Derive release time offsets from release inds and clipping to no more than the specified number of vesicles

        # vesicle_time_offsets = np.array([possible_release_times[inds][:number_of_vesicles] for inds in release_inds]) # With shunting
        vesicle_time_offsets = np.array([possible_release_times[inds] for inds in release_inds])  # No shunting

        # Transform neuron by adding vesicle time offsets to spike times
        if vesicle_time_offsets.ndim == 1:  # This takes care of possible issues with the time offset array
            transformed_neuron = np.hstack(neuron + vesicle_time_offsets)
        elif vesicle_time_offsets.ndim == 2:
            transformed_neuron = np.hstack(neuron[:, np.newaxis] + vesicle_time_offsets)
        # Remove spikes that exceed the maximal stimulus_duration
        transformed_neuron = transformed_neuron[transformed_neuron < stimulus_duration]
        # Handle release probability by excluding part of the vesicles
        if release_probability != 1:
            # Generate uniform 0-1 random values for each vesicle
            random_values = np.random.rand(transformed_neuron.shape[0])
            # Remove part of the vesicles
            vesicles_to_keep = random_values <= release_probability
            transformed_neuron = transformed_neuron[vesicles_to_keep]
        # Sort the vesicle release times
        transformed_neuron.sort()
        return transformed_neuron

    def _transform_stimulus():
        """
        Transform all neurons in a stimulus by generating vesicle release times for each spike in each neuron
        :return: transformed_neurons: array with all transformed neurons from the original stimulus
        """
        transformed_neurons = []  # Placeholder for all neurons in stimulus
        for neuron in stimulus:
            # Generate the transformed neuron
            transformed_neuron = _release_for_neuron(neuron)
            # Add to the stimulus placeholder
            transformed_neurons.append(transformed_neuron)
        transformed_neurons = np.array(transformed_neurons)
        return transformed_neurons

    all_transformed_stimuli = []  # Placeholder for transformed stimuli
    for i in range(num_transformed):
        # Generate transformed stimulus
        transformed_stimulus = _transform_stimulus()
        # Add to array of transformed stimuli
        all_transformed_stimuli.append(transformed_stimulus)
    if num_transformed == 1:
        all_transformed_stimuli = np.array(all_transformed_stimuli)[0]
    else:
        all_transformed_stimuli = np.array(all_transformed_stimuli)
    return all_transformed_stimuli



def stochastic_pool_release(stimulus: np.array, stimulus_duration: float, release_duration: float, max_pool_size: int,
                            max_fraction_released: float = 1, replenishment_rate: float = 15, num_transformed: int = 1,
                            longest_release_duration: float = 9, distribution_mode: float = 1,
                            mode_centrality: float = 3, decay_velocity: float = 2,
                            distribution_span: float = 15, dt_ms=0.01):
    """

    :param stimulus: A numpy array representing the stimulus in which each line (or object) is a neuron with spike times given in ms
    :param release_duration:
    :param max_pool_size:
    :param stimulus_duration:
    :param max_fraction_released:
    :param distribution_mode:
    :param mode_centrality:
    :param decay_velocity:
    :param distribution_span:
    :param longest_release_duration:
    :param num_transformed:
    :param replenishment_rate:
    :return:
    """
    time_bins = np.arange(0 + dt_ms, stimulus_duration + dt_ms, dt_ms)  # Vector of time bins
    # Calculate parameters for beta distribution and get representation of distribution
    beta_dist = get_beta_params(beta_mode=distribution_mode, beta_span=distribution_span,
                                mode_centrality=mode_centrality, decay_velocity=decay_velocity)
    # Deriving initial probabilities of release such that 1 vesicle is released at the longest_release_duration
    _, initial_probabilities = make_beta_bins(release_duration, beta_dist, dt_ms=dt_ms, time_bins=time_bins)
    # Calculate probability scaling factor to set expected number of vesicles at the expected stimulus_duration
    probability_scaling_factor = max_pool_size * beta_dist.cdf(longest_release_duration) # Scaled so as to give max_pool_size vesicles at longest_release duration
    initial_probabilities *= probability_scaling_factor  # Referred to as "P" in the equations

    # The above probablities essentially give us at each time instant, the chance that EXACTLY one vesicle will be released,
    # Thus, assuming independence of release, we can arrive at the independent probability of release for single vesicles
    p_, P_, n_ = sym.symbols(['p', 'P', 'n'], positive=True, real=True)

    def _derive_independent_probability(scaled_probs):
        expected_num_ves_released = np.round(max_pool_size * max_fraction_released).astype(int)
        actual_probs = np.zeros_like(scaled_probs)
        for i, P in enumerate(scaled_probs):
            sol = sym.solve([p_ >= 0, p_ <= 1,
                             expected_num_ves_released * (p_ * (1 - p_) ** (expected_num_ves_released - 1)) - P],
                            p_, quick=True)
            actual_probs[i] = sol.args[0].args[1]
        return actual_probs

    actual_probs = _derive_independent_probability(initial_probabilities)

    def _release_for_neuron(neuron: np.array) -> np.array:
        n_t = np.zeros_like(time_bins)  # Number of vesicles in the pool at each time instant
        p_t = np.zeros_like(time_bins)  # Independent release probability for vesicles at each time instant
        r_t = np.zeros_like(time_bins)  # Number of vesicles released at each time instant
        # Creating probability over time vector
        n_t[0] = max_pool_size
        for spike in neuron:
            start_ind = np.where(np.isclose(time_bins, spike))[0][0]
            end_ind = start_ind + actual_probs.shape[0]
            if end_ind > time_bins.shape[0]:
                end_ind = time_bins.shape[0]
                p_t[start_ind:] = actual_probs[:end_ind - start_ind]
            else:
                p_t[start_ind: end_ind] = actual_probs

        # Eulerian integration over all time points using simple loop structure
        for i, t in enumerate(time_bins[:-1]):
            replenished = (1 / replenishment_rate) * dt_ms * (1 - np.isclose(n_t[i], max_pool_size))
            # Calculate number of vesicles release
            released = r_t[i] = (np.random.rand(int(np.round(n_t[i], 5))) < p_t[i]).sum()  # Flip coin for each vesicle currently in pool
            # Calculate change in vesicle pool size
            dn = replenished - released

            # Calculate n at next time step
            n_t[i + 1] = n_t[i] + dn
        transformed_neuron = np.where(r_t)[0] * dt_ms
        return transformed_neuron

    def _transform_stimulus():
        """
        Transform all neurons in a stimulus by generating vesicle release times for each spike in each neuron
        :return: transformed_neurons: array with all transformed neurons from the original stimulus
        """
        transformed_neurons = []  # Placeholder for all neurons in stimulus
        for neuron in stimulus:
            # Generate the transformed neuron
            transformed_neuron = _release_for_neuron(neuron)
            # Add to the stimulus placeholder
            transformed_neurons.append(transformed_neuron)
        transformed_neurons = np.array(transformed_neurons)
        return transformed_neurons

    all_transformed_stimuli = []  # Placeholder for transformed stimuli
    for i in range(num_transformed):
        # Generate transformed stimulus
        transformed_stimulus = _transform_stimulus()
        # Add to array of transformed stimuli
        all_transformed_stimuli.append(transformed_stimulus)
    if num_transformed == 1:
        all_transformed_stimuli = np.array(all_transformed_stimuli)[0]
    else:
        all_transformed_stimuli = np.array(all_transformed_stimuli)
    return all_transformed_stimuli
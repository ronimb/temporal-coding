"""
This file contains all the functions required for transforming a stimulus
by transforming spikes into released vesicles
"""
# %%
import numpy as np
from scipy.stats import beta


# %%
def get_beta_params(beta_span: float, beta_mode: float,
                    mode_centrality: float, decay_velocity: float) -> tuple:
    """
    This function is used to derive the beta distribution A and B parameters
    for the customized heavy tailed beta_distribution

    :param beta_span: Maximal duration of the heavy tailed beta distribution
    :param beta_mode: The mode of the distribution
    :param mode_centrality: Parameter controlling how tight the probability distribution is around the mode - Higher values will result in more values closer to the mode
    :param decay_velocity: Parameter controlling how fast the probability decays after the mode, higher values will lead to less extreme values
    """
    # Calculate parameters for beta distribution
    beta_a = 1 + (mode_centrality / beta_span) * (decay_velocity - 1)
    beta_b = (beta_span * (beta_a - 1) - beta_mode * (beta_a - 2)) / beta_mode

    # Create scipy distribution object to represent beta distribution
    beta_dist = beta(a=beta_a, b=beta_b, scale=beta_span)

    return beta_b, beta_b, beta_dist


def fixed_release(stimulus: np.array, release_duration: float, number_of_vesicles: int, max_duration: float,
                  num_transformed: int = 1, release_probability: float = 1,
                  distribution_mode: int = 1, mode_centrality: int = 3, decay_velocity: int = 2) -> np.array:
    """
    This function takes a stimulus, and creates transformed versions of it by generating vesicle release processes
    at each spikes occurence in each of the stimulus' neurons.
    Returns a collection of transformed versions of the stimulus, where in each one the times correspond to times of vesicle release

    :param stimulus: A numpy array representing the stimulus in which each line (or object) is a neuron with spike times given in ms
    :param release_duration: Maximal duration of vesicle release process, Units : ms
    :param number_of_vesicles: Precise number of vesicles release for each spike
    :param max_duration: maximal duration of stimulus, Units : ms
    :param num_transformed: Number of transformed version of the original stimulus to generate
    :param release_probability: Probability of release for each vesicle, the fraction of vesicles released for each spike
    :param distribution_mode: The mode of the heavy-tailed beta distribution used to determine release times
    :param mode_centrality: Parameter controlling how tight the probability distribution is around the mode - Higher values will result in more values closer to the mode
    :param decay_velocity: Parameter controlling how fast the probability decays after the mode, higher values will lead to less extreme release times

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
        # Remove spikes that exceed the maximal duration
        transformed_neuron = transformed_neuron[transformed_neuron < max_duration]
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



def stochastic_release(stimulus: np.array, release_duration: float, number_of_vesicles: int,
                       max_duration: float, num_transformed: int = 1, release_probability: float = 1,
                       distribution_mode: float = 1, mode_centrality: float = 3, decay_velocity: float = 2,
                       distribution_span: float = 15, expected_duration: float = 9) -> np.array:
    """
    This function is used to generate transformed versions of a stimulus by releasing vesicles in a stochastic manner
    in response to the spikes in the original stimulus.
    The number of vesicles released is stochastic, and as such the number of vesicles specified in this functions
    input variables reflects the expected rather than the precise number of vesicles which will be released.

    :param stimulus: A numpy array representing the stimulus in which each line (or object) is a neuron with spike times given in ms
    :param release_duration: Maximal duration of vesicle release process, Units : ms
    :param number_of_vesicles: Expected number of vesicles release for each spike
    :param max_duration: maximal duration of stimulus, Units : ms
    :param num_transformed: Number of transformed version of the original stimulus to generate
    :param release_probability: Probability of release for each vesicle, the fraction of vesicles released for each spike
    :param distribution_mode: The mode of the heavy-tailed beta distribution used to determine release times
    :param mode_centrality: Parameter controlling how tight the probability distribution is around the mode - Higher values will result in more values closer to the mode
    :param decay_velocity: Parameter controlling how fast the probability decays after the mode, higher values will lead to less extreme release times
    :param distribution_span: Parameter controlling the length of the actual distribution, actual release times are from a clipped interval, MUST BE LARGER THAN RELEASE DURATION!
    :param expected_duration: The duration of time for which the expected number of vesicles released equals number_of_vesicles, MUST BET LARGER THAN RELEASE_DURATION, AND SMALLER THAN DISRIBUTION SPAN

    :return: all_transformed_stimuli: array where each object is a transformed stimulus
    """

    # Calculate parameters for beta distribution and get representation of distribution
    beta_a, beta_b, beta_dist = get_beta_params(beta_span=distribution_span, beta_mode=distribution_mode,
                                                mode_centrality=mode_centrality, decay_velocity=decay_velocity)
    # Calculate probability scaling factor to set expected number of vesicles at the expected duration
    probability_scaling_factor = number_of_vesicles * beta_dist.cdf(expected_duration)

    def _make_bins():
        """
        This function is used to generate the possible vesicle release times, along with
        the probablities of release for each possible release time
        """
        dt_ms = 0.001  # Hard coded time step in milliseconds

        # Create time bins
        bin_borders = np.arange(0, release_duration + dt_ms, dt_ms)  # Create border values in specified range
        centers = bin_borders[:-1] + (dt_ms / 2)  # Find centers between each each two borders

        # Calculate probability for each possible time value
        probabilities = beta_dist.cdf(bin_borders[1:]) - beta_dist.cdf(bin_borders[:-1])

        # Retain only bins within the possible release duration
        possible_inds = centers <= release_duration
        centers = centers[possible_inds]
        probabilities = probabilities[possible_inds]

        return centers, probabilities

    # Generate vesicle release time bin centers and their probabilities
    possible_release_times, release_times_probabilities = _make_bins()

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
        # Determine which bins fired using the time bin probabilities and the scaling factor
        release_inds = (random_values / probability_scaling_factor) <= release_times_probabilities
        # Derive release time offsets from release inds and clipping to no more than the specified number of vesicles
        vesicle_time_offsets = np.array([possible_release_times[inds][:number_of_vesicles] for inds in release_inds])
        # Transform neuron by adding vesicle time offsets to spike times
        if vesicle_time_offsets.ndim == 1: # This takes care of possible issues with the time offset array
            transformed_neuron = np.hstack(neuron + vesicle_time_offsets)
        elif vesicle_time_offsets.ndim == 2:
            transformed_neuron = np.hstack(neuron[:, np.newaxis] + vesicle_time_offsets)
        # Remove spikes that exceed the maximal duration
        transformed_neuron = transformed_neuron[transformed_neuron < max_duration]
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

    all_transformed_stimuli = [] # Placeholder for transformed stimuli
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

from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
# %%
def get_beta_params(beta_span: float, beta_mode: float,
                    mode_centrality: float, decay_velocity: float) -> tuple:
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

    return beta_b, beta_b, beta_dist

def plot_dist(dist):
    eps = 1e-55
    probs = np.linspace(eps, 1-eps, 10000)
    x = dist.ppf(probs)
    y = dist.pdf(x)
    plt.plot(x, y)

def _make_bins(beta_dist, release_duration):
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

    # Retain only bins within the possible release stimulus_duration
    possible_inds = centers <= release_duration
    centers = centers[possible_inds]
    probabilities = probabilities[possible_inds]

    return centers, probabilities
# %%
number_of_spikes = 100

spans = np.array([3, 6, 9])
number_of_vesicles = 20

decay_velocity = 2
mode_centrality = 3

distribution_span = 15

expected_duration = 9

a, b, beta_dist = get_beta_params(distribution_span, 1, mode_centrality, decay_velocity)
probability_scaling_factor = number_of_vesicles * beta_dist.cdf(expected_duration)
plt.figure()
plot_dist(beta_dist)
plt.vlines(spans, 0, .6)

time_offsets = {'3': [], '6': [], '9': []}
num_released = {'3': [], '6': [], '9': []}
for span in spans:
    possible_release_times, release_times_probabilities = _make_bins(beta_dist, span)
    random_values = np.random.rand(number_of_spikes, possible_release_times.shape[0])
    release_inds = (random_values / probability_scaling_factor) <= release_times_probabilities
    vesicle_time_offsets = np.array([possible_release_times[inds] for inds in release_inds])
    num_released[span] = np.array([spike_released.size for spike_released in vesicle_time_offsets])
    time_offsets[span] = np.hstack(vesicle_time_offsets)

plt.figure()
sns.distplot(time_offsets[3], norm_hist=False, kde=False)
sns.distplot(time_offsets [6], norm_hist=False, kde=False)
sns.distplot(time_offsets[9], norm_hist=False, kde=False)


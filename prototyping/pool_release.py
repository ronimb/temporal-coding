import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns;
from scipy.stats import beta
import sympy as sym
from multiprocessing import Pool
from time import time
from tools import sec_to_time

sns.set()


# %%
def nu_beta_params(beta_mode: float, beta_span: float = 15,
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


def nu_beta_bins(release_duration, beta_dist,
                 dt_ms=1e-2, time_bins=[]):
    """
    This function is used to generate the possible vesicle release times, along with
    the probablities of release for each possible release time
    """

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

    return probabilities, time_bins


# %% Prototype of production versions


def pool_release_p_decay(neuron, stimulus_duration, release_duration,
                         max_pool_size, starting_pool_size, tau_n,
                         d_s, max_p , init_p,
                         dt_ms=0.01, eps=1e-4,
                         debug=False):
    """

    :param neuron:
    :param stimulus_duration:
    :param max_pool_size:
    :param init_n:
    :param tau_n:
    :param max_p:
    :param init_p:
    :param tau_p:
    :param dt_ms:
    :return:
    """
    times = np.arange(0 + dt_ms, stimulus_duration + dt_ms, dt_ms)
    tau_p = d_s / (np.log(max_p) - np.log(eps))

    n_t = np.zeros_like(times)  # Number of vesicles in the pool at each time instant
    p_t = np.zeros_like(times)  # Independent release probability for vesicles at each time instant
    r_t = np.zeros_like(times)  # Number of vesicles released at each time instant

    # Setting initial values for n and p
    n_t[0] = starting_pool_size
    p_t[0] = init_p

    # Eulerian integration over all time points using simple loop structure
    # print(times[:-1].shape)
    time_last_spike = 0
    for i, t in enumerate(times[:-1]):
        # Determine when current replenishment process started
        replenished = (1 / tau_n) * dt_ms * (1 - np.isclose(n_t[i], max_pool_size))
        # Calculate number of vesicles release
        released = r_t[i] = (np.random.rand(int(np.round(n_t[i], 5))) < (p_t[i] * dt_ms)).sum()  # Flip coin for each vesicle currently in pool

        # Calculate change in vesicle pool size
        dn = replenished - released

        # Find last spike
        is_spiking = (t == neuron).any().astype(int)
        if is_spiking:
            time_last_spike = t
        # Calculate change in independent release probability
        p_decay = -((p_t[i]) / tau_p)  # Exponential decay of probability
        p_boost = (max_p - p_t[i]) * is_spiking  # Increase to max_p if spike just occured
        # p_boost = (1 - p_t[i]) * 
        dp = p_decay + p_boost

        # Calculate p and n at next time step
        n_t[i + 1] = n_t[i] + dn
        if (t - time_last_spike) > release_duration:
            p_t[i + 1] = 0
        else:
            p_t[i + 1] = np.clip(p_t[i] + dp, 0, 1)  # Probability can't exceed 1

    if debug:
        df = pd.DataFrame(dict(
            n=n_t,
            p=p_t,
            released=r_t,
        ),
            index=times
        )
        df['spikes'] = 0
        df.loc[neuron, 'spikes'] = 1
        return r_t, df
    else:
        return r_t


def pool_release_p_beta(neuron, stimulus_duration, release_duration,
                        starting_pool_size, max_pool_size, max_fraction_released, tau_n,
                        time_to_max_probability, longest_release_duration=9,
                        dt_ms=0.01, debug=False, **kwargs):
    # Preparing keyword arguments for passing to beta generating functions

    # Initializing variables
    # Todo: investigate time binning consistency
    time_bins = np.arange(0 + dt_ms, stimulus_duration + dt_ms, dt_ms)  # Vector of time bins

    n_t = np.zeros_like(time_bins)  # Number of vesicles in the pool at each time instant
    p_t = np.zeros_like(time_bins)  # Independent release probability for vesicles at each time instant
    r_t = np.zeros_like(time_bins)  # Number of vesicles released at each time instant

    # Setting initial values for n and
    n_t[0] = starting_pool_size

    # Generating the required beta function
    beta_dist = nu_beta_params(beta_mode=time_to_max_probability, **kwargs)
    # Deriving initial probabilities of release such that 1 vesicle is released at the longest_release_duration
    initial_probabilities, _ = nu_beta_bins(release_duration, beta_dist, dt_ms=dt_ms, time_bins=time_bins)
    # Scaling release probabilities such that max_pool_size vesicles are released instead
    scaling_factor = max_pool_size * beta_dist.cdf(
        longest_release_duration)  # Scaled so as to give max_pool_size vesicles at longest_release duration
    initial_probabilities *= scaling_factor  # Referred to as "P" in the equations

    # The above probablities essentially give us at each time instant, the chance that EXACTLY one vesicle will be released,
    # Thus, assuming independence of release, we can arrive at the independent probability of release for single vesicles
    p_, P_, n_ = sym.symbols(['p', 'P', 'n'], positive=True, real=True)

    def _derive_independent_probability(scaled_probs):
        num_ves_when_full = np.round(max_pool_size * max_fraction_released).astype(int)
        actual_probs = np.zeros_like(scaled_probs)
        for i, P in enumerate(scaled_probs):
            sol = sym.solve([p_ >= 0, p_ <= 1,
                             num_ves_when_full * (p_ * (1 - p_) ** (num_ves_when_full - 1)) - P],
                            p_, quick=True)
            actual_probs[i] = sol.args[0].args[1]
        return actual_probs

    actual_probs = _derive_independent_probability(initial_probabilities)
    num_release_bins = actual_probs.shape[0]
    # Generate independent release probability "p" for entire experiment
    for spike in neuron:
        start_ind = np.where(np.isclose(time_bins, spike))[0][0]
        end_ind = start_ind + actual_probs.shape[0]
        if end_ind > time_bins.shape[0]:
            end_ind = time_bins.shape[0]
            p_t[start_ind:] = actual_probs[:end_ind - start_ind]
        else:
            p_t[start_ind: end_ind] = actual_probs

    # Eulerian integration over all time points using simple loop structure
    # print(times[:-1].shape)
    for i, t in enumerate(time_bins[:-1]):
        replenished = (1 / tau_n) * dt_ms * (1 - np.isclose(n_t[i], max_pool_size))
        # Calculate number of vesicles release
        released = r_t[i] = (np.random.rand(int(np.round(n_t[i], 5))) < p_t[
            i]).sum()  # Flip coin for each vesicle currently in pool
        # Calculate change in vesicle pool size
        dn = replenished - released

        # Calculate p and n at next time step
        n_t[i + 1] = n_t[i] + dn

    if debug:
        df = pd.DataFrame(dict(
            n=n_t,
            p=p_t,
            released=r_t,
        ),
            index=time_bins
        )
        df['spikes'] = 0
        df.loc[neuron, 'spikes'] = 1
        return r_t, df
    else:
        return r_t


def plot_process(process_df, fig=None):
    if not fig:
        fig = plt.figure()
    grid_shape = (10, 1)
    tall_span = 3

    time = process_df.index
    spike_times = time[process_df.spikes.astype(bool)]
    ves_times = time[(process_df.released != 0).astype(bool)]

    ax_p = plt.subplot2grid(shape=grid_shape, loc=(0, 0), rowspan=tall_span, fig=fig)
    ax_p.plot(time, process_df.p)
    ax_p.vlines(spike_times, *ax_p.get_ylim(), linestyles='dotted', label='spikes')
    ax_p.set_title('Independent release probability')
    ax_p.set_xticklabels([])
    ax_p.set_ylabel('Release %P')

    ax_n = plt.subplot2grid(shape=grid_shape, loc=(3, 0), rowspan=tall_span, fig=fig)
    ax_n.plot(time, process_df.n.round(5).astype(int))
    ax_n.vlines(spike_times, *ax_n.get_ylim(), linestyles='dotted')
    ax_n.set_title('Pool contents')
    ax_n.set_xticklabels([])
    ax_n.set_ylabel('# Of vesicles in pool')

    ax_r = plt.subplot2grid(shape=grid_shape, loc=(6, 0), rowspan=tall_span, fig=fig)
    ax_r.plot(time, process_df.released)
    ax_r.vlines(spike_times, *ax_r.get_ylim(), linestyles='dotted')
    ax_r.set_title('Vesicles released')
    ax_r.set_ylabel('# Of vesicles released')
    ax_r.set_xticklabels([])

    ax_spikes = plt.subplot2grid(shape=grid_shape, loc=(9, 0), rowspan=1, fig=fig)
    ax_spikes.scatter(spike_times, [1] * spike_times.shape[0], marker='|', c='black', s=300, alpha=0.75)
    ax_spikes.scatter(ves_times, [1] * ves_times.shape[0], marker='.', c='blue', s=36, alpha=0.375, linewidths=0)
    ax_spikes.set_yticks([])
    ax_spikes.set_xlabel('Time (ms)')
    ax_spikes.set_xlim(*ax_r.get_xlim())
    ax_spikes.set_ylim(0.75, 1.25)
    ax_spikes.set_title('Spike and Release times')


# %% Test function definition
def vesicle_release_distribution(process_df):
    all_release_durations = process_df.release_duration.unique()
    results_df = {'num_released': [],
                  'offset': [],
                  'release_duration': []}
    for release_duration in all_release_durations:
        # Isolate the data from a single experiment (only one per release duration)
        release_duration_bool_ind = (process_df.release_duration == release_duration)
        release_duration_data = process_df[release_duration_bool_ind]
        # Get variables to work with on this experiment
        time = np.r_[release_duration_data.time]
        spikes_bool_ind = release_duration_data.spikes.astype(bool)
        spike_times = time[spikes_bool_ind]
        release_bool_ind = (release_duration_data.released != 0).astype(bool)
        ves_times = time[release_bool_ind]

        spike = spike_times[0]
        good_spikes = []
        while (spike + release_duration) <= time.max():
            good_spikes.append(spike)
            distances_of_next = spike_times - spike
            meet_criteria = spike_times[distances_of_next > release_duration]
            if meet_criteria.size > 0:
                spike = meet_criteria[0]  # Taking the first spike that matches as our next spike
            else:
                spike = time[-1]  # Terminating

        offsets = []
        for spike in good_spikes:
            ves_temporal_distance = ves_times - spike
            relevant_release_inds = (ves_temporal_distance > 0) & (ves_temporal_distance <= release_duration)

            # relevant_release_times = ves_times[relevant_release_inds]
            relevant_release_offsets = ves_temporal_distance[relevant_release_inds]
            offsets.append(relevant_release_offsets)
        n_released = np.array([o.size for o in offsets])

        for n, o in zip(n_released, offsets):
            results_df['release_duration'].extend([release_duration] * n)
            results_df['num_released'].extend([n] * n)
            results_df['offset'].extend(o)
    results_df = pd.DataFrame(results_df)

    return results_df


def plot_distribution(process_df, fig=None):
    results_df = vesicle_release_distribution(process_df)
    release_durations = np.unique(process_df.release_duration)
    time = np.unique(process_df.time)
    if not fig:
        fig = plt.figure()
    grid_shape = (8, 1)
    totals = [results_df[results_df.release_duration == r].shape[0]
              for r in np.unique(process_df.release_duration)]
    ax_1 = plt.subplot2grid(shape=grid_shape, loc=(0, 0), rowspan=4, fig=fig)
    sns.swarmplot(x='release_duration', y='offset', data=results_df, ax=ax_1)
    sns.boxenplot(x='release_duration', y='offset', data=results_df, ax=ax_1)
    # ax_1.set_xticks([])
    ax_1.xaxis.tick_top()
    ax_1.xaxis.set_label_position('top')
    ax_1.xaxis.set_ticklabels([*totals])
    ax_1.xaxis.set_tick_params(length=0)
    ax_1.set_xlabel('Total released')
    ax_1.set_title('Release offset distribution')

    ax_2 = plt.subplot2grid(shape=grid_shape, loc=(4, 0), rowspan=4, fig=fig)
    sns.boxplot(x='release_duration', y='num_released',
                data=results_df, ax=ax_2)
    sns.pointplot(x='release_duration', y='num_released',
                  data=results_df, ax=ax_2, color='black')
    ax_2.set_xlabel('Release Duration')
    ax_2.set_ylabel('# of vesicles released')
    ax_2.set_title('# Vesicles per spike distribution')
    #
    # for rel_dur in release_durations:
    #     ax_raster = plt.subplot2grid(shape=grid_shape, loc=(8, 0), rowspan=1, fig=fig)
    #     current_data = process_df[process_df.release_duration == rel_dur]
    #     spike_times = time[current_data.spikes.astype(bool)]
    #     ves_inds = (current_data.released != 0).astype(bool)
    #     ves_times = time[ves_inds]
    #     n_ves = current_data.released[ves_inds]
    #     max_released = n_ves.max()
    #     ax_raster.scatter(spike_times, [1] * spike_times.shape[0], marker='|', c='black', s=300, alpha=0.75)
    #
    #     d=0.1
    #     for i in range(max_released):
    #         n, m = divmod(i+1, 2)
    #         sign = -1 if (m == 0) else 1
    #         ax_raster.scatter(ves_times, [1 + sign * n * d] * ves_times.shape[0], marker='.', c='blue', s=36, alpha=0.375, linewidths=0)
    # ax_raster.set_xlabel('time')
    # ax_raster.set_yticks([])



# %% Mocking up test neuron
stimulus_duration = 5000
dt_ms = 0.01
bin_times = np.arange(0 + dt_ms, stimulus_duration + dt_ms, dt_ms)

freq = 15
neuron = np.random.rand(bin_times.shape[0]) <= (freq / (stimulus_duration) * dt_ms / (1000 / stimulus_duration))
neuron = bin_times[neuron]

# %% Testing beta derived p
stimulus_duration = stimulus_duration
release_durations = [3, 6, 9]
starting_pool_size = 20
tau_n = 30
max_pool_size = 20
max_fraction_released = 0.75
time_to_max_probability = 1
longest_release_duration = 9
dt_ms = 0.01

all_beta = {r: [] for r in release_durations}
for r in release_durations:
    print(f'Working on release_duration={r}', end='', flush=True)
    started = time()
    # params
    params_beta = dict(
        neuron=neuron,
        stimulus_duration=stimulus_duration,
        release_duration=r,
        tau_n=tau_n,
        starting_pool_size=starting_pool_size,
        max_pool_size=max_pool_size,
        max_fraction_released=max_fraction_released,
        time_to_max_probability=time_to_max_probability,
        longest_release_duration=longest_release_duration,
        dt_ms=dt_ms,
    )
    all_beta[r] = pool_release_p_beta(**params_beta, debug=True)[1]

    # fig1 = plt.figure()
    # plot_process(all_beta[r], fig=fig1)
    # fig1.suptitle(f'beta, release_duration={r}')
    print(f'    --- Done! took {sec_to_time(time() - started)}')

all_beta = pd.concat(all_beta)
all_beta.index.rename(['release_duration', 'time'], inplace=True)
all_beta.reset_index(inplace=True)
plot_distribution(all_beta)
# %% Testing decaying p
stimulus_duration = stimulus_duration
release_durations = [3, 6, 9]
max_pool_size = 20
starting_pool_size = max_pool_size
tau_n = 15
d_s = 45
max_p = 0.86
init_p = 0
dt_ms = 0.01
eps = 1e-4

all_decay = {r: [] for r in release_durations}
for r in release_durations:
    print(f'Working on release_duration={r}', end='', flush=True)
    # params
    params_decay = dict(
        neuron=neuron,
        stimulus_duration=stimulus_duration,
        release_duration=r,
        max_pool_size=max_pool_size,
        starting_pool_size=starting_pool_size,
        tau_n=tau_n,
        d_s=d_s,
        max_p=max_p,
        init_p=init_p,
        dt_ms=dt_ms,
        eps=eps
    )

    all_decay[r] = pool_release_p_decay(**params_decay, debug=True)[1]
    # fig2 = plt.figure()
    # plot_process(all_decay[r], fig=fig2)
    # fig2.suptitle(f'decay, release_duration={r}')
    print(f'    --- Done! took {sec_to_time(time() - started)}')
all_decay = pd.concat(all_decay)
all_decay.index.rename(['release_duration', 'time'], inplace=True)
all_decay.reset_index(inplace=True)
plot_distribution(all_decay)

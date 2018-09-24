import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import ms, Hz
from numba import jit, prange
from multiprocessing import Pool
from scipy.stats import beta
import pyspike as spk

b2.BrianLogger.suppress_name('method_choice')


# %%


def brian_poisson(rate, duration_ms, dt=1 * ms, n=1):
    """
    Generates a sample of poisson neurons with a specified average frequency.
    Returns the sample as a list of neurons

    :param rate: The average firing frequency of each neuron in the sample
    :param duration_ms: Length of a single "trial" in ms
    :param dt: The shortest time interval between spikes in the sample
    :param n: Number of neurons in the sample


    NOTE: the "rate" and "duration_ms" parameters can also vary between neurons,
          if this is the desired behaviour, they should be given as a list of length "n"
          where each entry corresponds to the parameters of a single neuron in the sample

    :return: List of neurons in the sample stimulus, in each neuron, firing times are specified in milliseconds
    """
    q = b2.units.fundamentalunits.Quantity  # Type used to represent units in brian

    # Check that all parameters comply with units expected by brian
    if not isinstance(rate, q):  # Rate is expected to be given in Hz
        if np.isscalar(rate):
            rate = rate * Hz
        else:
            rate = np.array(rate) * Hz

    if not isinstance(duration_ms, q):  # Duration is expected to be specified in ms
        if np.isscalar(duration_ms):
            duration_ms = duration_ms * ms
        else:
            duration_ms = np.array(duration_ms) * ms

    # Specify properties of brian neuron group
    neuron = b2.NeuronGroup(n, "rate : Hz", threshold='rand()<rate*dt', dt=dt)
    neuron.rate = rate
    spikes = b2.SpikeMonitor(neuron, record=True)  # Set up monitoring (required for gathering spike timing information)
    b2.run(duration_ms)  # Run simulation with the specified parameters
    # Extract spike timings from brian objects
    if n == 1:
        trains = spikes.spike_trains()[0] / dt
    else:
        trains = [train / dt for train in spikes.spike_trains().values()]
    return trains


def make_spk(rate, duration_ms, n=1, dt=1 * ms, exact_freq=True):
    """
    Creates a stimulus with n neurons of a desired frequency.
    This function returns each neuron as a PySpike object which can be used
    for evaluating distance between neurons, and uses brian internally for sample generation.

    :param rate: Desired rate of fire for neurons in the sample
    :param duration_ms: Length of a "trial" in milliseconds
    :param n: Number of neurons in the sample
    :param dt: The shortest time interval between spikes in the sample
    :param exact_freq: Whether frequency should be Exactly "rate" for each neuron, or "rate" on average.
    :return:
    """

    # Making sure the variable used for rate comparison is of the correct type
    if type(rate) == b2.units.fundamentalunits.Quantity:
        comp_rate = rate / Hz
    else:
        comp_rate = rate

    # Generating the spike-trains using brian
    trains = brian_poisson(rate, duration_ms, n=n, dt=dt)

    # Handling single or multiple train generation as well as exact or average frequency
    if n > 1:
        if exact_freq:
            trains = np.array(trains)
            actual_rates = [tr.size for tr in trains]
            # ToDO: Handle the actual rates considering the duration of the stimulus
            rate_match = np.equal(actual_rates, comp_rate)
            trains = trains[rate_match]
            real_N = trains.size
            while real_N < n:
                new_trains = np.array(brian_poisson(rate, duration_ms, n=n, dt=dt))
                new_rates = [tr.size for tr in new_trains]
                rate_match = np.equal(new_rates, comp_rate)
                new_trains = new_trains[rate_match]
                trains = np.append(trains, new_trains)
                real_N = trains.size
        tr_spk = [spk.SpikeTrain(train, duration_ms) for train in trains[0:n]]
    else:
        rate_match = trains.size == comp_rate
        while not (rate_match):
            trains = brian_poisson(rate, duration_ms, n=n, dt=dt)
            rate_match = trains.size == comp_rate
        tr_spk = spk.SpikeTrain(trains, duration_ms)
    return tr_spk


# OLD CONVERSION FUNCTION, NO LONGER USED
def convert_sample(sample):
    inds = []
    times = []
    counts = []
    for i, neuron in enumerate(sample):
        # neuron = np.round(neuron, 1)
        neuron = np.trunc(neuron * 10) / 10
        time, count = np.unique(neuron, return_counts=True)
        inds.extend([i] * time.shape[0])
        times.extend(time)
        counts.extend(count)
    return np.array([inds, times, counts])


def gen_with_temporal_shift(rate, num_neur, duration_ms=500, shift_size=5,
                            set1_size=100, set2_size=100):
    """

    :param rate:
    :param num_neur:
    :param duration_ms:
    :param shift_size:
    :param set1_size:
    :param set2_size:

    :return:
    """

    def shift_spikes(source, shift_size, num_samples):
        """

        :param source:
        :param shift_size:
        :param num_samples:
        :return:
        """
        samples = []
        for _ in range(num_samples):
            curr_sample = []
            for train in source:
                vals = np.random.randint(shift_size + 1, size=train.size)
                new_tr = train + vals
                edge = np.greater_equal(train + shift_size, duration_ms)
                if edge.any():
                    edge_vals = train[edge]
                    deltas = duration_ms - edge_vals
                    for i in range(np.size(edge_vals)):
                        edge_vals[i] += np.random.randint(deltas[i])
                    new_tr[edge] = edge_vals
                curr_sample.append(np.sort(np.unique(new_tr)))
            samples.append(curr_sample)
        return samples

    source_1 = brian_poisson(rate, duration_ms, n=num_neur)
    source_2 = brian_poisson(rate, duration_ms, n=num_neur)

    samples_1 = shift_spikes(source_1, shift_size, set1_size)
    samples_2 = shift_spikes(source_2, shift_size, set2_size)

    combined = np.concatenate([samples_1, samples_2])
    labels = np.concatenate([np.zeros(set1_size), np.ones(set2_size)])

    rand_inds = np.random.choice(np.arange(labels.size), labels.size, replace=False)

    labels = labels[rand_inds]
    combined = combined[rand_inds]

    return {'data': combined, 'labels': labels}


# %% Transformer class
class vesicle_beta_releaser:
    # These aree a set of parameters ive found to be suitable for our case
    _beta_defaults = dict(
        beta_mode=1,
        beta_span=15,
        mode_centrality=3,  # How "tight" the customized distribution is around the mode
        declination=2  # How fast the values decline past the mode
    )

    def __init__(self, mode='fixed', beta_params=_beta_defaults, release_duration_ms=6, num_ves=20,
                 signal_duration_ms=500, set_sizes=100):
        if not isinstance(beta_params, dict):
            raise TypeError(f'''
            beta_params must be a dictionary with any, some, or all of the following fields:
             {self._beta_defaults.keys()}''')
        # This makes sure all of the required parameters are set, even if only some are passed
        if beta_params is not self._beta_defaults:
            for parameter_name, parameter_value in self._beta_defaults.items():
                beta_params.setdefault(parameter_name, parameter_value)

        if mode == 'fixed':
            beta_params['beta_span'] = release_duration_ms

        self.beta_params = beta_params
        print(self.beta_params)

    def __str__(self):
        return 'T'

    def __repr__(self):
        return 'T'

    def fixed_rel(self):
        pass

    def stochastic_rel(self):
        def _make_bins():
            pass

    def transform_set(self):
        pass

    def transform_sample(self):
        pass


t = vesicle_beta_releaser()


# %%
def gen_with_vesicle_release(rate, num_neur, duration_ms=500,
                             release_duration=5, num_ves=20,
                             beta_params=dict(
                                 dist_mode=1,
                                 dist_span=15,  # This parameter is irrelevant for fixed release mode
                                 base_span=3,
                                 base_a=2),
                             set1_size=100, set2_size=100,
                             source_stims=[], mode='fixed'):
    beta_params.setdefault('base_span', 3)
    beta_params.setdefault('base_a', 2)
    if mode == 'fixed':
        beta_params.setdefault('dist_span', release_duration)
    elif mode == 'non_fixed':
        beta_params.setdefault('dist_span', 15)

    def release_fun(beta_params, release_duration, num_ves):
        # The base_span is the span for which the a parameter of the beta distribution equals base_a
        # Continue adding support for new vesicle release mode
        span = release_duration
        mode = beta_params['dist_mode']
        base_a = beta_params['base_a']
        base_span = beta_params['base_span']

        a = 1 + (base_span / span) * (base_a - 1)
        b = (span * (a - 1) - mode * (a - 2)) / mode
        return np.random.beta(a, b, num_ves) * span

    def spikes_to_ves(sample, beta_params, release_duration, num_ves=20):
        # ToDO: Continue adding support for new ves release mode
        ves_array = []
        for train in sample:
            num_spikes = train.shape[0]
            ves_offsets = release_fun(beta_params, (num_ves, num_spikes))
            ves_times = (train + ves_offsets).flatten()
            ves_times = ves_times[ves_times < duration_ms]
            ves_times.sort()
            ves_array.append(np.unique(ves_times))
        return np.array(ves_array)

    def make_bins(beta_params, release_duration, num_ves, dt_ms=0.01, max_span=9):
        # TODO: Make this a method of the future generator object and find a way to hand max_span and dt settings
        span = beta_params['dist_span']
        mode = beta_params['dist_mode']
        base_a = beta_params['base_a']
        base_span = beta_params['base_span']

        beta_a = 1 + (base_span / span) * (base_a - 1)
        beta_b = (span * (beta_a - 1) - mode * (beta_a - 2)) / mode
        beta_dist = beta(a=beta_a, b=beta_b, scale=span)

        multiplication_factor = num_ves / beta_dist.cdf(max_span)

        bin_borders = np.arange(0, max_span + dt_ms, dt_ms)
        bin_centers = bin_borders[:-1] + dt_ms / 2

        bin_probs = beta_dist.cdf(bin_borders[1:]) - beta_dist.cdf(bin_borders[:-1])

        release_duration_inds = bin_centers <= release_duration

        return bin_centers[release_duration_inds], bin_probs[release_duration_inds]

    def make_samples(source, num_samples, beta_params, release_duration, num_ves=20):

        if mode == 'non_fixed':
            bin_centers, bin_probs = make_bins(beta_params, release_duration, num_ves)

        samples = []
        for _ in range(num_samples):
            samples.append(spikes_to_ves(source, beta_params, num_ves))
        return samples

    if source_stims:
        source_1 = [train.get_spikes_non_empty() for train in source_stims[0]]
        source_2 = [train.get_spikes_non_empty() for train in source_stims[1]]
    else:
        source_1 = brian_poisson(rate, duration_ms, n=num_neur)
        source_2 = brian_poisson(rate, duration_ms, n=num_neur)

    samples_1 = make_samples(source_1, set1_size, beta_params, release_duration, num_ves)
    samples_2 = make_samples(source_2, set2_size, beta_params, release_duration, num_ves)

    combined = np.concatenate([samples_1, samples_2])
    labels = np.concatenate([np.zeros(set1_size), np.ones(set2_size)])

    rand_inds = np.random.choice(np.arange(labels.size), labels.size, replace=False)

    labels = labels[rand_inds]
    combined = combined[rand_inds]

    return {'data': combined, 'labels': labels}


def plot_sample(sample):
    for i, train in enumerate(sample):
        n_spikes = train.shape[0]
        ax = plt.scatter(train, n_spikes * [i], marker='|', c='blue')
        plt.yticks([])
    return ax


@jit(parallel=True)
def convert_single_sample(sample):
    inds = []
    times = []
    counts = []
    num_events = []
    for i in prange(sample.shape[0]):
        neuron = np.trunc(sample[i] * 10) / 10
        time, count = np.unique(neuron, return_counts=True)
        num_events.append(time.shape[0])
        inds.extend([i] * time.shape[0])
        times.extend(time)
        counts.extend(count)
    return np.array([inds, times, counts]), num_events


def convert_multi_samples(samples):
    num_neurons = samples['data'][0].shape[0]
    p = Pool(12)
    labels = samples['labels']
    res = p.map(convert_single_sample, samples['data'])
    ts = np.hstack([[x[0][0] + num_neurons * i, x[0][1], x[0][2]] for i, x in enumerate(res)])
    p.close()
    p.join()
    converted_samples = np.zeros(ts.shape[1],
                                 dtype={'names': ('inds', 'times', 'counts'),
                                        'formats': (int, float, int)})
    converted_samples['inds'] = ts[0]
    converted_samples['times'] = ts[1]
    converted_samples['counts'] = ts[2]
    return converted_samples, labels


def multi_shift(source, shifts, n, frac_shifts=[1],
                increase=False, jitter=False):
    ### FRAC_FIRE IS CURRENTLY IGNORED!!!!, implemented in shift_fwd ###
    if not (isinstance(frac_shifts, np.ndarray)):
        frac_shifts = np.array(frac_shifts)

    if not (isinstance(shifts, np.ndarray)):
        shifts = np.array(shifts)

    duration = source.t_end

    def shift_main(src):

        spiketimes = src.spikes[:, np.newaxis]

        num_shifts = len(shifts)
        num_fracshift = len(frac_shifts)
        num_spikes = spiketimes.size

        # Finding edges
        left_edge = np.where(spiketimes < shifts)
        right_edge = np.where((spiketimes + shifts) > duration)

        # Create the shift matrices
        shifts_bottom = np.tile(shifts,
                                reps=(num_spikes, n, num_fracshift, 1))
        shifts_top = shifts_bottom.copy()

        shifts_bottom[left_edge[0], :, :, left_edge[1]] = \
            spiketimes[left_edge[0]][:, np.newaxis]
        shifts_top[right_edge[0], :, :, right_edge[1]] = \
            ((duration) - spiketimes[right_edge[0]][:, np.newaxis])

        shifts_range = shifts_top + shifts_bottom

        # boolean matrix for fractional shifting - MAY REQUIRE CHANGING AFTER FRAC_FIRE IS INTEGRATED
        bool_frac_mat = np.random.rand(num_spikes, n, num_fracshift, num_shifts)
        bool_frac_mat = bool_frac_mat <= frac_shifts[:, np.newaxis]

        # Draw!
        shiftvals = np.random.rand(num_spikes, n, num_fracshift, num_shifts)

        # Fix edges!
        shiftvals = ((shiftvals * shifts_range) - shifts_bottom) * bool_frac_mat
        shiftvals.round(out=shiftvals)

        # Shift!
        shifted = shiftvals + spiketimes[:, np.newaxis, np.newaxis]
        shifted.sort()

        # Find uniques!
        uniques = [np.unique(shifted[:, i, j, k])
                   for i in range(n)
                   for j in range(num_fracshift)
                   for k in range(num_shifts)]

        # pyspike!
        shifted_spk = [spk.SpikeTrain(shifted_times, duration)
                       for shifted_times in uniques]
        return shifted_spk

    if increase:
        slots = np.setdiff1d(np.arange(source.t_start, source.t_end), source.spikes, assume_unique=True)
        sources = {0: source}
        for inc in increase:
            new_spikes = np.random.choice(slots, inc)
            sources[inc] = spk.SpikeTrain(np.sort(np.append(source.spikes, new_spikes)), duration)

        shifted = []
        for src in sources.values():
            shifted.extend(shift_main(src))

    else:
        shifted = shift_main(source)

    return shifted


def find_match_dists(sample_dists, desired_dist, eps=1e-3):
    absdiff = np.abs(sample_dists - desired_dist)

    ind = np.argmin(absdiff)

    meet_criteria = absdiff[ind] <= eps

    if meet_criteria:
        res = ind
    else:
        res = None

    return res


def calc_distance(samples, target=[], metric='distance'):
    if metric == 'distance':
        metric_fun = spk.spike_distance
        metric_matrix_fun = spk.spike_distance_matrix
    elif metric == 'isi':
        metric_fun = spk.isi_distance
        metric_matrix_fun = spk.isi_distance_matrix

    if not (target):  # In this case, create self distance matrix within set
        distance_matrix = metric_matrix_fun(samples)
        distance_vector = distance_matrix[np.triu_indices_from
        (distance_matrix, 1)]
        distance = {'matrix': distance_matrix,
                    'vector': distance_vector}

    elif isinstance(target,
                    spk.SpikeTrain):  # In this case, calculate distances between all trains and a reference train
        distance_vector = np.array(
            [metric_fun(tr, target) for tr in samples])
        distance = {'vector': distance_vector}

    else:  # In this case, calculate pairwise distances between different train sets
        n1 = len(samples)
        n2 = len(target)
        distance_matrix = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                distance_matrix[i, j] = metric_fun(samples[i], target[j])
        distance_vector = distance_matrix.flatten()
        distance = {'matrix': distance_matrix,
                    'vector': distance_vector}
    return distance


# %%
if __name__ == '__main__':
    rate = 50
    duration = 500
    num_neur = 10
    set_size = 100
    # data = gen_with_temporal_shift(rate, duration,
    #                                num_neur, set_size, set_size)
    beta_params = dict(span=5, mode=1)
    data = gen_with_vesicle_release(rate, num_neur, duration,
                                    beta_params=beta_params, num_ves=20,
                                    set1_size=set_size, set2_size=set_size)

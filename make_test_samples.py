import brian2 as b2
b2.BrianLogger.suppress_name('method_choice')
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import ms, Hz, second


def brian_poisson(rate, duration_ms, dt=1 * ms, n=1):
    """

    :param rate:
    :param duration_ms:
    :param dt:
    :param n:
    :return:
    """
    q = b2.units.fundamentalunits.Quantity
    if not isinstance(rate, q):
        if np.isscalar(rate):
            rate = rate * Hz
        else:
            rate = np.array(rate) * Hz
    if not isinstance(duration_ms, q):
        if np.isscalar(duration_ms):
            duration_ms = duration_ms * ms
        else:
            duration_ms = np.array(duration_ms) * ms

    neuron = b2.NeuronGroup(n, "rate : Hz", threshold='rand()<rate*dt', dt=dt)
    neuron.rate = rate
    spikes = b2.SpikeMonitor(neuron, record=True)
    b2.run(duration_ms)
    if n == 1:
        trains = spikes.spike_trains()[0] / dt
    else:
        trains = [train / dt for train in spikes.spike_trains().values()]
    return trains

def gen_with_temporal_shift(rate, num_neur, duration_ms=500, shift_size=5,
                            set1_size=100, set2_size=100):
    """

    :param num_neur:
    :param rate:
    :param duration_sec: Length in seconds of each spike train
    :param set1_size:
    :param set2_size:
    :return:
    """
    # TODO: move unit checking to brian_poisson, keep duration ms somehow

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

def gen_with_vesicle_release(rate, num_neur, duration_ms,
                             span, mode, num_ves,
                             set1_size=100, set2_size=100):

    def release_fun(span, mode, num_ves, base_a=2, base_span=3):
        # The base_span is the span for which the a parameter of the beta distribution equals base_a
        a = 1 + (base_span / span) * (base_a - 1)
        b = (span * (a - 1) - mode * (a - 2)) / mode
        return np.random.beta(a, b, num_ves) * span

    def spikes_to_ves(sample, span, mode=1, num_ves=20):
        ves_array = []
        for train in sample:
            num_spikes = train.shape[0]
            ves_offsets = release_fun(span, mode, (num_ves, num_spikes))
            ves_times = (train + ves_offsets).flatten()
            ves_times = ves_times[ves_times < duration_ms]
            ves_times.sort()
            ves_array.append(np.unique(ves_times))
        return np.array(ves_array)

    def make_samples(source, num_samples, span, mode=1, num_ves=20):
        samples = []
        for _ in range(num_samples):
            samples.append(spikes_to_ves(source, span, mode, num_ves))

    source_1 = brian_poisson(rate, duration_ms, n=num_neur)
    source_2 = brian_poisson(rate, duration_ms, n=num_neur)

    samples_1 = make_samples(source_1, set1_size, span, mode, num_ves)
    samples_2 = make_samples(source_2, set2_size, span, mode, num_ves)

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
# %%
if __name__ == '__main__':
    rate = 50
    duration = 500
    num_neur = 10
    set_size = 100
    data = gen_with_temporal_shift(rate, duration,
                                   num_neur, set_size, set_size)
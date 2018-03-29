import brian2 as b2
import numpy as np
from brian2.units import ms, Hz, second


def make_test_samples(rate, duration_sec,
                      num_neur=10, shift_size=5,
                      set1_size=500, set2_size=500):
    """

    :param num_neur:
    :param rate:
    :param duration_sec: Length in seconds of each spike train
    :param set1_size:
    :param set2_size:
    :return:
    """
    q = b2.units.fundamentalunits.Quantity
    if not isinstance(rate, q):
        if np.isscalar(rate):
            rate = rate * Hz
        else:
            rate = np.array(rate) * Hz
    if not isinstance(duration_sec, q):
        if np.isscalar(duration_sec):
            duration_sec = duration_sec * second
        else:
            duration_sec = np.array(duration_sec) * second
    duration_ms = duration_sec / ms

    def brian_poisson(rate, duration, dt=1 * ms, n=1):
        """

        :param rate:
        :param duration:
        :param dt:
        :param n:
        :return:
        """

        neuron = b2.NeuronGroup(n, "rate : Hz", threshold='rand()<rate*dt', dt=dt)
        neuron.rate = rate
        spikes = b2.SpikeMonitor(neuron, record=True)
        b2.run(duration)
        if n == 1:
            trains = spikes.spike_trains()[0] / dt
        else:
            trains = [train / dt for train in spikes.spike_trains().values()]
        return trains

    def shiftset(tr_set, shift_size, num_shifted):
        """

        :param tr_set:
        :param shift_size:
        :param num_shifted:
        :return:
        """
        all_shifted = []
        for _ in range(num_shifted):
            curr_shifted = []
            for tr in tr_set:
                vals = np.random.randint(shift_size + 1, size=tr.size)
                new_tr = tr + vals
                edge = np.greater_equal(tr + shift_size, duration_ms)
                if edge.any():
                    edge_vals = tr[edge]
                    deltas = duration_ms - edge_vals
                    for i in range(np.size(edge_vals)):
                        edge_vals[i] += np.random.randint(deltas[i])
                    new_tr[edge] = edge_vals
                curr_shifted.append(new_tr)
            all_shifted.append(curr_shifted)
        return all_shifted

    def booleanize(tr_set):
        """

        :param tr_set:
        :return:
        """
        booled = []
        tr_len = int(duration_ms)
        for train in tr_set:
            curr_tr = np.zeros(tr_len)
            curr_tr[train.astype(int)] = 1
            booled.append(curr_tr)
        return np.array(booled)

    samp1 = brian_poisson(rate, duration_sec, n=num_neur)
    samp2 = brian_poisson(rate, duration_sec, n=num_neur)

    shifted_1 = shiftset(samp1, shift_size, set1_size)
    shifted_2 = shiftset(samp2, shift_size, set2_size)

    shifted_1 = np.array([booleanize(tr_set) for tr_set in shifted_1])
    shifted_2 = np.array([booleanize(tr_set) for tr_set in shifted_2])
    combined = np.concatenate([shifted_1, shifted_2])
    labels = np.concatenate([np.zeros(set1_size), np.ones(set2_size)])

    rand_inds = np.random.choice(np.arange(labels.size), labels.size, replace=False)

    labels = labels[rand_inds]
    combined = combined[rand_inds]

    return {'data': combined, 'labels': labels}



if __name__ == '__main__':
    rate = 50
    duration = 1 * second
    num_neur = 10
    set_size = 500
    data = make_test_samples(rate, duration,
                             num_neur, set_size, set_size)
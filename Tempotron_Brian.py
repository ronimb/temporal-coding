import numpy as np
import brian2 as b2
from brian2.units import Hz, ms
from numba import jit, prange
from multiprocessing import Pool
import pickle
from make_test_samples import convert_single_sample, convert_multi_samples
# %% Helper functions



def return_subset(batch_size, samples, labels, num_samples, num_neurons):
    selected_inds = np.sort(np.random.choice(range(num_samples), batch_size, replace=False))
    batch_inds = (selected_inds * num_neurons + np.arange(num_neurons)[:, np.newaxis]).flatten()
    ind_locs = np.in1d(samples['inds'], batch_inds)
    subset = np.zeros(ind_locs.sum(), dtype={'names': ('inds', 'times', 'counts'),
                                             'formats': (int, float, int)})
    subset['inds'] = samples['inds'][ind_locs]

    samp_map = {v: i for i, v in enumerate(selected_inds)}
    subset['inds'], neur = np.divmod(subset['inds'], num_neurons)
    u, inv = np.unique(subset['inds'], return_inverse=True)
    subset['inds'] = np.array([samp_map[x] for x in u])[inv] * num_neurons + neur
    subset['times'] = samples['times'][ind_locs]
    subset['counts'] = samples['counts'][ind_locs]

    return subset, labels[selected_inds]

# %%
class Tempotron():
    def __init__(self, num_neurons, tau, threshold, duration=500):
        self.num_neurons = num_neurons
        self.tau = tau * ms
        self.threshold = threshold
        self.duration = duration * ms

        # Shared variables
        self.weights = np.random.normal(0, 1e-3, num_neurons)
        self.eqs = "dv/dt = -v / tau : 1"

        # Dummy variables
        self.networks = dict()
        self.test_samples = None
        self.test_labels = None

    def make_classification_network(self, num_samples, name):
        network_size = num_samples * self.num_neurons
        count_mat = np.zeros((int(self.duration / ms * 10), network_size), int)
        target = b2.NeuronGroup(N=num_samples, model=self.eqs, threshold='v>threshold', reset='v=0',
                                namespace={'tau': self.tau, 'threshold': self.threshold})
        driving = b2.SpikeGeneratorGroup(N=(num_samples * self.num_neurons),
                                         indices=[0], times=[0 * ms])
        # counts = b2.TimedArray(values=_count_mat, dt=b2.defaultclock.dt)
        synapses = b2.Synapses(source=driving, target=target,
                               model='w: 1', on_pre='v+=w*counts(t, i)')
        i = np.arange(network_size)
        j = np.repeat(range(num_samples), self.num_neurons)
        synapses.connect(j=j, i=i)
        synapses.w = np.tile(self.weights, reps=num_samples)

        spikes = b2.SpikeMonitor(target, record=True)
        voltage = b2.StateMonitor(target, 'v', record=True)

        net = b2.Network([target, driving, synapses, spikes, voltage])
        net.store()
        self.networks[name] = dict(net=net,
                                   count_mat=count_mat,
                                   synapses=synapses,
                                   v_mon=voltage,
                                   spike_mon=spikes,
                                   num_samples=num_samples,
                                   driving=driving)

    def feed_test_samples(self, samples):
        self.test_samples, self.test_labels = convert_multi_samples(samples)

    def accuracy(self, network_name):
        network = self.networks[network_name]
        network['net'].restore()
        network['driving'].set_spikes(self.test_samples['inds'], self.test_samples['times']*ms)
        network['synapses'].w = np.tile(self.weights, reps=network['num_samples'])
        counts = network['count_mat'].copy()
        counts[
            (self.test_samples['times'] * 10).astype(int),
            self.test_samples['inds'].astype(int)] = self.test_samples['counts']
        counts = b2.TimedArray(values=counts, dt=b2.defaultclock.dt)
        network['net'].run(self.duration)
        decision = network['spike_mon'].count != 0
        correct = (decision == self.test_labels)
        return correct

    def train(self):
        pass
# %%
if __name__ == '__main__':
    with open('/mnt/disks/data/test_big.pickle', 'rb') as f:
        samples = pickle.load(f)
    T = Tempotron(500, 2, 0.3)
    T.feed_test_samples(samples)
    T.make_classification_network(200, 'test')
    T.accuracy('test').mean()


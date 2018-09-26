import numpy as np
import matplotlib.pyplot as plt
import re
import brian2 as b2
from brian2.units import ms

import pickle
from Old.make_test_samples import convert_multi_samples
# %% Helper functions
def load_sample(loc):
    with open(loc, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        samples, labels = convert_multi_samples(data)
    else:
        samples, labels = [data[0], data[1]]
    return samples, labels


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

        def make_plot_network(self):
            count_mat = np.zeros((int(self.duration / ms * 10), self.num_neurons), int)
            target = b2.NeuronGroup(N=1, model=self.eqs, threshold='v>threshold', reset='v=0',
                                    namespace={'tau': self.tau, 'threshold': self.threshold})
            driving = b2.SpikeGeneratorGroup(N=self.num_neurons,
                                             indices=[0], times=[0 * ms])
            synapses = b2.Synapses(source=driving, target=target,
                                   model='w: 1', on_pre='v+=w*counts(t, i)')
            synapses.connect(i=range(num_neurons), j=[0]*num_neurons)
            synapses.w = self.weights
            spikes = b2.SpikeMonitor(target, record=True)
            voltage = b2.StateMonitor(target, 'v', record=True)
            net = b2.Network([target, driving, synapses, spikes, voltage])
            net.store()
            self.networks['plot'] = dict(net=net,
                                         synapses=synapses,
                                         count_mat=count_mat,
                                         v_mon=voltage,
                                         spike_mon=spikes,
                                         driving=driving)
        make_plot_network(self)

    def reset(self):
        self.weights = np.random.normal(0, 1e-3, self.num_neurons)

    def make_classification_network(self, num_samples, name):
        if name not in self.networks:
            network_size = num_samples * self.num_neurons
            count_mat = np.zeros((int(self.duration / ms * 10), network_size), int)
            target = b2.NeuronGroup(N=num_samples, model=self.eqs, threshold='v>threshold', reset='v=0',
                                    namespace={'tau': self.tau, 'threshold': self.threshold})
            driving = b2.SpikeGeneratorGroup(N=network_size,
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
        else:
            self.networks[name]['synapses'].w = np.tile(self.weights, reps=num_samples)

    def accuracy(self, network_name, samples, labels, return_decision=False):
        network = self.networks[network_name]
        network['net'].restore()
        network['driving'].set_spikes(samples['inds'], samples['times']*ms)
        network['synapses'].w = np.tile(self.weights, reps=network['num_samples'])
        counts = network['count_mat'].copy()
        counts[
            (samples['times'] * 10).astype(int),
            samples['inds'].astype(int)] = samples['counts']
        counts = b2.TimedArray(values=counts, dt=b2.defaultclock.dt)
        network['net'].run(self.duration)
        decisions = network['spike_mon'].count != 0
        correct = (decisions == labels)
        if return_decision:
            return correct, decisions
        else:
            return correct

    def make_train_network(self, batch_size, name):
        if name not in self.networks:
            network_size = batch_size * self.num_neurons
            target = b2.NeuronGroup(N=network_size, model=self.eqs,
                                    namespace={'tau': self.tau})
            driving = b2.SpikeGeneratorGroup(N=network_size,
                                             indices=[0], times=[0 * ms])
            count_mat = np.zeros((int(self.duration / ms * 10), network_size), int)
            synapses = b2.Synapses(driving, target, 'w: 1', on_pre='v+=1*counts(t, i)')
            synapses.connect(condition='i==j')
            synapses.w = np.tile(self.weights, reps=batch_size)
            voltage = b2.StateMonitor(target, 'v', record=True)
            net = b2.Network([target, driving, synapses, voltage])
            net.store()
            self.networks[name] = dict(net=net,
                                       count_mat=count_mat,
                                       synapses=synapses,
                                       v_mon=voltage,
                                       num_samples=batch_size,
                                       driving=driving)
        else:
            self.networks[name]['synapses'].w = np.tile(self.weights, reps=num_samples)


    def train(self, samples, labels, batch_size=50, num_reps=100, learning_rate=1e-3, verbose=False):
        num_samples = int(np.unique(samples['inds']).shape[0] / self.num_neurons)
        self.make_classification_network(batch_size, 'batch')
        self.make_train_network(batch_size, 'train')
        for ind in range(num_reps):
            if verbose:
                print(f'train rep #{ind+1}/{num_reps}')
            batch, batch_labels = return_subset(
                batch_size, samples, labels,
                num_samples=num_samples,
                num_neurons=self.num_neurons)
            self.networks['batch']['net'].restore()
            correct, decisions = self.accuracy('batch', batch, batch_labels, return_decision=True)
            v_max_times = np.argmax(self.networks['batch']['v_mon'].v, 1)
            if (correct.shape[0] == correct.sum()):
                if verbose:
                    print('No mistakes detected')
                continue
            v_max_t = v_max_times[~correct].max()
            self.networks['train']['net'].restore()
            self.networks['train']['synapses'].w = np.tile(self.weights, reps=batch_size)
            self.networks['train']['driving'].set_spikes(batch['inds'], batch['times'] * ms)
            counts = self.networks['train']['count_mat'].copy()
            counts[
                (batch['times'] * 10).astype(int),
                batch['inds'].astype(int)] = batch['counts']
            counts = b2.TimedArray(values=counts, dt=b2.defaultclock.dt)
            if (v_max_t != 0):
                self.networks['train']['net'].run(v_max_t * ms)
                voltage_contribs = self.networks['train']['v_mon'].v
                try:
                    voltage_contribs = voltage_contribs[
                        range(voltage_contribs.shape[0]), np.repeat(v_max_times, self.num_neurons)].\
                        reshape(batch_size, self.num_neurons)[~correct]
                    weight_upd = (voltage_contribs
                                  * (batch_labels - decisions)[~correct, np.newaxis]).mean(0) * learning_rate
                    self.weights += weight_upd
                except:
                    print(f"Error occured on:")
                    print(f"Learning rate: {learning_rate}")
                    print(f"Threshold: {self.threshold}")
                    print(f"Num neurons: {self.num_neurons}\n\n")
                    print(f"""voltage_contribs: {voltage_contribs.shape}
                    v_max_times: {v_max_times.shape}
                    num_neurons: {self.num_neurons}
                    batch_size: {batch_size}
                    correct: {correct.sum()}""")
                    continue
            elif verbose:
                print('Aww Crap')

    def plot_response(self, samples, samp_num):
        sample = samples[samples['inds']==samp_num]
        network = self.networks['plot']
        network['net'].restore()
        network['driving'].set_spikes(sample['inds'], sample['times']*ms)
        network['synapses'].w = self.weights
        counts = network['count_mat'].copy()
        counts[
            (sample['times'] * 10).astype(int),
            sample['inds'].astype(int)] = sample['counts']
        counts = b2.TimedArray(values=counts, dt=b2.defaultclock.dt)
        network['net'].run(self.duration)
        v = network['v_mon'].v[0]
        plt.figure()
        plt.plot(v)
        plt.hlines(xmin=0, xmax=self.duration*10, y=self.threshold, colors='k', linestyles='dashed')
# %%
if __name__ == '__main__':
    # loc = '/mnt/disks/data/test_big.pickle'
    loc = '/mnt/disks/data/18_07_samples/mock_identical/num_neur=30_rate=15_span=3/set0.pickle'
    loc = '/mnt/disks/data/18_07_samples/vesrel/num_neur=30_rate=15_distance=0.3_span=3/set0.pickle'
    samples, labels = load_sample(loc)

    num_neurons = int(re.findall(r"num_neur=(\d+)",loc)[0])
    num_samples = 200

    T = Tempotron(num_neurons, 2, 0.005)

    T.make_classification_network(num_samples, 'test')
    print(T.accuracy('test', samples, labels).mean())
    T.train(samples, labels, batch_size=50, num_reps=20, learning_rate=1e-3)
    print(T.accuracy('test', samples, labels).mean())


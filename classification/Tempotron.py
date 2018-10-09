# ToDo: documentation for Tempotron
# ToDO: rework helper function
# ToDO: rework plotting
import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2
from brian2.units import ms


# %% Helper functions
def return_subset(batch_size, samples, labels, num_samples, num_neurons):
    selected_inds = np.sort(np.random.choice(range(num_samples), batch_size, replace=False))
    batch_inds = (selected_inds * num_neurons + np.arange(num_neurons)[:, np.newaxis]).flatten()
    ind_locs = np.in1d(samples['index'], batch_inds)
    subset = np.zeros(ind_locs.sum(), dtype={'names': ('index', 'time', 'count'),
                                             'formats': (int, float, int)})
    subset['index'] = samples['index'][ind_locs]

    samp_map = {v: i for i, v in enumerate(selected_inds)}
    subset['index'], neur = np.divmod(subset['index'], num_neurons)
    u, inv = np.unique(subset['index'], return_inverse=True)
    subset['index'] = np.array([samp_map[x] for x in u])[inv] * num_neurons + neur
    subset['time'] = samples['time'][ind_locs]
    subset['count'] = samples['count'][ind_locs]

    return subset, labels[selected_inds]


# %%
class Tempotron():
    def __init__(self, number_of_neurons, tau, threshold, stimulus_duration):
        self.number_of_neurons = number_of_neurons
        self.tau = tau * ms
        self.threshold = threshold
        self.stimulus_duration = stimulus_duration * ms

        # Shared variables
        self.weights = np.random.normal(0, 1e-3, number_of_neurons)
        self.eqs = "dv/dt = -v / tau : 1"

        # Dummy variables
        self.networks = dict()

        def make_plot_network(self):
            count_mat = np.zeros((int(self.stimulus_duration / ms * 10), self.number_of_neurons), int)
            target = b2.NeuronGroup(N=1, model=self.eqs, threshold='v>threshold', reset='v=0',
                                    namespace={'tau': self.tau, 'threshold': self.threshold})
            driving = b2.SpikeGeneratorGroup(N=self.number_of_neurons,
                                             indices=[0], times=[0 * ms])
            synapses = b2.Synapses(source=driving, target=target,
                                   model='w: 1', on_pre='v+=w*counts(t, i)')
            synapses.connect(i=range(number_of_neurons), j=[0] * number_of_neurons)
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
        self.weights = np.random.normal(0, 1e-3, self.number_of_neurons)

    def make_classification_network(self, number_of_stimuli, network_name):
        if network_name not in self.networks:
            network_size = number_of_stimuli * self.number_of_neurons
            count_mat = np.zeros((int(self.stimulus_duration / ms * 10), network_size), int)
            target = b2.NeuronGroup(N=number_of_stimuli, model=self.eqs, threshold='v>threshold', reset='v=0',
                                    namespace={'tau': self.tau, 'threshold': self.threshold})
            driving = b2.SpikeGeneratorGroup(N=network_size,
                                             indices=[0], times=[0 * ms])
            # counts = b2.TimedArray(values=_count_mat, dt=b2.defaultclock.dt)
            synapses = b2.Synapses(source=driving, target=target,
                                   model='w: 1', on_pre='v+=w*counts(t, i)')
            i = np.arange(network_size)
            j = np.repeat(range(number_of_stimuli), self.number_of_neurons)
            synapses.connect(j=j, i=i)
            synapses.w = np.tile(self.weights, reps=number_of_stimuli)

            spikes = b2.SpikeMonitor(target, record=True)
            voltage = b2.StateMonitor(target, 'v', record=True)

            net = b2.Network([target, driving, synapses, spikes, voltage])
            net.store()
            self.networks[network_name] = dict(net=net,
                                               count_mat=count_mat,
                                               synapses=synapses,
                                               v_mon=voltage,
                                               spike_mon=spikes,
                                               number_of_stimuli=number_of_stimuli,
                                               driving=driving)
        else:
            self.networks[network_name]['synapses'].w = np.tile(self.weights, reps=number_of_stimuli)

    def accuracy(self, network_name, stimuli, labels, return_decision=False):
        network = self.networks[network_name]
        network['net'].restore()
        network['driving'].set_spikes(stimuli['index'], stimuli['time'] * ms)
        network['synapses'].w = np.tile(self.weights, reps=network['number_of_stimuli'])
        counts = network['count_mat'].copy()
        counts[
            (stimuli['time'] * 10).astype(int),
            stimuli['index'].astype(int)] = stimuli['count']
        counts = b2.TimedArray(values=counts, dt=b2.defaultclock.dt)
        network['net'].run(self.stimulus_duration)
        decisions = network['spike_mon'].count != 0
        correct = (decisions == labels)
        if return_decision:
            return correct, decisions
        else:
            return correct

    def make_train_network(self, batch_size, network_name):
        if network_name not in self.networks:
            network_size = batch_size * self.number_of_neurons
            target = b2.NeuronGroup(N=network_size, model=self.eqs,
                                    namespace={'tau': self.tau})
            driving = b2.SpikeGeneratorGroup(N=network_size,
                                             indices=[0], times=[0 * ms])
            count_mat = np.zeros((int(self.stimulus_duration / ms * 10), network_size), int)
            synapses = b2.Synapses(driving, target, 'w: 1', on_pre='v+=1*counts(t, i)')
            synapses.connect(condition='i==j')
            synapses.w = np.tile(self.weights, reps=batch_size)
            voltage = b2.StateMonitor(target, 'v', record=True)
            net = b2.Network([target, driving, synapses, voltage])
            net.store()
            self.networks[network_name] = dict(net=net,
                                               count_mat=count_mat,
                                               synapses=synapses,
                                               v_mon=voltage,
                                               number_of_stimuli=batch_size,
                                               driving=driving)
        else:
            self.networks[network_name]['synapses'].w = np.tile(self.weights, reps=batch_size)

    def train(self, stimuli, labels, batch_size=50, num_reps=100, learning_rate=1e-3, verbose=False):
        num_samples = int(np.unique(stimuli['index']).shape[0] / self.number_of_neurons)
        self.make_classification_network(batch_size, 'batch')
        self.make_train_network(batch_size, 'train')
        for ind in range(num_reps):
            if verbose:
                print(f'train rep #{ind+1}/{num_reps}')
            batch, batch_labels = return_subset(
                batch_size, stimuli, labels,
                num_samples=num_samples,
                num_neurons=self.number_of_neurons)
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
            self.networks['train']['driving'].set_spikes(batch['index'], batch['time'] * ms)
            counts = self.networks['train']['count_mat'].copy()
            counts[
                (batch['time'] * 10).astype(int),
                batch['index'].astype(int)] = batch['count']
            counts = b2.TimedArray(values=counts, dt=b2.defaultclock.dt)
            if (v_max_t != 0):
                self.networks['train']['net'].run(v_max_t * ms)
                voltage_contribs = self.networks['train']['v_mon'].v
                try:
                    voltage_contribs = voltage_contribs[
                        range(voltage_contribs.shape[0]), np.repeat(v_max_times, self.number_of_neurons)]. \
                        reshape(batch_size, self.number_of_neurons)[~correct]
                    weight_upd = (voltage_contribs
                                  * (batch_labels - decisions)[~correct, np.newaxis]).mean(0) * learning_rate
                    self.weights += weight_upd
                except:
                    print(f"Error occured on:")
                    print(f"Learning rate: {learning_rate}")
                    print(f"Threshold: {self.threshold}")
                    print(f"Num neurons: {self.number_of_neurons}\n\n")
                    print(f"""voltage_contribs: {voltage_contribs.shape}
                    v_max_times: {v_max_times.shape}
                    number_of_neurons: {self.number_of_neurons}
                    batch_size: {batch_size}
                    correct: {correct.sum()}""")
                    continue
            elif verbose:
                print('Aww Crap')

    def plot_response(self, stimuli, stim_num):
        sample = stimuli[stimuli['index'] == stim_num]
        network = self.networks['plot']
        network['net'].restore()
        network['driving'].set_spikes(sample['index'], sample['time'] * ms)
        network['synapses'].w = self.weights
        counts = network['count_mat'].copy()
        counts[
            (sample['time'] * 10).astype(int),
            sample['index'].astype(int)] = sample['count']
        counts = b2.TimedArray(values=counts, dt=b2.defaultclock.dt)
        network['net'].run(self.stimulus_duration)
        v = network['v_mon'].v[0]
        plt.figure()
        plt.plot(v)
        plt.hlines(xmin=0, xmax=self.stimulus_duration * 10, y=self.threshold, colors='k', linestyles='dashed')

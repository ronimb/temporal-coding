import numpy as np
import brian2 as b2
from brian2.units import ms
import pickle
import matplotlib.pyplot as plt
import time
import sys

from make_test_samples import *

# %% Generate samples for use in testing - REMOVE once done!
num_neurons = 500
rate = 100
samples_tempshift = gen_with_temporal_shift(rate=rate,
                                  num_neur=num_neurons,
                                  shift_size=5)

samples_ves = gen_with_vesicle_release(rate=rate,
                                       num_neur=num_neurons,
                                       beta_params=dict(
                                           span=5,
                                           mode=1),
                                       num_ves=20)

# %% helper functions
def flatten_sample_old(sample):
    inds = []
    times = []
    for i, neuron in enumerate(sample):
        inds.extend([i] * neuron.shape[0])
        times.extend(neuron)
    return np.array([inds, times])

def convert_sample(sample):
    inds = []
    times = []
    counts = []
    for i, neuron in enumerate(sample):
        neuron = np.trunc(neuron * 10) / 10
        time, count = np.unique(neuron, return_counts=True)
        inds.extend([i] * time.shape[0])
        times.extend(time)
        counts.extend(count)
    return np.array([inds, times, counts])

# %%
class Tempotron:
    def __init__(self, num_neurons, tau, threshold, duration=500):
        #TODO: Include t_max and v_max variables in the brian model, have them updated per-spike and try to create a custom event when t=t(end) to update the model weights
        #
        self.duration = duration * ms
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.tau = tau * ms

        # Helper variables
        self._count_mat = np.zeros((duration*10, self.num_neurons), int)
        # Synaptic efficacies
        self.weights = np.random.normal(0, 1e-3, num_neurons)

        # Model definition
        b2.start_scope()
        eqs = 'dv/dt = -v / tau: 1'
        self.counts = b2.TimedArray(self._count_mat, dt=b2.defaultclock.dt)

        # Evaluation network
        self.target = b2.NeuronGroup(1, eqs, threshold='v>threshold', reset='v=0',
                                     namespace={'tau': self.tau, 'threshold': self.threshold})
        self.driving = b2.SpikeGeneratorGroup(self.num_neurons, [0], [0] * ms)

        self.spike_response = 'v+=w*counts(t, i)'
        self.synapses = b2.Synapses(self.driving, self.target, 'w: 1', on_pre=self.spike_response)
        self.synapses.connect(i=list(range(self.num_neurons)), j=0)
        self.synapses.w = self.weights

        self.voltage = b2.StateMonitor(self.target, 'v', record=True)
        self.spikes = b2.SpikeMonitor(self.target, record=True)

        self.net = b2.Network(self.target, self.driving, self.synapses,
                              self.voltage, self.spikes)
        self.net.store()

        # Training network
        self.target_train = b2.NeuronGroup(self.num_neurons, eqs,
                                           namespace={'tau': self.tau})
        self.driving_train = b2.SpikeGeneratorGroup(self.num_neurons, [0], [0] * ms)

        self.train_response = 'v+=1*counts(t, i)'
        self.synapses_train = b2.Synapses(self.driving_train, self.target_train, 'w: 1', on_pre=self.train_response)
        self.synapses_train.connect(condition='i==j')
        self.synapses_train.w = self.weights

        self.voltage_train = b2.StateMonitor(self.target_train, 'v', record=True)

        self.net_train = b2.Network(self.target_train, self.driving_train, self.synapses_train,
                                    self.voltage_train)
        self.net_train.store()

        self.debug_0 = 0 # For debugging problematic samples

    def convert(func):
        """
        This function is used to flatten Neuron X Time input to a form usable as input to a brian SpikeGeneratorGroup
        Object
        """
        def is_flat(sample):
            return (sample.shape[0] == 3) & (sample.ndim == 2)

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

        def internal(*args, **kwargs):
            if is_flat(args[1]):
                return func(*args, **kwargs)
            else:
                args = list(args)
                args[1] = convert_sample(args[1])
                return func(*args, **kwargs)
        return internal

    def update_counts(self, sample):
        counts = self._count_mat.copy()
        counts[(sample[1] * 10).astype(int), sample[0].astype(int)] = sample[2]
        self.counts.values = counts

    def prep_network(self, mode='classify'):
        if mode == 'classify':
            self.net.restore()
            self.synapses.w = self.weights
        elif mode == 'train':
            self.net_train.restore()
            self.synapses_train.w = self.weights

    def set_driving(self, sample, mode='classify'):
        if mode == 'classify':
            self.driving.set_spikes(sample[0], sample[1] * ms)
        elif mode == 'train':
            self.driving_train.set_spikes(sample[0], sample[1] * ms)

    @convert
    def response(self, sample, plot=True, show=True, return_v=True):
        self.update_counts(sample)
        counts = self.counts # Required because for some reason timedarrays can't be included in networks
        self.prep_network()
        self.set_driving(sample)
        self.net.run(self.duration)
        if plot:
            plt.plot(self.voltage.t / ms, self.voltage.v[0])
            plt.hlines(self.threshold, 0, self.duration/ms)
        if show:
            plt.show()
        if return_v:
            return self.voltage.v[0]

    @convert
    def classify(self, sample):
        self.update_counts(sample)
        counts = self.counts
        self.prep_network()
        self.set_driving(sample)
        self.net.run(self.duration)
        decision = self.spikes.spike_trains()[0].shape[0] != 0
        return decision

    def accuracy(self, samples, labels):
        correct = []
        for sample, label in zip(samples, labels):
            decision = self.classify(sample)
            correct.append(decision == label)
        return np.mean(correct)

    @convert
    def train_sample(self, sample, label, learning_rate=1e-4):
        self.update_counts(sample)
        counts = self.counts
        self.prep_network()

        decision = self.classify(sample)
        match = (decision == label)

        if not(match):
            vmax_ind = self.voltage.v[0].argmax()
            t_vmax = self.voltage.t[vmax_ind]
            if t_vmax == 0:
                (print('t_vmax == 0 :('))
                self.debug_0 += 1
                return
            self.prep_network('train')
            self.set_driving(sample, mode='train')
            self.net_train.run(t_vmax)
            v_end = self.voltage_train.v[:, -1]

            weight_updates = learning_rate * (label - decision) * v_end
            self.weights += weight_updates

    def train(self, samples, labels, learning_rate=1e-4, n=[], iter=1):
        n_range = range(samples.shape[0])
        if not n:
            inds = n_range
        else:
            inds = np.random.choice(n_range, size=n, replace=False)
        for _ in range(iter):
            for i in inds:
                self.train_sample(samples[i], labels[i], learning_rate)

# %%
def sec_to_str(sec):
    m, s = divmod(sec, 60)
    return f'{m:0>2.0f}:{s:0>2.2f}'

def cycle(tempotron, samples, learning_rate=1e-4, n=[], iter=1):

    time_eval_1 = time.time()
    acc_before = tempotron.accuracy(samples['data'], samples['labels'])
    time_eval_1 = time.time() - time_eval_1
    weights_pre = tempotron.weights

    time_train = time.time()
    tempotron.train(samples['data'], samples['labels'], n=n, iter=iter)
    time_train = time.time() - time_train


    time_eval_2 = time.time()
    acc_after = tempotron.accuracy(samples['data'], samples['labels'])
    time_eval_2 = time.time() - time_eval_2
    weights_post = tempotron.weights

    print(f'accuracy before: {acc_before}')
    print(f'accuracy after: {acc_after}')

    print(f'''Times:
    \t First accuracy evaluation: {sec_to_str(time_eval_1)}
    \t Training: {sec_to_str(time_train)}
    \t Second accuracy evaluation: {sec_to_str(time_eval_2)}''')
    return acc_before, acc_after, {'weights_pre': weights_pre, 'weights_post': weights_post}
# %%
if __name__ == '__main__':
    # loc = r'E:\OneDrive\Documents\Masters\Parnas\temporal-coding\Data\n30_r15_tempshift_testset.pickle'
    # loc = r'C:\Users\ron\OneDrive\Documents\Masters\Parnas\temporal-coding\Data\n30_r15_tempshift_testset.pickle'
    # loc = sys.argv[1]
    # with open(loc, 'rb') as file:
        # samples = pickle.load(file)

    samples = samples_ves
    num_neurons = samples['data'][0].shape[0]

    # %%
    T = Tempotron(num_neurons, 2, 0.005)
    data = cycle(T, samples, iter=5)
    print(data[:2])
    data[2]['weights_pre'] == data[2]['weights_post']


# %%
'''Current issue:
When using a large number of inputs, particularly when there are many spikes,
imbalance between positive and negative weights may cause spikes close to the beginning of the stimulus
to drop the models voltage far below 0, such that the tempotrons vMax at the end occurs right at the beginning,
this leads to problems with training'''
# %%
import numpy as np
import brian2 as b2
from brian2.units import second, ms
import matplotlib.pyplot as plt

from make_test_samples import *


class Tempotron():
    """
    This class implements a variation of the tempotron model developed by Sompolinsky Et-Al.
    This implementation assumes a simple exponential decay of synaptic voltage according to:

    dv/dt = -v / tau

    The model parameters are:
        num_neurons  : The number of neurons (spike-trains) in the input samples
        tau          : The time-constant for voltage decay
        threshold    : The voltage threshold required for the tempotron neuron to spike
        duration     : Length of input sample in seconds, typically 1
        init_weights : Allows for customized initialization of model weights
    """
    def __init__(self, num_neurons, tau, threshold, baseline_v=0, duration=1, init_weights=[]):
        self.tau = tau * ms
        self.threshold = threshold
        self.baseline_v = baseline_v
        self.num_neurons = num_neurons
        self.duration = 1 * second

        if len(init_weights) == 0:
            self.weights = np.random.normal(0, 1e-3, num_neurons)
            # self.weights = np.random.uniform(-1 / self.num_neurons, 1 / self.num_neurons, self.num_neurons)
        else:
            self.weights = init_weights

        # Set up the equations for the model
        self.eqs = '''
        dv/dt = -v / tau: 1
        '''
        self.spikefun = 'v+=w'
        self.spikefun_train = 'v+=1'

    def flatten(func):
        """
        This function is used to flatten Neuron X Time input to a form usable as input to a brian SpikeGeneratorGroup
        Object
        """
        def is_flat(sample):
            return (sample.shape[0] == 2) & (sample.ndim == 2)

        def flatten_sample(sample):
            inds = []
            times = []
            for i, neuron in enumerate(sample):
                inds.extend([i] * neuron.shape[0])
                times.extend(neuron)
            return np.array([inds, times])

        def internal(*args, **kwargs):
            if is_flat(args[1]):
                return func(*args, **kwargs)
            else:
                args = list(args)
                args[1] = flatten_sample(args[1])
                return func(*args, **kwargs)
        return internal

    @flatten
    def classify(self, sample, debug=False):
        """
        Classify given sample with current model parameters
        :param sample: numpy array with 2 rows, one with the spike timings and another with the label of neuron
                       in the input that fired at that time
        :param debug: used for debugging purposes, if True, function returns a vector of the voltage throughout the trial

        :return:
        decision: whether the neuron fired(1) or not(0) in response to the input sample
        t_vmax  : the time at which maximal voltage was achieved in the trial
        """

        b2.start_scope()
        # Required for brian to find the parameters in the namespace
        threshold = self.threshold
        baseline_v = self.baseline_v
        tau = self.tau

        # Initialize model
        target = b2.NeuronGroup(1, self.eqs, threshold='v>threshold', reset='v=baseline_v')
        driving = b2.SpikeGeneratorGroup(self.num_neurons, sample[0], sample[1] * ms)
        synapses = b2.Synapses(driving, target, 'w: 1', on_pre=self.spikefun)
        synapses.connect(i=list(range(self.num_neurons)), j=0)
        synapses.w = self.weights

        # Set up monitoring in brian in order to use variable values
        spikes = b2.SpikeMonitor(target, record=True)
        voltage = b2.StateMonitor(target, 'v', record=True)

        # Run the model
        b2.run(self.duration)

        # Determine maximal voltage and the time at which it is obtained
        vmax_ind = voltage.v[0].argmax()
        t_vmax = voltage.t[vmax_ind]

        # Decision
        decision = len(spikes.spike_trains()[0]) != 0
        if debug:
            return decision, t_vmax, voltage.v[0]
        else:
            return decision, t_vmax

    @flatten
    def train_sample(self, sample, label, learning_rate=1e-4, debug=False):
        """
        Train the model with a single sample, this function feeds the sample into the model neuron and corrects the
        weights if a wrong classification is given.
        :param sample: numpy array with 2 rows, one with the spike timings and another with the label of neuron
                       in the input that fired at that time
        :param label: the real label (0 or 1) of the sample input
        :param learning_rate: a constant controlling the size of the weight adjustment in response to a wrong classification
        :param debug: used for debugging purposes, if True, returns the decision(0 or 1) for the input sample and the
                      weight update in response to it (all zeros if decision == label)
        """
        tau = self.tau  # Required for brian to find variable in the namespace

        # Computer model decision and the time of maximal voltage for the given input
        decision, t_vmax = self.classify(sample)

        # TODO: Remove this if once cause of error is found
        if t_vmax == 0:
            print('Encountered sample with t_vmax=0')
            print(f'''Weight attributes are:
            \tNumWeights: {self.weights.shape[0]}
            \tMax: {self.weights.max()}
            \tMin: {self.weights.min()}
            \tMean: {self.weights.mean()}''')
            return
        # Determine if the model decisions is equal to the label
        match = (decision == label)

        # Update weights if decision and label do not match
        weight_upd = np.zeros_like(self.weights)
        if not match:
            temp_neur = b2.NeuronGroup(self.num_neurons, self.eqs)
            driving = b2.SpikeGeneratorGroup(self.num_neurons, sample[0], sample[1] * ms)
            synapses = b2.Synapses(driving, temp_neur, on_pre=self.spikefun_train)
            synapses.connect(i=list(range(self.num_neurons)), j=list(range(self.num_neurons)))

            voltages = b2.StateMonitor(temp_neur, 'v', record=True)
            # Run until maximal voltage is obtained
            b2.run(t_vmax)
            # Find spikes contributions
            v_end = voltages.v[:, -1]
            # Update weights
            weight_upd = learning_rate * (label - decision) * v_end
            self.weights += weight_upd

        if debug:
            return decision, weight_upd

    @flatten
    def plot_response(self, sample):
        """
        Used for plotting response to a sample input, generates a plot of voltage over time in response to the input
        :param sample: numpy array with 2 rows, one with the spike timings and another with the label of neuron
                       in the input that fired at that time
        """
        # Required for brian to find the parameters in the namespace
        tau = self.tau
        threshold = self.threshold
        baseline_v = self.baseline_v

        # Initialize model
        b2.start_scope()
        target = b2.NeuronGroup(1, self.eqs, threshold='v>threshold', reset='v=baseline_v')
        driving = b2.SpikeGeneratorGroup(self.num_neurons, sample[0], sample[1] * ms)
        synapses = b2.Synapses(driving, target, 'w: 1', on_pre=self.spikefun)
        synapses.connect(i=list(range(self.num_neurons)), j=0)
        synapses.w = self.weights

        # Set up monitoring in brian in order to use variable values
        spikes = b2.SpikeMonitor(target, record=True)
        voltage = b2.StateMonitor(target, 'v', record=True)

        # Run model to obtain values
        b2.run(self.duration)

        # Plotting
        times = voltage.t / ms
        plt.plot(times, voltage.v[0])
        plt.hlines(threshold, times[0], times[-1])

    def accuracy(self, samples, labels, n=[], debug=False):
        """
        This function gets a set of samples and their true labels and feeds the successively to the model, checking
        for each sample whether the models decision is equal to the true label and calculating the fraction of the
        correct responses over the entire set

        :param samples: An array of input samples, each sample is a Neuron X Time array
        :param labels: An array of the true labels for all of the samples
        :param n: Number of samples from the set to use for calculating accuracy
        :param debug: used for debugging, if True returns times of maximal voltage and voltage traces obtained for each
                      sample, as well as the indexes of the samples fed to the model
        :return:
        """

        # Choose samples to use for calculation
        n_range = range(samples.shape[0])
        if not n:
            inds = n_range
        else:
            inds = np.random.choice(n_range, size=n, replace=False)

        # Iterate over samples
        correct = []
        if debug:
            times = []
            all_vs = []
            for i in inds:
                decision, t_vmax, voltages = self.classify(samples[i], debug=True)
                times.append(t_vmax)
                all_vs.append(voltages)
                correct.append(decision == labels[i])
            return np.mean(correct), times, all_vs, inds
        else:
            for i in inds:
                decision, _ = self.classify(samples[i])
                correct.append(decision == labels[i])
            return np.mean(correct)

    def train(self, samples, labels, learning_rate=1e-4, n=[], debug=False):
        """
        This function gets a set of samples and their true labels and successively trains the model on each of the samples

        :param samples: An array of input samples, each sample is a Neuron X Time array
        :param labels: An array of the true labels for all of the samples
        :param learning_rate: a constant controlling the size of the weight adjustment in response to a wrong classification
        :param n:  Number of samples from the set to use for training
        :param debug: used for debugging, if True function returns the decisions and weight_updates for each sample, as
                      well as the indexes of the samples used for training
        :return:
        """
        # Choose samples to use for calculation
        n_range = range(samples.shape[0])
        if not n:
            inds = n_range
        else:
            inds = np.random.choice(n_range, size=n, replace=False)

        # Iterate over samples
        if debug:
            decisions, w_upds = [[]]
            for i in inds:
                d, w = self.train_sample(samples[i], labels[i], learning_rate, debug=True)
                decisions.append(d)
                w_upds.append(w)
            return decisions, w_upds, inds
        else:
            for i in inds:
                self.train_sample(samples[i], labels[i], learning_rate)

# %%
if __name__ == '__main__':
    test_sample_params = dict(rate=150,
                              duration_sec=1,
                              num_neur=300,
                              shift_size=5,
                              set1_size=500,
                              set2_size=500)
    samples = make_test_samples(**test_sample_params)
    T = Tempotron(
        num_neurons=test_sample_params['num_neur'],
        tau=2,
        threshold=0.005, )
    # res = T.classify(samples['data'][0])
    # acc_pre= T.accuracy(samples['data'], samples['labels'], 200)
    # print('Accuracy before {}'.format(acc_pre))
    # T.train(samples['data'], samples['labels'], n=200)
    # acc_post = T.accuracy(samples['data'], samples['labels'], 200)
    # print('Accuracy after {}'.format(acc_post))

    def trial(samples, tempotron, n, train_reps):
        acc_pre = tempotron.accuracy(samples['data'], samples['labels'], n)
        for _ in range(train_reps):
            tempotron.train(samples['data'], samples['labels'], n=n)
        acc_post = tempotron.accuracy(samples['data'], samples['labels'], n)
        return (acc_pre, acc_post)
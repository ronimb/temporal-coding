#  This is used for plotting a stimulus which consists of multiple neurons
import numpy as np
import matplotlib.pyplot as plt


def plot_stimulus(stimulus, show=True, color='blue'):
    """

    :param stimulus: numpy array where each element is a single neurons spike times, specified in milliseconds
    :param show: whether to automatically display the plot or not, use false for comparing different neurons
    :param color: color in which to display the stimulus, use different colors for comparing different stimuli
    """
    for i, neuron in enumerate(stimulus):
        num_spikes = neuron.shape[0]
        plt.scatter(neuron, [i] * num_spikes, marker='|', c=color)
    if show:
        plt.show()

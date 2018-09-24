from Tempotron_Brian import Tempotron
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %%
def load_sample(loc):
    with open(loc, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_sample(vs, num=0):
    plt.plot(vs[num, :])
    plt.show()

def prc_plot(vs, newfig=False):
    prcs = np.linspace(0, 100, 1000)
    vals = np.percentile(vs, prcs)
    if newfig:
        plt.figure()
    plt.scatter(prcs, vals)
    plt.show()
# %%
data_folder = '/mnt/disks/data/08_07_18_vesrel/'
num_samps = 30
# %%
condition = 'num_neur=30_rate=100_span=3'
samples = load_sample(data_folder + condition + '/set0.pickle')
# %%
# TODO: Check variability between samples

Vs = np.zeros((num_samps, 20*5000))
T = Tempotron(30, 2, 0.25)
T.make_classification_network(200, 'test')
for i in range(num_samps):
    samples = load_sample(data_folder + condition + f'/set{i}.pickle')
    T.feed_test_samples(samples)
    T.accuracy('test')
    Vs[i, :] = T.networks['test']['v_mon'].v[0:20, :].flatten()
# %%
thresholds = [0.005, 0.01, 0.1, 0.5, 1]
for thresh in thresholds:
    T = Tempotron(30, 2, thresh)
    T.feed_test_samples(samples)
    T.make_classification_network(200, 'test')
    T.accuracy('test')
    v = T.networks['test']['v_mon'].v
    prc_plot(v)
    plt.legend(thresholds)

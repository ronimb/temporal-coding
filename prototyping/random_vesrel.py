import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import pandas as pd
# %%


def make_beta(span=9, mode=1, base_span=3, base_a=2):
    beta_a = 1 + (base_span / span) * (base_a - 1)
    beta_b = (span * (beta_a - 1) - mode * (beta_a - 2)) / mode

    print(f'Making beta distribution with a={beta_a:.3f}, b={beta_b:.3f}, scale={span}')
    beta_dist = beta(a=beta_a, b=beta_b, scale=span)
    return beta_dist


def show_beta(beta_dist, span=10):
    values = np.linspace(0, span, 1000)
    densities = beta_dist.pdf(values)

    plt.plot(values, densities, label='test')
    plt.show()


# %% Crating the beta distribution
beta_params = dict(
    span=15,
    mode=1,
    base_span=3,
    base_a=2
)
spans = [3, 6, 9]
b = make_beta(**beta_params)

print(f'3: {b.cdf(3)}\n6: {b.cdf(6)}\n9: {b.cdf(9)}')
show_beta(b, span=beta_params['span'])
# %%
## Code for generating based on fixed number of bins
# n_bins = 1500
# bin_borders = np.linspace(0, 9, n_bins + 1)
def vesicle_release_nonfixed(prob_dist, num_ves, span, dt_ms=0.01, maxspan=9):
    multiplication_factor = num_ves / prob_dist.cdf(maxspan)
    
    bin_borders = np.arange(0, maxspan + dt_ms, dt_ms)
    bin_centers = bin_borders[:-1] + dt_ms / 2

    bin_probs = prob_dist.cdf(bin_borders[1:]) - prob_dist.cdf(bin_borders[:-1])

    rand_vals = np.random.rand((bin_centers <= span).sum())

    rel_inds = rand_vals <= (bin_probs[:rand_vals.shape[0]] * multiplication_factor)

    return rel_inds
# %%
dt_ms = 0.01 # In ms

n_ves = 20
test_reps = 1000

multiplication_factor = n_ves / b.cdf(9)

bin_borders = np.arange(0, 9 + dt_ms, dt_ms)
bin_centers = bin_borders[:-1] + (dt_ms / 2)
bin_probs = {
    span:
        b.cdf(bin_borders[bin_borders <= span][1:]) - b.cdf(bin_borders[bin_borders <= span][:-1])
    for span in spans
}

rand_val_bins = {span: np.random.rand(test_reps, bin_probs[span].shape[0]) for span in spans}

rel_inds_bins = {span: rand_val_bins[span] <= bin_probs[span] * multiplication_factor for span in spans}

tot_rel_bins = pd.DataFrame({span: rel_inds_bins[span].sum(1) for span in spans})
tot_rel_bins.describe()
# %%
for i in range(30):
    inds = rel_inds_bins[9][i]
    plt.scatter(bin_centers[inds], [i + 0.1] * inds.sum(), c='b', marker='|')

# %% Simple testing
duration_ms = 500
freq = 30

stimulus = np.linspace(0,500, freq//2)

n_spikes = stimulus.shape[0]

rand_vals = np.random.rand(n_spikes, bin_probs[9].shape[0])

rel_inds = np.where(rand_vals <= bin_probs[9] * multiplication_factor)
rel_inds = [rel_inds[1][rel_inds[0] == i][:n_ves] for i in range(n_spikes)]

rel_offsets = [bin_centers[inds] for inds in rel_inds]

rel_times = np.hstack(np.add(stimulus, rel_offsets))

plt.figure()
plt.plot(stimulus, [1] * stimulus.shape[0], 'b|')
plt.plot(rel_times, [2] * rel_times.shape[0], 'b|')
"""
This file contains all the functions required for transforming a stimulus
by transforming spikes into released vesicles
"""
# %%
import brian2 as b2
from brian2.units import ms, Hz
from multiprocessing import Pool
import numpy as np


# %%
def vesicle_release_fixed(stimulus, release_duration, number_of_vesicles,
                          distribution_mode=1, mode_centrality=3, decay_velocity=2):
    def _release_for_neuron(neuron):
        """
        Used for vesicle release from single spike, returns vesicle release times
        """
        # Calculate parameters for beta distribution
        beta_a = 1 + (mode_centrality / release_duration) * (decay_velocity - 1)
        beta_b = (release_duration * (beta_a - 1) - distribution_mode * (beta_a - 2)) / distribution_mode

        # Generate release times
        number_of_spikes = neuron.shape[0]
        vesicle_times = np.random.beta(beta_a, beta_b, size=(number_of_vesicles, number_of_spikes))

        return vesicle_times


def vesicle_release_stochastic():
    pass

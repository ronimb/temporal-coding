# TODO: Write documentation for experiment_template
"""

"""
from generation import make_set_from_specs
from generation import transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Parameter specification
# These are the basic parameters for the spiking neuron stimuli from which the experiment originites
creation_params = dict(
    frequency=15,
    number_of_neurons=30,
    stimulus_duration=500,
)

# These parameters control how similar the basic two original stimuli will be, if random generation is desired,
# Assign each of these with None
origin_transform_function = transform.symmetric_interval_shift  # The function to use for transfrom stimulus_a to stimulus_b
origin_transform_params = dict(  # The parameters with which to execute the specified transformation function
    stimulus_duration=creation_params['stimulus_duration'],
    interval=(3, 5)
)

# These parameters set the function that will be used to generate the transformed version of both stimuli
set_transform_function = transform.stochastic_release  # The function to use when generating each transformed stimulus
set_transform_params = dict(  # The parameters with which to execute the specified transformation function
    release_duration=5,
    number_of_vesicles=20,
    stimulus_duration=creation_params['stimulus_duration'],
    release_probability=1,
    num_transformed=50
)

# %%

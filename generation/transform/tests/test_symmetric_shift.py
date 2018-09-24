from generation import make_stimulus
import numpy as np
from generation.transform.temporal_shift import symmetric_interval_shift
# %%
# Set parameters for stimulus generation
frequency = 50
duration = 0.5
num_neurons = 30
exact_frequency = False

# Set parameters for symmetric shift
interval = [3e-3, 5e-3]
num_shifted = 100

# Create stimulus
stimulus = make_stimulus(frequency, duration,
                         num_neurons, exact_frequency = False)




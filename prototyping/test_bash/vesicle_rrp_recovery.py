import numpy as np
import sympy as sym
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# %%
# Based on the results observed in: Homeostatic Matching and Nonlinear  Amplification at Identified Central Synapses

n_max = 10
p = 0.8

spikes_temporal_distance = {f: 1000 / f for f in (7, 20, 50)}

# At 7 hertz, uEPSCs are consistent (no decay) and are ~4 pA - 4 vesicles released
r7 = spikes_temporal_distance[7] / 4
# At 20 hertz, uEPSC are also consistent, but are ~2.5 pA - 2-3 vesicles released
r20 = spikes_temporal_distance[20] / 2.5

print({7: r7, 20:r20})

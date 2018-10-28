import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numba import jit

# %%
# Values as given in the article synaptic theory of working memory
tau_d = 0.2  # Sec
tau_f = 1.5  # Sec


def simple_decay(val, time, tau):
    return (1 - val) / tau
    # Such models may make use of the time variable as well, this can be used
    # to do model the effect of a spike, as well as to treat discretization


t = np.linspace(0, 2, 101)
val = 0
sol_d = odeint(simple_decay, val, t, (tau_d,))
sol_f = odeint(simple_decay, val, t, (tau_f,))

plt.figure()
plt.plot(t * 1000, sol_d)
plt.plot(t * 1000, sol_f)
plt.xlabel('ms')
plt.ylabel('fraction vesicles recovered')

# finding first time instant when recovery is complete (99%)
first_ms = t[np.isclose(sol_d[:, 0], 0.99, atol=0.001)][0] * 1000
print(first_ms)
# if it takes "first_ms" seconds for the entire pool to replenish, given a pool size one can calculate the recovery time for a single vesicle
num_ves = 20
ves_recovery_time = first_ms / num_ves
print(ves_recovery_time)
# %% prototyping the new release function
duration = 500
freq = 15
dt = 0.01
init_n = 0
tau_n = 15
tau_p = 1500
max_n = 20
init_p = 0
base_p = 0.15
times = np.arange(0, duration, dt)

spikes = np.random.rand(times.shape[0]) <= (freq * dt / 1000)
spikes = times[spikes]

def test_d():
    n_t = np.zeros_like(times)
    p_t = np.zeros_like(times)
    r_t = np.zeros_like(times)
    n_t[0] = init_n
    p_t[0] = init_p

    for i, t in enumerate(times[1:-1]):
        # Find when vesicle pool started replenishment
        past_times = times <= t  # Locate indexes of past time bins
        past_max_n_times = times[past_times][n_t[past_times] == max_n]  # Look for times in the pest when pool was full
        # Choose the latest of these times, if pool was never full, use beginning of experiment as reference
        if past_max_n_times.size == 0:
            time_since_max_n = t
        else:
            time_since_max_n = t - past_max_n_times[-1]

        # Calculate replenished
        replenished = 0 if n_t[i] == max_n else np.isclose(time_since_max_n % tau_n, 0).astype(int)

        # Calculate dp
        dp = -(p_t[i] / tau_p) + base_p * (t == spikes).any().astype(int)  # Temporary dp=0 for test purposes
        # Calculate p
        p_t[i + 1] = np.clip(p_t[i] + dp, 0, 1)
        # Calculate released
        # r_t[i] = np.round(p_t[i] * n_t[i])
        r_t[i] = (np.random.rand(int(n_t[i])) < p_t[i]).sum()
        # dn = replenished - released
        dn = replenished - r_t[i]
        n_t[i + 1] = n_t[i] + dn
    return n_t, p_t, r_t


n_t, p_t, r_t = test_d()

plt.figure()
plt.subplot(311)
plt.plot(times, p_t)
plt.subplot(312)
plt.plot(times, n_t)
plt.subplot(313)
plt.plot(times, r_t)




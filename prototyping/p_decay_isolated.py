import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;
from mpl_toolkits.mplot3d import Axes3D
sns.set()
from scipy.integrate import solve_ivp
from itertools import product
from multiprocessing import Pool
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# %%
def tau_p(p_max, d_s=15, eps=1e-4):
    return d_s / (np.log(p_max) - np.log(eps))


def p_t(t, p_max, **kwargs):
    p = p_max * np.exp(-t / tau_p(p_max, **kwargs))
    return p


def dn_dt(t, n, p_max, **kwargs):
    r_t = p_t(t, p_max, **kwargs) * n
    return -r_t


def n(time_bins, p_max, init_n=20, **kwargs):
    integrated = solve_ivp(fun=lambda t, y: dn_dt(t, y, p_max, **kwargs),
                           t_span=(0, time_bins[-1]),
                           y0=(init_n,),
                           t_eval=time_bins)
    return integrated.y.squeeze()


def p_t_wrong(t, d_s=15, d_lr=9, n_max=20, **kwargs):
    # Sadly, however fancy this function may be it is simply in correct because it operates
    # On the assumption that n is fixed at all time points to arrive at the expected value
    dummy_p_max = 1
    tau = tau_p(dummy_p_max, d_s, **kwargs)
    p = np.exp(-t / tau)
    p_sum_lr = tau - (tau * np.exp(-d_lr / tau))
    p = p / p_sum_lr * n_max
    return p


def dp_dt():
    pass


def dn_dt_wrong(t, n, **kwargs):
    r_t = p_t(t, **kwargs) * n
    return -r_t


def n_wrong(time_bins, init_n=20, **kwargs):
    integrated = solve_ivp(fun=lambda t, y: dn_dt(t, y, **kwargs),
                           t_span=(0, time_bins[-1]),
                           y0=(init_n,),
                           t_eval=time_bins)
    return integrated.y.squeeze()


# %% helper functions
def make_t_data(times, d_s, p_max, n_max, **kwargs):
    # Making sure our input variables can be interpreted as arrays if they are not already
    d_s_arr = np.r_[d_s]
    p_max_arr = np.r_[p_max]
    n_max_arr = np.r_[n_max]

    conditions = [
        {'d_s': d_s, 'p_max': p_max, 'n_max': n_max} for d_s, p_max, n_max in product(d_s_arr, p_max_arr, n_max_arr)]

    results = []
    for params in conditions:
        current_result = params.copy()
        param_dict = {key: params[key] for key in params if key != 'n_max'}
        current_p = p_t(times, **param_dict, **kwargs)
        current_n = n(times, init_n=params['n_max'], **param_dict, **kwargs)
        current_result.update({
            't': times,
            'p': current_p,
            'n': current_n,
        })
        results.append(pd.DataFrame(current_result))
    results = pd.concat(results)
    results['r'] = results.p * results.n
    return results

def _make_for_p_params(params, pmax_values, dt, **kwargs):
    times = np.arange(0 + dt, params['t'] + dt, dt)
    current_result = params.copy()
    current_p = p_t(params['t'], np.r_[pmax_values], d_s=params['d_s'], **kwargs)
    current_n = []
    for p_max in pmax_values:
        current_n.append(
            n(times, init_n=params['n_max'], p_max=p_max, d_s=params['d_s'], **kwargs)[-1])
    current_result.update({
        'p_max': pmax_values,
        'p': current_p,
        'n': current_n,
        't': [params['t']] * pmax_values.shape[0]
    })
    return pd.DataFrame(current_result)
def make_p_data(pmax_values, t, d_s, n_max, dt=0.01, **kwargs):
    # Making sure our input variables can be interpreted as arrays if they are not already
    t_arr = np.r_[t]
    d_s_arr = np.r_[d_s]
    n_max_arr = np.r_[n_max]

    conditions = [
        {'t': t, 'd_s': d_s, 'n_max': n_max} for t, d_s, n_max in product(t_arr, d_s_arr, n_max_arr)]

    # results = []
    # for params in conditions:
    #     results.append(_make_for_p_params(params, pmax_values, dt, **kwargs))
    with Pool(8) as P:
        results = P.starmap(_make_for_p_params, ((c, pmax_values, dt) for c in conditions))
    results = pd.concat(results)
    return results

def make_combo_data(times, combos, n_max, **kwargs):
    results = []
    for i, params in combos.iterrows():
        current_result = dict(params)
        current_result['n_max'] = n_max
        current_p = p_t(times, p_max=params['p_max'], d_s=params['d_s'], **kwargs)
        current_n = n(times, init_n=n_max, p_max=params['p_max'], d_s=params['d_s'], **kwargs)
        current_result.update({
            't': times,
            'p': current_p,
            'n': current_n,
        })
        results.append(pd.DataFrame(current_result))
    results = pd.concat(results)
    results['r'] = results.p * results.n
    return results

def plot_data(data, x='t', n_max=np.inf):
    pmax_values = np.unique(data.p_max)
    ds_values = np.unique(data.d_s)
    t_values = np.unique(data.t)
    if x == 't':
        ys = ['p', 'r', 'n']
        fig_shape = (len(ys), len(pmax_values))
        # fig, axarr = plt.subplots(len(ys), len(pmax_values), sharex=True, sharey='row')
        var_values = pmax_values
        var_name = 'p_max'
        hue = 'd_s'
    if (x == 'p_max') or (x == 'p'):
        x = 'p_max'
        ys = ['n', 'p']
        fig_shape = (len(ys), len(ds_values))
        # fig, axarr = plt.subplots(len(ys), len(t_values), sharex=True, sharey='row')
        var_values = ds_values
        var_name = 'd_s'
        hue = 't'

    fig, axarr = plt.subplots(*fig_shape, sharex=True, sharey='row')
    for i, (y, axrow) in enumerate(zip(ys, axarr)):
        if np.ndim(axrow) == 0:
            ax = axrow
            value = var_values
            plot_data = data[(data[var_name] == value[0]) & (data['n'] < n_max)]
            sns.lineplot(x=x, y=y, hue=hue, data=plot_data, ax=ax)
            if hue:
                ax.legend_.remove()
            if i == 0:
                ax.set_title(f'{var_name} = {value[0]}')
        else:
            for value, ax in zip(var_values, axrow):
                plot_data = data[(data[var_name] == value) & (data['n'] < n_max)]
                sns.lineplot(x=x, y=y, hue=hue, data=plot_data, ax=ax)
                if hue:
                    ax.legend_.remove()
                if i == 0:
                    ax.set_title(f'{var_name} = {value}')
    if hue:
        if axarr.ndim == 2:
            axarr[0, 0].legend(np.unique(data[hue]))
        else:
            axarr[0].legend(np.unique(data[hue]))

def plot_combo(combo_data):
    ds_values = np.unique(combo_data.d_s)
    pmax_values = np.unique(combo_data.p_max)

    x = 't'
    ys = ['p', 'r', 'n']
    hue = 'd_s'
    fig_shape = (len(ys), len(pmax_values))

    fig, axarr = plt.subplots(*fig_shape, sharex=True, sharey='row')
    for i, (y, axrow) in enumerate(zip(ys, axarr)):
        for j, (p_max, ax) in enumerate(zip(pmax_values, axrow)):
            plot_data = combo_data[combo_data['p_max'] == p_max]
            sns.lineplot(x=x, y=y, data=plot_data, ax=ax)
            if i==0:
                ax.set_title(f'p_max={p_max} | d_s={np.unique(plot_data.d_s)}')
# %% full dimensional analysis at t=[3, 6, 9]
d_lr = 9
n_max = 100

pmax_range = (0.5, 5)
n_pmax = 250
ds_range = (30, 1000)
n_ds = 250

dt = 0.01  # ms

t_values = [3, 6, 9]
# Setup logarithmically space values
pmax_values = np.linspace(*pmax_range, num=n_pmax)
ds_values = np.linspace(*ds_range, num=n_ds)
# pmax_values = np.logspace(*np.log10(pmax_range), num=n_pmax, endpoint=False)
# ds_values = np.logspace(*np.log10(ds_range), num=n_ds, endpoint=False).round()
# ds_values = np.logspace(np.log10(30), np.log10(1000), 4, endpoint=False).round()

mecha_data = make_p_data(pmax_values=pmax_values, t=t_values,
                     d_s=ds_values, n_max=n_max, dt=dt)

# %%
fig = plt.figure()
selected_data = mecha_data[mecha_data.t == 9]
selected_data['gap_69'] = mecha_data.n[mecha_data.t == 6] - selected_data.n
inds = (selected_data.n <= (n_max  - 20)) & (selected_data.n > (n_max - 20 - 1))
# selected_data = selected_data[inds]
ax = fig.add_subplot(1,2, 1, projection='3d')
ax2 = fig.add_subplot(1,2, 2, projection='3d')
ax.plot_trisurf(selected_data['p_max'], selected_data['d_s'], selected_data['n'], cmap=plt.cm.viridis, linewidth=0.2)
ax.set_title('n at t=9')
ax2.plot_trisurf(selected_data['p_max'], selected_data['d_s'], selected_data['gap_69'], cmap=plt.cm.viridis, linewidth=0.2)
ax2.set_title('6-9 gap')



# %%
plot_duration = 15
d_lr = 9
n_max = 20

dt = 0.01  # ms

times = np.arange(0 + dt, plot_duration + dt, dt)

combos = pd.DataFrame({
    'p_max': [1.55, 0.86, 0.68, 0.55, 0.46],
    'd_s': [25, 45, 80, 140, 400]
})

combo_data = make_combo_data(times, combos, n_max)
plot_combo(combo_data)

## (p_max =0.86, d_s=45)
# %%
d_lr = 9
n_max = 40

pmax_range = (0.01, 1)
n_pmax = 100
# ds_range = (25, 100)
n_ds = 4

dt = 0.01  # ms

t_values = [3, 6, 9]
# Setup logarithmically space values
pmax_values = np.logspace(*np.log10(pmax_range), num=n_pmax, endpoint=False)
ds_values = np.r_[30, 45, 60, 75]
# ds_values = np.logspace(np.log10(30), np.log10(1000), 4, endpoint=False).round()

p_data = make_p_data(pmax_values=pmax_values, t=t_values,
                     d_s=ds_values, n_max=n_max, dt=dt)

plot_data(p_data, 'p', n_max=10)

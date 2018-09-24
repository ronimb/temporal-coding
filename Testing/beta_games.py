import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy import stats
# %%
def betagames(span, base_span=3, base_a=2, mode=1, show=True):
    eps = 1e-55
    a = 1 + (base_span / span) * (base_a - 1)
    b = (span * (a - 1) - mode * (a - 2)) / mode
    dist = stats.beta(a, b)
    r = np.linspace(eps, 1-eps, 10000)
    x = dist.ppf(r)
    y = dist.pdf(x) / span
    x = x * span
    plt.plot(x, y)
    if show:
        plt.show()

def gen_emprical(span, base_span=3, base_a=2, mode=1, n=10000, plot=True, hist=True, show=True):
    a = 1 + (base_span / span) * (base_a - 1)
    b = (span * (a - 1) - mode * (a - 2)) / mode
    data = span * np.random.beta(a, b, n)
    if plot:
        sns.distplot(data, hist=hist)
        if show:
            plt.show()
    return data

# %%
base_a = 2
base_span = 3

spans = np.array([3, 6, 9])

a_vals = 1 + (base_span / spans) * (base_a - 1)
plt.figure()
for span in spans:
    betagames(span, base_span, base_a, show=False)
plt.legend(spans)
plt.title(f"Theoretical with base a = {base_a}")
plt.show()

plt.figure()
for span in spans:
    _ = gen_emprical(span, base_span, base_a, hist=True, show=False)
plt.legend(spans)
plt.title(f"Emprical with base a = {base_a}")
plt.show()

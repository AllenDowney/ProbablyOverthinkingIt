---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Chile Earthquake Magnitudes

Exploring the distribution of earthquake magnitudes in Chile using the relocated catalogue from the Centro Sismológico Nacional (CSN), 1982–mid-2020.

Potin et al. (2024), [doi:10.1785/0220240047](https://doi.org/10.1785/0220240047)


```python tags=["remove-cell"]
# Install empiricaldist if we don't already have it

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["remove-cell"]
from os.path import basename, exists


def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)


download(
    "https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/notebooks/utils.py"
)

```

```python tags=["hide-cell"]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from empiricaldist import Cdf, Surv
from utils import decorate, set_pyplot_params, make_surv, plot_error_bounds

set_pyplot_params()
```

## Load the catalogue

Data from [Potin et al. (2024)](https://doi.org/10.1785/0220240047): a revised, relocated Chilean seismic catalogue covering 1982 to mid-2020. See `chile/chilean_seismic_catalogue-1982-2020-publication_release/README.md` for column definitions and citation details.

```python tags=["hide-cell"]
DATA_DIR = "chile/chilean_seismic_catalogue-1982-2020-publication_release"
SEISMICITY_PATH = f"{DATA_DIR}/CHILE_SEISMICITY_RELOCATED.csv"

data = pd.read_csv(SEISMICITY_PATH)
data["magnitude_type"] = data["magnitude_type"].str.strip()
data.shape
```

```python tags=["hide-input"]
data.head()
```

```python tags=["hide-input"]
data.describe(include="all")
```

Magnitude is reported on several scales (`l` local, `c` coda, `w` and `ww` moment magnitude). Many early events have no magnitude estimate.

```python tags=["hide-cell"]
data["magnitude_type"].value_counts()
```

```python tags=["hide-cell"]
no_mag = data["magnitude"].isna() | (data["magnitude_type"] == "xx")
no_mag.mean()
```

Keep events with a valid magnitude for the distribution analysis.

```python tags=["hide-input"]
events = data.loc[~no_mag].copy()
events.groupby("magnitude_type")["magnitude"].describe()
```

The number of events per year has been increasing, which suggests better detection.

```python tags=["hide-input"]
events["year"].value_counts().sort_index().plot()
decorate(xlabel="Year", ylabel="Event count")
```

```python tags=["hide-input"]
for mtype, group in events.groupby("magnitude_type"):
    Cdf.from_seq(group["magnitude"]).plot(label=mtype)

decorate(xlabel="Magnitude", ylabel="CDF")
plt.legend()
plt.savefig("chile_eda_cdf.png", dpi=300)
```
The distributions for the different kinds of magnitudes are substantially different.
I'm not sure it's valid to combine them, but for a first pass analysis, that's what I'm going to do.

```python tags=["hide-cell"]
mags = events["magnitude"]
mags.describe()
```

As expected, there are only a few very large magnitudes.

```python tags=["hide-cell"]
(mags >= 7).sum(), (mags >= 7.5).sum(), mags.max()
```

Here's the tail distribution (fraction of values >= x for all x).

```python tags=["hide-cell"]
surv = make_surv(mags)
```

```python tags=["hide-cell"]
xlabel = "Magnitude"
ylabel = "Prob >= $x$"
```

```python tags=["hide-input"]
surv.plot()

decorate(xlabel=xlabel, ylabel=ylabel)
```

## Fit a Student t

Find the df parameter that best matches the tail, then the mu and sigma parameters that best match the percentiles.

```python tags=["hide-cell"]
from scipy.stats import t as t_dist


def truncated_t_sf(qs, df, mu, sigma):
    ps = t_dist.sf(qs, df, mu, sigma)
    surv_model = Surv(ps / ps[0], qs)
    return surv_model
```

```python tags=["hide-cell"]
from scipy.optimize import least_squares


def fit_truncated_t(df, surv):
    """Given df, find the best values of mu and sigma."""
    low, high = surv.qs.min(), surv.qs.max()
    qs_model = np.linspace(low, high, 1000)

    # match 20 percentiles from 10th to 99th
    ps = np.linspace(0.10, 0.9, 20)
    qs = surv.inverse(ps)

    def error_func_t(params, df, surv):
        mu, sigma = params
        surv_model = truncated_t_sf(qs_model, df, mu, sigma)

        error = surv(qs) - surv_model(qs)
        return error

    params = surv.mean(), surv.std()
    res = least_squares(error_func_t, x0=params, args=(df, surv), xtol=1e-3)
    assert res.success
    return res.x
```

```python tags=["hide-cell"]
from scipy.optimize import minimize


def minimize_df(df0, surv, bounds=[(1, 1e6)], ps=None):
    low, high = surv.qs.min(), surv.qs.max()
    qs_model = np.linspace(low, high * 1.0, 2000)

    if ps is None:
        t = surv.ps[0], surv.ps[-2]
        t = 0.1, surv.ps[-2]
        low, high = np.log10(t)
        ps = np.logspace(low, high, 30, endpoint=False)

    qs = surv.inverse(ps)

    def error_func_tail(params):
        (df,) = params
        mu, sigma = fit_truncated_t(df, surv)
        surv_model = truncated_t_sf(qs_model, df, mu, sigma)

        errors = np.log10(surv(qs)) - np.log10(surv_model(qs))
        return np.sum(errors**2)

    params = (df0,)
    res = minimize(error_func_tail, x0=params, bounds=bounds, tol=1e-3, method="Powell")
    assert res.success
    return res.x
```

Search for the value of df that best matches the tail behavior.

```python tags=["hide-cell"]
df = minimize_df(5, surv, [(1, 1000)])
df
```

Given that value of df, find mu and sigma to best match percentiles.

```python tags=["hide-cell"]
mu, sigma = fit_truncated_t(df, surv)
df, mu, sigma
```

Run the model out to magnitude 9.5

```python tags=["hide-cell"]
low, high = surv.qs.min(), 9.5
qs = np.linspace(low, high, 2000)
surv_model = truncated_t_sf(qs, df, mu, sigma)
```

On a linear scale:

```python tags=["hide-input"]
model_label = "$t$ model"

surv_model.plot(color="gray", alpha=0.4, label=model_label)
surv.plot(color="C1", ls="--", label="Data")

decorate(xlabel=xlabel, ylabel=ylabel)
plt.savefig("chile3.png", dpi=300)
```

The model fits the data well above magnitude 3.


Now on a log-y scale:

```python tags=["hide-cell"]
yticks = [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5]
ylabels = yticks
xticks = np.arange(1, 10)
xlabels = xticks
```

```python tags=["hide-input"]
surv_model.plot(color="gray", alpha=0.4, label=model_label)
plot_error_bounds(Surv(surv_model), mags.count(), color="gray", hatch="/")
surv.plot(color="C2", ls="--", label="Data")

decorate(xlabel=xlabel, ylabel=ylabel, yscale="log")
plt.xticks(xticks, xlabels)
plt.yticks(yticks, ylabels)
plt.savefig("chile4.png", dpi=300)
```


And both together

```python tags=["hide-cell"]
def plot_two(surv, surv_model, n, hatch=None, title=""):
    """Plots tail distributions on linear-y and log-y scales.

    Uses global variables model_label, ylabel, xlabel, xticks, yticks, xlabels, ylabels.
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    plt.sca(axs[0])
    surv_model.plot(color="gray", alpha=0.4, label=model_label)
    if n < 1000:
        plot_error_bounds(Surv(surv_model), n, color="gray")
    surv.plot(color="C1", ls="--", label="Data")
    decorate(ylabel=ylabel, title=title)

    plt.sca(axs[1])
    surv_model.plot(color="gray", alpha=0.4, label=model_label)
    plot_error_bounds(Surv(surv_model), n, color="gray", hatch=hatch)
    surv.plot(color="C2", ls="--", label="Data")
    decorate(xlabel=xlabel, ylabel=ylabel, yscale="log")
    plt.xticks(xticks, xlabels)
    plt.yticks(yticks, ylabels)

    plt.tight_layout()
    return axs
```

```python tags=["hide-input"]
plot_two(surv, surv_model, mags.count(), hatch="/")
plt.savefig("chile_t_model.png", dpi=300)
```

The model fits the tail behavior well and makes plausible predictions for very large earthquakes.


Copyright 2026 Allen B. Downey

Example related to Chapter 8 from [Probably Overthinking It](https://allendowney.github.io/ProbablyOverthinkingIt/).

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

```python tags=["hide-cell"]

```

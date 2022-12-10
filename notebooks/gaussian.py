import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from empiricaldist import Pmf, Cdf
from scipy.stats import norm
from scipy.optimize import least_squares


def make_cdf(seq):
    """Make a CDF from a sequence.
    
    Scale to show percentages rather than probabilities.
    
    Returns: Cdf
    """
    cdf = Cdf.from_seq(seq)
    cdf *= 100
    return cdf


def error_func(params, series):
    """Error function used to fit a Gaussian model.

    params: hypothetical mu and sigma
    series: Series of values

    returns: differences between CDF of values and Gaussian model
    """
    mu, sigma = params
    qs = series.quantile([0.1, 0.50, 0.9])
    cdf = Cdf.from_seq(series)
    error = cdf(qs) - norm.cdf(qs, mu, sigma)
    return error


def fit_model(series):
    """Fit a Gaussian model.

    series: Series of values

    returns: mu, sigma
    """
    params = series.mean(), series.std()
    res = least_squares(error_func, x0=params, args=(series,))
    return res.x


def normal_model(series, iters=201):
    """Fit a Gaussian model with error bounds.

    series: Series of values
    iters: number of random samples to generate

    returns: Cdf and arrays for the error bounds
    """
    n = len(series)
    # mu, sigma = series.mean(), series.std()
    mu, sigma = fit_model(series)
    cdf_series = make_cdf(series)

    a = np.empty(shape=(iters, len(cdf_series)))
    qs = cdf_series.qs

    for i in range(iters):
        sample = norm.rvs(mu, sigma, size=n)
        cdf = Cdf.from_seq(sample)
        a[i] = cdf(qs) * 100

    low, high = np.percentile(a, [2.5, 97.5], axis=0)
    return cdf_series, low, high


def make_plot(series, plot_model=True, **options):
    """Plot a Gaussian model with error bounds.
    
    series: Series of values
    plot_model: boolean, whether to plot the error bounds
    options: passed to Cdf.plot
    """
    cdf, low, high = normal_model(series)
    if plot_model:
        plt.fill_between(cdf.qs, low, high, lw=0, color="gray", alpha=0.3)

    cdf.plot(**options)

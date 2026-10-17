"""Fit truncated models to the tail distribution of log-scaled data.

Used by the firm-size examples (firm_size, firm_size_france). All models
are fit to log10 values above a lower cutoff, so each model is truncated
at the cutoff.

Fitting follows the same pattern for every model:

* location and scale are chosen to match percentiles of the data
  (vertical distances between tail distributions), and
* a shape parameter, if there is one, is chosen to match the largest values
  (squared differences between log tail probabilities).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.stats import gennorm, lomax, norm

from empiricaldist import TailDist
from utils import decorate, plot_error_bounds


def truncated_tail(dist, qs, *args):
    """Tail distribution of `dist` truncated at qs[0].

    dist: SciPy distribution
    qs: quantities, in increasing order
    args: shape, location, and scale parameters

    returns: TailDist
    """
    ps = dist.sf(qs, *args)
    # if the model probability underflows at the cutoff, the result is NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        return TailDist(ps / ps[0], qs)


def default_tail_ps(tail):
    """Tail probabilities used to fit and evaluate the largest values.

    returns: 30 values from 0.99 down to the second-largest value
    """
    t = 0.99, tail.ps[-2]
    return np.logspace(*np.log10(t), 30, endpoint=False)


def log_tail_error(tail, model, qs):
    """Sum of squared differences between log10 tail probabilities."""
    with np.errstate(invalid="ignore", divide="ignore"):
        errors = np.log10(tail(qs)) - np.log10(model(qs))
    total = np.sum(errors**2)
    # if the model probability underflows to 0, treat it as a bad fit
    return total if np.isfinite(total) else 1e10


def fit_loc_scale(dist, tail, shape=(), ps=None):
    """Find location and scale that match percentiles of the data.

    dist: SciPy distribution
    tail: TailDist of the data
    shape: tuple of shape parameters
    ps: probabilities to match; default 20 values from 0.1 to 0.9

    returns: array of location and scale
    """
    low, high = tail.qs.min(), tail.qs.max()
    qs_model = np.linspace(low, high, 1000)

    if ps is None:
        ps = np.linspace(0.1, 0.9, 20)
    qs = tail.inverse(ps)

    def error_func(params):
        model = truncated_tail(dist, qs_model, *shape, *params)
        return tail(qs) - model(qs)

    x0 = np.median(tail.qs), np.std(tail.qs)
    res = least_squares(error_func, x0=x0, xtol=1e-3)
    assert res.success
    return res.x


def fit_shape(dist, tail, shape0, bounds, ps=None):
    """Find the shape parameter that best matches the tail.

    dist: SciPy distribution with one shape parameter
    tail: TailDist of the data
    shape0: initial guess
    bounds: (low, high) bounds on the shape parameter
    ps: tail probabilities to match; default from `default_tail_ps`

    returns: shape parameter
    """
    low, high = tail.qs.min(), tail.qs.max()
    qs_model = np.linspace(low, high, 2000)
    qs = tail.inverse(default_tail_ps(tail) if ps is None else ps)

    def error_func(params):
        shape = tuple(params)
        loc_scale = fit_loc_scale(dist, tail, shape)
        model = truncated_tail(dist, qs_model, *shape, *loc_scale)
        return log_tail_error(tail, model, qs)

    res = minimize(error_func, x0=(shape0,), bounds=[bounds], tol=1e-3, method="Powell")
    assert res.success
    return res.x[0]


def pareto2_tail(qs, alpha, log_scale, log_cutoff):
    """Tail distribution of log10 values under a Pareto II model starting at the cutoff.

    qs: log10 quantities
    alpha: exponent
    log_scale: log10 of the scale parameter
    log_cutoff: log10 of the cutoff

    returns: TailDist
    """
    ps = lomax.sf(10**qs, alpha, loc=10**log_cutoff, scale=10**log_scale)
    return TailDist(ps, qs)


def fit_pareto2_scale(tail, alpha, log_cutoff, ps=None):
    """Given alpha, find the scale that matches percentiles of the data.

    returns: log10 of the scale parameter
    """
    low, high = tail.qs.min(), tail.qs.max()
    qs_model = np.linspace(low, high, 1000)

    if ps is None:
        ps = np.linspace(0.1, 0.9, 20)
    qs = tail.inverse(ps)

    def error_func(params):
        (log_scale,) = params
        model = pareto2_tail(qs_model, alpha, log_scale, log_cutoff)
        return tail(qs) - model(qs)

    res = least_squares(error_func, x0=(log_cutoff,), xtol=1e-3)
    assert res.success
    return res.x[0]


def fit_pareto2_alpha(tail, log_cutoff, alpha0=1, bounds=(0.2, 5), ps=None):
    """Find the exponent that best matches the tail, using the same criterion as `fit_shape`.

    returns: alpha
    """
    low, high = tail.qs.min(), tail.qs.max()
    qs_model = np.linspace(low, high, 2000)
    qs = tail.inverse(default_tail_ps(tail) if ps is None else ps)

    def error_func(params):
        (alpha,) = params
        log_scale = fit_pareto2_scale(tail, alpha, log_cutoff)
        model = pareto2_tail(qs_model, alpha, log_scale, log_cutoff)
        return log_tail_error(tail, model, qs)

    res = minimize(error_func, x0=(alpha0,), bounds=[bounds], tol=1e-3, method="Powell")
    assert res.success
    return res.x[0]


def model_errors(tail_data, tail_model):
    """Compute linear-scale and log-scale errors.

    * mean absolute difference between tail distributions at 99 percentiles
      from 1% to 99%, and
    * root mean squared difference between log10 tail probabilities at the
      probabilities from `default_tail_ps`.

    returns: tuple of (mean absolute error, RMS log10 error)
    """
    ps = np.linspace(0.01, 0.99, 99)
    qs = tail_data.inverse(ps)
    mae = np.mean(np.abs(tail_data(qs) - tail_model(qs)))

    qs_tail = tail_data.inverse(default_tail_ps(tail_data))
    with np.errstate(divide="ignore"):
        log_errors = np.log10(tail_data(qs_tail)) - np.log10(tail_model(qs_tail))
    rmse = np.sqrt(np.mean(log_errors**2))
    return mae, rmse


YTICKS_LOG = [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5]


def plot_two(tail_data, tail_model, n, model_label, xlabel, ylabel="Prob >= $x$"):
    """Plot data and model with linear-y and log-y scales, side by side.

    On the log-y scale, the shaded area shows how much variation we would
    expect due to sampling if the model were correct.
    """
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))

    plt.sca(axs[0])
    tail_model.plot(color="gray", alpha=0.4, label=model_label)
    tail_data.plot(color="C1", ls="--", label="Data")
    decorate(xlabel=xlabel, ylabel=ylabel)

    plt.sca(axs[1])
    tail_model.plot(color="gray", alpha=0.4, label=model_label)
    plot_error_bounds(tail_model, n, color="gray", hatch="/")
    tail_data.plot(color="C1", ls="--", label="Data")
    decorate(xlabel=xlabel, ylabel=ylabel, yscale="log")
    plt.yticks(YTICKS_LOG, YTICKS_LOG)


def fit_models(mags, log_cutoff=None):
    """Fit lognormal, log-gennorm, and Pareto II models; return parameters and errors.

    mags: log10 values above the cutoff
    log_cutoff: log10 of the cutoff; default is the smallest value

    returns: dict
    """
    tail = TailDist.from_seq(mags)
    qs = np.linspace(tail.qs.min(), tail.qs.max() + 0.5, 2000)
    if log_cutoff is None:
        log_cutoff = tail.qs.min()

    mu, sigma = fit_loc_scale(norm, tail, ps=np.linspace(0.1, 0.99, 20))
    _, rmse_lognormal = model_errors(tail, truncated_tail(norm, qs, mu, sigma))

    beta = fit_shape(gennorm, tail, shape0=2, bounds=(0.5, 20))
    mu_g, alpha_g = fit_loc_scale(gennorm, tail, shape=(beta,))
    _, rmse_gennorm = model_errors(tail, truncated_tail(gennorm, qs, beta, mu_g, alpha_g))
    if not np.isfinite(rmse_gennorm):
        # the fit did not converge
        beta = np.nan

    alpha = fit_pareto2_alpha(tail, log_cutoff)
    log_scale = fit_pareto2_scale(tail, alpha, log_cutoff)
    _, rmse_pareto = model_errors(tail, pareto2_tail(qs, alpha, log_scale, log_cutoff))

    return dict(
        n=len(mags),
        beta=beta,
        alpha=alpha,
        log_scale=log_scale,
        rmse_lognormal=rmse_lognormal,
        rmse_gennorm=rmse_gennorm,
        rmse_pareto=rmse_pareto,
    )

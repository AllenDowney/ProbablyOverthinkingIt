"""Supporting code for Probably Overthinking It

https://probablyoverthinking.it


"""

from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler

from scipy.optimize import minimize
from scipy.optimize import least_squares

from scipy.stats import gaussian_kde
from scipy.stats import binom
from scipy.stats import t as t_dist
from scipy.stats import norm

import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

from empiricaldist import Pmf, Cdf, Surv


def values(series):
    """Make a series of values and the number of times they appear.

    Returns a DataFrame because they get rendered better in Jupyter.

    series: Pandas Series

    returns: Pandas DataFrame
    """
    series = series.value_counts(dropna=False).sort_index()
    series.index.name = "values"
    series.name = "counts"
    return pd.DataFrame(series)


def write_table(table, label, **options):
    """Write a table in LaTex format.

    table: DataFrame
    label: string
    options: passed to DataFrame.to_latex
    """
    filename = f"tables/{label}.tex"
    fp = open(filename, "w", encoding="utf8")
    s = table.to_latex(**options)
    fp.write(s)
    fp.close()


def write_pmf(pmf, label):
    """Write a Pmf object as a table.

    pmf: Pmf
    label: string
    """
    df = pd.DataFrame()
    df["qs"] = pmf.index
    df["ps"] = pmf.values
    write_table(df, label, index=False)


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    legend = options.pop("legend", True)
    loc = options.pop("loc", "best")
    ax = plt.gca()
    ax.set(**options)

    handles, labels = ax.get_legend_handles_labels()
    if handles and legend:
        ax.legend(handles, labels, loc=loc)

    plt.tight_layout()


def anchor_legend(x, y):
    """Place the upper left corner of the legend box.

    x: x coordinate
    y: y coordinate
    """
    plt.legend(bbox_to_anchor=(x, y), loc="upper left", ncol=1)
    plt.tight_layout()


def savefig(root, **options):
    """Save the current figure.

    root: string filename root
    options: passed to plt.savefig
    """
    fmat = options.pop("format", None)
    if fmat:
        formats = [fmat]
    else:
        formats = ["pdf", "png"]

    for f in formats:
        fname = f"figs/{root}.{f}"
        plt.savefig(fname, **options)


# GAUSSIAN


def make_cdf(seq):
    """Make a CDF from a sequence.

    Scale to show percentages rather than probabilities.

    Returns: Cdf
    """
    cdf = Cdf.from_seq(seq)
    cdf *= 100
    return cdf


def make_normal_pmf(qs, mu, sigma):
    """Make a PMF for a normal distribution.

    qs: quantities
    mu, sigma: parameters

    returns Pmf
    """
    ps = norm.pdf(qs, mu, sigma)
    pmf_normal = Pmf(ps, qs)
    pmf_normal.normalize()
    return pmf_normal


def make_normal_model(pmf):
    """Make a normal model from a PMF.

    pmf: Pmf

    returns: Pmf
    """
    pmf = pmf / pmf.sum()
    mu, sigma = pmf.mean(), pmf.std()
    qs = np.linspace(0, pmf.qs.max(), 200)
    return make_normal_pmf(qs, mu, sigma)


def error_func_gauss(params, series):
    """Error function used to fit a Gaussian model.

    params: hypothetical mu and sigma
    series: Series of values

    returns: differences between CDF of values and Gaussian model
    """
    mu, sigma = params
    # TODO: check why we're only checking three points
    qs = series.quantile([0.1, 0.50, 0.9])
    cdf = Cdf.from_seq(series)
    error = cdf(qs) - norm.cdf(qs, mu, sigma)
    return error


def fit_gaussian(series):
    """Fit a Gaussian model.

    series: Series of values

    returns: mu, sigma
    """
    params = series.mean(), series.std()
    res = least_squares(error_func_gauss, x0=params, args=(series,))
    return res.x


def fit_normal(series):
    """Find the model that minimizes the errors in percentiles.

    series: Series of quantities

    returns: scipy.stats.norm object
    """

    def error_func(params, series):
        mu, sigma = params
        cdf = Cdf.from_seq(series)
        ps = np.linspace(0.01, 0.99)
        qs = series.quantile(ps)
        error = cdf(qs) - norm.cdf(qs, mu, sigma)
        return error

    params = series.mean(), series.std()
    res = least_squares(error_func, x0=params, args=(series,), ftol=1e3)
    assert res.success
    mu, sigma = res.x
    return norm(mu, sigma)


def gaussian_model(series, iters=201):
    """Fit a Gaussian model with error bounds.

    series: Series of values
    iters: number of random samples to generate

    returns: Cdf and arrays for the error bounds
    """
    n = len(series)
    # mu, sigma = series.mean(), series.std()
    mu, sigma = fit_gaussian(series)
    cdf_series = make_cdf(series)

    a = np.empty(shape=(iters, len(cdf_series)))
    qs = cdf_series.qs

    for i in range(iters):
        sample = norm.rvs(mu, sigma, size=n)
        cdf = Cdf.from_seq(sample)
        a[i] = cdf(qs) * 100

    low, high = np.percentile(a, [2.5, 97.5], axis=0)
    return cdf_series, low, high


def gaussian_plot(series, plot_model=True, **options):
    """Plot a Gaussian model with error bounds.

    series: Series of values
    plot_model: boolean, whether to plot the error bounds
    options: passed to Cdf.plot
    """
    cdf, low, high = gaussian_model(series)
    if plot_model:
        plt.fill_between(cdf.qs, low, high, lw=0, color="gray", alpha=0.3)

    cdf.plot(**options)


def add_dist_seq(seq):
    """Distribution of sum of quantities from PMFs.

    seq: sequence of Pmf objects

    returns: Pmf
    """
    total = seq[0]
    for other in seq[1:]:
        total = total.add_dist(other)
    return total


def make_mixture(pmf, pmf_seq):
    """Make a mixture of distributions.

    pmf: mapping from each hypothesis to its probability
         (or it can be a sequence of probabilities)
    pmf_seq: sequence of Pmfs, each representing
             a conditional distribution for one hypothesis

    returns: Pmf representing the mixture
    """
    df = pd.DataFrame(pmf_seq).fillna(0).transpose()
    df *= np.array(pmf)
    total = df.sum(axis=1)
    return Pmf(total)


def make_uniform(qs, name=None, **options):
    """Make a Pmf that represents a uniform distribution.

    qs: quantities
    name: string name for the quantities
    options: passed to Pmf

    returns: Pmf
    """
    pmf = Pmf(1.0, qs, **options)
    pmf.normalize()
    if name:
        pmf.index.name = name
    return pmf


# INSPECTION


def kdeplot(sample, xs, label=None, **options):
    """Use KDE to plot the density function.

    sample: NumPy array
    xs: NumPy array
    label: string
    """
    density = gaussian_kde(sample).evaluate(xs)
    plt.plot(xs, density, label=label, **options)
    plt.yticks([])
    decorate(ylabel="Likelihood")


def make_lowess(series, frac=0.5):
    """Use LOWESS to compute a smooth line.

    series: pd.Series

    returns: pd.Series
    """
    endog = series.values
    exog = series.index.values

    smooth = lowess(endog, exog, frac)
    index, data = np.transpose(smooth)

    return pd.Series(data, index=index)


def plot_series_lowess(series, plot_series=False, frac=0.7, **options):
    """Plots a series of data points and a smooth line.

    series: pd.Series
    color: string or tuple
    """
    color = options.pop("color", "C0")
    if "label" not in options:
        options["label"] = series.name

    x = series.index
    y = series.values
    if plot_series:
        plt.plot(x, y, "o", color=color, alpha=0.3, label="_")

    if not plot_series and len(series) == 1:
        plt.plot(x, y, ".", color=color, alpha=0.6, label=options["label"])

    if len(series) > 1:
        smooth = make_lowess(series, frac=frac)
        smooth.plot(color=color, **options)


# NBUE


def percentile_rows(series_seq, ps):
    """Computes percentiles from aligned series.

    series_seq: list of sequences
    ps: cumulative probabilities

    returns: Series of x-values, NumPy array with selected rows
    """
    df = pd.concat(series_seq, axis=1).dropna()
    xs = df.index
    array = df.values.transpose()
    array = np.sort(array, axis=0)
    nrows, _ = array.shape

    ps = np.asarray(ps)
    indices = (ps * nrows).astype(int)
    rows = array[indices]
    return xs, rows


def plot_percentiles(series_seq, ps=None, label=None, **options):
    """Plot the low, median, and high percentiles.

    series_seq: sequence of Series
    ps: percentiles to use for low, medium and high
    label: string label for the median line
    options: options passed plt.plot and plt.fill_between
    """
    ps = ps if ps is not None else [0.05, 0.5, 0.95]
    assert len(ps) == 3

    xs, rows = percentile_rows(series_seq, ps)
    low, med, high = rows
    plt.plot(xs, med, alpha=0.5, label=label, **options)
    plt.fill_between(xs, low, high, linewidth=0, alpha=0.2, **options)


# NBUE


def remaining_lifetimes_pmf(pmf, qs=None):
    """Compute remaining lifetimes from a PMF.

    pmf: Pmf
    qs: quantities

    returns: Series that maps from ages to average remaining lifetimes
    """
    qs = np.linspace(0, pmf.qs.max(), 200) if qs is None else qs

    series = pd.Series(index=qs, dtype=float)
    for q in qs:
        conditional = Pmf(pmf[pmf.qs >= q])
        conditional.normalize()
        series[q] = conditional.mean() - q

    return series


def plot_remaining_lifetimes(
    pmf_model, surv_km, surv_low, surv_high, label="", data_label="", qs=None
):
    """Plot remaining lifetimes with confidence intervals.

    pmf_model: Pmf
    surv_km: Surv object
    surv_low, surv_high: lower and upper bounds of CI
    label: string label for the model
    data_label: string label for the data
    """
    if pmf_model is not None:
        series = remaining_lifetimes_pmf(pmf_model, qs)
        series.plot(color="C4", ls=":", label=label)

    series = remaining_lifetimes_pmf(surv_km.make_pmf(), qs)
    series.plot(ls="--", color="C1", label=data_label)

    series_low = remaining_lifetimes_pmf(surv_low.make_pmf(), qs)
    series_high = remaining_lifetimes_pmf(surv_high.make_pmf(), qs)
    plt.fill_between(series.index, series_low, series_high, color="C1", alpha=0.1)

    decorate(xlabel="Current survival time", ylabel="Average remaining survival time")


# LONGTAIL


def make_surv(seq):
    """Make a non-standard survival function, P(X>=x)"""
    pmf = Pmf.from_seq(seq)
    surv = pmf.make_surv() + pmf

    # correct for numerical error
    surv.iloc[0] = 1
    return Surv(surv)


def truncated_normal_pmf(qs, mu, sigma):
    ps = norm.pdf(qs, mu, sigma)
    pmf_model = Pmf(ps, qs)
    pmf_model.normalize()
    return pmf_model


def truncated_normal_sf(qs, mu, sigma):
    ps = norm.sf(qs, mu, sigma)
    surv_model = Surv(ps / ps[0], qs)
    return surv_model


def fit_truncated_normal(surv):
    low, high = surv.qs.min(), surv.qs.max()
    qs_model = np.linspace(low, high, 1000)

    ps = np.linspace(0.1, 0.99, 20)
    qs = surv.inverse(ps)

    def error_func(params):
        print(params)
        mu, sigma = params

        pmf_model = truncated_normal_pmf(qs_model, mu, sigma)
        error = surv(qs) - pmf_model.make_surv()(qs)
        return error

    pmf = surv.make_pmf()
    pmf.normalize()
    params = pmf.mean(), pmf.std()
    res = least_squares(error_func, x0=params, xtol=1e-3)
    assert res.success
    return res.x


def empirical_error_bounds(surv, n, qs, con_level=0.95):
    """Find the bounds on a normal CDF analytically."""
    # find the correct probabilities
    ps = surv.make_cdf()(qs)

    # find the upper and lower percentiles of
    # a binomial distribution
    p_low = (1 - con_level) / 2
    p_high = 1 - p_low

    low = binom.ppf(p_low, n, ps) / n
    low[ps == 1] = 1
    high = binom.ppf(p_high, n, ps) / n
    return 1 - low, 1 - high


def normal_error_bounds(dist, n, qs, con_level=0.95):
    """Find the bounds on a normal CDF analytically.

    dist: scipy.stats.norm object
    n: sample size
    qs: quantities
    alpha: fraction excluded from the CI

    returns: tuple of arrays (low, high)
    """
    # find the correct probabilities
    ps = dist.cdf(qs)

    # find the upper and lower percentiles of
    # a binomial distribution
    p_low = (1 - con_level) / 2
    p_high = 1 - p_low

    low = binom.ppf(p_low, n, ps) / n
    low[ps == 1] = 1
    high = binom.ppf(p_high, n, ps) / n
    return low, high


def plot_error_bounds(surv, n, **options):
    underride(options, linewidth=0, alpha=0.1, capstyle="round")
    qs = np.linspace(surv.qs.min(), surv.qs.max(), 100)
    low, high = empirical_error_bounds(surv, n, qs)

    # plt.plot(xs, med, alpha=0.5, label=label, **options)
    plt.fill_between(qs, low, high, **options)


def truncated_t_pmf(qs, df, mu, sigma):
    ps = t_dist.pdf(qs, df, mu, sigma)
    pmf_model = Pmf(ps, qs)
    pmf_model.normalize()
    return pmf_model


def truncated_t_sf(qs, df, mu, sigma):
    ps = t_dist.sf(qs, df, mu, sigma)
    surv_model = Surv(ps / ps[0], qs)
    return surv_model


def fit_truncated_t(df, surv):
    """Given df, find the best values of mu and sigma."""
    low, high = surv.qs.min(), surv.qs.max()
    qs_model = np.linspace(low, high, 1000)
    ps = np.linspace(0.01, 0.9, 20)
    qs = surv.inverse(ps)

    def error_func_t(params, df, surv):
        # print(params)
        mu, sigma = params
        surv_model = truncated_t_sf(qs_model, df, mu, sigma)

        error = surv(qs) - surv_model(qs)
        return error

    pmf = surv.make_pmf()
    pmf.normalize()
    params = pmf.mean(), pmf.std()
    res = least_squares(error_func_t, x0=params, args=(df, surv), xtol=1e-3)
    assert res.success
    return res.x


def minimize_df(df0, surv, bounds=[(1, 1e6)], ps=None):
    low, high = surv.qs.min(), surv.qs.max()
    qs_model = np.linspace(low, high * 1.2, 2000)

    if ps is None:
        t = surv.ps[0], surv.ps[-2]
        low, high = np.log10(t)
        ps = np.logspace(low, high, 30, endpoint=False)

    qs = surv.inverse(ps)

    def error_func_tail(params):
        (df,) = params
        # print(df)
        mu, sigma = fit_truncated_t(df, surv)
        surv_model = truncated_t_sf(qs_model, df, mu, sigma)

        errors = np.log10(surv(qs)) - np.log10(surv_model(qs))
        return np.sum(errors ** 2)

    params = (df0,)
    res = minimize(error_func_tail, x0=params, bounds=bounds, tol=1e-3, method="Powell")
    assert res.success
    return res.x


# SIMPSON


def get_regression_result(results, varname="x"):
    """Get regression results for a given variable.

    result: regression result object
    varname: string

    returns: list of param, pvalue, stderr, conf_int
    """
    param = results.params[varname]
    pvalue = results.pvalues[varname]
    conf_int = results.conf_int().loc[varname].values
    stderr = results.bse[varname]
    return [param, pvalue, stderr, conf_int]


def valid_group(group, yvarname, min_values=100, min_nonplurality=20):
    """Check if a group meets the criteria for running regression.

    group: DataFrame
    yvarname: string

    returns: boolean
    """
    # make sure we have enough values
    num_valid = group[yvarname].notnull().sum()
    if num_valid < min_values:
        return False

    # make sure all the answers aren't the same
    counts = group[yvarname].value_counts()
    most_common = counts.max()

    # make sure the most common answer isn't too dominant
    nonplurality = num_valid - most_common
    if nonplurality < min_nonplurality:
        return False

    return True


def prepare_yvar(df, yvarname, yvalue=None):
    """

    df: DataFrame
    yvarname: column in df
    yvalue: value, sequence of values, or "continuous"

    """
    yvar = df[yvarname]
    if yvalue == "continuous":
        df["y"] = yvar
        return

    counts = yvar.value_counts()

    # if yvalue is not provided, use the most common value
    if yvalue is None:
        yvalue = counts.idxmax()

    # replace yvalue with 1, everything else with 0
    d = counts.copy()
    d[:] = 0
    d[yvalue] = 1
    df["y"] = yvar.replace(d)


def chunk_series(df, xvarname, size=300):
    """Break a population into chunks and compute a mean for each chunk.

    Sort by `xvarname`, break into chunks, and compute the mean of column "y"

    df: DataFrame
    xvarname: variable
    size: chunk size

    returns: pd.Series
    """
    subset = df[[xvarname, "y"]].dropna().sort_values(by=xvarname).reset_index()
    subset["chunk"] = subset.index // size
    groupby = subset.groupby("chunk")
    x = groupby[xvarname].mean()
    y = groupby["y"].mean()

    return pd.Series(y.values, x)


def run_subgroups(gss, xvarname, yvarname, gvarname, yvalue=None):
    """ """
    if xvarname == yvarname:
        return False, False, False, 0

    is_continuous = (yvarname == "log_realinc") or (yvalue == "continuous")

    # prepare the y variable
    if is_continuous:
        gss["y"] = gss[yvarname]
        ylabel = yvarname
    else:
        # if discrete, code so `yvalue` is 1;
        # all other answers are 0
        yvar = gss[yvarname]
        counts = yvar.value_counts()

        # if yvalue is not provided, use the most common value
        if yvalue is None:
            yvalue = counts.idxmax()

        d = counts.copy()
        d[:] = 0
        d[yvalue] = 1
        gss["y"] = yvar.replace(d)
        ylabel = yvarname + "=" + str(yvalue)

    gss["x"] = gss[xvarname]

    # run the overall model
    formula = "y ~ x"
    if is_continuous:
        results = smf.ols(formula, data=gss).fit(disp=False)
    else:
        results = smf.logit(formula, data=gss).fit(disp=False)

    # start the DataFrame
    columns = ["param", "pvalue", "stderr", "conf_inf"]
    result_df = pd.DataFrame(columns=columns, dtype=object)
    result_df.loc["all"] = get_regression_result(results)

    # run the subgroups
    grouped = gss.groupby(gvarname)
    for name, group in grouped:
        if not valid_group(group, yvarname):
            continue

        if is_continuous:
            results = smf.ols(formula, data=group).fit(disp=False)
        else:
            results = smf.logit(formula, data=group).fit(disp=False)
        result_df.loc[name] = get_regression_result(results)

    result_df.ylabel = ylabel
    return result_df


xvarname_binned = {
    "log_realinc": "log_realinc10",
    "year": "year5",
    # "age": "age5",
    "cohort": "cohort10",
}


def make_pivot_table(df, xvarname, yvarname, gvarname, yvalue=None):
    """Make a table by subgroup and a series overall."""
    prepare_yvar(df, yvarname, yvalue)
    xbinned = xvarname_binned.get(xvarname, xvarname)

    factor = 1 if yvalue == "continuous" else 100
    series_all = df.groupby(xbinned)["y"].mean() * factor
    series_all.name = "all"

    table = (
        df.pivot_table(index=xbinned, columns=gvarname, values="y", aggfunc="mean")
        * factor
    )
    table.name = yvarname
    table.index.name = xbinned
    table.columns.name = gvarname

    return series_all, table


def make_table(df, xvarname, yvarname, gvarname, yvalue=None):
    """Compute chunk series of yvarname vs xvarname grouped by gvarname.

    yvalue: which value or values from yvarname to count

    returns: map from group name to Series
    """
    prepare_yvar(df, yvarname, yvalue)
    factor = 1 if yvalue == "continuous" else 100

    table = {}
    for name, group in df.groupby(gvarname):
        if len(group) < 500:
            continue
        series = chunk_series(group, xvarname) * factor
        series.name = name
        if len(series):
            table[name] = series

    return table


def get_colors(n, palette="Purples"):
    """Get a gradient palette of colors with the given size.

    n: number of colors
    palette: string name of palette
    """
    palette = sns.color_palette(palette, n + 1)
    return palette[1:]


def visualize_table(overall, table, **options):
    """Plot the results from make_table.

    overall: Series
    table: map from group name to Series
    options: passed to plot
    """
    palette = options.pop("palette", "Purples")
    colors = get_colors(len(table), palette=palette)

    for i, series in enumerate(table.values()):
        label = f"{series.name}s"
        plot_series_lowess(series, color=colors[i], label=label, **options)

    plot_series_lowess(overall, color="gray", ls=":", label="Overall", **options)


def label_table(table, nudge={}, frac=0.7, **options):
    """Add direct labels for the lines in a table.

    table: map from group name to Series
    nudge: map from label to an offset in the y direction
    frac: parameter of make_lowess
    options: passed to plt.text
    """
    underride(options, va="center", alpha=0.8, fontsize="small")
    for series in table.values():
        label = f"{series.name}s"

        x = series.index[-1] + 0.5
        if len(series) > 1:
            smooth = make_lowess(series, frac=frac)
            y = smooth.iloc[-1]
        else:
            y = series.iloc[-1]

        y += nudge.get(label, 0)
        plt.text(x, y, label, **options)


def label_table_left(table, nudge={}, frac=0.7, **options):
    """Add direct labels for the lines in a table.

    table: map from group name to Series
    nudge: map from label to an offset in the y direction
    frac: parameter of make_lowess
    options: passed to plt.text
    """
    underride(options, va="center", ha="right", alpha=0.8)
    for series in table.values():
        label = f"{series.name}s"

        x = series.index[0] - 0.5
        if len(series) > 1:
            smooth = make_lowess(series, frac=frac)
            y = smooth.iloc[0]
        else:
            y = series.iloc[0]

        y += nudge.get(label, 0)
        plt.text(x, y, label, **options)


def make_counterfact(results, varname, inter, data):
    """ """
    results_counter = deepcopy(results)
    results_counter.params[varname] = 0
    results_counter.params["Intercept"] = inter

    data["pred"] = results_counter.predict(data)
    pred = data.groupby("year")["pred"].mean()
    return pred


def decorate_table(**options):
    decorate(**options)
    anchor_legend(1.02, 1.02)


def compress_table(table):
    """Make the header just one line."""
    table.columns.name = table.index.name
    table.index.name = None
    return table


Gray20 = (0.162, 0.162, 0.162, 0.7)
Gray30 = (0.262, 0.262, 0.262, 0.7)
Gray40 = (0.355, 0.355, 0.355, 0.7)
Gray50 = (0.44, 0.44, 0.44, 0.7)
Gray60 = (0.539, 0.539, 0.539, 0.7)
Gray70 = (0.643, 0.643, 0.643, 0.7)
Gray80 = (0.757, 0.757, 0.757, 0.7)
Pu20 = (0.247, 0.0, 0.49, 0.7)
Pu30 = (0.327, 0.149, 0.559, 0.7)
Pu40 = (0.395, 0.278, 0.62, 0.7)
Pu50 = (0.46, 0.406, 0.685, 0.7)
Pu60 = (0.529, 0.517, 0.742, 0.7)
Pu70 = (0.636, 0.623, 0.795, 0.7)
Pu80 = (0.743, 0.747, 0.866, 0.7)
Bl20 = (0.031, 0.188, 0.42, 0.7)
Bl30 = (0.031, 0.265, 0.534, 0.7)
Bl40 = (0.069, 0.365, 0.649, 0.7)
Bl50 = (0.159, 0.473, 0.725, 0.7)
Bl60 = (0.271, 0.581, 0.781, 0.7)
Bl70 = (0.417, 0.681, 0.838, 0.7)
Bl80 = (0.617, 0.791, 0.882, 0.7)
Gr20 = (0.0, 0.267, 0.106, 0.7)
Gr30 = (0.0, 0.312, 0.125, 0.7)
Gr40 = (0.001, 0.428, 0.173, 0.7)
Gr50 = (0.112, 0.524, 0.253, 0.7)
Gr60 = (0.219, 0.633, 0.336, 0.7)
Gr70 = (0.376, 0.73, 0.424, 0.7)
Gr80 = (0.574, 0.824, 0.561, 0.7)
Or20 = (0.498, 0.153, 0.016, 0.7)
Or30 = (0.498, 0.153, 0.016, 0.7)
Or40 = (0.599, 0.192, 0.013, 0.7)
Or50 = (0.746, 0.245, 0.008, 0.7)
Or60 = (0.887, 0.332, 0.031, 0.7)
Or70 = (0.966, 0.475, 0.147, 0.7)
Or80 = (0.992, 0.661, 0.389, 0.7)
Re20 = (0.404, 0.0, 0.051, 0.7)
Re30 = (0.495, 0.022, 0.063, 0.7)
Re40 = (0.662, 0.062, 0.085, 0.7)
Re50 = (0.806, 0.104, 0.118, 0.7)
Re60 = (0.939, 0.239, 0.178, 0.7)
Re70 = (0.985, 0.448, 0.322, 0.7)
Re80 = (0.988, 0.646, 0.532, 0.7)


color_list = [
    Bl30,
    Or70,
    Gr50,
    Re60,
    Pu20,
    Gray70,
    Re80,
    Gray50,
    Gr70,
    Bl50,
    Re40,
    Pu70,
    Or50,
    Gr30,
    Bl70,
    Pu50,
    Gray30,
]
color_cycle = cycler(color=color_list)


def convert_color_list():
    for r, g, b, a in color_list:
        r, g, b = (np.array([r, g, b]) * 255).astype(int)
        s = "%02x%02x%02x" % (r, g, b)
        print(f"'{s}', ", end="")


def set_pyplot_params():
    # create a figure and then close it so the jupyter inline backend doesn't clobber
    # the figure size and resolution
    # see https://github.com/ipython/ipython/issues/11098
    f = plt.figure()
    plt.close(f)

    # for the book
    # plt.rcParams["figure.figsize"] = (3.835, 3.835 / 2)
    # plt.rcParams["font.size"] = 7.9
    # plt.rcParams["legend.fontsize"] = 7.4
    # plt.rcParams["axes.titlesize"] = 8.4
    # plt.rcParams["font.sans-serif"] = ["Source Sans Pro"]

    plt.rcParams["figure.figsize"] = 6, 4
    plt.rcParams["figure.dpi"] = 75

    plt.rcParams["axes.titlesize"] = "medium"
    # Not available on Colab
    # plt.rcParams["font.sans-serif"] = ["Roboto"]
    plt.rcParams["font.size"] = 12

    plt.rcParams["axes.prop_cycle"] = color_cycle
    plt.rcParams["lines.linewidth"] = 1.5

    plt.rcParams["axes.titlelocation"] = "left"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["xtick.top"] = False
    plt.rcParams["xtick.bottom"] = False
    plt.rcParams["ytick.left"] = False
    plt.rcParams["ytick.right"] = False

    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.4
    plt.rcParams["legend.facecolor"] = "none"
    plt.rcParams["legend.edgecolor"] = "0.8"

    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["lines.markeredgewidth"] = 0
    
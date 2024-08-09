import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy.stats import binom
from scipy.special import expit

from statsmodels.nonparametric.smoothers_lowess import lowess

# pip install whittaker-eilers
from whittaker_eilers import WhittakerSmoother


plt.rcParams["figure.dpi"] = 75
plt.rcParams["figure.figsize"] = [6, 3.5]

plt.rcParams["axes.titlesize"] = "medium"

plt.rcParams["font.size"] = 12

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

plt.rcParams["lines.markersize"] = 5
plt.rcParams["lines.markeredgewidth"] = 0


def get_colors(n, palette="Purples"):
    """Get a gradient palette of colors with the given size.

    n: number of colors
    palette: string name of palette
    """
    palette = sns.color_palette(palette, n + 1)
    return palette[1:]


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


def make_prior(std=0.1):
    """Make a mesh of alphas and betas and a prior distribution.
    
    std: standard deviation of the beta prior

    returns: mesh of alphas, mesh of betas, joint prior
    """
    alphas = np.linspace(-4, 4, 111)
    alpha_prior = norm.pdf(alphas, 0, 3)
    alpha_prior /= alpha_prior.sum()

    betas = np.linspace(-1, 1, 101)
    beta_prior = norm.pdf(betas, 0, std)
    beta_prior /= beta_prior.sum()

    A, B = np.meshgrid(alphas, betas, indexing="ij")
    AP, BP = np.meshgrid(alpha_prior, beta_prior, indexing="ij")
    prior = AP * BP
    return A, B, prior


def bayesian_smooth(df, xvar, yvar, yval):
    """Compute a smoothed fit based on Bayesian logistic regression.

    df: DataFrame
    xvar: string
    yvar: string
    yval: value to compare to

    returns: actual, pred
    """
    # prepare the independent and dependent variables
    m = df[xvar].median()
    df["x"] = df[xvar] - m
    df["y"] = df[yvar] == yval

    # compute the number of successes and failures for each value of x
    ks = df.groupby("x")["y"].sum()
    ns = df.groupby("x")["y"].count()
    xs = ns.index

    # compute the actual percentage of successes
    actual = (ks / ns) * 100
    actual.index = xs + m

    # compute the posterior distribution of A and B
    A, B, posterior = make_prior()
    for x in xs:
        P = expit(A + B * x)
        likelihood = binom.pmf(ks[x], ns[x], P)
        posterior *= likelihood

    # normalize the posterior distribution
    posterior /= posterior.sum()

    # compute the expected percentage of successes for each x
    ps = [np.sum(posterior * expit(A + B * x)) for x in xs]
    pred = pd.Series(ps, actual.index) * 100
    return actual, pred


def make_lowess(series, frac=0.5):
    """Use LOWESS to compute a smooth line.

    series: pd.Series
    frac: fraction of data to use in each neighborhood

    returns: pd.Series
    """
    endog = series.values
    exog = series.index.values

    smooth = lowess(endog, exog, frac)
    index, data = np.transpose(smooth)

    return pd.Series(data, index=index)


def make_smooth(series, weights=None, order=2, lam=1e2):
    """Use Whittaker-Eilers to compute a smooth line.

    series: pd.Series
    weights: pd.Series, should be normalized
    order: int, order of the smoother
    lam: float, smoothness parameter, higher is smoother

    returns: pd.Series
    """
    ys = series.values
    xs = series.index.values

    smoother = WhittakerSmoother(
        lmbda=lam,
        order=order,
        data_length=len(ys),
        x_input=xs,
        weights=weights,
    )

    smoothed = smoother.smooth(ys)
    return pd.Series(smoothed, index=xs)


def make_entry(df, xvar, yvar, yval):
    """Make a DataFrame with a series and a smooth line.
    
    df: DataFrame
    xvar: string
    yvar: string
    yval: value to compare to

    returns: DataFrame
    """
    entry = pd.DataFrame(dtype=float)

    subset = df.dropna(subset=[xvar, yvar])
    counts = subset[xvar].value_counts()

    # if we have 2-3 points, use Bayesian smoothing
    if 1 < len(counts) <= 3:
        series, smooth = bayesian_smooth(subset, xvar, yvar, yval)
        entry["series"] = series
        entry["smooth"] = smooth
        return entry

    # otherwise use Whittaker-Eilers
    xtab = pd.crosstab(subset[xvar], subset[yvar], normalize="index")
    entry["series"] = xtab[yval] * 100

    weights = df.groupby(xvar)[yvar].count()
    entry["weights"] = weights / weights.sum()

    entry["smooth"] = make_smooth(entry["series"], entry["weights"], lam=100)

    return entry


def plot_series_smooth(series, smooth, plot_series=False, **options):
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
        plt.plot(x, y, "o", color=color, alpha=0.5, label="_")

    if not plot_series and len(series) == 1:
        plt.plot(x, y, "o", color=color, label=options["label"])

    if len(series) > 1:
        smooth.plot(color=color, **options)


def plot_by_year(df, yvar, yval):
    """Plot percentages by year and a smoothed line.

    df: DataFrame
    yvar: string
    yval: value to compare to
    """
    entry = make_entry(df, "year", yvar, yval)
    plot_series_smooth(
        entry["series"], entry["smooth"], plot_series=True, color="C0", label=""
    )


def plot_by_cohort(df, yvar, yval):
    """Plot percentages by cohort and a smoothed line.
    
    df: DataFrame
    yvar: string
    yval: value to compare to
    """
    entry = make_entry(df, "cohort", yvar, yval)

    plot_series_smooth(
        entry["series"],
        entry["smooth"],
        plot_series=True,
        color="C1",
        ls="--",
        label="",
    )


def make_table(df, xvar, yvar, gvar, yval=None):
    """Make a table of entries for each group in a DataFrame.

    df: DataFrame
    xvar: string, column on the x axis
    yvar: string, column on the y axis
    gvar: string, column that identifies groups
    yval: value to compare to

    returns: map from group name to Series
    """
    table = {}
    for name, group in df.groupby(gvar):
        if len(group) < 100:
            continue

        table[name] = make_entry(group, xvar, yvar, yval)

    return table


def plot_table(table, nudge={}, **options):
    """Plot the results from make_table.

    table: map from group name to Series
    nudge: map from label to an offset in the y direction
    options: passed to plot
    """
    palette = options.pop("palette", "Purples")
    colors = get_colors(len(table), palette=palette)

    for i, group in enumerate(table):
        entry = table[group]
        label = f"{group}s"
        plot_series_smooth(
            entry['series'], entry['smooth'], plot_series=False, color=colors[i], label=label, **options
        )
    label_table(table, nudge)


def label_table(table, nudge={}, **options):
    """Add direct labels for the lines in a table.

    table: map from group name to DataFrame
    nudge: map from label to an offset in the y direction
    options: passed to plt.text
    """
    underride(options, va="center", alpha=0.8, fontsize="small")
    for group in table:
        series = table[group]["series"]
        smooth = table[group]["smooth"]
        label = f"{group}s"

        x = series.index[-1] + 0.5
        if len(series) > 1:
            y = smooth.iloc[-1]
        else:
            y = series.iloc[-1]

        y += nudge.get(label, 0)
        plt.text(x, y, label, **options)


def plot_group(table, group):
    """Plot the data for one group.
    
    table: map from group name to DataFrame
    group: string
    """
    entry = table[group]
    plot_series_smooth(
        entry["series"],
        entry["smooth"],
        plot_series=True,
        color="C2",
        label=f"Born in the {group}s",
    )


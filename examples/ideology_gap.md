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

# Is the Ideology Gap Growing?

[This tweet](https://twitter.com/jburnmurdoch/status/1750849189834022932) from John Burn-Murdoch links to an article in the Financial Times, ["A new global gender divide is emerging"](https://www.ft.com/content/29fd9b5c-2f35-41bf-9d4c-994db4e12998), which includes this figure:


![](https://github.com/AllenDowney/ProbablyOverthinkingIt/blob/book/data/ft_ideology_gap.png?raw=true)

<!-- #region -->
The article claims:

> In the US, Gallup data shows that after decades where the sexes were each spread roughly equally across liberal and conservative world views, women aged 18 to 30 are now 30 percentage points more liberal than their male contemporaries. That gap took just six years to open up.

The figure says it is based on General Social Survey data and the text says it's based on Gallup data, so I'm not sure which it is. UPDATE: In [this tweet](https://twitter.com/jburnmurdoch/status/1751656332380262418) Burn-Murdoch explains that the figure shows Gallup data, backfilled with GSS data from before the Gallup series began.

And I don't know what it means that "All figures are adjusted for time trend in the overall population". UPDATE: See the section below that explains the adjustment.

Anyway, since I used GSS data in the last three chapters of [*Probably Overthinking It*](https://greenteapress.com/wp/probably-overthinking-it/), this analysis did not sound right to me.

This notebook is my attempt to replicate the analysis with GSS data. I conclude:

* The GSS data does not look like the figure in the FT.

* Men are a more likely to say that they are conservative, by 5-10 percentage points.

* The only evidence that the gap is growing depends entirely on a data point from 2022 that is almost certainly wrong.

* If we drop the 2022 data and apply moderate smoothing, we see no evidence that the gap is growing.


Most of the functions in this notebook are the ones I used to write *Probably Overthinking It*. All of [the notebooks for that book are available in this repository](https://allendowney.github.io/ProbablyOverthinkingIt/intro.html).
<!-- #endregion -->

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ideology_gap.ipynb)

```python tags=["hide-cell"]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```python tags=["hide-cell"]
from statsmodels.nonparametric.smoothers_lowess import lowess


def make_lowess(series, frac=0.5):
    """Use LOWESS to compute a smooth line.

    series: pd.Series

    returns: pd.Series
    """
    y = series.values
    x = series.index.values

    smooth = lowess(y, x, frac=frac)
    index, data = np.transpose(smooth)

    return pd.Series(data, index=index)
```

```python tags=["hide-cell"]
def plot_series_lowess(series, color, plot_series=True, frac=0.5, **options):
    """Plots a series of data points and a smooth line.

    series: pd.Series
    color: string or tuple
    """
    if "label" not in options:
        options["label"] = series.name

    if plot_series or len(series) == 1:
        x = series.index
        y = series.values
        plt.plot(x, y, "o", color=color, alpha=0.3, label="_")

    if not plot_series and len(series) == 1:
        x = series.index
        y = series.values
        plt.plot(x, y, "o", color=color, alpha=0.6, label=options["label"])

    if len(series) > 1:
        smooth = make_lowess(series, frac=frac)
        smooth.plot(color=color, **options)
```

```python tags=["hide-cell"]
def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()
```

## Read the Data

I'm using data I previous cleaned in [this notebook](https://github.com/AllenDowney/GssExtract/blob/main/notebooks/01_make_pacs_extract.ipynb).

```python tags=["remove-cell"]
from os.path import basename, exists
from pathlib import Path


def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + str(local))
    return filename
```

```python tags=["remove-cell"]
download("https://github.com/AllenDowney/GssExtract/raw/main/data/interim/gss_pacs_2022.hdf")
```

```python tags=["hide-cell"]
gss = pd.read_hdf("gss_pacs_2022.hdf", "gss")
gss.shape
```

The primary variable we'll use is [polviews](https://subsetdataexplorer.norc.org/variables/178/vshow), which asks

> We hear a lot of talk these days about liberals and conservatives. I'm going to show you a seven-point scale on which the political views that people might hold are arranged from extremely liberal--point 1--to extremely conservative--point 7. Where would you place yourself on this scale?

The points on the scale are Extremely liberal, Liberal, and Slightly liberal; Moderate;	Slightly conservative, Conservative, and Extremely conservative.


I'll clean `polviews`, replacing cases where the question was not asked or the respondent did not answer.

```python tags=["hide-cell"]
gss['polviews'].replace([0, 8, 9], np.nan, inplace=True)
```

To confirm that my cleaning did not introduce errors, I compared the following cross-tabulation with the one in the [GSS Explorer](https://subsetdataexplorer.norc.org/variables/178/vshow).

```python tags=["hide-input"]
year = gss["year"]
column = gss["polviews"]
xtab = pd.crosstab(column, year)
xtab
```

I'll lump the first three points into "Liberal" and the last three into "Conservative".

```python tags=["hide-cell"]
recode_polviews = {
    1: "Liberal",
    2: "Liberal",
    3: "Liberal",
    4: "Moderate",
    5: "Conservative",
    6: "Conservative",
    7: "Conservative",
}
```

```python tags=["hide-cell"]
gss["polviews3"] = gss["polviews"].replace(recode_polviews)
gss["polviews3"].value_counts()
```

```python tags=["hide-cell"]
def make_diff(df):
    """Compute % liberal minus % conservative.
    """
    year = df["year"]
    column = df["polviews3"]

    xtab = pd.crosstab(year, column, normalize='index')
    diff = xtab['Liberal'] - xtab['Conservative']
    
    return diff * 100
```

```python tags=["hide-cell"]
def make_mean(df):
    """Compute % liberal minus % conservative.
    """
    series = df.groupby('year')['polviews'].mean()
    
    return series
```

```python tags=["hide-cell"]
def decorate_plot(title):
    decorate(xlabel='Year',
             ylabel='% liberal - % conservative',
             title=title)
```

## Make the plot with all respondents

The following functions generate a figure like the one in the FT.

```python tags=["hide-cell"]
def make_plot(df, title=''):
    """Plot % liberal - % conservative for male and female respondents.
    """
    male = df.query('sex==1')
    female = df.query('sex==2')

    diff_male = make_diff(male)
    diff_female = make_diff(female)

    plot_series_lowess(diff_male, color='C0', label='Male')
    plot_series_lowess(diff_female, color='C1', label='Female')
    decorate_plot(title)
```

Generate the plot for all respondents.

```python tags=["remove-cell"]
def savefig(filename, **options):
    if 'dpi' not in options:
        options['dpi'] = 300
    plt.savefig(filename, **options)
```

```python tags=["hide-input"]
make_plot(gss, 'All respondents')
savefig('ideology_gap1.png')
```

In the general population, men are more likely to say they are conservative by 5-10 percentage points. 

The gap might have increased in the most recent data, depending on how seriously we take the last two points in a noisy series.


## Just young people

Now let's select just people under 30.

```python tags=["hide-cell"]
subset = gss.query('age < 30')
subset.shape
```

And make the same figure.

```python tags=["hide-input"]
make_plot(subset, 'Age < 30')
savefig('ideology_gap2.png')
```

The trends here are pretty much the same as in the general population.
Men are more likely to say they are conservative, by 5-10 percentage points.

It's possible that the gap has grown in the most recent data, but the evidence is weak and depends on how we draw a smooth curve through noisy data.

Anyway, there is no evidence the trend for men is going down, and the gap in the most recent data is nowhere near 30 percentage points.


Here are the sample sizes.

```python tags=["hide-input"]
year = subset["year"]
column = subset["sex"]
xtab = pd.crosstab(column, year)
xtab
```

## With Sampling Weights

In the previous figures, I have not taken into account the sampling weights.
I didn't expect them to make much difference, and they don't except for men in 2022 -- and as we'll see, there is almost certainly something wrong with that data point.

```python tags=["hide-cell"]
male = subset.query('sex==1')
female = subset.query('sex==2')

diff_male = make_diff(male)
diff_female = make_diff(female)
```

We only have weighted data since 1988, since that's how far back the `wtssps` variable goes.



```python tags=["hide-cell"]
recent = gss.dropna(subset=['wtssps']).query('age < 30')
```

```python tags=["hide-cell"]
def resample_rows_weighted(df, column):
    """Resamples a DataFrame using probabilities proportional to given column.

    df: DataFrame
    column: string column name to use as weights

    returns: DataFrame
    """
    weights = df[column]
    sample = df.sample(n=len(df), replace=True, weights=weights)
    return sample
```

```python tags=["hide-cell"]
def resample_by_year(df, column):
    """Resample rows within each year.

    df: DataFrame
    column: string name of weight variable

    returns DataFrame
    """
    grouped = df.groupby("year")
    samples = [resample_rows_weighted(group, column) for _, group in grouped]
    sample = pd.concat(samples, ignore_index=True)
    return sample
```

```python tags=["hide-cell"]
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
```

```python tags=["hide-cell"]
def plot_percentiles(series_seq, ps=None, label=None, **options):
    """Plot the low, median, and high percentiles.

    series_seq: sequence of Series
    ps: percentiles to use for low, medium and high
    label: string label for the median line
    options: options passed plt.plot and plt.fill_between
    """
    if ps is None:
        ps = [0.05, 0.5, 0.95]
    assert len(ps) == 3

    xs, rows = percentile_rows(series_seq, ps)
    low, med, high = rows
    plt.plot(xs, med, alpha=0.5, label=label, **options)
    plt.fill_between(xs, low, high, linewidth=0, alpha=0.2, **options)
```

```python tags=["hide-cell"]
def resample_diffs(df, query, iters=101):
    diffs = []
    for i in range(iters):
        sample = resample_by_year(df, "wtssps").query(query)
        diff = make_diff(sample)
        diffs.append(diff)
    return diffs
```

```python tags=["hide-cell"]
diffs_male = resample_diffs(recent, 'sex==1')
diffs_female = resample_diffs(recent, 'sex==2')
```

This figure shows the median of 101 resamplings and a 90% CI, along with the unweighted data.

```python tags=["hide-input"]
plot_percentiles(diffs_male)
plot_percentiles(diffs_female)

diff_male.plot(style='.', color='C0', label='Male')
diff_female.plot(style='.', color='C1', label='Female')

decorate_plot('Age < 30 with sampling weights')
savefig('ideology_gap3.png')
```

In most cases, the unweighted data falls in the CI of the weighted data, but for male respondents in 2022, the weighting moves the needle by almost 30 percentage points.

So something is not right there. I think the best option is to drop the 2022 data, but just for completeness, let's see what happens if we apply some smoothing.


## Resampling and smoothing

```python tags=["hide-cell"]
def resample_diffs_smooth(df, query, iters=101):
    diffs = []
    for i in range(iters):
        sample = resample_by_year(df, "wtssps").query(query)
        diff = make_diff(sample)
        smooth = make_lowess(diff)
        diffs.append(smooth)
    return diffs
```

```python tags=["hide-cell"]
diffs_male = resample_diffs_smooth(recent, 'sex==1')
diffs_female = resample_diffs_smooth(recent, 'sex==2')
```

Here's a version of the same plot with moderate smoothing, and dropping the unweighted data.

```python tags=["hide-input"]
plot_percentiles(diffs_male, label='Male')
plot_percentiles(diffs_female, label='Female')

decorate_plot('Age < 30 with sampling weights, smoothed')
savefig('ideology_gap4.png')
```

You could make the argument that this figure shows evidence for an increasing gap, but the error bounds are very wide, and as we'll see in the next figure, the entire effect is due to the likely error in the 2022 data.


## Resampling and smoothing without 2022

Finally, here's the analysis I think is the best choice, dropping the 2022 data for both men and women.

```python tags=["hide-cell"]
pre2022 = recent.query('year < 2022')
```

```python tags=["hide-cell"]
diffs_male = resample_diffs_smooth(pre2022, 'sex==1')
diffs_female = resample_diffs_smooth(pre2022, 'sex==2')
```

```python tags=["hide-input"]
plot_percentiles(diffs_male, label='Male')
plot_percentiles(diffs_female, label='Female')

decorate_plot('Age < 30 with sampling weights, smoothed, no 2022')
savefig('ideology_gap5.png')
```

Since the 1990s, both men and women have become more likely to identify as liberal.

Men are more likely to identify as conservative by 5-10 percentage points.

But this figure shows no evidence that the ideology gap is growing.


## Adjusting

In [this tweet](https://twitter.com/jburnmurdoch/status/1751569229516550403), Burn-Murdoch explains that the adjustment mentioned in the figure is to subtract off the overall trend.

Here's the overall trend, not including the 2022 data.

```python tags=["hide-input"]
diff_overall = make_diff(pre2022)

plot_series_lowess(diff_overall, color='C2', label='Overall')
decorate_plot('Overall')
```

```python tags=["hide-cell"]
def resample_diffs_smooth_adjusted(df, query, iters=101):
    diff_overall = make_diff(df)
    diff_overall_smooth = make_lowess(diff_overall)

    diffs = []
    for i in range(iters):
        sample = resample_by_year(df, "wtssps").query(query)
        diff = make_diff(sample)
        smooth = make_lowess(diff) - diff_overall_smooth
        diffs.append(smooth)
    return diffs
```

```python tags=["hide-cell"]
diffs_male = resample_diffs_smooth_adjusted(pre2022, 'sex==1')
diffs_female = resample_diffs_smooth_adjusted(pre2022, 'sex==2')
```

Here's what the adjusted data looks like.

```python tags=["hide-input"]
plot_percentiles(diffs_male, label='Male')
plot_percentiles(diffs_female, label='Female')

decorate_plot('Age < 30 with sampling weights, smoothed, no 2022, adjusted')
savefig('ideology_gap6.png')
```

This figure gives a stronger visual sense that the gap is growing, but even then, it is probably not more than 10 percentage points, and smaller than it was in the 1980s.

This way of showing the data makes it seem as if men are increasingly likely to say they are conservative, which is misleading.
They are increasingly likely to say they are liberal, but not increasing as fast as the overall average.


## What's wrong with 2022?

I don't know yet, but I'll add some explorations here.

For one thing, the magnitudes of the weights are substantially different than in previous years, which is why they are able to drag the results so far.

```python tags=["hide-input"]
subset.query('year>2010 and sex==2').groupby('year')['wtssall'].describe()
```

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

# Are You Normal? Hint: No.

<!-- #region tags=["remove-cell"] -->
This notebook contains the text from Chapter 1 of *[Probably Overthinking It: How to Use Data to Answer Questions, Avoid Statistical Traps, and Make Better Decisions](https://greenteapress.com/wp/probably-overthinking-it)* (University of Chicago Press, 2023), along with the supporting code used in the analysis.
<!-- #endregion -->

*Probably Overthinking It* is available from [Bookshop.org](https://bookshop.org/a/98697/9780226822587) and [Amazon](https://amzn.to/3Kp629E) (affiliate links).

If you like *Probably Overthinking It*, you'll like my book in progress, [*You Would Choose Now*](https://allendowney.com/ywcn), which uses decades of survey data to track changing attitudes in the United States. The early access edition is available now.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/gaussian.ipynb).

```python tags=["remove-cell"]
# Install empirical dist if we don't already have it

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["remove-cell"]
# download utils.py

from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)
        
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/notebooks/utils.py")
```

```python tags=["hide-cell"]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import set_pyplot_params

set_pyplot_params()
```

What does it mean to be normal? And what does it mean to be weird?
I think there are two factors that underlie our intuition for these ideas:

* "Normal" and "weird" are related to the idea of average. If by some measurement you are close to the average, you are normal; if you are far from the average, you are weird.

* "Normal" and "weird" are also related to the idea of rarity. If some ability or characteristic of yours is common, it is normal; if it's rare, it's weird.

Intuitively, most people think that these things go together; that is, we expect measurements close to the average to be common, and measurements far from average to be rare.

For many things this intuition is valid. 
For example, the average height of adults in the United States is about 170 cm.
Most people are close to this average: about 64\% of adults are within 10 cm plus or minus; 93\% are within 20 cm.
And few people are far from average: only 1\% of the population is shorter than 145 cm or taller than 195 cm.

missing

### Technical note

The numbers in the previous paragraph come from the [Behavioral Risk Factor Surveillance System](https://www.cdc.gov/brfss/index.html) (BRFSS).
Here is the [notebook I used to download and clean the data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_brfss.ipynb).

I generated a sample from the dataset, which we can download like this:

```python tags=["remove-cell"]
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/brfss_sample.hdf")
```

```python tags=["hide-cell"]
brfss = pd.read_hdf("brfss_sample.hdf", "brfss")
height = brfss["HTM4"]
height.describe()
```

```python tags=["hide-cell"]
from empiricaldist import Cdf

cdf_height = Cdf.from_seq(height)
cdf_height([150, 190])
```

```python tags=["hide-cell"]
cdf_height.inverse([0.01, 0.99])
```

```python tags=["hide-cell"]
from scipy.stats import norm

mu, sigma = height.mean(), height.std()
dist = norm(mu, sigma)

np.diff(dist.cdf([160, 180]))
```

```python tags=["hide-cell"]
np.diff(dist.cdf([150, 190]))
```

```python tags=["hide-cell"]
dist.ppf([0.01, 0.99])
```

But what is true when we consider a single characteristic turns out to be partly false when we consider a few characteristics at once, and spectacularly false when we consider more than a few.
In fact, when we consider the many ways people are different, we find that 

* People near the average are rare or non-existent,

* Everyone is far from average, and

* Everyone is roughly the same distance from average.

At least in a mathematical sense, no one is normal, everyone is weird, and everyone is the same amount of weird.

To show why this is true, we'll start with a single measurement and work our way up to hundreds, and then thousands.
I'll introduce the Gaussian curve, and we'll see that it fits many of these measurements remarkably well.

In addition to physical measurements from a survey of military personnel, we'll also consider psychological measurements, using survey results that quantify the Big Five personality traits.


## Present... Arms

How tall are you? How long are your arms? How far is it from the radiale landmark on your right elbow to the stylion landmark on your right wrist?

You might not know that last one, but the U.S. Army does. Or rather, they know the answer for the 6068 members of the armed forces they measured at the Natick Soldier Center (just a few miles from my house) as part of the Anthropometric Surveys of 2010-2011, which is abbreviated army-style as ANSUR-II.

In addition to the radiale-stylion length of each participant, the ANSUR dataset includes 93 other measurements "chosen as the most useful ones for meeting current and anticipated Army and [Marine Corps] needs."
The results were declassified in 2017 and are available to download.

We will explore all of the measurements in this dataset, starting with height.
The following figure shows the distribution of heights for the male and female participants in the survey.

Here's the ANSUR data, originally downloaded from [The OPEN Design Lab](https://www.openlab.psu.edu/ansur2/).

```python tags=["remove-cell"]
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/ANSURIIFEMALEPublic.csv")
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/ANSURIIMALEPublic.csv")
```

```python tags=["hide-cell"]
ansur_female = pd.read_csv("ANSURIIFEMALEPublic.csv", encoding="ISO-8859-1")
ansur_female.shape
```

```python tags=["hide-cell"]
ansur_male = pd.read_csv("ANSURIIMALEPublic.csv", encoding="ISO-8859-1")
ansur_male.shape
```

```python tags=["hide-input"]
ansur_female.head()
```

<!-- #region tags=["remove-cell"] -->
[*The Measurer’s Handbook*](http://tools.openlab.psu.edu/publicData/ANSURII-TR11-017.pdf) used in the ANSUR survey is 253 pages,
and includes details on "sampling strategy and measuring techniques"
as well as descriptions and diagrams of the landmarks and measurements between them.


<!-- #endregion -->

Since heights are in mm, I converted to cm.

```python tags=["hide-cell"]
height_male = ansur_male["stature"] / 10
height_female = ansur_female["stature"] / 10
```

The following function plots histograms using `plt.stairs`.

```python tags=["hide-cell"]
from utils import underride


def histplot(series, fill=True, **options):
    """Plot a histogram.
    
    series: Pandas Series
    fill: boolean, whether to fill below the curve
    options: passed to plt.stairs
    """
    underride(
        options,
        color="C0",
        alpha=0.7,
        lw=1.5,
    )
    bins = options.pop("bins", 40)
    counts, bins = np.histogram(series, bins=bins)
    counts = counts / counts.sum() * 100
    plt.stairs(counts, bins, **options)

    if fill:
        options.update(dict(label="", alpha=0.1))
        plt.stairs(counts, bins, fill=True, **options)
```

Here are the histograms of male and female heights.

```python tags=["hide-input"]
from utils import decorate

histplot(height_female, color="C1", label="Female")
histplot(height_male, color="C0", ls="--", label="Male")

decorate(
    xlabel="Adult Height (cm)",
    ylabel="Percent",
    title="Distributions of height in the U.S. Army",
)
```

The vertical axis shows the percentage of participants whose height falls in each one-centimeter interval.
Both distributions have the characteristic shape of the "bell curve".

One of the earliest discoveries in the history of statistics is that there is a simple model that matches the shape of these curves with remarkable accuracy.
Written as a mathematical formula it might not seem simple, depending on how you feel about the Greek alphabet.
So instead I will describe it as a game.

Please think of a number (but keep it small) and let's call it $x$.
Then follow these steps:

1. Square your number. For example, if you chose 2, the result is 4.

2. Take the result from the previous step and raise 10 to that power. If you chose 2, the result is $10^{4}$, which is 10,000.

3. Take that result and invert it. If you chose 2, the result is $1 / 10,000$.

Now I'll do the same calculation for a range of values from -2 to 2 and plot the results. 
Here's what it looks like:

```python tags=["hide-input"]
xs = np.linspace(-2, 2, 101)
ys = 1 / 10 ** (xs * xs)

plt.plot(xs, ys)
decorate(xlabel="Your number", ylabel="The result", title="A Gaussian curve")
```

This result is known as the Gaussian curve, after the mathematician Carl Friedrich Gauss.
Actually, it's just one of many Gaussian curves.
By changing the rules of the game we just played, we can shift the curve to the right or left, and make it wider or narrower.
And by shifting and stretching, we can often find a Gaussian curve that's a good match for real data.

In the following figure, the shaded areas show the distributions of height again; the lines show Gaussian curves I chose to match the data.


```python tags=["hide-cell"]
from scipy.stats import norm


def plot_normal_pdf(series, scale, **options):
    """Plot the PDF of a Gaussian distribution.
    
    series: Pandas series
    scale: multiplier used to scale the PDF
    options: passed to plt.plot
    """
    mu, sigma = series.mean(), series.std()
    qs = np.linspace(series.min(), series.max())
    ps = norm.pdf(qs, mu, sigma)
    ps *= scale / np.max(ps)
    plt.plot(qs, ps, alpha=0.8, **options)
```

The following function plots a Gaussian curve with the same mean and std as the data.


```python tags=["hide-cell"]
from scipy.stats import norm


def plot_normal_pdf(series, scale, **options):
    """Plot the PDF of a Gaussian distribution.
    
    series: Pandas series
    scale: multiplier used to scale the PDF
    options: passed to plt.plot
    """
    mu, sigma = series.mean(), series.std()
    qs = np.linspace(series.min(), series.max())
    ps = norm.pdf(qs, mu, sigma)
    ps *= scale / np.max(ps)
    plt.plot(qs, ps, alpha=0.8, **options)
```

The scale parameter is arbitrary -- I chose it to be roughly the same height as the histogram, which is also arbitrary. Avoiding this arbitrariness is one reason I prefer to use CDFs to compare the distribution of data with a model.

```python tags=["hide-input"]
histplot(height_female, color="C1", lw=0, alpha=0.3)
histplot(height_male, color="C0", lw=0, alpha=0.2)
plot_normal_pdf(height_female, 6.5, color="C1", label="Female")
plot_normal_pdf(height_male, 7.5, color="C0", ls="--", label="Male")

decorate(
    xlabel="Adult Height (cm)",
    ylabel="Percent",
    title="Distributions of height compared to Gaussian curves",
)
```

And they fit the data pretty well.
If you have seen results like this before, they might not surprise you; but maybe they should.

Mathematically, the Gaussian curve is simple. The operations we used to compute it -- squaring, raising to a power, and inverting numbers -- are familiar to people with a basic math education.

In contrast, people are complicated. Nearly everything about us is affected by hundreds or thousands of interacting causes and effects, including our genes and everything about our environment from conception to adulthood.

So if we measure a few thousand people and plot the results, and find that they fit a simple model so well, we should be surprised.

Furthermore, it's not just people and it's not just height.
If you choose individuals from almost any species and measure almost any part of them, the distribution of measurements will be approximately Gaussian.
In many cases, the approximation is remarkably close.
Naturally, you might wonder why.


## Why?

The answer comes in three parts:

* Physical characteristics like height depend on many factors, both genetic and environmental.

* The contribution of these factors tends to be additive; that is, the measurement is the sum of many contributions.

* In a randomly-chosen individual, the set of factors they have inherited or experienced is effectively random.

When we add up the contributions of these random factors, the resulting distribution tends to follow a Gaussian curve.


To show that that's true, I will use a model to simulate random factors, generate a set of random heights, and compare them to the measured heights of the participants in the ANSUR dataset.

In my model, there are 20 factors that influence height.
That's an arbitrary choice; it would also work with more or fewer.

For each factor, there are two possibilities, which you can think of as different genes or different environmental conditions.
To model an individual, I generate a random sequence of 0s and 1s that indicate the absence or presence of each factor.
For example:

* If there are two forms of a gene (known as alleles), 0 might indicate one of the alternatives, and 1 the other.

* If a particular nutrient contributes to height at some point in development, 0 and 1 might indicate the deficiency or sufficiency of the nutrient.  

* If a particular infection can detract from height, 0 and 1 might indicate the absence or presence of the infectious agent.


In the model, the contribution of each factor is a random value between -3 cm and 3 cm; that is, each factor causes a person to be taller or shorter by a few centimeters.

To simulate a population, I generate random factors (0s and 1s) for each individual, look up the contribution of each factor and add them up.
The result is a sample of simulated heights, which we can compare to the actual heights in the dataset.

The following figure shows the results.
The shaded areas show the distributions of the actual heights again; the lines show the distributions of the simulated heights.

The following function generates simulated values that are the sum of a number of random factors, shifted and scaled so they have the same mean and std as the given data.

```python tags=["hide-cell"]
def run_simulation(series):
    """Simulate the sum of randomly-selected effects.
    
    series: Pandas Series
    
    returns: Series of simulated values
    """
    # generate the effect of each coin toss
    num_effects = 20
    effects = np.random.uniform(-3, 3, size=(2, num_effects))

    # toss coins and add up the effects
    flips = np.random.choice([True, False], size=(len(series), num_effects))
    totals = np.where(flips, effects[0], effects[1]).sum(axis=1)

    # shift and scale the results to match the mean and std of the given series
    sim = (totals - totals.mean()) / totals.std()
    sim *= series.std()
    sim += series.mean()
    return sim
```

```python tags=["hide-cell"]
np.random.seed(10)
sim_female = run_simulation(height_female)
sim_male = run_simulation(height_male)
```

The result shows that values generated this way can resemble the actual data.

```python tags=["hide-input"]
histplot(height_female, color="C1", alpha=0.3, lw=0)
histplot(sim_female, fill=False, color="C1", alpha=0.8, label="Female")

histplot(height_male, color="C0", alpha=0.2, lw=0)
histplot(sim_male, fill=False, color="C0", alpha=0.8, ls="--", label="Male")

decorate(
    xlabel="Adult Height (cm)",
    ylabel="Percent",
    title="Distributions of height compared to simulation results",
)
```

The results from the simulation are a good fit for the data.
Before I ran this simulation, I had a pretty good idea what the results would look like because of the Central Limit Theorem, which states that the sum of a large number of random values follows a Gaussian distribution.
Mathematically, the theorem is only true if the random values come from the same distribution and they are not correlated with each other.

Of course, genetic and environmental factors are more complicated than that.
In reality, some contributions are bigger than others, so they don't all come from the same distribution.
Also, they are likely to be correlated with each other.
And their effects are not purely additive; they can interact with each other in more complicated ways.


However, even when the requirements of the Central Limit Theorem are not met exactly, the combined effect of many factors will be approximately Gaussian as long as:

* None of the contributions are much bigger than the others,

* The correlations between them are not too strong,

* The total effect is not too far from the sum of the parts.

Many natural systems satisfy these requirements, which is why so many distributions in the world are approximately Gaussian.


## Comparing Distributions

So far, I have represented distributions using histograms, which show a range of possible values on the horizontal axis and percentages on the vertical axis.
Before we go on, I want to present another way to visualize distributions that is particularly good for making comparisons.
This visualization is based on **percentile ranks**.

If you have taken a standardized test, you are probably familiar with percentile ranks.  
For example, if your percentile rank is 75, that means that you did as well as or better than 75\% of the people who took the test.

Or if you have a child, you might be familiar with percentile ranks from a pediatric growth chart.
For example, a 2-year-old boy who weighs 11 kg has percentile rank 10, which means he weighs as much or more than 10\% of children his age. 

Computing percentile ranks is not hard.
For example, if a female participant in the ANSUR survey is 160 cm tall, she is as tall or taller than 34\% of the female participants, so her percentile rank is 34.
If a male participant is 180 cm, he is as tall or taller than 75\% of the male participants, so his percentile rank is 75.

```python tags=["hide-cell"]
(height_female <= 160).mean() * 100
```

```python tags=["hide-cell"]
(height_male <= 180).mean() * 100
```

If we compute the percentile rank for each participant the same way, and plot these percentile ranks on the vertical axis, the results look like this:

```python tags=["hide-cell"]
from empiricaldist import Pmf, Cdf
from utils import make_cdf
```

```python tags=["hide-input"]
def plot_lookup(series, percentile_rank, **options):
    """Plot a line representing a lookup in a CDF.
    
    series: Pandas Series
    percentile_rank: which percentile to look up
    options: passed to plt.plot
    """
    options = underride(options, ls=":", color="gray", alpha=0.5)
    ymax = percentile_rank
    xmin = 140
    xmax = np.percentile(series.dropna(), percentile_rank)
    plt.plot([xmin, xmax, xmax], [ymax, ymax, 0], **options)

make_cdf(height_female).step(color="C1", label="Female")
plot_lookup(height_female, 33)

make_cdf(height_male).step(ls="--", label="Male")
plot_lookup(height_male, 75)

decorate(
    xlabel="Height (cm)",
    ylabel="Percentile rank",
    xlim=[139, 201],
    title="Distributions of height as CDFs",
    loc="lower right",
)
```

This way of representing a distribution is called a cumulative distribution function (CDF).
In this example, the solid curve shows the CDF of height for female participants; the dashed curve is the CDF for male participants.

The dotted lines indicate the height and percentile rank for two examples: in the CDF for female participants, height 160 cm corresponds to percentile rank 34; in the CDF for male participants, height 180 cm corresponds to percentile rank 75.

It might not be obvious yet why this way of plotting distributions is useful. 
The primary reason, in my opinion, is that it provides a good way to compare distributions.

<!-- #region tags=["remove-cell"] -->
For example, in the previous section, we compared the actual distribution of heights to the results from a simulation.
The two distributions are similar, but they differ in a few places.
It's hard to tell, visually, whether these differences are meaningful, or whether they might be due to random variation.
When we use a CDFs to compare distributions, they smooth out random variations.
At the same time, when the differences are substantial, the CDFs show them clearly.

The following figure shows the distributions of height and the values from the simulation, as in the previous section, this time represented with CDFs.
<!-- #endregion -->

```python tags=["hide-input"]
make_cdf(sim_female).plot(ls=":", color="gray", label="Simulation results")
make_cdf(height_female).step(color="C1", label="Female participants")

make_cdf(sim_male).plot(ls=":", color="gray")
make_cdf(height_male).step(ls="--", label="Male participants")

decorate(
    xlabel="Height (cm)",
    ylabel="Percentile rank",
    title="CDFs of height compared to simulation results",
    loc="lower right",
)
```

<!-- #region tags=["remove-cell"] -->
The dotted lines show the results from the simulations.
For both male and female participants, the simulations are similar to the data; we can barely see any space between them.
<!-- #endregion -->

For example, in the following figure, the lines show the CDFs of height again.
The shaded areas show the CDFs of Gaussian distributions I chose to fit the data.

```python tags=["hide-input"]
from utils import gaussian_plot

gaussian_plot(height_female, color="C1", label="Female participants")
gaussian_plot(height_male, style="--", label="Male participants")

decorate(
    xlabel="Height (cm)",
    ylabel="Percentile rank",
    title="CDFs of height compared to Gaussian curves",
    loc="lower right",
)
```

The width of the shaded areas shows how much variation we expect if we use the Gaussian model to generate simulated heights.
Where a line falls inside the shaded area, the data are consistent with the Gaussian model.
If it falls outside the shaded area, that indicates a deviation bigger than we would expect to see due to random variation.
In these examples, both curves fall within the bounds of variation we expect.

Before we go on, I want to say something about the word "deviation" in this context, which is used to mean a difference between the data and the model.
There are two ways to think about "deviation".
One, which is widespread in the history of statistics and natural philosophy, is that the model represents some kind of ideal, and if the world fails to meet this ideal, the fault lies in the world, not the model.

In my opinion, this is nonsense.
The world is complicated.
Sometimes we can describe it with simple models, and it is often useful when we can.
And sometimes we can find a reason the model fits the data, which can help explain why the world is as it is.
But when the world deviates from the model, that's a problem for the model, not a deficiency of the world.

Having said all that, let's see how big those deviations are.


## How Gaussian Is It?

Out of 94 measurements in the ANSUR dataset, 93 are well modeled by a Gaussian distribution; one is not.
In this chapter we'll explore the well behaved measurements.
In Chapter~\ref{extremes-outliers-and-goats} I'll reveal the exception.

For each measurement, I chose the Gaussian distribution that best fits the data and computed the maximum vertical distance between the CDF of the data and the CDF of the model.
Using the size of this deviation, I identified the measurements that are the most and the least Gaussian.

For many of the measurements, the distributions are substantially different for men and women, so I considered measurements from male and female participants separately.

Here are the names of the measurements.

```python tags=["hide-cell"]
measurements = [column for column in ansur_female.columns if column.lower() == column]
measurements
```

```python tags=["hide-cell"]
len(measurements)
```

To find the measurements that fit a Gaussian model the best and the worst, I use the following function, which computes the maximum vertical distance between the CDF of the data and the model.

```python tags=["hide-cell"]
from utils import fit_gaussian


def ks_distance(series):
    """Compute the Kolmogorov-Smirnov distance. 
    
    returns: the maximum vertical distance between data and model,
    and the quantile where it happens
    """
    mu, sigma = fit_gaussian(series)
    cdf = Cdf.from_seq(series)
    norm_cdf = norm.cdf(cdf.qs, mu, sigma)
    diff = np.abs(cdf - norm_cdf)
    idx_diff = diff.idxmax()
    return diff.max(), idx_diff
```

The following function computes this distance for all of the measurements.

```python tags=["hide-cell"]
def all_ks_distances(df, tag, measurements):
    """Compute K-S distances for all measurements.
    
    df: DataFrame
    tag: string added to the list of results
    measurements: list of column names in df
    
    returns: list of (distance, location, tag, varname) tuples
    """
    res = []

    for varname in measurements:
        series = df[varname]
        diff, idx_diff = ks_distance(series)
        res.append((diff, idx_diff, tag, varname))

    return res
```

Now we can make a list of results for all measurements, male and female.

```python tags=["hide-cell"]
res_female = all_ks_distances(ansur_female, "female", measurements)
```

```python tags=["hide-cell"]
res_male = all_ks_distances(ansur_male, "male", measurements)
```

```python tags=["hide-cell"]
res = res_female + res_male
res.sort()
```

```python tags=["hide-cell"]
ks = np.array([k for k, _, _, _ in res])
```

```python tags=["hide-cell"]
np.percentile(ks, [25, 50, 75])
```

Here are the measurements where the distance between the data and the model is smallest.

```python tags=["hide-cell"]
res[:6]
```

And here are the measurements where it's the biggest.

```python tags=["hide-cell"]
res[-5:]
```

Most of the best-fitting measurements are male, most of the worst-fitting are female. That's because the sample size is smaller for the female participants, which means there is more deviation from the model due to chance.

Of all measurements, the one that is the closest match to the Gaussian distribution is the popliteal height of the male participants, which is the "vertical distance from a footrest to the back of the right knee."
The following figure shows the distribution of these measurements as a dashed line and the Gaussian model as a shaded area.

To quantify how well the model fits the data, I computed the maximum vertical distance between them; in this example, it is 0.8 percentile ranks, at the location indicated by the vertical dotted line.
The deviation is not easily discernible.

```python tags=["hide-cell"]
series = ansur_male["poplitealheight"] / 10
diff, idx_diff = ks_distance(series)
diff * 100, idx_diff
```

```python tags=["hide-input"]
plt.axvline(idx_diff, color="gray", ls=":", alpha=0.3)
gaussian_plot(series, label="Male participants", style="--")
decorate(
    xlabel="Popliteal height (cm)",
    ylabel="Percentile rank",
    title="Distribution of popliteal height compared to a Gaussian curve",
    loc="lower right",
)
```

The measurement that is the worst match to the Gaussian model is the forearm length of the female participants, which is the distance I mentioned earlier between the  radiale landmark on the right elbow and the stylion landmark on the right wrist.

The following figure shows the distribution of these measurements and a Gaussian model.
The maximum vertical distance between them is 4.2 percentile ranks, at the location indicated by the vertical dotted line; it looks like there are more measurements between 24 and 25 cm than we would expect in a Gaussian distribution.

```python tags=["hide-cell"]
# "distance between the radiale landmark on the right elbow and
# the stylion landmark on the right wrist"
# basically, forearm length

series = ansur_female["interpupillarybreadth"] / 10
diff, idx_diff = ks_distance(series)
diff * 100, idx_diff
```

```python tags=["hide-cell"]
# "distance between the radiale landmark on the right elbow and
# the stylion landmark on the right wrist"
# basically, forearm length

series = ansur_female["interpupillarybreadth"] / 10
diff, idx_diff = ks_distance(series)
diff * 100, idx_diff
```

```python tags=["hide-input"]
plt.axvline(idx_diff, ls=":", color="gray", alpha=0.3)
gaussian_plot(
    ansur_female["interpupillarybreadth"] / 10, label="Female participants", color="C1"
)
decorate(
    xlabel="Interpupillary Breadth (cm)",
    ylabel="Percentile rank",
    title="Distribution of distance between eyes compared to a Gaussian curve",
    loc="lower right",
)
```

Out of almost 200 measurements, this is the one that is the *least* well modeled by the Gaussian distribution; even so, the deviation is small, and for many purposes, the Gaussian model would be good enough.

What I reported in the book is `radialestylionlength`, which is second on the list and seems to be legitimately the measurement that is the worst fit for the model.

```python tags=["hide-cell"]
# "distance between the radiale landmark on the right elbow and
# the stylion landmark on the right wrist"
# basically, forearm length

series = ansur_female["radialestylionlength"] / 10
diff, idx_diff = ks_distance(series)
diff * 100, idx_diff
```

```python tags=["hide-input"]
plt.axvline(idx_diff, ls=":", color="gray", alpha=0.3)
gaussian_plot(
    ansur_female["radialestylionlength"] / 10, label="Female participants", color="C1"
)
decorate(
    xlabel="Radiale-stylion length (cm)",
    ylabel="Percentile rank",
    title="Distribution of forearm length compared to a Gaussian curve",
    loc="lower right",
)
```


## The Myth of the "Average Man"

We've seen that variation in physical characteristics is often well modeled by a Gaussian distribution.
A characteristic of these distributions is that most people are close to the mean, with fewer and fewer people far from the mean in either direction.
But as I said in the introduction of this chapter: what is true when we consider a single characteristic turns out to be counter-intuitively false when we consider a few characteristics at once, and spectacularly false when we consider more than a few.
In particular, when we consider the many ways each individual differs from the average, we find that people close to average in every way are rare or non-existent.

This observation was made most famously by Gilbert Daniels in a Technical Report he wrote for the U.S. Air Force in 1952, with the title "The 'Average Man?'".
In the introduction he explains:

> The tendency to think in terms of the "average man" is a pitfall into which many persons blunder when attempting to apply human body size data to design problems. Actually, it is virtually impossible to find an "average man" in the Air Force population.

As evidence, he used data from the Air Force Anthropometric Survey of 1950, a precursor of the ANSUR dataset we've used in this chapter.
This dataset includes 131 measurements from 4063 Air Force "flying personnel", all male.
From these, Daniels selected 10 measurements "useful in clothing design".
Coyly, he mentions that we would get similar results if we chose measurements useful in airplane cockpit design, which was the actual but unstated purpose of the report.


Daniels finds that of the 4063 men, 1055 were of "approximately average" height, which he defines to be within the "middle 30\% of the total population".
Of those, 302 were of approximately average chest circumference.
Of those, 143 were of approximately average sleeve length.

He continues, filtering out anyone who falls outside the middle 30\% of the population by any of the other measurements.
In the end, he finds 3 who make it past the first eight measurements, 2 who make it past the first nine, and 0 who were approximately average on all ten.
If a uniform -- or cockpit -- was designed to fit the average man on all 10 measurements, it would fit no one.


Daniels suggests that "conclusions similar to those reported here would have been reached if the same type of analysis had been applied to body size data based on almost any group of people".
To see if he's right, we'll replicate his analysis using data from the participants in the ANSUR survey.

Many of the measurements in the ANSUR survey are the same as in the Air Force survey, but not all.
Of the ten Daniels chose, I found eight identical measurements in the ANSUR dataset and two replacements that are similar.

Here are the measurements from ANSUR that are the closest to the measurements in Daniels's report.

```python tags=["hide-cell"]
daniels_measurements = [
    "stature",
    "chestcircumference",
    "sleevelengthspinewrist",
    "crotchheight",
    "verticaltrunkcircumferenceusa",
    "hipbreadthsitting",
    "neckcircumference",
    "waistcircumference",
    "thighcircumference",
    "crotchlengthposterioromphalion",
]
```

```python tags=["hide-cell"]
np.any(ansur_male.describe().loc["count"] != 4082)
```

```python tags=["hide-cell"]
res = []

for varname in daniels_measurements:
    series = ansur_male[varname] / 10
    mu, sigma = series.mean(), series.std()
    num_std = 0.3
    low, high = np.array([-num_std, num_std]) * sigma + mu
    # low, high = np.percentile(series, [35, 65])
    nearly_average = series.between(low, high)
    frac = nearly_average.mean() * 100
    res.append((varname, mu, sigma, low, high, frac))
```

The following table shows the names of these measurements, the mean and standard deviations of the values, the low and high ends of the range considered "approximately average", and the percentage of survey participants who fall in the range.

You might notice that these percentages are lower than the 30\% Daniels claims.
In an appendix, he explains that he used "a range of plus or minus three-tenths of a standard deviation" from the mean.
He chose this range because it seemed "reasonable" to him and because it is the "equivalent of a full clothing size".
To reproduce his analysis, I followed the specification in the appendix.

```python tags=["hide-cell"]
columns = ["Measurement", "Mean", "Std Dev", "Low", "High", "% in range"]
index = [
    "Stature (height)",
    "Chest Circumference",
    "Sleeve Length",
    "Crotch Height",
    "Vertical Trunk Circ.",
    "Hip Breadth Sitting",
    "Neck Circumference",
    "Waist Circumference",
    "Thigh Circumference",
    "Crotch Length",
]
table = pd.DataFrame(res, columns=columns, index=index)
table.index.name = "Measurement"
```

```python tags=["hide-cell"]
table["Std Dev"].values.round(1)
```

```python tags=["hide-input"]
from utils import compress_table

compress_table(table.drop(columns=["Measurement"]).round(1))
```

```python tags=["hide-cell"]
table["% in range"].mean()
```

Here are the measurements where there are the biggest differences between the Daniels dataset and the ANSUR dataset (you can ignore the two very large differences, which are the result of using measurements that are defined differently).

The other differences suggest that members of the Army and Marines now are bigger than members of the Air Force in 1950.

```python tags=["hide-cell"]
mean_daniels = [175.5, 98.6, 85.5, 83.4, 164.6, 106.0, 38.0, 81.4, 56.9, 71.6]
```

```python tags=["hide-cell"]
(table["Mean"] - mean_daniels) / mean_daniels * 100
```

```python tags=["hide-cell"]
std_daniels = [6.2, 6.2, 3.8, 4.4, 7.3, 7.2, 1.9, 7.7, 4.4, 5.1]
```

```python tags=["hide-cell"]
(table["Std Dev"] - std_daniels) / std_daniels * 100
```

The following cells replicate the analysis in Daniels, computing the number of people who are close to the average in each measurement, considered in succession.

```python tags=["hide-cell"]
average_table = pd.DataFrame(index=ansur_male.index, dtype=bool)

for measurement, row in table.iterrows():
    varname, _, _, low, high, _ = row
    average_table[varname] = ansur_male[varname].between(low * 10, high * 10)

average_table.mean()
```

```python tags=["hide-input"]
average_table.head()
```

```python tags=["hide-input"]
array = np.cumprod(average_table.values, axis=1)

cumulative_table = pd.DataFrame(
    array, index=average_table.index, columns=average_table.columns, dtype=bool
)

cumulative_table
```

Now we can replicate Daniels's analysis using the measurements as "hurdles in a step-by-step elimination", starting from the top of the table and working down.

Of 4086 male participants, 949 are approximately average in height.
Of those, 244 are approximately average in chest circumference.
Of those, 87 are approximately average in sleeve length.

Approaching the finish line, we find 3 participants that make it past eight "hurdles", 2 that make it past nine, and 0 that make it past all ten.
Remarkably, the results from the last three hurdles are identical to the results from Daniels's report.
In the ANSUR dataset, if you design for the "average man", you design for no one.

```python tags=["hide-cell"]
cumulative_table.sum()
```

The same is true if you design for the average woman.
The ANSUR dataset includes fewer women than men: 1986 compared to 4086.
So we can use a more generous definition of "approximately average", including anyone within 0.4 standard deviations of the mean.

```python tags=["hide-cell"]
np.any(ansur_female.describe().loc["count"] != 1986)
```

```python tags=["hide-cell"]
res = []

for varname in daniels_measurements:
    series = ansur_female[varname] / 10
    mu, sigma = series.mean(), series.std()
    num_std = 0.4
    low, high = np.array([-num_std, num_std]) * sigma + mu
    # low, high = np.percentile(series, [35, 65])
    nearly_average = series.between(low, high)
    frac = nearly_average.mean() * 100
    res.append((varname, mu, sigma, low, high, frac))
```

```python tags=["hide-input"]
table = pd.DataFrame(res, columns=columns, index=index)
table.index.name = "Measurement"
table.drop(columns=["Measurement"]).round(1)
```

```python tags=["hide-input"]
average_table = pd.DataFrame(index=ansur_female.index, dtype=bool)

for measurement, row in table.iterrows():
    varname, _, _, low, high, _ = row
    average_table[varname] = ansur_female[varname].between(low * 10, high * 10)

average_table.head()
```

```python tags=["hide-input"]
array = np.cumprod(average_table.values, axis=1)

cumulative_table = pd.DataFrame(
    array, index=average_table.index, columns=average_table.columns, dtype=bool
)

cumulative_table
```

Even so, we find only 2 women who make it past the first eight hurdles, 1 who makes it past nine, and none that make it past all ten.

```python tags=["hide-cell"]
cumulative_table.sum()
```

So we can confirm Daniels's conclusion (with a small update):

> The "average [person]" is a misleading and illusory concept as a basis for design criteria, and is particularly so when more than one dimension is being considered.

And this is not only true for physical measurements, as we'll see in the next section.

```python tags=["hide-cell"]
# *The Measurer's Handbook* used in the ANSUR survey is 253 pages,
# and includes details on "sampling strategy and measuring techniques"
# as well as descriptions and diagrams of the landmarks and measurements between then.
```

## The Big Five

Measuring physical characteristics is relatively easy; measuring psychological characteristics is more difficult.
However, starting in the 1980s, psychologists developed a taxonomy based on five personality traits and surveys that measure them.
These traits, known as the "Big Five", are: extroversion, emotional stability, agreeableness, conscientiousness, and openness to Experience.

Emotional stability is sometimes reported on a reversed scale as "neuroticism", so high emotional stability corresponds to low neuroticism.
In psychological literature, "extroversion" is often written as "extraversion", but I will use the common spelling.

```python tags=["hide-cell"]
# Extroversion: 70
# Emotional stability: 93
# Agreeableness: 45
# Conscientiousness: 93
# Intellect/Imagination: 96
```

These traits were initially proposed, and gradually refined, with the goal of satisfying these requirements:

* They should be interpretable in the sense that they align with characteristics people recognize in themselves and others. For example, many people know what extroversion is and have a sense of how they compare to others on this scale.

* They should be stable in the sense that they are mostly unchanged over a person's adult life, and consistent in the sense that measurements based on different surveys yield similar results.

* The traits should be mostly uncorrelated with each other, which indicates that they actually measure five different things, not different combinations of a smaller number of things.

* They do not have to be complete; that is, there can be (and certainly are) other traits that are not measured by surveys of the Big Five.


At this point, the Big Five personality traits have been studied extensively and found to have these properties, at least to the degree we can reasonably hope for.

You can take a version of the Big Five Personality Test, called the International Personality Item Pool (IPIP) online.
This test is run by the Open-Source Psychometrics Project, which "exists for the promotion of open source assessments and open data". 
In 2018 they published anonymous responses from more than one million people who took the test and agreed to make their data available for research.

Some people who took the survey did not answer all of the questions; if we drop them, we have data from 873,173 who completed the survey. It is possible that dropping incomplete tests will disproportionately exclude people with low Conscientiousness, but let's keep it simple. 

The Big Five data is originally from the [Open-Source Psychometrics Project](https://openpsychometrics.org/_rawdata/).

Here is the [notebook I used to download and clean the data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_big_five.ipynb).

I did some preliminary cleaning and put the result in an HDF file.

```python tags=["remove-cell"]
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/big_five_data.hdf")
```

```python tags=["hide-cell"]
big_five = pd.read_hdf("big_five_data.hdf", "big_five")
big_five.shape
```

```python tags=["hide-cell"]
fit_gaussian(big_five["EXT"])
```

```python tags=["hide-cell"]
bar_options = dict(width=1, alpha=0.4)
cdf_options = dict(xlabel=None, xlim=[-25, 25], ylabel="Percentile rank", yticks=[])
pmf_options = dict(xlabel=None, xlim=[-25, 25], ylabel="Percent", yticks=[])
```

```python tags=["hide-cell"]
def plot_pmf(series, **options):
    """Plot a PMF and a Gaussian PDF with the same mean and std.
    
    series: Series
    options: passed to plt.plot 
    """
    pmf_series = Pmf.from_seq(series)
    mu, sigma = fit_gaussian(series)
    qs = np.linspace(-25, 25)
    ps = norm.pdf(qs, mu, sigma)
    plt.plot(qs, ps, color="gray", alpha=0.4)
    pmf_series.bar(**options)
```

```python tags=["hide-input"]
plot_pmf(big_five["EXT"], **bar_options)

decorate(title="Extroversion Histogram", **pmf_options)
```

```python tags=["hide-cell"]
def plot_cdf(series, **options):
    """Plot a CDF and a Gaussian CDF with the same mean and std.
    
    series: Series
    options: passed to plt.plot 
    """
    cdf_series = make_cdf(series)
    mu, sigma = fit_gaussian(series)
    qs = np.linspace(-25, 25)
    ps = norm.cdf(qs, mu, sigma) * 100
    plt.plot(qs, ps, color="gray", alpha=0.4)
    cdf_series.plot(**options)
```

```python tags=["hide-input"]
plot_cdf(big_five["EXT"])
```

The survey consists of 50 questions, with 10 questions intended to measure each of the five personality traits.
For example, one of the prompts related to extroversion is "I feel comfortable around people;" one of the prompts related to emotional stability is "I am relaxed most of the time."
People respond on a five-point scale from "Strongly Disagree" to "Strongly Agree".
I scored the responses like this:

* "Strongly agree" scores 2 points.

* "Agree" scores 1 point,

* "Neutral" scores 0 points,

* "Disagree" scores -1 point, and 

* "Strongly disagree" scores -2 points.

For some questions, the scale is reversed; for example, if someone strongly agrees that they are "quiet around strangers," that counts as -2 points on the extroversion score.

Since there are 10 questions for each trait, the maximum score is 20 and the minimum score is -20.
For each of the five traits, the following figure shows the distributions of total scores for more than 800,000 respondents.

```python tags=["hide-cell"]
traits = {
    "EXT": "Extroversion",
    "EST": "Emotional stability",
    "CSN": "Conscientiousness",
    "AGR": "Agreeableness",
    "OPN": "Openness",
}
```

```python tags=["hide-cell"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
```

```python tags=["hide-input"]
plt.figure(figsize=(7, 9))
shape = (5, 2)
colspan = 1

for i, trait in enumerate(traits):
    series = big_five[trait]
    plt.subplot2grid(shape, (i, 0), colspan=colspan)
    plot_pmf(series, color=colors[i], **bar_options)
    title = f"{traits[trait]}"
    decorate(title=title, **pmf_options)

    plt.subplot2grid(shape, (i, 1), colspan=colspan)
    plot_cdf(series, ls="--", color=colors[i])
    title = f"{traits[trait]} CDF"
    decorate(legend=False, **cdf_options)
    plt.gca().yaxis.set_label_position("right")
```

The figures on the left show histograms; the figures on the right show CDFs.
In both figures, the shaded line is a Gaussian distribution I chose to fit the data.
The Gaussian model fits the first three distributions well (extroversion, emotional stability, and conscientiousness) except in the extreme tails. 

The other two distributions are skewed to the left; that is, their tails extend farther left than right.
It's possible that these deviations from the Gaussian model are the result of measurement error.
In particular, agreeableness and openness might be subject to social desirability bias, which is the tendency of survey respondents to shade their answers to questions like these in the direction they know is more socially acceptable.


For example, two of the prompts related to agreeableness are 
"I am not interested in other people's problems," and
"I insult people."
It's possible that even callous, abrasive people are sensitive enough to avoid admitting these faults.

Similarly, two of the prompts related to openness are
"I use difficult words," and
"I have excellent ideas."
It's possible that some people with a limited vocabulary are mistaken about the quality of their ideas.

On the other hand, it seems like at least some unconscientious people are willing to admit that "I leave my belongings around," and "I shirk my duties."
So it may be that these measurements reflect actual distributions of these attributes in the population, and it just happens that those distributions are not very Gaussian. 


Regardless, let's see what happens if we apply Daniels's analysis to the Big Five data.
The following table shows the mean and standard deviation of the five scores, the range of values we'll consider "approximately average", and the percentage of the sample that falls in that range.

```python tags=["hide-cell"]
res = []

for trait in traits:
    series = big_five[trait]
    mu, sigma = series.mean(), series.std()
    num_std = 0.3
    low, high = np.array([-num_std, num_std]) * sigma + mu
    # low, high = np.percentile(series, [37, 63])
    nearly_average = series.between(low, high)
    percent = nearly_average.mean() * 100
    res.append((trait, mu, sigma, low, high, percent))
```

```python tags=["hide-input"]
columns = ["Trait", "Mean", "Std Dev", "Low", "High", "% in range"]
index = list(traits.values())
table = pd.DataFrame(res, columns=columns, index=index)
table.index.name = "Trait"

compress_table(table.drop(columns=["Trait"]).round(1))
```

```python tags=["hide-cell"]
average_table = pd.DataFrame(index=big_five.index, dtype=bool)

for trait_name, row in table.iterrows():
    trait, _, _, low, high, _ = row
    print(low, high)
    average_table[trait] = big_five[trait].between(low, high)

average_table.mean()
```

```python tags=["hide-input"]
average_table.head()
```

```python tags=["hide-input"]
array = np.cumprod(average_table.values, axis=1)

cumulative_table = pd.DataFrame(
    array, index=average_table.index, columns=average_table.columns, dtype=bool
)

cumulative_table.head()
```

For each trait, the "average" range contains 20-28% of the population. 
Now if we treat each trait as a hurdle and select people who are close to average on each one, the following table shows the results.

```python tags=["hide-cell"]
len(big_five)
```

```python tags=["hide-input"]
counts = cumulative_table.sum()
index = counts.index.to_series().replace(traits)
index.name = 'Trait'

results = pd.DataFrame({"Counts": counts.values}, index=index)
results["Percentages"] = counts.values / len(big_five) * 100

compress_table(results.round(1))
```

The first column shows the number of people who make it past each hurdle; the second column shows the percentages.

Of the 873,173 people we started with, about 204,000 are close to the average in extroversion.
Of those, about 47,000 are close to the average in emotional stability.
And so on, until we find 926 who are close to average on all five traits, which is barely one person in a thousand.

As Daniels showed with physical measurements, we have shown with psychological measurements: when we consider more than a few dimensions, people who are close to the average are rare or nonexistent.
In fact, nearly everyone is far from average by some measure.

But that's not all.
If we consider a large number of measurements, it's not just that everyone is far from average; it turns out that everyone is approximately the same distance from average.
We'll see why in the next section.


## We Are All Equally Weird

How weird are you? One way to quantify that is to count the number of measurements where you are far from average.

For example, using the Big Five data again, I counted the number of traits where each respondent falls outside the range we defined as "approximately average".
We can think of the result as a kind of "weirdness score", where 5 means they are far from average on all five traits, and 0 means they are far from average on none.

The following figure shows the distribution of these scores for the roughly 800,000 people who completed the Big Five survey.

```python tags=["hide-cell"]
def summarize(varname, series):
    """Summarize the values in a series.
    
    varname: string
    series: Series
    
    low and high are the bounds of the "average" range
    n is the number of people who fall in that range
    
    return: tuple of (varname, mean, std, low, high, n)"""
    mu, sigma = series.mean(), series.std()
    num_std = 0.3
    low, high = np.array([-num_std, num_std]) * sigma + mu
    nearly_average = series.between(low, high)
    n = nearly_average.sum()
    return varname, mu, sigma, low, high, n
```

```python tags=["hide-cell"]
def make_summary_table(df, measurements, columns):
    """Make a table summarizing a set of measurements.
    
    df: DataFrame
    measurements: list of column names
    
    returns: DataFrame with the columns from summarize
    """
    res = []

    for varname in measurements:
        series = df[varname]
        t = summarize(varname, series)
        res.append(t)

    return pd.DataFrame(res, columns=columns)
```

```python tags=["hide-cell"]
def make_standardized_table(df, summary_table):
    """Compute standardized measurements.
    
    Express measurements in terms of standard deviations from the mean.
    
    df: DataFrame to get the original values from
    summary_table: DataFrame to get the rows from
    
    returns: DataFrame
    """
    d = {}
    for measurement, row in summary_table.iterrows():
        varname, mean, std, low, high, _ = row
        d[varname] = np.abs(df[varname] - mean) / std
    return pd.DataFrame(d)
```

```python tags=["hide-cell"]
def make_weird_table(df, summary_table):
    """Make a table counting number of weirdness points.
    
    df: DataFrame to get the data from
    summary_table: DataFrame to get the rows from
    
    returns: DataFrame
    """
    d = {}
    for measurement, row in summary_table.iterrows():
        varname, mean, std, low, high, _ = row
        is_average = df[varname].between(low, high)
        d[varname] = np.where(is_average, 0, 1)
    return pd.DataFrame(d)
```

```python tags=["hide-input"]
columns = ["Trait", "Mean", "Std Dev", "Low", "High", "# in range"]
summary_table = make_summary_table(big_five, traits, columns)
summary_table
```

```python tags=["hide-input"]
std_table = make_weird_table(big_five, summary_table)
weirdness = std_table.sum(axis=1)

pmf = Pmf.from_seq(weirdness)
pmf *= 100
pmf.bar(**bar_options)

decorate(
    xlabel="Number of measurements far from average",
    ylabel="Percent",
    title="Distribution of weirdness, Big Five dataset",
)
```

```python tags=["hide-cell"]
pmf[4:6].sum()
```

As we've already seen, very few people are close to average on all five traits.
Almost everyone is weird in two or more ways, and the majority (68\%) are weird in four or five ways! 

```python tags=["hide-input"]
summary_table = make_summary_table(ansur_male, measurements, columns)
summary_table
```

```python tags=["hide-cell"]
weird_table = make_weird_table(ansur_male, summary_table)
weird_table.mean()
```

```python tags=["hide-input"]
weird_table.head()
```

The distribution of weirdness is similar with physical traits.
Using the 93 measurements in the ANSUR dataset, we can count the number of ways each participant deviates from average.
The following figure shows the distribution of these counts for the male ANSUR participants.


```python tags=["hide-input"]
weirdness = weird_table.sum(axis=1)
pmf = Pmf.from_seq(weirdness)
pmf *= 100
pmf.bar(**bar_options)

decorate(
    xlabel="Number of measurements far from average",
    xlim=[0, 95],
    ylabel="Percent",
    title="Distribution of weirdness, ANSUR dataset",
)
```

Nearly everyone in this dataset is "weird" in at least 40 ways, and 90\% of them are weird in at least 57 ways.
With enough measurements, being weird is normal.

In fact, as the number of measurements increases, the width of this distribution gets narrower; that is, the difference between the most normal person and the weirdest gets smaller.

To demonstrate, I'll use the ANSUR measurements to compute all possible ratios of two measurements.
Some of these ratios are more meaningful than others, but at least some of them are features that affect clothing design (like the ratio of waist circumference to chest circumference),
cockpit design (like the ratio of arm length to leg length), or perceived attractiveness (like the ratio of face width to face height).

With 93 measurements and 4278 ratios, there are a total of 4317 ways to be weird.
The following figure shows the distribution of weirdness scores for the male participants.

```python tags=["hide-cell"]
weirdness.mean()
```

```python tags=["hide-cell"]
np.percentile(weirdness, [10])
```

```python tags=["hide-input"]
d = {}
for i, varname1 in enumerate(measurements):
    for j, varname2 in enumerate(measurements):
        if i <= j:
            continue
        colname = varname1 + varname2
        d[colname] = ansur_male[varname1] / ansur_male[varname2]

ratio_frame = pd.DataFrame(d)
ratio_frame.head()
```

```python tags=["hide-cell"]
ratio_frame.shape
```

```python tags=["hide-cell"]
ratio_frame = pd.concat([ansur_male[measurements], ratio_frame], axis=1)
ratio_frame.shape
```

```python tags=["hide-input"]
summary_table2 = make_summary_table(ratio_frame, ratio_frame.columns, columns)
summary_table2.head()
```

```python tags=["hide-cell"]
weird_table2 = make_weird_table(ratio_frame, summary_table2)
weird_table2.mean()
```

```python tags=["hide-input"]
weirdness2 = weird_table2.sum(axis=1)
plt.hist(weirdness2, bins=30, fill=True, alpha=0.4)

decorate(
    xlabel="Number of measurements far from average",
    xlim=[0, 4371],
    ylabel="Percent",
    title="Distribution of weirdness, ANSUR Dataset with ratios",
)
```

With these measurements, all participants fall in a relatively narrow range of weirdness.
The most "normal" participant deviates from average in 2446 ways; the weirdest in 4038 ways.

Now, you might notice that the distribution of this weirdness score has the characteristic shape of a Gaussian curve, and that is not a coincidence.
Mathematically, as the number of measurements increases, the distribution of weirdness converges to a Gaussian distribution and the width of the distribution gets narrower.
In the limit, if we consider the nearly infinite ways people vary, we find that we are all equally weird.

```python tags=["hide-cell"]
weirdness2.quantile([0, 0.05, 0.5, 0.95, 1])
```

## But Some Are More Equal Than Others

In some ways, this chapter is a feel good story.
If at times in your life you have felt different, you are not alone; everyone is different in many ways.
And if you think you're normal and other people aren't, you are wrong on both counts.

But the real world is not like the mathematical world, and I want to be careful not to push this point too far.
The idea that we are all equally weird is only true if the different ways of being different are treated the same.
And that's not the case.

Considering the measurements in this chapter, some are more visible than others.
If your menton-sellion length is in the 90th percentile, people will notice your long face, and if you are a famous actress, Seth MacFarlane will make fun of it.
But if your lateral malleolus height is in the 90th percentile, you were probably not teased as a child because your ankle bone is unusually far from the ground.


Among personality traits, some kinds of variation are met more favorably than others.
For the most part, we accept introversion and extroversion as equally valid ways to be, but some of the other traits are more laden with value judgments.
In general, we consider it better to be conscientious than unreliable, and better to be agreeable than obnoxious.

Also, the world is designed to handle some kinds of variation better than others.
For example, while introversion might not be considered a moral failing, many introverts find that school and work environments reward extroversion more than might be merited.

The world handles some kinds of physical variation better than others, too.
In *Invisible Women*, Caroline Criado Perez writes about many ways our built environment, designed primarily with men in mind, does not always fit women.
Safety features in cars, tested on crash dummies with male proportions, protect women less effectively.
Many smartphones are difficult to use one-handed if your hands are smaller than average. 
And, as more than half of the population discovered during the COVID-19 pandemic, a lot of personal protective equipment (PPE) is "based on the sizes and characteristics of male populations from certain countries in Europe and the United States. As a result, most women, and also many men, experience problems finding suitable and comfortable PPE," according to the Trades Union Congress in the United Kingdom.


Furthermore, the way I defined a "weirdness score" is too blunt.
To be consistent with Daniels's analysis, I consider only two categories: approximately average or not.
In reality, our ability to handle variation has many levels.
If you are a little taller or shorter than average, you have to adjust your car seat, but you might not even notice the inconvenience.
But if you are too short to stand behind a "standard" lectern, or too tall to walk through a door without ducking, you will notice that the world is not designed for you.
And at greater extremes, when the mismatch between you and the world makes common tasks difficult or impossible, we call it a disability -- and implicitly put the blame on the person, not the built environment.

So, when I say we are all equally weird, my intent is to point out the ways variability defies our intuitive concepts of "normal" and "weird".
I don't mean to say that the world treats us as equally weird; it doesn't.


## Sources and related reading

* The ANSUR-II dataset is available from the [Open Design Lab at Penn State](https://www.openlab.psu.edu/ansur2/), published by the U.S. Army Natick Soldier Research, Development and Engineering Center in 2012. The measurements are described in the [*Measurer's Handbook: US Army and Marine Corps Anthropometric Surveys, 2010-2011*](http://tools.openlab.psu.edu/publicData/ANSURII-TR11-017.pdf) by Jennifer Hotzman and colleagues (2011), which is more interesting than you might expect, especially if you need to know the proper term of address for a Marine Master Gunnery Sergeant.

* My example from a pediatric growth chart is from the Centers for Disease Control and Prevention (CDC) [growth charts](https://www.cdc.gov/growthcharts/data/set1clinical/cj41c017.pdf) published in 2022.

* You can take a version of the Big Five Personality Test, called the International Personality Item Pool (IPIP), [online](https://openpsychometrics.org/tests/IPIP-BFFM) from the Open-Source Psychometrics Project (2019).

* Gilbert S. Daniels's 1952 paper "The 'Average Man'?" is available from the [Defense Technical Information Center](https://apps.dtic.mil/sti/pdfs/AD0010203.pdf) (Technical Report RDO No. 695-71, Air Force Aerospace Medical Research Lab, Wright-Patterson AFB, Ohio).

* Todd Rose wrote about the myth of the average person in *The end of average: How to succeed in a world that values sameness* (Penguin UK, 2016).

* Caroline Criado Perez described some of the ways the world is ill-designed for women in a [2019 article in *The Guardian*](https://www.theguardian.com/lifeandstyle/2019/feb/23/truth-world-built-for-men-car-crashes), which was based on her book *Invisible Women: Data bias in a world designed for men* (Abrams, 2019).

* The [report on PPE and women](https://www.tuc.org.uk/research-analysis/reports/personal-protective-equipment-and-women) was published by the Trades Union Congress (TUC) in 2017.

* Sara Hendren wrote about the ways the world is ill-designed for people who are far from average in *What Can a Body Do?: How We Meet the Built World* (Penguin, 2020).

<!-- #region tags=["remove-print"] -->
Probably Overthinking It

Copyright 2022 Allen Downey 


<!-- #endregion -->

```python tags=["hide-cell"]

```

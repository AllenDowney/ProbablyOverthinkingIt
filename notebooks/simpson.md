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

# Penguins, Pessimists, and Paradoxes

<!-- #region tags=["remove-cell"] -->
This notebook contains the text from Chapter 10 of *[Probably Overthinking It: How to Use Data to Answer Questions, Avoid Statistical Traps, and Make Better Decisions](https://greenteapress.com/wp/probably-overthinking-it)* (University of Chicago Press, 2023), along with the supporting code used in the analysis.
<!-- #endregion -->

*Probably Overthinking It* is available from [Bookshop.org](https://bookshop.org/a/98697/9780226822587) and [Amazon](https://amzn.to/3Kp629E) (affiliate links).

If you like *Probably Overthinking It*, you'll like my book in progress, [*You Would Choose Now*](https://allendowney.com/ywcn), which uses decades of survey data to track changing attitudes in the United States. The early access edition is available now.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/simpson.ipynb).

```python tags=["remove-cell"]
# Install empiricaldist if we don't already have it

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["remove-cell"]
# Install empiricaldist if we don't already have it

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
import seaborn as sns

from utils import set_pyplot_params, decorate

set_pyplot_params()

# Set the random seed so we get the same results every time
np.random.seed(17)
```

In 2021, the journalist I mentioned in the previous chapter wrote a series of misleading articles about the COVID pandemic.
I won't name him, but on April 1, appropriately, *The Atlantic* magazine identified him as "The Pandemic's Wrongest Man".

In November, this journalist posted an online newsletter with the title "Vaccinated English adults under 60 are dying at twice the rate of unvaccinated people the same age".
It includes a graph showing that the overall death rate among young, vaccinated people increased between April and September, 2021, and was actually higher among the vaccinated than among the unvaccinated.

As you might expect, this newsletter got a lot of attention. Among skeptics, it seemed like proof that vaccines were not just ineffective, but harmful. And among people who were counting on vaccines to end the pandemic, it seemed like a devastating setback.

Many people fact-checked the article, and at first the results held up to scrutiny: the graph accurately represented data published by the U.K. Office for National Statistics.
Specifically, it showed death rates, from any cause, for people in England between 10 and 59 years old, from March to September 2021.
And between April and September, these rates were higher among the fully vaccinated, compared to the unvaccinated, by a factor of almost 2.
So the journalist's description of the results was correct.

However, his conclusion that vaccines were causing increased mortality was completely wrong.
In fact, the data he reported shows that vaccines were safe and effective in this population, and demonstrably saved many lives.

How, you might ask, can increased mortality be evidence of saved lives?
The answer is Simpson's paradox.
If you have not seen it before, Simpson's paradox can seem impossible, but as I will try to show you, it is not just possible but ordinary -- and once you've seen enough examples, it is not even surprising.

We'll start with an easy case and work our way up.


## Old Optimists, Young Pessimists

Would you say that most of the time people try to be helpful, or that they are mostly just looking out for themselves? Almost every year since 1972, the General Social Survey (GSS) has posed that question to a representative sample of adult residents of the United States.

The following figure shows how the responses have changed over time.
The circles show the percentage of people in each survey who said that people try to be helpful.

```python tags=["hide-cell"]
# https://gssdataexplorer.norc.org/variables/439/vshow

# 1 = helpful
# 2 = look out for themselves
# 3 = depends
```

```python tags=["remove-cell"]
# This dataset is prepared in clean_simpson.ipynb

DATA_PATH = "https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/"

filename = "gss_simpson.hdf"
download(DATA_PATH + filename)
```

```python tags=["hide-cell"]
gss = pd.read_hdf(filename, "gss")
gss.shape
```

```python tags=["hide-cell"]
year = gss["year"] == 1978
series = gss.loc[year, "helpful"]
series.describe()
```

```python tags=["hide-cell"]
series.value_counts()
```

```python tags=["hide-input"]
from empiricaldist import Pmf

Pmf.from_seq(series)
```

```python tags=["hide-cell"]
year = gss["year"] == 1987
series = gss.loc[year, "helpful"]
series.describe()
```

```python tags=["hide-cell"]
series.value_counts()
```

```python tags=["hide-input"]
from empiricaldist import Pmf

Pmf.from_seq(series)
```

```python tags=["hide-cell"]
xtab = pd.crosstab(gss["year"], gss["helpful"], normalize="index")
series = xtab[1] * 100
series.name = "helpful"
series.max(), series.idxmax(), series.min(), series.idxmin()
```

```python tags=["hide-cell"]
title = """Would you say that most of the time people try to be helpful, 
or that they are mostly just looking out for themselves?
"""

short_title = "Would people be helpful or look out for themselves?"
```

```python tags=["hide-input"]
from utils import plot_series_lowess

plot_series_lowess(series, plot_series=True, color="C0", label="")

# plt.title(title, loc="left", fontdict=dict(fontsize=12))
decorate(
    xlabel="Year of survey",
    ylabel="Percent",
    title="Percent saying 'helpful' vs year of survey",
)
```

```python tags=["hide-cell"]
from utils import make_lowess

make_lowess(series)
```

The percentages vary from year to year, in part because the GSS surveys a different group of people for each survey.
Looking at the extremes: in 1978, about 58% said "helpful"; in 1996, it was only 40%.

The solid line in the figure shows a LOWESS curve, which is a statistical way to smooth out short-term variation and show the long-term trend.
Putting aside the extreme highs and lows, it looks like the fraction of optimists has decreased since 1990, from about 51% to 48%.

These results don't just depend on when you ask; they also depend on who you ask.
In particular, they depend strongly on when the respondents were born, as shown in the following figure.

```python tags=["hide-input"]
from utils import prepare_yvar
from utils import chunk_series

xvarname = "cohort"
yvarname = "helpful"
yvalue = 1

prepare_yvar(gss, yvarname, yvalue)
series = chunk_series(gss, xvarname, size=600) * 100
plot_series_lowess(
    series,
    plot_series=True,
    color="C1",
    ls="dashed",
    label="",
)

# plt.title(title, loc="left", fontdict=dict(fontsize=11))
decorate(
    xlabel="Year of birth",
    ylabel="Percent",
    title="Percent saying 'helpful' vs year of birth",
)
```

Notice that the x-axis here is the respondent's year of birth, not the year they participated in the survey.
The oldest person to participate in the GSS was born in 1883.
Because the GSS only surveys adults, the youngest, as of 2021, was born in 2003.
Again, the markers show variability in the survey results; the dashed line shows the long-term trend.

Starting with people born in the 1940s, Americans' faith in humanity has been declining consistently, from about 55% for people born before 1930 to about 30% for people born after 1990.
That is a substantial decrease!

```python tags=["hide-cell"]
make_lowess(series)
```

To get an idea of what's happening, let's zoom in on people born in the 1940s.
The following figure shows the fraction of people in this cohort who said people would try to be helpful, plotted over time.

```python tags=["hide-cell"]
subset = gss.dropna(subset=["cohort10"]).copy()
subset["cohort10"] = subset["cohort10"].astype(int)
```

```python tags=["hide-cell"]
from utils import make_table

xvarname = "year"
yvarname = "helpful"
gvarname = "cohort10"
yvalue = 1

prepare_yvar(gss, yvarname, yvalue)
overall = chunk_series(subset, xvarname) * 100
table = make_table(subset, xvarname, yvarname, gvarname, yvalue)
del table[1900]
del table[1910]
series = table[1940]
```

```python tags=["hide-cell"]
make_lowess(series)
```

```python tags=["hide-input"]
plot_series_lowess(series, plot_series=True, color="C2", label="Born in the 1940s")

decorate(
    xlabel="Year of survey",
    ylabel="Percent",
    title="Percent saying 'helpful' vs year of survey",
    loc="lower right",
)
```

Again, different people were surveyed each year, so there is a lot of variability. Nevertheless, there is a clear increasing trend.
When this cohort was interviewed in the 1970s, about 46% of them gave a positive response; when they were interviewed in the 2010s, about 61% did.

So that's surprising. So far, we've seen that

* People have become more pessimistic over time,

* And successive generations have become more pessimistic,

* But if we follow one cohort over time, they become more optimistic as they age.

Now, maybe there's something unusual about people born in the 1940s.
They were children during the aftermath of World War II and they were young adults during the rise of the Soviet Union and the Cold War.
These experiences might have made them more distrustful when they were interviewed in the 1970s.
Then, from the 1980s to the present, they enjoyed decades of relative peace and increasing prosperity.
And when they were interviewed in the 2010s, most were retired and many were wealthy.
So maybe that put them in a more positive frame of mind.

However, there is a problem with this line of conjecture: it wasn't just one cohort that trended toward the positive over time.
It was all of them.

The following figure shows the responses to this question grouped by decade of birth and plotted over time.
For clarity, I've dropped the circles showing the annual results and plotted only the smoothed lines.

```python tags=["hide-input"]
from utils import visualize_table, label_table

visualize_table(
    overall,
    table,
    xlabel="Year of survey",
    ylabel="Percent",
    title="Percent saying 'helpful' vs year of survey",
    plot_series=False,
)
nudge = {"1900s": 1, "1910s": 1, "1920s": 2}
label_table(table, nudge)
```

The solid lines show the trends within each cohort: people generally become more optimistic as they age.
The dashed line shows the overall trend: on average, people are becoming less optimistic over time.

This is an example of Simpson's paradox, named for Edward H. Simpson, a codebreaker who worked at Bletchley Park during World War II.
For people who have not seen it before, the effect often seems impossible:
if all of the groups are increasing, and we combine them into a single group, how can the aggregate decrease?

In this example, the explanation is *generational replacement*; as the oldest, more optimistic cohort dies off, it is replaced by the youngest, more pessimistic cohort.


The differences between these generations are substantial.
In the most recent data, only 39% of respondents born in the 1990s said people would try to be helpful, compared to 51% of those born in the 1960s and 64% of those born in the 1920s.

```python tags=["hide-cell"]
table[1920].iloc[-1]
```

```python tags=["hide-cell"]
table[1960].iloc[-1]
```

```python tags=["hide-cell"]
table[1990].iloc[-1]
```

And the composition of the population has changed a lot over 50 years.
The following figure shows the distribution of birth years for the respondents at the beginning of the survey in 1973, near the middle in 1990, and most recently in 2018.

```python tags=["hide-input"]
xtab = pd.crosstab(gss["year"], gss["cohort10"], normalize="index").replace(0, np.nan)

Pmf(xtab.loc[1973]).plot(ls=":", label="1973")
Pmf(xtab.loc[1990]).plot(color="C2", ls="--", label="1990")
Pmf(xtab.loc[2018]).plot(color="C1", label="2018")

decorate(xlabel="Year of birth", ylabel="Fraction of sample")
```

```python tags=["hide-input"]
from seaborn import kdeplot

options = dict(bw_adjust=2, cut=0, alpha=0.6)

kdeplot(gss.query("year==1973")["cohort"], ls=":", label="1973", **options)
kdeplot(gss.query("year==1990")["cohort"], color="C2", ls="--", label="1990", **options)
kdeplot(gss.query("year==2018")["cohort"], color="C1", label="2018", **options)

decorate(
    xlabel="Year of birth",
    ylabel="Relative likelihood",
    yticks=[],
    title="Distribution of birth year",
)
```

In the earliest years of the survey, most of the respondents were born between 1890 and 1950.
In the 1990 survey, most were born between 1901 and 1972.
In the 2018 survey, most were born between 1929 and 2000.

```python tags=["hide-cell"]
series = gss.query("year==1973")["cohort"]
series.min(), series.max()
```

```python tags=["hide-cell"]
series = gss.query("year==1990")["cohort"]
series.min(), series.max()
```

```python tags=["hide-cell"]
series = gss.query("year==2018")["cohort"]
series.min(), series.max()
```

With this explanation, I hope Simpson's paradox no longer seems impossible.
But if it is still surprising, let's look at another example.


## Real Wages

In April 2013 Floyd Norris, chief financial correspondent of *The New York Times*, wrote an article with the headline "Median Pay in U.S. Is Stagnant, but Low-Paid Workers Lose".
Using data from the U.S. Bureau of Labor Statistics, he computed the following changes in median real wages between 2000 and 2013, grouped by level of education:

*    High School Dropouts: -7.9%
*    High School Graduates, No College: -4.7%
*    Some College: -7.6%
*    Bachelor's or Higher: -1.2%

In all groups, real wages were lower in 2013 than in 2000, which is what the article was about.
But Norris also reported the overall change in the median real wage: during the same period, it *increased* by 0.9%.
Several readers wrote to say he had made a mistake.
If wages in all groups were doing down, how could overall wages go up?

To explain what's going on, I will replicate the results from Norris's article using data from the Current Population Survey (CPS) conducted by the U.S. Census Bureau and the Bureau of Labor Statistics.
This dataset includes real wages and education levels for almost 1.9 million participants between 1996 and 2021.
Real wages are adjusted for inflation and reported in constant 1999 dollars.

```python tags=["remove-cell"]
# this dataset is prepared in clean_cps.ipynb

filename = "ipums_cps.hdf"
download(DATA_PATH + filename)
```

```python tags=["hide-input"]
ipums_cps = pd.read_hdf(filename, "ipums_cps")
ipums_cps.head()
```

```python tags=["hide-cell"]
d = {
    "nohs": "No diploma",
    "hs": "High school",
    "bach": "Bachelor",
    "assc": "Associate",
    "college": "Some college",
    "adv": "Advanced",
}
ipums_cps = ipums_cps.replace({"degree": d})
```

```python tags=["hide-input"]
start = 1996
recent = ipums_cps[ipums_cps["year"] >= start]
recent.head()
```

```python tags=["hide-cell"]
recent["real_wage"].describe()
```

```python tags=["hide-cell"]
recent["degree"].describe()
```

```python tags=["hide-cell"]
recent["year"].describe()
```

```python tags=["hide-cell"]
xvarname = "year"
yvarname = "real_wage"
gvarname = "degree"
```

```python tags=["hide-cell"]
overall = recent.groupby("year")["real_wage"].mean()
overall.name = "overall"
```

```python tags=["hide-cell"]
from scipy.stats import linregress

res = linregress(overall.index, overall)
res.slope
```

```python tags=["hide-cell"]
table = pd.pivot_table(
    recent, index="year", columns="degree", values="real_wage", aggfunc="mean"
)
```

```python tags=["hide-cell"]
for name, column in table.items():
    res = linregress(column.index, column)
    print(name, res.slope)
```

```python tags=["hide-cell"]
columns = [
    "Advanced",
    "Bachelor",
    "Associate",
    "Some college",
    "High school",
    "No diploma",
]
```

The following figure shows average real wages for each level of education over time.

```python tags=["hide-input"]
# figwidth = 3.835
# plt.figure(figsize=(figwidth, 0.6 * figwidth))

nudge = {"High school": -1.7}

table2 = table / 1000
for column in table2.columns:
    series = table2[column]
    series.plot(label="", alpha=0.6)
    x = series.index[-1] + 0.5
    y = series.values[-1] + nudge.get(column, 0) - 1
    plt.text(x, y, column)

overall2 = overall / 1000
overall2.plot(ls=":", color="gray", label="")

decorate(
    xlabel="Year",
    ylabel="1999 dollars (thousands)",
    title="Average real wage income by education level",
    xlim=[1996, 2022],
)
plt.xticks([2000, 2010, 2020])
plt.yticks([10, 30, 50])
```

These results are consistent with Norris's: real wages in every group decreased over this period.
The decrease was steepest among people with an associate degree, by about $190 per year over 26 years.

The decrease was smallest among people with no high school diploma, about $30 per year.

However, if we combine the groups, real wages increased during the same period by about $80 per year, as shown by the dotted line in the figure.
Again, this seems like a contradiction.

In this example, the explanation is that levels of education changed substantially during this period.
The following figure shows the fraction of the population at each educational level, plotted over time.

```python tags=["hide-input"]
xtab = pd.crosstab(recent["year"], recent["degree"], normalize="index") * 100
xtab
```

```python tags=["hide-cell"]
columns2 = [
    "High school",
    "Bachelor",
    "Some college",
    "No diploma",
    "Advanced",
    "Associate",
]

left = [
    "High school",
    "Some college",
    "No diploma",
]
```

```python tags=["hide-input"]
nudge = {"No diploma": 0, "Advanced": -0.2, "Associate": -0.2}

for column in xtab.columns:
    series = xtab[column]
    series.plot(label="", alpha=0.6)
    if column in left:
        x = series.index[0] - 0.3
        y = series.values[0] + nudge.get(column, 0) - 0.3
        plt.text(x, y, column, ha="right")
    else:
        x = series.index[-1] + 0.3
        y = series.values[-1] + nudge.get(column, 0) - 0.3
        plt.text(x, y, column, ha="left")


decorate(
    xlabel="Year",
    ylabel="Percent",
    xlim=[1988, 2022],
    title="Percent of labor force at each education level",
)
plt.xticks([2000, 2010, 2020])
```

Compared to 1996, people in 2021 have more education.
The groups with associate degrees, bachelor's degrees, or advanced degrees have grown by 3-7 percentage points.
Meanwhile, the groups with no diploma, high school, or some college have shrunk by 2-10 percentage points.

So the overall increase in real wages is driven by changes in education levels, not by higher wages at any level.
For example, a person with a bachelor's degree in 2021 makes less money, on average, than a person with a bachelor's degree in 1996.
But a random person in 2021 is more likely to have a bachelor's degree (or more) than a person in 1996, so they make more money, on average.

```python tags=["hide-cell"]
xtab.loc[2021] - xtab.loc[1996]
```

In both examples so far, the explanation for Simpson's paradox is that the population is a mixture of groups and the proportions of the mixture change over time.
In the GSS examples, the population is a mixture of generations, and the older generations are replaced by younger generations.
In the CPS example, the population is a mixture of education levels, and the fraction of people at each level changes over time.

But Simpson's paradox does not require the variable on the $x$ axis to be time.
More generally, it can happen with any variables on the $x$ and $y$ axes and any set of groups.
To demonstrate, let's look at an example that doesn't depend on time.


## Penguins

In 2014 a team of biologists published measurements they collected from three species of penguins living near the Palmer Research Station in Antarctica.
For 344 penguins, the dataset includes body mass, flipper length, and the length and depth of their beaks: more specifically, the part of the beak called the culmen.

This dataset has become popular with machine learning experts because it is a fun way to demonstrate all kinds of statistical methods, especially classification algorithms.

Allison Horst, an environmental scientist who helped make this dataset freely available, used it to demonstrate Simpson's paradox.
The following figure shows the example she discovered:

```python tags=["remove-cell"]
filename = "penguins_raw.csv"
download(DATA_PATH + filename)
```

```python tags=["hide-cell"]
df = pd.read_csv(filename)
df.shape
```

```python tags=["hide-cell"]
def shorten(species):
    """Select the first word from a string."""
    return species.split()[0]


df["Species2"] = df["Species"].apply(shorten)
```

```python tags=["hide-cell"]
var1 = "Flipper Length (mm)"
var2 = "Culmen Length (mm)"
var3 = "Culmen Depth (mm)"
var4 = "Body Mass (g)"
```

```python tags=["hide-cell"]
from utils import Or70, Pu50, Gr30

colors = [Or70, Pu50, Gr30]
markers = ["s", "o", "^"]
```

```python tags=["hide-cell"]
def scatterplot(df, var1, var2):
    """Make a scatter plot grouped by species.
    
    df: DataFrame with penguin data
    var1: name of variable for x-axis
    var2: name of variable for y-axis
    """
    grouped = df.groupby("Species2")
    i = 0
    for species, group in grouped:
        plt.plot(
            group[var1],
            group[var2],
            markers[i],
            color=colors[i],
            label=species,
            alpha=0.2,
        )
        i += 1

    decorate(xlabel=var1, ylabel=var2)
    plt.legend(loc="upper left", bbox_to_anchor=(0.98, 1.0))
    plt.tight_layout()
```

```python tags=["hide-cell"]
def plot_line(group, result, **options):
    """Plot a linear regression line.

    group: DataFrame
    result: linear regression result
    """
    xs = group[var2].min(), group[var2].max()
    ys = np.array(xs) * result.slope + result.intercept
    plt.plot(xs, ys, alpha=0.5, **options)
```

```python tags=["hide-input"]
from scipy.stats import linregress

scatterplot(df, var2, var3)

data = df.dropna(subset=[var2, var3])
grouped = data.groupby("Species2")
i = 0
for species, group in grouped:
    result = linregress(group[var2], group[var3])
    plot_line(group, result, color=colors[i])
    i += 1

result = linregress(data[var2], data[var3])
plot_line(data, result, ls=":", color="gray")
```

Each marker represents the culmen length and depth of an individual penguin. 
The different marker shapes represent different species: Adélie, Chinstrap, and Gentoo.
The solid lines are the lines of best fit within each species; in all three, there is a positive correlation between length and depth.
The dotted line shows the line of best fit among all of the penguins; between species, there is a negative correlation between length and depth.


In this example, the correlation within species has some meaning: most likely, length and depth are correlated because both are correlated with body size.
Bigger penguins have bigger beaks.

However, when we combine samples from different species and find that the correlation of these variables is negative, that's a statistical artifact.
It depends on the species we choose and how many penguins of each species we measure.
So it doesn't really mean anything.

```python tags=["hide-cell"]
for species, group in grouped:
    result = linregress(group[var2], group[var3])
    print(species, result.slope)

result = linregress(data[var2], data[var3])
print("overall", result.slope)
```

These results are puzzling only if you think the trend within groups is necessarily the same as the the trend between groups.
Once you realize that's not true, the paradox is resolved.


Penguins might be interesting, but dogs take Simpson's paradox to a new level.
In 2007, a team of biologists at Leiden University in the Netherlands and the Natural History Museum in Berne, Switzerland described the unusual case of one Simpson's paradox nested inside another. They observe:

* When we compare species, there is a positive correlation between weight and lifespan: in general, bigger animals live longer.

* However, when we compare dog breeds, there is a negative correlation: bigger breeds have shorter lives, on average.

* However, if we compare dogs within a breed, there is a positive correlation again: a bigger dog is expected to live longer than a smaller dog of the same breed.

In summary, elephants live longer than mice, Chihuahuas live longer than Great Danes, and big beagles live longer than small beagles.
The trend is different at the level of species, breed, and individual because the relationship at each level is driven by different biological processes.

These results are puzzling only if you think the trend within groups is necessarily the same as the the trend between groups.
Once you realize that's not true, the paradox is resolved.


## Simpson's Prescription

Some of the most bewildering examples of Simpson's paradox come up in medicine, where a treatment might prove to be effective for men, effective for women, and ineffective or harmful for people.
To understand examples like that, let's start with an analogous example from the General Social Survey.

Since 1973, the GSS has asked respondents, "Is there any area right around here--that is, within a mile--where you would be afraid to walk alone at night?"
Out of more than 37,000 people who have answered this question, about 38% said "Yes".

Since 1974, they have also asked, "Should divorce in this country be easier or more difficult to obtain than it is now?"
About 49% of the respondents said it should be more difficult.

```python tags=["hide-cell"]
gss["divlaw"].notna().sum()
```

```python tags=["hide-cell"]
(gss["divlaw"].dropna() == 2).mean()
```

```python tags=["hide-input"]
xtab = pd.crosstab(gss["year"], gss["divlaw"], normalize="index")
plot_series_lowess(xtab[2])
```

```python tags=["hide-cell"]
gss["fear"].notna().sum()
```

```python tags=["hide-cell"]
(gss["fear"].dropna() == 1).mean()
```

```python tags=["hide-input"]
xtab = pd.crosstab(gss["year"], gss["fear"], normalize="index")
plot_series_lowess(xtab[1])
```

Now, if you had to guess, would you expect these responses to be correlated; that is, do you think someone who says they are afraid to walk alone at night might be more likely to say divorce should be more difficult?
As it turns out, they are slightly correlated:

* Of people who said they are afraid, about 50% said divorce should be more difficult.

* Of people who said they are not afraid, about 49% said divorce should be more difficult.

However, there are several reasons we should not take this correlation too seriously:

* The difference between the groups is too small to make any real difference, even if it were valid;

* It combines results from 1974 to 2018, so it ignores the changes in both responses over that time; and

* It is actually an artifact of Simpson's paradox.

To explain the third point, let's look at responses from men and women separately. The following figure shows this breakdown.

```python tags=["hide-cell"]
# This example doesn't work with the new 2021 data.
# Examples of Simpson's paradox are not always robust, but the
# point of this example is that it existed at a point in time.

gss = pd.read_hdf("gss_simpson.hdf", "gss")
gss.shape
```

```python tags=["hide-cell"]
data = gss.dropna(subset=["fear"]).copy()
```

```python tags=["hide-cell"]
data = data.replace({"fear": {2: 0}})
data["fear"].value_counts()
```

```python tags=["hide-cell"]
data = data.replace({"sex": {1: "Male", 2: "Female"}})
data["sex"].value_counts()
```

```python tags=["hide-cell"]
from utils import make_pivot_table

xvarname = "fear"
yvarname = "divlaw"
gvarname = "sex"
yvalue = 2

series, table = make_pivot_table(data, xvarname, yvarname, gvarname, yvalue)
series
```

```python tags=["hide-input"]
table
```

```python tags=["hide-cell"]
series
```

```python tags=["hide-input"]
from utils import anchor_legend

table["Female"].plot(color="C1")
table["Male"].plot(ls="--", color="C0")

series.plot(ls=":", color="gray", label="Overall")
plt.plot(0, series[0], "s", color="gray")
plt.plot(1, series[1], ">", color="gray")

title = "Should divorce be easier to obtain?"
decorate(xlabel="Afraid at night", ylabel="Percent saying yes", title=title)
plt.xticks([0, 1], ["No", "Yes"])
anchor_legend(1.02, 1.02)
None
```

Among women, people who are afraid are less likely to say divorce should be difficult.
And among men, people who are afraid are *also* less likely to say divorce should be difficult (very slightly).
In both groups, the correlation is negative, but when we put them together, the correlation is positive.
People who are afraid are more likely to say divorce should be difficult.

As with the other examples of Simpson's paradox, this might seem to be impossible.
But as with the other examples, it is not.

The key is to realize that there is a correlation between the sex of the respondent and their responses to both questions.
In particular, of the people who said they are not afraid, 41% are female; of the people who said they *are* afraid, 74% are female.
You can actually see these proportions in the figure:

* On the left, the square marker is 41% of the way between the male and female proportions.

* On the right, the triangle marker is 74% of the way between the male and female proportions.

In both cases, the overall proportion is a mixture of the male and female proportions. But it is a *different* mixture on the left and right.
That's why Simpson's paradox is possible.

```python tags=["hide-input"]
xtab = pd.crosstab(data["fear"], data["sex"], normalize="index")
xtab
```

With this example under our belts, we are ready to take on one of the best-known examples of Simpson's paradox.
It comes from a paper, published in 1986 by researchers at St Paul's Hospital in London, that compared two treatments for kidney stones: (A) open surgery and (B) percutaneous nephrolithotomy.
For our purposes, it doesn't matter what these treatments are; I'll just call them A and B.

Among 350 patients who received Treatment A, 78% had a positive outcome;
among another 350 patients who received Treatment B, 83% had a positive outcome.
So it seems like Treatment B is better.

```python tags=["hide-cell"]
index = [0, 1]
series = pd.Series([78, 83], index=index)
series
```

```python tags=["hide-input"]
columns = ["Small", "Large"]
numer = pd.DataFrame([[81, 192], [234, 55]], index=index, columns=columns)
numer
```

```python tags=["hide-input"]
denom = pd.DataFrame([[87, 263], [270, 80]], index=index, columns=columns)
denom
```

```python tags=["hide-cell"]
denom.T.sum()
```

```python tags=["hide-cell"]
series = numer.T.sum() / denom.T.sum() * 100
series
```

However, when they split the patients into groups, they found that the results were reversed:

* Among patients with relatively small kidney stones, Treatment A was better, with a success rate of 93% compared to 86%.

* Among patients with larger kidney stones, Treatment A was better again, with a success rate of 73% compared to 69%.

The following figure shows these results in graphical form.

```python tags=["hide-input"]
table = numer / denom * 100
table
```

```python tags=["hide-input"]
table["Small"].plot(color="C1")
table["Large"].plot(ls="--", color="C0")

series.plot(ls=":", color="gray", label="Overall")
plt.plot(0, series[0], "s", color="gray")
plt.plot(1, series[1], ">", color="gray")

decorate(
    xlabel="Treatment",
    ylabel="Percent",
    title="Kidney stone positive outcomes",
)
plt.xticks([0, 1], ["A", "B"])
anchor_legend(1.02, 1.02)
None
```

This figure resembles the previous one because the explanation is the same in both cases: the overall percentages, indicated by the square and triangle markers, are mixtures of the percentages from the two groups.
But they are different mixtures on the left and right.

For medical reasons, patients with larger kidney stones were more likely to receive Treatment A; patients with smaller kidney stones were more likely to receive Treatment B.

As a result, among people who received Treatment A, 75% had large kidney stones, which is why the square marker on the left is closer to the Large group.
And among people who received Treatment B, 77% had small kidney stones, which is why the triangle marker is closer to the Small group.

```python tags=["hide-input"]
props = denom.T / denom.T.sum()
props
```

I hope this explanation makes sense, but even if it does, you might wonder how to choose a treatment.
Suppose you are a patient and your doctor explains:

* For someone with a small kidney stone, Treatment A is better.

* For someone with a large kidney stone, Treatment A is better.

* But overall, Treatment B is better.

Which treatment should you choose?
Other considerations being equal, you should choose A because it has a higher chance of success for any patient, regardless of stone size.
The apparent success of B is a statistical artifact, the result of two causal relationships:

* Smaller stones are more likely to get Treatment B.

* Smaller stones are more likely to lead to a good outcome.

Looking at the overall results, Treatment B only looks good because it is used for easier cases; Treatment A only looks bad because it is used for the hard cases.
If you are choosing a treatment, you should look at the results within the groups, not between the groups.


## Do vaccines work? Hint: yes.

Now that we have mastered Simpson's paradox, let's get back to the example from the beginning of the chapter, the newsletter that went viral with the claim that "Vaccinated English adults under 60 are dying at twice the rate of unvaccinated people the same age".

The newsletter includes a graph based on data from the U.K. Office for National Statistics.
I downloaded the same data and replicated the graph, which looks like this:

```python tags=["hide-cell"]
# https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fbdca5329-b20b-4518-a733-fff84cc22124_1098x681.png
```

```python tags=["hide-cell"]
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/deathsbyvaccinationstatusengland
# https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/deathsbyvaccinationstatusengland/deathsoccurringbetween2januaryand24september2021/datasetfinalcorrected3.xlsx
```

```python tags=["remove-cell"]
filename = "datasetfinalcorrected3.xlsx"
download(DATA_PATH + filename)
```

```python tags=["hide-input"]
df = pd.read_excel(
    filename,
    sheet_name="Table 4",
    skiprows=3,
    skipfooter=12,
)
df.tail()
```

```python tags=["hide-cell"]
df = df.replace({"Age-specific rate per 100,000": {":": np.nan}})
```

```python tags=["hide-cell"]
df.columns
```

```python tags=["hide-cell"]
table = df.pivot_table(
    index="Week ending",
    columns=["Vaccination status", "Age group"],
    values="Age-specific rate per 100,000",
    aggfunc="median",
)
```

```python tags=["hide-input"]
table.head()
```

```python tags=["hide-cell"]
recent = table.index >= pd.Timestamp("2021-3-19")
```

```python tags=["hide-input"]
table[recent]["Second dose", "10-59"].plot(color="C1", ls="-", label="Vaccinated")
table[recent]["Unvaccinated", "10-59"].plot(color="C0", ls="--", label="Unvaccinated")

plt.margins(x=0.1, tight=False)
decorate(
    xlabel="",
    ylabel="Weekly deaths per 100k",
    title="All cause death rates, ages 10–59",
    ylim=[0.1, 4.6],
    loc="upper right",
)
```

The lines show weekly death rates from all causes, per 100,000 people age 10-59, from March to September 2021.
The solid line represents people who were fully vaccinated; the dashed line represents people who were unvaccinated.

The journalist I won't name concludes: "Vaccinated people under 60 are twice as likely to die as unvaccinated people. And overall deaths in Britain are running well above normal. I don't know how to explain this other than vaccine-caused mortality."
Well, I do.

There are two things wrong with his interpretation of the data:

* First, by choosing one age group and time interval, he has cherry-picked data that support his conclusion and ignored data that don't.

* Second, because the data combine a wide range of ages, from 10 to 59, he has been fooled by Simpson's paradox.

Let's debunk one thing at a time.
First, here's what the graph looks like if we include the entire dataset, which starts in January:

```python tags=["hide-cell"]
start = pd.Timestamp("2021-01-01")
end = pd.Timestamp("2021-09-30")
```

```python tags=["hide-cell"]
options = dict(xlabel="", ylabel="Weekly deaths per 100k", xlim=[start, end])
```

```python tags=["hide-input"]
table["Second dose", "10-59"].plot(color="C1", label="Vaccinated")
table["Unvaccinated", "10-59"].plot(ls="--", label="Unvaccinated")

decorate(title="All cause death rates, ages 10–59", ylim=[0.1, 4.7], **options)
```

Overall death rates were "running well above normal" in January and February, when almost no one in the 10-59 age range had been vaccinated.
Those deaths cannot have been caused by the vaccine; in fact, they were caused by a surge in the Alpha variant of COVID-19.

The unnamed journalist leaves out the time range that contradicts him; he also leaves out the age ranges that contradict him.
In all of the older age ranges, death rates were consistently lower among people who had been vaccinated.
The following figures show death rates for people in their 60s and 70s, and people over 80.

```python tags=["hide-input"]
ages = "60-69"
table["Unvaccinated", ages].plot(ls="--", label="Unvaccinated")
table["Second dose", ages].plot(label="Vaccinated")

decorate(title="All cause death rates, ages 60–69", **options)
```

```python tags=["hide-input"]
ages = "70-79"
table["Unvaccinated", ages].plot(ls="--", label="Unvaccinated")
table["Second dose", ages].plot(label="Vaccinated")

decorate(title="All cause death rates, ages 70–79", **options)
```

```python tags=["hide-input"]
ages = "80+"
table["Unvaccinated", ages].plot(ls="--", label="Unvaccinated")
table["Second dose", ages].plot(label="Vaccinated")

decorate(title="All cause death rates, ages 80+", **options)
```

Notice that the $y$ axes are on different scales; death rates are much higher in the older age groups.

```python tags=["hide-cell"]
ages = "80+"
ratio = table["Unvaccinated", ages] / table["Second dose", ages]
ratio.iloc[:10]
```

In all of these groups, death rates are substantially lower for people who are vaccinated, which is what we expect from a vaccine that has been shown in large clinical trials to be safe and effective.

So what explains the apparent reversal among people between 10 and 59 years old?
It is a statistical artifact caused by Simpson's paradox.
Specifically, it is caused by two correlations:

* Within this age group, older people were more likely to be vaccinated, and

* Older people were more likely to die of any cause, just because they are older.

Both of these correlations are strong.
For example, at the beginning of August, about 88% of people at the high end of this age range had been vaccinated; *none* of the people at the low end had.
And the death rate for people at the high end was 54 times higher than for people at the low end.


To show that these correlations are strong enough to explain the observed difference in death rates, I will follow an analysis by epidemiologist Jeffrey Morris, who refuted the unreliable journalist's claims within days of their publication.
He divides the excessively wide 10-59 year age group into 10 groups, each 5 years wide.

To estimate normal, pre-pandemic death rates in these groups, he uses 2019 data from the U.K. Office of National Statistics.
To estimate the vaccination rate in each group, he uses data from the coronavirus dashboard provided by the U.K. Health Security Agency.
Finally, to estimate the fraction of people in each age group, he uses data from a web site called PopulationPyramid, which organizes and publishes data from the United Nations Department of Economic and Social Affairs.
Not to get lost in the details, these are all reliable data sources.

Combining these datasets, Morris is able to compute the distribution of ages in the vaccinated and unvaccinated groups at the beginning of August 2021 (choosing a point near the middle of the interval in the original graph).
The following figure shows the results.

```python tags=["hide-cell"]
index = [
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
]
```

```python tags=["hide-cell"]
rate = [8.8, 23.1, 39.1, 48.4, 66.9, 97.3, 143.3, 219.6, 321.5, 478.2]
pct_vax = [0, 26.7, 59.6, 59.7, 63.3, 63.3, 74.6, 80.5, 85.6, 88.2]
pct_pop = [5.9, 5.5, 6.2, 6.8, 6.7, 6.6, 6.0, 6.6, 7.0, 6.6]
```

```python tags=["hide-cell"]
rate[-1] / rate[0]
```

```python tags=["hide-input"]
sheet = pd.DataFrame(dict(rate=rate, pct_vax=pct_vax, pct_pop=pct_pop), index)
sheet
```

```python tags=["hide-input"]
sheet["pct_pop"] /= sheet["pct_pop"].sum()
sheet
```

```python tags=["hide-input"]
sheet["pct_vax_age"] = sheet["pct_pop"] * sheet["pct_vax"]
sheet["pct_vax_age"] /= sheet["pct_vax_age"].sum()
sheet
```

```python tags=["hide-input"]
sheet["pct_novax_age"] = sheet["pct_pop"] * (100 - sheet["pct_vax"])
sheet["pct_novax_age"] /= sheet["pct_novax_age"].sum()
sheet
```

```python tags=["hide-input"]
from empiricaldist import Pmf

pmf_novax = Pmf(sheet["pct_novax_age"], copy=True)
pmf_vax = Pmf(sheet["pct_vax_age"], copy=True)

pmf_novax.plot(ls="--", color="C4", label="Unvaccinated")
pmf_vax.plot(color="C2", label="Vaccinated")

decorate(
    xlabel="Age group",
    ylabel="Fraction of population",
    title="Distribution of ages",
    ylim=[0, 0.26],
)
```

```python tags=["hide-input"]
xs = [12, 17, 22, 27, 32, 37, 42, 47, 52, 57]

ps = sheet["pct_novax_age"]
sns.kdeplot(
    x=xs, weights=ps, cut=0, ls="--", color="C4", label="Unvaccinated", alpha=0.6
)

ps = sheet["pct_vax_age"]
sns.kdeplot(x=xs, weights=ps, cut=0, color="C2", label="Vaccinated", alpha=0.6)

decorate(
    xlabel="Age group",
    ylabel="Likelihood",
    title="Distribution of ages",
)
plt.yticks([0.005, 0.015, 0.025])
```

```python tags=["hide-cell"]
pmf_vax.index = xs
pmf_novax.index = xs
pmf_vax.mean(), pmf_novax.mean()
```

At this point during the rollout of the vaccines, people who were vaccinated were more likely to be at the high end of the age range, and people who were unvaccinated were more likely to be at the low end.
Among the vaccinated, the average age was about 40; among the unvaccinated, it was 27.

Now, for the sake of this example, let's imagine that there are no deaths due to COVID, and no deaths due to the vaccine.
Based only on the distribution of ages and the death rates from 2019, we can compute the expected death rates for the vaccinated and unvaccinated groups.
The ratio of these two rates is about 2.4.

So, prior to the pandemic, we would expect people with the age distribution of the vaccinated to die at a rate 2.4 times higher than people with the age distribution of the unvaccinated, just because they are older.

In reality, the pandemic increased the death rate in both groups, so the actual ratio was smaller, about 1.8.
Thus, the results the journalist presented are not evidence that vaccines are harmful; as Morris concludes, they are "not unexpected, and can be fully explained by the Simpson's paradox artifact".

```python tags=["hide-input"]
sheet
```

```python tags=["hide-input"]
sheet["death_vax"] = sheet["rate"] * sheet["pct_vax_age"]
sheet["death_novax"] = sheet["rate"] * sheet["pct_novax_age"]
sheet
```

```python tags=["hide-cell"]
sheet["death_vax"].sum(), sheet["death_novax"].sum()
```

```python tags=["hide-cell"]
sheet["death_vax"].sum() / sheet["death_novax"].sum()
```

## Debunk Redux

So far we have used only data that was publicly available in November 2021, but if we take advantage of more recent data, we can get a clearer picture of what was happening then and what has happened since.

In more recent reports, data from the U.K. Office of National Statistics are broken into smaller age groups.
Instead of one group from ages 10 to 59, we have three groups from 18 to 39, 40 to 49, and 50 to 59.

```python tags=["hide-cell"]
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/deathsbyvaccinationstatusengland
# https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/deathsbyvaccinationstatusengland/deathsoccurringbetween1january2021and31march2022/referencetable20220516accessiblecorrected.xlsx
```

```python tags=["remove-cell"]
filename = "referencetable20220516accessiblecorrected.xlsx"
download(DATA_PATH + filename)
```

```python tags=["hide-input"]
df = pd.read_excel(
    filename,
    sheet_name="Table 2",
    skiprows=3,
)
df.tail()
```

```python tags=["hide-cell"]
df.columns
```

```python tags=["hide-cell"]
df["date"] = pd.to_datetime(df["Year"].astype(str) + " " + df["Month"])
```

```python tags=["hide-cell"]
rate = "Age-standardised mortality rate / 100,000 person-years"
df["rate"] = df[rate].replace("x", np.nan) / 52
```

```python tags=["hide-cell"]
count = "Count of deaths"
df = df.replace({count: {"<3": np.nan}})
```

```python tags=["hide-cell"]
all_cause = df["Cause of Death"] == "All causes"
all_cause_df = df[all_cause]
all_cause_df.shape
```

```python tags=["hide-input"]
table = all_cause_df.pivot_table(
    index="date",
    columns=["Age group", "Vaccination status"],
    values="rate",
    aggfunc="median",
)
table
```

```python tags=["hide-input"]
weight = all_cause_df.pivot_table(
    index="date",
    columns=["Age group", "Vaccination status"],
    values="Person-years",
    aggfunc="median",
)
weight
```

```python tags=["hide-cell"]
all_cause_df["Vaccination status"].value_counts()
```

```python tags=["hide-cell"]
unvax_stat = ["Unvaccinated"]

vax_stat = [
    "Second dose, less than 21 days ago",
    "Second dose, at least 21 days ago",
    "Third dose or booster, less than 21 days ago",
    "Third dose or booster, at least 21 days ago",
]
```

```python tags=["hide-cell"]
def weighted_sum(table, weight):
    """Weighted sum of table and weight.

    table: DataFrame
    weight: DataFrame

    returns: Series
    """
    # make sure that weight is 0 where table is NaN
    a = np.where(table.notna(), weight, 0)
    weight = pd.DataFrame(a, index=weight.index, columns=weight.columns)

    # normalize the weights rowwise
    total = weight.sum(axis=1)
    weight_norm = weight.divide(total, axis=0)

    # compute the weighted sum of table
    prod = table.fillna(0) * weight_norm
    return prod.sum(axis=1).replace(0, np.nan)
```

```python tags=["hide-cell"]
def get_groups(frame, groups, status):
    """Select age groups and vax status.

    frame: DataFrame (table or weights)
    groups: list of string age group names
    status: list of string vax status

    returns: DataFrame
    """
    seq = [frame[group][status] for group in groups]
    return pd.concat(seq, axis=1)
```

```python tags=["hide-cell"]
def make_plot(table, weight, groups):
    """Plot mortality for vaxed and unvaxed.

    table: DataFrame
    weight: DataFrame
    groups: list of string age groups
    """
    t = get_groups(table, groups, unvax_stat)
    w = get_groups(weight, groups, unvax_stat)
    unvax = weighted_sum(t, w)

    t = get_groups(table, groups, vax_stat)
    w = get_groups(weight, groups, vax_stat)
    vax = weighted_sum(t, w)

    ages = ",".join(groups)
    unvax.plot(ls="--", label="Unvaccinated")
    vax.plot(label="Vaccinated")

    decorate(
        xlabel="",
        ylabel="Weekly deaths per 100k",
        title=f"All cause death rates, ages {ages}",
    )

    # adjust the axes
    ax = plt.gca()
    low, high = plt.gca().get_ylim()
    plt.ylim(0, high)

    start = pd.Timestamp("2020-12-15")
    end = pd.Timestamp("2022-04-01")
    plt.xlim(start, end)
```

```python tags=["hide-input"]
# This figure shows what happens if we combine the youngest three groups,
# as in the original data
groups = ["18-39", "40-49", "50-59"]
make_plot(table, weight, groups)
```

```python tags=["hide-input"]
# the following figures look at the older age groups, which I won't present again

groups = ["90+"]
make_plot(table, weight, groups)
```

```python tags=["hide-input"]
groups = ["80-89"]
make_plot(table, weight, groups)
```

```python tags=["hide-input"]
groups = ["70-79"]
make_plot(table, weight, groups)
```

```python tags=["hide-input"]
groups = ["60-69"]
make_plot(table, weight, groups)
```

Let's start with the oldest of these subgroups. 
Among people in their 50s, we see that the all-cause death rate was substantially higher among unvaccinated people over the entire interval from March 2021 to April 2022.

```python tags=["hide-input"]
groups = ["50-59"]
make_plot(table, weight, groups)
plt.ylim([0, 24])
plt.legend(loc="upper right")
```

If we select people in their 40s, we see the same thing: death rates were higher for unvaccinated people over the entire interval.

```python tags=["hide-input"]
groups = ["40-49"]
make_plot(table, weight, groups)
plt.ylim([0, 7.8])
plt.legend(loc="upper right")
```

However, it is notable that death rates in both groups have declined since April or May 2021, and have nearly converged.
Possible explanations include improved treatment for coronavirus disease and the decline of more lethal variants of the COVID virus.
In the U.K., December 2021 is when the Delta variant was largely replaced by the Omicron variant, which is less deadly.

Finally, let's see what happened in the youngest group, people aged 18 to 39.

```python tags=["hide-input"]
groups = ["18-39"]
make_plot(table, weight, groups)
```

In this age group, like the others, death rates are generally higher among unvaccinated people.
There are a few months where the pattern is reversed, but we should not take it too seriously because:

* Death rates in this age group are very low, less than 1 per 100,000 per week, vaccinated or not.

* The apparent differences between the groups are even smaller, and might be the result of random variation.

* Finally, this age group is still broad enough that the results are influenced by Simpson's paradox. The vaccinated people in this group were older than the unvaccinated, and the older people in this group were about 10 times more likely to die, just because of their age.

In summary, the excess mortality among vaccinated people in 2021 is entirely explained by Simpson's paradox.
If we break the dataset into appropriate age groups, we see that death rates were higher among the unvaccinated in all age groups over the entire time from the rollout of the vaccine to the present.
And we can conclude that the vaccine saved many lives.

```python tags=["hide-cell"]
may = (df["Year"] == 2021) & (df["Month"] == "May")
may.sum()
```

```python tags=["hide-cell"]
fully = df["Vaccination status"] == "Second dose, at least 21 days ago"
fully.sum()
```

```python tags=["hide-cell"]
age = df["Age group"] == "18-39"
age.sum()
```

```python tags=["hide-input"]
df[may & fully & age]
```

```python tags=["hide-input"]
sheet
```

```python tags=["hide-cell"]
mortality = sheet["rate"] / 52
mortality
```

```python tags=["hide-cell"]
mortality["55-59"], mortality["20-24"]
```

```python tags=["hide-cell"]
mortality["55-59"] / mortality["20-24"]
```

```python tags=["hide-cell"]
sheet.loc["55-59", "pct_vax"], sheet.loc["20-24", "pct_vax"]
```

```python tags=["hide-cell"]
sheet.loc["55-59", "pct_vax"] / sheet.loc["20-24", "pct_vax"]
```

## Lives saved

```python tags=["remove-cell"]
filename = "referencetable20220516accessiblecorrected.xlsx"
download(DATA_PATH + filename)
```

```python tags=["hide-input"]
df = pd.read_excel(
    filename,
    sheet_name="Table 2",
    skiprows=3,
)
df.tail()
```

```python tags=["hide-cell"]
df["date"] = pd.to_datetime(df["Year"].astype(str) + " " + df["Month"])
```

```python tags=["hide-cell"]
all_cause = df["Cause of Death"] == "All causes"
all_cause.sum()
```

```python tags=["hide-cell"]
rate_col = "Age-standardised mortality rate / 100,000 person-years"
df = df.replace({rate_col: {"x": np.nan}})
df[rate_col].describe()
```

```python tags=["hide-cell"]
def estimate_saved_lives(period):
    """Estimate lives saved by comparing actual deaths to possible deaths.
    
    period: DataFrame with vaccination status, deaths, population, and rates
    
    returns: number of lives saved (actual deaths - possible deaths)
    """
    index = period["Vaccination status"]
    deaths = pd.Series(period["Count of deaths"].values, index=index)
    pop = pd.Series(period["Person-years"].values, index=index)
    rate = pd.Series(period[rate_col].values, index=index)

    # choose the minimum rate among the fully vaccinated / boosted
    possible_rate = rate.iloc[4:].min() / 1e5
    actual_deaths = deaths["Unvaccinated"]
    possible_deaths = possible_rate * pop["Unvaccinated"]
    return actual_deaths - possible_deaths
```

```python tags=["hide-input"]
res = []
for name, period in df[all_cause].groupby(["Age group", "date"]):
    saved = estimate_saved_lives(period)
    res.append((name[0], name[1], saved))

saved_df = pd.DataFrame(res, columns=["Age group", "date", "saved"])
saved_df
```

```python tags=["hide-cell"]
series = saved_df.groupby("date")["saved"].sum()
series
```

```python tags=["hide-cell"]
index = period["Vaccination status"]
deaths = pd.Series(period["Count of deaths"].values, index=index)
deaths

pop = pd.Series(period["Person-years"].values, index=index)
pop

rate = pd.Series(period[rate_col].values, index=index)
rate

boosted = "Third dose or booster, at least 21 days ago"
possible_rate = rate.iloc[4:].min() / 1e5
possible_rate

pop[boosted] * possible_rate

deaths[boosted]

actual_deaths = deaths["Unvaccinated"]

possible_deaths = possible_rate * pop["Unvaccinated"]
actual_deaths, possible_deaths
```

## Sources and related reading

* The General Social Survey (GSS) is a project of the University of Chicago's independent research organization NORC, with primary financing from the National Science Foundation. Data is available from the [GSS website](https://gss.norc.org/Get-The-Data).

* Floyd Norris wrote about changes in real wages in *The New York Times* ([Norris, 2013](https://www.nytimes.com/2013/04/27/business/economy/wage-disparity-continues-to-grow.html)). The following week, he wrote a blog article to respond to readers who thought he had made a mistake ([Norris, 2013](https://economix.blogs.nytimes.com/2013/05/01/can-every-group-be-worse-than-average-yes)).

* The IPUMS CPS data is from the U.S. Census Bureau: Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler, and Michael Westberry. Integrated Public Use Microdata Series, Current Population Survey: Version 11.0 [dataset]. Minneapolis, MN: IPUMS, 2023. https://doi.org/10.18128/D030.V11.0

* The penguin dataset was reported in "Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins" (Gorman et al., 2014). Allison Horst pointed out the penguin paradox in the [online documentation of the dataset](https://allisonhorst.github.io/palmerpenguins/articles/examples.html).

* The nested Simpson's paradox among dog breeds is described in "Do large dogs die young?" (Galis et al., 2007).

* The paper that evaluates treatments for kidney stones was published in 1986 in the *British Medical Journal* (Charig et al., 1986).

* Derek Thompson wrote about the misleading journalist in *The Atlantic* ([Thompson, 2021](https://www.theatlantic.com/ideas/archive/2021/04/pandemics-wrongest-man/618475)). Jeffrey Morris debunks the misleading newsletter in a series of blog articles ([Morris, 2021](https://www.covid-datascience.com/post/what-do-uk-data-say-about-real-world-impact-of-vaccines-on-all-cause-deaths)).

* Information about the prevalence of different COVID variants is from the U.K. Health Security Agency ([U.K. Health Security Agency, 2022](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1093275/covid-technical-briefing-44-22-july-2022.pdf)).

* The COVID data I used is from the U.K. Office for National Statistics ([U.K. Office for National Statistics, 2022](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/deathsbyvaccinationstatusengland)).


## Open data, open discussion

Reflecting on this episode, we can identify several elements that went wrong.
A journalist downloaded data he did not have the expertise to interpret correctly.
He wrote about it in a newsletter that was published with no editorial review.
Its spread was accelerated by a credulous podcaster who ignored numerous easily discoverable articles debunking its claims.
And, given the circulation it achieved, it contributed to a body of vaccine misinformation that almost certainly caused preventable deaths.

All that is true; nevertheless, it is also important to consider the elements that went right.

First, the U.K. government collected reliable data about the pandemic and made it widely available.
Other countries followed their example, albeit some more successfully than others. 
Groups like Our World in Data collected similar data from around the world and made it easier to use, which accelerated the ability of researchers to understand the pandemic, make predictions, and guide public policy decisions.
This data saved lives.

Second, after the misleading newsletter was posted on November 20, dozens of articles debunked its claims within days.
If you do a web search for the article today, the first page of results includes rebuttals published on November 21, 22, and 23.

Finally, even on social media sites that are infamous for spreading disinformation, public discussion of the article was not all bad.
Simpson's paradox is not easy to understand, but some of the explanations were very good.
I think it's a hopeful sign to see people grappling with statistical ideas in a public forum.

The marketplace of ideas will never be perfect, but this episode, which shows its weaknesses, also shows its strengths.
We are better off when our decisions are guided by evidence and reason, but that is only possible if we have access to data and open spaces to discuss difficult ideas.


## Israeli Data

This is an analysis I started with another dataset that demonstrates a Simpson's paradox if you don't separate age groups.

Downloaded from https://www.covid-datascience.com/post/israeli-data-how-can-efficacy-vs-severe-disease-be-strong-when-60-of-hospitalized-are-vaccinated

Originally from an [Israeli government COVID dashboard](https://datadashboard.health.gov.il/COVID-19/general) with headings translated by Jeffrey S. Morris, director of the Biostatistics Division at the University of Pennsylvania.

https://www.dbei.med.upenn.edu/bio/jeffrey-s-morris-phd

```python tags=["remove-cell"]
filename = "Israeli_data_August_15_2021.xlsx"
download(DATA_PATH + filename)
```

```python tags=["hide-input"]
df = pd.read_excel(filename, skiprows=2).loc[0:9]
df.index = df["age group"]
del df["age group"]
df
```

```python tags=["hide-cell"]
df.columns
```

```python tags=["hide-cell"]
overall = df.sum()
overall
```

```python tags=["hide-cell"]
overall_risk_vax = overall["severe vax"] / overall["population vax"] * 1e5
overall_risk_vax
```

```python tags=["hide-cell"]
overall_risk_unvax = overall["severe unvax"] / overall["population unvax"] * 1e5
overall_risk_unvax
```

```python tags=["hide-cell"]
age_group = df.index
risk_vax = pd.Series(df["severe vax 100k"].values, index=age_group)
risk_unvax = pd.Series(df["severe unvax per 100k"].values, index=age_group)
risk_vax
```

```python tags=["hide-cell"]
table = pd.DataFrame(columns=["age_group"])
```

Probably Overthinking It

Copyright 2022 Allen Downey 

The code in this notebook and `utils.py` is under the [MIT license](https://mit-license.org/).

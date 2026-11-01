"""The Chapter 12 (Overton) pipeline, factored out of overton.md.

Same method as the book: fifteen questions, difficulties from the logit
of each question's prevalence, a grid posterior for each respondent's
efficacy with a wide Gaussian prior, and conservatism reported as the
posterior mean number of conservative responses out of 15.

Runs the pipeline on one or more datasets, prints the key numbers from
the chapter side by side, and saves the chapter's figures for each
dataset in overton_figs/<label>/.

Also runs two checks on the 2024 data:

- Mode sensitivity: since 2022, five questions were asked only in person
  or by phone. Rerun (a) with data through 2018 only and (b) without those
  five questions after 2018, and (c) without them in every year.
- Direct weighting: run on the unresampled extract, with weighted means
  for the difficulties and the yearly averages.

Usage: python overton_pipeline.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.special import expit, logit
from scipy.stats import norm

from overton_data import CONSERVATIVE_VALUES, EXTRACT, RESAMPLED, WEIGHT
from utils import (
    chunk_series,
    decorate,
    label_table,
    make_counterfact,
    make_lowess,
    make_table,
    plot_series_lowess,
    prepare_yvar,
    run_subgroups,
    set_pyplot_params,
    visualize_table,
)

HERE = Path(__file__).resolve().parent
BOOK_FILE = HERE / "gss_pacs_resampled.hdf"
FIG_DIR = HERE / "overton_figs"

ORDER = [
    "homosex",
    "cappun",
    "grass",
    "abany",
    "prayer",
    "premarsx",
    "divlaw",
    "natfare",
    "natenvir",
    "pornlaw",
    "racopen",
    "letdie1",
    "fepol",
    "gunlaw",
    "sexeduc",
]

ISSUE_DICT = {
    "abany": "Abortion",
    "homosex": "Homosexuality",
    "premarsx": "Premarital sex",
    "prayer": "School prayer",
    "natfare": "Welfare spending",
    "grass": "Legal cannabis",
    "natenvir": "Environment",
    "divlaw": "Divorce",
    "cappun": "Capital punishment",
    "racopen": "Fair housing law",
    "letdie1": "Assisted suicide",
    "fepol": "Women in politics",
    "gunlaw": "Gun law",
    "sexeduc": "Sex education",
    "pornlaw": "Pornography",
}

RECODE_POLVIEWS = {
    1: "Liberal",
    2: "Liberal",
    3: "Liberal",
    4: "Moderate",
    5: "Conservative",
    6: "Conservative",
    7: "Conservative",
}

POLVIEW_NAMES = ["Liberal", "Moderate", "Conservative"]
muted = sns.color_palette("muted", 5)
POLVIEW_MAP = dict(zip(POLVIEW_NAMES, [muted[0], muted[4], muted[3]]))
MARKER_MAP = dict(zip(POLVIEW_NAMES, ["o", "^", "s"]))
LINE_MAP = dict(zip(POLVIEW_NAMES, [":", "-.", "--"]))
OPTIONS = dict(frac=0.75, plot_series=False)

# asked only in person or by phone since 2022, and not at all in 2021
IN_PERSON_ONLY = ["grass", "prayer", "divlaw", "racopen", "fepol"]

# grid of efficacies and the prior
XS = np.linspace(-6, 6, 21)
PRIOR = norm.pdf(XS, 0, 4)
PRIOR /= PRIOR.sum()


def load_gss(filename, key):
    """Read one resampling and add cohort10 and polviews3."""
    gss = pd.read_hdf(filename, key)
    gss["year"] = gss["year"].astype(int)

    bins = np.arange(1889, 2011, 10)
    labels = bins[:-1] + 1
    gss["cohort10"] = pd.cut(gss["cohort"], bins, labels=labels).astype(float)
    gss = gss.dropna(subset=["cohort10"]).copy()
    gss["cohort10"] = gss["cohort10"].astype(int)

    gss["polviews3"] = gss["polviews"].replace(RECODE_POLVIEWS)
    return gss


def make_questions(gss, codes=None):
    """1 for a conservative response, 0 for other responses, NaN if not answered.

    codes: map from question to conservative codes; default, the book's fifteen
    """
    if codes is None:
        codes = {varname: CONSERVATIVE_VALUES[varname] for varname in ORDER}
    questions = pd.DataFrame(dtype=float, index=gss.index)
    for varname, values in codes.items():
        questions[varname] = gss[varname].isin(values).astype(float)
        questions.loc[gss[varname].isna(), varname] = np.nan
    return questions


def weighted_means(questions, weights):
    """Weighted mean of each column, ignoring NaN."""
    valid = questions.notna()
    totals = (questions.fillna(0).mul(weights, axis=0)).sum()
    return totals / valid.mul(weights, axis=0).sum()


def estimate_conservatism(questions, weights=None):
    """Grid posterior for each respondent.

    weights: if provided, difficulties come from weighted means

    returns: posterior mean and std of the number of conservative responses,
             and the difficulties
    """
    if weights is None:
        means = pd.to_numeric(questions.mean())
    else:
        means = weighted_means(questions, weights)
    difficulty = -logit(means)

    X, D = np.meshgrid(XS, difficulty)
    P = expit(X - D)
    Q = 1 - P
    ns = P.sum(axis=0)

    n, m = questions.shape
    res = np.empty((n, m, len(XS)))
    a = questions.fillna(2).astype(int).values
    ii, jj = np.nonzero(a == 0)
    res[ii, jj, :] = Q[jj]
    ii, jj = np.nonzero(a == 1)
    res[ii, jj, :] = P[jj]
    ii, jj = np.nonzero(a == 2)
    res[ii, jj, :] = 1

    product = res.prod(axis=1) * PRIOR
    posterior = product / product.sum(axis=1)[:, None]

    con = (posterior * ns).sum(axis=1)
    deviations = ns - con[:, None]
    std_con = np.sqrt((posterior * deviations**2).sum(axis=1))
    return con, std_con, difficulty


def run_pipeline(gss, weights=None, codes=None):
    """Add the conservatism column; return the questions and difficulties."""
    # drop questions nobody answered (used by the drop-5-all-years variant);
    # then conservatism is out of the number of remaining questions
    questions = make_questions(gss, codes).dropna(axis=1, how="all")
    con, std_con, difficulty = estimate_conservatism(questions, weights)
    answered = questions.notna().sum(axis=1)
    gss["conservatism"] = pd.Series(con, gss.index)
    gss.loc[answered < 5, "conservatism"] = np.nan
    return questions, difficulty


def count_signs(params):
    """Number of subgroups with significant negative and positive slopes."""
    sig = params.iloc[1:]["pvalue"] < 0.02
    neg = (params.iloc[1:]["param"] < 0) & sig
    pos = (params.iloc[1:]["param"] > 0) & sig
    return len(params) - 1, neg.sum(), pos.sum()


def fit_period_cohort(gss):
    """OLS of conservatism on year and cohort, as in the chapter."""
    data = pd.DataFrame(dict(y=gss["conservatism"], cohort=gss["cohort"], year=gss["year"]))
    results = smf.ols("y ~ year + cohort", data=data).fit()
    return results, data


def key_numbers(gss, questions, difficulty):
    """The numbers the chapter reports or depends on."""
    k = {}
    k["rows"] = len(gss)
    k["first year"] = gss["year"].min()
    k["last year"] = last = gss["year"].max()
    answered = questions.notna().sum(axis=1)
    k["frac answered >= 5"] = (answered >= 5).mean()
    k["conservatism mean"] = gss["conservatism"].mean()
    k["conservatism std"] = gss["conservatism"].std()

    augmented = pd.concat([questions, gss["polviews3"]], axis=1)
    table = augmented.groupby("polviews3").mean().transpose() * 100
    gap = table["Conservative"] - table["Liberal"]
    k["con-lib gap min"] = gap.min()
    k["con-lib gap max"] = gap.max()
    k["questions"] = questions.shape[1]
    for varname in ["homosex", "sexeduc"]:
        k[f"difficulty {varname}"] = difficulty.get(varname, np.nan)

    # percent identifying as conservative: overall, first and last year
    xtab = pd.crosstab(gss["year"], gss["polviews3"], normalize="index") * 100
    k["% conservative ID, first year"] = xtab["Conservative"].iloc[0]
    k["% conservative ID, last year"] = xtab["Conservative"].iloc[-1]

    # conservatism vs age within cohorts
    groups, neg, pos = count_signs(
        run_subgroups(gss, "age", "conservatism", "cohort10", "continuous")
    )
    k["age slopes: groups / neg / pos"] = f"{groups} / {neg} / {pos}"

    # conservatism vs year
    prepare_yvar(gss, "conservatism", "continuous")
    series = chunk_series(gss, "year", size=2000)
    smooth = make_lowess(series)
    k["smoothed conservatism, start"] = smooth.iloc[0]
    k["smoothed conservatism, end"] = smooth.iloc[-1]

    groups, neg, pos = count_signs(
        run_subgroups(gss, "year", "conservatism", "cohort10", "continuous")
    )
    k["year slopes: groups / neg / pos"] = f"{groups} / {neg} / {pos}"

    results, _ = fit_period_cohort(gss)
    k["cohort / year effect"] = results.params["cohort"] / results.params["year"]

    # conservatism by alignment over time
    table = gss.pivot_table(index="year", columns="polviews3", values="conservatism")
    con = make_lowess(table["Conservative"])
    lib = make_lowess(table["Liberal"])
    diff = con - lib
    k["con-lib gap 1974"] = diff[1974]
    k["con-lib gap 2018"] = diff[2018]
    k["con-lib gap 2021"] = diff.get(2021, np.nan)
    k[f"con-lib gap {last}"] = diff[last]
    for column in ["Conservative", "Moderate", "Liberal"]:
        smooth = make_lowess(table[column])
        k[f"change, {column}"] = smooth.iloc[-1] - smooth.iloc[1]
    k["Liberal 1974"] = make_lowess(table["Liberal"])[1974]
    k["Moderate 2000"] = make_lowess(table["Moderate"])[2000]
    k["Conservative 2018"] = make_lowess(table["Conservative"])[2018]
    k["Conservative 2021"] = make_lowess(table["Conservative"]).get(2021, np.nan)
    k[f"Conservative {last}"] = make_lowess(table["Conservative"])[last]
    return k


def savefig(outdir, name):
    plt.savefig(outdir / name, dpi=150)
    plt.close()


def make_figures(gss, questions, outdir):
    """The chapter's figures, with the chapter's file names."""
    outdir.mkdir(parents=True, exist_ok=True)
    last = gss["year"].max()

    # overton1: alignment vs age
    xtab = pd.crosstab(gss["age"], gss["polviews3"], normalize="index") * 100
    for column in ["Moderate", "Conservative", "Liberal"]:
        plot_series_lowess(xtab[column], ls=LINE_MAP[column], color=POLVIEW_MAP[column], **OPTIONS)
    decorate(xlabel="Age", ylabel="Percent", title="Political alignment vs age")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    savefig(outdir, "overton1.png")

    # overton2: percent conservative vs age, by cohort
    table = make_table(gss, "age", "polviews3", "cohort10", "Conservative")
    table.pop(1900, None)
    table.pop(1910, None)
    series = chunk_series(gss, "age") * 100
    visualize_table(series, table, frac=0.8)
    label_table(table, frac=0.8)
    decorate(
        xlabel="Age",
        title="Percentage identifying as conservative, grouped by decade of birth",
        legend=False,
    )
    savefig(outdir, "overton2.png")

    # overton3: percent conservative responses by question and alignment
    augmented = pd.concat([questions, gss["polviews3"]], axis=1)
    table = augmented.groupby("polviews3").mean().transpose() * 100
    issue_names = [ISSUE_DICT.get(v, v) for v in questions.columns]
    figwidth = 3.835
    plt.figure(figsize=(figwidth, 0.8 * figwidth))
    plt.hlines(issue_names, table["Liberal"], table["Conservative"], color="gray", alpha=0.2)
    for polviews, color in POLVIEW_MAP.items():
        plt.plot(table[polviews], issue_names, MARKER_MAP[polviews], color=color, alpha=0.9, label=polviews)
    decorate(xlabel="Percent choosing conservative responses")
    savefig(outdir, "overton3.png")

    # overton4: conservatism vs age, by alignment
    series = gss.groupby("age")["conservatism"].mean()
    table = gss.pivot_table(index="age", columns="polviews3", values="conservatism")
    for column in ["Conservative", "Moderate", "Liberal"]:
        plot_series_lowess(table[column], ls=LINE_MAP[column], color=POLVIEW_MAP[column], **OPTIONS)
    plot_series_lowess(series, ls="-", color="gray", alpha=0.5, label="overall")
    decorate(xlabel="Age", title="Conservative responses vs age, grouped by alignment")
    plt.yticks([3, 5, 7, 9])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    savefig(outdir, "overton4.png")

    # overton5: conservatism vs age, by cohort
    table = make_table(gss, "age", "conservatism", "cohort10", "continuous")
    series = chunk_series(gss, "age")
    visualize_table(series, table, frac=0.8)
    label_table(table, frac=0.8)
    decorate(
        xlabel="Age",
        title="Conservative responses vs age, grouped by decade of birth",
        legend=False,
    )
    plt.yticks([5, 7, 9])
    savefig(outdir, "overton5.png")

    # overton6: conservatism vs year
    prepare_yvar(gss, "conservatism", "continuous")
    series_conservatism = chunk_series(gss, "year", size=2000)
    plot_series_lowess(series_conservatism, plot_series=True, ls=":", color="C0", alpha=0.5, label="")
    decorate(xlabel="Year of survey", title="Conservative responses vs year")
    plt.yticks([5, 6, 7, 8])
    savefig(outdir, "overton6.png")

    # overton12: each question vs year
    over_time = questions.groupby(gss["year"]).mean()
    for column in over_time.columns:
        plot_series_lowess(over_time[column] * 100, color="C0", ls="-", alpha=0.5)
    decorate(
        xlabel="Year of survey",
        ylabel="Percent",
        title="Conservative responses vs year, all questions",
        legend=False,
    )
    savefig(outdir, "overton12.png")

    # overton7: conservatism vs year, by cohort
    cohort_table = make_table(gss, "year", "conservatism", "cohort10", "continuous")
    visualize_table(series_conservatism, cohort_table)
    label_table(cohort_table)
    decorate(
        xlabel="Year of survey",
        title="Conservative responses vs year, grouped by decade of birth",
        legend=False,
    )
    savefig(outdir, "overton7.png")

    # overton8: counterfactual period-only and cohort-only models
    results, data = fit_period_cohort(gss)
    data["predict"] = results.predict(data)
    pred = data.groupby("year")["predict"].mean()
    y = 1972
    c = data.query("year==1972")["cohort"].mean()
    b0, b1, b2 = results.params
    cohort_only = make_counterfact(results, "year", b0 + b1 * y, data)
    period_only = make_counterfact(results, "cohort", b0 + b2 * c, data)
    series_conservatism.plot(style="o", color="C0", alpha=0.2, label="Data")
    period_only.plot(ls="--", color="C1", label="Period only")
    cohort_only.plot(ls=":", color="C2", label="Cohort only")
    pred.plot(color="C0", label="Both")
    decorate(
        xlabel="Year of survey",
        title="Conservative responses vs year, with counterfactual models",
        ylim=[4.5, 8.5],
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    savefig(outdir, "overton8.png")

    # overton9: alignment vs year
    xtab = pd.crosstab(gss["year"], gss["polviews3"], normalize="index") * 100
    for column in ["Moderate", "Conservative", "Liberal"]:
        plot_series_lowess(xtab[column], ls=LINE_MAP[column], color=POLVIEW_MAP[column], **OPTIONS)
    decorate(xlabel="Year of survey", title="Political alignment vs year", ylabel="Percent", ylim=[16, 45])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    savefig(outdir, "overton9.png")

    # overton10b and overton11*: conservatism vs year, by alignment, with a cohort
    table = gss.pivot_table(index="year", columns="polviews3", values="conservatism")
    level = make_lowess(table["Liberal"])[1974]

    def plot_alignment():
        for column in ["Conservative", "Moderate", "Liberal"]:
            plot_series_lowess(table[column], ls=LINE_MAP[column], color=POLVIEW_MAP[column], **OPTIONS)

    plot_alignment()
    plt.hlines(level, 1974, last, color="gray", alpha=0.5)
    decorate(xlabel="Year of survey", title="Conservative responses vs year, grouped by alignment")
    plt.yticks([3, 5, 7, 9])
    savefig(outdir, "overton10b.png")

    for cohort, name in [(1940, "overton11"), (1960, "overton11b"), (1980, "overton11d")]:
        if cohort not in cohort_table:
            continue
        plot_alignment()
        plot_series_lowess(cohort_table[cohort], color="gray", alpha=0.5, label=f"{cohort}s")
        decorate(xlabel="Year of survey", title="Conservative responses vs year, grouped by alignment")
        plt.yticks([3, 5, 7, 9])
        savefig(outdir, f"{name}.png")


def through_2018(gss):
    """Variant: drop 2021 and later."""
    return gss[gss["year"] <= 2018].copy()


def drop_in_person_only(gss):
    """Variant: drop the in-person-only questions after 2018."""
    gss = gss.copy()
    gss.loc[gss["year"] > 2018, IN_PERSON_ONLY] = np.nan
    return gss


def drop_in_person_all_years(gss):
    """Variant: drop the in-person-only questions in every year."""
    gss = gss.copy()
    gss[IN_PERSON_ONLY] = np.nan
    return gss


def run(label, filename, key, figures=True, transform=None, codes=None):
    gss = load_gss(filename, key)
    if transform is not None:
        gss = transform(gss)
    questions, difficulty = run_pipeline(gss, codes=codes)
    k = key_numbers(gss, questions, difficulty)
    if figures:
        make_figures(gss, questions, FIG_DIR / label)
    return pd.Series(k, name=label)


def yearly_table(gss, weights=None):
    """Mean conservatism by year, overall and by alignment."""
    if weights is None:
        weights = pd.Series(1.0, gss.index)
    data = gss.assign(w=weights).dropna(subset=["conservatism"])
    data["wc"] = data["w"] * data["conservatism"]

    def wmean(df):
        return df["wc"].sum() / df["w"].sum()

    table = data.groupby(["year", "polviews3"]).apply(wmean).unstack()
    table["All"] = data.groupby("year").apply(wmean)
    return table[["All", "Conservative", "Moderate", "Liberal"]]


def resampled_yearly(transform=None):
    """yearly_table averaged over the three resamplings."""
    tables = []
    for key in ["gss0", "gss1", "gss2"]:
        gss = load_gss(RESAMPLED, key)
        if transform is not None:
            gss = transform(gss)
        run_pipeline(gss)
        tables.append(yearly_table(gss))
    return sum(tables) / len(tables)


def weighted_yearly():
    """yearly_table from the unresampled extract, with weights."""
    gss = load_gss(EXTRACT, "gss")
    run_pipeline(gss, weights=gss[WEIGHT])
    return yearly_table(gss, gss[WEIGHT])


def checks():
    """Mode sensitivity and direct weighting, yearly means from 2010."""
    tables = {
        "resampled": resampled_yearly(),
        "through 2018": resampled_yearly(through_2018),
        "drop 5 after 2018": resampled_yearly(drop_in_person_only),
        "drop 5 all years": resampled_yearly(drop_in_person_all_years),
        "weighted": weighted_yearly(),
    }
    combined = pd.concat(tables, axis=1).loc[2010:]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(combined)
    combined.to_csv(FIG_DIR / "yearly_checks.csv")


def main():
    np.random.seed(17)
    set_pyplot_params()
    columns = [
        run("book", BOOK_FILE, "gss0"),
        run("new", RESAMPLED, "gss0"),
        run("new1", RESAMPLED, "gss1", figures=False),
        run("new2", RESAMPLED, "gss2", figures=False),
        run("thru2018", RESAMPLED, "gss0", figures=False, transform=through_2018),
        run("drop5", RESAMPLED, "gss0", figures=False, transform=drop_in_person_only),
        run("drop5all", RESAMPLED, "gss0", figures=False, transform=drop_in_person_all_years),
    ]
    summary = pd.concat(columns, axis=1)
    with pd.option_context("display.width", 150, "display.float_format", "{:.3f}".format):
        print(summary)
    summary.to_csv(FIG_DIR / "key_numbers.csv")
    checks()


if __name__ == "__main__":
    main()

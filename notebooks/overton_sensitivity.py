"""Sensitivity of the Overton results to the choice of questions.

Same method as the book; only the question set changes.

1. Drift by question: slope of the log odds of a conservative response
   versus year, for each question, compared with the average slope.
2. Leave one out: rerun without each of the fifteen questions.
3. Random subsets: rerun with random subsets of ten questions.

To compare question sets of different sizes, conservatism is rescaled to
"out of 15": the posterior mean number of conservative responses times
15 / (number of questions).

Writes CSV files to overton_figs/.

Usage: python overton_sensitivity.py
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import logit

from overton_pipeline import (
    FIG_DIR,
    IN_PERSON_ONLY,
    ORDER,
    RESAMPLED,
    count_signs,
    estimate_conservatism,
    load_gss,
    make_questions,
)
from utils import chunk_series, make_lowess, prepare_yvar, run_subgroups

N_SUBSETS = 50
SUBSET_SIZE = 10


def question_drift(gss, questions):
    """Slope of logit(percent conservative) vs year for each question."""
    prevalence = questions.groupby(gss["year"]).mean()
    rows = []
    for varname in ORDER:
        series = prevalence[varname].dropna().clip(0.005, 0.995)
        data = pd.DataFrame(dict(y=logit(series.values), year=series.index))
        results = smf.ols("y ~ year", data=data).fit()
        rows.append(
            dict(
                question=varname,
                first=series.index.min(),
                last=series.index.max(),
                waves=len(series),
                pct_first=series.iloc[0] * 100,
                pct_last=series.iloc[-1] * 100,
                slope_per_decade=results.params["year"] * 10,
                in_person_only=varname in IN_PERSON_ONLY,
            )
        )
    drift = pd.DataFrame(rows).set_index("question")
    drift["relative"] = drift["slope_per_decade"] - drift["slope_per_decade"].mean()
    return drift.sort_values("slope_per_decade")


def add_conservatism(gss, questions, subset):
    """Estimate conservatism from a subset of the questions, out of 15."""
    sub = questions[subset]
    con, _, _ = estimate_conservatism(sub)
    con = con * len(ORDER) / len(subset)
    answered = sub.notna().sum(axis=1)
    gss["conservatism"] = pd.Series(con, gss.index)
    gss.loc[answered < 5, "conservatism"] = np.nan


def summarize(gss):
    """The results the chapter depends on."""
    s = {}
    last = gss["year"].max()
    table = gss.pivot_table(index="year", columns="polviews3", values="conservatism")
    s["Liberal 1974"] = make_lowess(table["Liberal"])[1974]
    s["Moderate 2000"] = make_lowess(table["Moderate"])[2000]
    s[f"Conservative {last}"] = make_lowess(table["Conservative"])[last]
    s["Overton gap"] = s[f"Conservative {last}"] - s["Liberal 1974"]
    gap = make_lowess(table["Conservative"]) - make_lowess(table["Liberal"])
    s["Con-Lib 1974"] = gap[1974]
    s[f"Con-Lib {last}"] = gap[last]

    prepare_yvar(gss, "conservatism", "continuous")
    smooth = make_lowess(chunk_series(gss, "year", size=2000))
    s["start"] = smooth.iloc[0]
    s["end"] = smooth.iloc[-1]
    s["decline"] = s["start"] - s["end"]

    _, neg, pos = count_signs(run_subgroups(gss, "age", "conservatism", "cohort10", "continuous"))
    s["age slopes neg"], s["age slopes pos"] = neg, pos
    _, neg, pos = count_signs(run_subgroups(gss, "year", "conservatism", "cohort10", "continuous"))
    s["year slopes neg"], s["year slopes pos"] = neg, pos
    return s


def run_subset(gss, questions, subset):
    add_conservatism(gss, questions, subset)
    return summarize(gss)


def leave_one_out(gss, questions):
    rows = {"all 15": run_subset(gss, questions, ORDER)}
    for varname in ORDER:
        subset = [q for q in ORDER if q != varname]
        rows[f"without {varname}"] = run_subset(gss, questions, subset)
    return pd.DataFrame(rows).T


def random_subsets(gss, questions, seed=17):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_SUBSETS):
        subset = sorted(rng.choice(ORDER, SUBSET_SIZE, replace=False))
        row = run_subset(gss, questions, subset)
        row["questions"] = " ".join(subset)
        rows.append(row)
    return pd.DataFrame(rows)


def resampling_spread():
    """The same summaries for the three resamplings, all fifteen questions."""
    rows = {}
    for key in ["gss0", "gss1", "gss2"]:
        gss = load_gss(RESAMPLED, key)
        rows[key] = run_subset(gss, make_questions(gss), ORDER)
    return pd.DataFrame(rows).T


def main():
    gss = load_gss(RESAMPLED, "gss0")
    questions = make_questions(gss)

    drift = question_drift(gss, questions)
    drift.to_csv(FIG_DIR / "sensitivity_drift.csv")
    print(drift.round(2))

    loo = leave_one_out(gss, questions)
    loo.to_csv(FIG_DIR / "sensitivity_leave_one_out.csv")
    print(loo.round(2))

    subsets = random_subsets(gss, questions)
    subsets.to_csv(FIG_DIR / "sensitivity_random_subsets.csv", index=False)
    print(subsets.drop(columns="questions").describe().round(2))

    spread = resampling_spread()
    spread.to_csv(FIG_DIR / "sensitivity_resamplings.csv")
    print(spread.round(2))


if __name__ == "__main__":
    main()

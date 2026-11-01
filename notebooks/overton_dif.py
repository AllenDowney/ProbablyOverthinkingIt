"""Checks recommended by the independent review of the confidence questions.

1. Measurement invariance over time (differential item functioning). For a
   target question, estimate each respondent's conservatism from the other
   questions (set v2, without the target; weighted grid, original extract),
   then fit a weighted logistic regression of the target's coded answer on
   that score, decade, and their interaction. If the question measures
   conservatism the same way in every decade, the probability of the coded
   answer at a fixed score shouldn't change by decade. Reported: predicted
   percent at three fixed scores (the 25th, 50th, and 75th percentiles of
   the score in the 1970s), by decade.
   Caveat: the other questions provide the reference scale, so this assumes
   they are invariant themselves.

2. Interview mode in the 2020s. The conservative-liberal gap on each
   question's coded answer in 2022 and 2024, separately for web respondents
   (mode 4) and interviewer-administered ones (in person or by phone).

Targets: the seven confidence questions that pass the stability rule, and
issue questions for comparison.

Writes overton_figs/dif_by_decade.csv and dif_mode.csv.

Usage: python overton_dif.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from overton_candidates import EXTRACT
from overton_pipeline import FIG_DIR, RECODE_POLVIEWS, estimate_conservatism
from overton_question_sets import SET_V2

CONFIDENCE = ["conpress", "conlabor", "conclerg", "conbus", "conarmy", "consci", "contv"]
ISSUES = ["homosex", "abany", "grass", "helppoor", "eqwlth", "cappun", "natenvir", "gunlaw", "fepol"]
MIN_ANSWERED = 5


def load():
    gss = pd.read_hdf(EXTRACT, "gss")
    coded = pd.read_hdf(FIG_DIR / "candidates_coded.hdf", "coded")
    gss["w"] = gss["wtssall"].fillna(gss["wtssps"])
    gss["w"] /= gss["w"].mean()
    gss["decade"] = (gss["year"] // 10 * 10).astype(int)
    gss["polviews3"] = gss["polviews"].replace(RECODE_POLVIEWS)
    return gss, coded


def reference_score(gss, coded, target):
    """Conservatism from the v2 questions other than target, out of the number used."""
    names = [v for v in SET_V2 if v != target]
    questions = coded[names]
    con, _, _ = estimate_conservatism(questions, weights=gss["w"])
    answered = questions.notna().sum(axis=1)
    return pd.Series(con, gss.index).where(answered >= MIN_ANSWERED)


def dif(gss, coded, target):
    df = pd.DataFrame(dict(y=coded[target], theta=reference_score(gss, coded, target),
                           decade=gss["decade"], w=gss["w"])).dropna()
    model = smf.glm("y ~ theta * C(decade)", data=df, family=sm.families.Binomial(), freq_weights=df["w"])
    results = model.fit()
    levels = df.loc[df["decade"] == 1970, "theta"].quantile([0.25, 0.5, 0.75]).values
    rows = []
    for decade in sorted(df["decade"].unique()):
        new = pd.DataFrame(dict(theta=levels, decade=decade))
        pred = results.predict(new) * 100
        rows.append(dict(question=target, decade=decade, p25=pred.iloc[0], p50=pred.iloc[1], p75=pred.iloc[2],
                         n=int((df["decade"] == decade).sum())))
    return pd.DataFrame(rows)


def mode_gaps(gss, coded, target):
    rows = []
    recent = gss[gss["year"].isin([2022, 2024]) & gss["polviews3"].isin(["Liberal", "Conservative"])]
    for mode_name, in_mode in [("web", recent["mode"] == 4), ("interviewer", recent["mode"].isin([1, 2, 3]))]:
        sub = recent[in_mode]
        y = coded.loc[sub.index, target]
        ok = y.notna()
        sub, y = sub[ok], y[ok]
        shares = {}
        for side in ["Conservative", "Liberal"]:
            mask = sub["polviews3"] == side
            # some questions weren't asked in one mode (e.g. fepol only in person since 2022)
            shares[side] = np.average(y[mask], weights=sub.loc[mask, "w"]) * 100 if mask.any() else np.nan
        rows.append(dict(question=target, mode=mode_name, n=len(sub),
                         conservative=shares["Conservative"], liberal=shares["Liberal"],
                         gap=shares["Conservative"] - shares["Liberal"]))
    return rows


def main():
    gss, coded = load()
    targets = CONFIDENCE + ISSUES
    table = pd.concat([dif(gss, coded, t) for t in targets])
    table.to_csv(FIG_DIR / "dif_by_decade.csv", index=False)
    modes = pd.DataFrame([row for t in targets for row in mode_gaps(gss, coded, t)])
    modes.to_csv(FIG_DIR / "dif_mode.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.pivot(index="question", columns="decade", values="p50").reindex(targets).round(0))
        print(modes.pivot(index="question", columns="mode", values="gap").reindex(targets).round(0))


if __name__ == "__main__":
    main()

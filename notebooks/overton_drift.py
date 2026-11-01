"""Item drift check for the Overton update.

For the same respondents, compare conservatism estimated from all fifteen
questions with estimates from ten (the in-person-only five removed) and from
the five alone, holding the difficulties and the out-of-15 scale fixed.
If the difficulties were stable over time, the differences would be small
and steady. Also computes the logit gap between the five and the ten by year.

Writes overton_figs/drift_by_year.csv.

Usage: python overton_drift.py
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from overton_pipeline import (
    FIG_DIR,
    IN_PERSON_ONLY,
    PRIOR,
    RESAMPLED,
    XS,
    estimate_conservatism,
    load_gss,
    make_questions,
)


def estimate_with_difficulties(questions, difficulty):
    """Posterior mean number of conservative responses, given difficulties.

    Uses all of `difficulty` for the scale, so the result is out of
    len(difficulty) even if some questions are blank. Works in log space.
    """
    X, D = np.meshgrid(XS, difficulty)
    P = expit(X - D)
    ns = P.sum(axis=0)
    a = questions.fillna(2).astype(int).values
    loglike = np.zeros((len(questions), len(XS)))
    for j in range(questions.shape[1]):
        col = a[:, j, None]
        loglike += np.where(col == 1, np.log(P[j]), np.where(col == 0, np.log(1 - P[j]), 0))
    posterior = np.exp(loglike - loglike.max(axis=1, keepdims=True)) * PRIOR
    posterior /= posterior.sum(axis=1, keepdims=True)
    return (posterior * ns).sum(axis=1)


def main():
    gss = load_gss(RESAMPLED, "gss0")
    questions = make_questions(gss)
    others = [q for q in questions.columns if q not in IN_PERSON_ONLY]

    con15, _, difficulty = estimate_conservatism(questions)
    without5 = questions.copy()
    without5[IN_PERSON_ONLY] = np.nan
    con10 = estimate_with_difficulties(without5, difficulty)
    only5 = questions.copy()
    only5[others] = np.nan
    con5 = estimate_with_difficulties(only5, difficulty)

    # respondents with enough answers in both sets
    enough = (questions[IN_PERSON_ONLY].notna().sum(axis=1) >= 3) & (
        questions[others].notna().sum(axis=1) >= 5
    )
    df = pd.DataFrame(dict(year=gss["year"], c15=con15, c10=con10, c5=con5))[enough]
    table = df.groupby("year")[["c15", "c10", "c5"]].mean()
    table["ten minus fifteen"] = table["c10"] - table["c15"]
    table["five minus fifteen"] = table["c5"] - table["c15"]
    table["n"] = df.groupby("year").size()

    # logit gap between the five and the ten, using yearly percentages
    pct = questions.groupby(gss["year"]).mean().clip(0.01, 0.99)
    L = logit(pct)
    table["logit gap, five minus ten"] = L[IN_PERSON_ONLY].mean(axis=1) - L[others].mean(axis=1)

    table.to_csv(FIG_DIR / "drift_by_year.csv")
    with pd.option_context("display.width", 150):
        print(table.round(2))


if __name__ == "__main__":
    main()

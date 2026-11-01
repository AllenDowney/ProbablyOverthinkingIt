"""Larger question sets chosen by the same rule as set v2.

The rule (overton_question_sets.py): opinion questions asked in at least 17
of the 33 waves, original wording for the spending items, one question per
hand-assigned topic, ranked by the polviews coefficient. Set v2 is the top
15; here we take the top N for several N, nested, and report all of them,
ending with every candidate that passes the rules.

Scores are rescaled to "out of 15" so sets of different sizes are
comparable. Respondents who answered fewer than 5 questions are dropped,
as in the book.

Sensitivity variant (not a change to the rule): the same procedure with
the confidence-in-institutions items (con*) left out of the pool, because
`conpress` (rank 20) drifts strongly toward its "conservative" response.

Writes overton_figs/larger_sets.csv and larger_sets_questions.csv.

Usage: python overton_larger_sets.py
"""

import numpy as np
import pandas as pd

from overton_data import CANDIDATES_RESAMPLED, KEYS
from overton_pipeline import FIG_DIR, load_gss, make_questions, through_2018
from overton_question_sets import SET_V2
from overton_sensitivity import add_conservatism, summarize
from overton_topics import topic
from utils import make_lowess

MIN_WAVES = 17
SIZES = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
CUTOFFS = {2018: through_2018, 2024: None}
CONFIDENCE = r"^con(press|labor|clerg|educ|bus|army|sci|tv|finan|fed|legis|judge|medic)$"


def ranked_questions(exclude=None, drop=()):
    """Candidates in rank order after the coverage, wording, and topic rules.

    exclude: regex of variables to leave out; drop: names to leave out (e.g.
    the ones that fail the stability rule). Both apply before the topic rule,
    so a question of the same topic can take an excluded one's place.
    """
    ranking = pd.read_csv(FIG_DIR / "candidates_ranking.csv", index_col=0)
    if exclude is not None:
        ranking = ranking[~ranking.index.str.match(exclude)]
    ranking = ranking[~ranking.index.isin(list(drop))]
    ranking = ranking[ranking["waves"] >= MIN_WAVES]
    ranking = ranking[~ranking.index.str.match(r"^nat.*y$")]
    ranking = ranking.assign(topic=[topic(v) for v in ranking.index])
    return ranking.drop_duplicates("topic")


def codes_for(ranked):
    return {v: [float(c) for c in str(ranked.loc[v, "conservative"]).split()] for v in ranked.index}


def run_sizes(ranked, pool):
    codes = codes_for(ranked)
    sizes = [n for n in SIZES if n < len(ranked)] + [len(ranked)]
    rows = []
    for year, transform in CUTOFFS.items():
        for key in KEYS:
            gss = load_gss(CANDIDATES_RESAMPLED, key)
            if transform is not None:
                gss = transform(gss)
            questions = make_questions(gss, codes)
            for n in sizes:
                subset = list(ranked.index[:n])
                add_conservatism(gss, questions, subset)
                s = summarize(gss)
                table = gss.pivot_table(index="year", columns="polviews3", values="conservatism")
                for column in ["Conservative", "Moderate", "Liberal"]:
                    smooth = make_lowess(table[column])
                    s[f"change, {column}"] = smooth.iloc[-1] - smooth.iloc[1]
                s["Overton gap 2018"] = make_lowess(table["Conservative"])[2018] - s["Liberal 1974"]
                s.update(pool=pool, cutoff=year, key=key, n=n, kept=gss["conservatism"].notna().mean())
                rows.append(s)
    return rows


def main():
    np.random.seed(17)
    ranked = ranked_questions()
    assert set(ranked.index[:15]) == set(SET_V2)
    no_conf = ranked_questions(CONFIDENCE)
    pd.concat({"all": ranked, "no confidence": no_conf}).to_csv(FIG_DIR / "larger_sets_questions.csv")

    rows = run_sizes(ranked, "all") + run_sizes(no_conf, "no confidence")
    results = pd.DataFrame(rows)
    results.to_csv(FIG_DIR / "larger_sets.csv", index=False)

    columns = [
        "Liberal 1974", "Moderate 2000", "Overton gap 2018", "Overton gap",
        "change, Conservative", "change, Moderate", "change, Liberal",
        "decline", "age slopes neg", "age slopes pos", "year slopes neg", "year slopes pos",
    ]
    summary = results.groupby(["pool", "cutoff", "n"])[columns].mean()
    with pd.option_context("display.width", 300):
        print(summary.round(2).to_string())


if __name__ == "__main__":
    main()

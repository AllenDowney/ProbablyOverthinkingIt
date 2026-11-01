"""Grid results with difficulties from one m1 fit on the whole candidate pool.

In a 1PL model a question's difficulty doesn't depend on which other
questions are in the set, so one fit of m1 (overton_irt.py, set "pool":
66 questions) gives difficulties for every set: v1, v2, and the nested
top-N sets from overton_larger_sets.py, with and without the confidence
items. For each set, the grid is run twice on the same respondents:
- heuristic: difficulties = -logit(prevalence), as in the book;
- m1 pool: posterior mean difficulties from the pool fit.
Only the difficulties differ. Scores are rescaled to out of 15;
respondents who answered fewer than 5 of the set's questions are dropped.

Also compares the v1 and v2 difficulties from the pool fit with those
from the fits on v1 and v2 alone (overton_irt_compare.py), because m1's
scale depends on the spread of respondents' estimates, which depends on
how many questions they answered.

Writes overton_figs/pool_difficulties.csv (difficulties) and
pool_difficulty_results.csv (results).

Usage: python overton_pool_difficulties.py
"""

import numpy as np
import pandas as pd

from overton_data import CANDIDATES_RESAMPLED, KEYS
from overton_drift import estimate_with_difficulties
from overton_irt import RESULTS_DIR, question_sets
from overton_irt_compare import item_means, results_for
from overton_larger_sets import CONFIDENCE, SIZES, ranked_questions
from overton_pipeline import FIG_DIR, load_gss, make_questions

MIN_ANSWERED = 5


def subsets():
    """Map from label to list of questions."""
    sets = question_sets()
    out = {"v1": list(sets["v1"]), "v2": list(sets["v2"])}
    for pool, exclude in [("all", None), ("no confidence", CONFIDENCE)]:
        ranked = list(ranked_questions(exclude).index)
        sizes = [n for n in SIZES if n < len(ranked)] + [len(ranked)]
        for n in sizes:
            out[f"{pool} {n}"] = ranked[:n]
    return out


def main():
    np.random.seed(17)
    codes = question_sets()["pool"]
    labels = subsets()
    rows, tables = [], []
    for key in KEYS:
        gss = load_gss(CANDIDATES_RESAMPLED, key)
        questions = make_questions(gss, codes)
        names = list(questions.columns)
        prevalence = questions.mean()
        sources = {
            "heuristic": pd.Series(-np.log(prevalence / (1 - prevalence)).values, names),
            "m1 pool": pd.Series(item_means(pd.read_csv(RESULTS_DIR / f"pool_{key}_m1_items.csv", index_col=0), names), names),
        }
        table = pd.DataFrame(sources)
        for name in ["v1", "v2"]:
            own = pd.read_csv(RESULTS_DIR / f"{name}_{key}_m1_items.csv", index_col=0)
            table[f"m1 {name}"] = [own.loc[f"d[{v}]", "mean"] if f"d[{v}]" in own.index else np.nan for v in names]
        table["key"] = key
        tables.append(table)

        for label, subset in labels.items():
            sub = questions[subset]
            few = sub.notna().sum(axis=1) < MIN_ANSWERED
            for source, d in sources.items():
                con = estimate_with_difficulties(sub, d[subset].values) * 15 / len(subset)
                s = results_for(gss, pd.Series(con, gss.index).mask(few))
                s.update(key=key, set=label, n=len(subset), source=source)
                rows.append(s)
            print(key, label, round(rows[-2]["Overton gap"], 2), round(rows[-1]["Overton gap"], 2))

    difficulties = pd.concat(tables)
    difficulties.to_csv(FIG_DIR / "pool_difficulties.csv")
    results = pd.DataFrame(rows)
    results.to_csv(FIG_DIR / "pool_difficulty_results.csv", index=False)

    # scale of the pool difficulties relative to the heuristic and the per-set fits
    d0 = difficulties[difficulties["key"] == "gss0"]
    for name in ["v1", "v2"]:
        sub = d0.dropna(subset=[f"m1 {name}"])
        print(name, "slope pool vs heuristic", round(np.polyfit(sub["heuristic"], sub["m1 pool"], 1)[0], 2),
              "| per-set vs heuristic", round(np.polyfit(sub["heuristic"], sub[f"m1 {name}"], 1)[0], 2))
    print("all 66, slope pool vs heuristic", round(np.polyfit(d0["heuristic"], d0["m1 pool"], 1)[0], 2))

    columns = ["Liberal 1974", "Moderate 2000", "Overton gap", "Con-Lib 1974", "change, Conservative",
               "change, Moderate", "change, Liberal", "decline", "year slopes neg", "year slopes pos"]
    summary = results.groupby(["set", "source"], sort=False)[columns].mean().unstack("source")
    with pd.option_context("display.width", 300, "display.max_rows", 100):
        print(summary[["Overton gap", "change, Conservative", "year slopes pos"]].round(2).to_string())


if __name__ == "__main__":
    main()

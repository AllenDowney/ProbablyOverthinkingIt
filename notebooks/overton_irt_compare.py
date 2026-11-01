"""Compare the book's grid method with Ricardo Vieira's IRT models (Phase 4).

One change at a time, on the same respondents (weighted resamplings of the
candidate extract) and question sets (v1, v2):

- grid, heuristic: the book's method, difficulties = -logit(prevalence).
- grid, m1 difficulties: the same grid, with difficulties from m1
  (posterior means). The only change is the difficulties.
- m1: posterior mean count from m1 (also carries the uncertainty in the
  difficulties; otherwise the same model as the grid).
- m2: posterior mean count from the hierarchical model (adds pooling).
- m2, all respondents: m2 without the book's rule of dropping respondents
  who answered fewer than 5 questions.

Reads the fits written by overton_irt.py (run on Unity). Writes
overton_figs/irt_compare.csv and irt_difficulties.csv.

Usage: python overton_irt_compare.py
"""

import json

import numpy as np
import pandas as pd

from overton_data import CANDIDATES_RESAMPLED, KEYS
from overton_drift import estimate_with_difficulties
from overton_irt import RESULTS_DIR
from overton_pipeline import FIG_DIR, load_gss, make_questions
from overton_question_sets import SETS
from overton_sensitivity import summarize
from utils import make_lowess

MIN_ANSWERED = 5


def load_fit(name, key, model):
    stem = RESULTS_DIR / f"{name}_{key}_{model}"
    items = pd.read_csv(f"{stem}_items.csv", index_col=0)
    respondents = pd.read_parquet(f"{stem}_respondents.parquet").set_index("index")
    diagnostics = json.load(open(f"{stem}_diagnostics.json"))
    return items, respondents, diagnostics


def item_means(items, names):
    """Posterior mean difficulties in the order of `names`."""
    return np.array([items.loc[f"d[{v}]", "mean"] for v in names])


def results_for(gss, conservatism):
    gss["conservatism"] = conservatism
    s = summarize(gss)
    table = gss.pivot_table(index="year", columns="polviews3", values="conservatism")
    for column in ["Conservative", "Moderate", "Liberal"]:
        smooth = make_lowess(table[column])
        s[f"change, {column}"] = smooth.iloc[-1] - smooth.iloc[1]
    s["kept"] = gss["conservatism"].notna().mean()
    s["mean"] = gss["conservatism"].mean()
    return s


def main():
    np.random.seed(17)
    rows, difficulties, diagnostics = [], [], []
    for name, codes in SETS.items():
        for key in KEYS:
            gss = load_gss(CANDIDATES_RESAMPLED, key)
            questions = make_questions(gss, codes)
            names = list(questions.columns)
            answered = questions.notna().sum(axis=1)
            few = answered < MIN_ANSWERED
            heuristic = -np.log(questions.mean() / (1 - questions.mean())).values

            fits = {model: load_fit(name, key, model) for model in ["m1", "m2"]}
            d1 = item_means(fits["m1"][0], names)

            variants = {}
            variants["grid, heuristic"] = pd.Series(estimate_with_difficulties(questions, heuristic), gss.index).mask(few)
            variants["grid, m1 difficulties"] = pd.Series(estimate_with_difficulties(questions, d1), gss.index).mask(few)
            for model, (items, respondents, diag) in fits.items():
                count = respondents["count_mean"].reindex(gss.index)
                variants[model] = count.mask(few)
                diagnostics.append(diag)
            variants["m2, all respondents"] = fits["m2"][1]["count_mean"].reindex(gss.index)

            for variant, conservatism in variants.items():
                s = results_for(gss, conservatism)
                s.update(set=name, key=key, variant=variant)
                rows.append(s)
                print(name, key, variant, round(s["Overton gap"], 2))

            table = pd.DataFrame({"heuristic": heuristic, "m1": d1, "m2": item_means(fits["m2"][0], names)}, index=names)
            table["set"], table["key"] = name, key
            difficulties.append(table)

    results = pd.DataFrame(rows)
    results.to_csv(FIG_DIR / "irt_compare.csv", index=False)
    pd.concat(difficulties).to_csv(FIG_DIR / "irt_difficulties.csv")
    pd.DataFrame(diagnostics).to_csv(FIG_DIR / "irt_diagnostics.csv", index=False)

    columns = [
        "kept", "mean", "Liberal 1974", "Moderate 2000", "Overton gap", "Con-Lib 1974",
        "change, Conservative", "change, Moderate", "change, Liberal", "decline",
        "age slopes neg", "age slopes pos", "year slopes neg", "year slopes pos",
    ]
    order = list(dict.fromkeys(results["variant"]))
    summary = results.groupby(["set", "variant"], sort=False)[columns].mean()
    with pd.option_context("display.width", 300):
        print(summary.round(2).to_string())
        print(pd.DataFrame(diagnostics).to_string())


if __name__ == "__main__":
    main()

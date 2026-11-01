"""Figures for the Overton lab notebook (overton_lab.md).

Reads the CSV files written by overton_drift.py, overton_sensitivity.py,
and overton_larger_sets.py, and the m1 fit on the whole pool (overton_irt.py),
and saves figures in overton_figs/lab/.

Usage: python overton_lab_figs.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from overton_pipeline import FIG_DIR
from utils import decorate, make_lowess, set_pyplot_params

LAB_DIR = FIG_DIR / "lab"


def savefig(name):
    plt.tight_layout()
    plt.savefig(LAB_DIR / name, dpi=150)
    plt.close()


def plot_drift_by_question():
    drift = pd.read_csv(FIG_DIR / "sensitivity_drift.csv", index_col=0)
    drift = drift.sort_values("slope_per_decade", ascending=False)
    drift["y"] = np.arange(len(drift))
    plt.figure(figsize=(6, 4.5))
    plt.axvline(0, color="gray", lw=1, alpha=0.5)
    for flag, color, label in [(False, "C0", "Asked in all modes"), (True, "C1", "In person or phone only since 2022")]:
        sub = drift[drift["in_person_only"] == flag]
        plt.plot(sub["slope_per_decade"], sub["y"], "o", color=color, label=label)
    plt.yticks(drift["y"], drift.index)
    decorate(
        xlabel="Change in log odds of a conservative response per decade",
        title="Drift by question, 1972-2024",
    )
    plt.legend(loc="lower left")
    savefig("drift_by_question.png")


def plot_drift_by_year():
    table = pd.read_csv(FIG_DIR / "drift_by_year.csv", index_col=0)
    series = table["ten minus fifteen"]
    plt.figure(figsize=(6, 3.5))
    plt.axhline(0, color="gray", lw=1, alpha=0.5)
    plt.plot(series.index, series.values, "o", color="C0", alpha=0.4)
    make_lowess(series, frac=0.5).plot(color="C0")
    decorate(
        xlabel="Year of survey",
        ylabel="Difference (out of 15)",
        title="Estimate from ten questions minus estimate from fifteen",
    )
    savefig("drift_by_year.png")


def plot_overton_gap():
    loo = pd.read_csv(FIG_DIR / "sensitivity_leave_one_out.csv", index_col=0)
    subsets = pd.read_csv(FIG_DIR / "sensitivity_random_subsets.csv")
    spread = pd.read_csv(FIG_DIR / "sensitivity_resamplings.csv", index_col=0)

    rows = [
        ("Three resamplings,\nall 15 questions", spread["Overton gap"]),
        ("Leave one\nquestion out", loo["Overton gap"].drop("all 15")),
        ("Random sets\nof 10 questions", subsets["Overton gap"]),
    ]
    rng = np.random.default_rng(17)
    plt.figure(figsize=(6, 3.5))
    plt.axvline(0, color="gray", lw=1, alpha=0.5)
    plt.axvline(loo.loc["all 15", "Overton gap"], color="C0", lw=1, ls=":")
    for i, (label, values) in enumerate(rows):
        jitter = rng.uniform(-0.15, 0.15, len(values))
        plt.plot(values, i + jitter, "o", color="C0", alpha=0.5)
    plt.yticks(range(len(rows)), [label for label, _ in rows])

    # label the extremes of the leave-one-out row
    gaps = loo["Overton gap"].drop("all 15")
    for name in [gaps.idxmin(), gaps.idxmax()]:
        plt.text(gaps[name], 1.25, name, ha="center", fontsize="small")

    decorate(
        xlabel="Conservative 2024 minus Liberal 1974 (out of 15)",
        title="How far is the 2024 conservative from the 1974 liberal?",
    )
    savefig("overton_gap.png")


def plot_gap_by_size():
    results = pd.read_csv(FIG_DIR / "larger_sets.csv")
    results = results[results["cutoff"] == 2024]
    plt.figure(figsize=(6, 3.5))
    plt.axhline(0, color="gray", lw=1, alpha=0.5)
    for pool, color, label in [("all", "C0", "All candidates"), ("no confidence", "C1", "Confidence items left out")]:
        sub = results[results["pool"] == pool]
        plt.plot(sub["n"], sub["Overton gap"], ".", color=color, alpha=0.5)
        plt.plot(sub.groupby("n")["Overton gap"].mean(), "-", color=color, label=label)
    decorate(
        xlabel="Number of questions (top N by the selection rule)",
        ylabel="Conservative 2024 - Liberal 1974",
        ylim=[0, 3],
    )
    savefig("gap_by_size.png")


def plot_pool_difficulties(key="gss0"):
    """Difficulties from the m1 fit on the pool vs the heuristic, -logit(prevalence)."""
    from overton_data import CANDIDATES_RESAMPLED
    from overton_irt import RESULTS_DIR, question_sets
    from overton_irt_compare import item_means
    from overton_larger_sets import CONFIDENCE
    from overton_pipeline import load_gss, make_questions

    questions = make_questions(load_gss(CANDIDATES_RESAMPLED, key), question_sets()["pool"])
    names = list(questions.columns)
    prevalence = questions.mean()
    heuristic = -np.log(prevalence / (1 - prevalence)).values
    fitted = item_means(pd.read_csv(RESULTS_DIR / f"pool_{key}_m1_items.csv", index_col=0), names)
    confidence = pd.Index(names).str.match(CONFIDENCE)

    slope, intercept = np.polyfit(heuristic, fitted, 1)
    xs = np.array([heuristic.min(), heuristic.max()])
    plt.figure(figsize=(5, 4.5))
    plt.plot(xs, xs, ":", color="gray", label="Identity")
    plt.plot(xs, intercept + slope * xs, "-", color="0.6", label=f"Fitted line, slope {slope:.2f}")
    plt.plot(heuristic[~confidence], fitted[~confidence], "o", color="C0", alpha=0.7, label="Other questions")
    plt.plot(heuristic[confidence], fitted[confidence], "s", color="C1", alpha=0.7, label="Confidence questions")
    decorate(xlabel="Heuristic difficulty, -logit(prevalence)", ylabel="m1 difficulty (posterior mean)")
    savefig("pool_difficulties.png")
    print("slope", round(slope, 2), "intercept", round(intercept, 2),
          "corr", round(np.corrcoef(heuristic, fitted)[0, 1], 3))


def main():
    set_pyplot_params()
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    plot_drift_by_question()
    plot_drift_by_year()
    plot_overton_gap()
    plot_gap_by_size()
    plot_pool_difficulties()


if __name__ == "__main__":
    main()

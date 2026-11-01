"""Composite figures for Phase 4b, in the CultureWar book style.

Uses CultureWar's figure helpers (header, registration marks, direct line
labels, export), imported from ~/CultureWar/notebooks, so the figures match
that book as its style evolves. Run in the CultureWar env. Reads only the
fit results in overton_apc/results (no imports from this repo's utils,
whose name collides with CultureWar's).

Figures, for each question set, in overton_figs/apc/:
- time_series_{set}: period model by polviews3, with the 1974 liberal
  level marked (dotted, unlabeled).
- time_series7_{set}: period model by the 7-point polviews.
- cohort_component_{set} and period_component_{set}: period-cohort model,
  all respondents and by polviews3.
- age_series_{set}: age model, one line per decade of surveys.
- age_by_decade_{set} and age_by_generation_{set}: age model fit separately
  for each decade of birth (1900s-2000s) and each generation (Pew's
  boundaries), plotted against age.
- year_by_decade_{set} and year_by_generation_{set}: the period model fit
  separately for the same groups, plotted against survey year, with the
  overall average dotted.
- pc_by_decade_{set} and pc_by_generation_{set}: the period component of a
  period-cohort model fit within each group (standardized to the group's
  birth years), with the overall average dotted.
- within_cohort_slope_by_n and overton_gap_by_n: the within-cohort slope
  and the Overton gap against the number of questions, for the primary
  specification (no confidence questions) and the sensitivity version.

Usage: python overton_apc_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path.home() / "CultureWar/notebooks"))
import utils as cw  # noqa: E402  CultureWar's utils
from composed_standardized_effects.effects import _add_direct_line_labels, _add_inline_line_labels  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "overton_apc" / "results"
OUT_DIR = HERE / "overton_figs" / "apc"

SOURCE = "GSS"
Y_AXIS = "Conservative responses, out of 15"
YLIM = (0, 15)  # the whole scale, the same for every figure; legends go above the lines
# the colors of Chapter 12 (seaborn "muted" 0, 4, 3)
GROUP_COLORS = {"All": "0.6", "Liberal": "#4878d0", "Moderate": "#956cb4", "Conservative": "#d65f5f"}
POLVIEWS7 = ["Extremely liberal", "Liberal", "Slightly liberal", "Moderate",
             "Slightly conservative", "Conservative", "Extremely conservative"]
# label placements on the time series, (x, offset in points); clear of the lines for v1 and v2
TIME_SERIES_LABELS = {"Conservative": (2000, 6), "Moderate": (2020, -4), "Liberal": (2000, -6)}
DECADE_LABELS = {"1972-1979": "1970s", "1980-1989": "1980s", "1990-1999": "1990s",
                 "2000-2009": "2000s", "2010-2019": "2010s", "2020-2024": "2020s"}


def read_series(run):
    return pd.read_csv(RESULTS_DIR / f"{run}_series.csv")


def read_components(run):
    return pd.read_csv(RESULTS_DIR / f"{run}_components.csv")


def band(ax, x, df, color, label, z):
    ax.fill_between(x, df["lo90"], df["hi90"], color=color, alpha=0.3, lw=0, zorder=z)
    ax.plot(x, df["mean"], color=color, label=label, lw=cw.FIGURE_LINEWIDTH * 1.5, zorder=z + 1)


def new_figure():
    cw.apply_figure_style()
    return plt.subplots(figsize=cw.FIGURE_FIGSIZE)


def finish(fig, ax, name, subject, figure_type, xlabel, labels="right"):
    ax.set_xlabel(xlabel)
    cw.apply_book_grid(ax)
    ax.set_ylim(*YLIM)
    ax.set_yticks(range(0, 16, 3))
    cw.add_figure_header(subject, figure_type, Y_AXIS, SOURCE, left_align=True)
    # direct labels for up to four lines (a side, or placements along the lines);
    # a legend in rows for more ("legend:<corner>:<columns>")
    if isinstance(labels, tuple):  # ("sides", {label: "left" or "right"})
        _add_direct_line_labels(ax, sides=labels[1])
    elif isinstance(labels, dict):
        _add_inline_line_labels(ax, labels)
    elif labels.startswith("legend:"):
        _, loc, ncol = labels.split(":")
        ax.legend(loc=loc, ncol=int(ncol), frameon=False, fontsize="small", handlelength=1.5, columnspacing=1.2)
    elif labels:
        _add_direct_line_labels(ax, side=labels)
    cw.save_figure(fig, OUT_DIR / name)


def group_runs(prefix, name):
    runs = {"All": f"{prefix}_{name}"}
    runs.update({g: f"{prefix}_{name}_polviews3-{g}" for g in ["Liberal", "Moderate", "Conservative"]})
    return runs


def time_series(name):
    fig, ax = new_figure()
    # the three groups only, as in Chapter 12 (all respondents is close to the moderates)
    runs = group_runs("period", name)
    del runs["All"]
    for z, (group, run) in enumerate(runs.items()):
        df = read_series(run)
        band(ax, df["year"], df, GROUP_COLORS[group], group, 3 + 2 * z)
    # the level of the 1974 liberal, over the range of the data
    lib = read_series(f"period_{name}_polviews3-Liberal").set_index("year")["mean"]
    # 1974, or the first year with data (the core economic questions start in 1975)
    ref_year = 1974 if 1974 in lib.index else lib.index.min()
    ax.hlines(lib.loc[ref_year], lib.index.min(), lib.index.max(), color="0.5", ls=":", lw=1, zorder=2)
    finish(fig, ax, f"time_series_{name}", "Conservatism by political alignment", "time series", "Survey year",
           labels=TIME_SERIES_LABELS)


def time_series7(name):
    fig, ax = new_figure()
    cmap = plt.get_cmap("coolwarm")
    for i, label in enumerate(POLVIEWS7):
        df = read_series(f"period_{name}_polviews-{i + 1}")
        band(ax, df["year"], df, cmap(i / 6), label, 3 + 2 * i)
    finish(fig, ax, f"time_series7_{name}", "Conservatism by political label", "time series", "Survey year",
           labels="legend:upper left:4")


def components(name):
    for component, xlabel, title in [("cohort", "Birth cohort", "Conservatism by birth cohort"),
                                     ("period", "Survey year", "Conservatism within cohorts")]:
        fig, ax = new_figure()
        for z, (group, run) in enumerate(group_runs("period_cohort", name).items()):
            df = read_components(run)
            df = df[df["component"] == component]
            if component == "cohort":
                df = df[df["index"] >= 1900]
            band(ax, df["index"], df, GROUP_COLORS[group], group, 3 + 2 * z)
        # the lines are too close together for direct labels that work in both sets
        labels = "legend:lower left:2"
        finish(fig, ax, f"{component}_component_{name}", title, f"{component} component", xlabel, labels=labels)


def age_series(name):
    fig, ax = new_figure()
    cmap = plt.get_cmap("viridis")
    windows = list(DECADE_LABELS)
    for i, window in enumerate(windows):
        path = RESULTS_DIR / f"age_{name}_window-{window}_series.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        band(ax, df["age"], df, cmap(i / (len(windows) - 1) * 0.9), DECADE_LABELS[window], 3 + 2 * i)
    finish(fig, ax, f"age_series_{name}", "Conservatism by age", "age series", "Age", labels="legend:upper left:4")


DECADES = [(f"{d}s", f"{d}-{d + 9}") for d in range(1900, 2001, 10)]
GENERATIONS = [("Greatest", "1883-1927"), ("Silent", "1928-1945"), ("Boomers", "1946-1964"),
               ("Gen X", "1965-1980"), ("Millennials", "1981-1996"), ("Gen Z", "1997-2012")]


def by_birth_group(name, groups, index, ends, fname, subject, offsets=None, overall=False):
    """One fit per birth group, plotted against age or survey year.

    index: "age" (age model), "year" (period model), or "pc" (the period
    component of a period-cohort model fit within the group, standardized
    to the group's mix of birth years, so the line doesn't move because
    members enter or leave the group). Within a group, age
    and survey year move together, but a group spans several birth years,
    so the two are separate fits, not shifted copies.
    ends: {label: "start" or "end"}, which end of each line gets its label.
    Line ends are staggered across groups (by age, the early groups start
    late and the late groups end early; by year, the reverse), so labels
    go at the staggered ends. offsets: {label: points} to nudge a label.
    overall: also plot the overall average (the period model for all
    respondents) as a dotted line; only for index="year".
    Colors run from yellow (earliest births) to purple (latest).
    """
    fig, ax = new_figure()
    cmap = plt.get_cmap("viridis")
    offsets = offsets or {}
    prefix = {"age": "age", "year": "period", "pc": "period_cohort"}[index]
    if overall:
        df = read_series(f"period_{name}")
        ax.plot(df["year"], df["mean"], color="0.4", ls=":", lw=cw.FIGURE_LINEWIDTH * 1.5, zorder=2)
    placements = {}
    for i, (label, cohorts) in enumerate(groups):
        run = f"{prefix}_{name}_cohorts-{cohorts}"
        if index == "pc":
            df = read_components(run)
            df = df[df["component"] == "period"].rename(columns={"index": "year"})
        else:
            df = read_series(run)
        x = df["age"] if index == "age" else df["year"]
        color = cmap(0.9 * (1 - i / (len(groups) - 1)))
        band(ax, x, df, color, label, 3 + 2 * i)
        placements[label] = (0, offsets.get(label, 0), ends[label])
    if index == "age":
        ax.set_xlim(12, 92)
        figure_type, xlabel = "age series", "Age"
    else:
        ax.set_xlim(1965, 2031)
        figure_type = "time series" if index == "year" else "period component"
        xlabel = "Survey year"
    finish(fig, ax, fname, subject, figure_type, xlabel, labels=placements)


def split_ends(groups, n_early, early, late):
    """Label the first n_early groups at one end of their lines, the rest at the other."""
    return {label: (early if i < n_early else late) for i, (label, _) in enumerate(groups)}


def by_n_table():
    """Within-cohort slope and Overton gap for each set size, both specifications.

    Within-cohort slope: straight-line slope of the standardized period
    component (national period-cohort model). Overton gap: the period
    model's conservative line in the last year minus its liberal line in
    1974, with an interval from the two fits treated as independent.
    """
    rows = []
    for spec, prefix in [("primary", "p"), ("sensitivity", "s")]:
        for path in sorted(RESULTS_DIR.glob(f"period_cohort_{prefix}*_summary.json")) + [
                RESULTS_DIR / "period_cohort_v2_summary.json"]:
            summary = json.load(open(path))
            name = summary["set"]
            if not (RESULTS_DIR / f"period_{name}_polviews3-Liberal_series.csv").exists():
                continue
            n = 15 if name == "v2" else int(name[1:])
            lib = read_series(f"period_{name}_polviews3-Liberal").set_index("year")
            con = read_series(f"period_{name}_polviews3-Conservative").set_index("year")
            last = con.index.max()
            gap = con.loc[last, "mean"] - lib.loc[1974, "mean"]
            # 90% half-widths to standard deviations, combined as independent
            sd = ((con.loc[last, "hi90"] - con.loc[last, "lo90"]) ** 2 +
                  (lib.loc[1974, "hi90"] - lib.loc[1974, "lo90"]) ** 2) ** 0.5 / (2 * 1.645)
            rows.append(dict(spec=spec, n=n, set=name,
                             slope=summary["period_slope_per_decade"],
                             slope_lo90=summary["period_slope_lo90"], slope_hi90=summary["period_slope_hi90"],
                             gap=gap, gap_lo90=gap - 1.645 * sd, gap_hi90=gap + 1.645 * sd))
    return pd.DataFrame(rows).sort_values(["spec", "n"])


def plot_by_n(table, column, fname, subject, y_axis, ylim):
    fig, ax = new_figure()
    ax.axhline(0, color="0.6", lw=1, zorder=1)
    for spec, color, label in [("primary", "C0", "Without confidence questions"),
                               ("sensitivity", "C1", "With confidence questions")]:
        df = table[table["spec"] == spec]
        ax.fill_between(df["n"], df[f"{column}_lo90"], df[f"{column}_hi90"], color=color, alpha=0.3, lw=0)
        ax.plot(df["n"], df[column], "o-", color=color, label=label, lw=cw.FIGURE_LINEWIDTH * 1.5, ms=4)
    ax.set_xlabel("Number of questions (top N by the selection rule)")
    cw.apply_book_grid(ax)
    ax.set_ylim(*ylim)
    cw.add_figure_header(subject, "by question set", y_axis, SOURCE, left_align=True)
    ax.legend(loc="lower left", frameon=False, fontsize="small")
    cw.save_figure(fig, OUT_DIR / fname)


def by_n_figures():
    table = by_n_table()
    table.to_csv(HERE / "overton_figs" / "apc_by_n.csv", index=False)
    print(table.round(2).to_string(index=False))
    plot_by_n(table, "slope", "within_cohort_slope_by_n", "Change within cohorts",
              "Conservative responses per decade, out of 15", (-0.6, 0.2))
    plot_by_n(table, "gap", "overton_gap_by_n", "Conservatives today vs liberals in 1974",
              "Difference in conservative responses, out of 15", (-1, 4))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # the figures by set are for v1 and v2; the larger sets go in the by-N figures
    sets = sorted({json.load(open(p))["set"] for p in RESULTS_DIR.glob("*_summary.json")} & {"v1", "v2"})
    for name in sets:
        time_series(name)
        time_series7(name)
        components(name)
        age_series(name)
        if (RESULTS_DIR / f"age_{name}_cohorts-1900-1909_series.csv").exists():
            # by age: early groups labeled where they start, late groups where they end
            ends = split_ends(DECADES, 4, "start", "end")
            by_birth_group(name, DECADES, "age", ends, f"age_by_decade_{name}",
                           "Conservatism by age and decade of birth", offsets={"1960s": -8})
            ends = split_ends(GENERATIONS, 2, "start", "end")
            by_birth_group(name, GENERATIONS, "age", ends, f"age_by_generation_{name}",
                           "Conservatism by age and generation")
        if (RESULTS_DIR / f"period_{name}_cohorts-1900-1909_series.csv").exists():
            # by year: early groups labeled where they end, late groups where they start
            ends = split_ends(DECADES, 6, "end", "start")
            by_birth_group(name, DECADES, "year", ends, f"year_by_decade_{name}",
                           "Conservatism by decade of birth", overall=True)
            ends = split_ends(GENERATIONS, 3, "end", "start")
            by_birth_group(name, GENERATIONS, "year", ends, f"year_by_generation_{name}",
                           "Conservatism by generation", overall=True)
        if (RESULTS_DIR / f"period_cohort_{name}_cohorts-1900-1909_components.csv").exists():
            ends = split_ends(DECADES, 6, "end", "start")
            by_birth_group(name, DECADES, "pc", ends, f"pc_by_decade_{name}",
                           "Conservatism within cohorts, by decade of birth", overall=True)
            ends = split_ends(GENERATIONS, 3, "end", "start")
            by_birth_group(name, GENERATIONS, "pc", ends, f"pc_by_generation_{name}",
                           "Conservatism within cohorts, by generation", overall=True)
        print("figures for", name)
    if any(RESULTS_DIR.glob("period_cohort_p*_summary.json")):
        by_n_figures()


if __name__ == "__main__":
    main()

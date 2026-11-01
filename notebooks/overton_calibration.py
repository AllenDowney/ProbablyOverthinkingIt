"""Calibration check recommended by the review of the difficulties
(overton_difficulties_design.md, 2026-11-01).

For each question set, compares the scores under each choice of difficulties
(heuristic; EM calibrated on the primary pool, the production choice; EM on
all 66 candidates; EM per set; m1 on all 66), using the data written by
`overton_apc.py prepare --difficulties ...`. Every version has the same
respondents, so the differences are what the difficulties do.

Reported, from weighted yearly means (not model-based): the Overton gap
(conservatives in 2024 minus liberals in the first year, 1974 or 1975 for
econA), and the change in conservatives and liberals over the same span.

Also the model-based version, from the period and period-cohort fits
(overton_apc.py, Unity array 66468744 for the EM versions): within-cohort
slope (straight-line slope of the standardized period component), its
decomposition, and the Overton gap and change in conservatives from the
period models by polviews3.

Writes overton_figs/calibration_check.csv and calibration_models.csv.

Usage: python overton_calibration.py
"""

import json

import numpy as np
import pandas as pd

from overton_apc import DATA_DIR, RESULTS_DIR, SCALE, data_name
from overton_pipeline import FIG_DIR

SETS = ["v1", "p15", "p30", "p43", "s50", "econA", "econB", "econC", "social"]
CHOICES = ["heuristic", "em", "em66", "emset", "m1pool"]


def yearly(df):
    df = df.assign(c=df["y"] * SCALE)
    table = df.groupby(["year", "polviews3"]).apply(lambda g: np.average(g["c"], weights=g["weight"]))
    return table.unstack()


def model_table():
    """Model-based results for each set and choice of difficulties that has been fit."""
    rows = []
    for name in SETS:
        for choice in CHOICES:
            set_name = data_name(name, choice)
            path = RESULTS_DIR / f"period_cohort_{set_name}_summary.json"
            if not path.exists():
                continue
            pc = json.load(open(path))
            row = dict(set=name, difficulties=choice,
                       slope=pc["period_slope_per_decade"], slope_lo90=pc["period_slope_lo90"],
                       slope_hi90=pc["period_slope_hi90"], total_change=pc["total_change"],
                       generational_replacement=pc["generational_replacement"],
                       changing_minds=pc["changing_minds"])
            series = {}
            for side in ["Liberal", "Conservative"]:
                run = RESULTS_DIR / f"period_{set_name}_polviews3-{side}_series.csv"
                if run.exists():
                    series[side] = pd.read_csv(run).set_index("year")["mean"]
            if len(series) == 2:
                lib, con = series["Liberal"], series["Conservative"]
                first = 1974 if 1974 in lib.index else lib.index.min()
                row.update(overton_gap=con.iloc[-1] - lib.loc[first],
                           change_conservative=con.iloc[-1] - con.loc[first],
                           change_liberal=lib.iloc[-1] - lib.loc[first])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    rows = []
    for name in SETS:
        for choice in CHOICES:
            path = DATA_DIR / f"{data_name(name, choice)}.parquet"
            if not path.exists():
                continue
            table = yearly(pd.read_parquet(path))
            first = 1974 if 1974 in table.index else table.index.min()
            last = table.index.max()
            rows.append(dict(set=name, difficulties=choice,
                             overton_gap=table.loc[last, "Conservative"] - table.loc[first, "Liberal"],
                             change_conservative=table.loc[last, "Conservative"] - table.loc[first, "Conservative"],
                             change_liberal=table.loc[last, "Liberal"] - table.loc[first, "Liberal"]))
    results = pd.DataFrame(rows)
    results.to_csv(FIG_DIR / "calibration_check.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(results.pivot(index="set", columns="difficulties", values="overton_gap")
              .reindex(index=SETS, columns=CHOICES).round(2))
        print(results.pivot(index="set", columns="difficulties", values="change_conservative")
              .reindex(index=SETS, columns=CHOICES).round(2))

    models = model_table()
    models.to_csv(FIG_DIR / "calibration_models.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(models.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

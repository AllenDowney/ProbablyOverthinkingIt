"""Summary table of the period, period-cohort, and age fits (Phase 4b).

Reads overton_apc/results (fits from overton_apc.py, run on Unity) and
writes overton_figs/apc_summary.csv. The figures are made by
overton_apc_figures.py (CultureWar style, CultureWar env).

Usage: python overton_apc_report.py
"""

import json

import pandas as pd

from overton_apc import RESULTS_DIR
from overton_pipeline import FIG_DIR


def load_summaries():
    rows = [json.load(open(path)) for path in sorted(RESULTS_DIR.glob("*_summary.json"))]
    return pd.DataFrame(rows)


def main():
    summaries = load_summaries()
    summaries.to_csv(FIG_DIR / "apc_summary.csv", index=False)

    columns = ["run", "slope_per_decade", "slope_lo90", "slope_hi90", "change_first_last",
               "total_change", "generational_replacement", "changing_minds",
               "period_change_per_decade", "period_change_per_decade_lo90", "period_change_per_decade_hi90",
               "divergences", "rhat_max", "ess_bulk_min", "elapsed_seconds"]
    with pd.option_context("display.width", 300, "display.max_rows", 100):
        print(summaries.reindex(columns=columns).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Run the Overton pipeline with question sets v1 and v2 on the same data.

Data: three weighted resamplings of the candidate extract (full GSS file,
R3a), so both sets use the same respondents. Same method as the book;
only the questions change.

Writes overton_figs/compare_sets_key_numbers.csv, compare_sets_yearly.csv,
and the chapter's figures for v2 in overton_figs/v2/.

Usage: python overton_compare_sets.py
"""

import numpy as np
import pandas as pd

from overton_data import CANDIDATES_RESAMPLED, KEYS, make_candidates_resampled
from overton_pipeline import FIG_DIR, load_gss, run, run_pipeline, set_pyplot_params, yearly_table
from overton_question_sets import SETS


def yearly(codes):
    """Mean conservatism by year and alignment, averaged over the resamplings."""
    tables = []
    for key in KEYS:
        gss = load_gss(CANDIDATES_RESAMPLED, key)
        run_pipeline(gss, codes=codes)
        tables.append(yearly_table(gss))
    return sum(tables) / len(tables)


def main():
    np.random.seed(17)
    set_pyplot_params()
    if not CANDIDATES_RESAMPLED.exists():
        make_candidates_resampled()

    columns = []
    for name, codes in SETS.items():
        for key in KEYS:
            figures = key == "gss0"
            label = f"{name}_{key}"
            col = run(name if figures else label, CANDIDATES_RESAMPLED, key, figures=figures, codes=codes)
            col.name = label
            columns.append(col)
    summary = pd.concat(columns, axis=1)
    summary.to_csv(FIG_DIR / "compare_sets_key_numbers.csv")

    tables = {name: yearly(codes) for name, codes in SETS.items()}
    combined = pd.concat(tables, axis=1)
    combined.to_csv(FIG_DIR / "compare_sets_yearly.csv")

    with pd.option_context("display.width", 200, "display.max_rows", 60, "display.float_format", "{:.2f}".format):
        print(summary)
        print(combined.loc[[1974, 1980, 1990, 2000, 2010, 2018, 2021, 2022, 2024]])


if __name__ == "__main__":
    main()

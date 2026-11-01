"""Complete the square: question sets v1 and v2, data through 2018 and 2024.

The through-2024 runs come from overton_compare_sets.py; this script runs
both sets on the same resamplings cut at 2018 (the book's last year), so
the two changes (newer data, different questions) can be separated.

Writes overton_figs/compare_square_key_numbers.csv (all four cells),
compare_square_summary.csv (selected numbers, mean of the resamplings),
and the chapter's figures for v2 through 2018 in overton_figs/v2_2018/.

Usage: python overton_compare_square.py
"""

import numpy as np
import pandas as pd

from overton_data import CANDIDATES_RESAMPLED, KEYS
from overton_pipeline import FIG_DIR, run, set_pyplot_params, through_2018
from overton_question_sets import SETS

SUMMARY_ROWS = [
    "smoothed conservatism, start",
    "smoothed conservatism, end",
    "age slopes: groups / neg / pos",
    "year slopes: groups / neg / pos",
    "con-lib gap 1974",
    "con-lib gap 2018",
    "change, Conservative",
    "change, Moderate",
    "change, Liberal",
    "Liberal 1974",
    "Moderate 2000",
    "Conservative 2018",
    "Conservative 2018 - Liberal 1974",
    "Conservative last - Liberal 1974",
]


def main():
    np.random.seed(17)
    set_pyplot_params()

    columns = []
    for name, codes in SETS.items():
        for key in KEYS:
            figures = name == "v2" and key == "gss0"
            label = f"{name}_2018_{key}"
            col = run(f"{name}_2018" if figures else label, CANDIDATES_RESAMPLED, key,
                      figures=figures, transform=through_2018, codes=codes)
            col.name = label
            columns.append(col)
    cut = pd.concat(columns, axis=1)

    full = pd.read_csv(FIG_DIR / "compare_sets_key_numbers.csv", index_col=0)
    full.columns = [c.replace("_gss", "_2024_gss") for c in full.columns]
    square = pd.concat([cut, full], axis=1)

    def number(row):
        return pd.to_numeric(square.loc[row], errors="coerce")

    last = number("last year").astype(int)
    con_last = pd.Series({c: number(f"Conservative {last[c]}")[c] for c in square.columns})
    square.loc["Conservative 2018 - Liberal 1974"] = number("Conservative 2018") - number("Liberal 1974")
    square.loc["Conservative last - Liberal 1974"] = con_last - number("Liberal 1974")
    square.to_csv(FIG_DIR / "compare_square_key_numbers.csv")

    # mean over resamplings for numeric rows; the slope counts from each resampling
    cells = ["v1_2018", "v1_2024", "v2_2018", "v2_2024"]
    summary = {}
    for cell in cells:
        cols = [c for c in square.columns if c.startswith(cell)]
        values = {}
        for row in SUMMARY_ROWS:
            if "slopes" in row:
                values[row] = "; ".join(square.loc[row, cols].astype(str))
            else:
                values[row] = pd.to_numeric(square.loc[row, cols]).mean()
        summary[cell] = values
    summary = pd.DataFrame(summary)
    summary.to_csv(FIG_DIR / "compare_square_summary.csv")

    with pd.option_context("display.width", 200, "display.max_colwidth", 30):
        print(summary.map(lambda v: f"{v:.2f}" if isinstance(v, float) else v))


if __name__ == "__main__":
    main()

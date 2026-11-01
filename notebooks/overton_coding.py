"""Code each in-scope candidate as conservative response or not.

For each response, the difference in the percentage choosing it
(Conservative minus Liberal) is computed within each year and averaged
over the years, weighted by the number of respondents. Weights: wtssall,
or wtssps where wtssall is missing.

Rule: split the response scale, in order, at the point that best separates
conservatives and liberals; the conservative side is the side
conservatives choose more. `separation` is the difference for that split.
For comparison, `by_response` is the set of responses that conservatives
choose more, one at a time (not always contiguous).

Volunteered answers that aren't positions on the question ("other",
"depends", "neither", ...) always count as not conservative, as in the book.

The coding is compared with the book's codings for its fifteen questions
and with the groupings in CultureWar's variable catalog. Close calls (a
response whose difference is near zero) are flagged for review.

Only the conservative/liberal difference is used; no trends.

Writes overton_figs/candidates_coding.csv.

Usage: python overton_coding.py
"""

import re

import numpy as np
import pandas as pd

from overton_candidates import EXTRACT
from overton_data import CONSERVATIVE_VALUES
from overton_pipeline import FIG_DIR

# volunteered answers that aren't positions on the question
NOT_A_POSITION = [
    "other",
    "depends",
    "depends on age",
    "neither",
    "wdnt belong",
    "wouldnt vote",
    "r pays no income tax (vol.)",
]

CLOSE_CALL = 1.0  # percentage points

# scale order where the codes aren't in order (low to high)
SCALE_ORDER = {
    "divlaw": [1, 3, 2],  # easier, stay as is, more difficult
    "courts": [1, 3, 2],  # too harshly, about right, not harshly enough
}


def parse_labels(value_labels):
    """Map from code to label, from the gssrdoc value_labels string."""
    pairs = re.findall(r"\[(\d+)\] ([^/]+)", str(value_labels))
    return {float(code): label.strip() for code, label in pairs}


def parse_selected(selected):
    """CultureWar selected_values as a set of codes, or None."""
    if not isinstance(selected, str) or not selected.startswith("["):
        return None
    return {float(x) for x in selected.strip("[]").split(",") if x.strip()}


def response_differences(gss, varname):
    """Conservative minus Liberal percentage choosing each response, averaged over years."""
    data = gss[["year", "polviews3", "w", varname]].dropna()
    data = data[data["polviews3"].isin(["Conservative", "Liberal"])]
    rows = []
    for year, group in data.groupby("year"):
        shares = group.groupby("polviews3").apply(
            lambda df: df.groupby(varname)["w"].sum() / df["w"].sum()
        )
        if not isinstance(shares, pd.Series):
            shares = shares.stack()
        shares = shares.unstack(0).fillna(0)
        if not {"Conservative", "Liberal"} <= set(shares.columns):
            continue
        diff = (shares["Conservative"] - shares["Liberal"]) * 100
        rows.append((len(group), diff))
    weights = np.array([n for n, _ in rows], dtype=float)
    table = pd.concat([d for _, d in rows], axis=1).fillna(0)
    return (table * weights).sum(axis=1) / weights.sum()


def best_cut(positions, order):
    """Contiguous split of an ordered scale that best separates the groups.

    returns: the conservative side (a list of codes) and the size of the
             difference for that split (percentage points)
    """
    ordered = positions.reindex(order)
    cum = ordered.cumsum().iloc[:-1]
    k = cum.abs().values.argmax()
    low, high = list(order[: k + 1]), list(order[k + 1 :])
    if cum.iloc[k] > 0:
        return low, cum.iloc[k]
    return high, -cum.iloc[k]


def code_question(gss, varname, labels):
    diff = response_differences(gss, varname)
    not_position = {v for v in diff.index if labels.get(v, "").lower() in NOT_A_POSITION}
    positions = diff.drop(list(not_position))
    order = [float(v) for v in SCALE_ORDER.get(varname, sorted(positions.index))]
    conservative, separation = best_cut(positions, order)
    by_response = sorted(positions.index[positions > 0])
    close = sorted(positions.index[positions.abs() < CLOSE_CALL])
    return diff, sorted(conservative), separation, by_response, sorted(not_position), close


def describe(values, labels):
    return "; ".join(f"{v:g} {labels.get(v, '')}" for v in values)


def main():
    scope = pd.read_csv(FIG_DIR / "candidates_scope.csv", index_col=0)
    scope = scope[scope["scope"] == "opinion"]

    gss = pd.read_hdf(EXTRACT, "gss")
    gss["w"] = gss["wtssall"].fillna(gss["wtssps"])
    recode = {1: "Liberal", 2: "Liberal", 3: "Liberal", 4: "Moderate"}
    gss["polviews3"] = gss["polviews"].map(lambda x: recode.get(x, "Conservative") if pd.notna(x) else np.nan)

    rows = []
    for varname, meta in scope.iterrows():
        labels = parse_labels(meta["value_labels"])
        diff, conservative, separation, by_response, not_position, close = code_question(
            gss, varname, labels
        )
        valid = set(diff.index)
        con = set(conservative)

        book = CONSERVATIVE_VALUES.get(varname)
        book_agrees = None if book is None else set(book) == con

        cw = parse_selected(meta["cw_selected_values"])
        if cw is None:
            cw_agrees = None
        else:
            cw = cw & valid
            cw_agrees = cw == con or (valid - set(not_position) - cw) == con

        rows.append(
            dict(
                variable=varname,
                description=meta["description"],
                conservative=" ".join(f"{v:g}" for v in conservative),
                conservative_labels=describe(conservative, labels),
                separation=separation,
                by_response=" ".join(f"{v:g}" for v in by_response),
                contiguous=sorted(by_response) == sorted(conservative),
                not_position=" ".join(f"{v:g}" for v in not_position),
                differences=" ".join(f"{v:g}:{d:+.1f}" for v, d in diff.items()),
                close_calls=describe(close, labels),
                book_coding=None if book is None else " ".join(map(str, book)),
                agrees_with_book=book_agrees,
                cw_selected=None if cw is None else " ".join(f"{v:g}" for v in sorted(cw)),
                agrees_with_cw=cw_agrees,
            )
        )

    table = pd.DataFrame(rows).set_index("variable")
    table.to_csv(FIG_DIR / "candidates_coding.csv")
    print(len(table), "questions coded")
    print("agrees with book:", table["agrees_with_book"].value_counts(dropna=True).to_dict())
    print("agrees with CultureWar grouping:", table["agrees_with_cw"].value_counts(dropna=True).to_dict())
    print("with close calls:", (table["close_calls"] != "").sum())
    print("per-response rule not contiguous:", (~table["contiguous"]).sum())


if __name__ == "__main__":
    main()

"""Check whether each candidate's liberal-conservative valence is stable over time.

The method assumes each question has a fixed conservative answer (one coding
for all years). For each ranked candidate (overton_ranking.py), this computes
the conservative-liberal gap on the coded answer in each decade: the weighted
percent giving the coded conservative answer among conservatives minus among
liberals (polviews3), with a 90% interval from the Kish effective sample
sizes. Decades with fewer than 200 liberal and conservative respondents are
left out.

Flags (decided before looking at which questions they catch, 2026-10-31):
- sign change: one decade's gap credibly positive (interval above 0) and
  another's credibly negative (interval below 0);
- fades: the gap is credibly positive in some decade and, in a later decade,
  its interval includes 0. Two versions: in the last decade observed, or in
  any later decade.

Only the conservative-liberal gap is used, the quantity the selection is
based on, not trends in the answers.

Writes overton_figs/candidates_stability.csv.

Usage: python overton_stability.py
"""

import numpy as np
import pandas as pd

from overton_candidates import EXTRACT
from overton_pipeline import FIG_DIR, RECODE_POLVIEWS

MIN_RESPONDENTS = 200
Z90 = 1.645


def weighted_share(y, w):
    """Weighted mean and its standard error, using Kish's effective sample size."""
    p = np.average(y, weights=w)
    n_eff = w.sum() ** 2 / (w**2).sum()
    return p, np.sqrt(p * (1 - p) / n_eff)


def decade_gaps(y, alignment, weight, decade):
    df = pd.DataFrame(dict(y=y, p=alignment, w=weight, decade=decade)).dropna()
    df = df[df["p"].isin(["Liberal", "Conservative"])]
    rows = []
    for d, group in df.groupby("decade"):
        if len(group) < MIN_RESPONDENTS:
            continue
        con = group[group["p"] == "Conservative"]
        lib = group[group["p"] == "Liberal"]
        if len(con) == 0 or len(lib) == 0:
            continue
        pc, sc = weighted_share(con["y"].values, con["w"].values)
        pl, sl = weighted_share(lib["y"].values, lib["w"].values)
        gap, se = (pc - pl) * 100, np.hypot(sc, sl) * 100
        rows.append(dict(decade=d, gap=gap, lo=gap - Z90 * se, hi=gap + Z90 * se, n=len(group)))
    return pd.DataFrame(rows)


def flags(gaps):
    positive = gaps["lo"] > 0
    negative = gaps["hi"] < 0
    includes_zero = ~positive & ~negative
    sign_change = positive.any() and negative.any()
    if positive.any():
        first_positive = positive.idxmax()
        later = includes_zero.loc[first_positive:].iloc[1:]
        fades_any = later.any()
        fades_last = bool(includes_zero.iloc[-1]) and gaps.index[-1] > first_positive
    else:
        fades_any = fades_last = False
    return sign_change, fades_last, fades_any


def main():
    ranking = pd.read_csv(FIG_DIR / "candidates_ranking.csv", index_col=0)
    coded = pd.read_hdf(FIG_DIR / "candidates_coded.hdf", "coded")
    gss = pd.read_hdf(EXTRACT, "gss")
    weight = gss["wtssall"].fillna(gss["wtssps"])
    alignment = gss["polviews"].replace(RECODE_POLVIEWS)
    decade = (gss["year"] // 10 * 10).astype(int)

    rows = []
    for varname in ranking.index:
        gaps = decade_gaps(coded[varname], alignment, weight, decade).reset_index(drop=True)
        if len(gaps) == 0:
            continue
        sign_change, fades_last, fades_any = flags(gaps)
        rows.append(
            dict(
                variable=varname,
                rank=ranking.loc[varname, "rank"],
                description=ranking.loc[varname, "description"],
                decades=len(gaps),
                gaps=" ".join(f"{d}s:{g:+.0f}" for d, g in zip(gaps["decade"], gaps["gap"])),
                min_gap=gaps["gap"].min(),
                max_gap=gaps["gap"].max(),
                sign_change=sign_change,
                fades_last=fades_last,
                fades_any=fades_any,
            )
        )
    table = pd.DataFrame(rows).set_index("variable").sort_values("rank")
    table.to_csv(FIG_DIR / "candidates_stability.csv")

    flagged = table[table["sign_change"] | table["fades_any"]]
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 80):
        print(len(table), "candidates;",
              table["sign_change"].sum(), "sign change;",
              table["fades_last"].sum(), "fade in last decade;",
              table["fades_any"].sum(), "fade in any later decade")
        print(flagged[["rank", "description", "gaps", "sign_change", "fades_last", "fades_any"]].round(1).to_string())


if __name__ == "__main__":
    main()

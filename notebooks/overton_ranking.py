"""Rank the Overton candidates and check redundancy.

Coding: the book's codings for its fifteen questions; for the others, the
best contiguous cut from overton_coding.py.

Score: the book's find_questions model, with the coded split as the
outcome: logit(conservative ~ year + cohort + polviews), weighted
(wtssall, or wtssps where wtssall is missing; weights scaled to mean 1).
Ranked by the polviews coefficient.

Redundancy: tetrachoric correlation between two coded questions among
respondents who answered both. Pairs in SAME_TOPIC (alternate wordings on
different ballots) count as redundant.

Mode flag: asked in 2022 or 2024 but with no web responses.

No trends are computed here.

Writes overton_figs/candidates_ranking.csv.

Usage: python overton_ranking.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from overton_candidates import EXTRACT
from overton_data import CONSERVATIVE_VALUES
from overton_pipeline import FIG_DIR

MIN_OVERLAP = 200


def load():
    gss = pd.read_hdf(EXTRACT, "gss")
    gss["w"] = gss["wtssall"].fillna(gss["wtssps"])
    gss["w"] /= gss["w"].mean()
    return gss


def conservative_codes(coding):
    """Map from variable to the codes that count as conservative."""
    codes = {}
    for varname, row in coding.iterrows():
        if varname in CONSERVATIVE_VALUES:
            codes[varname] = [float(v) for v in CONSERVATIVE_VALUES[varname]]
        else:
            codes[varname] = [float(v) for v in str(row["conservative"]).split()]
    return codes


def code_responses(gss, codes):
    """1 for a conservative response, 0 for other responses, NaN if not answered."""
    coded = pd.DataFrame(index=gss.index)
    for varname, values in codes.items():
        coded[varname] = gss[varname].isin(values).astype(float).where(gss[varname].notna())
    return coded


def score(gss, coded, varname):
    data = pd.DataFrame(
        dict(y=coded[varname], year=gss["year"], cohort=gss["cohort"], polviews=gss["polviews"], w=gss["w"])
    ).dropna()
    model = smf.glm("y ~ year + cohort + polviews", data=data, family=sm.families.Binomial(), freq_weights=data["w"])
    results = model.fit()
    return results.params["polviews"], len(data)


def tetrachoric(x, y):
    """Tetrachoric correlation of two binary series (NaN where either is missing)."""
    both = pd.concat([x, y], axis=1).dropna()
    if len(both) < MIN_OVERLAP:
        return np.nan, len(both)
    a, b = both.iloc[:, 0], both.iloc[:, 1]
    p1, p2, p11 = a.mean(), b.mean(), (a * b).mean()
    if min(p1, p2) in (0, 1) or p11 in (0, min(p1, p2)):
        return np.nan, len(both)
    t1, t2 = norm.ppf(1 - p1), norm.ppf(1 - p2)

    def error(rho):
        # P(both conservative) = P(Z1 > t1, Z2 > t2)
        cov = [[1, rho], [rho, 1]]
        upper = 1 - norm.cdf(t1) - norm.cdf(t2) + multivariate_normal.cdf([t1, t2], cov=cov)
        return upper - p11

    return brentq(error, -0.999, 0.999), len(both)


def mode_flag(gss, varname):
    """True if asked in 2022 or 2024 but with no web responses."""
    recent = gss[gss["year"].isin([2022, 2024]) & gss[varname].notna()]
    if len(recent) == 0:
        return False
    return (recent["mode"] == 4).sum() == 0


def main():
    coding = pd.read_csv(FIG_DIR / "candidates_coding.csv", index_col=0)
    scope = pd.read_csv(FIG_DIR / "candidates_scope.csv", index_col=0)
    gss = load()
    codes = conservative_codes(coding)
    coded = code_responses(gss, codes)

    rows = []
    for varname in codes:
        param, n = score(gss, coded, varname)
        meta = scope.loc[varname]
        rows.append(
            dict(
                variable=varname,
                description=meta["description"],
                param=param,
                n=n,
                waves=meta["waves"],
                first=meta["first"],
                last=meta["last"],
                conservative=" ".join(f"{v:g}" for v in codes[varname]),
                separation=coding.loc[varname, "separation"],
                in_person_only=mode_flag(gss, varname),
                in_book_15=varname in CONSERVATIVE_VALUES,
            )
        )
    table = pd.DataFrame(rows).set_index("variable").sort_values("param", ascending=False)
    table["rank"] = range(1, len(table) + 1)
    table.to_csv(FIG_DIR / "candidates_ranking.csv")
    coded.to_hdf(FIG_DIR / "candidates_coded.hdf", key="coded", mode="w", complevel=6)

    with pd.option_context("display.width", 200, "display.max_rows", 80):
        print(table.head(60).drop(columns=["description"]).round(3))

    # calibration pairs for the redundancy threshold
    pairs = [
        ("abany", "abnomore"), ("abany", "absingle"), ("abany", "abpoor"), ("abany", "abdefect"),
        ("abany", "abrape"), ("abany", "abhlth"),
        ("natfare", "natenvir"), ("homosex", "premarsx"), ("premarsx", "teensex"),
        ("premarsx", "xmarsex"), ("homosex", "colhomo"), ("homosex", "spkhomo"),
        ("colhomo", "spkhomo"), ("colhomo", "libhomo"), ("helpblk", "natrace"),
        ("affrmact", "fejobaff"), ("helppoor", "eqwlth"), ("fefam", "fepresch"),
    ]
    print("\nTetrachoric correlations for calibration:")
    for x, y in pairs:
        r, n = tetrachoric(coded[x], coded[y])
        print(f"  {x:9s} {y:9s} r={r:5.2f}  n={n}")


if __name__ == "__main__":
    main()

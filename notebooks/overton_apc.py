"""Period, period-cohort, and age models of conservatism (Phase 4b).

The outcome is each respondent's estimated conservatism from the grid
method (posterior mean number of conservative responses), divided by the
number of questions, so it's a mean between 0 and 1. The models are ported
from CultureWar (commit 0f54767), with the same priors and defaults and the
continuous-outcome ("mean") likelihood from its Task 165:

- period: CultureWar's time_model. logit(mu_t) = alpha + period_t (RW2 on
  an annual grid) + epsilon_t (year noise, sigma_short). Each surveyed
  year's weighted mean ~ Normal(mu_t, s_t / sqrt(n_eff_t)), with the
  weighted SD s_t plugged in.
- period_cohort: CultureWar's model10. logit(mu_ct) = alpha + cohort_c
  (RW2 on annual birth cohorts) + period_t (RW2) + epsilon_t. Each cohort x
  year cell's weighted mean ~ Normal(mu_ct, sigma_resid_t / sqrt(n_eff_ct)).
- age: CultureWar's cohort_model, which is the time model with another
  index; here the index is age, within a window of survey years.

Weights: wtsmerge (wtssall, or wtssps where it's missing), on the original
extract, not resampled. The grid's difficulties use weighted prevalence.
n_eff is Kish's effective sample size.

Two steps; the fit needs only numpy, pandas, and PyMC (CultureWar env):

    python overton_apc.py prepare              # local: write respondent data
    python overton_apc.py fit --model period --set v1 [--by polviews3=Liberal]
    python overton_apc.py fit --model age --set v1 --window 1972-1979
    python overton_apc.py fit --model age --set v1 --cohorts 1946-1964

Results go to overton_apc/results/{run}_*.csv and _diagnostics.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
APC_DIR = HERE / "overton_apc"
DATA_DIR = APC_DIR / "data"
RESULTS_DIR = APC_DIR / "results"
MIN_ANSWERED = 5
# the domain sets are small (4 questions in econA), so they use a minimum of 2 (Allen, 2026-10-31)
MIN_ANSWERED_BY_SET = {"econA": 2, "econB": 2, "econC": 2, "social": 2}

ECON_A = ["helppoor", "helpnot", "helpsick", "eqwlth"]
ECON_B = ECON_A + ["natenvir", "nateduc", "natmass", "natpark", "tax", "getahead"]
ECON_C = ECON_B + ["helpblk", "natdrug"]
NOT_SOCIAL = ECON_C + ["natarms", "nataid"]
DOMAIN_SETS = {"econA": ECON_A, "econB": ECON_B, "econC": ECON_C, "social": None}  # social: the rest of the pool
SCALE = 15  # results are reported as conservative responses out of 15, whatever the set size

# defaults from CultureWar time_model/settings.py and model10/settings.py
PERIOD_SETTINGS = dict(sigma_t_scale=0.125, slope_init_scale=0.2, sigma_short_scale=0.5, target_accept=0.9)
COHORT_SETTINGS = dict(sigma_c_scale=0.125, sigma_t_scale=0.125, slope_init_scale=0.2,
                       sigma_short_scale=0.2, sigma_resid_scale=0.5, target_accept=0.95)


def question_sets():
    """v1, v2, and the nested top-N sets from the ranked pool.

    p{N}: primary specification, the top N after the stability rule and
    without the confidence-in-institutions questions (decided 2026-10-31,
    after independent review). s{N}: sensitivity, the top N after the
    stability rule, keeping the confidence questions that pass it. Sizes run
    in steps of 5 up to the whole pool. econA, econB, econC, social: domain
    subsets of the primary pool (core, broad, and inclusive economic; social
    without the "both" and "neither" questions).
    """
    from overton_larger_sets import CONFIDENCE, codes_for, ranked_questions
    from overton_question_sets import SETS

    stability = pd.read_csv(HERE / "overton_figs/candidates_stability.csv", index_col=0)
    unstable = set(stability.index[stability["sign_change"] | stability["fades_any"]])
    sets = dict(SETS)
    for prefix, exclude_confidence in [("p", True), ("s", False)]:
        ranked = ranked_questions(CONFIDENCE if exclude_confidence else None, drop=unstable)
        codes = codes_for(ranked)
        names = list(ranked.index)
        sizes = list(range(15, len(names), 5)) + [len(names)]
        for n in sizes:
            sets[f"{prefix}{n}"] = {v: codes[v] for v in names[:n]}

    # social and economic subsets of the primary pool, prespecified after review
    # (overton_question_domains.md, 2026-10-31)
    pool = sets[f"p{len(names_primary(sets))}"]
    for name, members in DOMAIN_SETS.items():
        members = members or [v for v in pool if v not in NOT_SOCIAL]
        sets[name] = {v: pool[v] for v in members}
    return sets


def names_primary(sets):
    """Names of the full primary pool (the largest p{N} set)."""
    return max((v for k, v in sets.items() if k.startswith("p")), key=len)


def prepare(names=None):
    """Weighted grid estimates for each question set, on the original extract."""
    from overton_data import CANDIDATES
    from overton_pipeline import RECODE_POLVIEWS, estimate_conservatism, make_questions

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gss = pd.read_hdf(CANDIDATES, "gss")
    gss["year"] = gss["year"].astype(int)
    gss["weight"] = gss["wtssall"].fillna(gss["wtssps"])
    sets = question_sets()
    for name in names or sets:
        codes = sets[name]
        questions = make_questions(gss, codes)
        con, _, _ = estimate_conservatism(questions, weights=gss["weight"])
        answered = questions.notna().sum(axis=1)
        df = gss[["year", "id", "age", "cohort", "polviews", "weight"]].copy()
        df["polviews3"] = df["polviews"].replace(RECODE_POLVIEWS)
        df["n_questions"] = questions.shape[1]
        df["conservatism"] = con
        df["answered"] = answered
        min_answered = MIN_ANSWERED_BY_SET.get(name, MIN_ANSWERED)
        df = df[(answered >= min_answered) & df["cohort"].notna() & df["age"].notna()]
        df["y"] = df["conservatism"] / df["n_questions"]
        df.to_parquet(DATA_DIR / f"{name}.parquet")
        print(name, len(df), df.groupby("year")["y"].mean().iloc[[0, -1]].round(3).to_dict())


def weighted_cells(df, keys):
    """Weighted mean, weighted variance, and Kish n_eff of y in each cell."""

    def summarize(g):
        w = g["weight"].values
        y = g["y"].values
        mean = np.average(y, weights=w)
        var = np.average((y - mean) ** 2, weights=w)
        return pd.Series(dict(p_weighted=mean, var_weighted=var, n=len(g), n_eff=w.sum() ** 2 / (w**2).sum()))

    return df.groupby(keys).apply(summarize).reset_index()


def rw2(name, n, sigma, slope_init_scale, dims):
    """CultureWar's mean-centered RW2 (double cumsum, non-centered, free initial slope)."""
    import pymc as pm

    z_delta = pm.Normal(f"{name}_z_delta", 0.0, 1.0, shape=n - 2)
    delta = sigma * z_delta
    v0 = pm.Normal(f"{name}_slope_init", 0.0, slope_init_scale)
    rw1 = v0 + pm.math.concatenate([pm.math.zeros(1), pm.math.cumsum(delta)])
    f_raw = pm.math.concatenate([pm.math.zeros(1), pm.math.cumsum(rw1)])
    return pm.Deterministic(name, f_raw - pm.math.mean(f_raw), dims=dims)


def build_series_model(cells, index, grid, s=PERIOD_SETTINGS):
    """Time model (period or age): RW2 over `grid`, year noise, Normal on cell means."""
    import pymc as pm

    idx = cells[index].map({v: i for i, v in enumerate(grid)}).values
    se = np.sqrt(np.maximum(cells["var_weighted"].values, 1e-4) / np.maximum(cells["n_eff"].values, 1.0))
    with pm.Model(coords={"grid": grid}) as m:
        alpha = pm.Normal("alpha", 0.0, 1.0)
        sigma_t = pm.Exponential("sigma_t", 1.0 / s["sigma_t_scale"])
        trend = rw2("trend", len(grid), sigma_t, s["slope_init_scale"], dims="grid")
        eta = pm.Deterministic("eta", alpha + trend, dims="grid")
        pm.Deterministic("mu", pm.math.sigmoid(eta), dims="grid")
        sigma_short = pm.HalfNormal("sigma_short", s["sigma_short_scale"])
        epsilon_z = pm.ZeroSumNormal("epsilon_z", sigma=1.0, dims="grid")
        epsilon = pm.Deterministic("epsilon", sigma_short * epsilon_z, dims="grid")
        pm.Normal("y_mean", mu=pm.math.sigmoid(eta[idx] + epsilon[idx]), sigma=se, observed=cells["p_weighted"].values)
    return m


def build_period_cohort_model(cells, years, cohorts, s=COHORT_SETTINGS):
    """model10: RW2 cohort + RW2 period + year noise, Normal on cohort x year cell means."""
    import pymc as pm

    t = cells["year"].map({v: i for i, v in enumerate(years)}).values
    c = cells["cohort"].map({v: i for i, v in enumerate(cohorts)}).values
    with pm.Model(coords={"year": years, "cohort_grid": cohorts}) as m:
        alpha = pm.Normal("alpha", 0.0, 1.0)
        sigma_c = pm.Exponential("sigma_c", 1.0 / s["sigma_c_scale"])
        cohort = rw2("cohort", len(cohorts), sigma_c, s["slope_init_scale"], dims="cohort_grid")
        sigma_t = pm.Exponential("sigma_t", 1.0 / s["sigma_t_scale"])
        period = rw2("period", len(years), sigma_t, s["slope_init_scale"], dims="year")
        sigma_short = pm.Exponential("sigma_short", 1.0 / s["sigma_short_scale"])
        epsilon_z = pm.ZeroSumNormal("epsilon_z", sigma=1.0, dims="year")
        epsilon = pm.Deterministic("epsilon", sigma_short * epsilon_z, dims="year")
        p_obs = pm.math.sigmoid(alpha + cohort[c] + period[t] + epsilon[t])
        sigma_resid = pm.HalfNormal("sigma_resid", s["sigma_resid_scale"], dims="year")
        se = sigma_resid[t] / np.sqrt(cells["n_eff"].values)
        pm.Normal("y_mean", mu=p_obs, sigma=se, observed=cells["p_weighted"].values)
    return m


def sample(m, target_accept, tune, draws, chains, seed, override=None):
    import nutpie

    target_accept = override or target_accept
    compiled = nutpie.compile_pymc_model(m)
    return nutpie.sample(compiled, tune=tune, draws=draws, chains=chains, seed=seed,
                         target_accept=target_accept, progress_bar=False)


def draws_of(trace, name):
    """Posterior draws stacked over chains: (draws, ...)."""
    values = trace["posterior"][name].values
    return values.reshape(-1, *values.shape[2:])


def interval(x, axis=0):
    return np.mean(x, axis=axis), np.quantile(x, 0.05, axis=axis), np.quantile(x, 0.95, axis=axis)


def diagnostics_for(trace, names):
    import arviz as az

    rhat = az.rhat(trace, var_names=names)
    ess = az.ess(trace, var_names=names)
    return dict(
        divergences=int(trace["sample_stats"]["diverging"].values.sum()),
        rhat_max=float(max(np.nanmax(rhat[v].values) for v in names)),
        ess_bulk_min=float(min(np.nanmin(ess[v].values) for v in names)),
    )


def fit_series(df, index, grid, scale, stem, args):
    cells = weighted_cells(df, [index])
    m = build_series_model(cells, index, grid)
    trace = sample(m, PERIOD_SETTINGS["target_accept"], args.tune, args.draws, args.chains, args.seed, args.target_accept)
    mu = draws_of(trace, "mu") * scale
    mean, lo, hi = interval(mu)
    out = pd.DataFrame({index: grid, "mean": mean, "lo90": lo, "hi90": hi}).merge(
        cells.assign(observed=cells["p_weighted"] * scale)[[index, "observed", "n", "n_eff"]], how="left")
    out.to_csv(f"{stem}_series.csv", index=False)

    # linear slope of the latent mean per decade (count scale), over the values with data
    x = np.asarray(grid, float)
    observed = np.isin(x, cells[index].values)
    xs = x[observed] - x[observed].mean()
    slope = (mu[:, observed] - mu[:, observed].mean(axis=1, keepdims=True)) @ xs / (xs @ xs) * 10
    summary = dict(zip(["slope_per_decade", "slope_lo90", "slope_hi90"], map(float, interval(slope))))
    summary["change_first_last"] = float(np.mean(mu[:, observed][:, -1] - mu[:, observed][:, 0]))
    return trace, summary, ["alpha", "sigma_t", "sigma_short", "trend"]


def fit_period_cohort(df, scale, stem, args):
    years = list(range(df["year"].min(), df["year"].max() + 1))
    cohorts = list(range(int(df["cohort"].min()), int(df["cohort"].max()) + 1))
    df = df.assign(cohort=df["cohort"].astype(int))
    cells = weighted_cells(df, ["cohort", "year"])
    m = build_period_cohort_model(cells, years, cohorts)
    trace = sample(m, COHORT_SETTINGS["target_accept"], args.tune, args.draws, args.chains, args.seed, args.target_accept)

    alpha = draws_of(trace, "alpha")[:, None, None]
    cohort = draws_of(trace, "cohort")
    period = draws_of(trace, "period")

    # weighted cohort composition of each surveyed year
    survey_years = sorted(df["year"].unique())
    comp = df.groupby(["year", "cohort"])["weight"].sum().unstack(fill_value=0).reindex(columns=cohorts, fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0).loc[survey_years].values  # (years, cohorts)

    def level(comp_row, t_idx):
        """Population mean (count scale) with cohort mix comp_row and period at year index t_idx."""
        p = 1 / (1 + np.exp(-(alpha[:, 0, 0, None] + cohort + period[:, t_idx, None])))
        return (p * comp_row).sum(axis=1) * scale

    ti = {y: years.index(y) for y in survey_years}
    y0, y1 = survey_years[0], survey_years[-1]
    c0, c1 = comp[0], comp[-1]
    p0, p1 = level(c0, ti[y0]), level(c1, ti[y1])
    fix_cohort = level(c0, ti[y1])  # period changes, cohort mix held at y0
    fix_period = level(c1, ti[y0])  # cohort mix changes, period held at y0
    total = p1 - p0
    d_cohort = 0.5 * ((fix_period - p0) + (total - (fix_cohort - p0)))
    d_period = total - d_cohort
    decades = (y1 - y0) / 10

    # period component, standardized to the pooled cohort mix of all years
    pooled = df.groupby("cohort")["weight"].sum().reindex(cohorts, fill_value=0).values
    pooled = pooled / pooled.sum()
    std_period = np.stack([level(pooled, j) for j in range(len(years))], axis=1)
    # cohort component, at the average period
    p_c = 1 / (1 + np.exp(-(alpha[:, 0] + cohort + period.mean(axis=1, keepdims=True)))) * scale

    rows = []
    for j, y in enumerate(years):
        rows.append(dict(component="period", index=y, **dict(zip(["mean", "lo90", "hi90"], interval(std_period[:, j])))))
    for k, c in enumerate(cohorts):
        rows.append(dict(component="cohort", index=c, **dict(zip(["mean", "lo90", "hi90"], interval(p_c[:, k])))))
    pd.DataFrame(rows).to_csv(f"{stem}_components.csv", index=False)

    summary = {}
    for label, x in [("total_change", total), ("generational_replacement", d_cohort), ("changing_minds", d_period),
                     ("period_change_per_decade", d_period / decades)]:
        mean, lo, hi = interval(x)
        summary.update({label: float(mean), f"{label}_lo90": float(lo), f"{label}_hi90": float(hi)})
    summary["window"] = f"{y0}-{y1}"

    # the within-cohort slope: a straight line fit to the standardized period
    # component over the surveyed years, per decade, on the count scale (each draw)
    x = np.array(survey_years, float) - np.mean(survey_years)
    sp = std_period[:, [ti[y] for y in survey_years]]
    slope = (sp - sp.mean(axis=1, keepdims=True)) @ x / (x @ x) * 10
    summary.update(dict(zip(["period_slope_per_decade", "period_slope_lo90", "period_slope_hi90"],
                            map(float, interval(slope)))))

    # the same on the logit scale
    pv = period[:, [ti[y] for y in survey_years]]
    slope = (pv - pv.mean(axis=1, keepdims=True)) @ x / (x @ x) * 10
    summary.update(dict(zip(["period_logit_slope_per_decade", "lo90", "hi90"], map(float, interval(slope)))))
    return trace, summary, ["alpha", "sigma_c", "sigma_t", "sigma_short", "cohort", "period", "sigma_resid"]


def fit(args):
    df = pd.read_parquet(DATA_DIR / f"{args.set}.parquet")
    label = [args.model, args.set]
    if args.by:
        column, value = args.by.split("=")
        values = pd.Series([value]).astype(df[column].dtype).iloc[0] if column != "polviews3" else value
        df = df[df[column] == values]
        label.append(f"{column}-{value}")
    if args.window:
        first, last = map(int, args.window.split("-"))
        df = df[df["year"].between(first, last)]
        label.append(f"window-{args.window}")
    if args.cohorts:
        first, last = map(int, args.cohorts.split("-"))
        df = df[df["cohort"].between(first, last)]
        label.append(f"cohorts-{args.cohorts}")
    run = "_".join(label) + args.tag
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / run
    n_questions = int(df["n_questions"].iloc[0])
    scale = SCALE

    start = time.time()
    if args.model == "period":
        grid = list(range(df["year"].min(), df["year"].max() + 1))
        trace, summary, names = fit_series(df, "year", grid, scale, stem, args)
    elif args.model == "age":
        df = df.assign(age=df["age"].astype(int))
        grid = list(range(df["age"].min(), df["age"].max() + 1))
        trace, summary, names = fit_series(df, "age", grid, scale, stem, args)
    else:
        trace, summary, names = fit_period_cohort(df, scale, stem, args)

    summary.update(run=run, model=args.model, set=args.set, by=args.by, window=summary.get("window", args.window),
                   cohorts=args.cohorts, age_range=f"{int(df['age'].min())}-{int(df['age'].max())}",
                   respondents=len(df), n_questions=n_questions, elapsed_seconds=round(time.time() - start, 1),
                   tune=args.tune, draws=args.draws, chains=args.chains, seed=args.seed,
                   target_accept=args.target_accept)
    summary.update(diagnostics_for(trace, names))
    with open(f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare", "fit"])
    parser.add_argument("--model", choices=["period", "period_cohort", "age"], default="period")
    parser.add_argument("--set", default="v1")
    parser.add_argument("--sets", nargs="*", default=None, help="prepare: which sets (default all)")
    parser.add_argument("--by", default=None, help="subgroup, e.g. polviews3=Liberal or polviews=1")
    parser.add_argument("--window", default=None, help="years, e.g. 1972-1979 (age model)")
    parser.add_argument("--cohorts", default=None, help="years of birth, e.g. 1946-1964 (age model by cohort)")
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-accept", type=float, default=None, help="default: CultureWar's setting for the model")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.sets)
    else:
        fit(args)


if __name__ == "__main__":
    main()

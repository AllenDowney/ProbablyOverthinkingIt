"""Ricardo Vieira's IRT models for the Overton analysis (Phase 4).

Two models from the bayesian-bakeoff code, both 1PL (Rasch):
    logit P(conservative response) = e_i - d_j
- m1, unpooled: d_j ~ N(0, 1), e_i ~ N(0, 4). The same model as the
  book's grid method (whose prior on e is also N(0, 4)), except that the
  difficulties are estimated rather than set to -logit(prevalence).
- m2, hierarchical: e_i ~ N(0, e_std), e_std ~ HalfNormal(4); written
  non-centered (e = e_std * z), which samples better for respondents who
  answered few questions.

Unlike the talk, all respondents who answered at least one question are
fit, not a subsample. The data are the weighted resamplings, so there are
no weights in the model.

Two steps, so the cluster job needs only numpy, pandas, and PyMC:

    python overton_irt.py prepare [--sets pool]   # local: write the coded answers
    python overton_irt.py fit --set v1 --key gss0 --model m1   # Unity

prepare writes overton_irt/data/{set}_{key}.npz. fit writes
overton_irt/results/{set}_{key}_{model}_*: item summaries (CSV),
respondent summaries (parquet), diagnostics (JSON), and the posterior of
the item parameters (NetCDF). The respondent draws aren't saved; their
summaries are computed in the job.

Respondent summaries: posterior mean and sd of e, and of the expected
number of conservative responses, sum_j expit(e_i - d_j), the same
quantity the grid method reports.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IRT_DIR = HERE / "overton_irt"
DATA_DIR = IRT_DIR / "data"
RESULTS_DIR = IRT_DIR / "results"


def question_sets():
    """v1, v2, and "pool": every candidate that passes the selection rules
    (overton_larger_sets.py, including the confidence items) plus any v1
    question not among them. A 1PL difficulty doesn't depend on which other
    questions are in the set, so the pool fit gives difficulties for any
    subset."""
    from overton_larger_sets import codes_for, ranked_questions
    from overton_question_sets import SETS, SET_V1

    pool = {v: [float(c) for c in codes] for v, codes in SET_V1.items()}
    pool.update(codes_for(ranked_questions()))
    return dict(SETS, pool=pool)


def prepare(names=None):
    """Write the coded answers for each question set and resampling."""
    from overton_data import CANDIDATES_RESAMPLED, KEYS
    from overton_pipeline import load_gss, make_questions
    from scipy.special import logit

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sets = question_sets()
    for name in names or sets:
        codes = sets[name]
        for key in KEYS:
            gss = load_gss(CANDIDATES_RESAMPLED, key)
            questions = make_questions(gss, codes)
            answered = questions.notna().sum(axis=1)
            keep = answered > 0
            questions = questions[keep]
            long = questions.reset_index(drop=True).stack().reset_index()
            long.columns = ["resp", "item", "y"]
            items = list(questions.columns)
            np.savez_compressed(
                DATA_DIR / f"{name}_{key}.npz",
                resp=long["resp"].values.astype(np.int32),
                item=long["item"].map({v: j for j, v in enumerate(items)}).values.astype(np.int32),
                y=long["y"].values.astype(np.int8),
                items=np.array(items),
                index=questions.index.values,
                answered=answered[keep].values.astype(np.int16),
                heuristic=-logit(questions.mean().values),
            )
            print(name, key, len(questions), len(long))


def build_model(data, model):
    import pymc as pm

    items = list(data["items"])
    n = len(data["index"])
    coords = {"item": items, "resp": np.arange(n)}
    with pm.Model(coords=coords) as m:
        d = pm.Normal("d", 0, 1, dims="item")
        if model == "m1":
            e = pm.Normal("e", 0, 4, dims="resp")
        elif model == "m2":
            e_std = pm.HalfNormal("e_std", 4)
            z = pm.Normal("z", 0, 1, dims="resp")
            e = pm.Deterministic("e", e_std * z, dims="resp")
        else:
            raise ValueError(model)
        pm.Bernoulli("y", logit_p=e[data["resp"]] - d[data["item"]], observed=data["y"])
    return m


def expected_count(e, d, chunk=5000):
    """Mean and sd over draws of sum_j expit(e_i - d_j); e is (draws, n), d is (draws, items)."""
    from scipy.special import expit

    n = e.shape[1]
    mean, sd = np.empty(n), np.empty(n)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        count = expit(e[:, sl, None] - d[:, None, :]).sum(axis=2)
        mean[sl], sd[sl] = count.mean(axis=0), count.std(axis=0)
    return mean, sd


def fit(name, key, model, tune, draws, chains, seed, subsample=None, tag=""):
    import arviz as az
    import nutpie
    import pymc as pm

    loaded = np.load(DATA_DIR / f"{name}_{key}.npz", allow_pickle=True)
    data = {k: loaded[k] for k in loaded.files}
    if subsample:
        # smoke tests: a random subset of respondents
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(data["index"]), subsample, replace=False))
        new_idx = -np.ones(len(data["index"]), dtype=np.int32)
        new_idx[chosen] = np.arange(subsample)
        rows = new_idx[data["resp"]] >= 0
        data["resp"] = new_idx[data["resp"][rows]]
        data["item"], data["y"] = data["item"][rows], data["y"][rows]
        data["index"], data["answered"] = data["index"][chosen], data["answered"][chosen]

    m = build_model(data, model)
    start = time.time()
    compiled = nutpie.compile_pymc_model(m)
    trace = nutpie.sample(compiled, tune=tune, draws=draws, chains=chains, seed=seed, progress_bar=False)
    elapsed = time.time() - start

    post = trace["posterior"]
    e = post["e"].values.reshape(-1, len(data["index"]))
    d = post["d"].values.reshape(-1, len(data["items"]))
    count_mean, count_sd = expected_count(e, d)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"{name}_{key}_{model}{tag}"

    item_vars = ["d"] + (["e_std"] if model == "m2" else [])
    summary = az.summary(trace, var_names=item_vars)
    summary.to_csv(f"{stem}_items.csv")

    respondents = pd.DataFrame(
        dict(
            index=data["index"],
            answered=data["answered"],
            e_mean=e.mean(axis=0),
            e_sd=e.std(axis=0),
            count_mean=count_mean,
            count_sd=count_sd,
        )
    )
    respondents.to_parquet(f"{stem}_respondents.parquet")

    diverging = trace["sample_stats"]["diverging"].values
    e_rhat = az.rhat(trace, var_names=["e"])["e"].values
    e_ess = az.ess(trace, var_names=["e"])["e"].values
    diagnostics = dict(
        set=name, key=key, model=model, tune=tune, draws=draws, chains=chains, seed=seed,
        respondents=len(data["index"]), observations=len(data["y"]),
        elapsed_seconds=round(elapsed, 1),
        divergences=int(diverging.sum()),
        items_rhat_max=float(summary["r_hat"].max()),
        items_ess_bulk_min=float(summary["ess_bulk"].min()),
        e_rhat_max=float(np.nanmax(e_rhat)),
        e_ess_bulk_min=float(np.nanmin(e_ess)),
    )
    with open(f"{stem}_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    # posterior of the item parameters only; the respondent draws are large
    az.extract(trace, var_names=item_vars, combined=False).to_netcdf(f"{stem}_items.nc")
    print(json.dumps(diagnostics, indent=2))
    print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare", "fit"])
    parser.add_argument("--set", default="v1")
    parser.add_argument("--sets", nargs="*", default=None, help="prepare: which sets (default all)")
    parser.add_argument("--key", default="gss0")
    parser.add_argument("--model", default="m1", choices=["m1", "m2"])
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.sets)
    else:
        fit(args.set, args.key, args.model, args.tune, args.draws, args.chains, args.seed, args.subsample, args.tag)


if __name__ == "__main__":
    main()

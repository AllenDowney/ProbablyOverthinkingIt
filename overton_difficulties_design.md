# Which question difficulties should the conservatism score use?

*A design question. Allen Downey, 2026-11-01.*

## Background

We are updating an analysis of the "Overton paradox" (Chapter 12 of *Probably Overthinking It*) with General Social Survey (GSS) data through 2024. Each respondent's conservatism is estimated from their answers to a set of opinion questions, each with a fixed "conservative" answer, using a one-parameter (Rasch) item response model:

P(conservative answer to question j | respondent i) = expit(θᵢ − dⱼ)

where θᵢ is the respondent's conservatism and dⱼ is the question's difficulty. Respondents answer different subsets of questions (the GSS uses split ballots), and the model puts them on a common scale. The score is computed on a grid of θ values with a wide prior (Normal, SD 4), using the difficulties as fixed inputs. It's reported as the expected number of conservative answers out of 15. The score then feeds everything else: trends by alignment, the Overton gap (the average self-described conservative in 2024 minus the average liberal in 1974), and the age–period–cohort models.

So the question is where the difficulties come from.

## The options we have tried

1. **Heuristic (the book's method):** dⱼ = −logit(pⱼ), where pⱼ is the weighted share of conservative answers among the people asked question j. This is fast and simple, and it treats every question's difficulty as if every respondent were average.
2. **Bayesian fit (m1):** Ricardo Vieira's PyMC model, with difficulties and respondents' θ estimated jointly by MCMC on all 66 candidate questions. There were three runs, one on each weighted resampling of the data (the model has no survey weights). Each took about 11 hours on the Unity cluster, and their difficulties are averaged.
3. **Heuristic times a constant:** the heuristic stretched by 1.135, the slope of the Bayesian difficulties on the heuristic ones.
4. **EM on the grid** (new; a prototype, not yet in the pipeline): marginal maximum likelihood, the standard estimator for this model (Bock and Aitkin's EM algorithm). It alternates two steps:
   - compute each respondent's posterior on the grid, with the current difficulties and the same prior as the score;
   - for each question, solve for the difficulty at which the expected share of conservative answers among the people asked equals the observed share.

   It takes the survey weights directly, runs on the original data with no resampling, and converges in 5 iterations (about 6 seconds on the 66-question pool). EM can be fit on the whole 66-question pool, like the Bayesian fit, or separately on each question set.

## What we found

**The Bayesian difficulties are the heuristic ones, stretched and slightly tilted by year.**

![Bayesian vs heuristic difficulties](notebooks/overton_figs/lab/pool_difficulties.png)

- **Correlation 0.999, slope 1.14, intercept about 0.** The stretch is the expected attenuation: averaging expit(θ − d) over respondents who vary in θ pulls the observed share toward 50%, so −logit(p) understates |d| by a factor of about √(1 + πσ²/8). A slope of 1.14 corresponds to σ ≈ 0.87.
- **The rest is "who was asked".** After removing the stretch, each question's leftover difference (SD 0.045) correlates −0.9 with the average survey year of the people asked. A question asked mostly in recent years was asked of a more liberal population, so its conservative answer is rarer and the heuristic calls it harder than it is. The model accounts for who answered; the heuristic doesn't. The extremes are `discaff` (mean year 2012, −0.17) and `racmar` (1986, +0.09).
- **EM reproduces the Bayesian difficulties:** correlation 1.000, slope 1.00, average difference 0.017 (the Bayesian posterior SD of each difficulty is about 0.02). For the heuristic, the average difference is 0.127. Fit on each set separately, EM also matches the stretch of earlier Bayesian fits on single sets (slope 1.41 vs 1.42 for the book's 15).

**Effect on the results.** These are weighted yearly means, out of 15 conservative answers. They aren't the model-based estimates, but every row uses the same respondents, so the differences between rows are what the difficulties do. The sets:
- v1: the book's 15 questions;
- p15–p43: the top N of the primary pool, where p43 is the full pool and the basis of the headline;
- s50: the sensitivity pool, which keeps the confidence questions;
- econA: core economic questions, starting in 1975;
- econB: broad economic questions;
- social: the social questions.

| Set | Difficulties | Overton gap | Change, conservatives | Change, liberals |
|---|---|---|---|---|
| v1 | heuristic | 1.72 | −0.85 | −3.00 |
| | scaled 1.135 | 1.71 | −0.86 | −3.00 |
| | Bayesian (pool) | 1.70 | −0.87 | −3.00 |
| | EM (pool) | 1.70 | −0.87 | −3.00 |
| | EM (per set) | 1.69 | −0.89 | −3.01 |
| p15 | heuristic | 2.27 | −0.31 | −3.42 |
| | Bayesian / EM (pool) | 2.29 | −0.30 | −3.40 |
| | EM (per set) | 2.34 | −0.28 | −3.34 |
| p43 | heuristic | 1.91 | −0.64 | −2.81 |
| | scaled 1.135 | 1.88 | −0.68 | −2.83 |
| | Bayesian / EM (pool) | 1.80 | −0.76 | −2.90 |
| | EM (per set) | 1.66 | −0.91 | −3.03 |
| s50 | heuristic | 2.09 | −0.01 | −2.39 |
| | Bayesian / EM (pool) | 1.98 | −0.13 | −2.48 |
| econA | heuristic | 3.79 | +0.93 | −1.46 |
| | Bayesian / EM (pool) | 3.76 | +0.89 | −1.49 |
| | EM (per set) | 3.55 | +0.71 | −1.64 |
| econB | heuristic | 2.63 | +0.47 | −1.84 |
| | Bayesian / EM (pool) | 2.61 | +0.45 | −1.85 |
| social | heuristic | 1.63 | −1.20 | −2.75 |
| | scaled 1.135 | 1.55 | −1.28 | −2.81 |
| | Bayesian / EM (pool) | 1.47 | −1.36 | −2.88 |
| | EM (per set) | 1.22 | −1.63 | −3.09 |

(Bayesian and EM pool difficulties agree to 0.01 or better in every row. Full table, including the scaled heuristic for every set, in `notebooks/overton_figs/difficulty_options.csv` and `em_per_set.csv`; EM pool difficulties in `em_difficulties.csv`.)

The comparison on the three resamplings, using the book's lowess-smoothed summary, shows the same pattern: switching from heuristic to Bayesian difficulties lowers the Overton gap by 0.0 to 0.15. For example, p43 goes from 0.90 to 0.75 and social from 0.30 to 0.22 (`notebooks/overton_figs/pool_difficulty_results.csv`).

**What this shows:**
- **No conclusion changes.** Every sign is the same, and the differences are small next to the effect of the question set (the Overton gap ranges from about 1.7 to 2.3 across N in this table).
- **The shift is systematic, not noise.** Every set moves the same way, toward a smaller gap and a larger move left among conservatives. That follows from the "who was asked" correction: questions asked mostly in recent years get lower difficulties. The size of the shift depends on how many such questions a set has: it's largest for p43, social, and s50, and negligible for the book's 15.
- **Stretching alone fixes about half of it.** The rest is the "who was asked" correction.
- **The scale of the difficulties depends on the questions they're fitted with**, and this matters more than heuristic vs model. Fit on the 66-question pool, the difficulties are stretched by 1.14; fit on a single set, by 1.2 to 1.9. With one trait and equal discriminations, the stretch reflects how strongly the set's questions hang together. The four core economic questions are one battery and hang together tightly (1.94); a 66-question mix of topics less so. On social questions, the Overton gap goes from 1.47 (pool difficulties) to 1.22 (per set).

## Cost

| Option | Time | Weights | Accounts for who was asked | Notes |
|---|---|---|---|---|
| Heuristic | instant | yes | no | the book's method; simple to explain |
| Scaled heuristic | instant | yes | no | the scale factor has to come from somewhere (a model fit) |
| Bayesian (m1) | about 11 h per resampling, on a cluster | via resampling | yes | full posterior for each difficulty; has to be rerun whenever the pool or the data change |
| EM, pool | seconds | yes | yes | point estimates only; difficulty uncertainty (SD about 0.02) ignored, as in every other option except the Bayesian fit |
| EM, per set | seconds per set | yes | yes | each set gets its own scale; scores from different sets are then on slightly different footings |

## The design question

**Which difficulties should the score use?**

- (a) **Heuristic**, as in the book. Simple and transparent, but with two known biases: attenuation (the difficulties are compressed) and "who was asked" (the difficulties depend on when a question was asked). Report the Bayesian or EM comparison as a sensitivity check.
- (b) **Bayesian (m1) pool difficulties.** The most principled on paper; slow, and unweighted except through resampling.
- (c) **EM, fit once on the whole pool.** It gives the same difficulties as (b) in seconds, with weights. One set of difficulties for every question set, so scores from different sets differ only in which questions they use. Option (b) then becomes the validation of (c) rather than a production step.
- (d) **EM, fit per set.** Each set's difficulties are the maximum-likelihood estimates for the model actually used to score it. The scale then differs from set to set, so comparisons between sets (e.g., social vs economic) mix different scales.
- (e) **Something else.** For example, a two-parameter model that lets each question have its own discrimination. This would remove the dependence of the scale on the set, but it changes the method.

Sub-questions:
1. **Pool vs per set.** In a Rasch model, a question's difficulty shouldn't depend on which other questions are in the set; that it does here is a sign the questions differ in how strongly they relate to the trait. Is the pool fit the right reference scale for all sets, or should each set be scored with its own fit?
2. **Is it worth leaving the book's method?** The heuristic is easy to explain in a book chapter and a blog post. EM is one more paragraph in the technical appendix. The differences are small, but they are systematic, and they run in one direction.

## Our lean

Option (c): EM fit once on the whole pool. It gives the principled difficulties at no cost, uses the survey weights, and removes the dependence on 11-hour runs (which then serve as validation). It keeps one scale across sets. We'd keep the heuristic as a switch for comparison with the book. Per-set EM (d) would go in as a sensitivity check, because it's the alternative assumption about the scale (each set its own trait). Reviewers' views on sub-question 1 would be especially useful.

## Outcome (2026-11-01)

**Review recommendation (adopted):** option (c), weighted EM with fixed, pooled difficulties, but calibrated on the **43 primary questions** rather than all 66 candidates. The calibration pool should agree with the question-selection decisions; otherwise questions excluded on construct grounds (the confidence questions) still shape the scale. Questions outside the primary pool are put on its scale by anchored calibration: the 43 difficulties are held fixed, and the others are estimated. These are the 7 confidence questions in the sensitivity sets and `natfare` in the book's set. The heuristic stays as a switch for comparison with the book, and EM on all 66 and per-set EM are sensitivity analyses. No two-parameter model unless these comparisons reveal a problem. Pooled calibration is a common calibration convention; it doesn't show that every set measures the same trait. Keep weighted yearly comparisons and lowess-smoothed or model-based estimates clearly distinguished.

**Implemented:** `notebooks/overton_em.py` and `overton_apc.py --difficulties {heuristic, em, em66, emset, m1pool}`. `em` is the production choice.

**Calibration check, weighted yearly means** (`notebooks/overton_calibration.py`, `overton_figs/calibration_check.csv`). For the primary pool, per-set EM is the same as `em`. There is no heuristic p15 file, but p15 is the same set as v2.

| Set | Heuristic | EM, 43 (production) | EM, 66 | EM, per set | Bayesian, 66 |
|---|---|---|---|---|---|
| Overton gap: | | | | | |
| v1 | 1.72 | 1.69 | 1.70 | 1.69 | 1.70 |
| p43 | 1.91 | 1.66 | 1.80 | 1.66 | 1.80 |
| s50 | 2.09 | 1.86 | 1.98 | 1.93 | 1.98 |
| econA | 3.79 | 3.73 | 3.76 | 3.55 | 3.76 |
| econB | 2.63 | 2.50 | 2.60 | 2.56 | 2.61 |
| social | 1.63 | 1.33 | 1.48 | 1.22 | 1.47 |
| Change, conservatives: | | | | | |
| p43 | −0.64 | −0.90 | −0.76 | −0.90 | −0.76 |
| econA | +0.93 | +0.87 | +0.90 | +0.71 | +0.89 |
| social | −1.20 | −1.51 | −1.36 | −1.63 | −1.36 |

- **The choice of calibration pool matters about as much as the choice of estimator.** On the primary pool, calibrating on 43 instead of 66 questions lowers the Overton gap from 1.80 to 1.66; the heuristic to EM on 66 lowers it from 1.91 to 1.80. The 43-question calibration stretches the difficulties 8% more than the 66 (slope 1.08). The 23 extra questions, which are weaker and include the confidence battery, make the common trait less coherent.
- **So the reviewer's empirical test doesn't resolve the concern:** the 43 and 66 calibrations are not nearly identical. As recommended, the 43-question calibration is primary, on the grounds of consistency with the selection rules, and the 66 is reported as a sensitivity.
- **All differences are in the same direction and smaller than the effect of the question set.** No sign changes.

**Model-based check** (period and period–cohort fits; Unity array 66468744; `overton_calibration.py`). Within-cohort slope per decade, the decomposition of total change, and the Overton gap from the period models by alignment, out of 15:

| Set | Difficulties | Within-cohort slope [90%] | Generational replacement | Changing minds | Overton gap | Change, conservatives | Change, liberals |
|---|---|---|---|---|---|---|---|
| p43 (full primary pool) | heuristic | −0.09 [−0.12, −0.07] | −1.93 | −0.39 | 1.59 | −0.92 | −3.07 |
| | **EM, 43 (production)** | **−0.14 [−0.17, −0.12]** | −1.92 | −0.68 | **1.31** | −1.17 | −3.34 |
| | EM, 66 | −0.12 [−0.14, −0.09] | −1.93 | −0.53 | 1.44 | −1.05 | −3.21 |
| s50 (with confidence) | heuristic | −0.01 [−0.04, +0.02] | −1.75 | −0.14 | 1.76 | −0.39 | −2.65 |
| | EM, 43 + anchored | −0.05 [−0.09, −0.02] | −1.75 | −0.39 | 1.55 | −0.59 | −2.86 |
| social | heuristic | −0.14 [−0.17, −0.12] | −2.25 | −0.52 | 1.05 | −1.60 | −3.27 |
| | EM, 43 | −0.19 [−0.22, −0.17] | −2.24 | −0.82 | 0.74 | −1.91 | −3.53 |
| | EM, 66 | −0.17 [−0.20, −0.14] | −2.25 | −0.68 | 0.88 | −1.77 | −3.41 |
| | EM, per set | −0.21 [−0.24, −0.19] | −2.24 | −0.95 | 0.61 | −2.04 | −3.63 |
| econA (core economic) | heuristic | +0.24 [+0.18, +0.29] | −1.36 | +0.84 | 3.78 | +1.11 | −1.89 |
| | EM, 43 | +0.24 [+0.18, +0.29] | −1.36 | +0.80 | 3.73 | +1.07 | −1.93 |
| | EM, 66 | +0.24 [+0.18, +0.29] | −1.36 | +0.82 | 3.76 | +1.10 | −1.91 |
| | EM, per set | +0.22 [+0.17, +0.28] | −1.35 | +0.65 | 3.56 | +0.92 | −2.03 |
| econB (broad economic) | heuristic | +0.07 [+0.02, +0.12] | −1.05 | +0.24 | 2.33 | +0.54 | −2.14 |
| | EM, 43 | +0.02 [−0.02, +0.06] | −1.04 | +0.08 | 2.19 | +0.37 | −2.32 |
| | EM, per set | +0.01 [−0.02, +0.04] | −1.04 | +0.13 | 2.24 | +0.31 | −2.27 |
| econC (inclusive economic) | heuristic | +0.07 [+0.03, +0.11] | −0.93 | +0.29 | 2.15 | +0.62 | −2.38 |
| | EM, 43 | +0.01 [−0.02, +0.05] | −0.93 | +0.13 | 2.00 | +0.41 | −2.55 |
| | EM, per set | +0.01 [−0.02, +0.04] | −0.93 | +0.17 | 2.06 | +0.36 | −2.47 |

- **The signs of the age–period–cohort results are the same under every calibration except for broad and inclusive economic (below); the sizes differ.** With the production difficulties, the full primary pool's within-cohort slope goes from −0.09 to −0.14 per decade and the Overton gap from 1.59 to 1.31. Generational replacement doesn't change (−1.92 vs −1.93); the difference is all in the within-cohort ("changing minds") part.
- **The sensitivity pool (with the confidence questions) is now credibly negative:** −0.05 [−0.09, −0.02], where it was −0.01 [−0.04, +0.02] with the heuristic.
- **Core economic is unchanged:** +0.24 under every calibration except per set (+0.22). People get more economically conservative within cohorts, by every measure.
- **Broad and inclusive economic change qualitatively:** +0.07, credibly positive with the heuristic, becomes +0.01 to +0.02 with intervals that include zero. The positive within-cohort result for economic questions now rests on the core set alone, which the review already recommended as the primary economic result.
- **43 vs 66 calibration, model-based:** slope −0.14 vs −0.12, gap 1.31 vs 1.44 (full pool); −0.19 vs −0.17, 0.74 vs 0.88 (social). It's a real difference but a modest one, and in the same direction as everything else.
- **Diagnostics:** 65 fits (6000 draws, target accept 0.99): at most 39 divergences (`period_s50-em`), R-hat ≤ 1.013, ESS ≥ 566. Full table: `overton_figs/calibration_models.csv`.

## Questions for a second round (2026-11-01)

The design decision above was made on principle, before the model-based results. These questions are about whether it's still the best-justified choice, given what the check showed. They are not about which choice gives which result.

1. **Calibration pool, now that the check is in.** The review's test was whether calibrating on the 43 primary questions and on all 66 candidates give nearly the same results. They don't: the 43-question calibration stretches the difficulties 8% more. Is calibrating on the 43 still the better-justified choice? Or does the difference mean the reference scale is a convention with no clear right answer, and both should be reported with equal standing?
2. **Is the set-dependence of the scale the "substantive problem" that would justify a two-parameter model?** The stretch relative to the heuristic is 1.14 on the 66 candidates, 1.23 on the 43, and 1.3 to 1.9 for individual sets (1.94 for the four core economic questions). In a correctly specified Rasch model, it wouldn't depend on the set. The review recommended against a 2PL unless the checks revealed a problem.
3. **Domain comparisons.** For claims about a domain (e.g., "people become more economically conservative within cohorts"), which is the more honest estimate: the domain's questions scored on the common calibration, or calibrated on their own (each domain its own trait)? On which grounds should the primary version be chosen and the other reported as a sensitivity?
4. **Uncertainty in the difficulties.** Every option except the Bayesian fit treats the difficulties as known. Their posterior SD is about 0.02, but the choice of calibration moves results more than that. Should the reported intervals reflect the uncertainty in the difficulties (e.g., a bootstrap over the calibration), the choice of calibration (reported as a range), or neither?

## Outcome of the second round (2026-11-01)

**Decision (Allen, following the review).**
- **Calibration:** weighted EM on the 43 primary questions stays the primary specification, with anchored calibration for questions outside it. EM on all 66 and per-set EM are sensitivity analyses. The 43 and 66 don't get equal standing.
  - Agreement between the two was never a condition for preferring one. The comparison measures how much the choice matters; the choice itself rests on construct validity.
  - Describe the 43-question calibration as a reference convention chosen on construct-validity grounds, not as "the true difficulties". The one-dimensional Rasch model is an approximation.
- **No two-parameter model.** The set-dependence of the scale justifies a diagnostic look at model misfit, not a new production model. A 2PL addresses unequal discrimination but not multiple dimensions, and it gives up the interpretation of the score as an expected number of conservative answers, with every question contributing equally.
- **Domains: common calibration for every domain score** (the 43-question calibration). Per-set calibration is a sensitivity analysis, reported where it changes magnitude or interpretation. The two answer different questions:
  - common: how domain attitudes changed against one broad conservatism scale, the question aligned with the Overton analysis;
  - per set: how a domain changed as its own trait.

  Don't treat the difference between social and economic slopes as a difference in one quantity; the domains have different questions and possibly different latent structure. Consider also reporting changes in the expected fraction of conservative answers.
- **Economic claims are scoped to the core set.** The within-cohort conservative shift holds for the four core questions on government economic responsibility under every calibration. The broader economic sets show little within-cohort change under the production calibration. So the result is about those four questions, not "economic conservatism" in general.
- **Uncertainty:** intervals are conditional on the calibration. Calibration choice is shown separately, as specification sensitivity, never merged into an interval. Sources are kept distinct:
  - sampling: model intervals;
  - difficulty estimation: an optional bootstrap check;
  - calibration choice: a sensitivity table;
  - question selection: the main robustness analysis.

**Future work (proposed by the review; not yet done).** In the review's order of priority:
1. **Flexible latent distribution.** The N(0, 4) distribution in the EM is the population distribution the item probabilities are integrated over, not just a computational prior. Over 50 years with uneven question coverage, a fixed distribution can push period differences into the difficulties. Check by estimating the latent distribution empirically (as in Bock and Aitkin), or letting it vary by broad survey period with common difficulties. In particular: does the "who was asked" adjustment survive?
2. **Core vs broad economic, by question.** Plot or tabulate the trends of the individual questions in the core and broad economic sets to see which questions account for the flat broad-economic result. This doesn't change the prespecified sets.
3. **Residual associations within domains.** For example, do the four core economic questions stay associated after conditioning on estimated conservatism? Use residual correlations or a posterior predictive check of item pairs. If the associations are substantial, the next step is a multidimensional model, not a 2PL.
4. **Divergences.** Inspect `period_s50-em` (39 divergences) and check whether they affect the reported estimands.
5. **Bootstrap of the calibration (optional).** Refit the EM difficulties within each survey bootstrap replicate, then recompute the Overton gap and within-cohort slopes, to confirm that difficulty uncertainty is negligible downstream.

The review suggests keeping these targeted, without expanding the work into a full psychometric study.


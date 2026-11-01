# Overton Update: Lab Notebook

This is a running log of the analyses for updating Chapter 12 of *Probably Overthinking It*, "Chasing the Overton Window," with General Social Survey (GSS) data through 2024. It is a working document, not a finished report. Entries are dated and in the order the work was done, and later entries sometimes revise earlier ones.

**The method, in brief.** The chapter uses fifteen GSS questions that distinguish liberals and conservatives. Each respondent's answers are coded as conservative or not. A one-parameter item response theory (IRT) model estimates each respondent's latent conservatism:

- Each question's difficulty is fixed at the logit of the share of conservative responses.
- Each respondent gets a grid posterior with a wide Gaussian prior.
- Conservatism is reported as the expected number of conservative responses out of 15.

Respondents who answered fewer than five questions are dropped. The chapter's Overton result is that the average liberal of 1974 gives about as many conservative responses as the average moderate of 2000, and the average conservative now. The opinions that count as liberal, moderate, and conservative move together over time.

**Code.** The scripts are in the `notebooks/` folder of the book's repository:

- `overton_data.py` prepares the data.
- `overton_pipeline.py` is the chapter's pipeline.
- `overton_drift.py` checks for item drift.
- `overton_sensitivity.py` tests sensitivity to the choice of questions.
- `overton_lab_figs.py` makes the figures in this document.

---

## 2026-10-30: Data

**Source.** GSS 1972–2024 cumulative file, Release 3a (July 2026). It comes through an extract from another project with 404 variables and 75,699 respondents. All fifteen questions, `polviews`, `age`, and `cohort` are in the extract, and all fifteen questions were asked in 2024.

**Weights.** As in the book, I make three resamplings of the data. Within each year, rows are drawn with probability proportional to the sampling weight. The book used `wtssall`, filled with `wtssps` from 2021 on, where `wtssall` is not available; that's the same as `wtsmerge` in the new extract.

**Checks.**
- The response codes for all fifteen questions and `polviews` are the same as in the book's file.
- Percent conservative by year and question, averaged over the three resamplings, is identical to the book's file for 1972–2021, because the same random seeds draw the same rows.
- 2022 differs by at most 2.5 percentage points. That's resampling noise; the raw 2022 counts are identical.
- One revision between releases: `racopen` has about twice as many 2022 responses in R3a (1,173 vs 588).

## 2026-10-30: Reproducing the book

`overton_pipeline.py` is the chapter's code as functions, with no changes to the method. On the book's data file (which runs through 2022, although the chapter text says 2021), every key number matches the notebook's saved outputs:

- mean conservatism 6.833;
- Conservative − Liberal difference by question, 11.4 to 28.0 percentage points;
- 7 of 12 birth cohorts with a significant negative slope of conservatism vs age, none positive;
- smoothed conservatism from 7.96 to 5.09;
- the cohort effect is 3.3 times the period effect;
- Moderate in 2000: 6.76; Conservative in 2021: 6.99.

## 2026-10-30: Through 2024

The same pipeline on the new data. The first column is the book's file; the other three are the three resamplings of the new data.

| | Book (to 2022) | New, `gss0` | New, `gss1` | New, `gss2` |
|---|---|---|---|---|
| Smoothed conservatism, start → end | 7.96 → 5.09 | 7.99 → 4.97 | 8.06 → 5.08 | 8.01 → 5.12 |
| Cohorts with negative / positive age slope | 7 / 0 of 12 | 8 / 0 | 7 / 0 | 8 / 0 |
| Cohorts with negative / positive year slope | 8 / 0 of 12 | 8 / 0 | 8 / 0 | 9 / 0 |
| Cohort effect / period effect | 3.3 | 2.9 | 3.2 | 3.4 |
| Liberal 1974 | 6.41 | 6.41 | 6.65 | 6.39 |
| Moderate 2000 | 6.76 | 6.76 | 6.75 | 6.73 |
| Conservative, last year | 6.93 (2022) | 6.94 (2024) | 7.15 | 7.07 |
| Conservative − Liberal, 1974 → last year | 2.47 → 3.95 | 2.48 → 4.16 | 2.29 → 4.35 | 2.69 → 4.20 |

The main results hold. People don't become more conservative as they age, nearly every cohort becomes more liberal over time, the center moves, and the gap between conservatives and liberals keeps growing. The differences between resamplings are not negligible, so the final report should give ranges.

```{figure} overton_figs/new/overton10b.png
:align: left
Conservatism by year, grouped by self-identified alignment, through 2024. The gray line marks the level of the average liberal in 1974.
```

**Yearly means** (average of three resamplings):

| Year | All | Conservative | Moderate | Liberal | Questions answered |
|------|-----|--------------|----------|---------|--------------------|
| 2018 | 5.21 | 6.94 | 5.00 | 3.32 | 9.6 |
| 2021 | 5.40 | 7.79 | 5.27 | 3.13 | 6.3 |
| 2022 | 5.00 | 7.00 | 4.84 | 2.81 | 7.7 |
| 2024 | 5.29 | 7.66 | 4.67 | 2.93 | 7.2 |

Overall conservatism went up in 2024, mostly because self-identified conservatives moved toward conservative responses. Since 2021, respondents have answered fewer of the fifteen questions, so the estimates lean more on the prior.

**Youngest cohorts.** People born in the 1990s are more likely to identify as conservative than they used to be (weighted): 21% in 2021, 26% in 2022, and 30% in 2024. Liberal identification went from 42% to 39% to 29%. The book said the 1980s and 1990s cohorts showed no trend toward conservative identity; for the 1990s cohort, that's no longer clear. The 2000s cohort is too small to say anything (about 200 respondents per year).

## 2026-10-30: Interview mode

The GSS changed how it conducts interviews in 2021:

- **2021:** almost all interviews were on the web.
- **2022:** about half in person and half on the web (1,562 in person, 1,633 web).
- **2024:** also mixed (1,174 in person, 1,762 web).

The interview mode variable (`mode`) is not in the extract, so I read it from the raw R3a file.

**Five of the fifteen questions were not asked on the web.** `grass`, `prayer`, `divlaw`, `racopen`, and `fepol` were asked only in person or by phone in 2022 and 2024, and not at all in 2021. Since 2022, the answers to these questions come only from people who did the interview in person or by phone.

**The 2024 shift is mostly among in-person respondents.** Weighted percent giving the conservative response:

| Question | 2022 in person | 2022 web | 2024 in person | 2024 web |
|---|---|---|---|---|
| `homosex` | 38 | 36 | 50 | 39 |
| `abany` | 47 | 40 | 49 | 37 |
| `natfare` | 30 | 38 | 25 | 36 |
| `cappun` | 62 | 61 | 62 | 63 |

Conservative self-identification moved the other way: in person, 36.9% → 32.2%; on the web, 32.0% → 36.3%. So the kind of people who chose each mode changed between 2022 and 2024.

## 2026-10-30: Checks on weighting and mode

Yearly mean conservatism (All), under four versions of the analysis:

| Year | Resampled | Weighted, no resampling | Through 2018 only | Without the five, all years (out of 10) |
|---|---|---|---|---|
| 2010 | 6.14 | 6.09 | 6.14 | 4.21 |
| 2018 | 5.21 | 5.23 | 5.21 | 3.68 |
| 2021 | 5.40 | 5.40 | | 3.52 |
| 2022 | 5.00 | 5.05 | | 3.50 |
| 2024 | 5.29 | 5.25 | | 3.57 |

- **Weighting.** Using the weights directly gives the same results as resampling, within about 0.1 in every year and group, so the resampling is fine.
- **Through 2018.** If we stop before the mode change, the book's results hold: 6 of 11 cohorts with a negative age slope and none positive; smoothed conservatism from 7.94 to 5.33.
- **Without the five in-person-only questions in every year** (scores out of 10): the decline continues through 2022 and flattens in 2024. In 2024, conservatives move toward conservative responses (4.87 in 2018 → 5.23) while moderates keep falling (3.57 → 3.16) and liberals are flat. That looks more like polarization than a general shift to the right. The within-cohort results hold.
- **Without the five after 2018 only**, recent years come out 0.3–0.4 higher (2022: 5.40 vs 5.00). I first called this misleading, because the question set changes in 2021. On reflection, that's not the problem: the model is designed to handle respondents who answer different subsets. The difference is a symptom of item drift (next entry).

## 2026-10-30: Item drift

The IRT model can compare respondents who answered different subsets of questions because it assumes each question's difficulty is fixed. The questions are a fixed ruler. If that holds, dropping some questions for some respondents makes their estimates less precise, but not biased.

To test it, I estimated each respondent's conservatism three ways, holding the difficulties and the out-of-15 scale fixed: from all fifteen questions, from the ten, and from the five. The sample is respondents who answered at least three of the five and at least five of the ten. If the ruler were fixed, the difference between the estimates would be small and steady.

```{figure} overton_figs/lab/drift_by_year.png
:align: left
For the same respondents, conservatism estimated from ten questions minus the estimate from all fifteen, by year. If question difficulties were stable, this would be near zero throughout.
```

The difference is about −0.1 from the 1970s to the 2000s, then grows: +0.2 in 2010–2016, +0.34 in 2018, +0.51 in 2022, and +0.39 in 2024. On the logit scale, the five questions went from about 0.3 above the ten (more conservative responses) in the 1980s–2000s to 0.5 below in 2022. The five moved toward liberal responses faster than the others. This happened before the mode change; mode just made it matter, because since 2022 web respondents never see these five questions.

**Interpretation.** The question set can change for random reasons, as with the ballot design, where each respondent gets a random subset. Then drift only adds noise. When the question set changes systematically, drift becomes bias. Questions get retired, added, or asked in one mode only. Questions are retired *because* they drifted to consensus, so retirement and drift go together by design. That's the selection problem that motivates Phase 3.

## 2026-10-30: Sensitivity to the choice of questions

**Drift by question.** For each question, I fit the slope of the log odds of a conservative response over time:

```{figure} overton_figs/lab/drift_by_question.png
:align: left
Change in the log odds of a conservative response per decade, by question, 1972–2024. Orange: questions asked only in person or by phone since 2022.
```

Every question except `gunlaw` moved toward liberal responses. The spread is wide, from −0.51 per decade (`homosex`) to +0.01 (`gunlaw`). Three of the five in-person-only questions (`grass`, `racopen`, `fepol`) are among the four fastest; the other two (`divlaw`, `prayer`) are near the middle.

**Leave one out and random subsets.** I reran the pipeline without each question, and with 50 random sets of 10 questions. Scores are rescaled to be out of 15 so the sets are comparable. The method is otherwise unchanged. The "Overton gap" is the 2024 conservative minus the 1974 liberal. The book's claim is that it's near zero.

```{figure} overton_figs/lab/overton_gap.png
:align: left
The 2024 conservative minus the 1974 liberal, in conservative responses out of 15, for the three resamplings, for each question left out, and for 50 random sets of 10 questions. Dotted line: all fifteen questions.
```

| | All 15, three resamplings | Leave one out (15 runs) | Random sets of 10 (50 runs) |
|---|---|---|---|
| Overton gap (Conservative 2024 − Liberal 1974) | 0.50 to 0.68 | 0.31 to 1.04 | −1.27 to 1.54 (median 0.64) |
| Decline in smoothed conservatism | 2.89 to 3.01 | 2.57 to 3.44 | 2.29 to 4.18 |
| Cohorts with negative / positive age slope | 7–8 / 0 | 7–10 / 0 | 1–11 / 0–1 (median 7 / 0) |
| Cohorts with negative / positive year slope | 8–9 / 0 | 7–9 / 0 | 4–10 / 0–1 (median 7 / 0) |
| Conservative − Liberal, 2024 | 4.16 to 4.35 | 4.05 to 4.52 | 2.87 to 4.86 |

**What holds and what doesn't.**
- **The direction of every result holds for every question set.** People don't become more conservative with age (at most one significant positive cohort slope in any run), conservatism declines over time, and conservatives and liberals move apart.
- **The size of the Overton result depends on the questions.** Across random sets of ten, the 2024 conservative ranges from 1.3 responses *less* conservative than the 1974 liberal to 1.5 *more*. That's 3–4 times the spread between resamplings. Sets dominated by fast-drifting questions (`homosex`, `grass`, `racopen`, `fepol`, `premarsx`) give negative gaps; sets of slow ones (`gunlaw`, `natenvir`, `letdie1`, `cappun`) give large positive gaps.
- So "the average conservative now is about where the average liberal was in 1974" depends on which questions are on the list. It's about the issues where opinion moved. On issues where opinion hardly moved, such as gun permits or environmental spending, today's conservatives are clearly more conservative than the liberals of 1974.

## 2026-10-30: Question selection, step 0 (the book's ranking)

Before selecting a new set of questions, I reran the book's `find_questions()` on the book's data. Its call was commented out in the chapter notebook, so the ranking was never saved. For each of 114 candidates (`union_` isn't in the file), it codes the most common response as 1, fits `logit(y ~ year + cohort + polviews)`, and ranks by the absolute value of the `polviews` coefficient. Results: `overton_figs/selection_step0.csv`.

The book's fifteen questions rank between 4th and 47th:

| Question | Rank | Question | Rank | Question | Rank |
|---|---|---|---|---|---|
| `homosex` | 4 | `natfare` | 20 | `letdie1` | 33 |
| `sexeduc` | 5 | `natenvir` | 21 | `fepol` | 37 |
| `abany` | 7 | `prayer` | 26 | `racopen` | 40 |
| `premarsx` | 13 | `divlaw` | 28 | `pornlaw` | 47 |
| `cappun` | 14 | `gunlaw` | 31 | | |
| `grass` | 19 | | | | |

So "the top fifteen" doesn't reproduce the book's set. The higher-ranked questions that were left out fall into groups, each reflecting a judgment that a written rule needs to make explicit:

- **Circular:** presidential vote (`pres04`, `pres08`, `pres12`, ranks 1–3).
- **Religious belief and practice:** `god` (9), `savesoul` (18), `reborn` (22), `postlife` (38), `relactiv` (39), `attend` (41).
- **Behavior rather than opinion:** `hunt` (32), `owngun` (36), `pistol` (43).
- **The same battery:** the other abortion items (`abdefect`, `abnomore`, `abrape`, `absingle`, `abpoor`, `abhlth`, ranks 10–24); sexual morality (`teensex` 6, `xmarsex` 17, while `homosex` and `premarsx` were kept); civil liberties (`colhomo` 30, `spkhomo` 35, `libhomo` 44, `colath` 45, `colcom` 46).
- **Spending items:** `natenrgy` (8), `natheal` (25), `natchld` (27), `nateduc` (34), `natcity` (42). Two spending items were kept (`natfare`, `natenvir`), so the spending battery was treated as several topics rather than one.
- **Shorter-running:** `natenrgy` (7 waves, from 2010), `natchld` (12 waves, from 2000), `affrmact` (23; 15 waves, from 1994), `fejobaff` (29; 13 waves, from 1996). Under the new span rule (at least 8 waves over at least 15 years), `natchld`, `affrmact`, and `fejobaff` would be eligible.

## 2026-10-30: Question selection, step 1 (candidate pool)

**Gates** (`overton_candidates.py`). From the full GSS 1972–2024 cumulative file (R3a, 6,943 variables), read in chunks, a variable is a candidate if:
- it has at least 8 waves with at least 100 respondents who answered it and `polviews`;
- those waves span at least 15 years;
- it has 2 to 7 distinct valid values;
- it isn't an administrative variable.

432 pass. That includes all of the book's fifteen and 107 of its 114 candidates; the 7 that fail are the presidential votes, `attend`, `relexp`, `relactiv`, and `natenrgy`. The passing variables are in a 17 MB extract (`data/gss_overton_candidates_r3a.hdf`). Metadata comes from the CultureWar repo: codebook text, value labels, and subjects from gssrdoc, and response groupings from CultureWar's variable catalog where available. Questions with more than 7 response categories (e.g., 0–10 scales) are excluded by the gate.

**Scope** (`overton_scope.py`, explicit lists; no responses or trends used):

| Scope | Count |
|---|---|
| Opinion (in scope) | 133 |
| Borderline (needs a decision) | 40 |
| Excluded: demographic, household, interview, administrative | 116 |
| Excluded: behavior or experience | 76 |
| Excluded: personal circumstances or well-being | 28 |
| Excluded: religion | 24 |
| Excluded: knowledge, expectation, circular | 15 |

Borderline groups:
- alternate wordings of the spending items (`nat*y`, asked on other ballots);
- social trust (`trust`, `fair`, `helpful`);
- child-rearing values (`obey`, `thnkself`, ...);
- anomie (`anomia5`–`anomia7`);
- group stereotypes (`wlthblks`, `workblks`, `intlblks`, and the White versions);
- perceived likelihood of discrimination (`discaffm`, `discaffw`);
- attitudes toward science;
- `getahead`, `aged`, `tax`, `world1`;
- two follow-up questions asked depending on an earlier answer (`rachaf`, `racmost`).

**Decisions on the borderline items (2026-10-30):** all are in scope except `rachaf` and `racmost`, which depend on an earlier answer. The alternate wordings of the spending items (`nat*y`) are separate candidates. Because each wording is asked on a different ballot, no one answers both, so correlation can't detect redundancy; for redundancy, each pair counts as one topic. That gives **171 opinion questions** in the candidate pool.

Of the original 133 opinion questions, 79 were among the book's candidates. New ones include `courts`, the police-violence items (`polhitok`, `polabuse`, `polmurdr`, `polescap`, `polattak`), `helppoor`/`helpsick`/`helpnot`/`helpblk`, `racdif1`–`racdif4`, `spanking`, `pillok`, the suicide items, `wirtap`, `busing`, `racmar`, `racpush`, and `racseg`.

## 2026-10-30: Question selection, steps 2–3 (coding, ranking, topics)

**Coding** (`overton_coding.py`). For each response, I computed the difference between the percentage of conservatives and of liberals who choose it, within each year, then averaged over years (weighted). Each ordered scale is split where it best separates the two groups (contiguous cut); the conservative side is the side conservatives choose more. Volunteered answers such as "other", "depends", and "neither" count as not conservative, as in the book.
- `divlaw` and `courts` use a hand-specified order, because their codes aren't in scale order.
- The rule matches the book's coding for 13 of the 15. For `homosex` and `premarsx` it leaves out "wrong only sometimes"; these are near ties (separation 31.6 vs 28.4 points, and 27.1 vs 26.4).
- Decision: keep the book's codings for its fifteen, so that set v1 and set v2 differ only in which questions are included.
- It differs from CultureWar's groupings on 18 of 93 questions, always in where the scale is cut, never in direction.
- A per-response version of the rule (each response conservative if conservatives choose it more) gives non-adjacent groupings for 16 questions, so I didn't use it.

**Ranking** (`overton_ranking.py`). The book's model with the coded split as the outcome: weighted logit of the conservative response on `year + cohort + polviews`, ranked by the `polviews` coefficient. The top of the ranking is the `help*` battery and the spending items in their alternate wordings: `homosex` 0.455, `helppoor` 0.437, `helpnot` 0.421, `sexeduc` 0.393, `wrkwayup` 0.380, `natenviy` 0.380, ...

**Redundancy by correlation doesn't work.** Tetrachoric correlations among respondents who answered both questions:

| Pair | r |
|---|---|
| `abany` with `abnomore`, `absingle`, `abpoor` | 0.92–0.95 |
| `abany` with `abrape`, `abdefect` | 0.76–0.78 |
| `abany` with `abhlth` | 0.70 |
| `homosex` / `premarsx` (separate topics in the book) | 0.72 |
| `homosex` / `colhomo` | 0.71 |
| `eqwlth` / `goveqinc` (near-duplicates) | 0.58 |
| `helppoor` / `helpnot` | 0.64 |
| `natfare` / `natenvir` | 0.21 |

No threshold reproduces the book's judgments, so topics are assigned by hand (`overton_topics.py`; draft).

**Coverage.** Decision: a question must have been asked in at least half the waves (17 of the 33 years with `polviews`). For the spending items, the original wording (from 1973) represents the topic rather than the 1984 alternate.

**Draft set v2** (top 15, one per topic): `homosex`, `helppoor`, `helpnot`, `sexeduc`, `helpsick`, `eqwlth`, `abany`, `pillok`, `helpblk`, `natarms`, `racdif1`, `premarsx`, `grass`, `natenvir`, `cappun`. Seven are from the book's set. Decision (2026-10-30): keep the `help*` questions as separate topics, as drafted, and keep the other groupings. This is set v2, frozen in `overton_question_sets.py`. Metadata for both sets is in `overton_questions.yaml`.

## 2026-10-30: Set v1 vs set v2

Same method and same respondents (three weighted resamplings of the candidate extract); only the questions differ. For v1 the results are identical to the earlier run through 2024. `overton_compare_sets.py`; figures for v2 in `overton_figs/v2/`.

| | v1 (book's 15) | v2 (new 15) |
|---|---|---|
| Smoothed conservatism, start → end | 7.99–8.06 → 4.97–5.12 | 8.52–8.66 → 5.84–5.88 |
| Cohorts with negative / positive age slope (of 12) | 7–8 / 0 | 7–8 / 0 |
| Cohorts with negative / positive year slope (of 12) | 8–9 / 0 | 8 / 0 |
| Cohort effect / period effect | 2.9–3.4 | 4.2–5.4 |
| Change, Liberal / Moderate / Conservative | −3.6 to −3.8 / −3.5 to −3.7 / −1.8 to −1.96 | −3.8 / −3.1 to −3.3 / −0.8 to −1.0 |
| Liberal 1974 | 6.38–6.65 | 6.75–6.81 |
| Moderate 2000 | 6.73–6.76 | 7.43–7.47 |
| Conservative 2024 | 6.94–7.15 | 8.61–8.66 |
| Moderate 2000 − Liberal 1974 | 0.09–0.35 | 0.64–0.71 |
| Conservative 2024 − Liberal 1974 (Overton gap) | 0.50–0.68 | 1.84–1.87 |
| Con−Lib gap, 1974 → 2024 | 2.3–2.7 → 4.2–4.35 | 2.7–2.9 → 5.6 |

```{figure} overton_figs/v2/overton10b.png
:align: left
Set v2: conservatism by year and alignment. The gray line marks the level of the average liberal in 1974.
```

**What holds.** People don't become more conservative as they age, every cohort with a significant slope gets more liberal over time, the overall decline is about the same size (2.7–2.8 vs 3.0), and liberals and moderates move about as much as in v1.

**What changes.** With v2, self-identified conservatives barely move: about −0.9 over 50 years, vs −1.9 with v1. The gap between conservatives and liberals doubles (2.8 → 5.6). The 1974 liberal now corresponds to the moderate of about 2010, not 2000, and today's conservative is clearly more conservative than the 1974 liberal (gap 1.85, outside the range from random subsets of the book's questions, −1.3 to 1.5). The new questions in v2 are mostly about the role of government (`helppoor`, `helpnot`, `helpsick`, `eqwlth`, `helpblk`) and defense spending, where conservatives' views have changed little. The book's set was weighted toward social issues, where views moved more.

**So far:** the qualitative Overton pattern (the center moves, so the same views get relabeled) holds with v2, especially for liberals and moderates. The book's strongest version ("today's conservative is about where the 1974 liberal was") depends on a bundle weighted toward social issues.

## 2026-10-30: Completing the square (question set × data cutoff)

The v1 vs v2 comparison changed two things relative to the book: newer data and different questions. To separate them, I ran both sets on the same resamplings cut at 2018, the book's last year. Difficulties are estimated from whatever data is used, so the 2018 runs don't use the 2021–2024 responses at all. `overton_compare_square.py`; figures for v2 through 2018 in `overton_figs/v2_2018/`.

Means over the three resamplings; slope counts for each resampling.

| | v1, 2018 | v1, 2024 | v2, 2018 | v2, 2024 |
|---|---|---|---|---|
| Smoothed conservatism, start → end | 7.96 → 5.38 | 8.02 → 5.06 | 8.63 → 6.45 | 8.58 → 5.86 |
| Cohorts with negative / positive age slope | 6–7 / 0 of 11 | 7–8 / 0 of 12 | 3–4 / 0–1 of 11 | 7–8 / 0 of 12 |
| Cohorts with negative / positive year slope | 8–9 / 0 of 11 | 8–9 / 0 of 12 | 4 / 0 of 11 | 8 / 0 of 12 |
| Change, Liberal / Moderate / Conservative | −2.9 / −3.1 / −1.9 | −3.7 / −3.6 / −1.9 | −2.6 / −2.4 / −0.8 | −3.8 / −3.2 / −0.9 |
| Liberal 1974 | 6.47 | 6.48 | 6.74 | 6.79 |
| Moderate 2000 | 6.76 | 6.74 | 7.46 | 7.45 |
| Conservative 2018 − Liberal 1974 | 0.67 | 0.83 | 2.00 | 2.04 |
| Conservative 2024 − Liberal 1974 | | 0.57 | | 1.86 |
| Con−Lib gap, 1974 → 2018 | 2.6 → 3.5 | 2.5 → 3.8 | 2.8 → 4.6 | 2.8 → 5.0 |

Changes run from the first to the last smoothed year, so they span 44 years in the 2018 columns and 50 in the 2024 columns.

```{figure} overton_figs/v2_2018/overton10b.png
:align: left
Set v2, data through 2018: conservatism by year and alignment. The gray line marks the level of the average liberal in 1974.
```

**The question set accounts for the change in the Overton gap; the newer data doesn't.** Through 2018, the 2018 conservative is 0.67 responses more conservative than the 1974 liberal with v1, and 2.00 with v2. Adding 2021–2024 barely moves either number (0.83, 2.04 for the same comparison). Moderate 2000 and Liberal 1974 are also essentially unchanged by the cutoff. So the weaker Overton result in v2 comes from the questions, not from the last three waves.

**The newer data matters for the within-cohort trends with v2.** With v1, cohorts get more liberal over time through 2018, and the 2021–2024 data only adds to that. With v2 through 2018, the cohorts born 1930–1970 are flat (per-decade year slopes between −0.06 and +0.04, mostly not significant; the 1940s cohort has a small significant *positive* age slope in two of three resamplings). Only the oldest (1910s, 1920s) and youngest (1980s, 1990s) cohorts decline. With data through 2024, those middle cohorts decline too (−0.03 to −0.18 per decade, mostly significant). Per-decade slopes for resampling 0:

| Cohort | v1 2018 | v1 2024 | v2 2018 | v2 2024 |
|---|---|---|---|---|
| 1930 | −0.17* | −0.19* | −0.02 | −0.07* |
| 1940 | −0.10* | −0.15* | +0.04 | −0.03 |
| 1950 | −0.09* | −0.18* | +0.02 | −0.09* |
| 1960 | −0.24* | −0.26* | −0.06 | −0.11* |
| 1970 | −0.22* | −0.23* | −0.01 | −0.18* |

(Year slopes; * means p < 0.02, the threshold in `count_signs`.)

**Is the 2018–2024 drop in v2 an interview-mode artifact?** Probably not entirely. Mean v2 conservatism in 2018 (all in person) was 6.31. In 2022 it was 5.86 among in-person and phone respondents and 6.04 among web respondents, so the drop shows up within the in-person mode, and web respondents are, if anything, more conservative. The in-person samples after 2020 are smaller and may differ in composition, so this doesn't settle it.

**So far:**
- The book's claim that today's conservative is about where the 1974 liberal was depends on the question set (v1 vs v2), not on the data cutoff.
- With v2, the claim that people get more liberal as they age, within cohorts, rests mostly on the last three waves for the middle cohorts. Through 2018, v2 says they stayed about the same. "People don't get more conservative as they age" holds in all four cells.
- With the role-of-government questions in v2, the attitudes of the middle cohorts barely moved from 1972 to 2018, and the change in the population came mostly from cohort replacement.

## 2026-10-30: Larger question sets, same rule

Same selection rule as set v2 (at least 17 waves, original wording for spending items, one per topic, ranked by the `polviews` coefficient), taking the top N for N = 15, 20, …, 40. The sets are nested, and the top 15 is set v2. Scores are rescaled to out of 15. `overton_larger_sets.py`; questions in rank order in `overton_figs/larger_sets_questions.csv`.

Questions 16–40, in order: `xmarsex`, `spanking`, `fefam`, `discaff`, `conpress`, `prayer`, `gunlaw`, `pornlaw`, `divlaw`, `suicide4`, `letdie1`, `popular`, `conlabor`, `colhomo`, `courts`, `fepol`, `nateduc`, `fepresch`, `conclerg`, `natmass`, `racopen`, `colcom`, `colath`, `coneduc`, `workblks`. All of the book's 15 are in the top 36.

**One question breaks the pattern.** `conpress` (rank 20), coded conservative for "hardly any" confidence in the press, went from 15% to 57%, by far the largest drift *toward* the conservative response among the 40 (+0.37 log odds per decade; the next largest is `coneduc`, +0.18). Adding it at N = 20 makes self-identified conservatives flat (change −0.1) and produces significant *positive* year slopes in about 4 of 12 cohorts. Falling confidence in the press is a trend across the whole population. On this coding it counts as a conservative shift, but it may say more about institutions than about views. Whether to keep it is a judgment about scope, so I didn't change the rule. As a sensitivity run, I repeated the procedure with the confidence-in-institutions items (`con*`) left out of the pool.

Means over the three resamplings, data through 2024:

| N | Overton gap, all | Overton gap, no `con*` | Conservative change, all | Conservative change, no `con*` | Cohorts with positive year slope, all / no `con*` |
|---|---|---|---|---|---|
| 15 | 1.86 | 1.86 | −0.9 | −0.9 | 0 / 0 |
| 20 | 2.49 | 1.91 | −0.1 | −1.0 | 4 / 0 |
| 25 | 2.46 | 2.01 | −0.1 | −0.6 | 4.3 / 1 |
| 30 | 2.08 | 1.47 | −0.4 | −1.1 | 3 / 0 |
| 35 | 1.69 | 1.21 | −0.6 | −1.5 | 0 / 0 |
| 40 | 1.45 | 0.89 | −0.9 | −1.5 | 0 / 0 |

Liberals change by −2.8 to −3.8 and moderates by −1.9 to −3.2 in every row. Through 2018 the pattern is similar (`overton_figs/larger_sets.csv`).

```{figure} overton_figs/lab/gap_by_size.png
:align: left
The 2024 conservative minus the 1974 liberal, out of 15, for the top N questions by the selection rule, up to all 65 candidates (52 without the confidence items). Dots are resamplings. For comparison, the book's fifteen give 0.5–0.7. (Figure updated with the next entry.)
```

**What this shows:**
- **Liberals and moderates move in every set.** The decline among liberals and moderates, and the absence of any increase with age within cohorts in most sets, hold throughout.
- **The size of the Overton gap depends on N, with no plateau by 40** (but see the next entry: it levels off after 40). Questions ranked lower are more often social and civil-liberties items that drifted a lot (`colhomo`, `fepol`, `fepresch`, `racopen`, `workblks`). As they come in, conservatives move more and the gap shrinks: from about 1.9 at N = 15 to 0.9 at N = 40 without the confidence items, closer to the book's 0.5–0.7.
- **The highest-ranked questions, the ones that best separate conservatives from liberals, are the ones conservatives changed least on.** Questions ranked lower separate the groups less and drifted more. So "today's conservative vs the 1974 liberal" depends on how far down the ranking we go, and there's no obvious place to stop.
- **The within-cohort result is less robust than it looked.** With `conpress` in the set, several cohorts get significantly more conservative over time. Without the confidence items, only N = 25 has any positive slopes (1 cohort on average).

**Decisions (2026-10-30):** report both pools, with and without the confidence items, and decide later which one the article uses. Report the Overton gap as a function of N rather than as a single number.

## 2026-10-30: Larger sets, extended to every candidate

Only 65 candidates pass the rules (52 without the confidence items), so I extended the curve to the whole list. Same script and figure as the previous entry.

Means over the three resamplings, data through 2024:

| N | Overton gap, all | Overton gap, no `con*` | Conservative change, all | Conservative change, no `con*` | Cohorts with positive year slope, all / no `con*` |
|---|---|---|---|---|---|
| 40 | 1.45 | 0.89 | −0.9 | −1.5 | 0 / 0 |
| 45 | 1.31 | 0.90 | −0.8 | −1.3 | 0 / 0 |
| 50 | 1.43 | 0.96 | −0.5 | −1.0 | 1 / 0 |
| 52 (all, no `con*`) | | 0.94 | | −1.0 | / 0 |
| 55 | 1.25 | | −0.6 | | 1 / |
| 60 | 1.39 | | −0.3 | | 2.3 / |
| 65 (all) | 1.42 | | −0.1 | | 3.7 / |

Questions 41–65 by rank: `nataid`, `conbus`, `conarmy`, `natpark`, `polescap`, `consci`, `tax`, `contv`, `getahead`, `racmar`, `confinan`, `racpres`, `colmil`, `natdrug`, `natsoc`, `librac`, `confed`, `trust`, `conlegis`, `aged`, `natspac`, `conjudge`, `natcrime`, `conmedic`, `natroad`. Below about rank 50 the `polviews` coefficients are small (0.02–0.08, and −0.01 for `natroad`), so these questions barely separate conservatives from liberals. With equal weights in the count, they add mostly noise.

**The gap levels off after N = 40.** Without the confidence items, it's 0.89–0.96 from N = 40 to the full 52. With them, it's 1.25–1.45 from 40 to 65. So from 40 on the answer depends on the pool more than on N: about 0.9 without the confidence items and about 1.4 with them. Both are above the book's 0.5–0.7, and below set v2's 1.9.

**Conservatives' change depends on the confidence items.** Without them, conservatives move −1.0 to −1.5 from N = 30 on, and no cohort gets significantly more conservative over time. With them, more confidence items come in past 40 (`conbus`, `conarmy`, `consci`, `contv`, `confinan`, …). Confidence in most institutions fell over the period, so with the full 65, conservatives' change goes to about zero, and about 4 of 12 cohorts get significantly more conservative over time.

**Summary of the question-set results so far (data through 2024):**

| Set | Conservative 2024 − Liberal 1974 |
|---|---|
| Book's 15 | 0.5–0.7 |
| Random 10 of the book's 15 | −1.3 to 1.5 |
| Set v2 (top 15 by the rule) | 1.9 |
| Top 40 to all, without confidence items | 0.9–1.0 |
| Top 40 to all, with confidence items | 1.25–1.45 |

## 2026-10-31: Phase 4, Ricardo's IRT models (estimated difficulties, then pooling)

Ricardo Vieira's two 1PL models from the bake-off, fit with PyMC and nutpie on Unity (CultureWar env), on all respondents who answered at least one question (about 72,000; the talk used a subsample), for sets v1 and v2 and all three resamplings:
- **m1:** `d_j ~ N(0, 1)`, `e_i ~ N(0, 4)`. The book's model, with difficulties estimated instead of set to −logit(prevalence).
- **m2:** `e_i ~ N(0, e_std)`, `e_std ~ HalfNormal(4)`; pools respondents toward a common mean. Written non-centered (`e = e_std * z`), the same model.

Each fit took 1.5–2.5 hours (4 chains, 2000 tune, 1000 draws, 4 CPUs, about 10 GB). No divergences in any fit. m2: worst R-hat 1.004 for the item parameters, ESS ≥ 1,083. m1: item R-hat up to 1.035 and ESS 171–339, because in m1 the difficulties and the respondent parameters can shift together and only the priors pin down the location; the posterior means are still precise. Respondent parameters: R-hat ≤ 1.013, ESS ≥ 2,988. Code: `overton_irt.py` (prepare locally, fit on Unity), `unity/` (sync and Slurm scripts), `overton_irt_compare.py` (comparison). Results in `overton_figs/irt_compare.csv`, `irt_difficulties.csv`, `irt_diagnostics.csv`.

Each variant changes one thing from the row above it. Means over the three resamplings; respondents who answered fewer than 5 questions are dropped except in the last row.

| Set | Variant | Overton gap | Con−Lib 1974 | Decline | Change, Con / Lib | Gap ÷ Con−Lib 1974 |
|---|---|---|---|---|---|---|
| v1 | grid, heuristic difficulties (book) | 0.57 | 2.49 | 2.96 | −1.89 / −3.67 | 0.23 |
| v1 | grid, m1 difficulties | 0.51 | 2.49 | 2.96 | −1.94 / −3.67 | 0.21 |
| v1 | m1 | 0.52 | 2.50 | 2.99 | −1.94 / −3.73 | 0.21 |
| v1 | m2 (pooled) | 0.40 | 1.64 | 1.76 | −1.22 / −2.16 | 0.24 |
| v1 | m2, all respondents | 0.41 | 1.64 | 1.76 | −1.20 / −2.12 | 0.25 |
| v2 | grid, heuristic difficulties | 1.86 | 2.79 | 2.72 | −0.92 / −3.76 | 0.66 |
| v2 | grid, m1 difficulties | 1.94 | 2.76 | 2.55 | −0.80 / −3.62 | 0.70 |
| v2 | m1 | 1.94 | 2.79 | 2.59 | −0.83 / −3.68 | 0.69 |
| v2 | m2 (pooled) | 1.36 | 1.67 | 1.59 | −0.31 / −2.46 | 0.82 |
| v2 | m2, all respondents | 1.34 | 1.65 | 1.58 | −0.31 / −2.48 | 0.81 |

**Step A: estimated difficulties change almost nothing.** The m1 difficulties are spread out more than the heuristic ones (about 1.4 times, e.g. `sexeduc` 1.95 → 2.77), as expected: −logit(prevalence) is attenuated because respondents vary.

Why: prevalence is the logistic curve averaged over respondents, p_j = mean over i of expit(e_i − d_j), and averaging over a spread of e flattens the curve. For e ~ N(μ, σ), p_j ≈ expit((μ − d_j)/k) with k = √(1 + πσ²/8), so the heuristic difficulties are the model's divided by k. m2 estimates σ, and the prediction matches the fitted slope of m2 on the heuristic difficulties: k = 1.15–1.16 vs slope 1.16 for v1, and 1.19 vs 1.18–1.19 for v2. m1 doesn't estimate σ. Under its wide N(0, 4) prior, respondents with all-conservative or all-liberal answers get extreme values of e (SD of e about 2.0–2.2), so its difficulties are stretched more (slope 1.42–1.46). But the counts barely move, because the count is computed on the same scale as the difficulties. The grid with m1's difficulties and m1 itself agree to within 0.05 on every number, so carrying the uncertainty in the difficulties doesn't matter either. Across all numbers, the book's heuristic difficulties hold up.

**Step B: pooling compresses everything, unevenly.** Posterior `e_std` is 0.92–0.93 for v1 and 1.03–1.04 for v2, close to Ricardo's 0.9. That's much narrower than the book's N(0, 4) prior, so m2 pulls each respondent toward the overall mean, more for people who answered fewer questions. For v2, the spread of the counts relative to m1 is 0.39 for respondents with 1–4 answers and 0.75 for 13–15. Group means of the posterior means shrink with them: every difference (decline, Con−Lib gap, changes) drops by about 40%.

The shrinkage depends on how many questions people answered, and that changes over time. With v2, respondents in 1974–1978 answered about 6 questions (the `help*` items were on fewer ballots then) and later respondents about 9–10. So early respondents are pulled toward the mean more, which shrinks the early gap between conservatives and liberals and makes conservatives look like they changed less (−0.31 vs −0.83 with m1). For v1, the ratio of the Overton gap to the 1974 gap barely changes (0.21 → 0.24); for v2 it rises (0.69 → 0.82). For v1, the respondents who answered fewer questions are in 2021–2024 (about 7 vs 9–10).

**What this means.** Pooling helps estimate *individuals* (Ricardo's respondent 309). But averaging the pooled posterior means within groups (year × alignment) underestimates the differences between groups, because the prior pulls everyone toward one common mean, not toward their group's mean. And the bias varies over time with the number of questions answered. So m2 as specified isn't a better estimator of the trends the chapter reports. The book's wide prior shrinks much less, so its group averages are close to unbiased, at the cost of noisier individual estimates. Getting pooling without this bias means pooling toward group-specific means (by year and alignment, or a smooth function of year), which puts the structure of the analysis into the model. That's the "trend inside the model" alternative already listed as a future option in the plan, so it's a method decision to discuss.

**So far:**
- The book's heuristic difficulties are fine: estimating them changes the results by less than the spread between resamplings.
- Pooling toward a single mean distorts the group trends and should not replace the grid as is.
- The direction of every result holds in every variant: decline overall, all cohorts with significant slopes declining (except in "m2, all respondents", where the extra respondents with 1–4 answers produce 1–2 positive slopes), conservatives changing less than liberals.

## 2026-10-31: Phase 4b, period, period–cohort, and age models

Each respondent's estimated conservatism (the grid's posterior mean count, out of 15) is the outcome. Models ported from CultureWar (commit `0f54767`): the period model (`time_model`), the period–cohort model (`model10`), and the age model (`cohort_model` with age as the index, within a decade of surveys), all with the continuous-outcome ("mean") likelihood. Weighted with `wtsmerge` on the original extract (no resampling); difficulties from weighted prevalence; respondents who answered at least 5 questions (67,768 for v1, 66,375 for v2). Fit on Unity: 42 fits, period national and by `polviews3` and `polviews`, period–cohort national and by `polviews3`, age by decade, for v1 and v2. Code: `overton_apc.py`, `unity/overton_apc_*`, `overton_apc_report.py` (table, `overton_figs/apc_summary.csv`), `overton_apc_figures.py` (figures).

**Sampling.** At CultureWar's default target accept of 0.9, the period and age fits had many divergences (up to 15% of draws, R-hat up to 1.38 for `polviews` 5–7), the `sigma_t` funnel CultureWar documents. Rerun at 0.99 with 4000 tuning steps (CultureWar's production setting is 0.95–0.99): at most 29 divergences in 8000 draws, R-hat ≤ 1.007, except the national v2 period fit (R-hat 1.03, ESS 160, no divergences), rerun with 6000 draws: R-hat 1.006, estimates unchanged. Period–cohort fits at their default 0.95: ≤ 78 divergences in 8000 (< 1%), R-hat ≤ 1.010, ESS ≥ 415. All 42 pass.

**Period model** (slope of the latent mean, conservative responses per decade, 90% interval):

| | v1 | v2 |
|---|---|---|
| All | −0.59 [−0.63, −0.56] | −0.49 [−0.53, −0.44] |
| Liberal | −0.77 | −0.77 |
| Moderate | −0.69 | −0.59 |
| Conservative | −0.33 | −0.12 |
| Extremely liberal … extremely conservative (7 points) | −0.63, −0.78, −0.75, −0.69, −0.52, −0.29, −0.15 | −0.63, −0.78, −0.72, −0.59, −0.32, −0.04, +0.04 |

The more conservative the label, the less the people using it have changed; with v2, people who call themselves conservative or extremely conservative haven't changed at all.

```{figure} overton_figs/apc/time_series_v1.png
:align: left
Period model by alignment, set v1. Bands are 90% intervals. The dotted line is the 1974 liberal.
```

**Period–cohort model: the Overton test.** The period component is the change *within* cohorts over time (aging plus the effects of the times, which this model can't separate), standardized to a fixed cohort mix. The decomposition splits the total change from the first to the last survey into generational replacement and changing minds (CultureWar's method: the average of the two orderings).

| | Within-cohort change per decade [90%] | Total change | Generational replacement | Changing minds |
|---|---|---|---|---|
| v1, all | **−0.13 [−0.20, −0.07]** | −2.84 | −2.17 | −0.67 |
| v1, liberal | −0.43 [−0.50, −0.36] | −3.35 | −1.20 | −2.15 |
| v1, moderate | −0.31 [−0.37, −0.24] | −3.38 | −1.86 | −1.52 |
| v1, conservative | +0.10 [+0.01, +0.19] | −1.33 | −1.82 | +0.50 |
| v2, all | **−0.05 [−0.14, +0.04]** | −2.25 | −2.01 | −0.24 |
| v2, liberal | −0.49 [−0.60, −0.39] | −3.57 | −1.10 | −2.46 |
| v2, moderate | −0.30 [−0.39, −0.21] | −3.24 | −1.75 | −1.49 |
| v2, conservative | +0.27 [+0.19, +0.35] | −0.36 | −1.71 | +1.35 |

```{figure} overton_figs/apc/period_component_v1.png
:align: left
Period component (change within cohorts), set v1.
```

```{figure} overton_figs/apc/cohort_component_v1.png
:align: left
Cohort component, set v1.
```

**Age model** (slope of conservatism against age within each decade of surveys, per decade of age):

| | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|
| v1 | +0.58 | +0.51 | +0.54 | +0.45 | +0.46 | +0.42 |
| v2 | +0.50 | +0.34 | +0.43 | +0.38 | +0.46 | +0.53 |

All intervals exclude zero.

```{figure} overton_figs/apc/age_series_v1.png
:align: left
Age model by decade of surveys, set v1.
```

**What this shows:**
- **The paradox, in two numbers.** In every decade, older people are more conservative: +0.4 to +0.6 conservative responses per decade of age. But within cohorts, the change over time is negative (v1: −0.13 per decade) or about zero (v2: −0.05, interval includes zero). If people got more conservative as they aged, the within-cohort change would be positive. So the age gradient in every cross-section comes from differences between cohorts, not from aging.
- **For the single-number answer** to "do people get more conservative as they get older": no. With the book's questions, within-cohort change is −0.13 per decade, with an interval that excludes zero; with v2, it's indistinguishable from zero. In both, it's far from the +0.4 to +0.6 per decade that the cross-sectional age gradient would suggest.
- **Most of the overall change is generational replacement:** 76% for v1 (−2.17 of −2.84), 89% for v2 (−2.01 of −2.25).
- **By alignment, conservatives get *more* conservative within cohorts** (v1 +0.10, v2 +0.27 per decade), while liberals and moderates get more liberal. These are people who currently use each label, so this mixes changing minds with people re-sorting into labels; the national within-cohort trend doesn't have that problem.
- **Compared with the book's lowess curves** (Phase 2, resampled data): the period model gives the same picture: overall decline of about 2.8 for v1 (book's method on the same data: 3.0), the moderate line crossing the 1974 liberal level around 2010, and the conservative line ending above it.

**Figures.** CultureWar style, made with CultureWar's own helpers (header, registration marks, export), imported in the CultureWar env. Colors for liberal, moderate, and conservative are Chapter 12's. All figures use the full 0–15 scale so they can be compared. In `overton_figs/apc/`, for v1 and v2: `time_series`, `time_series7` (by the 7-point label), `period_component`, `cohort_component`, `age_series`.

## 2026-10-31: Conservatism by age within birth groups (Chapter 12, `overton5`)

The Chapter 12 figure "Conservative responses vs age, grouped by decade of birth", redone with the age model fit separately for each birth group (set v1, weighted, no resampling), in two versions: by decade of birth (1900s–2000s, 11 groups) and by generation, with Pew's boundaries (Greatest before 1928, Silent 1928–45, Boomers 1946–64, Gen X 1965–80, Millennials 1981–96, Gen Z 1997 on). Separate fits let each group have its own shape, as the book's lowess curves did; the period–cohort model would force all cohorts to share one trend over time. Within a group, age and survey year move together, so each line is the group's trajectory over time, indexed by age. 17 fits on Unity (`unity/overton_apc_cohort_jobs.txt`), target accept 0.99: at most 4 divergences, R-hat ≤ 1.016, ESS ≥ 476.

Slope of conservatism against age (per decade of age, 90% interval):

| Group | Ages observed | Respondents | Slope |
|---|---|---|---|
| 1900s | 64–89 | 1,848 | +0.08 [−0.09, +0.25] |
| 1910s | 54–89 | 3,772 | −0.06 [−0.15, +0.04] |
| 1920s | 44–89 | 5,490 | −0.02 [−0.10, +0.05] |
| 1930s | 34–89 | 6,562 | −0.16 [−0.22, −0.10] |
| 1940s | 24–84 | 10,496 | −0.09 [−0.14, −0.03] |
| 1950s | 18–74 | 13,938 | −0.16 [−0.21, −0.11] |
| 1960s | 18–64 | 10,636 | −0.24 [−0.29, −0.18] |
| 1970s | 18–54 | 6,771 | −0.23 [−0.33, −0.14] |
| 1980s | 18–44 | 4,850 | −0.45 [−0.61, −0.29] |
| 1990s | 18–34 | 2,422 | +0.00 [−0.49, +0.51] |
| 2000s | 18–24 | 449 | −2.1 [−4.1, −0.1] |
| Greatest | 46–89 | 10,506 | +0.12 [+0.06, +0.17] |
| Silent | 28–89 | 13,199 | −0.08 [−0.13, −0.04] |
| Boomers | 18–78 | 24,942 | −0.10 [−0.14, −0.06] |
| Gen X | 18–59 | 11,994 | −0.24 [−0.30, −0.17] |
| Millennials | 18–43 | 6,208 | −0.23 [−0.39, −0.07] |
| Gen Z | 18–27 | 919 | +0.22 [−0.97, +1.45] |

```{figure} overton_figs/apc/age_by_decade_v1.png
:align: left
Age model by decade of birth, set v1.
```

```{figure} overton_figs/apc/age_by_generation_v1.png
:align: left
Age model by generation, set v1.
```

**What this shows:**
- **Every decade from the 1930s to the 1980s got more liberal as it aged**, with intervals that exclude zero; the 1900s–1920s are flat; the 1990s, 2000s, and Gen Z are seen over too few years to say. Same for the generations from Silent to Millennials.
- **The one positive slope, the Greatest generation (+0.12), is probably a composition effect.** That group spans 45 birth years (1883–1927). At age 46 it's only people born in 1926–27 (surveyed in 1972–73); at 89 it's mostly earlier births, which were more conservative. That alone would make the line rise with age even if no one changed. The decades inside it (1900s, 1910s, 1920s) are flat. The book's statement "None became more conservative" holds for decades of birth; for generations it needs this caveat, or the Greatest generation could be split or dropped.
- **Wider groups mix more birth years**, so generations (16–19 years, 45 for Greatest) are more exposed to this than decades. The decade version is the cleaner test; the generation version is the more readable one.

**By survey year (added 2026-10-31).** The same groups fit separately with the period model and plotted against survey year, with the overall average (the national period model) as a dotted line. Within a group, age and year move together, but a group spans several birth years, so these are separate fits, not shifted copies of the age fits. 17 fits: at most 8 divergences, R-hat ≤ 1.010. Figures: `year_by_decade_v1`, `year_by_generation_v1`. Plotted by age, a common aging pattern would show up as lines with the same shape at the same ages; by year, a common period effect would show up as the lines moving together in the same years.

| Group | Slope by age (per decade of age) | Slope by year (per decade) |
|---|---|---|
| Greatest | +0.12 | −0.21 |
| 1900s | +0.08 | −0.08 |
| 1910s | −0.06 | −0.15 |
| 1920s | −0.02 | −0.16 |
| 1930s | −0.16 | −0.21 |
| 1940s | −0.09 | −0.19 |
| 1950s | −0.16 | −0.15 |
| 1960s | −0.24 | −0.23 |
| 1970s | −0.23 | −0.24 |
| 1980s | −0.45 | −0.71 |
| 1990s | +0.00 | −1.06 |
| 2000s | −2.1 | −0.2 (interval −4.0 to +3.4) |
| Silent | −0.08 | −0.26 |
| Boomers | −0.10 | −0.14 |
| Gen X | −0.24 | −0.34 |
| Millennials | −0.23 | −0.89 |
| Gen Z | +0.22 | −0.55 (interval −2.6 to +1.5) |

**Caveat: the membership of each group changes along its line, in opposite directions on the two axes.** By age, the old end of each line is only the group's earliest births (e.g. the 1990s at 34 are only people born in 1990, surveyed in 2024), and earlier births are more conservative, so age slopes are biased up. By year, young groups gain their later, more liberal births as they turn 18, and old groups lose their earliest births as they age out, so year slopes are biased down. The truth is between; the two agree where the bias is small (1950s–1970s) and differ most for wide groups (Greatest, 45 birth years) and groups observed over few years (1980s, 1990s, Millennials). Decision (Allen, 2026-10-31): keep the fits as they are and explain this in the text. Alternatives considered: fit each group only over the range where all its birth years are observed; model-based trajectories from the period–cohort model; narrower groups.

**Period–cohort model within each group (added 2026-10-31).** Allen's reading of the by-age and by-year figures: the lines are mostly straight, with no shape that repeats at the same ages (an aging pattern) or in the same years (a period pattern), so the data don't distinguish age from period effects, and the period–cohort model's constraint, one shared trend over time for all birth years in a group, is a reasonable choice. (Strictly, straight lines don't show that age and period effects are absent: the linear parts of the two are exactly what can't be separated. So the period component is within-cohort change, aging and the times combined.) Fitting the model within each group and standardizing the period component to the group's mix of birth years removes the composition shift. 17 fits; at the default target accept of 0.95, 1–5% of draws diverged (the 2000s: R-hat 1.08), so they were rerun at 0.99: at most 76 divergences in 8000 draws, R-hat ≤ 1.010, ESS ≥ 548. Estimates barely changed.

Slopes per decade. "PC" is a straight line fit to the standardized period component over the group's survey years, comparable with the other two columns; "PC, endpoints" is the change from first to last survey divided by decades, from the decomposition (with a 90% interval).

| Group | By age | By year | PC | PC, endpoints [90%] |
|---|---|---|---|---|
| Greatest | +0.12 | −0.21 | −0.10 | −0.08 [−0.23, +0.08] |
| 1900s | +0.08 | −0.08 | −0.00 | −0.02 [−0.46, +0.36] |
| 1910s | −0.06 | −0.15 | −0.06 | −0.06 [−0.23, +0.12] |
| 1920s | −0.02 | −0.16 | −0.11 | −0.10 [−0.30, +0.12] |
| Silent | −0.08 | −0.26 | −0.21 | −0.19 [−0.28, −0.09] |
| 1930s | −0.16 | −0.21 | −0.19 | −0.18 [−0.29, −0.05] |
| 1940s | −0.08 | −0.19 | −0.16 | −0.11 [−0.20, +0.00] |
| Boomers | −0.10 | −0.14 | −0.10 | −0.02 [−0.09, +0.06] |
| 1950s | −0.16 | −0.15 | −0.12 | −0.07 [−0.15, +0.01] |
| 1960s | −0.24 | −0.23 | −0.21 | −0.22 [−0.32, −0.13] |
| Gen X | −0.24 | −0.34 | −0.25 | −0.32 [−0.48, −0.16] |
| 1970s | −0.23 | −0.23 | −0.14 | −0.16 [−0.33, −0.01] |
| 1980s | −0.45 | −0.71 | −0.66 | −0.59 [−0.91, −0.22] |
| Millennials | −0.23 | −0.89 | −0.75 | −0.67 [−1.04, −0.31] |
| 1990s | +0.00 | −1.06 | −0.79 | −0.73 [−1.63, +0.21] |
| Gen Z | +0.22 | −0.55 | −0.58 | −0.54 [−2.13, +1.03] |
| 2000s | −2.11 | −0.22 | −1.07 | −1.00 [−4.67, +2.52] |

```{figure} overton_figs/apc/pc_by_decade_v1.png
:align: left
Period component of a period–cohort model fit within each decade of birth, standardized to the decade's birth years, set v1. Dotted: overall average.
```

```{figure} overton_figs/apc/pc_by_generation_v1.png
:align: left
The same by generation.
```

**What this shows:**
- **The within-group slopes fall between the by-age and by-year slopes, or close to one of them, for 15 of 17 groups**, as they should if those two are biased in opposite directions. The exceptions are the 1950s (−0.12 vs −0.16 and −0.15) and the 1970s (−0.14 vs −0.23 and −0.23), slightly less negative than both.
- **No group gets more conservative.** Every slope is negative or near zero. The Greatest generation's positive slope by age (+0.12) becomes −0.10, confirming it was composition.
- **For the young groups, the by-year slopes were closer to the truth** (1980s: −0.71 by year, −0.66 PC, −0.45 by age; Millennials −0.89, −0.75, −0.23): the bigger bias was in the by-age fits, where the old end of each line is only the earliest births.
- **The two summaries differ for groups whose lines rise and then fall** (Boomers: −0.10 as a line, −0.02 from endpoints). The interval reported by the fit is for the endpoint version; for the final version, add a posterior interval for the straight-line slope to `overton_apc.py`.

Figures: `overton_figs/apc/age_by_decade_v1`, `age_by_generation_v1`, `year_by_decade_v1`, `year_by_generation_v1`, `pc_by_decade_v1`, `pc_by_generation_v1`. Colors run from yellow (earliest births) to purple (latest). Some labels in the by-year figures overlap other lines; placement is not final. Labels: the earlier groups are labeled where their lines start, in the right part of the plot, and the later groups where their lines end, in the left part; the staggered starts and ends keep the labels apart.

## 2026-10-31: Question pool: stability rule, and the confidence questions

### Stability rule (adopted)

The method assumes each question has a fixed conservative answer. For every ranked candidate (171), `overton_stability.py` computes the conservative–liberal gap on the coded answer by decade (weighted percent of conservatives giving it minus liberals, 90% intervals, decades with at least 200 liberal and conservative respondents). A question is excluded if the gap credibly changes sign (credibly positive in one decade, credibly negative in another) or fades (credibly positive, then a later decade whose interval includes zero). It uses only the gap, the quantity the selection was based on, not trends. Results: `overton_figs/candidates_stability.csv`.

It flags 15 of the 65 questions in the pool, all ranked 39th or lower: six confidence questions (`coneduc`, `confinan`, `confed`, `conlegis`, `conjudge`, `conmedic`) and nine others whose gaps were small to begin with (`racpres`, `colmil`, `natsoc`, `librac`, `trust`, `aged`, `natspac`, `natcrime`, `natroad`). The top 38 are untouched. Decision (Allen, 2026-10-31): adopt it.

### Confidence-in-institutions questions: excluded (after independent review)

**The problem.** The stability rule keeps seven confidence questions, including `conpress` (pool position 20), whose gap widens (7 → 45 points) without changing sign. Adding `conpress` alone to the top 20 makes self-described conservatives' 50-year change about zero and produces significant within-cohort *increases* in four cohorts. With all confidence questions, the Overton gap levels off at about 1.4 for N ≥ 40; without them, about 0.9 (book's 15: 0.5–0.7).

**Process.** The confidence questions were judged in scope when the selection rule was frozen (2026-10-30), and the first decision was to report both pools and choose later. They were reconsidered only after seeing `conpress`'s effect on the results, and exclusion moves the results toward the book's original finding. Because of that, the question went out for independent evaluation, with a self-contained description of the background, rules, data, and arguments on both sides, and no recommendation (`overton_confidence_review.md`). Allen's instruction: do what the review recommends.

**The review's recommendation** (2026-10-31), in summary:
- Exclude all 13 confidence questions by scope, as the primary specification, and report the version that includes them as a sensitivity analysis, visibly, not in a footnote.
- The reason is construct validity, not the effect on results: confidence in "the people running" an institution is an evaluation of the institution, whose leadership, behavior, and political alignment change, not a position on an issue. A respondent can lose confidence in the press without any change in ideology, so the score would respond to changes in the institutions, not just in the respondents. The Supreme Court and the executive branch are the clearest cases, but the same applies to the press, science, and religion even where the sign of the gap doesn't change.
- The stability rule tests a weaker condition than the model requires: a stable sign doesn't show that an answer represents the same degree of conservatism in every period (differential item functioning over time). Keep the rule for the other questions, but don't use it as the only test of whether a question belongs.
- On the post hoc decision: address it with transparency, not by keeping questions that don't fit the construct. State that the confidence questions passed the original scope screen; explain the distinction that prompted reconsideration; say the decision was made after seeing results; report both versions; apply the revised scope rule from now on. Describe the procedure as rule-based with a documented post hoc revision, not as fully prespecified.
- Don't present the Overton gap as a precise quantity independent of measurement choices: the variation across question sets is probably larger than the model uncertainty. Show the gap against the number of questions, with one curve for each specification.
- Two checks: measurement invariance for `conpress` over time, and the 2020s jump in the gaps within interview modes.

**Decision (2026-10-31):** following the review, the primary specification excludes all 13 confidence questions (revised scope rule: evaluations of institutions are out, alongside religious belief and behaviors). The version that includes them is reported as a sensitivity analysis. The stability rule applies to everything else.

### The review's two checks (`overton_dif.py`)

**Measurement invariance over time.** For each target question, conservatism is estimated from the other set-v2 questions (weighted grid, original extract), and a weighted logistic regression of the target's coded answer on that score, decade, and their interaction gives the percent choosing the coded answer at a fixed score. At the median score of the 1970s (`overton_figs/dif_by_decade.csv`):

| Question | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|
| `conpress` (hardly any) | 18 | 24 | 40 | 45 | 51 | 67 |
| `contv` (hardly any) | 25 | 28 | 39 | 43 | 42 | 52 |
| `consci` (only some / hardly any) | 56 | 57 | 58 | 58 | 61 | 70 |
| `conarmy` (a great deal) | 37 | 34 | 43 | 53 | 59 | 50 |
| `conclerg` (a great deal / only some) | 84 | 80 | 80 | 80 | 80 | 74 |
| `conbus` (a great deal) | 25 | 27 | 27 | 22 | 20 | 17 |
| `conlabor` (hardly any) | 33 | 36 | 35 | 31 | 30 | 31 |
| `homosex` (always / almost always / sometimes wrong) | 89 | 90 | 82 | 74 | 59 | 52 |
| `grass` (not legal) | 76 | 81 | 78 | 67 | 47 | 35 |
| `fepol` (women not suited) | 49 | 38 | 24 | 26 | 21 | 20 |
| `abany` (no) | 66 | 64 | 60 | 66 | 66 | 63 |
| `gunlaw` (oppose permits) | 27 | 27 | 20 | 21 | 32 | 43 |
| `eqwlth` (government shouldn't, 4–7) | 51 | 55 | 60 | 61 | 65 | 68 |

- **`conpress` changed a lot at a fixed level of conservatism:** among people with the same score, "hardly any confidence" went from 18% to 67%. That's the reviewer's second possibility, a change specific to the question, which supports excluding it. `contv` is similar (25 → 52).
- **But item drift at a fixed score is not unique to the confidence questions.** `grass` (76 → 35), `homosex` (89 → 52), and `fepol` (49 → 20) drift strongly toward the *liberal* answer; `gunlaw` (27 → 43) and `eqwlth` (51 → 68) toward the conservative one; `abany` is stable. This is the item drift found earlier (2026-10-30), seen from another angle. `conpress`'s drift is among the largest, and it runs toward its coded conservative answer.
- Caveat (as the reviewer noted): the reference score comes from questions that drift too, so this assumes they're invariant on average.

**Interview mode in the 2020s** (conservative–liberal gap on the coded answer, 2022 and 2024 pooled, `overton_figs/dif_mode.csv`):

| Question | Interviewer (in person or phone) | Web |
|---|---|---|
| `conpress` | 38 | 48 |
| `consci` | 28 | 41 |
| `contv` | 13 | 27 |
| `conclerg` | 17 | 31 |
| `conlabor` | 17 | 31 |
| `homosex` | 41 | 46 |
| `abany` | 52 | 58 |
| `eqwlth` | 40 | 60 |
| `natenvir` | 39 | 55 |
| `helppoor` | 26 | 39 |

- **Gaps are larger on the web for nearly every question**, confidence and issue questions alike (exception: `conarmy`, 10 vs 8): part of the 2020s jump is the change in interview mode.
- **But the jump persists within interviewer-administered surveys:** `conpress`'s gap is 38 points there, against 21 in the 2010s (all interviewer-administered), and `consci`'s 28, against 14.

### What gets reported

- Primary: the pool without the confidence questions, with the stability rule.
- Sensitivity: with the confidence questions (those passing the stability rule).
- The Overton gap (and the within-cohort slope) as a function of N, one curve for each specification, not a single number.
- The description of the selection procedure says the scope rule was revised after seeing results, why, and that the review was sought for that reason.

## 2026-10-31: Within-cohort slope and Overton gap as functions of N

**Sets.** Primary specification (`p{N}`): the ranked pool without the 13 confidence questions and without the 15 that fail the stability rule (the topic rule applied after both, so a question of the same topic can move up; the social-trust topic drops out, since `trust`, `fair`, and `helpful` all fail). 43 questions. Sensitivity (`s{N}`): the same, plus the seven confidence questions that pass the stability rule; 50 questions. Nested top-N sets, N = 15, 20, …, all; the top 15 is set v2 in both. The primary pool's questions 16–43, in order: `xmarsex`, `spanking`, `fefam`, `discaff`, `prayer`, `gunlaw`, `pornlaw`, `divlaw`, `suicide4`, `letdie1`, `popular`, `colhomo`, `courts`, `fepol`, `nateduc`, `fepresch`, `natmass`, `racopen`, `colcom`, `colath`, `workblks`, `nataid`, `natpark`, `polescap`, `tax`, `getahead`, `racmar`, `natdrug`.

**Method.** Weighted grid estimates (book's difficulties from weighted prevalence; original extract, no resampling), scores rescaled to out of 15. For each set: the national period–cohort model (within-cohort slope: a straight line fit to the standardized period component over the survey years, per decade, with a 90% interval from the posterior draws; new in `overton_apc.py`, as decided) and the period model by `polviews3` (Overton gap: the conservative line in 2024 minus the liberal line in 1974; interval treating the two fits as independent). 60 fits on Unity at target accept 0.99 (`unity/overton_apc_sets_jobs.txt`); 17 had R-hat > 1.01 or ESS < 400 at 2000 draws, with no divergences, and were rerun at 8000 draws (`unity/overton_apc_sets_rerun_jobs.txt`). Final: at most 78 divergences in 8000 draws, R-hat ≤ 1.011, ESS ≥ 415. Table: `overton_figs/apc_by_n.csv`; figures: `overton_figs/apc/within_cohort_slope_by_n`, `overton_gap_by_n`.

| N | Within-cohort slope, primary | Sensitivity | Overton gap, primary | Sensitivity |
|---|---|---|---|---|
| 15 (v2) | −0.08 [−0.12, −0.04] | same | 2.32 [1.84, 2.79] | same |
| 20 | −0.03 [−0.06, +0.01] | +0.08 [+0.04, +0.11] | 2.39 | 2.88 |
| 25 | +0.02 [−0.01, +0.05] | +0.10 [+0.07, +0.13] | 2.51 | 2.85 |
| 30 | −0.06 [−0.09, −0.04] | +0.04 [+0.01, +0.07] | 2.14 | 2.70 |
| 35 | −0.10 [−0.13, −0.07] | −0.02 [−0.05, +0.01] | 1.90 | 2.08 |
| 40 | −0.12 [−0.15, −0.09] | −0.06 [−0.10, −0.03] | 1.63 | 1.80 |
| 45 | | −0.03 [−0.06, −0.00] | | 1.63 |
| all (43 / 50) | −0.09 [−0.12, −0.07] | −0.01 [−0.04, +0.02] | 1.59 [1.23, 1.95] | 1.76 [1.43, 2.10] |

For reference, the book's 15 questions (v1) on the same footing: within-cohort slope −0.16 [−0.20, −0.12].

```{figure} overton_figs/apc/within_cohort_slope_by_n.png
:align: left
Within-cohort slope (conservative responses per decade, out of 15) against the number of questions, with and without the confidence questions. Bands are 90% intervals (sampling only).
```

```{figure} overton_figs/apc/overton_gap_by_n.png
:align: left
Overton gap (2024 conservative minus 1974 liberal) against the number of questions.
```

**What this shows:**
- **Primary specification: no set shows people getting more conservative within cohorts.** The slope is negative at N = 15 and from N = 30 on (−0.06 to −0.12, intervals excluding zero), and indistinguishable from zero at N = 20 and 25 (the interval at 25 is −0.01 to +0.05). Questions 21–25 (`prayer`, `gunlaw`, `pornlaw`, `divlaw`, `suicide4`) include `gunlaw`, one of the questions that drifted toward its conservative answer at a fixed level of conservatism (2026-10-31 invariance check).
- **With the confidence questions, the slope is credibly positive at N = 20–30** (+0.04 to +0.10), the pattern that prompted the review, and near zero beyond.
- **The Overton gap is positive in every set**, larger than the book's (which used smoothed curves on resampled data, so not directly comparable), and declines as N grows: about 2.3–2.5 up to N = 25, about 1.6 at N = 40–43. It doesn't reach the book's "about the same" in any set. The gap and its decline are similar in both specifications; the confidence questions mainly affect the within-cohort slope.
- **The intervals are sampling uncertainty only.** The variation across N is larger, as the review pointed out; the curves, not a single N, are the result.
- **Settled vs not.** The qualitative claims hold in the primary specification at every N: the center moves toward liberal, the cross-sectional age gradient is a cohort effect, and people don't get more conservative as they age. The strong version ("today's conservative is about where the 1974 liberal was") doesn't hold with the broader question sets; today's conservative is 1.6–2.5 responses (out of 15) more conservative than the 1974 liberal.

These use the book's difficulties; they'll be redone with the pool difficulties from m1 when those fits finish.

**Decision: how to present it (Allen, 2026-10-31).** The headline numbers come from the full primary pool (43 questions): it's the one point on the curve with no choice of where to stop. The by-N curves show robustness, with the claims framed separately:
- "People don't get more conservative as they age" is robust: the within-cohort slope is never credibly positive at any N, and the full pool (−0.09) is near the strong end; other stopping points give the same or values closer to zero.
- The Overton gap is positive at every N, and the full pool gives the *smallest* value (1.6; up to 2.5 at N = 15–25), so the text says plainly that its size depends on the questions and that the headline uses the lowest value.
- Both curves (with and without the confidence questions) are shown prominently, as the review asked; the book's 15 questions (−0.16) are mentioned as more negative than any set in the new pool; and the text notes that every question counts equally, including weakly separating ones at the bottom of the pool, which is a reason someone might prefer a smaller N.

## 2026-10-31: Social vs economic questions

**Classification and review.** The 43 primary-pool questions were classified as social, economic, both, or neither (`overton_question_domains.md`) and sent out for review before any analysis. The review kept the classification (race questions social, `helpblk` both; domestic spending economic, defense and foreign aid neither) and asked for three prespecified economic specifications plus a social-only comparison, without expanding the pool:
- **A, core economic** (4): `helppoor`, `helpnot`, `helpsick`, `eqwlth`, the role of government in the economy.
- **B, broad economic** (10): A + `natenvir`, `nateduc`, `natmass`, `natpark`, `tax`, `getahead`.
- **C, inclusive economic** (12): B + `helpblk`, `natdrug`.
- **Social** (29): the rest, without both and neither.

Minimum answers for these sets: 2 (Allen; set A has only 4 questions). Set A has 39,930 respondents (the `help*` questions start in 1975 and are on fewer ballots); the others about 70,000. Scores on the out-of-15 scale (the expected fraction of conservative answers × 15). Book's difficulties, weighted, no resampling. Fits: period national, period by `polviews3`, period–cohort national, for each set (20 fits, 6000 draws, target accept 0.99): at most 19 divergences, R-hat ≤ 1.010, ESS ≥ 627. Table: `overton_figs/apc_domains.csv`.

| | Within-cohort slope [90%] | Total change | Generational replacement | Changing minds | Change: liberal / moderate / conservative | Con−Lib gap, first → last | Overton gap |
|---|---|---|---|---|---|---|---|
| A, core economic | **+0.24** [+0.18, +0.29] | −0.44 | −1.36 | +0.84 | −1.9 / −1.1 / **+1.1** | 2.7 → 5.7 | 3.8 |
| B, broad economic | **+0.07** [+0.02, +0.12] | −0.78 | −1.05 | +0.24 | −2.1 / −1.0 / **+0.5** | 1.8 → 4.5 | 2.3 |
| C, inclusive economic | **+0.07** [+0.03, +0.11] | −0.63 | −0.93 | +0.29 | −2.4 / −1.0 / **+0.6** | 1.5 → 4.5 | 2.2 |
| Social | **−0.14** [−0.17, −0.12] | −2.72 | −2.25 | −0.52 | −3.3 / −3.2 / −1.6 | 2.7 → 4.3 | 1.05 |
| Full pool (43) | −0.09 [−0.12, −0.07] | | −1.93 | −0.39 | −3.1 / −2.7 / −0.9 | 2.5 → 4.7 | 1.6 |
| Book's 15 (v1) | −0.16 [−0.20, −0.12] | −2.82 | −2.16 | −0.67 | −3.3 / −3.3 / −1.5 | 2.8 → 4.6 | 1.3 |

(Total change and the decomposition run from the first to the last survey; the changes by alignment from the first to the last year of each period model; set A's first year is 1975, so its Overton gap uses the 1975 liberal.)

```{figure} overton_figs/apc/time_series_econA.png
:align: left
Core economic questions (set A) by alignment. Dotted: the liberal level in the first year (1975).
```

```{figure} overton_figs/apc/time_series_social.png
:align: left
Social questions by alignment. Dotted: the 1974 liberal.
```

**What this shows:**
- **Two different stories.** On social questions, the Overton pattern holds in its classic form: everyone moves toward the liberal answers (liberals −3.3, moderates −3.2, conservatives −1.6), people get more liberal within cohorts (−0.14 per decade), and today's conservative is only about 1 answer (of 15) more conservative than the 1974 liberal; the moderate line crosses the 1974 liberal level around 2010. On core economic questions, opinion polarizes instead: liberals move left (−1.9, mostly after 2000), moderates change little (−1.1), and conservatives move *right* (+1.1); the conservative–liberal gap doubles.
- **On economic questions, people get *more* conservative within cohorts:** +0.24 per decade for the core set, +0.07 for the broader ones, with intervals excluding zero. The population still moves slightly left overall, because generational replacement (−1.36) outweighs the within-cohort change (+0.84). So the claim "people don't get more conservative as they age" holds for social questions and for the full pool, but not for economic questions. Within-cohort change combines aging and the times, so this could be people becoming more economically conservative as they age (income, property, taxes) or a period effect (the economic politics of the 1980s onward); the model can't separate them.
- **The full pool's result is mostly the social questions:** 29 of its 43 questions are social, and its numbers sit between the two. The book's 15 were also mostly social.
- **Core vs broad economic differ**, as the reviewer anticipated: adding the spending items, `tax`, and `getahead` cuts the within-cohort slope from +0.24 to +0.07 and the conservatives' move from +1.1 to +0.5. The spending items are relative to current spending, so they can move when spending changes; that's one reason to treat A as the primary economic result, as the review recommended.
- **Caveats:** set A is four questions from one GSS battery, asked of about 40,000 respondents, so each respondent's score is coarse (0 to 4 conservative answers); the difficulties are the book's, not yet the pool estimates; interview mode isn't modeled.

**Implication for the headline (to discuss).** The decision of 2026-10-31 framed "people don't get more conservative as they age" as robust across question sets. That's true across set sizes in the primary pool, but not across domains: on core economic questions, the within-cohort change is positive. The article and appendix should say so.

## Next

- **Headline framing.** Decide how the article and appendix present the domain split: on social questions (and the full pool), people don't get more conservative within cohorts; on core economic questions, they do (+0.24 per decade), and opinion polarizes.
- **Pool difficulties.** When the m1 fits on all pool questions finish (Unity array 66448475), feed their difficulties to the grid (`overton_pool_difficulties.py`) and redo the by-N, period–cohort, and domain results.
- **Mode sensitivity** for the final set: through 2018, and without the questions asked only in person.
- **Birth-group figures** (by decade of birth and by generation) with the final question set; fine-tune labels only then.

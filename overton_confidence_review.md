# Should the GSS confidence-in-institutions questions count toward a measure of conservatism?

*A request for independent evaluation. Allen Downey, 2026-10-31.*

## 1. Background

### The Overton paradox

Chapter 12 of *Probably Overthinking It* (Downey, 2023) describes what it calls the Overton paradox, using data from the General Social Survey (GSS):

1. Older people are more likely to say they are conservative.
2. Older people hold more conservative views, on average.
3. But as people get older, their views do not become more conservative; on average, they become more liberal.
4. Explanation: the center of public opinion (the "Overton window") moves toward the liberal side over time, so the labels shift. A person whose views stay put, or move liberal more slowly than the center, finds themselves further to the right of the center and is more likely to call themselves conservative.

A memorable version of the claim: the average self-described conservative today holds views about as conservative as the average self-described liberal in the 1970s.

### How conservatism is measured

- **Questions.** The book used 15 GSS opinion questions on which self-identified liberals and conservatives (the 7-point `polviews` scale, grouped into liberal, moderate, and conservative) answer most differently. Each question has a coding: the set of answers that count as "conservative", the side chosen more often by conservatives. The coding is fixed for all years.
- **Score.** A one-parameter item response model (Rasch). Each question's difficulty is set from how common the conservative answer is overall; each respondent's latent conservatism is estimated on a grid with a wide prior. The reported score is the posterior mean number of conservative answers out of 15. Respondents answer different subsets of questions (the GSS uses split ballots), and the model puts them on a common scale.
- **Trends.** Scores are then analyzed by survey year, birth cohort, age, and self-identified alignment.

### What we are doing now

We are updating the analysis with GSS data through 2024 (the book stopped at 2018), and checking how much the conclusions depend on the choice of questions. Work so far:

- **The book's results hold through 2024** with the book's 15 questions.
- **The direction of the results is robust to the question set; the size of the Overton gap is not.** The "Overton gap" here is the average self-described conservative in 2024 minus the average self-described liberal in 1974, in conservative answers out of 15; the book's claim is that it is near zero. With random subsets of the book's 15 questions, it ranges from −1.3 to +1.5.
- **A new, rule-based selection of questions** (below), to avoid choosing questions by hand.
- **Model checks.** Estimating the difficulties with MCMC instead of the book's shortcut changes the results by less than the variation between bootstrap resamplings. Age–period–cohort models (Bayesian random-walk smoothers, survey-weighted) confirm that: in every decade of surveys, older respondents are more conservative; but within birth cohorts, conservatism declines or stays flat over time, and most of the overall change comes from generational replacement.

The remaining decision is the question set. This document is about one part of it.

## 2. How questions are selected

The rules, decided before looking at how any question's answers changed over time:

1. **Pool.** Every variable in the GSS cumulative file (1972–2024, release 3a) that passes mechanical gates: asked in at least 8 survey years with at least 100 respondents who also answered `polviews`, spanning at least 15 years, with 2–7 distinct answers. 432 variables pass.
2. **Scope.** Opinion questions only. Excluded by category: demographics; behaviors and experiences (e.g. gun ownership); religious belief and practice; party identification and votes (circular with `polviews`); administrative variables; follow-up questions that depend on an earlier answer. 171 remain. The confidence-in-institutions questions were judged in scope at this step.
3. **Coding.** The book's coding for its 15 questions; for the others, the contiguous cut of the answer scale that best separates conservatives from liberals.
4. **Ranking.** By how strongly the coded answer is associated with `polviews`: a weighted logistic regression of the coded answer on survey year, birth year, and `polviews` (the book's own procedure), ranked by the `polviews` coefficient. No trends are computed for selection.
5. **Coverage.** Asked in at least 17 of the 33 survey years with `polviews`.
6. **Redundancy.** At most one question per topic (e.g. one abortion question, one question about government help for the poor), assigned by hand; for spending items, the original 1973 wording.
7. **Stability (added 2026-10-31).** The method assumes each question's conservative answer stays the conservative answer. For each candidate we computed the conservative–liberal gap on the coded answer in each decade (percent of conservatives giving it minus percent of liberals, weighted, with 90% intervals). A question is excluded if its gap is credibly positive in one decade and credibly negative in another (sign change), or credibly positive in one decade and indistinguishable from zero in a later one (fading). This removed 15 of the 65 candidates that pass rules 1–6, all ranked 39th or lower; the top 38 are unaffected.

After rules 1–6, 65 questions remain; ranked in order, they form nested sets ("top N"). Because the size of the Overton gap depends on which questions are included, we report results as a function of N rather than for one set.

## 3. The question

**Should the GSS confidence-in-institutions questions be excluded from the pool?**

The battery: "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?" Answers: a great deal / only some / hardly any. Thirteen institutions pass rules 1–6.

### Why it matters

The confidence questions are among those whose answers changed the most over 50 years. One of them, confidence in the press (`conpress`), enters the ranked list at position 20. Adding it alone changes the results noticeably: at N = 20, the change in the average self-described conservative over 50 years goes from about −1.0 to about −0.1 conservative answers, and four birth cohorts (of 12) show a statistically significant *increase* in conservatism over time, where otherwise none does. The effect on the Overton gap:

| N (top N questions) | Overton gap, with confidence questions | Overton gap, without | Change in average conservative, 50 years: with / without |
|---|---|---|---|
| 15 | 1.86 | 1.86 | −0.9 / −0.9 |
| 20 | 2.49 | 1.91 | −0.1 / −1.0 |
| 25 | 2.46 | 2.01 | −0.1 / −0.6 |
| 30 | 2.08 | 1.47 | −0.4 / −1.1 |
| 40 | 1.45 | 0.89 | −0.9 / −1.5 |
| 50 | 1.43 | 0.96 | −0.5 / −1.0 |
| all (65 with, 52 without) | 1.42 | 0.94 | −0.1 / −1.0 |

(Conservative answers out of 15; means over three weighted bootstrap resamplings; computed before the stability rule was added. For comparison, the book's 15 questions give an Overton gap of 0.5–0.7.)

Without the confidence questions, the gap levels off around 0.9 from N = 40 on; with them, around 1.4. So the decision affects how strong a version of the claim the data support.

### The 13 questions

"Coded conservative" is the answer chosen more often by conservatives over the whole period. "Trend" is the percent giving that answer in the first and last survey years. "Gap" is the conservative–liberal difference in that percent, by decade (percentage points). "Pool position" is the question's place in the ranked list after rules 1–6 (– if a question of the same topic ranked higher; here each institution is its own topic). Flags are from the stability rule (rule 7).

| Institution | Variable | Pool position | Coded conservative | Trend (first → last) | Gap by decade, 1970s → 2020s | Stability flag |
|---|---|---|---|---|---|---|
| Press | `conpress` | 20 | hardly any | 15% → 55% | +7, +11, +9, +18, +21, +45 | none |
| Organized labor | `conlabor` | 28 | hardly any | 27% → 23% | +13, +14, +12, +18, +21, +22 | none |
| Organized religion | `conclerg` | 34 | a great deal / only some | 84% → 65% | +3, +6, +10, +9, +18, +26 | none |
| Education | `coneduc` | 39 | hardly any | 8% → 26% | +4, +1, +6, +4, +7, +20 | fades |
| Major companies | `conbus` | 42 | a great deal | 31% → 15% | +12, +10, +7, +7, +9, +7 | none |
| Military | `conarmy` | 43 | a great deal | 33% → 43% | +8, +6, +11, +19, +18, +16 | none |
| Scientific community | `consci` | 46 | only some / hardly any | 59% → 64% | −0, +2, +5, +10, +14, +37 | none |
| Television | `contv` | 48 | hardly any | 22% → 48% | +4, +5, +9, +11, +4, +23 | none |
| Banks and financial institutions | `confinan` | 51 | a great deal / only some | 89% → 75% | +4, +5, +3, +2, +13, +8 | fades |
| Executive branch | `confed` | 57 | hardly any | 19% → 46% | −4, −10, +10, −11, +15, +31 | sign change |
| Congress | `conlegis` | 59 | hardly any | 16% → 52% | +4, +2, +4, −0, +1, +18 | fades |
| U.S. Supreme Court | `conjudge` | 62 | a great deal / only some | 84% → 63% | −4, −3, −4, +3, −3, +21 | sign change |
| Medicine | `conmedic` | 64 | only some / hardly any | 46% → 74% | −1, −3, −0, +2, +2, +17 | sign change |

Observations:
- **Six are already excluded by the stability rule** (education, banks, executive branch, Congress, Supreme Court, medicine). The question is about the other seven, above all `conpress`, which has the largest effect.
- **Confidence fell for most institutions.** Under the fixed codings, that decline counts as a *conservative* shift for eight institutions (press, education, military, science, TV, executive branch, Congress, medicine: the coded conservative answer is low confidence, or high confidence in the military) and as a *liberal* shift for five (labor, religion, business, banks, Supreme Court).
- **For most of the seven that pass the stability rule, the gap widens over time, mostly in the 2020s** (press +21 → +45; science +14 → +37; TV +4 → +23; religion +18 → +26). Many issue questions also widen in the same period (e.g. abortion for any reason +12 → +55; government reducing income differences +17 → +53), which is part of why a rule based on the size of the change in the gap doesn't separate the two kinds of question.
- **The 2020s data are different.** Since 2021, most GSS interviews have been self-administered on the web rather than face to face, and the decade's gaps for many questions jump. We have checked interview mode for the main results, but not specifically for these questions.

### Arguments for excluding them

1. **They are evaluations of institutions, not positions on issues.** The method assumes each question has a fixed conservative answer. A question like "should abortion be legal for any reason?" means the same thing in 1974 and 2024. Confidence in "the people running" an institution depends on who runs it, what it does, and how it's covered, all of which change. The clearest case is the Supreme Court: for decades liberals and conservatives expressed about the same confidence; after the Court's rightward turn, low confidence became a liberal answer. Liberals losing confidence in a conservative Court is not a shift toward conservatism, but a fixed coding counts it as one (or the opposite). Confidence in the executive branch flips with the party of the president.
2. **A general decline in trust is a different dimension.** Confidence in most institutions fell across the political spectrum. A broad decline in trust is not a movement along the left–right dimension the score is meant to measure, but with a fixed coding, it is counted as a conservative shift for some institutions and a liberal shift for others, depending on which side happened to be less trusting over the whole period.
3. **They are a category, like the ones already excluded.** The scope rules already exclude whole categories (religious belief, behaviors) because they aren't positions on political issues; the confidence battery could be treated the same way.
4. **The stability check already catches about half of them**, including the clearest cases (Supreme Court, executive branch). The rest share the same structure and are vulnerable to the same problem even if it hasn't shown up as a sign change yet.

### Arguments for keeping them

1. **They passed the rules that were set in advance.** The confidence questions were judged in scope, and the seven that remain pass the stability check. Removing them after seeing that they change the results is the kind of decision the rule-based selection was meant to prevent; it also moves the results toward the book's original finding.
2. **The seven that remain have a stable direction.** Conservatives have expressed less confidence in the press than liberals in every decade since the 1970s, and more in the military and in organized religion. If distrust of the press is a long-standing conservative position, its spread is arguably part of what the analysis is meant to measure, not noise.
3. **Exclusion by category is a judgment call.** Many issue questions also depend on context (e.g. spending on the military, or on crime, depends on current levels of spending and crime); the line between "evaluation" and "position" is not sharp.
4. **Reporting both versions is possible**, at the cost of a less clear headline.

### What we are asking

Given the purpose of the measure (a score of political conservatism, built from opinion questions with a fixed coding, used to compare people across 50 years), and given the information above:

1. **Should the confidence-in-institutions questions be excluded from the pool?** Options:
   - (a) exclude all 13 by scope (evaluations of institutions are out);
   - (b) exclude only the six flagged by the stability rule, keeping the other seven, including `conpress`;
   - (c) something else, e.g. a different criterion that distinguishes the questions that should stay from those that shouldn't.
2. **However the question is decided, what should be reported**, given that the decision is being made after seeing its effect on the results?
3. **Is there an additional check** that would bear on the decision, for example on the 2020s jump in the gaps?

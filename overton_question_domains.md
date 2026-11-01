# Classifying GSS opinion questions as social or economic

*A request for review. Allen Downey, 2026-10-31.*

## Purpose

We are updating an analysis of the "Overton paradox" (Chapter 12 of *Probably Overthinking It*): using the General Social Survey (GSS), 1972–2024, we measure each respondent's conservatism as the number of conservative answers to a set of opinion questions (a one-parameter item response model, with a fixed "conservative" answer for each question), then track how conservatism changes over time, across birth cohorts, and by self-described political label.

The main results use the 43 questions below, chosen by a fixed rule (questions on which self-described liberals and conservatives answer most differently, asked in at least half the survey years, one per topic, with a stable liberal–conservative direction over time). Earlier work suggests that the results differ between kinds of questions: on questions about the role of government in the economy, self-described conservatives have changed little since the 1970s; on questions about sex, family, religion, and personal morality, everyone has moved a lot.

We plan to rerun the analysis on the economic questions alone (and possibly the social questions alone). For that, we need a classification of each question that doesn't depend on the results. We're asking for review of the classification below before running anything.

## Categories

- **Economic:** the role of government in the economy, redistribution and the welfare state, taxes, and government spending on domestic programs; also beliefs about economic opportunity that underlie these positions.
- **Social:** sexuality, reproduction, marriage and family, gender roles, religion in public life, drugs, crime and punishment, guns, end of life, civil liberties, and race relations (beliefs about race, discrimination, and civil rights laws).
- **Both:** questions that combine the two in a way that can't be separated, e.g. government economic help for a particular racial group.
- **Neither:** questions that fit neither dimension, mainly foreign and defense policy.

Two judgment calls to look at in particular:

1. **Spending questions** ("are we spending too much, too little, or about the right amount on…"). We classified these as economic when the program is domestic (environment, education, mass transit, parks), as *both* when the program addresses a social problem (drug addiction), and as *neither* when it's foreign or defense (military, foreign aid). An alternative is to call all spending questions economic, since they measure support for government spending in general.
2. **Race questions.** We classified beliefs about race and civil rights laws (discrimination, affirmative action, open housing, intermarriage, stereotypes) as social, and government aid to Black Americans as both. An alternative is a separate race category, as in some of the literature.

## The questions

Questions are listed in rank order (how strongly the answer is associated with self-described political views). "Conservative answer" is the answer, or group of answers, that counts as conservative, chosen as the side that self-described conservatives choose more often. Our confidence in each classification is high unless marked.

The spending questions share a stem: "We are faced with many problems in this country, none of which can be solved easily or inexpensively. I'm going to name some of these problems, and for each one I'd like you to tell me whether you think we're spending too much money on it, too little money, or about the right amount."

| # | Variable | Question | Conservative answer | Classification | Notes |
|---|---|---|---|---|---|
| 1 | `homosex` | What about sexual relations between two adults of the same sex: do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all? | Always, almost always, or sometimes wrong | Social | |
| 2 | `helppoor` | Some people think that the government in Washington should do everything possible to improve the standard of living of all poor Americans (1). Other people think it is not the government's responsibility, and that each person should take care of himself (5). Where would you place yourself? | 4–5 (not the government's responsibility) | Economic | |
| 3 | `helpnot` | Some people think that the government in Washington is trying to do too many things that should be left to individuals and private businesses. Others think that the government should do even more to solve our country's problems. Where would you place yourself? (1–5) | 4–5 (government does too much) | Economic | |
| 4 | `sexeduc` | Would you be for or against sex education in the public schools? | Against | Social | |
| 5 | `helpsick` | Some people think that it is the responsibility of the government in Washington to see to it that people have help in paying for doctors and hospital bills. Others think that these matters are not the responsibility of the federal government and that people should take care of these things themselves. Where would you place yourself? (1–5) | 3–5 | Economic | |
| 6 | `eqwlth` | Some people think that the government in Washington ought to reduce the income differences between the rich and the poor, perhaps by raising the taxes of wealthy families or by giving income assistance to the poor. Others think that the government should not concern itself with reducing this income difference. (1 = should reduce, 7 = should not concern itself) | 4–7 | Economic | |
| 7 | `abany` | Should it be possible for a pregnant woman to obtain a legal abortion if the woman wants it for any reason? | No | Social | |
| 8 | `pillok` | Do you strongly agree, agree, disagree, or strongly disagree that methods of birth control should be available to teenagers between the ages of 14 and 16 if their parents do not approve? | Disagree or strongly disagree | Social | |
| 9 | `helpblk` | Some people think that Black people have been discriminated against for so long that the government has a special obligation to help improve their living standards. Others believe that the government should not be giving special treatment to Black people. Where would you place yourself? (1–5) | 4–5 (no special treatment) | Both | Government economic help, targeted by race |
| 10 | `natarms` | Spending on the military, armaments, and defense | Too little or about right | Neither | Defense policy. Lower confidence: could be economic (spending), as in judgment call 1 |
| 11 | `racdif1` | On the average, Black people have worse jobs, income, and housing than white people. Do you think these differences are mainly due to discrimination? | No | Social | Lower confidence: a belief about the causes of economic inequality; could be both |
| 12 | `premarsx` | If a man and woman have sex relations before marriage, do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all? | Always, almost always, or sometimes wrong | Social | |
| 13 | `grass` | Do you think the use of marijuana should be made legal or not? | Not legal | Social | |
| 14 | `natenvir` | Spending on improving and protecting the environment | About right or too much | Economic | Lower confidence: environmental policy is sometimes treated as its own dimension |
| 15 | `cappun` | Do you favor or oppose the death penalty for persons convicted of murder? | Favor | Social | |
| 16 | `xmarsex` | What is your opinion about a married person having sexual relations with someone other than the marriage partner: always wrong, almost always wrong, wrong only sometimes, or not wrong at all? | Always wrong | Social | |
| 17 | `spanking` | Do you strongly agree, agree, disagree, or strongly disagree that it is sometimes necessary to discipline a child with a good, hard spanking? | Strongly agree or agree | Social | |
| 18 | `fefam` | It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family. (agree/disagree) | Strongly agree or agree | Social | |
| 19 | `discaff` | What do you think the chances are these days that a white person won't get a job or promotion while an equally or less qualified Black person gets one instead? | Very or somewhat likely | Social | Lower confidence: about jobs and affirmative action; could be both |
| 20 | `prayer` | The United States Supreme Court has ruled that no state or local government may require the reading of the Lord's Prayer or Bible verses in public schools. What are your views on this: do you approve or disapprove of the court ruling? | Disapprove | Social | |
| 21 | `gunlaw` | Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun? | Oppose | Social | |
| 22 | `pornlaw` | Which of these statements comes closest to your feelings about pornography laws: there should be laws against the distribution of pornography whatever the age; laws against distribution to persons under 18; no laws forbidding distribution? | Laws against distribution whatever the age | Social | |
| 23 | `divlaw` | Should divorce in this country be easier or more difficult to obtain than it is now? | More difficult | Social | |
| 24 | `suicide4` | Do you think a person has the right to end his or her own life if this person is tired of living and ready to die? | No | Social | |
| 25 | `letdie1` | When a person has a disease that cannot be cured, do you think doctors should be allowed by law to end the patient's life by some painless means if the patient and his family request it? | No | Social | |
| 26 | `popular` | If you had to choose, which thing on this list would you pick as the most important for a child to learn to prepare him or her for life? (ranking of five qualities, including "to be well-liked or popular") | Ranks "to be well-liked or popular" least important | Social | Child-rearing values |
| 27 | `colhomo` | And what about a man who admits that he is a homosexual? Should such a person be allowed to teach in a college or university, or not? | Not allowed | Social | Civil liberties |
| 28 | `courts` | In general, do you think the courts in this area deal too harshly or not harshly enough with criminals? | Not harshly enough | Social | |
| 29 | `fepol` | Tell me if you agree or disagree with this statement: Most men are better suited emotionally for politics than are most women. | Agree | Social | |
| 30 | `nateduc` | Spending on improving the nation's education system | About right or too much | Economic | |
| 31 | `fepresch` | A preschool child is likely to suffer if his or her mother works. (agree/disagree) | Strongly agree or agree | Social | |
| 32 | `natmass` | Spending on mass transportation | About right or too much | Economic | |
| 33 | `racopen` | Suppose there is a community-wide vote on the general housing issue. There are two possible laws to vote on: (A) one law says that a homeowner can decide for himself whom to sell his house to, even if he prefers not to sell to Black people; (B) the second law says that a homeowner cannot refuse to sell to someone because of their race or color. Which law would you vote for? | A (homeowner decides) | Social | Lower confidence: civil rights law that also involves property rights; could be both |
| 34 | `colcom` | Now, I should like to ask you some questions about a man who admits he is a Communist. Suppose he is teaching in a college. Should he be fired, or not? | Fired | Social | Civil liberties |
| 35 | `colath` | There are always some people whose ideas are considered bad or dangerous by other people. For instance, somebody who is against all churches and religion. Should such a person be allowed to teach in a college or university, or not? | Not allowed | Social | Civil liberties |
| 36 | `workblks` | (7-point scale) Do people in this group tend to be hard-working or lazy? Black people | 5–7 (toward lazy) | Social | Racial stereotype |
| 37 | `nataid` | Spending on foreign aid | Too much | Neither | Foreign policy. Lower confidence: could be economic (spending) |
| 38 | `natpark` | Spending on parks and recreation | About right or too much | Economic | |
| 39 | `polescap` | Are there any situations you can imagine in which you would approve of a policeman striking an adult male citizen? If the citizen was attempting to escape from custody? | Yes | Social | |
| 40 | `tax` | Do you consider the amount of federal income tax which you have to pay as too high, about right, or too low? | Too high | Economic | About the respondent's own taxes, so partly self-interest rather than a policy position |
| 41 | `getahead` | Some people say that people get ahead by their own hard work; others say that lucky breaks or help from other people are more important. Which do you think is most important? | Hard work | Economic | Lower confidence: a belief about economic opportunity, not a policy; could be neither |
| 42 | `racmar` | Do you think there should be laws against marriages between Black people and white people? | Yes | Social | |
| 43 | `natdrug` | Spending on dealing with drug addiction | About right or too much | Both | A social problem addressed by spending; lower confidence |

## Summary

| Classification | Count | Questions |
|---|---|---|
| Economic | 10 | `helppoor`, `helpnot`, `helpsick`, `eqwlth`, `natenvir`, `nateduc`, `natmass`, `natpark`, `tax`, `getahead` |
| Social | 29 | the rest, except below |
| Both | 2 | `helpblk`, `natdrug` |
| Neither | 2 | `natarms`, `nataid` |

Lower-confidence classifications: `natarms`, `racdif1`, `natenvir`, `discaff`, `racopen`, `nataid`, `getahead`, `natdrug`; also `tax` (self-interest).

## Questions for the reviewer

1. Do you agree with the classification of each question? Please flag any you would change, especially the lower-confidence ones.
2. For the two judgment calls (spending questions; race questions), which alternative would you choose, and why?
3. For an analysis of "economic conservatism" alone, should *both* questions be included with the economic ones, excluded, or run both ways?
4. Ten economic questions (twelve with *both*) is a small set. Is that enough for a separate analysis, in your view, or should we look for additional economic questions that didn't make the main list (for example, other spending items or questions that were asked in fewer years)?

## Outcome (2026-10-31)

The review kept the proposed classification, with these points:

- Race questions stay social; `helpblk` is both.
- Spending: domestic programs economic; military and foreign aid neither; drug addiction both.
- The reviewer suggested `natenvir` and `nateduc` could be both, and `getahead` neither (a belief, not a policy), but noted that consistency favors keeping the domestic spending questions together. Rather than relabel them, the ambiguous questions are handled by prespecified specifications.
- Don't expand the pool to get more economic questions.

**Final classification**, as used, and membership in the analysis sets (defined in `notebooks/overton_apc.py`):

| Classification | Questions | Sets |
|---|---|---|
| Economic, core | `helppoor`, `helpnot`, `helpsick`, `eqwlth` | A, B, C |
| Economic, broad | `natenvir`, `nateduc`, `natmass`, `natpark`, `tax`, `getahead` | B, C |
| Both | `helpblk`, `natdrug` | C |
| Neither | `natarms`, `nataid` | none (full pool only) |
| Social | `homosex`, `sexeduc`, `abany`, `pillok`, `racdif1`, `premarsx`, `grass`, `cappun`, `xmarsex`, `spanking`, `fefam`, `discaff`, `prayer`, `gunlaw`, `pornlaw`, `divlaw`, `suicide4`, `letdie1`, `popular`, `colhomo`, `courts`, `fepol`, `fepresch`, `racopen`, `colcom`, `colath`, `workblks`, `polescap`, `racmar` | Social |

- **A, core economic** (primary economic specification): the role of government in the economy.
- **B, broad economic:** A plus domestic spending, `tax`, and `getahead`.
- **C, inclusive economic:** B plus the two "both" questions.
- **Social:** the 29 social questions, without both and neither.

All 43 are in the full pool, which is the headline specification.

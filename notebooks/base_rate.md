---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Fairness and Fallacy


<!-- #region tags=["remove-cell"] -->
This notebook contains the text from Chapter 9 of *[Probably Overthinking It: How to Use Data to Answer Questions, Avoid Statistical Traps, and Make Better Decisions](https://greenteapress.com/wp/probably-overthinking-it)* (University of Chicago Press, 2023), along with the supporting code used in the analysis.
<!-- #endregion -->

*Probably Overthinking It* is available from [Bookshop.org](https://bookshop.org/a/98697/9780226822587) and [Amazon](https://amzn.to/3Kp629E) (affiliate links).

If you like *Probably Overthinking It*, you'll like my book in progress, [*You Would Choose Now*](https://allendowney.com/ywcn), which uses decades of survey data to track changing attitudes in the United States. The early access edition is available now.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/base_rate.ipynb).

```python tags=["hide-cell"]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the random seed so we get the same results every time
np.random.seed(17)
```

In the criminal justice system, we use algorithms to guide decisions about who should be released on bail or kept in jail, and who should be kept in prison or released on parole.

Of course, we want those algorithms to be fair. 
For example, suppose we use an algorithm to predict whether a candidate for parole is likely to commit another crime, if released. 
If the algorithm is fair:

* Its predictions should mean the same thing for different groups of people. For example, if the algorithm predicts that a group of women and a group of men are equally likely to reoffend, we expect the numbers of women and men who actually reoffend to be the same, on average.

* Also, since the algorithm will make mistakes -- 
some people who reoffend will be assigned low probabilities, and some people who do not reoffend will be assigned high probabilities -- it should make these mistakes at the same rate in different groups of people.


It is hard to argue with either of these requirements.
Suppose the algorithm assigns a probability of 30\% to 100 women and 100 men; if 20 of the women commit another crime, and 40 of the men do, it seems like the algorithm is unfair.

Or suppose black candidates are more likely to be wrongly assigned a high probability, and white candidates are more likely to be wrongly assigned a low probability.
That seems unfair, too.

But here's the problem: it is not possible for an algorithm -- or a human -- to satisfy both requirements.
Unless two groups commit crimes at precisely the same rate, any classification that is equally predictive for both groups will necessarily make different kinds of errors between the groups.
And if we calibrate it to make the same kind of errors, the meaning of the predictions will be different.
To understand why, we have to understand the *base rate fallacy*.


In this chapter, I'll demonstrate the base rate fallacy using three examples:

* Suppose you take a medical test and the result is positive, which indicates that you have a particular disease. If the test is 99\% accurate, you might think there is a 99\% chance that you have the disease. But the actual probability could be much less.

* Suppose a driver is arrested because a testing device found that their blood alcohol was above the legal limit. If the device is 99\% accurate, you might think there's a 99\% chance they are guilty. In fact, the probability depends strongly on the reason the driver was stopped.

* Suppose you hear that 70\% of people who die from a disease had been vaccinated against it. You might think that the vaccine was not effective. In fact, such a vaccine might prevent 80\% of deaths or more, and save a large number of lives.

These examples might be surprising if you have not seen them before, but once you understand what's going on, I think they will make sense. 
At the end of the chapter, we'll return to the problem of algorithms and criminal justice.


## Medical Tests

During the COVID-19 pandemic, we all got a crash course in the science and statistics of infectious disease. 
One of the things we learned about is the accuracy of medical testing and the possibility of errors, both false positives and false negatives.

As an example, suppose a friend tells you they have tested positive for COVID, and they want to know whether they are really infected, or whether the result might be a false positive. What information would you need to answer their question?

One thing you clearly need to know is the accuracy of the test, but that turns out to be a little tricky. In the context of medical testing, we have to consider two kinds of accuracy: sensitivity and specificity.


* **Sensitivity** is the ability of the test to detect the presence of an infection, usually expressed as a probability. For example, if the sensitivity of the test is 87\%, that means 87 out of 100 people who are actually infected will get a positive test result, on average. The other 13 will get a *false negative*.

* **Specificity** is the ability of the test to indicate the absence of an infection. For example, if the specificity is 98\%, that means 98 out of 100 people who are not infected will get a negative test result, on average. The other 2 will get a *false positive*.

I did not make those numbers up. 
They are the reported sensitivity of a particular rapid antigen test, the kind used for at-home testing, in December 2021.
By the numbers, it sounds like the test is accurate, so if your friend tested positive, you might think they are likely to be infected.

But that's not necessarily true. It turns out that there is another piece of information we need to consider: the *base rate*, which is the probability that your friend was infected, based on everything we know about them *except* the outcome of the test.


For example, if they live someplace where the infection rate is high, we know they have been in a room with someone who was infected, and they currently have symptoms, the base rate might be quite high.
If they have been in strict isolation for 14 days and have no symptoms, it would be quite low.

To see why it matters, let's consider a case where the base rate is relatively low, like 1\%. And let's imagine a group of 1000 people who all take the test. In a group this size, we expect 10 people to be infected, because 10 out of 1000 is 1\%.

Of the 10 who are actually infected, we expect 9 to get a positive test result, because the sensitivity of the test is 87\%.

Of the other 990, we expect 970 to get a negative test result, because the specificity is 98\%. But that means we expect 20 people to get a false positive.

Before we go on, let's put the numbers we have so far in a table.

```python tags=["hide-input"]
table = pd.DataFrame(index=["Infected", "Not infected"])
table["# of people"] = 10, 990
table["Prob positive"] = 0.87, 0.02
table["# positive"] = (
    (table["# of people"] * table["Prob positive"]).round().astype(int)
)
table
```

The first column is the number of people in each group: infected or not.

The second column is the probability of a positive test for each group.
For someone who is actually infected, the probability of a positive test is 0.87, because sensitivity is 87\%.
For someone who is not infected, the probability of a *negative* test is 0.98, because specificity is 98\%, so the probability of a positive test is 0.02.

The third column is the product of the first two columns, which is the number of positive tests we expect in each group, on average.
Out of 1000 test results, 9 are true positives and 20 are false positives, for a total of 29.

Now we are ready to answer your friend's question: Given a positive test result, what is the probability that they are actually infected?
In this example, the answer is 9 out of 29, or 31\%.

Here's the table again, with a fourth column showing the probability of actual infection and the complementary probability that the test result is a false positive.

```python tags=["hide-input"]
total = table["# positive"].sum()
table["% of positive tests"] = (table["# positive"] / total).round(3) * 100
table
```

Although the sensitivity and specificity of the test are high, after a positive result, the probability that your friend is infected is only 31\%.
The reason it's so low is that the base rate in this example is only 1\%.


## More prevalence

To see why it matters, let's change the scenario. Suppose your friend has mild flu-like symptoms; in that case it seems more likely that they are infected, compared to someone with no symptoms.
Let's say it is ten times more likely, so the probability that your friend is infected is 10\% before we get the test results.
In that case, out of 1000 people with the same symptoms, we would expect 100 to be infected.
If we modify the first column of the table accordingly, here are the results.

```python tags=["hide-cell"]
def make_test_table(prior, likelihood, as_int=True):
    """Create a table showing test results for infected and not infected groups.
    
    prior: tuple of (infected_count, not_infected_count)
    likelihood: tuple of (sensitivity, 1 - specificity)
    as_int: if True, round positive test counts to integers
    """
    table = pd.DataFrame(index=["Infected", "Not infected"])
    table["# of people"] = prior
    table["Prob positive"] = likelihood
    table["# positive"] = table["# of people"] * table["Prob positive"]
    if as_int:
        table["# positive"] = table["# positive"].astype(int)
    total = table["# positive"].sum()
    table["% of positive tests"] = (table["# positive"] / total).round(3) * 100
    return table
```

```python tags=["hide-input"]
sens = 0.87
spec = 0.98
prior = 100, 900
likelihood = sens, 1 - spec
make_test_table(prior, likelihood)
```

Now the probability is about 83\% that your friend is actually infected, and about 17\% that the result is a false positive.
This example demonstrates two things:

1. The base rate makes a big difference, and

2. Even with an accurate test and a 10\% base rate, the probability of a false positive is still surprisingly high.


If the test is more sensitive, that helps, but maybe not as much as you expect.
For example, another brand of rapid antigen tests claims 95\% sensitivity, substantially better than the first brand, which was 87\%.
With this test, assuming the same specificity, 98\%, and the same base rate, 10\%, here's what we get.

```python tags=["hide-input"]
sens = 0.95
spec = 0.98
prior = 100, 900
likelihood = sens, 1 - spec
make_test_table(prior, likelihood)
```

Increasing the sensitivity from 87\% to 95\% has only a small effect: the probability that the test result is a false positive goes from 17\% to 16\%.


## More specificity

Increasing specificity has a bigger effect.
For example, lab tests that use PCR (polymerase chain reaction) are highly specific, about as close to 100\% as can be.
However, in practice it is always possible that a specimen is contaminated, a device malfunctions, or a result is reported incorrectly.

For example, in a retirement community near my house in Massachusetts, 18 employees and one resident tested positive for COVID in August 2020.
But all 19 turned out to be false positives, produced by a lab in Boston that was suspended by the Department of Public Health after they reported at least 383 false positive results.

It's hard to say how often something like that goes wrong, but if it happens one time in 1000, the specificity of the test would be 99.9\%.
Let's see what effect that has on the results.

```python tags=["hide-input"]
sens = 0.95
spec = 0.999
prior = 100, 900
likelihood = sens, 1 - spec
make_test_table(prior, likelihood, as_int=False)
```

With 95\% sensitivity, 99.9\% specificity, and 10\% base rate, the probability is about 99\% that your friend is actually infected, given a positive PCR test result.

However, the base rate still matters. Suppose you tell me that your friend lives in New Zealand where (at least at the time I am writing) the rate of COVID infection is very low. In that case the base rate for someone with mild flu-like symptoms might be 1 in 1000.

Here's the table with 95\% sensitivity, 99.9\% specificity, and base rate 1 in 1000.

```python tags=["hide-input"]
sens = 0.95
spec = 0.999
prior = 1, 999
likelihood = sens, 1 - spec
make_test_table(prior, likelihood, as_int=False)
```

In this example, the numbers in the third column aren't integers, but that's okay.
The calculation works the same way. Out of 1000 tests, we expect 0.95 true positives, on average, and 0.999 false positives. 
So the probability is about 49\% that a positive test is correct.
That's lower than most people think, including most doctors.


## Bad medicine

A 2014 paper in *The Journal of the American Medical Association* reports the result of a sneaky experiment.
The researchers asked a "convenience sample" of doctors (probably their friends and colleagues) the following question:

> "If a test to detect a disease whose prevalence is 1/1000 has a false positive rate of 5\%, what is the chance that a person found to have a positive result actually has the disease, assuming you know nothing about the person's symptoms or signs?"

What they call "prevalence" is what I've been calling "base rate". And what they call the "false positive rate" is the complement of specificity, so a false positive rate of 5\% corresponds to a specificity of 95\%.

Before I tell you the results of the experiment, let's work out the answer to the question.  We are not given the sensitivity of the test, so I'll make the optimistic assumption that it is 99\%. The following table shows the results.

```python tags=["hide-input"]
sens = 0.99
spec = 0.95
prior = 1, 999
likelihood = sens, 1 - spec
make_test_table(prior, likelihood, as_int=False)
```

The correct answer is about 2\%. 

Now, here are the results of the experiment:

> "Approximately three-quarters of respondents answered the question incorrectly. In our study, 14 of 61 respondents (23\%) gave a correct response. [...] the most common answer was 95\%, given by 27 of 61 respondents."

If the correct answer is 2\% and the most common response is 95\%, that is an alarming level of misunderstanding.


To be fair, the wording of the question might have been confusing. Informally, "false positive rate" could mean either:

* The fraction of uninfected people who get a positive test result,

* The fraction of positive test results that are false.

The first is the technical definition of "false positive rate"; the second is called the "false discovery rate". But even statisticians have trouble keeping these terms straight, and doctors are experts at medicine, not statistics.


However, even if the respondents misunderstood the question, their confusion could have real consequences for patients. 
In the case of COVID testing, a false positive test result might lead to an unnecessary period of isolation, which would be disruptive, expensive, and possibly harmful.
A state investigation of the lab that produced hundreds of false positive results concluded that their failures "put patients at immediate risk of harm".

Other medical tests involve similar risks.
For example, in the case of cancer screening, a false positive might lead to additional tests, unnecessary biopsy or surgery, and substantial costs, not to mention emotional difficulty for the patient and their family.

Doctors and patients need to know about the base rate fallacy.
As we'll see in the next section, lawyers, judges, and jurors do, too.


## Driving Under the Influence

The challenges of the base rate fallacy have become more salient as some states have cracked down on "drugged driving".

In September 2017 the American Civil Liberties Union (ACLU) filed suit against Cobb County, Georgia on behalf of four drivers who were arrested for driving under the influence of cannabis.
All four were evaluated by Officer Tracy Carroll, who had been trained as a "Drug Recognition Expert" (DRE) as part of a program developed by the Los Angeles Police Department in the 1970s.

At the time of their arrest, all four insisted that they had not smoked or ingested any cannabis products, and when their blood was tested, all four results were negative; that is, the blood tests found no evidence of recent cannabis use.

In each case, prosecutors dismissed the charges related to impaired driving.
Nevertheless, the arrests were disruptive and costly, and the plaintiffs were left with a permanent and public arrest record.


At issue in the case is the assertion by the ACLU that, "Much of the DRE protocol has never been rigorously and independently validated."

So I investigated that claim.
What I found was a collection of studies that are, across the board, deeply flawed.
Every one of them features at least one methodological error so blatant it would be embarrassing at a middle school science fair.

As an example, the lab study most often cited to show that the DRE protocol is valid was conducted at Johns Hopkins University School of Medicine in 1985.
It concludes, "Overall, in 98.7\% of instances of judged intoxication the subject had received some active drug". 
In other words, in the cases where one of the Drug Recognition Experts believed that a subject was under the influence, they were right 98.7\% of the time.


That sounds impressive, but there are several problems with this study.
The biggest is that the subjects were all "normal, healthy" male volunteers between 18 and 35 years old, who were screened and "trained on the psychomotor tasks and subjective effect questionnaires used in the study".

By design, the study excluded women, anyone older than 35, and anyone in poor health.
Then the screening excluded anyone who had any difficulty passing a sobriety test while they were sober -- for example, anyone with shaky hands, poor coordination, or poor balance.

But those are exactly the people most likely to be falsely accused.
How can you estimate the number of false positives if you exclude from the study everyone likely to yield a false positive?
You can't.


Another frequently-cited study reports that "When DREs claimed drugs other than alcohol were present, they [the drugs] were almost always detected in the blood (94\% of the time)".
Again, that sounds impressive until you look at the methodology.

Subjects in this study had already been arrested because they were suspected of driving while impaired, most often because they had failed a field sobriety test.

Then, while they were in custody, they were evaluated by a DRE, that is, a different officer trained in the drug evaluation procedure.
If the DRE thought that the suspect was under the influence of a drug, the suspect was asked to consent to a blood test; otherwise they were released.


Of 219 suspects, 18 were released after a DRE performed a "cursory examination" and concluded that there was no evidence of drug impairment.

The remaining 201 suspects were asked for a blood sample.
Of those, 22 refused and 6 provided a urine sample only.

Of the 173 blood samples, 162 were found to contain a drug other than alcohol.
That's about 94\%, which is the statistic they reported.


But the base rate in this study is extraordinarily high, because it includes only cases that were suspected by the arresting officer and then confirmed by the DRE.
With a few generous assumptions, I estimate that the base rate is 86\%; in reality, it was probably higher.


To estimate the base rate, let's assume:

* All 18 of the suspects who were released were, in fact, not under the influence of a drug, and

* The 28 suspects who refused a blood test were impaired at the same rate as the 173 who agreed, 94\%.

Both of these assumptions are generous; that is, they probably overestimate the accuracy of the DREs.
Even so, they imply that 188 out of 219 blood tests would have been positive, if they had been tested.
That's a base rate of 86\%.

```python tags=["hide-cell"]
def percent(y, n):
    """Calculate percentage: 100 * y / (y + n)."""
    return 100 * y / (y + n)
```

```python tags=["hide-cell"]
no_drug = 11
drug = 173 - 11
drug
```

```python tags=["hide-cell"]
rate = percent(drug, no_drug)
rate
```

```python tags=["hide-cell"]
refused = 28
total_pos = rate / 100 * refused + drug
total_pos
```

```python tags=["hide-cell"]
total_pos / 219
```

Because the suspects who were released were not tested, there is no way to estimate the sensitivity of the test, but let's assume it's 99\%, so if a suspect is under the influence of a drug, there is a 99\% chance a DRE would detect it. In reality, it is probably lower.

With these generous assumptions, we can use the following table to estimate the sensitivity of the DRE protocol.

```python tags=["hide-input"]
col1 = "Suspects"
col2 = "Prob positive"
col3 = "Cases"
col4 = "Percent"
table = pd.DataFrame(
    index=["Impaired", "Not impaired"], columns=[col1, col2, col3, col4]
)

table[col1] = 86, 14
table[col2] = 0.99, 0.4
table[col3] = table[col1] * table[col2]
total = table[col3].sum()
table[col4] = (table[col3] / total).round(3) * 100

table
```

With 86\% base rate, we expect 86 impaired suspects out of 100, and 14 unimpaired.
With 99\% sensitivity, we expect the DRE to detect about 85 true positives.
And with 60\% specificity, we expect the DRE to wrongly accuse 5.6 suspects.
Out of 91 positive tests, 85 would be correct; that's about 94\%, as reported in the study.

But this accuracy is only possible because the base rate in the study is so high.
Remember that most of the subjects had been arrested because they had failed a field sobriety test.
Then they were tested by a DRE, who was effectively offering a second opinion.


But that's not what happened when Officer Tracy Carroll arrested Katelyn Ebner, Princess Mbamara, Ayokunle Oriyomi, and Brittany Penwell.
In each of those cases, the driver was stopped for driving erratically, which is evidence of possible impairment.
But when Officer Carroll began his evaluation, that was the only evidence of impairment.

So the relevant base rate is not 86\%, as in the study; it is the fraction of erratic drivers who are under the influence of drugs.
And there are many other reasons for erratic driving, including distraction, sleepiness, and the influence of alcohol.
It's hard to say which explanation is most common.
I'm sure it depends on time and location.
But as an example, let's suppose it is 50\%; the following table shows the results with this base rate.

```python tags=["hide-input"]
table = pd.DataFrame(
    index=["Impaired", "Not impaired"], columns=[col1, col2, col3, col4]
)

table[col1] = 50, 50
table[col2] = 0.99, 0.4
table[col3] = table[col1] * table[col2]
total = table[col3].sum()
table[col4] = (table[col3] / total).round(3) * 100

table
```

With 50\% base rate, 99\% sensitivity, and 60\% specificity, the predictive value of the test is only 71\%; under these assumptions, almost 30\% of the accused would be innocent.
In fact, the base rate, sensitivity, and specificity are probably lower, which means that the value of the test is even worse.

The suit filed by the ACLU was not successful.
The court decided that the arrests were valid because the results of the field sobriety tests constituted "probable cause" for an arrest.
As a result, the court did not consider the evidence for, or  against, the validity of the DRE protocol.
The ACLU has appealed the decision.


## Vaccine Effectiveness

Now that we understand the base rate fallacy, we're ready to untangle a particularly confusing example of COVID disinformation.
In October 2021, a journalist appeared on a well-known podcast with a surprising claim. He said, "In the [United Kingdom] 70-plus percent of the people who die now from COVID are fully vaccinated."

The incredulous host asked, "Seventy percent?" and the journalist repeated, "Seven in ten of the people [who died] -- I want to keep saying because nobody believes it but the numbers are there in the government documents -- the vast majority of people in Britain who died in September were fully vaccinated."

Then, to his credit, he showed a table from a report published by Public Health England in September 2021.
From the table, he read off the number of deaths in each age group:
"1270 out of 1500 in the over 80 category [...] 607 of 800 of the 70 year-olds [...] They were almost all fully vaccinated. Most people who die of this now are fully vaccinated in the UK. Those are the numbers."

It's true; those are the numbers.
But the implication that the vaccine is useless, or actually harmful, is wrong.
In fact, we can use these numbers, along with additional information from the same table, to compute the effectiveness of the vaccine and estimate the number of lives it saved.

<!-- #region -->
Berenson: The vaccine still appears to have some protective effect .. that would still imply that the vaccine was doing *some* good.


Rogan: So when you say that most of the people who are dying are vaccinated is that because the [] rate of vaccination is very high?"

Berenson "Yes, *but*... there's another complexity here -- and this is the part that the vaccine [advocates?] never admit -- when you get to a place like Britain or Israel where almost everybody in that age range is vaccinated, who's not being vaccinated? Do you think there's a lot of people in the old age home who are saying, 'You know what, I'm insisting on my personal rights; you can't vaccinate me.'  Some 88 year old, no. The only people who aren't being vaccinated in that age group? Are probably too sick or too close to the end of their lives ..."

Rogan: Isn't that speculative, though?

Berenson: You caught me, because you're right, it is speculative. That is my speculation that there is this difference in these two groups.
<!-- #endregion -->

```python tags=["hide-cell"]
death_vax = 1272
death_unvax = 1521 - death_vax
death_vax + death_unvax
```

```python tags=["hide-cell"]
death_vax / (death_vax + death_unvax)
```

```python tags=["hide-cell"]
rate_vax = 495
rate_unvax = 1560

effectiveness = 1 - (rate_vax / rate_unvax)
effectiveness
```

Let's start with the oldest age group, people who were 80 or more years old.
In this group, there were 1521 deaths attributed to COVID during the four week period from August 23 to September 19, 2021.
Of the people who died, 1272 had been fully vaccinated.
The others were either unvaccinated or partially vaccinated; for simplicity I'll consider them all not fully vaccinated.
So, in this age group, 84% of the people who died had been fully vaccinated.
On the face of it, that sounds like the vaccine was not effective.

However, the same table also reports death rates among the vaccinated and unvaccinated, that is, the number of deaths as a fraction of the population in each age group.
During the same four week period, the death rates due to COVID were 1,560 per million people among the unvaccinated and 495 per million among the vaccinated.
So, the death rate was substantially lower among the vaccinated.

The following table shows these death rates in the second column, and the number of deaths in the third column.
Given these numbers, we can work forward to compute the fourth column, which shows again that 84% of the people who died had been vaccinated.

We can also work backward to compute the first column, which shows that there were about 2.57 million people in this age group who had been vaccinated, and only 0.16 million who had not.
So, more than 94% of this age group had been vaccinated.

```python tags=["hide-input"]
col1 = "Population"
col2 = "Death rate"
col3 = "Deaths"
col4 = "Percent"
table = pd.DataFrame(
    index=["Vaccinated", "Not vaccinated"], columns=[col1, col2, col3, col4]
)

table[col3] = death_vax, death_unvax
table[col2] = rate_vax, rate_unvax
table[col1] = table[col3] / table[col2]
total = table[col3].sum()
table[col4] = (table[col3] / total * 100).round(1)

table.round(2)
```

```python tags=["hide-cell"]
percent_vaccinated = table[col1] / table[col1].sum()
percent_vaccinated
```

From this table, we can also compute the effectiveness of the vaccine, which is the fraction of deaths the vaccine prevented.
The difference in the death rate from 1560 per million to 495 is a decrease of 68%.
By definition, this decrease is the "effectiveness" of the vaccine in this age group.

Finally, we can estimate the number of lives saved by answering a counterfactual question: if the death rate among the vaccinated had been the same as the death rate among the unvaccinated, how many deaths would there have been?
The answer is that there would have been 4,009 deaths.
In reality, there were 1,272, so we can estimate that the vaccine saved about 2,737 lives in this age group, in just four weeks.

In the United Kingdom right now, there are a lot of people visiting parents and grandparents at their homes, rather than a cemetery, because of the COVID vaccine.

```python tags=["hide-cell"]
counterfact = (
    table.loc["Vaccinated", "Population"] * table.loc["Not vaccinated", "Death rate"]
)
counterfact
```

```python tags=["hide-cell"]
# actual number, not rate, in one month
lives_saved = counterfact - table.loc["Vaccinated", "Deaths"]
death_vax, lives_saved
```

Of course, this analysis is based on some assumptions, most notably that the vaccinated and unvaccinated were similar except for their vaccination status.
That might not be true: people with high risk or poor general health might have been more likely to seek out the vaccine.
If so, our estimate would be too low, and the vaccine might have saved more lives.
If not, and people in poor health were *less* likely to be vaccinated, our estimate would be too high.
I'll leave it to you to judge which is more likely.


We can repeat this analysis with the other age groups.
The following table shows the number of deaths in each age group and the percentage of the people who died who were vaccinated.

I've omitted the "Under 18" age group because there were only 6 deaths in this group, 4 among the unvaccinated and 2 with unknown status.
With such small numbers, we can't make useful estimates for the death rate or effectiveness of the vaccine.

```python tags=["hide-cell"]
ages = [
    #    "Under 18",
    "18 to 29",
    "30 to 39",
    "40 to 49",
    "50 to 59",
    "60 to 69",
    "70 to 79",
    "80+",
]
```

```python tags=["hide-cell"]
# cases per 28 days, August 23 to September 19

deaths_vax = np.array([5, 10, 30, 102, 258, 607, 1272])
deaths_total = np.array([17, 48, 104, 250, 411, 801, 1521])
deaths_unvax = deaths_total - deaths_vax

deaths_vax.sum(), deaths_total.sum(), deaths_vax.sum() / deaths_total.sum()
```

```python tags=["hide-input"]
table2 = pd.DataFrame(index=ages, dtype=float)
table2["Deaths vax"] = deaths_vax
table2["Deaths total"] = deaths_total
percent = (deaths_vax / deaths_total) * 100
table2["Percent"] = percent.round(0).astype(int)
table2
```

Adding up the columns, there were a total of 3,152 deaths, 2,284 of them among the vaccinated.
So 72% of the people who died had been vaccinated, as the unnamed journalist reported.
Among people over 80, it was even higher, as we've already seen.

However, in the younger age groups, the percentage of deaths among the vaccinated is substantially lower, which is a hint that this number might reflect something about the groups, not about the vaccine.

To compute something about the vaccine, we can use death rates rather than number of deaths. 
The following table shows death rates per million people, reported by Public Health England for each age group, and the implied effectiveness of the vaccine, which is the percent reduction in death rate.

```python tags=["hide-cell"]
rates_vax = [1, 2, 5, 14, 45, 131, 495]
rates_unvax = [3, 12, 38, 124, 231, 664, 1560]
```

```python tags=["hide-input"]
table3 = pd.DataFrame(index=ages, dtype=float)
table3["Death rate vax"] = rates_vax
table3["Death rate unvax"] = rates_unvax
effectiveness = 100 * (1 - np.array(rates_vax) / rates_unvax)
table3["Effectiveness"] = effectiveness.round(0).astype(int)
table3
```

The effectiveness of the vaccine is more than 80% in most age groups.
In the youngest group it is 67%, but that might be inaccurate because the number of deaths is low and the estimated death rates are not precise.
In the oldest group it is 68%, which suggests that the vaccine is less effective for older people, possibly because their immune systems are weaker.
However, a treatment that reduces the probability of dying by 68% is still very good.

Effectiveness is nearly the same in most age groups because it reflects primarily something about the vaccines and only secondarily something about the groups.

Now, given the number of deaths and death rates, we can infer the number of people in each age group who were vaccinated or not, and the percentage who had been vaccinated.

```python tags=["hide-input"]
table4 = pd.DataFrame(index=ages, dtype=float)
table4["# vax (millions)"] = deaths_vax / rates_vax
table4["# unvax (millions)"] = deaths_unvax / rates_unvax
pop_total = table4.sum().sum()
percent = 100 * table4["# vax (millions)"] / table4.sum(axis=1)
table4["Percent vax"] = percent.round(0).astype(int)
table4.round(1)
```

By August 2021, nearly everyone in England over 60 years old had been vaccinated.
In the younger groups, the percentages were lower, but even in the youngest group it was more than half.

With this, it becomes clear why most deaths were among the vaccinated:

* Most deaths were in the oldest age groups, and

* In those age groups, almost everyone was vaccinated.

Taking this logic to the extreme, if everyone is vaccinated, we expect all deaths to be among the vaccinated.

In the vocabulary of this chapter, the percentage of deaths among the vaccinated depends on the effectiveness of the vaccine *and* the base rate of vaccination in the population.
If the base rate is low, as in the younger groups, the percentage of deaths among the vaccinated is low.
If the base rate is high, as in the older groups, the percentage of deaths is high.
Because this percentage depends so strongly on the properties of the group, it doesn't tell us much about the properties of the vaccine.

Finally, we can estimate the number of lives saved in each age group.
First we compute the hypothetical number of the vaccinated who would have died if their death rate had been the same as among the unvaccinated, then we subtract off the actual number of deaths.

```python tags=["hide-input"]
table5 = pd.DataFrame(index=ages, dtype=float)
table5["Hypothetical deaths"] = table4["# vax (millions)"] * table3["Death rate unvax"]
table5["Actual deaths"] = deaths_vax
table5["Lives saved"] = table5["Hypothetical deaths"] - table5["Actual deaths"]
table5.round(0).astype(int)
```

```python tags=["hide-cell"]
lives_saved = table5["Lives saved"].sum()
lives_saved
```

```python tags=["hide-cell"]
pop_total
```

```python tags=["hide-cell"]
pop_total / lives_saved * 1e6
```

In total, the COVID vaccine saved more than 7000 lives in a four-week period, in a relevant population of about 48 million.

If you created a vaccine that saved 7000 lives in less than a month, in just one country, you would feel pretty good about yourself.
And if you used misleading statistics to persuade a large, international audience that they should not get that vaccine, you should feel very bad.


## Predicting Crime

If we understand the base rate fallacy, we can correctly interpret medical and impaired driving tests, and we can avoid being misled by headlines about COVID vaccines.
We can also shed light on an ongoing debate about the use of data and algorithms in the criminal justice system.

In 2016 a team of journalists at ProPublica published a now-famous article about COMPAS, which is a statistical tool used in some states to inform decisions about which defendants should be released on bail before trial, how long convicted defendants should be imprisoned, and whether prisoners should be released on probation.

COMPAS uses information about defendants to generate a "risk score" which is supposed to quantify the probability that the defendant will commit another crime if released.

The authors of the ProPublica article used public data to assess the accuracy of COMPAS risk scores.
They explain:

> We obtained the risk scores assigned to more than 7,000 people arrested in Broward County, Florida, in 2013 and 2014 and checked to see how many were charged with new crimes over the next two years, the same benchmark used by the creators of the algorithm.

They published the data they obtained, so we can use it to replicate their analysis and do our own.


NOTE: The statistics reported in the rest of this chapter are from the [Recidivism Case Study](https://allendowney.github.io/RecidivismCaseStudy/), to be published as part of [Elements of Data Science](https://allendowney.github.io/ElementsOfDataScience).


If we think of COMPAS as a diagnostic test, a high risk score is like a positive test result and a low risk score is like a negative result.
Under those definitions, we can use the data to compute the sensitivity and specificity of the test.
As it turns out, they are not very good:

* *Sensitivity*: Of the people who were charged with another crime during the period of observation, only 63\% were given high risk scores.

* *Specificity*: Of the people who were *not* charged with another crime, only 68\% were given low risk scores.

Now suppose you are a judge considering a bail request from a defendant who has been assigned a high risk score. Among other things, you would like to know the probability that they will commit a crime if released. Let's see if we can figure that out.

As you might guess by now, we need another piece of information: the base rate. In the sample from Broward County, it is 45\%; that is, 45\% of the defendants released from jail were charged with a crime within two years.

The following table shows the results with this base rate, sensitivity, and specificity.

```python tags=["hide-cell"]
def make_risk_table(prior, likelihood, as_int=True):
    """Create a table showing risk scores for charged and not charged groups.
    
    prior: array of (charged_count, not_charged_count)
    likelihood: tuple of (sensitivity, 1 - specificity) for high risk scores
    as_int: if True, round high risk counts to integers
    """
    table = pd.DataFrame(index=["Charged again", "Not charged"])
    table["# of people"] = prior.astype(int)
    table["P(high risk)"] = likelihood
    table["# high risk"] = table["# of people"] * table["P(high risk)"]
    if as_int:
        table["# high risk"] = table["# high risk"].astype(int)
    total = table["# high risk"].sum()
    table["Percent"] = (table["# high risk"] / total).round(3) * 100
    return table
```

```python tags=["hide-input"]
sens = 0.63
spec = 0.68
prev = 0.45
prior = np.array([prev, 1 - prev]) * 1000
likelihood = sens, 1 - spec
make_risk_table(prior, likelihood)
```

Out of 1000 people in this dataset, 450 will be charged with a crime, on average; the other 550 will not.

Based on the sensitivity and specificity of the test, we expect 283 of the offenders to be assigned a high risk score, along with 175 of the non-offenders. So, of all people with high risk scores, about 62\% will be charged with another crime.

This result is called the "positive predictive value", or PPV, because it quantifies the accuracy of a positive test result. In this case, 62\% of the positive tests turn out to be correct.

We can do the same analysis with low risk scores.

```python tags=["hide-input"]
sens = 1 - 0.63
spec = 1 - 0.68
prev = 0.45
prior = np.array([prev, 1 - prev]) * 1000
likelihood = sens, 1 - spec
table = make_risk_table(prior, likelihood)
table.columns = ['# of people', 'P(low risk)', '# low risk', 'Percent']
table
```

Out of 450 offenders, we expect 166 to get an incorrect low score. Out of 550 non-offenders, we expect 374 to get a correct low score. 
So, of all people with low risk scores, 69\% were not charged with another crime. 

This result is called the "negative predictive value" of the test, or NPV, because it indicates what fraction of negative tests are correct.


On one hand, these results show that risk scores provide useful information.
If someone gets a high risk score, the probability is 62\% that they will be charged with a crime. 
If they get a low risk score, it is only 31\%. 
So, people with high risk scores are about twice as likely to re-offend.

On the other hand, these results are not as accurate as we would like when we make decisions that affect people's lives so seriously.
And they might not be fair.


## Comparing Groups

The authors of the ProPublica article considered whether COMPAS has the same accuracy for different groups. With respect to racial groups, they find:

> ... In forecasting who would re-offend, the algorithm made mistakes with black and white defendants at roughly the same rate but in very different ways.
>
> * The formula was particularly likely to falsely flag black defendants as future criminals, wrongly labeling them this way at almost twice the rate as white defendants.
>
> * White defendants were mislabeled as low risk more often than black defendants.

This discrepancy suggests that the use of COMPAS in the criminal justice system is racially biased.

I will use the data they obtained to replicate their analysis, and we will see that their numbers are correct.
But interpreting these results turns out to be complicated; I think it will be clearer if we start by considering sex, and then race.


In the data from Broward County, 81\% of defendants are male and 19\% are female. The sensitivity and specificity of the risk scores is almost the same in both groups:

* Sensitivity is 63\% for male defendants and 61\% for female defendants.

* Specificity is close to 68\% for both groups.

But the base rate is different: about 47\% of male defendants were charged with another crime, compared to 36\% of female defendants.

In a group of 1000 male defendants, the following table shows the number we expect to get a high risk score and the fraction of them that will re-offend.

```python tags=["hide-input"]
sens = 0.63
spec = 0.68
prev = 0.47
prior = np.array([prev, 1 - prev]) * 1000
likelihood = sens, 1 - spec
make_risk_table(prior, likelihood)
```

Of the high risk male defendants, about 64\% were charged with another crime.

Here is the corresponding table for 1000 female defendants.

```python tags=["hide-input"]
sens = 0.61
spec = 0.68
prev = 0.36
prior = np.array([prev, 1 - prev]) * 1000
likelihood = sens, 1 - spec
make_risk_table(prior, likelihood)
```

Of the high risk female defendants, only 52\% were charged with another crime. 

And that's what we should expect: if the test has the same sensitivity and specificity, but the groups have different base rates, the test will have different predictive values in the two groups.


Now let's consider racial groups. As the ProPublica article reports, the sensitivity and specificity of COMPAS are substantially different for white and black defendants:

* Sensitivity for white defendants is 52\%; for black defendants it is 72\%.

* Specificity for white defendants is 77\%; for black defendants it is 55\%.

The complement of sensitivity is the "false negative rate", or FNR, which in this context is the fraction of offenders who were wrongly classified as low risk. The false negative rate for white defendants is 48\% (the complement of 52%); for black defendants it is 28\%

And the complement of specificity is the "false positive rate", or FPR, which is the fraction of non-offenders who were wrongly classified as high risk. The false positive rate for white defendants is 23\% (the complement of 77%); for black defendants it is 45\%.


In other words, black non-offenders were almost twice as likely to bear the cost of an incorrect high score. And black offenders were substantially less likely to get the benefit of an incorrect low score.

That seems patently unfair. As U.S. Attorney General Eric Holder wrote in 2014 (as quoted in the ProPublica article), "Although these measures were crafted with the best of intentions, I am concerned that they inadvertently undermine our efforts to ensure individualized and equal justice [and] they may exacerbate unwarranted and unjust disparities that are already far too common in our criminal justice system and in our society."

But that's not the end of the story.


## Fairness is Hard to Define

A few months after the ProPublica article, the Washington Post published a response with the expositive title: "A computer program used for bail and sentencing decisions was labeled biased against blacks. It's actually not that clear."

It acknowledges that the results of the ProPublica article are correct: the false positive rate for black defendants is higher and the false negative rate is lower.
But it points out that the PPV and NPV are the nearly the same for both groups:

* *Positive predictive value*: Of people with high risk scores, 59\% of white defendants and 63\% of black defendants were charged with another crime. 

* *Negative predictive value*: Of people with low risk scores, 71\% of white defendants and 65\% of black defendants were *not* charged again.

So in this sense the test is fair: a high risk score in either group means the same thing; that is, it corresponds to roughly the same probability of recidivism.
And a low risk score corresponds to roughly the same probability of non-recidivism.


Strangely, COMPAS achieves one kind of fairness based on sex, and another kind of fairness based on race.

* For male and female defendants, the error rates (false positive and false negative) are roughly the same, but the predictive values are different.

* For black and white defendants, the error rates are substantially different, but the predictive values (PPV and NPV) are about the same.

The COMPAS algorithm is a trade secret, so there is no way to know why it is designed this way, or even whether the discrepancy is deliberate.
But the discrepancy is not inevitable. 
COMPAS could be calibrated to have equal error rates in all four groups, or equal predictive values.

However, it cannot have the same error rates *and* the same predictive values.
We have already seen why: if the error rates are the same and the base rates are different, we get different predictive values. 
And, going the other way, if the predictive values are the same and the base rates are different, we get different error rates.


At this point it is tempting to conclude that algorithms are inherently unfair, so maybe we should rely on humans instead of algorithms. But this option is not as appealing as it might seem.

First, it doesn't actually solve the problem: just like an algorithm, a human judge cannot achieve equal error rates for different groups and equal predictive values at the same time. The math is inescapable by man or machine.


Second, if the task is to use data to generate predictions, humans are almost always worse than algorithms. To see why, let's consider the reasons a person and an algorithm might disagree:

1. A human might consider additional information that's not available to the algorithm. For example, a judge might assess whether a defendant seems contrite, based on their behavior in court.

2. A human might consider the same information as an algorithm, but give different weight to different factors. For example, a judge might give more weight to age than the algorithm, and less weight to prior arrests.

3. A human might be influenced by factors that don't affect algorithms, like political beliefs, personal biases, and mood.


Taking these in turn: 

1. If a human judge uses more information than the algorithm, the additional information may or may not be valid. If not, it provides no advantage. If so, it could be included in the algorithm. For example, if judges record their belief about whether a defendant is contrite or not, we could check whether their assessment is actually predictive, and if so, we could add it to the algorithm.

2. If a human judge gives more weight to some factors, relative to the algorithm, and less weight to others, the results are unlikely to be better. After all, figuring out which factors are predictive, and how much weight to give each one, is exactly what algorithms are designed to do, and they are generally better at it than humans.


3. Finally, there is ample evidence that judges differ from each other in consistent ways, and differ from themselves over time. The outcome of a case should not depend on whether it is assigned to a harsh or a lenient judge, or whether it is heard before or after lunch. And it certainly should not depend on prejudices the judge may have based on race, sex, and other group membership.

I don't mean to say that algorithms are guaranteed to be free of this kind of prejudice. If they are based on previous outcomes, and if those outcomes are subject to bias, algorithms can replicate and perpetuate that bias.


For example, the dataset used by ProPublica to validate COMPAS indicates whether each defendant was *charged* with another crime during the period of observation. But what we really want to know is whether the defendant *committed* another crime, and that is not the same thing.

Not everyone who commits a crime gets charged -- not even close. The probability of getting charged for a particular crime depends on the type of crime and location; the presence of witnesses and their willingness to work with police; the decisions of police about where to patrol, what crimes to investigate, and who to arrest; and decisions of prosecutors about who to charge. 
It is likely that every one of these factors depends on the race and sex of the defendant.


This kind of data bias is a problem for algorithms like COMPAS.
But it is also a problem for humans: exposed to biased data, we tend to make biased judgments. The difference is that humans can handle less data, and we are less good at extracting reliable information from it. Trained with the same data, an algorithm will be about as biased as the average judge, less biased than the worst judge, and less noisy than any judge.

Also, algorithms are easier to correct than humans. If we discover that an algorithm is biased, and we can figure out how, we can often unbias it. 
If we could do that with humans, the world would be a better place.


For all of these reasons, I think algorithms like COMPAS have a place in the criminal justice system.
But that brings us back to the question of calibration.


## Fairness is Hard to Achieve

Even if you think we should not use predictive algorithms in the criminal justice system, the reality is that we do. So at least for now we have a difficult question to answer:

1. Should we calibrate algorithms so predictive values are the same in all groups, and accept different error rates (as we see with black and white defendants)?

2. Or should we calibrate them so error rates are the same in all groups, and accept different predictive values (as we see with male and female defendants)?

3. Or should we compromise between the extremes, and accept different error rates *and* different predictive values.

If we choose either of the first two options, we run into two problems: the number of groups is large, and every defendant belongs to several of them.

Consider a defendant who is a 50-year old African-American woman. What is the false positive rate for her group? As we've already seen, FPR for black defendants is 45\%. But for black women it's 40\%, for women older than 45 it's 15\%, and for black women older than 45 it's 24\%.

We have the same problem with the false negative rate. For example, FNR for white defendants is 48\%, but for white women it is 43\%, for women younger than 25, it's 18\% and for white women younger than 25, it's just 4\%!


Predictive values (PPV and NPV) don't differ as much between groups, but if you search for the extremes, you can find substantial differences.
Among the subgroups I looked at (excluding very small groups):

* COMPAS has the highest positive predictive value for black men younger than 25, 70\%. It has the lowest PPV for Hispanic defendants older than 45, 29\%.

* It has the highest negative predictive value for white women younger than 25, 95\%, and the lowest NPV for men under 25 whose racial category is "Other", 49\%.

With six racial categories, three age groups, and two sexes, there are 81 subgroups.
It is not possible to calibrate any algorithm to achieve the same error rates or the same predictive values in all of these groups.


So, suppose we use an algorithm that allows error rates and predictive values to vary between groups.
How should we design it, and how should we evaluate it? 
Let me suggest a few principles to start with:

1. If one of the goals of incarceration is to reduce crime, it is better to keep in prison someone who will commit another crime, if released, than someone who will not. Of course we don't know with certainty who will re-offend, but we can make probablistic predictions.

2. The public interest is better served if our predictions are accurate, otherwise we will keep more people in prison than necessary, or suffer more crime than necessary, or both.

3. However, we should be willing to sacrifice some accuracy in the interest of justice. For example, suppose we find that, comparing male and female defendants who are alike in every other way, women are more likely to re-offend. In that case, including sex in the algorithm might improve its accuracy. Nevertheless, we might decide to exclude this information on the grounds that using it would violate the principle of equality before the law.

4. The criminal justice system should be fair, and it should be perceived to be fair. However, we have seen that there are conflicting definitions of fairness, and it is mathematically impossible to satisfy all of them.


Even if we agree that these principles should guide our decisions, they provide a framework for a discussion rather than a resolution.

For example, reasonable people could disagree about what factors should be included in the algorithm. I suggested that sex should be excluded even if it improves the accuracy of the predictions. For the same reason, we might choose to exclude race.

But what about age? If two defendants are similar except that one is 25 years old and the other is 50, the younger person is substantially more likely to re-offend. So an algorithm that includes age will be more accurate than one that does not. And on the face of it, releasing someone from prison because they are old does not seem obviously unjust.
But a person does not choose their age any more than they choose their race or sex. So I'm not sure what principle justifies the decision to include age while excluding race and sex. 

The point of this example is that these decisions are hard because they depend on values that are not universal.


Fortunately, we have tools for making decisions when people disagree, including public debate and representative democracy. But the key words in that sentence are "public" and "representative". The algorithms we use in the criminal justice system should be a topic of public discussion, not a trade secret. And the debate should include everyone involved, including perpetrators and victims of crime.


## All about the base rate

Sometimes the base rate fallacy is funny. There's a very old joke that goes like, "I read that 21% of car crashes are caused by drunk drivers. Do you know what that means? It means that 79% are caused by sober drivers. Those sober drivers aren't safe -- get them off the road!"

And sometimes the base rate fallacy is obvious, like the xkcd comic that says "Remember, right-handed people commit 90% of all base rate errors".

But often it is more subtle.
When someone says a medical test is accurate, they usually mean that is it sensitive and specific: that is, likely to be positive if the condition it detects is present, and likely to be negative if the condition is absent.
And those are good properties for a test to have.

But they are not enough to tell us what we really want to know, which is whether a particular result is correct. For that, we need the base rate, and it often depends on the circumstances of the test.

For example, if you go to a doctor because you have symptoms of a particular disease and they test for the disease, that's a diagnostic test. If the test is sensitive and specific, and the result is positive, it's likely that you have the disease. 

But if you go to the doctor for a regular checkup, you have no symptoms, and they test for a rare disease, that's a screening test. In that case, if the result is positive, the probability that you have the disease might be small, even if the test is highly specific.

And it's important for you to know this, because there's a good chance your doctor does not.


## Sources and related reading

* The sensitivity and specificity of COVID antigen tests was reported in the Wirecutter feature of [*The New York Times*](https://www.nytimes.com/wirecutter/reviews/at-home-covid-test-kits) (Lee, 2021).

* The false positive COVID tests from the Orig3n lab were reported in [*The Boston Globe*](https://www.bostonglobe.com/2020/09/08/nation/state-suspends-covid-19-testing-orig3n-boston-based-lab-responsible-least-383-false-positive-results/) (Moore, 2020).

* The experiment that asked doctors to interpret the results of a medical test was described in [*JAMA Internal Medicine*](https://doi.org/10.1001/jamainternmed.2014.1059) (Manrai et al., 2014).


* The story of the drivers in Georgia who were wrongly arrested for driving under the influence of cannabis was reported by [11 Alive, WXIA-TV in Atlanta, Georgia](https://www.11alive.com/article/news/investigations/the-drug-whisperer-drivers-arrested-while-stone-cold-sober/85-502132144) (Keefe and King, 2018).

* The complaint filed on their behalf, and updates on the appeal, are available from the [ACLU website](https://acluga.org/ebner-v-cobb-county) (2017).

* The 1985 lab test of the DRE protocol is reported in a [technical report from the National Highway Traffic Safety Administration](http://www.decp.us/pdfs/Bigelow_1985_DRE_validation_study.pdf) (Bigelow et al., 1985). The 1986 field test is reported in [another technical report](https://trid.trb.org/view/466854) (Compton, 1986).



* The podcast where the unnamed journalist committed the base rate fallacy is [*The Joe Rogan Experience*](https://open.spotify.com/episode/1VNcMVzwgdU2gXdbw7yqCL?si=WjK0_EQ7TXGgFdVmda_gRw) (2021).

* The data he reported is from [Public Health England](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1019992/Vaccine_surveillance_report_-_week_38.pdf) (2021).


* The ProPublica article about COMPAS is available online: ["Machine Bias"](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) (Angwin et al., 2016). The response in *The Washington Post* was published in 2016 (Corbett-Davies et al., 2016).

* My discussion of Machine Bias is based on a case study that is part of the *Elements of Data Science* curriculum (Downey, 2021). The Python code I used to analyze the data is available in [Jupyter notebooks](https://allendowney.github.io/RecidivismCaseStudy) (Downey, 2021).

* For a discussion of the vagaries of human judges, I recommend Chapter 1 of [*Noise: A Flaw in Human Judgment*](https://books.google.com/books?id=vCZMzQEACAAJ) (Kahneman, Sibony, and Sunstein, 2021).

* The company that produces COMPAS was called Northpointe in 2016. After the ProPublica article was published, the name of the company was changed to Equivant.


* The [xkcd comic about the base rate fallacy](https://xkcd.com/2476/) is number 2476 (Munroe, 2021). The hover text is "Sure, you can talk about per-capita adjustment, but if you want to solve the problem, it's obvious that this is the group you need to focus on."

* A January 2022 feature in [*TheUpshot*](https://www.nytimes.com/2022/01/01/upshot/pregnancy-birth-genetic-testing.html) discusses the high false positive rates of screening tests for rare genetic disorders (Kliff and Bhatia, 2022).

* A 2022 article in [*The New York Times*](https://www.nytimes.com/2022/06/10/health/cancer-blood-tests.html) discusses potential harms of increased screening for rare cancers (Kolata, 2022).


Probably Overthinking It

Copyright 2022 Allen Downey 

The code in this notebook and `utils.py` is under the [MIT license](https://mit-license.org/).

```python tags=["hide-cell"]

```

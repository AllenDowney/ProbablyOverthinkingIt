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

# Field Sobriety Tests and the Base Rate Fallacy

In Chapter 9 of [Probably Overthinking It](https://allendowney.github.io/ProbablyOverthinkingIt/base_rate.html#driving-under-the-influence) I wrote about Drug Recognition Experts (DREs), who are law enforcement officers trained to recognize impaired drivers.

I reviewed the research papers that were supposed to evaluate the accuracy of DREs and I summarized my impressions like this:

> What I found was a collection of studies that are, across the board, deeply flawed. Every one of them features at least one methodological error so blatant it would be embarrassing at a middle school science fair.

Recently the related topic of Field Sobriety Tests (FSTs) came up in this [Reddit discussion](https://www.reddit.com/r/videos/comments/1r5ngps/sober_people_being_arrested_nationwide_for_dui/), which links to [this TV news report](https://www.youtube.com/watch?v=nmSnBOGbOzk) about sober drivers who were arrested based on FST results.


The TV report refers to this [2023 paper in JAMA Psychiatry](https://jamanetwork.com/journals/jamapsychiatry/fullarticle/2807719).
Because it's recent, published in a good quality journal, and called "Evaluation of Field Sobriety Tests for Identifying Drivers Under the Influence of Cannabis: A Randomized Clinical Trial", I thought it might address the problems I found in previous research.

Unfortunately, it has the same problems:

* Selection bias: It excludes as subjects people with conditions that might cause them to fail an FST while sober -- but these are exactly the people most vulnerable to false positive results.

* Wrong metrics: The paper focuses on the true positive and false positive rates, and neglects the predictive value of the test -- which is more relevant to the policy question.

* Unrealistic base rate: In the test conditions, two thirds of the participants were impaired, which is almost certainly higher than the relevant fraction in the real world.

Despite all that, the false positive rate they reported is 49%, which means that **nearly half of the sober participants were wrongly classified as impaired**.

Let's look at each of these problems more closely.


[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/sobriety.ipynb).

```python tags=["hide-cell"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## False Positives

The study tested 184 participants, 121 randomly assigned to the THC group and 63 to the placebo group.
The THC group smoked cannabis cigarettes containing THC; the placebo group smoked cigarettes with almost none.
Each participant was evaluated by one officer, who was "blinded to treatment assignment".
The paper reports

> Officers classified 98 participants (81.0%) in the THC group and 31 (49.2%) in the placebo group as FST impaired.

The following table summarizes these results as a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix):

```python tags=["hide-input"]
TP = 98
FP = 31
FN = 23
TN = 32

confusion_matrix = pd.DataFrame(
    {
        'FST Positive': [TP, FP],
        'FST Negative': [FN, TN]
    },
    index=['THC Group', 'Placebo Group']
)

confusion_matrix['Total'] = confusion_matrix.sum(axis=1)
confusion_matrix.loc['Total'] = confusion_matrix.sum(axis=0)
confusion_matrix
```

Let's start with the most obvious problem: of 63 people in the placebo group, 31 were wrongly classified as impaired, so the false positive rate was 49%.

```python tags=["hide-cell"]
fpr = FP / (FP + TN)
print(f"False positive rate: {fpr:.1%}")
```

Although the tests "were administered by certified DRE instructors (n = 11), the highest training level for impaired driving detection", the results for sober participants were no better than a coin toss.
That's pretty bad, but in reality it's probably worse, because of selection bias. 


## Selection Bias

The study recruited 261 people who met these requirements: "age 21 to 55 years, cannabis use 4 or more times in the past month, holding a valid driver's license, and driving at least 1000 miles in the past year."

But it excluded 62 recruits for reasons including "history of traumatic brain injury [and] significant medical conditions or psychiatric conditions".
They also excluded people with a positive urine test for nonprescription drugs or substance use disorder in the past year.

That's a problem because people with these kinds of medical conditions are more likely to fail an FST -- even if they are not actually impaired.
By excluding them, the study excludes exactly the people most vulnerable to a false positive result.

A better experiment would recruit a representative sample of drivers, including people older than 55 and people with conditions that make it hard to pass a field sobriety test.
The TV report highlights an example: an autistic man who was arrested for DUI because his autism-related differences were mistaken for impairment.
I assume he would have been excluded from the study.


To see how much difference the selection criteria could make, suppose 20 of the excluded participants (about one third) had been assigned to the placebo group.
And suppose that because of their conditions 16 of them were wrongly classified as impaired -- that's 80%, somewhat higher than the rate among included participants.

That would increase the number of false positives by 16 and the number of true negatives by 4, so the unbiased false positive rate might be 57%.

```python tags=["hide-cell"]
unbiased_fpr = (FP + 16) / (FP + 16 + TN + 4)
print(f"False positive rate: {unbiased_fpr:.1%}")
```

This is just a guess: it's not clear how many were excluded specifically for medical conditions or how many of the excluded would have failed the FST.
But this calculation gives us a sense of how big the bias could be.

As I wrote in *Probably Overthinking It*:

> How can you estimate the number of false positives if you exclude from the study everyone likely to yield a false positive? You can't.


And that brings us to the next problem.


## Predictive Value

The paper reports: 

> Officers classified 98 participants (81.0%) in the THC group and 31 (49.2%) in the placebo group as FST impaired at the first evaluation

They quantify this difference as 31.8 percentage points, with 95% CI, 16.4-47.2 percentage points, and report a p-value < .001.
Based on this analysis, they conclude:

> FSTs administered by highly trained law enforcement officers differentiated between individuals receiving THC vs placebo

This conclusion is true in the sense that the difference in percentages is statistically significant, but the policy question is not whether THC exposure changes FST performance under laboratory conditions. **The question is whether an FST result provides sufficiently strong evidence to justify detention or arrest**.

For that, the false positive rate is relevant, and as we have discussed, it is probably more than 50%.

But even more important is the positive predictive value (PPV), which is the probability that a positive test is correct.
In the confusion matrix, there are 129 positive tests, of which 98 are correct and 31 incorrect, so the PPV is 98 out of 129, about 76%.

```python tags=["hide-cell"]
ppv = TP / (TP + FP)
print(f"PPV: {ppv:.1%}")
```

Of the people who failed the FST, 76% were actually impaired.
That might sound good enough for probable cause, but that conclusion is misleading because there is still another problem -- the base rate.


## Base Rate

In the study, two thirds of the participants were impaired.
In the real world, it is unlikely that two thirds of drivers are impaired -- or even two thirds of drivers who take an FST.
So the base rate in the study is too high.

To see why that matters, we have to do a little math.
First we'll use the confusion matrix to compute one more metric, sensitivity, which is the percentage of impaired participants who were classified correctly.

```python tags=["hide-cell"]
sensitivity = TP / (TP + FN)
print(f"Sensitivity: {sensitivity:.1%}")
```

We can use sensitivity, along with the false positive rate we already computed, to figure out the positive predictive value of a test with a more realistic base rate.

Of all people pulled over and given a field sobriety test, how many do you think are impaired by THC?
That's a hard question to answer, so we'll try a couple of values.

First, suppose the base rate is one third, rather than the two thirds in the study.
If we imagine 100 drivers:

* If 33 are impaired, and sensitivity is 81%, we expect 27 true positive results.

* If 67 are not impaired, and the false positive rate is 49%, we expect 33 false positive results.

In that case the positive predictive value is `27 / (27 + 33)`, which means that only 45% of positive tests are correct.
If we put those numbers in a table, the calculation might be clearer.

```python tags=["hide-input"]
col1 = "Tests"
col2 = "Prob pos"
col3 = "Pos tests"
col4 = "Percent"
table = pd.DataFrame(
    index=["Impaired", "Not impaired"], columns=[col1, col2, col3, col4]
)

table[col1] = 33, 67
table[col2] = sensitivity, fpr
table[col3] = table[col1] * table[col2]
total = table[col3].sum()
table[col4] = (table[col3] / total) * 100

table.round(3)
```

With a lower base rate, PPV is lower, which means that a positive test is weaker evidence of impairment.
But even 45% might be too high.

If we suppose that 15% of drivers who take an FST are impaired, we can run the numbers again.

```python tags=["hide-input"]
table = pd.DataFrame(
    index=["Impaired", "Not impaired"], columns=[col1, col2, col3, col4]
)

table[col1] = 15, 85
table[col2] = sensitivity, fpr
table[col3] = table[col1] * table[col2]
total = table[col3].sum()
table[col4] = (table[col3] / total) * 100

table.round(3)
```

With 15% base rate, the predictive value of the test is only 23% -- which means 77% of drivers identified as impaired would actually be sober.

In reality, the base rate depends on the context.
At a checkpoint where every driver is stopped, the base rate might be lower than 15%.
If a driver is stopped for driving erratically, the base rate might be relatively high.
But even then, it is unlikely to be as high as 66%, as in the study.


## Discussion

The JAMA Psychiatry study provides valuable data, but it suffers from the same methodological problems as previous DRE validation studies:

1. **High false positive rate**: Nearly half of sober participants were incorrectly classified as impaired.

2. **Selection bias**: The study excluded exactly the people most likely to be falsely accused, making it impossible to assess the true false positive rate in the general population.

3. **Unrealistic base rate**: The base rate in the study is higher than what we expect in real-world use, which inflates the predictive value of the test.

Although I have been critical of the study, I agree with their interpretation of the results:

> ...the substantial overlap of FST impairment between groups and the high frequency at which FST impairment was suspected to be due to THC suggest that absent other indicators, **FSTs alone may be insufficient to identify THC-specific driving impairment**.

Emphasis mine.


## Notes

In my interpretation of the results, I follow the methodology of the study, which treats assignment to the THC group as ground truth -- that is, we assume that participants in the THC group were actually impaired and participants in the placebo group were not.
And the paper reports:

> Median self-reported highness (scale of 0 to 100, with higher scores indicating more impairment) at 30 minutes was 64 (IQR, 32-76) for the THC group and 13 (IQR, 1-28) for the placebo group (P < .001).

The THC group felt that they were more impaired, but based on the IQRs, it looks like there might be overlap.
That complicates the interpretation of "impaired", but for this analysis I use the study's operational definition.


Copyright 2026 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

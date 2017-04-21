
"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy as np
import pandas as pd

import thinkbayes2
import thinkplot

from collections import Counter

FORMATS = ['pdf', 'eps', 'png']


class SurvivalFunction(object):
    """Represents a survival function."""

    def __init__(self, ts, ss, label=''):
        self.ts = ts
        self.ss = ss
        self.label = label

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, t):
        return self.Prob(t)

    def Prob(self, t):
        """Returns S(t), the probability that corresponds to value t.
        t: time
        returns: float probability
        """
        return np.interp(t, self.ts, self.ss, left=1.0)

    def Probs(self, ts):
        """Gets probabilities for a sequence of values."""
        return np.interp(ts, self.ts, self.ss, left=1.0)

    def Items(self):
        """Sorted list of (t, s) pairs."""
        return zip(self.ts, self.ss)

    def Render(self):
        """Generates a sequence of points suitable for plotting.
        returns: tuple of (sorted times, survival function)
        """
        return self.ts, self.ss

    def MakeHazardFunction(self, label=''):
        """Computes the hazard function.

        This simple version does not take into account the
        spacing between the ts.  If the ts are not equally
        spaced, it is not valid to compare the magnitude of
        the hazard function across different time steps.

        label: string

        returns: HazardFunction object
        """
        lams = pd.Series(index=self.ts)

        prev = 1.0
        for t, s in zip(self.ts, self.ss):
            lams[t] = (prev - s) / prev
            prev = s

        return HazardFunction(lams, label=label)

    def MakePmf(self, filler=None):
        """Makes a PMF of lifetimes.

        filler: value to replace missing values

        returns: Pmf
        """
        cdf = thinkbayes2.Cdf(self.ts, 1-self.ss)
        pmf = thinkbayes2.Pmf()
        for val, prob in cdf.Items():
            pmf.Set(val, prob)

        cutoff = cdf.ps[-1]
        if filler is not None:
            pmf[filler] = 1-cutoff

        return pmf

    def RemainingLifetime(self, filler=None, func=thinkbayes2.Pmf.Mean):
        """Computes remaining lifetime as a function of age.
        func: function from conditional Pmf to expected liftime
        returns: Series that maps from age to remaining lifetime
        """
        pmf = self.MakePmf(filler=filler)
        d = {}
        for t in sorted(pmf.Values())[:-1]:
            pmf[t] = 0
            if pmf.Total():
                pmf.Normalize()
                d[t] = func(pmf) - t

        return pd.Series(d)


def MakeSurvivalFromSeq(values, label=''):
    """Makes a survival function based on a complete dataset.

    values: sequence of observed lifespans
    
    returns: SurvivalFunction
    """
    counter = Counter(values)
    ts, freqs = zip(*sorted(counter.items()))
    ts = np.asarray(ts)
    ps = np.cumsum(freqs, dtype=np.float)
    ps /= ps[-1]
    ss = 1 - ps
    return SurvivalFunction(ts, ss, label)


def MakeSurvivalFromCdf(cdf, label=''):
    """Makes a survival function based on a CDF.

    cdf: Cdf
    
    returns: SurvivalFunction
    """
    ts = cdf.xs
    ss = 1 - cdf.ps
    return SurvivalFunction(ts, ss, label)


class HazardFunction(object):
    """Represents a hazard function."""

    def __init__(self, d, label=''):
        """Initialize the hazard function.

        d: dictionary (or anything that can initialize a series)
        label: string
        """
        self.series = pd.Series(d)
        self.label = label

    def __len__(self):
        return len(self.series)

    def __getitem__(self, t):
        return self.series[t]

    def Get(self, t, default=np.nan):
        return self.series.get(t, default)

    def Render(self):
        """Generates a sequence of points suitable for plotting.

        returns: tuple of (sorted times, hazard function)
        """
        return self.series.index, self.series.values

    def MakeSurvival(self, label=''):
        """Makes the survival function.

        returns: SurvivalFunction
        """
        ts = self.series.index
        ss = (1 - self.series).cumprod()
        sf = SurvivalFunction(ts, ss, label=label)
        return sf

    def Extend(self, other):
        """Extends this hazard function by copying the tail from another.
        other: HazardFunction
        """
        last_index = self.series.index[-1] if len(self) else 0
        more = other.series[other.series.index > last_index]
        self.series = pd.concat([self.series, more])

    def Truncate(self, t):
        """Truncates this hazard function at the given value of t.
        t: number
        """
        self.series = self.series[self.series.index < t]


def ConditionalSurvival(pmf, t0):
    """Computes conditional survival function.

    Probability that duration exceeds t0+t, given that
    duration >= t0.

    pmf: Pmf of durations
    t0: minimum time

    returns: tuple of (ts, conditional survivals)
    """
    cond = thinkbayes2.Pmf()
    for t, p in pmf.Items():
        if t >= t0:
            cond.Set(t-t0, p)
    cond.Normalize()
    return MakeSurvivalFromCdf(cond.MakeCdf())


def PlotConditionalSurvival(durations):
    """Plots conditional survival curves for a range of t0.

    durations: list of durations
    """
    pmf = thinkbayes2.Pmf(durations)
    
    times = [8, 16, 24, 32]
    thinkplot.PrePlot(len(times))

    for t0 in times:
        sf = ConditionalSurvival(pmf, t0)
        label = 't0=%d' % t0
        thinkplot.Plot(sf, label=label)

    thinkplot.Show()


def PlotSurvival(complete):
    """Plots survival and hazard curves.

    complete: list of complete lifetimes
    """
    thinkplot.PrePlot(3, rows=2)

    cdf = thinkbayes2.Cdf(complete, label='cdf')
    sf = MakeSurvivalFromCdf(cdf, label='survival')
    print(cdf[13])
    print(sf[13])

    thinkplot.Plot(sf)
    thinkplot.Cdf(cdf, alpha=0.2)
    thinkplot.Config()

    thinkplot.SubPlot(2)
    hf = sf.MakeHazardFunction(label='hazard')
    print(hf[39])
    thinkplot.Plot(hf)
    thinkplot.Config(ylim=[0, 0.75])


def PlotHazard(complete, ongoing):
    """Plots the hazard function and survival function.

    complete: list of complete lifetimes
    ongoing: list of ongoing lifetimes
    """
    # plot S(t) based on only complete pregnancies
    sf = MakeSurvivalFromSeq(complete)
    thinkplot.Plot(sf, label='old S(t)', alpha=0.1)

    thinkplot.PrePlot(2)

    # plot the hazard function
    hf = EstimateHazardFunction(complete, ongoing)
    thinkplot.Plot(hf, label='lams(t)', alpha=0.5)

    # plot the survival function
    sf = hf.MakeSurvival()

    thinkplot.Plot(sf, label='S(t)')
    thinkplot.Show(xlabel='t (weeks)')


def EstimateHazardFunction(complete, ongoing, label='', verbose=False):
    """Estimates the hazard function by Kaplan-Meier.

    http://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator

    complete: list of complete lifetimes
    ongoing: list of ongoing lifetimes
    label: string
    verbose: whether to display intermediate results
    """
    if np.sum(np.isnan(complete)):
        raise ValueError("complete contains NaNs")
    if np.sum(np.isnan(ongoing)):
        raise ValueError("ongoing contains NaNs")

    hist_complete = Counter(complete)
    hist_ongoing = Counter(ongoing)

    ts = list(hist_complete | hist_ongoing)
    ts.sort()

    at_risk = len(complete) + len(ongoing)

    lams = pd.Series(index=ts)
    for t in ts:
        ended = hist_complete[t]
        censored = hist_ongoing[t]

        lams[t] = ended / at_risk
        if verbose:
            print('%0.3g\t%d\t%d\t%d\t%0.2g' % 
                  (t, at_risk, ended, censored, lams[t]))
        at_risk -= ended + censored

    return HazardFunction(lams, label=label)


def EstimateHazardNumpy(complete, ongoing, label=''):
    """Estimates the hazard function by Kaplan-Meier.

    Just for fun, this is a version that uses NumPy to
    eliminate loops.

    complete: list of complete lifetimes
    ongoing: list of ongoing lifetimes
    label: string
    """
    hist_complete = Counter(complete)
    hist_ongoing = Counter(ongoing)

    ts = set(hist_complete) | set(hist_ongoing)
    at_risk = len(complete) + len(ongoing)

    ended = [hist_complete[t] for t in ts]
    ended_c = np.cumsum(ended)
    censored_c = np.cumsum([hist_ongoing[t] for t in ts])

    not_at_risk = np.roll(ended_c, 1) + np.roll(censored_c, 1)
    not_at_risk[0] = 0

    at_risk_array = at_risk - not_at_risk
    hs = ended / at_risk_array

    lams = dict(zip(ts, hs))

    return HazardFunction(lams, label=label)

from __future__ import print_function, division

import matplotlib.pyplot as plt

import numpy as np
from numpy.fft import fft, ifft


class Pmf:
    
    def __init__(self, d=None):
        """Initializes the distribution.

        d: map from values to probabilities
        """
        self.d = {} if d is None else d

    def items(self):
        """Returns a sequence of (value, prob) pairs."""
        return self.d.items()
    
    def __repr__(self):
        """Returns a string representation of the object."""
        cls = self.__class__.__name__
        return '%s(%s)' % (cls, repr(self.d))

    def __getitem__(self, value):
        """Looks up the probability of a value."""
        return self.d.get(value, 0)

    def __setitem__(self, value, prob):
        """Sets the probability associated with a value."""
        self.d[value] = prob

    def __add__(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf or a scalar

        returns: new Pmf
        """
        if other == 0:
            return self

        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf[v1 + v2] += p1 * p2
        return pmf
    
    __radd__ = __add__

    def total(self):
        """Returns the total of the probabilities."""
        return sum(self.d.values())

    def normalize(self):
        """Normalizes this PMF so the sum of all probs is 1.

        Args:
            fraction: what the total should be after normalization

        Returns: the total probability before normalizing
        """
        total = self.total()
        for x in self.d:
            self.d[x] /= total
        return total
    
    def mean(self):
        """Computes the mean of a PMF."""
        return sum(p * x for x, p in self.items())

    def var(self, mu=None):
        """Computes the variance of a PMF.

        mu: the point around which the variance is computed;
                if omitted, computes the mean
        """
        if mu is None:
            mu = self.mean()

        return sum(p * (x - mu) ** 2 for x, p in self.items())

    def expect(self, func):
        """Computes the expectation of a given function, E[f(x)]

        func: function
        """
        return sum(p * func(x) for x, p in self.items())

    def display(self):
        """Displays the values and probabilities."""
        for value, prob in self.items():
            print(value, prob)
            
    def plot_pmf(self, **options):
        """Plots the values and probabilities."""
        xs, ps = zip(*sorted(self.items()))
        plt.plot(xs, ps, **options)


class Cdf:
    
    def __init__(self, xs, ps):
        self.xs = xs
        self.ps = ps

    def __repr__(self):
        return 'Cdf(%s, %s)' % (repr(self.xs), repr(self.ps))

    def __getitem__(self, x):
        return self.cumprobs([x])[0]
    
    def cumprobs(self, values):
        """Gets probabilities for a sequence of values.

        values: any sequence that can be converted to NumPy array

        returns: NumPy array of cumulative probabilities
        """
        values = np.asarray(values)
        index = np.searchsorted(self.xs, values, side='right')
        ps = self.ps[index-1]
        ps[values < self.xs[0]] = 0.0
        return ps

    def values(self, ps):
        """Returns InverseCDF(p), the value that corresponds to probability p.

        ps: sequence of numbers in the range [0, 1]

        returns: NumPy array of values
        """
        ps = np.asarray(ps)
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError('Probability p must be in range [0, 1]')

        index = np.searchsorted(self.ps, ps, side='left')
        return self.xs[index]
    
    def sample(self, shape):
        """Generates a random sample from the distribution.
        
        shape: dimensions of the resulting NumPy array
        """
        ps = np.random.random(shape)
        return self.values(ps)
    
    def maximum(self, k):
        """Computes the CDF of the maximum of k samples from the distribution."""
        return Cdf(self.xs, self.ps**k)
    
    def display(self):
        """Displays the values and cumulative probabilities."""
        for x, p in zip(self.xs, self.ps):
            print(x, p)
            
    def plot_cdf(self, **options):
        """Plots the cumulative probabilities."""
        plt.plot(self.xs, self.ps, **options)


class CharFunc:
    
    def __init__(self, hs):
        """Initializes the CF.
        
        hs: NumPy array of complex
        """
        self.hs = hs

    def __mul__(self, other):
        """Computes the elementwise product of two CFs."""
        return CharFunc(self.hs * other.hs)
        
    def make_pmf(self, thresh=1e-11):
        """Converts a CF to a PMF.
        
        Values with probabilities below `thresh` are dropped.
        """
        ps = ifft(self.hs)
        d = dict((i, p) for i, p in enumerate(ps.real) if p > thresh)
        return Pmf(d)
    
    def plot_cf(self, **options):
        """Plots the real and imaginary parts of the CF."""
        n = len(self.hs)
        xs = np.arange(-n//2, n//2)
        hs = np.roll(self.hs, len(self.hs) // 2)
        plt.plot(xs, hs.real, label='real', **options)
        plt.plot(xs, hs.imag, label='imag', **options)
        plt.legend()


def compute_cumprobs(d):
    """Computes cumulative probabilities.
    
    d: map from values to probabilities
    """
    xs, freqs = zip(*sorted(d.items()))
    xs = np.asarray(xs)
    ps = np.cumsum(freqs, dtype=np.float)
    ps /= ps[-1]
    return xs, ps

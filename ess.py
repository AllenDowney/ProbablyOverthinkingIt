"""

Copyright 2015 Allen B. Downey
MIT License: http://opensource.org/licenses/MIT
"""

from __future__ import print_function, division

import numpy as np
import pandas as pd

import thinkstats2
import matplotlib.pyplot as plt

import random
import string

import statsmodels.formula.api as smf

from iso_country_codes import COUNTRY
for k, v in COUNTRY.items():
    COUNTRY[k] = v.title()
COUNTRY['RU'] = 'Russia'
COUNTRY['GB'] = 'UK'
COUNTRY['CZ'] = 'Czech Rep'

def get_country(code):
    return COUNTRY[code]

# colors by colorbrewer2.org
BLUE1 = '#a6cee3'
BLUE2 = '#1f78b4'
GREEN1 = '#b2df8a'
GREEN2 = '#33a02c'
PINK = '#fb9a99'
RED = '#e31a1c'
ORANGE1 = '#fdbf6f'
ORANGE2 = '#ff7f00'
PURPLE1 = '#cab2d6'
PURPLE2 = '#6a3d9a'
YELLOW = '#ffff99'
BROWN = '#b15928'


def country_name(code):
    return COUNTRY[code]


def read_cycle(filename):
    """Reads a file containing ESS data and selects columns.
    
    filename: string
    
    returns: DataFrame
    """ 
    df = pd.read_stata(filename, convert_categoricals=False)

    if 'hinctnta' not in df.columns:
        df['hinctnta'] = df.hinctnt
        
    if 'inwyr' not in df.columns:
        df['inwyr'] = df.inwyye
        
    cols = ['cntry', 'inwyr', 'tvtot', 'tvpol', 'rdtot', 'rdpol', 
            'nwsptot', 'nwsppol', 'netuse', 
            'rlgblg', 'rlgdgr', 'eduyrs', 'hinctnta', 'yrbrn', 
            'eisced', 'pspwght', 'pweight']
    df = df[cols]
    return df


def read_all_cycles():
    filenames = ['ESS1e06_4.dta', 'ESS2e03_4.dta', 'ESS3e03_5.dta', 
                 'ESS4e04_3.dta', 'ESS5e03_2.dta']

    cycles = [read_cycle(filename) for filename in filenames]
    return cycles


def clean_cycle(df):
    """Cleans data from one cycle.
    
    df: DataFrame
    """
    df.tvtot.replace([77, 88, 99], np.nan, inplace=True)
    df.rdtot.replace([77, 88, 99], np.nan, inplace=True)
    df.nwsptot.replace([77, 88, 99], np.nan, inplace=True)
    df.netuse.replace([77, 88, 99], np.nan, inplace=True)
    df.tvpol.replace([66, 77, 88, 99], np.nan, inplace=True)
    df.rdpol.replace([66, 77, 88, 99], np.nan, inplace=True)
    df.nwsppol.replace([66, 77, 88, 99], np.nan, inplace=True)
    df.eduyrs.replace([77, 88, 99], np.nan, inplace=True)
    df.rlgblg.replace([7, 8, 9], np.nan, inplace=True)
    df.rlgdgr.replace([77, 88, 99], np.nan, inplace=True)
    df.hinctnta.replace([77, 88, 99], np.nan, inplace=True)
    df.yrbrn.replace([7777, 8888, 9999], np.nan, inplace=True)
    df.inwyr.replace([9999], np.nan, inplace=True)
    
    df['hasrelig'] = (df.rlgblg==1).astype(int)
    df.loc[df.rlgblg.isnull(), 'hasrelig'] = np.nan
    
    df['yrbrn60'] = df.yrbrn - 1960
    df['inwyr07'] = df.inwyr - 2007 + np.random.uniform(-0.5, 0.5, len(df))



def resample(df):
    """Resample data by country.
    
    df: DataFrame
    
    returns: map from country code to DataFrame
    """
    res = {}
    grouped = df.groupby('cntry')
    for code, group in grouped:
        sample = group.sample(len(group), weights=group.pspwght, replace=True)
        sample.index = range(len(group))
        res[code] = sample
    return res


def check_variables(code, group):
    """Print variables missing from a group.
    
    code: group code (country code)
    group: DataFrame
    """
    varnames = ['cntry', 'tvtot', 'tvpol', 'rdtot', 'rdpol', 
                'nwsptot', 'nwsppol', 'netuse', 'inwyr07', 
                'rlgblg', 'rlgdgr', 'eduyrs', 'hinctnta', 
                'yrbrn', 'pspwght', 'pweight']
    for var in varnames:
        n = len(group[var].dropna())
        if (n < 100):
            print(code, var, len(group[var].dropna()))


def remove_missing(cycle_maps):
    """Cleans up some problems with missing data.

    cycle_maps: list of maps from country code to DataFrame
    """
    del cycle_maps[0]['FR']
    del cycle_maps[0]['DE']
    del cycle_maps[1]['FR']
    del cycle_maps[1]['FI']

    ee = cycle_maps[4]['EE']
    ee.inwyr07 = 3 + np.random.uniform(-0.5, 0.5, len(ee))


def replace_var_with_rank(code, df, old, new):
    """Replaces a scale variable with a rank from 0-1.
    
    Creates a new column.
    
    code: country code
    df: DataFrame
    old: old variable name
    new: new variable name
    """
    # jitter the data
    series = df[old] + np.random.uniform(-0.25, 0.25, len(df))
    
    # if there's no data, just put in random values
    if len(series.dropna()) < 10:
        #print(name, old)
        df[new] = np.random.random(len(df))
        return
    
    # map from values to ranks
    cdf = thinkstats2.Cdf(series)
    df[new] = cdf.Probs(series)
    
    # make sure NaN maps to NaN
    df.loc[df[old].isnull(), new] = np.nan
    
    
def replace_with_ranks(cycle_map):
    """Replace variables within countries.
    
    cycle_map: map from country code to DataFrame
    """
    for code, group in cycle_map.items():
        replace_var_with_rank(code, group, 'hinctnta', 'hincrank')
        replace_var_with_rank(code, group, 'eduyrs', 'edurank')
        

def fill_var(df, old, new):
    """Fills missing values.
    
    Creates a new column
    
    df: DataFrame
    old: old variable name
    new: new variable name
    """
    # find the NaN rows
    null = df[df[old].isnull()]
    
    # sample from the non-NaN rows
    fill = df[old].dropna().sample(len(null), replace=True)
    fill.index = null.index
    
    # replace NaNs with the random sample
    df[new] = df[old].fillna(fill)
    

OLD_NAMES = ['hasrelig', 'rlgdgr', 'yrbrn60', 'edurank', 'hincrank',
             'tvtot', 'rdtot', 'nwsptot', 'netuse', 'inwyr07']
    
NEW_NAMES = [old_name + '_f' for old_name in OLD_NAMES]


def fill_vars_by_country(cycle_map):
    for code, group in cycle_map.items():
        [fill_var(group, old, new) 
         for old, new in zip(OLD_NAMES, NEW_NAMES)]
        

def concat_groups(cycle_map):
    """Concat all countries in a cycle.
    
    cycle_map: map from country code to DataFrame
    
    returns: DataFrame
    """
    return pd.concat(cycle_map.values(), ignore_index=True)


def run_model(df, formula):
    model = smf.logit(formula, data=df)    
    results = model.fit(disp=False)
    return results


def extract_res(res, var='netuse_f'):
    param = res.params[var]
    pvalue = res.pvalues[var]
    stars = '**' if pvalue < 0.01 else '*' if pvalue < 0.05 else ''
    return res.nobs, param, stars


def run_logits(grouped, formula, var):
    for code, group in grouped:
        country = get_country(code).ljust(14)
        model = smf.logit(formula, data=group)    
        results = model.fit(disp=False)
        nobs, param, stars = extract_res(results, var=var)
        arrow = '<--' if stars and param > 0 else ''
        print(country, nobs, '%0.3g'%param, stars, arrow, sep='\t')


def run_ols(grouped, formula, var):
    for code, group in grouped:
        model = smf.ols(formula, data=group)    
        results = model.fit(disp=False)
        nobs, param, stars = extract_res(results, var=var)
        arrow = '<--' if stars and param > 0 else ''
        print(code, len(group), '%0.3g    '%param, stars, arrow, sep='\t')


def read_and_clean():
    cycles = read_all_cycles()
    for cycle in cycles:
        clean_cycle(cycle)
    return cycles


def resample_and_fill(cycles):
    # each cycle_map is a map from country code to DataFrame
    cycle_maps = [resample(cycle) for cycle in cycles]

    remove_missing(cycle_maps)

    for i, cycle_map in enumerate(cycle_maps):
        replace_with_ranks(cycle_map)

    for i, cycle_map in enumerate(cycle_maps):
        fill_vars_by_country(cycle_map)

    dfs = [concat_groups(cycle_map) for cycle_map in cycle_maps]
    df = pd.concat(dfs, ignore_index=True)
    return df


def random_name():
    """Generates a random string of letters.

    returns: string
    """
    t = [random.choice(string.ascii_letters) for i in range(6)]
    return ''.join(t)


def add_frames(store, cycles, num):
    """Generates filled resamples and put them in the store.

    store: h5 store object
    cycles: list of DataFrames
    num: how many resamples to generate
    """
    for i in range(num):
        name = random_name()
        print(name)
        df = resample_and_fill(cycles)
        store.put(name, df)


class Country:
    def __init__(self, code, nobs):
        self.code = code
        self.name = country_name(code)
        self.nobs = nobs
        self.mean_seq = []
        self.param_seq = []
        self.param2_seq = []
        self.range_seq = []
        self.range2_seq = []

    def add_mean(self, means):
        self.mean_seq.append(means)
        
    def add_params(self, params):
        self.param_seq.append(params)
        
    def add_params2(self, params):
        self.param2_seq.append(params)
        
    def add_ranges(self, ranges):
        self.range_seq.append(ranges)
        
    def add_ranges2(self, ranges):
        self.range2_seq.append(ranges)
        
    def get_means(self, varname):
        t = [mean[varname] for mean in self.mean_seq]
        return np.array(t)

    def get_params(self, varname):
        t = [params[varname] for params in self.param_seq]
        return np.array(t)

    def get_params2(self, varname):
        t = [params[varname] for params in self.param2_seq]
        return np.array(t)

    def get_ranges(self, varname):
        t = [ranges[varname] for ranges in self.range_seq]
        return np.array(t)

    def get_ranges2(self, varname):
        t = [ranges[varname] for ranges in self.range2_seq]
        return np.array(t)


def process_frame(df, country_map, reg_func, formula, model_num):
    """Processes one frame.
    
    df: DataFrame
    country_map: map from code to Country
    reg_func: function used to compute regression
    formula: string Patsy formula
    model_num: which model we're running
    """
    grouped = df.groupby('cntry')
    for code, group in grouped:
        country = country_map[code]
        country.add_mean(group.mean())

        # run the model
        model = reg_func(formula, data=group)    
        results = model.fit(disp=False)

        # extract parameters and range of effect sizes
        if model_num == 1:
            country.add_params(results.params)
            add_ranges(country, group, results)
        else:
            country.add_params2(results.params)
            add_ranges2(country, group, results)


def process_all_frames(store, country_map, num, 
                       reg_func, formula, model_num):
    """Loops through the store and processes frames.
    
    store: store
    country_map: map from code to Country
    num: how many resamplings to process
    reg_func: function used to compute regression
    formula: string Patsy formula
    model_num: which model we're running
    """
    for i, key in enumerate(store.keys()):
        if i >= num:
            break
        print(i, key)
        df = store.get(key)
        df['yrbrn60_f2'] = df.yrbrn60_f ** 2
        process_frame(df, country_map, reg_func, formula, model_num)
        

def extract_params(country_map, param_func, varname=None):
    """Extracts parameters.
    
    country_map: map from country code to Country
    param_func: function that takes country and returns param list
    varname: name of variable to get the mean of
    
    returns: list of (code, name, param, low, high, mean) tuple
    """
    t = []
    for code, country in sorted(country_map.items()):
        name = country.name

        params = param_func(country)
        param = np.median(params)
        low = np.percentile(params, 2.5)
        high = np.percentile(params, 97.5)
    
        if varname is not None:
            means = country.get_means(varname)
            mean = np.median(means)
        else:
            mean = np.nan
    
        t.append((code, name, param, low, high, mean))
    
    t.sort(key=lambda x: x[2])
    return t


def extract_vars(country_map, exp_var, dep_var):
    def param_func(country):
        return country.get_params(exp_var)

    t = extract_params(country_map, param_func, dep_var)
    return t


def extract_vars2(country_map, exp_var, dep_var):
    def param_func(country):
        return country.get_params2(exp_var)

    t = extract_params(country_map, param_func, dep_var)
    return t


def plot_params(params, ys, codes, color):
    """Plots parameters using country codes on top of white squares.

    t: list of (code, name, param, low, high, mean)
    color: string
    hlines: whether to plot lines for the confidence intervals
    """

    # plot white squares
    plt.plot(params, ys, 'ws', markeredgewidth=0, markersize=15)

    # plot codes as text
    for param, y, code in zip(params, ys, codes):
        plt.text(param, y, code, fontsize=10, color=color, 
                 horizontalalignment='center',
                 verticalalignment='center')

    return ys


def plot_cis(t, color='blue'):
    """Plots confidence intervals.

    t: list of (code, name, param, low, high, mean)
    color: string
    """
    plt.figure(figsize=(8, 8))
    n = len(t)
    ys = np.arange(1, n+1, dtype=float)
    codes, names, params, lows, highs, means = zip(*t)

    # plot confidence intervals
    plt.hlines(ys, lows, highs, color=color, linewidth=2)
    plot_params(params, ys, codes, color)
    plt.vlines(0, 0, n+1, color='gray', alpha=0.5)
    plt.yticks(ys, names)


STYLE_MAP = {}
STYLE_MAP['inwyr07_f'] = (GREEN1, 'year asked')
STYLE_MAP['yrbrn60_f'] = (GREEN2, 'year born')
STYLE_MAP['hincrank_f'] = (ORANGE1, 'income')
STYLE_MAP['edurank_f'] = (ORANGE2, 'education')
STYLE_MAP['tvtot_f'] = (RED, 'television')
STYLE_MAP['rdtot_f'] = (BLUE1, 'radio')
STYLE_MAP['nwsptot_f'] = (BLUE2, 'newspaper')
STYLE_MAP['netuse_f'] = (PURPLE2, 'Internet')
STYLE_MAP['delta'] = (PURPLE2, 'Internet')
STYLE_MAP['hasrelig_f'] = (BROWN, 'affiliation')
STYLE_MAP['rlgdgr_f'] = (BROWN, 'religiosity')


def plot_cdfs(country_map, extract_func, cdfnames):
    """Plots cdfs for estimated parameters or ranges.

    country_map: map from code to Country
    extract_func: function to extract params or ranges
    cdfnames: list of string variable names to plot cdfs of
    """
    def extract(exp_var):
        t = extract_func(country_map, exp_var, None)
        t.sort(key=lambda x: x[2])
        return t

    def plot(t, color, label):
        n = len(t)
        ys = np.arange(1, n+1, dtype=float)
        codes, names, params, lows, highs, means = zip(*t)
    
        cdf = thinkstats2.Cdf(params)
        print(cdf.Mean(), cdf.Percentile(50))
        plt.plot(cdf.xs, cdf.ps, label=label,
                 linewidth=3,
                 color=color, alpha=0.6)
        
        # it's possible to plot the country codes on top of the CDFs,
        # but turns out not to look so great
        # plot_params(params, ys, codes, color)

    plt.figure(figsize=(8, 8))

    for varname in cdfnames:
        t = extract(varname)
        color, label = STYLE_MAP[varname]
        ys = plot(t, color, label)

    plt.vlines(0, 0, 1, color='gray', linewidth=2, alpha=0.4)


def plot_scatter(t, color='blue'):
    """Makes a scatter plot.

    t: list of (code, name, param, low, high, mean)
    color: string
    factor: what to multiply the parameter by
    """
    plt.figure(figsize=(8, 8))

    codes, names, params, lows, highs, means = zip(*t)

    for param, mean, code in zip(params, means, codes):
        plt.text(param, mean, code, fontsize=10, color=color, 
                 horizontalalignment='center',
                 verticalalignment='center')
        
    corr = np.corrcoef(params[2:-2], means[2:-2])[0][1]
    print(corr)


def make_countries(store):
    keys = store.keys()
    key = random.choice(keys)
    df = store.get(key)

    grouped = df.groupby('cntry')
    country_map = {}

    for code, group in grouped:
        country_map[code] = Country(code, len(group))
        print(country_map[code].name)
        
    return country_map


class Range():
    __slots__ = ['low', 'middle', 'high', 'width']

    def __init__(self, *args):
        self.__dict__.update(zip(self.__slots__, args))


def compute_range(country, group, results, varname):
    """Computes the range in the dependent variable.
    
    country: Country object
    group: DataFrame
    results: regression results
    varname: explanatory variable
    
    returns: Range object
    """
    def logistic(results):
        return hasattr(results, 'prsquared')

    def predict(results, df):
        pred = results.predict(df)[0]

        # if the prediction is from logistic regression, multiply
        # by 100 to get percentage points
        if logistic(results):
            pred *= 100

        return pred

    def set_to_percentile(df, varname, percentile):
        val = cdf.Percentile(percentile)
        df[varname] = val

        # when you vary yrbrn60_f, you have to vary yrbrn60_f2
        # at the same time
        if varname == 'yrbrn60_f':
            set_to_percentile(df, 'yrbrn60_f2', percentile)

    # start with all values set to their mean
    df = group.mean()
    middle = predict(results, df)

    cdf = thinkstats2.Cdf(group[varname])

    # change one variable to its 25th percentile
    set_to_percentile(df, varname, 25)
    low = predict(results, df)
    
    # change to the 75th percentile
    set_to_percentile(df, varname, 75)
    high = predict(results, df)

    if logistic(results):
        width = high-low
    else:
        # compute width in terms of standard deviatons
        #std = np.std(results.model.endog)
        #width = (high - low) / std
        width = high-low

    return Range(low, middle, high, width)


def add_ranges(country, group, results):
    """Adds model 1 ranges for each variable to the country object.
    
    country: Country object
    group: DataFrame
    results: regression results    
    """
    ranges = {}
    for varname in results.params.index:
        if varname in ['Intercept', 'yrbrn60_f2']:
            continue
        ranges[varname] = compute_range(country, group, 
                                        results, varname)
    
    country.add_ranges(ranges)

    
def add_ranges2(country, group, results):
    """Adds model2 ranges for each variable to the country object.
    
    country: Country object
    group: DataFrame
    results: regression results    
    """
    ranges = {}
    for varname in results.params.index:
        if varname == 'Intercept':
            continue
        ranges[varname] = compute_range(country, group, 
                                        results, varname)
    
    country.add_ranges2(ranges)


def extract_ranges(country_map, exp_var, dep_var):
    """Extracts results for plotting.
    
    country_map: map from code to Country object
    exp_var: string
    dep_var: string
    
    returns: list of (codes, names, params, lows, highs, means)
    """
    def param_func(country):
        ranges = country.get_ranges(exp_var)
        widths = [r.width for r in ranges]
        return widths
    
    t = extract_params(country_map, param_func, dep_var)
    return t


def extract_ranges2(country_map, exp_var, dep_var):
    def param_func(country):
        ranges = country.get_ranges2(exp_var)
        widths = [r.width for r in ranges]
        return widths
    
    t = extract_params(country_map, param_func, dep_var)
    return t


def classify_countries(country_map, varname, extract_func):
    t = extract_func(country_map, varname, None)
    codes, names, params, lows, highs, means = zip(*t)
    signs = np.sign(params)
    sigs = np.sign(np.array(lows) * np.array(highs))
    d = {}
    for sign in [-1, 1]:
        for sig in [-1, 1]:
            d[sign, sig] = sum((signs==sign) & (sigs==sig))
    d[1, -1] += sum(signs==0)
    return d


def make_table(country_map, varnames, extract_func):
    keys = [(-1, 1), (-1, -1), (1, -1), (1, 1)]
    ts = []
    for varname in varnames:
        d = classify_countries(country_map, varname, extract_func)
        t = [varname]
        t.extend([d[key] for key in keys])
        t.append(sum(d.values()))
        ts.append(t)
    
    ts.sort(key=lambda t: t[1], reverse=True)
    return ts


def print_table(ts, sep='  \t', end='\n'):
    print('varname', 'neg*', 'neg', 'pos', 'pos*',
          sep=sep, end=end)
    print('---------', '----', '---', '---', '----', 
          sep=sep, end=end)
    for t in ts:
        print(*t, sep=sep, end=end)


def main():
    cycles = read_and_clean()
    store = pd.HDFStore('ess.resamples.h5')
    print(store)
    add_frames(store, cycles, n=101)


if __name__ == '__main__':
    main()

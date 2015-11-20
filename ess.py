"""

Copyright 2015 Allen B. Downey
MIT License: http://opensource.org/licenses/MIT
"""

from __future__ import print_function, division

import numpy as np
import pandas as pd

import thinkstats2
import thinkplot
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from iso_country_codes import COUNTRY
for k, v in COUNTRY.items():
    COUNTRY[k] = v.title()
COUNTRY['RU'] = 'Russia'


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



def run_all_models():
    formula = ('hasrelig_f ~ inwyr07_f + yrbrn60_f + edurank_f + hincrank_f +'
               'tvtot_f + rdtot_f + nwsptot_f + netuse_f')
    res = run_model(df, formula)
    res.summary()

    formula = ('hasrelig_f ~ inwyr07_f + yrbrn60_f + edurank_f + hincrank_f +'
               'tvtot_f + rdtot_f + nwsptot_f + netuse_f')

    run_logits(grouped, formula, 'netuse_f')


    # In[41]:

    run_logits(grouped, formula, 'hincrank_f')


    # In[42]:

    run_logits(grouped, formula, 'edurank_f')

    formula = ('rlgdgr_f ~ inwyr07_f + yrbrn60_f + edurank_f + hincrank_f +'
               'tvtot_f + rdtot_f + nwsptot_f + netuse_f')



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
    t = [random.choice(string.ascii_letters) for i in range(6)]
    return ''.join(t)


def add_frames(store, cycles, n):
    for i in range(n):
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

    def add_mean(self, means):
        self.mean_seq.append(means)
        
    def add_params(self, params):
        self.param_seq.append(params)
        
    def add_params2(self, params):
        self.param2_seq.append(params)
        
    def get_means(self, varname):
        t = [mean[varname] for mean in self.mean_seq]
        return np.array(t)

    def get_params(self, varname):
        t = [params[varname] for params in self.param_seq]
        return np.array(t)

    def get_params2(self, varname):
        t = [params[varname] for params in self.param2_seq]
        return np.array(t)


def make_country_map():
    keys = store.keys()
    key = random.choice(keys)
    df = store.get(key)

    grouped = df.groupby('cntry')
    country_map = {}

    for code, group in grouped:
        country_map[code] = Country(code, len(group))
        print(country_map[code].name)

    return country_map


formula1 = ('hasrelig_f ~ inwyr07_f + yrbrn60_f + '
            'edurank_f + hincrank_f +'
            'tvtot_f + rdtot_f + nwsptot_f + netuse_f')

formula2 = ('rlgdgr_f ~ inwyr07_f + yrbrn60_f + '
            'edurank_f + hincrank_f +'
            'tvtot_f + rdtot_f + nwsptot_f + netuse_f')

def process_frame(df):
    grouped = df.groupby('cntry')
    for code, group in grouped:
        country = country_map[code]
        country.add_mean(group.mean())
    
        model = smf.logit(formula1, data=group)    
        results = model.fit(disp=False)
        country.add_params(results.params)
        
        model = smf.ols(formula2, data=group)    
        results = model.fit(disp=False)
        country.add_params2(results.params)


def process_all_frames():

    for key in store.keys():
        print(key)
        df = store.get(key)
        process_frame(df)


def extract_params(country_map, param_func, varname):
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
    
        means = country.get_means(varname)
        mean = np.median(means)
    
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


def plot_cis(t):
    plt.figure(figsize=(8,8))

    n = len(t)
    ys = n - np.arange(n)
    codes, names, params, lows, highs, means = zip(*t)
    plt.hlines(ys, lows, highs, color='blue', linewidth=2, alpha=0.5)
    plt.plot(params, ys, 'ws', markeredgewidth=0, markersize=15)

    for param, y, code in zip(params, ys, codes):
        plt.text(param, y, code, fontsize=10, color='blue', 
                 horizontalalignment='center',
                 verticalalignment='center')

    plt.vlines(0, 0, n+1, color='gray', alpha=0.5)
    plt.yticks(ys, names)


def plot_scatter(t):
    plt.figure(figsize=(8,8))

    codes, names, params, lows, highs, means = zip(*t)

    for param, mean, code in zip(params, means, codes):
        plt.text(param, mean, code, fontsize=10, color='blue', 
                 horizontalalignment='center',
                 verticalalignment='center')
        
    corr = np.corrcoef(params, means)[0][1]
    print(corr)




def main():
    read_and_clean()


if __name__ == '__main__':
    main()

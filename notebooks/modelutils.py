'''
modelutils.py

A small collection of functions for inspecting, diagnosing, and interpreting models in PyMC3.
Inspired in part by functionality implemented by quap and precis from the `rethinking` R package by Richard McElreath.
Includes functions to extract normal (Laplace) approximations to the maximum a posteriori estimates, and to plot intervals for comparison
across several parameters.
'''

import pymc3 as pm, pandas as pd, numpy as np, matplotlib.pyplot as plt, scipy as sp
from scipy import stats

def make_normal_approx(vars):
    '''
    Gets the normal approximation for the posterior near its maximum for the specified variables.
    Useful for getting quick summaries without sampling from, e.g. linear models.
    Don't use for complex models!
    Parameters:
        vars: a list of variable names
    Returns:
        a DataFrame with the MAP estimates, standard deviations, and compatible intervals estimated from the Hessian
    '''
    map_est = pm.find_MAP()
    std_est = (1/pm.find_hessian(map_est, vars=vars))**0.5
    data = []
    for var in vars:
        i = vars.index(var)
        cv = sp.stats.norm.ppf(0.97)
        data.append([map_est[var.name].round(3), std_est[i,i].round(3), (map_est[var.name] - std_est[i, i] * cv).round(3), (map_est[var.name] + std_est[i, i] * cv).round(3)])
    summary = pd.DataFrame(data, columns = ['map', 'sd', 'hdi_3%', 'hdi_97%'], index = vars)
    return summary

def plot_estimates(sumtab, vars = None):
    '''
    Makes a "nice" plot of the summary table from a PyMC3 sample trace. Plots posterior means and highest density intervals for parameters.
    If vars is None then it plots all of them..
    
    Parameters:
        sumtab: a summary table gotten from calling pm.summary on a trace
        vars: a list of variable names
    
    Returns: None
    '''
    
    lowest = min(sumtab['hdi_3%'])
    lowest -= 0.1*np.abs(lowest)
    highest = max(sumtab['hdi_97%'])
    highest += 0.1*np.abs(highest)
    fig = plt.figure(figsize=(10, 0.5 * len(sumtab)))
    plt.xlim(lowest, highest)
    #idx = range(len(sumtab)-1,-1,-1)
    idx = range(len(sumtab))
    if 'mean' in sumtab.columns:
        plt.plot(sumtab['mean'], idx, 'o', color = 'black')
    if 'map' in sumtab.columns:
        plt.plot(sumtab['map'], idx, 'o', color = 'black')
    for i in range(len(sumtab)-1,-1,-1):
        plt.hlines(y=i, xmin=sumtab['hdi_3%'][i], xmax=sumtab['hdi_97%'][i], color = 'gray')
    plt.yticks(ticks=idx, labels=sumtab.index)
    plt.grid(axis='x')
    plt.show()
'''
modelutils.py

A small collection of functions for inspecting, diagnosing, and interpreting models in PyMC3.
Inspired in part by functionality implemented by quap and precis from the `rethinking` R package by Richard McElreath.

'''

import pymc3 as pm
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats

class ModelQuap:
    def __init__(self, var_list, map_est, cov_est):
        self.var_names = [var.name for var in var_list]
        self.mode = map_est
        self.cov = cov_est
        index = []
        data = []
        i = 0
        for var in var_list:
            if not var.distribution.shape:
                #i = var_list.index(var)
                cv = sp.stats.norm.ppf(0.97)
                data.append([np.float64(map_est[var.name]), cov_est[i,i] ** 0.5,
                            (map_est[var.name] - cv * cov_est[i, i] ** 0.5),
                            (map_est[var.name] + cv * cov_est[i, i] ** 0.5)])
                i += 1
                index.append(var.name)
            else:
                for j in range(var.distribution.shape[0]):
                    cv = sp.stats.norm.ppf(0.97)
                    mode = map_est[var.name][j]
                    error = cv * cov_est[i, i] ** 0.5
                    data.append([mode, cov_est[i, i] ** 0.5,
                               mode - error, mode + error])
                    i += 1
                    index.append(var.name + '[' + str(j) + ']')
        self.summary_table = pd.DataFrame(data, 
                                          columns=['mean', 'sd', 'hdi_3%', 'hdi_97%'],
                                          index=index)

    def summary(self):
        return self.summary_table.round(3)

    def get_mode(self, var_names = None):
        if var_names is not None:
            return {var:self.mode[var] for var in var_names}
        else:
            return {var:self.mode[var] for var in self.var_names}

    def get_cov(self, var_names = None):
        return self.cov

    def extract_samples(self, n = 1000):
        mean = np.array([self.mode[name] for name in self.var_names])
        return sp.stats.multivariate_normal(mean=mean, cov=self.cov).rvs(n)

    def plot_forest(self, var_names = None, figsize = None, scale = 1):
        if var_names is None:
            var_names = self.summary_table.index

        if figsize is None:
            figsize_x = 5
            figsize_y = 0.8 * len(var_names)
            figsize = (scale * figsize_x, scale * figsize_y)

        plt.figure(figsize=figsize)
        plt.scatter(self.summary_table.loc[:, 'mean'], range(len(var_names)), edgecolors='k', facecolors='none')

        ax = plt.gca()
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i in range(len(var_names)):
            plt.hlines(y=i,
                       xmin=self.summary_table.loc[var_names[i], 'hdi_3%'],
                       xmax=self.summary_table.loc[var_names[i], 'hdi_97%'],
                       color='k',
                       ls='-',
                       lw=1)
            plt.plot([self.summary_table.loc[var_names[i], 'hdi_3%'], self.summary_table.loc[var_names[i], 'hdi_3%']], [i+0.02, i-0.02], 'k-', lw=1)
            plt.plot([self.summary_table.loc[var_names[i], 'hdi_97%'], self.summary_table.loc[var_names[i], 'hdi_97%']], [i+0.02, i-0.02], 'k-', lw=1)
        plt.title('94% HDI')
        plt.yticks(ticks=range(len(var_names)), labels=var_names, size=12)
        plt.xticks(size=12)
        plt.show()


def quap(var_list = None):
    '''
    Gets the quadratic approximation for the posterior near its maximum for the specified variables.
    Use inside a model context as you would pm.sample().

    Parameters:
        var_list: a list of variable names. If this is not set, use the full list of free RVs.
    Returns:
        a ModelQuap object
    '''
    if var_list is None:
        var_list = pm.Model.get_context().unobserved_RVs
        for var in var_list:
            if var.name.endswith('__'):
                var_list.remove(var)
    map_est = pm.find_MAP()
    cov_est = np.linalg.inv(pm.find_hessian(map_est, vars = var_list))
    approx = ModelQuap(var_list, map_est, cov_est)
    return approx
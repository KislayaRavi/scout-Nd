"""Gradient-estimator drift / convergence-rate analysis.

For the square (and cubic) target function, plots the error
``|dU/d(mu) - df/dx|`` vs the noise variance ``sigma^2`` on a log-log scale,
fits a slope, and saves ``drift_plot_square_fn.pdf`` (and ``drift_plot_cubic_fn.pdf``).
"""

import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
from objective_function import *
# import seaborn as sns
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20

rc('font', size=MEDIUM_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure titles


def target_function(x):
    return np.sum(x**2, axis=1)

def get_derivative(function, sigma):
    dim = 1
    constraints = None
    obj = Baseline1(dim, function, constraints, num_samples=512, qmc=True, correct_constraint_derivative=True)
    x = np.zeros(2*dim)
    x[dim:] = sigma*np.ones(dim)
    _, grad = obj.function_wrapper(torch.tensor(x, dtype=torch.float64))
    print(grad)
    return grad[:dim]

if __name__ == '__main__':
    sigmas = np.array([0, -4, -8, -12])
    seeds = np.arange(6)
    actual_derivative = 0
    list_outer = []
    for sigma in sigmas:
        list_inner = []
        for seed in seeds:
            np.random.seed(seed)
            derivative = get_derivative(target_function, sigma)
            difference = np.abs(derivative - actual_derivative)
            list_inner.append(difference[0])
        list_outer.append(list_inner)
        # print(f'Sigma: {np.exp(sigma)}, derivative: {derivative}, difference: {difference}')
    # print(len(list_outer), len(np.exp(sigmas)))
    plt.errorbar(np.exp(sigmas), np.mean(list_outer, axis=1), yerr=np.std(list_outer, axis=1), capsize=5)
    X = np.log10(np.exp(sigmas))
    Y = np.log10(np.mean(list_outer, axis=1))
    slope, intercept = np.polyfit(X,Y,1)
    print("Slope and intercept: ", slope, intercept)
    plt.xlabel(r'$\sigma^2$')
    plt.ylabel(r'$|\frac{\partial U}{\partial \mu} - \frac{\partial f}{\partial x}|$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('drift_plot_square_fn.pdf', bbox_inches='tight')
    plt.show()
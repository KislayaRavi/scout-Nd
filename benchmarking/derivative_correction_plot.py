"""Visualise the constraint-derivative correction smoothing effect for different noise variances.

Produces ``derivative_correction.pdf``: the (max(C(x), 0)) constraint together with its
Gaussian-smoothed versions for several values of ``sigma^2``.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scoutNd.objective_function import Baseline1
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# plt.rcParams["mathtext.fontset"] = "cm"
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 20


rc('font', size=BIGGER_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

def function(x):
    X = x[:, None]
    Y = 1 - X 
    Y[X>1] = 0.
    return Y

def plot_convolution(phi):
    num_points = 100
    x = np.linspace(0, 2, num_points)
    obj = Baseline1(1, function, [], num_samples=256, qmc=True, correct_constraint_derivative=True)
    X, Y = np.zeros(2), np.zeros(num_points)
    for i in range(num_points):
        X[0] = x[i]
        X[1] = phi
        X = torch.tensor(X, dtype=torch.float32)
        Y[i], _ = obj.function_wrapper(X)
    plt.plot(x, Y, label='$\sigma^2= e^{'+ str(phi) + '}$')


if __name__ == '__main__':
    x = np.linspace(0, 2, 100)
    y = function(x)
    plt.plot(x, y, label=r'$\max(\mathcal{C}(x), 0)$')
    plot_convolution(-1)
    plot_convolution(-2)
    plot_convolution(-3)
    plt.legend()
    plt.xlim([0, 2])
    plt.xlabel('$x$')
    plt.ylabel('Objective value')
    plt.legend(bbox_to_anchor=(0, 1.02,1, 0.2), loc="lower left", ncol=2, mode='expand')
    plt.grid()
    plt.savefig('derivative_correction.pdf', bbox_inches='tight')
    plt.show()
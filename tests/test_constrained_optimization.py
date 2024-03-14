import numpy as np
import torch
from scoutNd.objective_function import *
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os
import time
datetime = time.strftime("%Y%m%d-%H%M%S")

from scoutNd.stochastic_optimizer import Stochastic_Optimizer
from scoutNd.objective_function import *


def sphere(x):
    X = np.atleast_2d(x)
    val1 = np.sum(X**2, axis=1)
    # val2 = np.random.normal(0, 0.0001, val1.shape)
    val2 = 0
    return val1 + val2

def linear_constraint(X):
    x = np.atleast_2d(X)
    return 1 - x[:, 0] - x[:, 1]

if __name__ == '__main__':
    dim = 32
    #constraints = [linear_constraint]
    # constraints = None
    obj = Baseline1(dim, sphere,constraints=[linear_constraint], num_samples=64, qmc=True, correct_constraint_derivative=True)
    optimizer = Stochastic_Optimizer(obj, natural_gradients= True, verbose=True,tolerance_sigma = 1e-04)
    optimizer.create_optimizer('Adam', lr=1e-1)
    optimizer.optimize(num_lambdas =5, num_steps_per_lambda = 200)
    #print(optimizer.get_objective_constraint_evolution())
    path = os.getcwd() + '/tests/tmp'
    #optimizer.save_results(path)
    print(optimizer.get_final_state())
    optimizer.plot_results(path, 'dim_32_sphere')
    #optimizer.plot_results()
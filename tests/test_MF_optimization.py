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
from scoutNd.multifidelity_objective import MultifidelityObjective


def sphere(x):
    X = np.atleast_2d(x)
    val1 = np.sum(X**2, axis=1)
    # val2 = np.random.normal(0, 0.0001, val1.shape)
    val2 = 0.0
    return val1 + val2

def sphere_lf(x):
    X = np.atleast_2d(x)
    val1 = np.sum(X**2, axis=1)
    val3 = np.random.normal(0, 0.01, val1.shape)
    val2 = 0.001*np.sum(X, axis=1)
    return val1 + val2 + val3

def linear_constraint(X):
    x = np.atleast_2d(X)
    return 1 - x[:, 0] - x[:, 1]

if __name__ == '__main__':
    dim = 16
    constraints = [linear_constraint]
    #constraints = None
    obj = MultifidelityObjective(dim, [sphere_lf, sphere], constraints, qmc=True)
    obj.set_num_samples([64, 8])
    optimizer = Stochastic_Optimizer(obj,natural_gradients= True, verbose=True)
    optimizer.create_optimizer('Adam', lr=1e-2)
    optimizer.optimize(num_lambdas =10, num_steps_per_lambda = 300)
    #optimizer.optimize(num_steps = 300)
    print(optimizer.get_final_state())
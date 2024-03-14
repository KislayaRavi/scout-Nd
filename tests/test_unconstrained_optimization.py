import numpy as np
import torch
from scoutNd.objective_function import *
import matplotlib.pyplot as plt
from copy import deepcopy
import sys,os
import time
import pytest
datetime = time.strftime("%Y%m%d-%H%M%S")

from scoutNd.stochastic_optimizer import Stochastic_Optimizer
from scoutNd.objective_function import *

# TODO: someone still getting differnt results upon differnt runs. Seeds dont work?
torch.manual_seed(0)
np.random.seed(0)


def sphere(x):
    X = np.atleast_2d(x)
    val1 = np.sum(X**2, axis=1)
    # val2 = np.random.normal(0, 0.0001, val1.shape)
    val2 = 0
    return val1 + val2

def linear_constraint(X):
    x = np.atleast_2d(X)
    return 1 - x[:, 0] - x[:, 1]

def test_unconstrained_optimization():
    dim = 2
    #constraints = [linear_constraint]
    # constraints = None
    obj = Baseline1(dim, sphere,constraints=None, num_samples=32, qmc=True, correct_constraint_derivative=True)
    optimizer = Stochastic_Optimizer(obj, natural_gradients= True, verbose=True, tolerance_sigma = 1e-04)
    optimizer.create_optimizer('Adam', lr=1e-1)
    optimizer.optimize(num_steps = 500)
    assert optimizer.get_final_state() == 0

if __name__ == '__main__':
    dim = 2
    #constraints = [linear_constraint]
    # constraints = None
    obj = Baseline1(dim, sphere,constraints=None, num_samples=8, qmc=True, correct_constraint_derivative=True)
    optimizer = Stochastic_Optimizer(obj, natural_gradients= True, verbose=True, tolerance_sigma = 1e-03)
    optimizer.create_optimizer('Adam', lr=1e-2)
    optimizer.optimize(num_steps = 1000)
    print(optimizer.get_final_state())
    path = os.getcwd() + '/tests/tmp'
    optimizer.plot_results(path,'dim_2_sphere')
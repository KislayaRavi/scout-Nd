"""Run a constrained Bayesian-optimisation baseline (``bayes_opt``) on the benchmark suite.

Used as a comparison against SCOUT-Nd; results are pickled for ingestion by ``moore_plot.py``.
"""

from typing import Any
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds, NonlinearConstraint
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from bayes_opt import BayesianOptimization as bo
from collections import OrderedDict
from benchmarks import *
from common_data import *

    

def get_data_profile(values, dim, fx_star, eps_f):
    d = 0
    for idx, value in enumerate(values):
        if np.abs(value - fx_star) <= eps_f:
            d = idx / (dim + 1)
            # print(dim, idx)
            break 
    return d


def constrain_fn1_bo(**kwargs):
    return  -1*(kwargs['x0'] + kwargs['x1'])

def get_num_init_points(dim):
    if dim <= 2:
        num_init = 32
    elif dim <= 4:
        num_init = 64
    elif dim <= 8:
        num_init = 128
    else:
        num_init = 256
    return num_init

def optimize_cbo(obj_fn:AbstractBenchmark):
    dim = obj_fn.get_dim()
    num_samples = get_num_samples(dim)
    obj_fn.set_num_samples(num_samples)
    lower_bound, upper_bound = obj_fn.get_bounds()
    pbounds = OrderedDict()
    for i in range(dim):
        var_name = 'x' + str(i)
        pbounds[var_name] = (lower_bound[i], upper_bound[i])
    neg_obj_fn = lambda **kwargs: -1. * obj_fn.evaluate_hf_mean(list(kwargs.values()))
    if obj_fn.get_name() == 'Constraint Sphere Function':
        coefficient = -1
        constraint = NonlinearConstraint(constrain_fn1_bo, [-np.inf], [-1.])
        optimizer = bo(f=neg_obj_fn, constraint=constraint, pbounds=pbounds)
    else:
        optimizer = bo(f=neg_obj_fn, pbounds=pbounds)
    optimizer.maximize(init_points=get_num_init_points(dim), n_iter=2000 - get_num_init_points(dim))
    fx_star = obj_fn.get_optimum_value() 
    results = OrderedDict()
    results['name'] = obj_fn.get_name()
    results['dim'] = deepcopy(dim) 
    results['x_star'] = deepcopy(obj_fn.get_optimum_value()) 
    results['fx_star'] = deepcopy(fx_star)
    results['eval_points'] = deepcopy(obj_fn.eval_points)
    results['eval_vals'] = deepcopy(obj_fn.eval_val)
    print(obj_fn.get_name(), obj_fn.get_global_optimum())
    print(optimizer.max)
    # print('Constraint value', linear_constraint.residual(res.x))
    return results


if __name__ == '__main__':
    list_benchmarks = get_list_objective_functions()
    # num_benchmarks = len(seeds) * len(list_benchmarks)
    benchmark_id = 0
    accumulated_results = OrderedDict()
    for seed in seeds:
        np.random.seed(seed)
        for obj_fn in list_benchmarks:
            results = optimize_cbo(obj_fn)
            benchmark_id = benchmark_id + 1
            benchmark_name = 'benchmark' + str(benchmark_id)
            accumulated_results[benchmark_name] = results  
    # with open(method+'.pickle', "wb") as outfile:
    #     pickle.dump(accumulated_results, outfile, protocol=pickle.HIGHEST_PROTOCOL)
"""Run SciPy optimisers (``trust-constr`` etc.) on the benchmark suite as a baseline.

Used as a comparison against SCOUT-Nd; results are pickled for ingestion by ``moore_plot.py``.
"""

from typing import Any
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds, NonlinearConstraint
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
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


def constrain_fn_bo(**kwargs):
    return  -1*(kwargs['x0'] + kwargs['x1'])


def optimize_scipy(obj_fn:AbstractBenchmark, arguments, method='trust-constr'):
    dim = obj_fn.get_dim()
    num_samples = get_num_samples(dim)
    obj_fn.set_num_samples(num_samples)
    lower_bound, upper_bound = obj_fn.get_bounds()
    bnds = Bounds(lower_bound, upper_bound)
    if obj_fn.get_name() == 'Rosenbrock function':
        initial_val = -1.0* np.ones(dim)
    else:
        initial_val = 0.25* np.ones(dim)
    if obj_fn.get_name() == 'Constraint Sphere Function':
        coefficient = -1
        if dim > 2:
            linear_constraint = LinearConstraint([coefficient]*2 + [0.]*(dim-2), [-np.inf], [coefficient])
        else: 
            linear_constraint = LinearConstraint([coefficient]*2, [-np.inf], [coefficient])
        res = minimize(obj_fn.evaluate_hf_mean, initial_val, method=method, 
                       constraints=[linear_constraint], bounds=bnds, options=arguments)
    else:
        res = minimize(obj_fn.evaluate_hf_mean, initial_val, method=method, bounds=bnds, options=arguments)
    fx_star = obj_fn.get_optimum_value() 
    results = OrderedDict()
    results['name'] = obj_fn.get_name()
    results['dim'] = deepcopy(dim) 
    results['x_star'] = deepcopy(obj_fn.get_optimum_value()) 
    results['fx_star'] = deepcopy(fx_star)
    results['eval_points'] = deepcopy(obj_fn.eval_points)
    results['eval_vals'] = deepcopy(obj_fn.eval_val)
    print(obj_fn.get_name(), obj_fn.get_global_optimum())
    print(res.x, res.fun)
    # print('Constraint value', linear_constraint.residual(res.x))
    return results

def run_scipy_method(method):
    list_benchmarks = get_list_objective_functions()
    # num_benchmarks = len(seeds) * len(list_benchmarks)
    benchmark_id = 0
    accumulated_results = OrderedDict()
    arguments = {'maxiter': 2000}
    if method == 'trust-constr':
        arguments['xtol'] = 1e-12
    if method == 'COBYLA':
        arguments['tol'] = 1e-12
    if method == 'SLSQP':
        arguments['ftol'] = 1e-12
        arguments['eps'] = 1e-3  # This is the most important parameter.
    for seed in seeds:
        np.random.seed(seed)
        for obj_fn in list_benchmarks:
            results = optimize_scipy(obj_fn, arguments, method=method)
            benchmark_id = benchmark_id + 1
            benchmark_name = 'benchmark' + str(benchmark_id)
            accumulated_results[benchmark_name] = results  
    with open(method+'.pickle', "wb") as outfile:
        pickle.dump(accumulated_results, outfile, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    method_names = ['SLSQP', 'COBYLA']
    # method_names = ['SLSQP']
    for method in method_names:
        print("Running method: ", method)
        run_scipy_method(method)

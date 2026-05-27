"""Run the multi-fidelity SCOUT-Nd optimiser on the benchmark suite.

Uses ``MultifidelityObjective`` with low- and high-fidelity sample allocations,
pickles per-benchmark results for later comparison in ``moore_plot.py``.
"""

import numpy as np
from collections import OrderedDict
import pickle
from copy import deepcopy
from common_data import seeds
from common_data import *
from benchmarks import *
from stochastic_optimizer import Stochastic_Optimizer
from objective_function import Baseline1
from multifidelity_objective import MultifidelityObjective


def run_optimizer(obj_fn:AbstractBenchmark):
    dim = obj_fn.get_dim()
    num_samples_lf = get_num_samples(dim)
    num_samples_hf = int(num_samples_lf/4) 
    verbose, natural_gradients = False, False
    constraints = obj_fn.get_constraints()
    obj = MultifidelityObjective(dim, [obj_fn.evaluate_lf, obj_fn.evaluate_hf], constraints, qmc=True, correct_constraint_derivative=True)
    obj.set_num_samples([num_samples_lf, num_samples_hf])
    if obj_fn.get_name() == 'Rosenbrock function':
        optimizer = Stochastic_Optimizer(obj, natural_gradients=natural_gradients, verbose=verbose, initial_val=-1*np.ones(2*dim))
    else:
        optimizer = Stochastic_Optimizer(obj, natural_gradients=natural_gradients, verbose=verbose)
    optimizer.create_optimizer('Adam', lr=1e-1)
    fx_star = obj_fn.get_optimum_value() 
    results = OrderedDict()
    results['dim'] = deepcopy(dim) 
    results['x_star'] = deepcopy(obj_fn.get_optimum_value()) 
    results['fx_star'] = deepcopy(fx_star)
    num_resets=5
    if len(constraints) == 0:
        for i in range(num_resets):
            print('Reset Number:', i)
            optimizer.reset_sigma()
            optimizer.optimize(num_steps=int(2000/num_resets))
            eval_points= []
            for i in range(len(optimizer.stored_results)):
                eval_points.append(optimizer.stored_results[i].detach().numpy()[:dim])
            results['eval_points'] = np.array(eval_points)
            results['eval_vals'] = obj_fn(results['eval_points'])
            results['name'] = obj_fn.get_name()
            print(obj_fn.get_name(), obj_fn.get_global_optimum())
            print(optimizer.stored_results[-1], results['eval_vals'][-1] )
    else:
        optimizer.optimize(num_steps_per_lambda=int(500), num_lambdas=4)
        eval_points= []
        for i in range(len(optimizer.stored_results)):
            eval_points.append(optimizer.stored_results[i].detach().numpy()[:dim])
        results['eval_points'] = np.array(eval_points)
        results['eval_vals'] = obj_fn(results['eval_points'])
        results['name'] = obj_fn.get_name()
        print(obj_fn.get_name(), obj_fn.get_global_optimum())
        print(optimizer.stored_results[-1], results['eval_vals'][-1] )
    return results
    


if __name__ == '__main__':
    list_benchmarks = get_list_objective_functions()
    # num_benchmarks = len(seeds) * len(list_benchmarks)
    benchmark_id = 0
    accumulated_results = OrderedDict()
    for seed in seeds:
        np.random.seed(seed)
        for obj_fn in list_benchmarks:
            # print(obj_fn.normalize_params(np.array([[-5, 3], [-15, -3]])))
            results = run_optimizer(obj_fn)
            benchmark_id = benchmark_id + 1
            benchmark_name = 'benchmark' + str(benchmark_id)
            accumulated_results[benchmark_name] = results
    # with open('MF_Scout.pickle', "wb") as outfile:
    #     pickle.dump(accumulated_results, outfile, protocol=pickle.HIGHEST_PROTOCOL)
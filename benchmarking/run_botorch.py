"""Run BoTorch (Gaussian-process Bayesian optimisation) baseline on the benchmark suite.

Used as a comparison against SCOUT-Nd; results are pickled for ingestion by ``moore_plot.py``.
"""

import torch
import numpy as np
import pickle
from collections import OrderedDict
from copy import deepcopy
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelList
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, ConstrainedExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.optim.initializers import initialize_q_batch_nonneg
from common_data import seeds

dist = torch.distributions.Normal(0, 1e-3)

class Sphere():

    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.eval_points = []
        self.eval_val = []

    def hf(self, X):
        return -1.0 * torch.sum(torch.square(X), dim=-1, keepdim=True)
    
    def expected_hf(self, X, num_samples=50):
        val = self.hf(X)
        temp = torch.zeros((X.shape[0], 1))
        for j in range(X.shape[0]):
            self.count += num_samples
            for i in range(num_samples):
                self.eval_points.append(X[..., j, -1].numpy())
                self.eval_val.append(val[..., j, -1].numpy())
            noise = torch.tensor(np.random.normal(0, 0.1, (num_samples,1)))
            temp[j] = torch.mean(val[j]+noise)
        return temp

    def hf_no_noise(self, X):
        return -1.0 * torch.sum(torch.square(X), dim=-1, keepdim=True)
         
    def __call__(self, *args, **kwds):
        return self.expected_hf(args[0])

# def sphere(X):
#     return -1.0 * torch.sum(torch.square(X), dim=-1, keepdim=True)

def constraint1(X):
    val = -1*torch.sum(X[..., :2], dim=-1, keepdim=True)
    val = val + torch.ones_like(val)
    return val

def constraint2(X):
    val = torch.sum(X[..., :2], dim=-1, keepdim=True)
    val = val - torch.ones_like(val)
    return val

def create_gp(train_X, train_Y):
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

def get_best(train_X, train_Y, constraint_Y, tolerance=0.1):
    selected_indices = (constraint_Y < tolerance).ravel()
    selected_X = train_X[selected_indices] 
    selected_Y = train_Y[selected_indices]
    selected_constraint_Y = constraint_Y[selected_indices]
    # selected_X, selected_Y, selected_constraint_Y = train_X, train_Y, constraint_Y
    if selected_Y.shape[0] == 0:
        best_X = train_X[0]
        best_Y = train_Y[0]
        best_constraint = constraint_Y[0]
    else:
        max_index = torch.argmax(selected_Y)
        best_X = selected_X[max_index]
        best_Y = selected_Y[max_index]
        best_constraint = selected_constraint_Y[max_index]
    return best_X, best_Y, best_constraint

def optimize_cbo(dim, seed, constraint_number):
    num_starting_points = 10 * dim
    np.random.seed(seed)
    if constraint_number == 1:
        constraint= constraint1
        if dim == 2:
            x_star, fx_star = [0.5, 0.5], -0.5
        else: 
            x_star, fx_star = [0.5]*2 + [0.]*(dim-2), -0.5
    else:
        constraint = constraint2
        x_star, fx_star = [0.]*dim, 0.
    def log_pf(X, best_constraint): 
        val = constraint(X)
        # print(X, val)
        # temp = torch.max(torch.zeros_like(val), val)
        temp = torch.max(torch.zeros_like(val), val - best_constraint)
        return temp + torch.log(dist.cdf(val)+1e-16)
    bounds = torch.stack([-torch.ones(dim), torch.ones(dim)])
    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(num_starting_points, dim)
    sphere = Sphere(dim)
    train_Y = sphere(train_X)
    constraint_Y = constraint(train_X)
    best_X, best_f, best_constraint = get_best(train_X, train_Y, constraint_Y)
    model = create_gp(train_X, train_Y)
    acq = ExpectedImprovement(model, best_f)
    N = 5
    q = 1
    # generate a large number of random q-batches
    Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(100 * N, q, dim)
    Yraw = acq(Xraw)  # evaluate the acquisition function on these q-batches

    # apply the heuristic for sampling promising initial conditions
    X = initialize_q_batch_nonneg(Xraw, Yraw, N)

    # we'll want gradients for the input
    X.requires_grad_(True)
    # set up the optimizer, make sure to only pass in the candidate set here
    optimizer = torch.optim.Adam([X], lr=0.01)

    num_bo_steps = 5 * dim
    for step in range(num_bo_steps):
        X_traj = []  # we'll store the results
        # run a basic optimization loop
        for i in range(75):
            optimizer.zero_grad()
            # this performs batch evaluation, so this is an N-dim tensor
            losses = -acq(X)  # torch.optim minimizes
            constraint_loss = log_pf(X, best_constraint)
            loss = losses.sum() + constraint_loss.sum()

            loss.backward()  # perform backward pass
            optimizer.step()  # take a step

            # clamp values to the feasible set
            for j, (lb, ub) in enumerate(zip(*bounds)):
                X.data[..., j].clamp_(lb, ub)  # need to do this on the data not X itself

            # store the optimization trajecatory
            X_traj.append(X.detach().clone())

            # if (i + 1) % 15 == 0:
            #     print(f"Iteration {i+1:>3}/75 - Loss: {loss.item():>4.3f}")
        X_next = X_traj[-1][torch.argmax(acq(X_traj[-1]))]
        train_X = torch.concatenate((train_X, X_next), dim=0)
        Y_next = sphere(X_next)
        Y_next_constraint = constraint(X_next)
        train_Y = torch.concatenate((train_Y, Y_next), dim=0)
        constraint_Y = torch.concatenate((constraint_Y, Y_next_constraint), dim=0)
        best_X, best_f, best_constraint = get_best(train_X, train_Y, constraint_Y)
        # print("Proposal:", X_next, Y_next)
        print("Current best:", best_X, best_f)
        model = create_gp(train_X, train_Y)
        acq = ExpectedImprovement(model, best_f)
    results = OrderedDict()
    results = OrderedDict()
    results['dim'] = deepcopy(dim) 
    results['x_star'] = deepcopy(x_star) 
    results['fx_star'] = deepcopy(fx_star)
    results['eval_points'] = deepcopy(sphere.eval_points)
    results['eval_vals'] = deepcopy(sphere.eval_val)
    return results

if __name__ == '__main__':
    dim_list = [2, 4, 8, 16, 32]
    method = 'CBO'
    num_benchmarks = 2 * len(seeds) * len(dim_list)
    benchmark_id = 0
    accumulated_results = OrderedDict()
    for seed in seeds:
        for dim in dim_list:
            print(dim)
            results = optimize_cbo(dim, seed, 1)
            benchmark_id = benchmark_id + 1
            benchmark_name = 'benchmark' + str(benchmark_id)
            accumulated_results[benchmark_name] = results
        for dim in dim_list:
            results = optimize_cbo(dim, seed, 2)
            benchmark_id = benchmark_id + 1
            benchmark_name = 'benchmark' + str(benchmark_id)
            accumulated_results[benchmark_name] = results
    with open(method+'.pickle', "wb") as outfile:
        pickle.dump(accumulated_results, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    
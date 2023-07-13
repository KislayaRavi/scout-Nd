import numpy as np
from scipy.optimize import  minimize
import torch 
import matplotlib.pyplot as plt
seed = 0

# TODO: Make an abstract class. Then create abstract function to incorporate multi-fidelity
# TODO: Perform profiling, optimize the code for the most time-consuming part

class Stochastic_Optimizer(object):

    def __init__(self, dim, func, distribution='gaussian'):
        self.dim, self.func, self.distribution = dim, func, distribution
        self.f_list = [self.func]

    def grad_logpdf_one_sample(self, mean, scaled_sigma, sample):
        m = torch.tensor(mean, requires_grad=True)
        ss = torch.tensor(scaled_sigma, requires_grad=True)
        dist = torch.distributions.MultivariateNormal(m, torch.diag(torch.exp(ss)))
        val = dist.log_prob(torch.tensor(sample))
        val.backward()
        grad_mean, grad_sigma = m.grad, ss.grad
        return np.array(np.concatenate([grad_mean, grad_sigma]))

    def grad_logpdf(self, mean, scaled_sigma, samples):
        num_samples = len(samples)
        grad_array = np.zeros((num_samples, 2*self.dim))
        for i in range(num_samples):
            grad_array[i, :] = self.grad_logpdf_one_sample(mean, scaled_sigma, samples[i, :])
        return grad_array

    def approximate_derivative_baseline(self, f_val, grad_logpdf_val, num_samples):
        s = np.sum(f_val)
        B = (s - f_val)/(num_samples - 1)
        f_var_red = np.reshape(f_val - B, (num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return grad_val

    def function_wrapper(self, x, num_samples=100):
        mean, scaled_sigma = x[:self.dim], x[self.dim:]
        #TODO: Write a separate class for sampling
        if self.dim == 1:
            samples = np.random.normal(mean, np.exp(scaled_sigma), size=(num_samples, self.dim))
        else:
            samples = np.random.multivariate_normal(mean, np.diag(np.exp(scaled_sigma)), size=(num_samples,))
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        mean_array, grad_array = np.zeros(len(self.f_list)), np.zeros((len(self.f_list), 2*self.dim))
        for idx, func in enumerate(self.f_list):
            f_val = func(samples)
            mean_array[idx] = np.mean(f_val)
            grad_array[idx, :] = self.approximate_derivative_baseline(f_val, grad_logpdf_val, num_samples)
        # Potential Idea: Polynomial Chaos Expansion with sparse grid to calculate expectation
        return np.sum(mean_array), np.sum(grad_array, axis=0)

    def optimize_no_constraints(self, maxiter=100, method='BFGS'):
        options = {'maxiter':maxiter}
        mean_starting_point = np.ones(self.dim)
        scaled_sigma_starting_point = 0.1*np.ones(self.dim) #If we make this large then the estimation of gradient is very bad
        starting_point = np.concatenate([mean_starting_point, scaled_sigma_starting_point])
        results = minimize(self.function_wrapper, starting_point, jac=True, method=method, options=options)
        return results
    
    #TODO: Make this private
    def update_function_list(self, lambdas, constraints):
        self.f_list = [self.func]
        for idx, constraint in enumerate(constraints):
            self.f_list.append(lambda x: lambdas[idx]*np.max([np.zeros(len(x)), constraint(x)], axis=0))

    def optimize_with_constraints(self, constraints, maxiter=50, method='BFGS'):
        options = {'maxiter':maxiter}
        mean_starting_point = np.ones(self.dim)
        num_constraints = len(constraints)
        c = np.zeros(num_constraints) - 2
        scaled_sigma_starting_point = np.zeros(self.dim) #If we make this large then the estimation of gradient is very bad
        # TODO: Following process of changing lambda is not good. Maybe simply increase lambda
        for i in range(5):
            lambdas = 10**c
            self.update_function_list(lambdas, constraints)
            if i==0:
                starting_point = np.concatenate([mean_starting_point, scaled_sigma_starting_point])
            else:
                starting_point = results.x
            results = minimize(self.function_wrapper, starting_point, jac=True, method=method, options=options)
            c = c + 1
            print('Updated optimum point and c after ', i, ' optimization ', results.x, c)
            print('Gradient at optimum', self.function_wrapper(results.x))
            # print('penalty', self.penalty(results.x, lambdas, constraints))
        return results


def sphere(x):
    val1 = np.sum(x**2, axis=1)
    val2 = np.random.normal(0, 0.0001, val1.shape)
    # val2 = 0
    return val1 + val2

# TODO: try other benchmarks

def linear_constraint(X):
    x = np.atleast_2d(X)
    return 1 - x[:, 0] - x[:, 1]

if __name__ == '__main__':
    so = Stochastic_Optimizer(5, sphere)
    constraints = [linear_constraint]
    # num_constraints = len(constraints)
    # c = np.zeros(num_constraints)
    # lambdas = 10**c
    # so.update_function_list(lambdas, constraints)
    # x = np.ones(4) * -2
    # x[0] = 0.5
    # x[1] = 0.5
    # print(so.function_wrapper(x))
    results = so.optimize_with_constraints(constraints)
    print('Final result', results.x)

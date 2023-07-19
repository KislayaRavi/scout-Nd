import numpy as np
import torch 
from scipy.stats import qmc
from abc import ABC, abstractmethod
   
class ObjectiveAbstract(ABC):

    def __init__(self, dim, func: callable, constraints: list, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol', log_lambdas=None):
        self.dim, self.func, self.distribution = dim, func, distribution
        self.constraints = constraints
        if constraints is None:
            self.num_constraints = 0
        else:
            self.num_constraints = len(constraints)
        self.num_samples = num_samples
        self.qmc, self.qmc_engine = qmc, qmc_engine 
        self.update_lambdas(log_lambdas)

    def deterministic_penalty_definition(self, constraint, mean, samples, grad_logpdf_val):
        c_val = constraint(mean)
        if c_val <= 0:
            return 0, np.zeros(2*self.dim)
        else:
            return self.estimator_mean_and_derivative(constraint, mean, samples, grad_logpdf_val)

    def stochastic_penalty_definition(self, constraint, mean, samples, grad_logpdf_val): 
        # use this when constraint is stochastic (not yet used in the code)
        mean_constraint, grad_constraint = self.estimator_mean_and_derivative(constraint, mean, samples, grad_logpdf_val)
        if mean_constraint <= 0:
            return 0, np.zeros(2*self.dim)
        return mean_constraint, grad_constraint
    
    def get_penalty(self, mean, samples, grad_logpdf_val):
        if self.constraints is None:
            return 0
        else:
            num_constraints = len(self.constraints)
            mean_array, grad_array = np.zeros(num_constraints), np.zeros((num_constraints, 2*self.dim))
            for idx, constraint in enumerate(self.constraints):
                mean_array[idx], grad_array[idx, :] = self.deterministic_penalty_definition(constraint, mean, samples, grad_logpdf_val)
            return np.sum(self.lambdas*mean_array), np.sum(self.lambdas[:,None]*grad_array, axis=0)
        
    def update_lambdas(self, log_lambdas):
        if log_lambdas is None:
            self.lambdas = np.exp(np.zeros(len(self.constraints)))
        else:
            self.lambdas = np.exp(log_lambdas)
            assert len(self.lambdas) == len(self.constraints), 'Number of constraints should be equal to number of lambdas'
    
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

    def sampler(self, mean, scaled_sigma):
        if self.qmc:
            dist = qmc.MultivariateNormalQMC(mean=mean, cov=np.diag(np.exp(scaled_sigma)))
            samples = dist.random(self.num_samples)
        else:
            if self.dim == 1:
                samples = np.random.normal(mean, np.exp(scaled_sigma), size=(self.num_samples, self.dim))
            else:
                samples = np.random.multivariate_normal(mean, np.diag(np.exp(scaled_sigma)), size=(self.num_samples,))
        return samples

    def function_wrapper(self, x):
        mean, scaled_sigma = x[:self.dim].detach().numpy(), x[self.dim:].detach().numpy()
        samples = self.sampler(mean, scaled_sigma)
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        mean_func, grad_func = self.estimator_mean_and_derivative(self.func, mean, samples, grad_logpdf_val)
        mean_penalty, grad_penalty = self.get_penalty(mean, samples, grad_logpdf_val)
        return mean_func + mean_penalty, grad_func + grad_penalty

    @abstractmethod
    def estimator_mean_and_derivative(self, func, mean, samples, grad_logpdf_val):
        pass

class NoVarianceReduction(ObjectiveAbstract):

    def __init__(self, dim, func, constraints, **kwargs):
        super().__init__(dim, func, constraints, **kwargs)
  

    def estimator_mean_and_derivative(self, func, mean, samples, grad_logpdf_val):
        f_val = func(samples)
        f_var_red = np.reshape(f_val, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return np.mean(f_val), grad_val


class SFBiasedBaseline(ObjectiveAbstract):

    def __init__(self, dim, func, constraints, **kwargs):
        super().__init__(dim, func, constraints, **kwargs)
  

    def estimator_mean_and_derivative(self, func, mean, samples, grad_logpdf_val):
        f_val = func(samples)
        s = np.sum(f_val)
        B = (s - f_val)/(self.num_samples - 1)
        f_var_red = np.reshape(f_val - B, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return np.mean(f_val), grad_val
    
    
class SFUnbiasedBaseline(ObjectiveAbstract):

    def __init__(self, dim, func, constraints, **kwargs):
        super().__init__(dim, func, constraints, **kwargs)
  

    def estimator_mean_and_derivative(self, func, mean, samples, grad_logpdf_val):
        f_val, f_mean = func(samples), func(mean)
        f_var_red = np.reshape(f_val - f_mean, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0) 
        # TODO: Look at the noise(variance) of grad_val. I expect this to be higher than biased version. 
        return np.mean(f_val), grad_val
    
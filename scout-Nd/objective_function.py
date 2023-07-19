import numpy as np
import torch 
from scipy.stats import qmc
from abc import ABC, abstractmethod
   
class ObjectiveAbstract(ABC):

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol', lambdas=None):
        self.dim, self.func, self.distribution = dim, func, distribution
        self.constraints = constraints
        self.num_samples = num_samples
        self.qmc, self.qmc_engine = qmc, qmc_engine 
        self.update_lambdas(lambdas)

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
        
    def update_lambdas(self, lambdas):
        if lambdas is None:
            self.lambdas = np.exp(np.zeros(len(self.constraints)))
        else:
            self.lambdas = np.exp(lambdas)
            assert len(lambdas) == len(self.constraints), 'Number of constraints should be equal to number of lambdas'
    
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

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol', lambdas=None):
        super().__init__(dim, func, constraints, distribution=distribution, num_samples=num_samples, qmc=qmc, qmc_engine=qmc_engine, lambdas=lambdas)
  

    def estimator_mean_and_derivative(self, func, mean, samples, grad_logpdf_val):
        f_val = func(samples)
        f_var_red = np.reshape(f_val, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return np.mean(f_val), grad_val


class SFBiasedBaseline(ObjectiveAbstract):

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol', lambdas=None):
        super().__init__(dim, func, constraints, distribution=distribution, num_samples=num_samples, qmc=qmc, qmc_engine=qmc_engine, lambdas=lambdas)
  

    def estimator_mean_and_derivative(self, func, mean, samples, grad_logpdf_val):
        f_val = func(samples)
        s = np.sum(f_val)
        B = (s - f_val)/(self.num_samples - 1)
        f_var_red = np.reshape(f_val - B, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return np.mean(f_val), grad_val
    
    
class SFUnbiasedBaseline(ObjectiveAbstract):

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol', lambdas=None):
        super().__init__(dim, func, constraints, distribution=distribution, num_samples=num_samples, qmc=qmc, qmc_engine=qmc_engine, lambdas=lambdas)
  

    def estimator_mean_and_derivative(self, func, mean, samples, grad_logpdf_val):
        f_val, f_mean = func(samples), func(mean)
        f_var_red = np.reshape(f_val - f_mean, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0) 
        # TODO: Look at the noise(variance) of grad_val. I expect this to be higher than biased version. 
        return np.mean(f_val), grad_val
    

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
    dim = 10
    constraints = [linear_constraint]
    lambdas = -1 * np.ones(len(constraints))
    # constraints = None
    obj = SFBiasedBaseline(dim, sphere, constraints, num_samples=128, lambdas=lambdas) # Biased performs better than unbiased
    # obj = SFUnbiasedBaseline(dim, sphere, constraints, num_samples=128)
    # obj = NoVarianceReduction(dim, sphere, constraints, num_samples=64, qmc=True, qmc_engine='Halton') # QMC engine can be Sobol, Halton
    initial_val = np.ones(2*dim)
    initial_val[dim:] = -1
    parameters = [torch.tensor(initial_val, requires_grad=True)]
    optimiser = torch.optim.Adam(parameters, lr=1e-2) # if lambda is high, then lr should be low
    for reps in range(5):
        for i in range(100):
            val, grad = obj.function_wrapper(parameters[0])
            parameters[0].grad = torch.tensor(grad) #torch.ones_like(parameters[0]) #This is line which will break in GPU, one has to explicitly mention .to(device) with correct device of cpu or gpu
            optimiser.step()
        print(lambdas, parameters)  
        lambdas = lambdas + 1
        obj.update_lambdas(lambdas) 
        # print(obj.function_wrapper(parameters[0]))
    # x = 2*np.ones(2*dim)
    # x[dim:] = -5
    # print(obj.function_wrapper(torch.tensor(x)))
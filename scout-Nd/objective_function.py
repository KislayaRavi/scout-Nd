import numpy as np
import torch 
from scipy.stats import qmc
from abc import ABC, abstractmethod
   
class ObjectiveAbstract(ABC):

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol'):
        self.dim, self.func, self.distribution = dim, func, distribution
        self.f_list = [self.func]
        self.num_samples = num_samples
        self.qmc, self.qmc_engine = qmc, qmc_engine
        if constraints is not None:
            for constraint in constraints:
                self.f_list.append(constraint)
    
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

    @abstractmethod
    def function_wrapper(self, x):
        pass

class NoVarianceReduction(ObjectiveAbstract):

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol'):
        super().__init__(dim, func, constraints, distribution=distribution, num_samples=num_samples, qmc=qmc, qmc_engine=qmc_engine)
  

    def approximate_derivative_biased_baseline(self, f_val, grad_logpdf_val, num_samples):
        f_var_red = np.reshape(f_val, (num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return grad_val
    
    def function_wrapper(self, x):
        mean, scaled_sigma = x[:self.dim].detach().numpy(), x[self.dim:].detach().numpy()
        samples = self.sampler(mean, scaled_sigma)
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        mean_array, grad_array = np.zeros(len(self.f_list)), np.zeros((len(self.f_list), 2*self.dim))
        for idx, func in enumerate(self.f_list):
            f_val = func(samples)
            mean_array[idx] = np.mean(f_val)
            grad_array[idx, :] = self.approximate_derivative_biased_baseline(f_val, grad_logpdf_val, self.num_samples)
        # Potential Idea: Polynomial Chaos Expansion with sparse grid to calculate expectation
        return np.sum(mean_array), np.sum(grad_array, axis=0)



class SFBiasedBaseline(ObjectiveAbstract):

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol'):
        super().__init__(dim, func, constraints, distribution=distribution, num_samples=num_samples, qmc=qmc, qmc_engine=qmc_engine)
  

    def approximate_derivative_biased_baseline(self, f_val, grad_logpdf_val, num_samples):
        s = np.sum(f_val)
        B = (s - f_val)/(num_samples - 1)
        f_var_red = np.reshape(f_val - B, (num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return grad_val
    
    def function_wrapper(self, x):
        mean, scaled_sigma = x[:self.dim].detach().numpy(), x[self.dim:].detach().numpy()
        samples = self.sampler(mean, scaled_sigma)
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        mean_array, grad_array = np.zeros(len(self.f_list)), np.zeros((len(self.f_list), 2*self.dim))
        for idx, func in enumerate(self.f_list):
            f_val = func(samples)
            mean_array[idx] = np.mean(f_val)
            grad_array[idx, :] = self.approximate_derivative_biased_baseline(f_val, grad_logpdf_val, self.num_samples)
        # Potential Idea: Polynomial Chaos Expansion with sparse grid to calculate expectation
        return np.sum(mean_array), np.sum(grad_array, axis=0)
    
    
class SFUnbiasedBaseline(ObjectiveAbstract):

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=128, qmc=True, qmc_engine='Sobol'):
        super().__init__(dim, func, constraints, distribution=distribution, num_samples=num_samples, qmc=qmc, qmc_engine=qmc_engine)
  

    def approximate_derivative_unbiased_baseline(self, f_val, f_mean, grad_logpdf_val, num_samples):
        f_var_red = np.reshape(f_val - f_mean, (num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0) 
        # TODO: Look at the noise(variance) of grad_val. I expect this to be higher than biased version. 
        return grad_val
    
    def function_wrapper(self, x):
        mean, scaled_sigma = x[:self.dim].detach().numpy(), x[self.dim:].detach().numpy()
        #TODO: Write a separate class for sampling
        samples = self.sampler(mean, scaled_sigma)
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        mean_array, grad_array = np.zeros(len(self.f_list)), np.zeros((len(self.f_list), 2*self.dim))
        for idx, func in enumerate(self.f_list):
            f_val = func(samples)
            f_mean = func(mean)
            mean_array[idx] = np.mean(f_val)
            grad_array[idx, :] = self.approximate_derivative_unbiased_baseline(f_val, f_mean, grad_logpdf_val, self.num_samples)
        # Potential Idea: Polynomial Chaos Expansion with sparse grid to calculate expectation
        return np.sum(mean_array), np.sum(grad_array, axis=0)

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
    dim = 15
    constraints = [linear_constraint]
    # constraints = None
    obj = SFBiasedBaseline(dim, sphere, constraints, num_samples=128) # Biased performs better than unbiased
    # obj = SFUnbiasedBaseline(dim, sphere, constraints, num_samples=64)
    # obj = NoVarianceReduction(dim, sphere, constraints, num_samples=64, qmc=True, qmc_engine='Halton') # QMC engine can be Sobol, Halton
    parameters = [torch.tensor(np.ones(2*dim), requires_grad=True)]
    optimiser = torch.optim.Adam(parameters, lr=1e-1)
    for i in range(100):
        val, grad = obj.function_wrapper(parameters[0])
        parameters[0].grad = torch.tensor(grad) #torch.ones_like(parameters[0]) #This is line which will break in GPU, one has to explicitly mention .to(device) with correct device of cpu or gpu
        optimiser.step()   
    print(parameters)
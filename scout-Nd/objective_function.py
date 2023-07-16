import numpy as np
import torch 

class Objective():

    @staticmethod
    def forward(ctx, input, obj):
        mean, scaled_sigma = input[:obj.dim], input[obj.dim:]
        if obj.dim == 1:
            samples = np.random.normal(mean, np.exp(scaled_sigma), size=(obj.num_samples, obj.dim))
        else:
            samples = np.random.multivariate_normal(mean, np.diag(np.exp(scaled_sigma)), size=(obj.num_samples,))
        ctx.save_for_backward(mean)
        ctx.save_for_backward(scaled_sigma)
        ctx.save_for_backward(samples)
        mean_array = np.zeros(len(obj.f_list))
        f_val_list = []
        for idx, func in enumerate(obj.f_list):
            f_val_list.append(func(samples))
            mean_array[idx] = np.mean(f_val_list[-1])
        ctx.save_for_backward(f_val_list)
        ctx.save_for_backward(obj)
        print('Forward call')
        return np.sum(mean_array)

    @staticmethod
    def backward(ctx, grad_output):
        mean, scaled_sigma, samples, f_val_list, obj,  = ctx.saved_tensors
        grad_logpdf_val = obj.grad_logpdf(mean, scaled_sigma, samples)
        grad_array = np.zeros((len(obj.f_list), 2*obj.dim))
        for idx, func in enumerate(obj.f_list):
            f_val = f_val_list[idx]
            grad_array[idx, :] = obj.approximate_derivative_biased_baseline(f_val, grad_logpdf_val, obj.num_samples)
        # Potential Idea: Polynomial Chaos Expansion with sparse grid to calculate expectation
        return np.sum(grad_array, axis=0)
    
    

class SFBiasedBaseline():

    def __init__(self, dim, func, constraints, distribution='gaussian', num_samples=100):
        self.dim, self.func, self.distribution = dim, func, distribution
        self.f_list = [self.func]
        self.num_samples = num_samples
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

    def approximate_derivative_biased_baseline(self, f_val, grad_logpdf_val, num_samples):
        s = np.sum(f_val)
        B = (s - f_val)/(num_samples - 1)
        f_var_red = np.reshape(f_val - B, (num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return grad_val
    
    def function_wrapper(self, x):
        mean, scaled_sigma = x[:self.dim].detach().numpy(), x[self.dim:].detach().numpy()
        #TODO: Write a separate class for sampling
        if self.dim == 1:
            samples = np.random.normal(mean, np.exp(scaled_sigma), size=(self.num_samples, self.dim))
        else:
            samples = np.random.multivariate_normal(mean, np.diag(np.exp(scaled_sigma)), size=(self.num_samples,))
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        mean_array, grad_array = np.zeros(len(self.f_list)), np.zeros((len(self.f_list), 2*self.dim))
        for idx, func in enumerate(self.f_list):
            f_val = func(samples)
            mean_array[idx] = np.mean(f_val)
            grad_array[idx, :] = self.approximate_derivative_biased_baseline(f_val, grad_logpdf_val, self.num_samples)
        # Potential Idea: Polynomial Chaos Expansion with sparse grid to calculate expectation
        return np.sum(mean_array), np.sum(grad_array, axis=0)
    


def sphere(x):
    val1 = np.sum(x**2, axis=1)
    # val2 = np.random.normal(0, 0.0001, val1.shape)
    val2 = 0
    return val1 + val2


def linear_constraint(X):
    x = np.atleast_2d(X)
    return 1 - x[:, 0] - x[:, 1]

if __name__ == '__main__':
    dim = 15
    constraints = [linear_constraint]
    obj = SFBiasedBaseline(dim, sphere, constraints)
    parameters = [torch.tensor(np.ones(2*dim), requires_grad=True)]
    optimiser = torch.optim.Adam(parameters, lr=1e-1)
    for i in range(100):
        val, grad = obj.function_wrapper(parameters[0])
        parameters[0].grad = torch.tensor(grad)#torch.ones_like(parameters[0])
        optimiser.step()   
    print(parameters)
import numpy as np
import torch 
from scipy.stats import qmc
from abc import ABC, abstractmethod
from copy import deepcopy
   
class ObjectiveAbstract(ABC):
    """"Base class for objective function in *scout-Nd*.

    """

    def __init__(self, dim: int, func: callable, constraints: list, 
                 distribution='gaussian', num_samples=128, qmc=True, 
                 qmc_engine='Sobol', log_lambdas: np.ndarray=None,
                 correct_constraint_derivative=True, adaptive_sample_size: bool=False):
        """Constructor.

        Parameters
        ----------
        dim : int
            Dimension of the original objective function.
        func : callable
            The objective function.
        constraints : list
            List of callables representing the left-hand-side of less than equal to inequality constraint.
        distribution : str, optional
            name of the distribution of the parameters, by default 'gaussian'.
        num_samples : int, optional
            Number of samples used to estimate the gradient, by default 128.
        qmc : bool, optional
            Boolean variable to turn on/off the QMC sampling strategy, by default True.
        qmc_engine : str, optional
            Name of the QMC sampling method, by default 'Sobol'.
        log_lambdas : np.ndarray, optional
            Exponential logarithm of the penalty coefficients, by default None.
        correct_constraint_derivative : bool, optional
            If constrained is satisfied then set the derivative of the w.r.t $\mu$ to zero, by default True. 
        """
        self.dim, self.func, self.distribution = dim, func, distribution
        self.constraints = []
        #self.lambda_list = [] # storage for the lambda values
        if constraints is None:
            self.num_constraints = 0
        else:
            self.num_constraints = len(constraints)
            self.create_constraints_list(constraints)
        self.num_samples = num_samples
        self.qmc, self.qmc_engine = qmc, qmc_engine 
        self.correct_constraint_derivative = correct_constraint_derivative
        if self.num_constraints > 0:
            self.update_lambdas(log_lambdas)
        self.constrain_values = None  # placeholder for constraint values, size : no_constraints x 1
        self.objective_value = None
        self.adaptive_sample_size = adaptive_sample_size        

    def create_constraints_list(self, constraints: list) -> None:
        """Create the attribute of list of constraints for the class.

        Parameters
        ----------
        constraints : list
            List of callables representing the left-hand-side of less than equal to inequality constraint.
        """
        for i in range(self.num_constraints):
            # def constraint_wrapper(x):
            #     y = constraints[i](x)
            #     y[y<0] = 0
            #     return y
            # self.constraints.append(constraint_wrapper)
            self.constraints.append(lambda x, i=i: np.maximum(0, constraints[i](x)))

    # def deterministic_penalty_definition(self, constraint: callable, mean: float, 
    #                                      samples: np.ndarray, grad_logpdf_val: np.ndarray):
    def deterministic_penalty_definition(self, constraint: callable, mean: float, samples: np.ndarray):
        """Penalty term for determinitic constraints.

        Parameters
        ----------
        constraint : callable
            Callables representing the left-hand-side of less than equal to inequality constraint.
        mean : float
            Mean of the samples.
        samples : np.ndarray
            Samples drawn from the distribution to estimate the gradient.
        grad_logpdf_val : np.ndarray
            Gradient of logarithm of density evaluated at samples.

        Returns
        -------
        float
            Mean of the objective function evaluations at the sampled points.
        np.ndarray
            Estimated gradient of the objective evaluation at mean
        """
        # mean_val, grad = self.estimator_mean_and_derivative(constraint, mean, samples, grad_logpdf_val)
        # c_val = constraint(mean)
        # if c_val <= 0 and self.correct_constraint_derivative:
        #     mean_val = 0
        #     grad[:self.dim] = np.zeros(self.dim)
        # return mean_val, grad
        c_val = constraint(mean)
        if c_val <= 0 and self.correct_constraint_derivative:
            return np.zeros(len(samples))
        return constraint(samples)

    def stochastic_penalty_definition(self, constraint: callable, mean: float, 
                                       samples: np.ndarray, grad_logpdf_val: np.ndarray): 
        """Penalty term for stochastic constraints.

        Parameters
        ----------
        constraint : callable
            Callables representing the left-hand-side of less than equal to inequality constraint.
        mean : float
            Mean of the samples.
        samples : np.ndarray
            Samples drawn from the distribution to estimate the gradient.
        grad_logpdf_val : np.ndarray
            Gradient of logarithm of density evaluated at samples.

        Returns
        -------
        float
            Mean of the objective function evaluations at the sampled points.
        np.ndarray
            Estimated gradient of the objective evaluation at mean
        """
        # use this when constraint is stochastic (not yet used in the code)
        # Correction of derivative in stochastic case is complicated.
        # We need to define a robustness measure.
        mean_val, grad = self.estimator_mean_and_derivative(constraint, mean, samples, grad_logpdf_val)
        return mean_val, grad
    
    # def get_penalty(self, mean: float, samples: np.ndarray, grad_logpdf_val: np.ndarray):
    def get_penalty(self, mean: float, samples: np.ndarray):
        """Accumulates penalty from all the constraints, multiplies it with lambdas and returns the total penalty. 

        Parameters
        ----------
        mean : float
            Mean of the samples.
        samples : np.ndarray
            Samples drawn from the distribution to estimate the gradient.
        grad_logpdf_val : np.ndarray
            Gradient of logarithm of density evaluated at samples.

        Returns
        -------
        float
            Mean of the objective function evaluations at the sampled points.
        """
        # if self.num_constraints == 0:
        #     return 0, 0
        # else:
        #     mean_array, grad_array = np.zeros(self.num_constraints), np.zeros((self.num_constraints, 2*self.dim))
        #     for idx, constraint in enumerate(self.constraints):
        #         mean_array[idx], grad_array[idx, :] = self.deterministic_penalty_definition(constraint, mean, samples, grad_logpdf_val)
        #     self.constrain_values = mean_array #storing constraint values at the current design point
        #     return np.sum(self.lambdas*mean_array), np.sum(self.lambdas[:,None]*grad_array, axis=0)
        if self.num_constraints == 0:
            return np.zeros((len(samples),1))
        else:
            constraint_sample_values = np.zeros((len(samples), self.num_constraints))
            for idx, constraint in enumerate(self.constraints):
                constraint_sample_values[:, idx] = self.deterministic_penalty_definition(constraint, mean, samples)
            self.constrain_values = np.mean(constraint_sample_values, axis=0) #storing constraint values at the current design point
            return np.sum(self.lambdas*constraint_sample_values, axis=1)

    def update_lambdas(self, log_lambdas: np.ndarray):
        """Updates the scaling of penalty term

        Parameters
        ----------
        log_lambdas : np.ndarray
            Exponential logarithm of the penalty coefficients
        """
        if log_lambdas is None:
            self.lambdas = np.exp(np.zeros(len(self.constraints)))
        else:
            self.lambdas = np.exp(log_lambdas)
            assert len(self.lambdas) == len(self.constraints), 'Number of constraints should be equal to number of lambdas'
        #self.lambda_list.append(self.lambdas)
    
    def grad_logpdf_one_sample(self, mean: np.ndarray, scaled_sigma: np.ndarray, sample: np.ndarray):
        """Evaluates the gradient of logarithm of the distribution with respect to the mean and the scaled sigma for a given sample.

        Parameters
        ----------
        mean : np.ndarray
            Mean of the distribution.
        scaled_sigma : np.ndarray
            Logarithm of the variance of the distribution.
        sample : np.ndarray
            Location where the gradient is evaluated.

        Returns
        -------
        np.ndarray
            Gradient of logarithm of the distribution with respect to the mean and the scaled sigma.
        """
        m = torch.tensor(mean, requires_grad=True)
        ss = torch.tensor(scaled_sigma, requires_grad=True)
        dist = torch.distributions.MultivariateNormal(m, torch.diag(torch.exp(ss)**2))
        val = dist.log_prob(torch.tensor(sample))
        val.backward()
        grad_mean, grad_sigma = m.grad, ss.grad
        return np.array(np.concatenate([grad_mean, grad_sigma]))

    def grad_logpdf(self, mean: np.ndarray, scaled_sigma: np.ndarray, samples: np.ndarray):
        """Evaluates the gradient of logarithm of the distribution with respect to the mean and the scaled sigma for all the samples.

        Parameters
        ----------
        mean : np.ndarray
            Mean of the distribution.
        scaled_sigma : np.ndarray
            Logarithm of the variance of the distribution.
        sample : np.ndarray
            Locations where the gradients are evaluated.

        Returns
        -------
        np.ndarray
            Gradient of logarithm of the distribution with respect to the mean and the scaled sigma for all the samples.
        """
        num_samples = len(samples)
        grad_array = np.zeros((num_samples, 2*self.dim))
        for i in range(num_samples):
            grad_array[i, :] = self.grad_logpdf_one_sample(mean, scaled_sigma, samples[i, :])
        return grad_array
    
    def get_variance(self, scaled_sigma: np.ndarray):
        """Returns the variance of the distribution.

        Parameters
        ----------
        scaled_sigma : np.ndarray
            Logarithm of the variance of the distribution.

        Returns
        -------
        np.ndarray
            Variance of the distribution.
        """
        return np.exp(scaled_sigma)**2

    def sampler(self, mean, scaled_sigma, num_samples):
        """Draws the samples from a Gaussian distribution with given mean and variance.
        The stratergy is eith QMC or Vanilla sampling methods. This is chosen during creation of the object.

        Parameters
        ----------
        mean : np.ndarray
            Mean of the distribution.
        scaled_sigma : np.ndarray
            Logarithm of the variance of the distribution.

        Returns
        -------
        np.ndarray
            Samples drawn from the Gaussian distribution with given mean and variance.
        """
        if self.qmc:
            dist = qmc.MultivariateNormalQMC(mean=mean, cov=np.diag(self.get_variance(scaled_sigma)))
            samples = dist.random(num_samples)
        else:
            if self.dim == 1:
                samples = np.random.normal(mean, np.sqrt(self.get_variance(scaled_sigma)), size=(num_samples, self.dim)) #this function need std. dev while others need variance
            else:
                samples = np.random.multivariate_normal(mean, np.diag(self.get_variance(scaled_sigma)), size=(num_samples,))
        return samples

    def function_wrapper(self, x:np.ndarray, **kwargs):
        """_summary_

        Parameters
        ----------
        x : np.ndarray
            Concatenated array of the mean and the scaled sigma where the objective function and penalty needs to be evaluated.

        Returns
        -------
        np.ndarray
            Sum of mean of function and the penalty.
        np.ndarray
            Sum of gradient of the function and the penalty.
        """
        mean, scaled_sigma = x[:self.dim].detach().numpy(), x[self.dim:].detach().numpy()
        if self.adaptive_sample_size:
            aug_func_mean, aug_func_grad = self.new_estimator_mean_and_derivative(mean, scaled_sigma, **kwargs)
        else:
            aug_func_mean, aug_func_grad = self.estimator_mean_and_derivative(self.func, mean, scaled_sigma)
        return aug_func_mean, aug_func_grad

    @abstractmethod
    def estimator_mean_and_derivative(self, func, mean, samples, scaled_sigma):
        pass

    @abstractmethod
    def new_estimator_mean_and_derivative(self, mean, scaled_sigma):
        pass

class NoVarianceReduction(ObjectiveAbstract):
    """Class to evaluate the gradient without variance reduction.

    """

    def __init__(self, dim: int, func: callable, constraints: list, **kwargs):
        """Constructor

        Parameters
        ----------
         dim : int
            Dimension of the original objective function.
        func : callable
            The objective function.
        constraints : list
            List of callables representing the left-hand-side of less than equal to inequality constraint.
        """
        super().__init__(dim, func, constraints, **kwargs)
  

    def estimator_mean_and_derivative(self, func: callable, mean: np.ndarray, 
                                      samples: np.ndarray, grad_logpdf_val: np.ndarray):
        """_summary_

        Parameters
        ----------
        func : callable
            Function whose gradient needs evaluations.
        mean : np.ndarray
            mean of the samples
        samples : np.ndarray
            Samples drawn from the distribution.
        grad_logpdf_val : np.ndarray
            Gradient of logarithm of density evaluated at samples.

        Returns
        -------
        np.ndarray
            Estimated mean of function.
        np.ndarray
            Estimated gradient of the function.
        """
        f_val = func(samples)
        f_var_red = np.reshape(f_val, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return np.mean(f_val), grad_val
    
    def new_estimator_mean_and_derivative(self, mean, scaled_sigma):
        pass


class Baseline1(ObjectiveAbstract):
    """Objective function where the gradient evaluation is done using the method describe in Welling paper.
    https://openreview.net/pdf?id=r1lgTGL5DE
    This is the recommened baseline to use.
    """

    def __init__(self, dim: int, func: callable, constraints: list, **kwargs):
        """Constructor

        Parameters
        ----------
         dim : int
            Dimension of the original objective function.
        func : callable
            The objective function.
        constraints : list
            List of callables representing the left-hand-side of less than equal to inequality constraint.
        """
        super().__init__(dim, func, constraints, **kwargs)
        self.sample_size_list = []
  

    def estimator_mean_and_derivative(self, func: callable, mean: np.ndarray, scaled_sigma: np.ndarray):
        """_summary_

        Parameters
        ----------
        func : callable
            Function whose gradient needs evaluations.
        mean : np.ndarray
            mean of the samples
        samples : np.ndarray
            Samples drawn from the distribution.
        grad_logpdf_val : np.ndarray
            Gradient of logarithm of density evaluated at samples.

        Returns
        -------
        np.ndarray
            Estimated mean of function.
        np.ndarray
            Estimated gradient of the function.
        """
        samples = self.sampler(mean, scaled_sigma, self.num_samples)
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        f_val = func(samples)
        constraints_val = self.get_penalty(mean, samples)
        augmented_val = f_val.ravel() + constraints_val.ravel()
        s = np.sum(augmented_val)
        B = (s - augmented_val)/(self.num_samples - 1)
        f_var_red = np.reshape(augmented_val - B, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        self.sample_size_list.append(self.num_samples)
        return np.mean(augmented_val), grad_val

    def new_estimator_mean_and_derivative(self, mean: np.ndarray, scaled_sigma: np.ndarray, 
                                          lr=0.01, eps_kl=0.01, smallest_num_samples=8, 
                                          biggest_num_samples=256, natural_gradients=False):
        num_samples = self.num_samples
        samples = self.sampler(mean, scaled_sigma, num_samples)
        f_val = self.func(samples)
        constraints_val = self.get_penalty(mean, samples)
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        # print(np.mean(constraints_val), mean)
        while True:
            augmented_val = f_val.ravel() + constraints_val.ravel()
            s = np.sum(augmented_val)
            B = (s - augmented_val)/(num_samples - 1)
            f_var_red = np.reshape(augmented_val - B, (num_samples,1))
            temp = f_var_red * grad_logpdf_val
            grad_mean = np.mean(temp, axis=0)
            grad_var = np.var(temp, axis=0)  
            if natural_gradients:
                # print('Using natural gradients')
                suggested_num_samples = np.sum((lr**2 * grad_var[:self.dim] * self.get_variance(scaled_sigma))/(2* eps_kl))
            else: 
                suggested_num_samples = np.sum((lr**2 * grad_var[:self.dim])/(2* eps_kl * self.get_variance(scaled_sigma)))
            min_num_samples = int(max(smallest_num_samples, suggested_num_samples))
            min_num_samples = min(biggest_num_samples, min_num_samples)
            # print("Current num samples and suggested num samples", num_samples, min_num_samples, grad_var.shape, grad_var[:self.dim])
            if min_num_samples - num_samples < 5:
                self.num_samples = min_num_samples
                # print("Final number of samples", num_samples)
                break
            extra_num_samples = min_num_samples - num_samples
            num_samples = min_num_samples
            new_samples = self.sampler(mean, scaled_sigma, extra_num_samples)
            new_f_val = self.func(new_samples)
            new_constraints_val = self.get_penalty(mean, new_samples)
            f_val = np.concatenate((f_val, new_f_val),axis=0)
            constraints_val = np.concatenate((constraints_val, new_constraints_val), axis=0)
            new_grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, new_samples)
            grad_logpdf_val = np.concatenate((grad_logpdf_val, new_grad_logpdf_val), axis=0)
            samples = np.concatenate((samples, new_samples), axis=0)
        self.sample_size_list.append(num_samples)
        return np.mean(f_val), grad_mean
    
    def reset_values(self):
        self.f_val = np.array([])
        self.constraints_val = np.array([])
        self.grad_logpdf_val = np.array([])
        self.samples = np.array([])
        # self.sample_size_list = []
    
    def mf_objective(self, num_samples: int, mean: np.ndarray, scaled_sigma: np.ndarray, natural_gradients: bool=False):
        samples = self.sampler(mean, scaled_sigma, num_samples)
        if len(self.samples) == 0:
            self.samples = deepcopy(samples)
        else:
            self.samples = np.concatenate((samples, self.samples), axis=0)
        if len(self.f_val) == 0:
            self.f_val = self.func(samples)
        else:   
            self.f_val = np.concatenate((self.f_val, self.func(samples)), axis=0)
        if len(self.constraints_val) == 0:
            self.constraints_val = self.get_penalty(mean, samples)
        else:
            self.constraints_val = np.concatenate((self.constraints_val, self.get_penalty(mean, samples)), axis=0)
        self.grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, self.samples)
        augmented_val = self.f_val.ravel() + self.constraints_val.ravel()
        s = np.sum(augmented_val)
        B = (s - augmented_val)/(len(self.samples) - 1)
        f_var_red = np.reshape(augmented_val - B, (len(augmented_val),1))
        temp = f_var_red * self.grad_logpdf_val
        self.grad_mean = np.mean(temp, axis=0)
        # variance = np.var(temp, axis=0)
        # print("Shape of variance", variance.shape, np.mean(self.f_val), self.grad_mean)
        if natural_gradients:
            self.var_of_meangrad = 0.5*np.sum(np.var(temp, axis=0)[:self.dim]*self.get_variance(scaled_sigma))
        else:
            self.var_of_meangrad = 0.5*np.sum(np.var(temp, axis=0)[:self.dim]/self.get_variance(scaled_sigma))
        # print("Variance of the mean gradient", self.var_of_meangrad)
    
    
class Baseline2(ObjectiveAbstract):
    """Objective function where the gradient evaluation is done by direction subtraction from mean value.

    """

    def __init__(self, dim: int, func: callable, constraints: list, **kwargs):
        """Constructor

        Parameters
        ----------
         dim : int
            Dimension of the original objective function.
        func : callable
            The objective function.
        constraints : list
            List of callables representing the left-hand-side of less than equal to inequality constraint.
        """
        super().__init__(dim, func, constraints, **kwargs)
  

    def estimator_mean_and_derivative(self, func: callable, mean: np.ndarray, 
                                      samples: np.ndarray, grad_logpdf_val: np.ndarray):
        """_summary_

        Parameters
        ----------
        func : callable
            Function whose gradient needs evaluations.
        mean : np.ndarray
            mean of the samples
        samples : np.ndarray
            Samples drawn from the distribution.
        grad_logpdf_val : np.ndarray
            Gradient of logarithm of density evaluated at samples.

        Returns
        -------
        np.ndarray
            Estimated mean of function.
        np.ndarray
            Estimated gradient of the function.
        """
        f_val, f_mean = func(samples), func(mean)
        f_var_red = np.reshape(f_val - f_mean, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0) 
        # TODO: Look at the noise(variance) of grad_val. I expect this to be higher than biased version. 
        return np.mean(f_val), grad_val
    
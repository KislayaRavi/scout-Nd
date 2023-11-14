import numpy as np
import torch 
from scipy.stats import qmc
from abc import ABC, abstractmethod
   
class ObjectiveAbstract(ABC):
    """"Base class for objective function in *scout-Nd*.

    """

    def __init__(self, dim: int, func: callable, constraints: list, 
                 distribution='gaussian', num_samples=128, qmc=True, 
                 qmc_engine='Sobol', log_lambdas: np.ndarray=None,
                 correct_constraint_derivative=True):
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

    def create_constraints_list(self, constraints: list) -> None:
        """Create the attribute of list of constraints for the class.

        Parameters
        ----------
        constraints : list
            List of callables representing the left-hand-side of less than equal to inequality constraint.
        """
        for i in range(self.num_constraints):
            def constraint_wrapper(x):
                y = constraints[i](x)
                y[y<0] = 0
                return y
            self.constraints.append(constraint_wrapper)

    def deterministic_penalty_definition(self, constraint: callable, mean: float, 
                                         samples: np.ndarray, grad_logpdf_val: np.ndarray):
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
        mean_val, grad = self.estimator_mean_and_derivative(constraint, mean, samples, grad_logpdf_val)
        c_val = constraint(mean)
        if c_val <= 0 and self.correct_constraint_derivative:
            mean_val = 0
            grad[:self.dim] = np.zeros(self.dim)
        return mean_val, grad

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
    
    def get_penalty(self, mean: float, samples: np.ndarray, grad_logpdf_val: np.ndarray):
        """Accumulates penalty from all the constraints, multiplies it with lambdas and returns the total penalty and its gradient. 

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
        np.ndarray
            Estimated gradient of the objective evaluation at mean
        """
        if self.num_constraints == 0:
            return 0, 0
        else:
            mean_array, grad_array = np.zeros(self.num_constraints), np.zeros((self.num_constraints, 2*self.dim))
            for idx, constraint in enumerate(self.constraints):
                mean_array[idx], grad_array[idx, :] = self.deterministic_penalty_definition(constraint, mean, samples, grad_logpdf_val)
            return np.sum(self.lambdas*mean_array), np.sum(self.lambdas[:,None]*grad_array, axis=0)
        
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

    def sampler(self, mean, scaled_sigma):
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
            dist = qmc.MultivariateNormalQMC(mean=mean, cov=np.diag(np.exp(scaled_sigma)))
            samples = dist.random(self.num_samples)
        else:
            if self.dim == 1:
                samples = np.random.normal(mean, np.exp(scaled_sigma), size=(self.num_samples, self.dim))
            else:
                samples = np.random.multivariate_normal(mean, np.diag(np.exp(scaled_sigma)**2), size=(self.num_samples,))
        return samples

    def function_wrapper(self, x:np.ndarray):
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
        samples = self.sampler(mean, scaled_sigma)
        grad_logpdf_val = self.grad_logpdf(mean, scaled_sigma, samples)
        mean_func, grad_func = self.estimator_mean_and_derivative(self.func, mean, samples, grad_logpdf_val)
        mean_penalty, grad_penalty = self.get_penalty(mean, samples, grad_logpdf_val)
        return mean_func + mean_penalty, grad_func + grad_penalty

    @abstractmethod
    def estimator_mean_and_derivative(self, func, mean, samples, grad_logpdf_val):
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


class Baseline1(ObjectiveAbstract):
    """Objective function where the gradient evaluation is done using the method describe in Welling paper.

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
        s = np.sum(f_val)
        B = (s - f_val)/(self.num_samples - 1)
        f_var_red = np.reshape(f_val - B, (self.num_samples,1))
        grad_val = np.mean(f_var_red * grad_logpdf_val, axis=0)
        return np.mean(f_val), grad_val
    
    
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
    
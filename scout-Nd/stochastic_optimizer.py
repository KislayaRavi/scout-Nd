import numpy as np
from scipy.optimize import  minimize
import torch
from objective_function import *
import matplotlib.pyplot as plt
seed = 0


class Stochastic_Optimizer():
    """Class that encapsulates all the optimization parameters.

    """

    def __init__(self, objective: ObjectiveAbstract, **kwargs:dict):
        """Initializer
        
        Parameters
        ----------
        objective : ObjectiveAbstract
            Object of type ObjectiveAbstract that contains all the information about the function and constraints.
        """
        self.objective = objective
        self.dim = self.objective.dim
        if 'initial_val' in kwargs.keys():
            self.set_initial_val(kwargs['initial_val'])
        else:
            x0 = np.ones(2*self.dim)
            x0[self.dim:] = -1
            self.set_initial_val(x0)
        self.stored_results = [self.initial_val]        
        self.optimizer = None

    def set_initial_val(self, initial_val:np.ndarray):
        """_summary_

        Parameters
        ----------
        initial_val : np.ndarray
            Starting point of the optimization algorithm.
        """
        assert len(initial_val) == 2*self.dim, 'Incorrect length of initial values, it should be equal to 2*dim'
        self.initial_val = torch.tensor(initial_val, requires_grad=True)
        self.parameters = [self.initial_val] # TODO: transfer it to objective. This does not make sense in optimizer

    def create_optimizer(self, name_optimizer:str, **optimizer_parameters:dict):
        name_optimizer = name_optimizer.lower()
        if name_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'adamax':
            self.optimizer = torch.optim.Adamax(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'asgd':
            self.optimizer = torch.optim.ASGD(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'nadam':
            self.optimizer = torch.optim.NAdam(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'radam':
            self.optimizer = torch.optim.RAdam(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'rprop':
            self.optimizer = torch.optim.Rprop(self.parameters, **optimizer_parameters)
        elif name_optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters, **optimizer_parameters)
        else:
            raise NotImplementedError('Either the name of optimizer is wrong or it is not yet implemented in the code')

    def optimize_fixed_lambda(self, log_lambdas:np.ndarray, num_steps_per_lambda:int):
        """Performs optimization for fix value of lambdas for the given number of steps.

        Parameters
        ----------
        log_lambdas : np.ndarray
            The exponential logarithm of the penalty scaling factor.
        num_steps_per_lambda : int
            Number of optimization steps with fixed value of lambdas.
        """
        self.objective.update_lambdas(log_lambdas)
        for i in range(num_steps_per_lambda):
            val, grad = obj.function_wrapper(self.parameters[0])
            self.parameters[0].grad = torch.tensor(grad) # Tis could be problemtic in GPUs when device is not set correctly
            self.optimizer.step()
            self.stored_results.append(self.parameters[0])

    def constrained_optimization(self, initial_log_lambdas:int=-1, num_lambdas:int=4, num_steps_per_lambda:int=100):
        """Function that performs constrained optimization.

        Parameters
        ----------
        initial_log_lambdas : int, optional
            starting values of exponential logarithm of lambda, by default -1
        num_lambdas : int, optional
            Number of lambda updates, by default 4
        num_steps_per_lambda : int, optional
            Number of optimization steps with fixed value of lambdas., by default 100
        """
        if self.optimizer is None:
            print('Optimizer is not created, reverting to default Adam optimizer with lr 1e-2')
            self.optimizer = torch.optim.Adam(self.parameters[0], lr=1e-2)
        lambdas = initial_log_lambdas*np.ones(self.objective.num_constraints) 
        for i in range(num_lambdas):
            self.optimize_fixed_lambda(lambdas, num_steps_per_lambda)
            print(self.objective.lambdas, self.get_final_state())
            lambdas = lambdas + 1

    def unconstrained_optimization(self, num_steps: int=100):
        """Function that performs unconstrained optimization.

        Parameters
        ----------
        num_steps : int, optional
            number of optimization steps, by default 100
        """
        for i in range(num_steps):
            val, grad = obj.function_wrapper(self.parameters[0])
            self.parameters[0].grad = torch.tensor(grad) # Tis could be problemtic in GPUs when device is not set correctly
            self.optimizer.step()
            self.stored_results.append(self.parameters[0])
    
    def optimize(self, **kwargs):
        """Function to optmize the objective function
        """
        if self.objective.num_constraints > 0:
            self.constrained_optimization(**kwargs)
        else:
            self.unconstrained_optimization(**kwargs)
    
    def get_final_state(self):
        """Reurns the finals optimization results.

        Returns
        -------
        dict
            Dictionary of optimum mean and variance.
        """
        return {'mean':self.stored_results[-1][:self.dim], 'variance':self.stored_results[-1][self.dim:]}
    

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
    # constraints = None
    obj = Baseline1(dim, sphere, constraints, num_samples=128)
    optimizer = Stochastic_Optimizer(obj)
    optimizer.create_optimizer('Adam', lr=1e-2)
    optimizer.optimize()
    print(optimizer.get_final_state())
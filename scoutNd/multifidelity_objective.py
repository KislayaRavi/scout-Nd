import numpy as np
from scoutNd.objective_function import *
from scoutNd.stochastic_optimizer import *


class MultifidelityObjective():
    """Class for multi-fidelity objective function
    """

    def __init__(self, dim: int, f_list: list, constraints: list,
                 distribution: str='gaussian', qmc: bool=True,
                 qmc_engine: str='Sobol', log_lambdas: np.ndarray=None,
                 correct_constraint_derivative: bool=True,
                 category_var_red: str='Baseline1') -> None:
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
        category_var_red : str, optional
            Type of variance reduction technique to be used in gradient approxiation.
        """
        
        self.dim = dim
        if constraints is None:
            self.num_constraints = 0
        else:
            self.num_constraints = len(constraints)
        self.num_fidelities, self.f_list = len(f_list), f_list
        self.list_objective = []
        self.num_samples = [1]*self.num_fidelities
        self.initialize_list(f_list, constraints, distribution, qmc, qmc_engine,
                             log_lambdas, correct_constraint_derivative, 
                             category_var_red)
        if self.num_constraints > 0:
            self.update_lambdas(log_lambdas)
        self.constrain_values = []  # placeholder for constraint values, size : no_constraints x 1
        self.objective_value = []

    def set_num_samples(self, num_samples: np.ndarray)-> None:
        """Sets the array of the number of samples to be used for each estimator in the telscopic sum.

        Parameters
        ----------
        num_samples : np.ndarray
            Array of the number of samples to evaluate each estimator in the telescopic sum.
        """
        assert len(num_samples) == self.num_fidelities, 'List of number of samples should be equal to the number of fidelities'
        self.num_samples = num_samples
        for idx, l in enumerate(self.list_objective):
            l.num_samples = num_samples[idx]

    def initialize_list(self, f_list: list, constraints: list,
                        distribution: str, qmc: bool,
                        qmc_engine: str, log_lambdas: np.ndarray,
                        correct_constraint_derivative: bool,
                        category_var_red: str) -> None:
        """Initializes the list of objective function in the telescopic sum formula for multi-fidelity gradient approximation.

        Parameters
        ----------
        f_list : list
            List of function callables, fidelity increases with index
        constraints : list
            List of callables representing the left-hand-side of less than equal to inequality constraint.
        distribution : str
            name of the distribution of the parameters.
        qmc : bool
            Boolean variable to turn on/off the QMC sampling strategy.
        qmc_engine : str
            Name of the QMC sampling method.
        log_lambdas : np.ndarray
            Exponential logarithm of the penalty coefficients.
        correct_constraint_derivative : bool
            If constrained is satisfied then set the derivative of the w.r.t $\mu$ to zero. 
        category_var_red : str
            Type of variance reduction technique to be used in gradient approxiation.
        """

        self.list_objective.append(self.create_sf_objective_object(f_list[0],
                                                                   constraints, self.num_samples[0],
                                                                   distribution, qmc,
                                                                   qmc_engine, log_lambdas,
                                                                   correct_constraint_derivative,
                                                                   category_var_red))
        for i in range(1, self.num_fidelities):
            def func_def(x):
                if type(f_list[i](x)) == list:
                    return np.array(f_list[i](x)) - np.array(f_list[i-1](x))
                else:
                    return f_list[i](x) - f_list[i-1](x)
                
            self.list_objective.append(
                self.create_sf_objective_object(func_def, None, self.num_samples[i], distribution,
                                                qmc, qmc_engine, None,
                                                correct_constraint_derivative,
                                                category_var_red))

    def create_sf_objective_object(self, func: callable, constraints: list, num_samples:int,
                                   distribution: str, qmc:str, qmc_engine:str, 
                                   log_lambdas: np.ndarray,
                                   correct_constraint_derivative: str,
                                   category_var_red: str) -> ObjectiveAbstract:

        """Initializes the list of objective function in the telescopic sum formula for multi-fidelity gradient approximation.

        Parameters
        ----------
        f_list : list
            List of function callables, fidelity increases with index
        constraints : list
            List of callables representing the left-hand-side of less than equal to inequality constraint.
        num_samples : int
            Number of samples used to estimate the gradient.
        distribution : str
            name of the distribution of the parameters.
        qmc : bool
            Boolean variable to turn on/off the QMC sampling strategy.
        qmc_engine : str
            Name of the QMC sampling method.
        log_lambdas : np.ndarray
            Exponential logarithm of the penalty coefficients.
        correct_constraint_derivative : bool
            If constrained is satisfied then set the derivative of the w.r.t $\mu$ to zero. 
        category_var_red : str
            Type of variance reduction technique to be used in gradient approxiation.

        Returns
        -------
        ObjectiveAbstract
            Return object based on the choice of variance reduction technique

        Raises
        ------
        ValueError
            Raises error when the name of the variance reduction technique is wrong.
        """
        category = category_var_red.lower()
        if category == 'novariancereduction':
            return NoVarianceReduction(self.dim, func, constraints, 
                                       num_samples=num_samples, 
                                       distribution=distribution,
                                       qmc=qmc, qmc_engine=qmc_engine,
                                       log_lambdas=log_lambdas,
                                       correct_constraint_derivative=correct_constraint_derivative)
        elif category == 'baseline1':
            return Baseline1(self.dim, func, constraints, 
                             num_samples=num_samples, distribution=distribution,
                             qmc=qmc, qmc_engine=qmc_engine, log_lambdas=log_lambdas,
                             correct_constraint_derivative=correct_constraint_derivative)
        elif category == 'baseline2':
            return Baseline2(self.dim, func, constraints, 
                             num_samples=num_samples, distribution=distribution,
                             qmc=qmc, qmc_engine=qmc_engine, log_lambdas=log_lambdas,
                             correct_constraint_derivative=correct_constraint_derivative)
        else:
            raise ValueError('Wrong name of the variance reduction technique, ' +
                'Options are: NoVarianceReduction, Basline1 and Baseline2')

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
        # Parallelisation over multi-fidelitues will be interesting to implement
        # one of the options could be multi-processing
        # But it will intefere with parallelisation of external program, which will be more harmful
        # So, I have not implemented it here.
        val, grad = 0. , np.zeros(2*self.dim)
        for i in range(self.num_fidelities):
            temp_val, temp_grad = self.list_objective[i].function_wrapper(x)
            val = val + temp_val
            grad = grad + temp_grad
        # adding objective and constraint values. Ugly now. takes values from the lowest fidelity
        self.constrain_values = self.list_objective[0].constrain_values
        self.objective_value = self.list_objective[0].objective_value
        return val, grad

    def update_lambdas(self, log_lambdas: np.ndarray):
        """Updates the scaling of penalty term

        Parameters
        ----------
        log_lambdas : np.ndarray
        """
        if log_lambdas is None:
            self.lambdas = np.exp(np.zeros(self.num_constraints))
        else:
            self.lambdas = np.exp(log_lambdas)
            assert len(self.lambdas) == self.num_constraints, 'Number of constraints should be equal to number of lambdas'
        self.list_objective[0].update_lambdas(log_lambdas)




if __name__ == '__main__':

    def sphere(x):
        X = np.atleast_2d(x)
        val1 = np.sum(X**2, axis=1)
        # val2 = np.random.normal(0, 0.0001, val1.shape)
        val2 = 0.0
        #print(f'val1 is {val1} and val2 is {val2}')
        return val1 + val2

    def sphere_lf(x):
        X = np.atleast_2d(x)
        val1 = np.sum(X**2, axis=1)
        val3 = np.random.normal(0, 0.01, val1.shape)
        val2 = 0.001*np.sum(X, axis=1)
        return val1 + val2 + val3

    def linear_constraint(X):
        x = np.atleast_2d(X)
        return 1 - x[:, 0] - x[:, 1]


    dim = 16
    constraints = [linear_constraint]
    #constraints = None
    obj = MultifidelityObjective(dim, [sphere_lf, sphere], constraints, qmc=True)
    obj.set_num_samples([64, 8])
    optimizer = Stochastic_Optimizer(obj,natural_gradients= True, verbose=True)
    optimizer.create_optimizer('Adam', lr=1e-2)
    optimizer.optimize(num_lambdas =10, num_steps_per_lambda = 300)
    #optimizer.optimize(num_steps = 300)
    print(optimizer.get_final_state())

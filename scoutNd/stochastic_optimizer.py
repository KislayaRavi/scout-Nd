import numpy as np
import torch
from scoutNd.objective_function import *
from scoutNd.viz import variable_evolution
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import time
datetime = time.strftime("%Y%m%d-%H%M%S")

 #seed = 0


class Stochastic_Optimizer():
    """Class that encapsulates all the optimization parameters.

    """

    def __init__(self, objective: ObjectiveAbstract, **kwargs:dict):
        """Initializer
        kwargs can take the following parameters:
        - initial_val: np.ndarray
            Starting point of the optimization algorithm.
        - natural_gradients: bool
            If True, natural gradients are used.
        - verbose: bool
            If True, the optimizer will print the output.
        - tolerance_L_x: float
            Tolerance of L(x) per lambda.
        - tolerance_sigma: float
            Tolerance of sigma^2.
        
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
        self.stored_results = [self.initial_val] # stores evolution of thetas
        self.stored_f_x = [] # stores evolution of f(x) i.e obj + lambda_i c_i(x)
        self.stored_objective_mean = []
        self.stored_constraints_mean = []
        self.stored_lambdas = []
        self.optimizer = None
        self.iteration = 0
        if 'natural_gradients' in kwargs.keys():
            self.natural_gradients = kwargs['natural_gradients']
            print("Natural gradients are being used")
        else:
            self.natural_gradients = False
            print("Natural gradients are not being used")
        if 'verbose' in kwargs.keys():
            self.verbose = kwargs['verbose']
            if self.verbose:
                print("Verbose mode is on.")
        else:
            self.verbose = False
            print("Verbose mode is off.")

        # set tolerance for a given lambda 
        # if 'tolerance_L_x' in kwargs.keys():
        #     self.tolerance_L_x = kwargs['tolerance_L_x']
        # else:
        #     self.tolerance_L_x = 1e-04
        #     print(f'L_2 norm Tolerance of L(x) per lambda is set to {self.tolerance_L_x}')
        if 'tolerance_theta' in kwargs.keys():
            self.tolerance_theta = kwargs['tolerance_theta']
        else:
            self.tolerance_theta = 1e-04
            print(f'L_2 norm Tolerance of theta is set to {self.tolerance_theta}')
        
        # set tolerance for sigma^2
        if 'tolerance_sigma' in kwargs.keys():
            self.tolerance_sigma = kwargs['tolerance_sigma']
        else:
            self.tolerance_sigma = 5e-03
            print(f'L_2 norm Tolerance of sigma^2 is set to {self.tolerance_sigma}')

        # set tolerance for constraints
        if 'tol_constraints' in kwargs.keys():
            self.tol_constraints = kwargs['tol_constraints']
        else:
            self.tol_constraints = 1e-03
            print(f'L_2 norm Tolerance of constraints is set to {self.tol_constraints}')

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
        """Performs optimization for fix value of lambdas for the given number of steps. This is the inner loop of the optimization.
        Synchronous to the "unconstrained_optmization" function below. Changes made here should be replicated there.

        Parameters
        ----------
        log_lambdas : np.ndarray
            The exponential logarithm of the penalty scaling factor.
        num_steps_per_lambda : int
            Number of optimization steps with fixed value of lambdas.
        """
        theta_norm_convergance_criterion = False
        condition_1 = False
        self.objective.update_lambdas(log_lambdas)
        for i in range(num_steps_per_lambda):
            val, grad = self.objective.function_wrapper(self.parameters[0])
            if self.natural_gradients:
                grad = self._get_fim_adjusted_gradient(self.parameters[0][self.dim:].detach().numpy(), grad)
            self.parameters[0].grad = torch.tensor(grad) # Tis could be problemtic in GPUs when device is not set correctly
            self.optimizer.step()
            self.stored_results.append(deepcopy(self.parameters[0])) # storing thetas
            self.stored_f_x.append(val) # storing f(x) values
            self.stored_objective_mean.append(self.objective.objective_value)
            self.stored_constraints_mean.append(self.objective.constrain_values)
            self.stored_lambdas.append(self.objective.lambdas)
            # keep adding the iteration number
            self.iteration+=1 # FIM kicks in after a certain value of iteration.
            #self.iteration = i

            # print the output
            if self.verbose and i%25 == 0:
                print(f'Iteration: {self.iteration}, lambdas: {self.objective.lambdas}, L(x): {val}, f(x): {self.objective.objective_value}, C(x) : {self.objective.constrain_values}, theta_mean : {self.stored_results[-1][:self.dim]}, theta_beta : {self.stored_results[-1][self.dim:]} ')

            # convergance criterion here
            # if ||f_x -f_x_prev|| < 1e-06 break
            #if len(self.stored_f_x) > 1 and np.linalg.norm(val - self.stored_f_x[-2]) <= self.tolerance_L_x:

            # break condition 1: 20 iterations have passed and each entry of the constraint is more than the tolearnce. meaning lambda is not able to decrease the constraint. this wouldnt be triggered when
            # we start from constraint satisfaction region
            if i > 20 and all(val > self.tol_constraints for val in self.stored_constraints_mean[-10:]):
            #if self.iteration > 20 and np.linalg.norm(self.stored_constraints_mean[-5:]) > self.tol_constraints:
                condition_1 = True
                print(f"-------------------------------------------\n"
                f"Inner loop termination criterion 1. The Lambda is not enough and needs to be increased. iteration number j: {i} for the lambda : {self.objective.lambdas}\n"
                f"-------------------------------------------\n")
                break
                

            # break condition 2: if || theta_i - theta_i-1|| < tol break
            if len(self.stored_results) >2:
                with torch.no_grad():
                    # write RMSe with torch
                    rmse = torch.sqrt(torch.mean(torch.square(self.stored_results[-1][:self.dim] - self.stored_results[-2][:self.dim]))) 
                if rmse.item() <= self.tolerance_theta:
                    # print(f"----------------------------------------\n"
                    # f"L2 error of L(x) (augmented objective) convergance criterion met at iteration: {self.iteration} for lambda : {self.objective.lambdas} with norm: {np.linalg.norm(val - self.stored_f_x[-2])} \n"
                    #     f"----------------------------------------\n")
                    print(f"----------------------------------------\n"
                    f"Inner loop termination condition 2. L2 error of mean(x) convergance criterion met at iteration # j: {i} for lambda : {self.objective.lambdas} with RMSE: {rmse.item()} \n"
                        f"----------------------------------------\n")
                    theta_norm_convergance_criterion = True
                    break
        if theta_norm_convergance_criterion==False and condition_1 ==False:
            print(f"-------------------------------------------\n"
              f"Inner loop termination condition 3. Total no. of iterations crtirion met. The maximum number of inner-loop iterations : {num_steps_per_lambda} is reached for the given lambda : {self.objective.lambdas}\n"
              f"-------------------------------------------\n")

                
    
    def _get_fim_adjusted_gradient(self, phi:np.ndarray, grad:np.ndarray):
        """Computes the adjusted gradient using the Fisher information matrix.

        Parameters
        ----------
        phi : np.ndarray
            sigma**2  = e^(2*phi)
        grad : np.ndarray [2d,1]
            Gradient of the objective function.

        Returns
        -------
        np.ndarray
            Adjusted gradient.
        """
        assert phi.shape == (self.dim,), 'Incorrect shape of phi'
        assert grad.shape == (2*self.dim,), 'Incorrect shape of grad'
        fim = self._fisher_information_matrix(phi)

        # --- by preinverting the matrix
        fim_inv = np.matrix(fim).I
        tilda_grad_U = np.matmul(fim_inv, grad).base # to return an array instead of matrix

        # --- by solving the linear system
        #tilda_grad_U_ = np.linalg.solve(fim, grad)
        return tilda_grad_U
    
    def _fisher_information_matrix(self,phi):
        """Computes the Fisher information matrix.

        Parameters
        ----------
        phi : np.ndarray 
            sigma**2  = e^(2*phi)

        Returns
        -------
        np.ndarray
            Fisher information matrix.
        """
        assert phi.shape == (self.dim,), 'Incorrect shape of phi'
        tmp = np.exp(-2*phi)
        fisher_diag = np.hstack((tmp,2*np.ones(self.dim)))
        fim = np.diag(fisher_diag)

        #TODO add damped fim here
        fim_dampening_coeff = 1e-1
        dampening_coeff_lower_bound = 1e-8
        fim_decay_start = 50
        if self.iteration > fim_decay_start:
            tmp = fim_dampening_coeff*np.exp(-(self.iteration - fim_decay_start)/fim_decay_start)
            dampening_coeff = max(tmp, dampening_coeff_lower_bound)
        else:
            dampening_coeff = fim_dampening_coeff
        
        fim = fim + dampening_coeff*np.eye(2*self.dim)
        # check if FIM is invertible
        # if np.linalg.cond(fim) > 1/sys.float_info.epsilon:
        #     raise ValueError(f'Fisher information matrix is not invertible, it is: {fim}')
        return fim

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
        sigma_norm_convergance_criterion = False
        if self.optimizer is None:
            print('Optimizer is not created, reverting to default Adam optimizer with lr 1e-2')
            self.optimizer = torch.optim.Adam(self.parameters[0], lr=1e-2)
        lambdas = initial_log_lambdas*np.ones(self.objective.num_constraints) 
        for i in range(num_lambdas):
            self.optimize_fixed_lambda(lambdas, num_steps_per_lambda)
            print(self.objective.lambdas, self.get_final_state())
            lambdas = lambdas + 1

            # convergance criterion for the outher loop
            # if cnstraint are less the specied value and (||sigma^2|| <-= tol break)
            #l2_norm = torch.norm(torch.exp(self.parameters[0][self.dim:]))
            # TODO: can add || |x_lambda* - x_{lambda-1}*|| < tol also as a convergance criterion
            with torch.no_grad():
                rms = torch.sqrt(torch.mean(torch.square(torch.exp(self.parameters[0][self.dim:]))))
            if rms.item() <= self.tolerance_sigma:
            #if np.linalg.norm(self.stored_constraints_mean[-1])<=self.tol_constraints  and rms.item() <= self.tolerance_sigma:
            #if np.linalg.norm(self.stored_constraints_mean[-1])<=self.tol_constraints  and l2_norm.item() <= self.tolerance_sigma:
                print(f"----------------------------------------\n"
                      f"Outer loop terminating. L2 norm of sigma^2 convergance criterion met at iteration: {self.iteration}\n"
                      f"----------------------------------------\n")
                sigma_norm_convergance_criterion = True
                break
        if not sigma_norm_convergance_criterion:
            print(f"-------------------------------------------\n"
               f" Outer loop terminating. Max no. of iterations crtirion met. The number of iterations : {self.iteration} is reached for the given lambda : {self.objective.lambdas}\n"
                f"-------------------------------------------\n")


    def unconstrained_optimization(self, num_steps: int=100):
        """Function that performs unconstrained optimization.
        Synchronous to the "get_fixed_lambda" function above. 

        Parameters
        ----------
        num_steps : int, optional
            number of optimization steps, by default 100
        """
        sigma_convergance_criterion = False
        for i in range(num_steps):
            val, grad = self.objective.function_wrapper(self.parameters[0])
            if self.natural_gradients:
                grad = self._get_fim_adjusted_gradient(self.parameters[0][self.dim:].detach().numpy(), grad)
            self.parameters[0].grad = torch.tensor(grad) # Tis could be problemtic in GPUs when device is not set correctly
            self.optimizer.step()
            # store numpy of parmeters in stored_results
            self.stored_results.append(deepcopy(self.parameters[0]))
            self.stored_f_x.append(val) # storing f(x) values
            self.stored_constraints_mean.append(self.objective.objective_value)
            self.iteration = i

            # print the output
            if self.verbose and i%25 == 0:
                print(f'Iteration: {i}, L(x): {val}, f(x): {self.objective.objective_value}, theta_mean : {self.stored_results[-1][:self.dim]}, theta_beta : {self.stored_results[-1][self.dim:]} ')

            # convergance criterion if (||sigma^2|| <-= 10^-06 break)
            #l2_norm = torch.norm(torch.exp(self.parameters[0][self.dim:]))
            
            with torch.no_grad():
                rms = torch.sqrt(torch.mean(torch.square(torch.exp(self.parameters[0][self.dim:]))))
            if i%100 == 0:
                print(f"RMS of sigma^2 is {rms.item()} at iteration: {i}")
            #if l2_norm.item() <= self.tolerance_sigma:
            if rms.item() <= self.tolerance_sigma:
                print(f"----------------------------------------\n"
                      f"Root mean sq. of sigma^2 is {rms.item()}, convergance criterion met at iteration: {i}\n"
                      f"----------------------------------------\n")
                sigma_convergance_criterion = True
                break
        if not sigma_convergance_criterion:
            print(f"-------------------------------------------\n"
                f"no. of iterations crtirion met. The maximum number of iterations : {num_steps} is reached.\n"
                f"-------------------------------------------\n")
    
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
    
    def get_objective_constraint_evolution(self):
            """
            Returns the evolution of the objective and constraint values.

            Returns:
                If there are constraints, returns a tuple containing the augmented objective mean,
                objective mean, and constraints mean and lambdas as numpy arrays.
                If there are no constraints, returns a tuple containing the augmented objective mean
                and objective mean as numpy arrays.
            """
           
            aug_objective_mean = np.array(self.stored_f_x)
            objective_mean = np.array(self.stored_objective_mean)
            

            if self.objective.num_constraints > 0:
                constraints_mean = np.array(self.stored_constraints_mean)
                lambdas = np.array(self.stored_lambdas)
                return aug_objective_mean, objective_mean, constraints_mean, lambdas
            else:
                return aug_objective_mean, objective_mean
        
    def get_design_variable_evolution(self):
            """
            Returns the evolution of the design variables.

            Returns:
                x_mean (ndarray): [Nx dim] with N being of step. The evolution of the mean design variables.
                x_beta (ndarray): The evolution of the beta design variables.
            """
            # convert list of tensors in store_results to numpy array
            des_variables = np.array([result.detach().numpy() for result in self.stored_results])
            x_mean = des_variables[:, :self.dim]
            x_beta = des_variables[:, self.dim:]
            return x_mean, x_beta
    
    def save_results(self, path:str):
        """Saves the results of the optimization in the given path.

        Parameters
        ----------
        path : str
            Path to save the results.
        """
        
        x_mean, x_beta = self.get_design_variable_evolution()
        np.save(f'{path}/design_variable_mean_evolution_{datetime}.npy', x_mean)
        np.save(f'{path}/design_variable_beta_evolution_{datetime}.npy', x_beta)


        np.save(f'{path}/final_state_{datetime}.npy', self.get_final_state())

        if self.objective.num_constraints > 0:
            aug_obj, obj, constraints, lambdas = self.get_objective_constraint_evolution()
            np.save(f'{path}/constraint_evolution_{datetime}.npy', constraints)
            np.save(f'{path}/lambda_evolution_{datetime}.npy', lambdas)
            np.save(f'{path}/objective_evolution_{datetime}.npy', obj)
            np.save(f'{path}/augmented_objective_evolution_{datetime}.npy', aug_obj)
        else:
            aug_obj, obj = self.get_objective_constraint_evolution()
            np.save(f'{path}/objective_evolution_{datetime}.npy', obj)
            np.save(f'{path}/augmented_objective_evolution_{datetime}.npy', aug_obj)

        
    def plot_results(self, path:str, save_name:str):
            """
            Plot the results of the stochastic optimizer.

            Parameters:
            path (str): The path where the plots will be saved.
            save_name (str): The name of the saved plot file.
            """
            x_mean, x_beta = self.get_design_variable_evolution()

            if self.objective.num_constraints > 0:
                aug_obj, obj, constraints, lambdas = self.get_objective_constraint_evolution()
                ve = variable_evolution(L_x=aug_obj, f_x=obj, mu=x_mean, beta=x_beta, path=path, save_name=save_name,C_x=constraints,lambdas=lambdas)
            else:
                aug_obj, obj = self.get_objective_constraint_evolution()
                ve = variable_evolution(L_x=aug_obj, f_x=obj, mu=x_mean, beta=x_beta, path=path, save_name=save_name)   
            ve.plot_all()

        

        
        
    

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
    dim = 32
    constraints = [linear_constraint]
    # constraints = None
    obj = Baseline1(dim, sphere, constraints, num_samples=16, qmc=True, correct_constraint_derivative=True)
    optimizer = Stochastic_Optimizer(obj)
    optimizer.create_optimizer('Adam', lr=1e-2)
    optimizer.optimize()
    print(optimizer.get_final_state())

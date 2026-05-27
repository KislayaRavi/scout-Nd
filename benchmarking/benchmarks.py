"""Benchmark function suite (18 functions, HF/LF variants) used by the SCOUT-Nd paper studies.

Each class derives from ``AbstractBenchmark`` and exposes ``evaluate_hf`` (high-fidelity),
``evaluate_lf`` (low-fidelity), ``get_bounds``, ``get_global_optimum`` and ``get_constraints``.
Source of formulae: https://www.sfu.ca/~ssurjano/optimization.html
"""

from typing import Any
import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import product

'''Source : https://www.sfu.ca/~ssurjano/optimization.html
'''



class AbstractBenchmark(metaclass=ABCMeta):

    def __init__(self, dim, add_noise=False):
        self._dim = dim
        self._add_noise = add_noise
        self.noise_level = 0.0001
        # Following variables are used for scipy and cbo
        self.count, self.eval_val, self.eval_points = 0, [], []

    def get_dim(self):
        return self._dim
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.evaluate_hf(args[0])

    def normalize_params(self, x):
        lower, upper = self.get_bounds()
        return (x - lower) / (upper - lower)
    
    def denormalize_params(self, x):
        lower, upper = self.get_bounds()
        return x * (upper - lower) + lower
    
    # This function is used for scipy and cbo
    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    # This function is used for scipy and cbo
    def evaluate_hf_mean(self, X):
        val = []
        self.count += self.num_samples
        # print(X)
        for i in range(self.num_samples):
            self.eval_points.append(X)
            hf_val = self.evaluate_hf(X)
            val.append(hf_val)
            self.eval_val.append(hf_val)
        return np.mean(val)

    def evaluate_hf(self, parameter):
        X = np.atleast_2d(parameter)
        assert X.shape[1] == self._dim, "Incorrect dimension"
        Y = np.zeros((X.shape[0],1))
        for idx, x in enumerate(X):
            Y[idx, 0] = self.evaluate_one_hf(x)
        if self._add_noise:
            Y += np.random.normal(0, self.noise_level, Y.shape)
        return Y 

    def evaluate_lf(self, parameter):
        X = np.atleast_2d(parameter)
        assert X.shape[1] == self._dim, "Incorrect dimension"
        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            Y[idx, 0] = self.evaluate_one_lf(x)
        if self._add_noise:
            Y += np.random.normal(0, self.noise_level, Y.shape)
        return Y 

    def plot(self, category='hf'):
        if self._dim == 1:
            self.plot_1d(category=category)
        elif self._dim == 2:
            self.plot_2d(category=category)
        else:
            raise ValueError("Cannot plot for more that 2 dimensions")
        plt.show()

    def plot_1d(self, num_mesh=100, category='hf'):
        lower, upper = self.get_bounds()
        x = np.linspace(lower, upper, num_mesh)
        if category=='hf':
            y = self.evaluate_hf(x)
        elif category=='lf':
            y = self.evaluate_lf(x)
        else:
            y = self.evaluate_hf(x) - self.evaluate_lf(x)
        plt.plot(x, y)
        plt.title(self.get_name())
        plt.xlabel(r'$x$')
        plt.ylabel(r'$f(x)$')

    def plot_2d(self, num_mesh=100, category='hf'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lower, upper = self.get_bounds()
        x = np.linspace(lower[0], upper[0], num_mesh)
        y = np.linspace(lower[1], upper[1], num_mesh)
        X, Y = np.meshgrid(x, y)
        temp = np.array(list(product(x,y)))
        if category=='hf':
            zs = self.evaluate_hf(temp)
        elif category=='lf':
            zs = self.evaluate_lf(temp)
        else:
            zs = self.evaluate_hf(temp) - self.evaluate_lf(temp)
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$f(x_1, x_2)$')
        ax.set_title(self.get_name())
    
    def get_optimum_value(self):
        return self.evaluate_one_hf(self.get_global_optimum())
    
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def evaluate_one_hf(self, x):
        pass

    @abstractmethod
    def evaluate_one_lf(self, x):
        pass 

    @abstractmethod
    def get_global_optimum(self):
        pass 

    @abstractmethod
    def get_bounds(self):
        pass 

    @abstractmethod
    def get_constraints(self):
        pass

class SphereFunction(AbstractBenchmark):
    '''The function is convex and unimodal, which helps optimization algorithms efficiently find the global minimum.
    '''

    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)

    def get_name(self):
        return "Sphere Function"

    def evaluate_one_hf(self, x):
        return np.sum(x**2)

    def evaluate_one_lf(self, x):
        return 1.1*np.sum(x**2)

    def get_global_optimum(self):
        return np.zeros(self._dim)

    def get_bounds(self):
        return np.ones(self._dim)*-6, np.ones(self._dim)*6

    def get_constraints(self):
        return []

class ConstraintSphereFunction(AbstractBenchmark):
    '''The function is convex and unimodal, which helps optimization algorithms efficiently find the global minimum.
    '''

    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)
        if dim < 2:
            raise ValueError("Dimension should be at least 2")

    def get_name(self):
        return "Constraint Sphere Function"

    def evaluate_one_hf(self, x):
        return np.sum(x**2)

    def evaluate_one_lf(self, x):
        return 1.1*np.sum(x**2)

    def get_global_optimum(self):
        if self._dim > 2:
            x_star = [0.5]*2 + [0.]*(self._dim-2)
        else:
            x_star = [0.5]*2
        return np.array(x_star)

    def get_bounds(self):
        return np.ones(self._dim)*-6, np.ones(self._dim)*6
    
    def constraint_func(self, X):
        x = np.atleast_2d(X)
        return 1 - x[:, 0] - x[:, 1]

    def get_constraints(self):
        return [lambda x: self.constraint_func(x)]



class AckleyFunction(AbstractBenchmark):
    '''The function poses a risk for optimization algorithms, particularly hillclimbing algorithms, 
    to be trapped in one of its many local minima.
    Recommended variable values are: a = 20, b = 0.2 and c = 2*np.pi
    '''

    def __init__(self, dim, a=20, b=0.2, c=2*np.pi, **kwargs):
        super().__init__(dim, **kwargs)
        self.a, self.b, self.c = a, b, c 
        self.inv_d = 1/dim
        self.constant = self.a + np.exp(1)

    def get_name(self):
        return "Ackley Function"

    def evaluate_one_hf(self, x):
        t1 = -self.a * np.exp(-self.b * np.sqrt(self.inv_d * np.sum(x**2)))
        t2 = -np.exp(self.inv_d * np.sum(np.cos(self.c * x)))
        return t1 + t2 + self.constant 

    def evaluate_one_lf(self, x):
        t1 = -self.a* np.exp(-self.b * np.sqrt(self.inv_d * np.sum(x**2)))
        t2 = -np.exp(self.inv_d * np.sum(np.cos(self.c * x)))
        return 1.1*t1 + 0.9*t2 + self.constant

    def get_global_optimum(self):
        return np.zeros(self._dim)

    def get_bounds(self):
        return np.ones(self._dim)*-10, np.ones(self._dim)*10

    def get_constraints(self):
        return []


class BukinFunction_N6(AbstractBenchmark):
    '''The sixth Bukin function has many local minima, all of which lie in a ridge. 
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Bukin Function number 6"

    def evaluate_one_hf(self, x):
        # self.denormalize_params(x)
        return 100 * np.sqrt(np.abs(x[1] - 0.01*x[0]*x[0])) + 0.01*np.abs(x[0] + 10)

    def evaluate_one_lf(self, x):
        # self.denormalize_params(x)
        return 98 * np.sqrt(np.abs(x[1] - 0.01*x[0]*x[0])) + 0.01*np.abs(x[0] + 9.9)

    def get_global_optimum(self):
        return np.array([-10, 1])

    def get_bounds(self):
        return np.array([-15, -3]), np.array([-5, 3])
    
    def get_constraints(self):
        return []


class DropWave(AbstractBenchmark):
    '''Mutltimodal and highly complex function
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Drop-Wave function"

    def evaluate_one_hf(self, x):
        return -(1+ np.cos(12*np.sqrt(x[0]**2 + x[1]**2))) / (2 + 0.5*(x[0]**2 + x[1]**2))
    
    def evaluate_one_lf(self, x):
        return -1.1*(1+ np.cos(11.9*np.sqrt(x[0]**2 + x[1]**2))) / (2 + 0.55*(x[0]**2 + x[1]**2))

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-2, -2]), np.array([2, 2])
    
    def get_constraints(self):
        return []

class Rastrigin(AbstractBenchmark):
    '''The Rastrigin function has several local minima. 
    It is highly multimodal, but locations of the minima are regularly distributed.
    '''

    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)

    def get_name(self):
        return "Rastrigin function"

    def evaluate_one_hf(self, x):
        return 10 * self._dim + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    def evaluate_one_lf(self, x):
        return 10 * self._dim + 1.1*np.sum(x**2 - 10*np.cos(2*np.pi*x))

    def get_global_optimum(self):
        return np.array([0]*self._dim)

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])
    
    def get_constraints(self):
        return []


class Bohachevsky1(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Bohachevsky function 1"

    def evaluate_one_hf(self, x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7

    def evaluate_one_lf(self, x):
        return x[0]**2 + 2.1*x[1]**2 - 0.32*np.cos(3*np.pi*x[0]) - 0.41*np.cos(4*np.pi*x[1]) + 0.7

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])
    
    def get_constraints(self):
        return []

class Bohachevsky2(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Bohachevsky function 2"

    def evaluate_one_hf(self, x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0])*np.cos(4*np.pi*x[1]) + 0.3

    def evaluate_one_lf(self, x):
        return x[0]**2 + 2.1*x[1]**2 - 0.31*np.cos(3*np.pi*x[0])*np.cos(4*np.pi*x[1]) + 0.3

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])
    
    def get_constraints(self):
        return []


class Bohachevsky3(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Bohachevsky function 3"

    def evaluate_one_hf(self, x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0] + 4*np.pi*x[1]) + 0.3

    def evaluate_one_lf(self, x):
        return x[0]**2 + 2.1*x[1]**2 - 0.31*np.cos(3*np.pi*x[0] + 4*np.pi*x[1]) + 0.3

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])
    
    def get_constraints(self):
        return []


class Zakharov(AbstractBenchmark):
    '''Zakharov function is plate shaped function
    It has only one optimum at (0,0, ..d times)
    '''

    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)

    def get_name(self):
        return "Zakharov function"

    def evaluate_one_hf(self, x):
        temp = np.sum([0.5*(i+1)*x[i] for i in range(self._dim)])
        return np.sum(x**2) + temp**2 + temp**4 
    
    def evaluate_one_lf(self, x):
        temp = np.sum([0.5*(i+1)*x[i] for i in range(self._dim)])
        return np.sum(x**2) + 1.1*temp**2 + temp**4

    def get_global_optimum(self):
        return np.array([0]*self._dim)

    def get_bounds(self):
        return np.array([-3]*self._dim), np.array([3]*self._dim)

    def get_constraints(self):
        return []

class SixHumpCamel(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Six Hump Camel function"

    def evaluate_one_hf(self, x):
        return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 - 4)*x[1]**2

    def evaluate_one_lf(self, x):
        return (4 - 2.2*x[0]**2 + x[0]**4/3.2)*x[0]**2 + 1.1*x[0]*x[1] + (4.1*x[1]**2 - 4)*x[1]**2

    def get_global_optimum(self):
        return np.array([0.0898, -0.7126])#, np.array([-0.0898, 0.7126])   # This is the second optimum which is equally strong

    def get_bounds(self):
        return np.array([-1, -1]), np.array([1, 1])
    
    def get_constraints(self):
        return []
    
class ThreeHumpCamel(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Three Hump Camel function"

    def evaluate_one_hf(self, x):
        return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2

    def evaluate_one_lf(self, x):
        return 2.1*x[0]**2 - 1.06*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2

    def get_global_optimum(self):
        return np.array([0., 0.])

    def get_bounds(self):
        return np.array([-1, -1]), np.array([1, 1])
    
    def get_constraints(self):
        return []

class Rosenbrock(AbstractBenchmark):
    '''Rosenbrock function is Valley or banana shaped function. 
    '''

    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)

    def get_name(self):
        return "Rosenbrock function"

    def evaluate_one_hf(self, x):
        return np.sum([100*(x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2 for i in range(1, self._dim)])

    def evaluate_one_lf(self, x):
        return 0.2 + 1.01*np.sum([100*(x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2 for i in range(1, self._dim)])

    def get_global_optimum(self):
        return np.ones(self._dim)

    def get_bounds(self):
        return np.array([-4]*self._dim), np.array([4]*self._dim)
    
    def get_constraints(self):
        return []
    

class Beale(AbstractBenchmark):
    ''' The Beale function is multimodal, with sharp peaks at the corners of the input domain. 
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Beale function"

    def evaluate_one_hf(self, x):
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

    def evaluate_one_lf(self, x):
        return (1.54 - x[0] + x[0]*x[1])**2 + (2.29 - x[0] + x[0]*x[1]**2)**2 + (2.675 - x[0] + x[0]*x[1]**3)**2

    def get_global_optimum(self):
        return np.array([3., 0.5])

    def get_bounds(self):
        return np.array([-4, -4]), np.array([4, 4])
    
    def get_constraints(self):
        return []


class Easom(AbstractBenchmark):
    '''Easom function has several local minimum.
    The global minimum has a small area in relative search space
    '''

    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def get_name(self):
        return "Easom function"

    def evaluate_one_hf(self, x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 -(x[1] - np.pi)**2)
    
    def evaluate_one_lf(self, x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 -(x[1] - np.pi)**2)*1.1 + 0.2

    def get_global_optimum(self):
        return np.array([np.pi, np.pi])

    def get_bounds(self):
        return np.array([0, 0]), np.array([6, 6])
    
    def get_constraints(self):
        return []

class Hartmann3d(AbstractBenchmark):
    '''The Hartmann 3-D function has three local minima.'''

    def __init__(self, add_noise=False):
        super().__init__(3, add_noise)
        self.alpha = np.array([1, 1.2, 3, 3.2])
        self.A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        self.P = 10**-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
    
    def get_name(self):
        return "Hartmann 3-D function"
    
    def evaluate_one_hf(self, x):
        return -sum(self.alpha * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1)))
    
    def evaluate_one_lf(self, x):
        return -sum(1.1* self.alpha * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1))) + 0.1
    
    def get_global_optimum(self):
        return np.array([0.114614, 0.555649, 0.852547])
    
    def get_bounds(self):
        return np.array([0]*3), np.array([1]*3)
    
    def get_constraints(self):
        return []
    
class Hartmann4d(AbstractBenchmark):
    '''The Hartmann 4-D function has four local minima.'''

    def __init__(self, add_noise=False):
        super().__init__(4, add_noise)
        self.alpha = np.array([1, 1.2, 3, 3.2])
        self.A = np.array([[10, 3, 17, 3.5], [0.05, 10, 17, 0.1], [3, 3.5, 1.7, 10], [17, 8, 0.05, 10]])
        self.P = 10**-4 * np.array([[1312, 1696, 5569, 124], [2329, 4135, 8307, 3736], [2348, 1451, 3522, 2883], [4047, 8828, 8732, 5743]])
    
    def get_name(self):
        return "Hartmann 4-D function"
    
    def evaluate_one_hf(self, x):
        return -sum(self.alpha * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1)))
    
    def evaluate_one_lf(self, x):
        return -sum(1.1* self.alpha * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1))) + 0.1
    
    def get_global_optimum(self):
        return np.array([0.1873, 0.1906, 0.5566, 0.2647])
    
    def get_bounds(self):
        return np.array([0]*4), np.array([1]*4)
    
    def get_constraints(self):
        return []


class Hartmann6d(AbstractBenchmark):
    '''The Hartmann 6-D function has six local minima.'''

    def __init__(self, add_noise=False):
        super().__init__(6, add_noise)
        self.alpha = np.array([1, 1.2, 3, 3.2])
        self.A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
        self.P = 10**-4 * np.array([[1312, 1696, 5569, 124, 8283, 5887], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
    
    def get_name(self):
        return "Hartmann 6-D function"
    
    def evaluate_one_hf(self, x):
        return -sum(self.alpha * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1)))
    
    def evaluate_one_lf(self, x):
        return -sum(1.05* self.alpha * np.exp(-np.sum(self.A * (x - self.P)**2, axis=1))) + 0.1
    
    def get_global_optimum(self):
        return np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    
    def get_bounds(self):
        return np.array([0]*6), np.array([1]*6)
    
    def get_constraints(self):
        return []

def get_benchmark(benchmark_name:str, dim:int, add_noise:bool=True):
    if benchmark_name == 'Bohachevsky1':
        return Bohachevsky1(add_noise=add_noise)
    elif benchmark_name == 'Bohachevsky2':
        return Bohachevsky2(add_noise=add_noise)
    elif benchmark_name == 'SixHumpCamel':
        return SixHumpCamel(add_noise=add_noise)
    elif benchmark_name == 'ThreeHumpCamel':
        return ThreeHumpCamel(add_noise=add_noise)
    elif benchmark_name == 'Beale':
        return Beale(add_noise=add_noise)
    elif benchmark_name == 'Hartmann3d':
        return Hartmann3d(add_noise=add_noise)
    elif benchmark_name == 'Hartmann4d':
        return Hartmann4d(add_noise=add_noise)
    elif benchmark_name == 'SphereFunction':
        return SphereFunction(dim, add_noise=add_noise)
    elif benchmark_name == 'ConstreinedSphereFunction':
        return ConstraintSphereFunction(dim, add_noise=add_noise)  
    elif benchmark_name == 'Ackley':
        return AckleyFunction(dim, add_noise=add_noise)
    elif benchmark_name == 'Rosenbrock':
        return Rosenbrock(dim, add_noise=add_noise) 
    elif benchmark_name == 'Zakharov':
        return Zakharov(dim, add_noise=add_noise)
    else:
        raise ValueError(f'Unknown benchmark {benchmark_name}')

def get_list_objective_functions():
    dim_list = [2, 4, 8, 16, 32]
    # dim_list = [2,4]
    dim_ridge = [2, 4, 8, 16]
    obj_list = []
    add_noise = True
    obj_list.append(Bohachevsky1(add_noise=add_noise))
    obj_list.append(Bohachevsky2(add_noise=add_noise))
    obj_list.append(SixHumpCamel(add_noise=add_noise))
    obj_list.append(ThreeHumpCamel(add_noise=add_noise))
    obj_list.append(Beale(add_noise=add_noise))
    obj_list.append(Hartmann3d(add_noise=add_noise))
    obj_list.append(Hartmann4d(add_noise=add_noise))
    # obj_list.append(Hartmann6d(add_noise=add_noise))
    for dim in dim_list:
        obj_list.append(SphereFunction(dim, add_noise=add_noise))
        obj_list.append(ConstraintSphereFunction(dim, add_noise=add_noise))
        obj_list.append(AckleyFunction(dim, add_noise=add_noise))
    for dim in dim_ridge:
        obj_list.append(Rosenbrock(dim, add_noise=add_noise))
        obj_list.append(Zakharov(dim, add_noise=add_noise))
    return obj_list

if __name__ == '__main__':
    # benchmark = AckleyFunction(2, a=20, b=0.2, c=2*np.pi)
    # benchmark = BukinFunction_N6()
    # benchmark = DropWave()
    # benchmark = Rastrigin(2)
    # benchmark = Bohachevsky1()
    # benchmark = Bohachevsky2()
    benchmark = Zakharov(32)
    # benchmark = SixHumpCamel()
    # benchmark = Easom()
    # benchmark.plot(category='lf')
    print(benchmark(np.ones(32)))

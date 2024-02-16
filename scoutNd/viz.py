import numpy as np
import os
import torch
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
from itertools import product

#from scoutNd.stochastic_optimizer import Stochastic_Optimizer
#from scoutNd.objective_function import Baseline1
# add bm, amsmath and all that to matplotlib


rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#params= {'text.latex.preamble' : r'\usepackage{amsmath,bm, amsfonts}'}
#plt.rcParams.update(params)
rc('text.latex', preamble=r'\usepackage{amsmath,bm,amsfonts}')
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20

rc('font', size=MEDIUM_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure titles

torch.set_default_dtype(torch.float64)
# fix the seed
seed = 666
np.random.seed(seed)
torch.random.manual_seed(seed)
import time
datetime = time.strftime("%Y%m%d-%H%M%S")

class variable_evolution:
    def __init__(self, L_x:np.ndarray, f_x:np.ndarray, mu:np.ndarray, beta:np.ndarray, path:str,save_name :str, C_x =None, lambdas =None, **kwargs):
        """"""
        self.L_x = L_x # Augmented objective function
        self.f_x = f_x # Objective function
        self.C_x = C_x # Constraints
        self.mu = mu # mean of the design variables
        self.beta = beta # variance of the design variables
        self.lambdas = lambdas # penalty term multipliers
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path # path to save the plots
        self.save_name = save_name # name of the file to save the plots
    
    def aug_objective(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.L_x, label='Aug. Objective Function')
        # TODO : plot moving average

        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'$\mathbb{E}[\mathcal{L}(\mathbf{\cdot})]$')
        ax.legend()
        ax.grid()
        plt.tight_layout() 
        plt.savefig(f'{self.path}/{self.save_name}_aug_objective_evolution_{datetime}.pdf')

    def obective(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.f_x, label='Objective Function')
        # TODO : plot moving average

        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'$\mathbb{E}[f(\mathbf{\cdot})]$')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(f'{self.path}/{self.save_name}_objective_evolution_{datetime}.pdf')

    def constraints(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # iterate over number of columns
        for i in range(self.C_x.shape[1]):
            #ax.plot(self.C_x[:,i], label=r'$\mathcal{C}_{i+1}$')
            ax.plot(self.C_x[:,i], label=rf'$\mathcal{{C}}_{{{i+1}}}$')
        ax.axhline(y=0, color='red', linestyle='--')
        # TODO : plot moving average

        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'$\mathbb{E}[\mathcal{C}(\mathbf{\cdot})]$')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(f'{self.path}/{self.save_name}_constraints_evolution_{datetime}.pdf')

    def mean(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.mu)
        #ax.set_title(r'$\mu$ evolution')
        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'$\mathbf{\mu}$')
        #ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(f'{self.path}/{self.save_name}_mean_evolution_{datetime}.pdf')
    
    def variance(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        var = np.exp(self.beta)
        ax.semilogy(var)
        #ax.set_title(r'$\beta$ evolution')
        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'$\sigma^2$')
        #ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(f'{self.path}/{self.save_name}_variance_evolution_{datetime}.pdf')

    def plot_lambdas(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogy(self.lambdas)
        #ax.set_title(r'$\lambda$ evolution')
        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'$\lambda$')

        ax.grid()
        plt.tight_layout()
        plt.savefig(f'{self.path}/{self.save_name}_lambda_evolution_{datetime}.pdf')

    # TODO: add Lambda evolution plot

    # a function to plot all the evolution
    def plot_all(self):
        self.aug_objective()
        self.obective()
        #self.constraints()
        self.mean()
        self.variance()
        #self.plot_lambdas()

        if self.C_x is not None:
            self.constraints()
            self.plot_lambdas()


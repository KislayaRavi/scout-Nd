
from benchmarks import AckleyFunction
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
from itertools import product

from stochastic_optimizer import Stochastic_Optimizer
from objective_function import Baseline1
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
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

# function = AckleyFunction(dim=2)
# print(function.get_name())
# function.evaluate_hf(np.array([0,0]))
# fig = function.plot()

dim = 2
def function(x:np.array):
    fn = AckleyFunction(dim=dim)
    val1 = fn.evaluate_hf(x)
    noise = False
    if noise:
        val2 = np.random.normal(0, 0.0001, val1.shape)
        return val1+ val2
    else:
        return val1

optimize= False
gaussian_conv = False
optimize_compare = False
check_fim = True

if optimize == True:
    constraints = None
    # constraints = None
    obj = Baseline1(dim, function, constraints, num_samples=128, qmc=True, correct_constraint_derivative=True)
    optimizer = Stochastic_Optimizer(obj)
    x0 = np.ones(2*dim)
    x0[dim:] = -10
    #x0[dim:] = 0
    #x0 = np.zeros(2*dim)
    optimizer.set_initial_val(x0)
    optimizer.create_optimizer('Adam', lr=1e-2)
    optimizer.optimize(num_steps=300)
    print(optimizer.get_final_state())

if gaussian_conv:
    sigma_set = [0,-10]
    obj = Baseline1(dim, function, None, num_samples=64, qmc=True, correct_constraint_derivative=True)

    # evaluate function in x in [-30,30] for 2 d
    mesh_size = 100
    x = np.linspace(-30, 30, mesh_size)
    y = np.linspace(-30, 30, mesh_size)
    X, Y = np.meshgrid(x, y)
    temp = np.array(list(product(x,y)))

    # plot without smootheing
    fn = AckleyFunction(dim=dim)
    zs = fn.evaluate_hf(temp)
    Z = zs.reshape(mesh_size, mesh_size)
    
    # do a 3d contour plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.plot_wireframe(X, Y, Z, color='black', alpha=0.3, linewidth=0.3, rcount=mesh_size/5, ccount=mesh_size/5)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$f(x_1, x_2)$')
    plt.tight_layout()
    plt.savefig('ackley_no_conv.pdf')

    #plt.show()


    # x = np.zeros(2*dim)
    # x[dim:] = sigma*np.ones(dim)
    for sigma in sigma_set:
        x = np.zeros(2*dim)
        z_conv = np.zeros(mesh_size*mesh_size)
        for i in range(mesh_size*mesh_size):
            x[:dim] = temp[i]
            x[dim:] = sigma*np.ones(dim)
            fval, _ = obj.function_wrapper(torch.tensor(x, dtype=torch.float64))
            z_conv[i] = fval
        Z_s = z_conv.reshape(mesh_size, mesh_size)

        # do a 3d contour plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_s, cmap='viridis')
        ax.plot_wireframe(X, Y, Z_s, color='black', alpha=0.3, linewidth=0.3,rcount=mesh_size/5, ccount=mesh_size/5)
        ax.set_xlabel(r'$\mu_1$')
        ax.set_ylabel(r'$\mu_2$')
        ax.set_zlabel(r'$U(\mu,\sigma)$')
        plt.tight_layout()
        plt.savefig(f'ackley_conv_' +f'sigma_{sigma}' +'.pdf')
        #plt.show()
    
    
#if optimize_compare:
def compare_optimize(sigma,mean_init :int,plot_name:str,**kwargs):
    constraints = None
    # constraints = None
    obj = Baseline1(dim, function, constraints, num_samples=64, qmc=True, correct_constraint_derivative=True)
    
    #x0 = 3*np.ones(2*dim)
    x0 = mean_init*np.ones(2*dim)
    #sigma = -10
    x0[dim:] = sigma
    #x0[dim:] = sigma
    #x0 = np.zeros(2*dim)
    optimizer = Stochastic_Optimizer(obj, initial_val=x0, **kwargs)
    #optimizer.set_initial_val(x0)
    optimizer.create_optimizer('Adam', lr=1e-1)
    num_steps = 500
    optimizer.optimize(num_steps=num_steps)
    mean_evo = np.zeros((num_steps, dim))
    var_evo = np.zeros((num_steps, dim))
    # convert list to array
    for i in range(num_steps):
        mean_evo[i,:] = optimizer.stored_results[i].detach().numpy()[:dim]
        var_evo[i,:] = np.exp(optimizer.stored_results[i].detach().numpy()[dim:])

    # do a 2d contour plot for the Ackley fn
    
    # evaluate function in x in [-30,30] for 2 d
    mesh_size = 100
    x = np.linspace(-10, 10, mesh_size)
    y = np.linspace(-10, 10, mesh_size)
    X, Y = np.meshgrid(x, y)
    temp = np.array(list(product(x,y)))

    # plot without smootheing
    fn = AckleyFunction(dim=dim)
    zs = fn.evaluate_hf(temp)
    Z = zs.reshape(mesh_size, mesh_size)

    # 2d surfac plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.contourf(X, Y, Z, cmap='viridis', levels=20)
    # plot the evolution of the design variables
    ax.plot(mean_evo[:,0], mean_evo[:,1], 'x', markersize=6, color='black')
    # last marker red
    ax.plot(mean_evo[-1,0], mean_evo[-1,1], 'x', markersize=8, color='red')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    
    # add colorbar
    cbar = plt.colorbar(h, ax=ax)
    plt.tight_layout()
    plt.savefig(f'figures/ackley_optimization_mean_evolution_contour_{plot_name}' + f'sigma_{sigma}'+'.pdf')
    plt.show()

    # plot mean evolution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    iterations = np.linspace(0, num_steps, num_steps)
    ax.plot(mean_evo[1:,0],'r', label=r'$\mu_1$')
    ax.plot(mean_evo[1:,1],'b', label=r'$\mu_2$')
    ax.set_xlabel(r'$iterations$')
    ax.set_ylabel(r'$\mu$')
    ax.legend()
    ax.grid()
    plt.tight_layout()  
    plt.savefig(f'figures/ackley_optimization_mean_evolution_{plot_name}' + f'sigma_{sigma}'+'.pdf')
    plt.show()

    # plot var evolution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    iterations = np.linspace(0, num_steps, num_steps)
    ax.semilogy(var_evo[1:,0],'r', label=r'$\sigma_1^2$')
    ax.semilogy(var_evo[1:,1],'b', label=r'$\sigma_2^2$')
    ax.set_xlabel(r'$iterations$')
    ax.set_ylabel(r'$\sigma^2$')
    ax.legend()
    ax.grid()
    plt.tight_layout()  
    plt.savefig(f'figures/ackley_optimization_var_evolution_{plot_name}' + f'sigma_{sigma}'+'.pdf')
    plt.show()

if check_fim:
    compare_optimize(sigma=0,mean_init=0.1,plot_name = 'fim_true_start_0dot1_',natural_gradients=True)
    #compare_optimize(sigma=0,mean_init=0.1,plot_name = 'fim_false_start_0dot1_',natural_gradients=False)



    



        #print(optimizer.get_final_state())















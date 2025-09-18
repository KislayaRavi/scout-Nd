
from scoutNd.benchmarks import AckleyFunction
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
from itertools import product

from scoutNd.stochastic_optimizer import Stochastic_Optimizer
from scoutNd.objective_function import Baseline1
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
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
# double precision for both numpy and tensor
torch.set_default_dtype(torch.float64)
# fix the seed
seed = 666
np.random.seed(seed)
torch.random.manual_seed(seed)


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
compare_fim_fx = False

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
def compare_optimize(sigma,mean_init :int,plot_name:str,no_sample=64,**kwargs):
    constraints = None
    # constraints = None
    obj = Baseline1(dim, function, constraints, num_samples=no_sample, qmc=True, correct_constraint_derivative=True)
    
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
    num_actual_steps = len(optimizer.stored_results)
    mean_evo = np.zeros((num_actual_steps, dim))
    var_evo = np.zeros((num_actual_steps, dim))
    # convert list to array
    for i in range(num_actual_steps):
        mean_evo[i,:] = optimizer.stored_results[i].detach().numpy()[:dim]
        var_evo[i,:] = np.exp(optimizer.stored_results[i].detach().numpy()[dim:])
    
    # plotting no of fn evalutions vs the f(x)
    val = np.array(optimizer.stored_f_x)
    # convert the list to an array
    # plot var evolution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    no_calls = np.linspace(0, no_sample*(num_actual_steps-1), num_actual_steps-1)
    ax.semilogy(no_calls,val, label=r'$\sigma_1^2$')
    #ax.semilogy(var_evo[1:,1],'b', label=r'$\sigma_2^2$')
    ax.set_xlabel(r'$iterations$')
    ax.set_ylabel(r'$\sigma^2$')
    ax.legend()
    ax.grid()
    plt.tight_layout()  
    # plt.savefig(f'figures/ackley_optimization_var_evolution_{plot_name}' + f'sigma_{sigma}'+'.pdf')
    plt.show()



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
    # plt.savefig(f'figures/ackley_optimization_mean_evolution_contour_{plot_name}' + f'sigma_{sigma}'+'.pdf')
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
    # plt.savefig(f'figures/ackley_optimization_mean_evolution_{plot_name}' + f'sigma_{sigma}'+'.pdf')
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
    # plt.savefig(f'figures/ackley_optimization_var_evolution_{plot_name}' + f'sigma_{sigma}'+'.pdf')
    plt.show()

def compare_natural_fn_vals(sigma,mean_init :int,plot_name:str,no_sample=64,plotting=True,**kwargs):
    constraints = None
    # constraints = None
    obj = Baseline1(dim, function, constraints, num_samples=no_sample, qmc=True, correct_constraint_derivative=True)
    
    #x0 = 3*np.ones(2*dim)
    x0 = mean_init*np.ones(2*dim)
    #sigma = -10
    x0[dim:] = sigma
    #x0[dim:] = sigma
    #x0 = np.zeros(2*dim)
    optimizer = Stochastic_Optimizer(obj, initial_val=x0, natural_gradients=True, **kwargs)
    #optimizer.set_initial_val(x0)
    lr = 0.1
    optimizer.create_optimizer('Adam', lr=lr)
    #num_steps = 300
    num_steps = 1000
    optimizer.optimize(num_steps=num_steps)
    
    
    val_nat_grad = np.array(optimizer.stored_f_x)

    # natural grad flase
    optimizer_ = Stochastic_Optimizer(obj, initial_val=x0, natural_gradients=False,**kwargs)
    #optimizer.set_initial_val(x0)
    optimizer_.create_optimizer('Adam', lr=lr)
    
    optimizer_.optimize(num_steps=num_steps)

    val_no_nat_grad = np.array(optimizer_.stored_f_x)

    # plotting no of fn evalutions vs the f(x)
    if plotting:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        no_calls = np.linspace(0, no_sample*num_steps, num_steps)
        #ax.semilogy(no_calls,val_no_nat_grad, label=r'Scout')
        #ax.semilogy(no_calls,val_nat_grad, label=r'Scount NG')
        ax.plot(no_calls,val_no_nat_grad, label=r'Scout')
        ax.plot(no_calls,val_nat_grad, label=r'Scount NG')
        # horizontal line with grey dotted at y =0 
        ax.axhline(y=0, color='grey', linestyle='--', label=r'True value')
        #ax.semilogy(var_evo[1:,1],'b', label=r'$\sigma_2^2$')
        ax.set_xlabel('Number of function calls', fontsize=15)
        ax.set_ylabel(r'$f(x)$')
        ax.legend()
        ax.grid()
        plt.tight_layout()  
        # plt.savefig(f'figures/ackley_optimization_f_x_lr_{lr}_{plot_name}' + f'sigma_{sigma}'+'.pdf')
        plt.show()
    return val_nat_grad, val_no_nat_grad

if check_fim:
    compare_optimize(sigma=0,mean_init=0.1,plot_name = 'fim_true_start_0dot1_',natural_gradients=True)
    #compare_optimize(sigma=0,mean_init=0.1,plot_name = 'fim_false_start_0dot1_',natural_gradients=False)

if compare_fim_fx:
    #compare_natural_fn_vals(sigma=0,mean_init=5,no_sample=64,plot_name = 'fim_comparision')
    no_samples = [2,4,8,16,32,64]
    #no_samples = [2,2]
    no_ng =[]
    ng = []
    for i in no_samples:
        val_nat_grad, val_no_nat_grad = compare_natural_fn_vals(sigma=0,mean_init=5,no_sample=i,plot_name = 'fim_comparision', plotting=False)
        no_ng.append(val_no_nat_grad)
        ng.append(val_nat_grad)
    no_ng = np.vstack(no_ng)
    ng = np.vstack(ng)
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    no_step = 1000
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    for i in range(len(no_samples)):
        # linspace for no of space
        iterations = np.linspace(0, no_step, no_step)
        ax.plot(iterations,no_ng[i,:], ':',color=CB_color_cycle[i], label=f'no NG $(S = {no_samples[i]})$')
        ax.plot(iterations,ng[i,:], '-',color=CB_color_cycle[i], label=f'NG $(S = {no_samples[i]})$')
    # horizontal line with grey dotted at y =0
    ax.axhline(y=0, color='grey', linestyle='--', label=r'True value')
    ax.set_xlabel('Number of steps', fontsize=15)
    ax.set_ylabel(r'$f(x)$')
    # plot the legend outside the plot on the right side, the main plot should be square
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    # th plot x and y should be equal

    #ax.legend(bbox_to_anchor=(-0.1, 1.05), loc='lower left', borderaxespad=0., ncols=5)
    #ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.savefig(f'figures/ackley_optimization_f_x_lr_fim_comparision_different_samples'+'.pdf')
    plt.show()
    

    


# 1. what dampoening factor to choose.
# 2. very large terms in the diagonal.
# 3. what should be the value of the coeff. 
# 4. turn off the natural gradeints for starting 40% of the iterations? 
# - use sigma stopping criterion. no need to push for cond number bettwe
# - exp decay for nat grad, or delayed start (based on moving avergae of the fn lets say)
# - Multi level MC, check the var reduction for it. gil thesis eq 73-76
        #print(optimizer.get_final_state())















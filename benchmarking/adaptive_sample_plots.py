"""Adaptive vs fixed sample size study.

Runs single-fidelity and multi-fidelity SCOUT-Nd optimisers across a sweep of sample sizes,
saves the convergence trajectories under ``./sample_size_study/``, and produces the
``ackley_adaptive_*.pdf`` figures comparing computational cost vs objective value.
"""

import numpy as np
import os
from collections import OrderedDict
import pickle
from common_data import *
from scoutNd.stochastic_optimizer import Stochastic_Optimizer
from scoutNd.objective_function import Baseline1
from scoutNd.multifidelity_objective import MultifidelityObjective
from benchmarks import *
from matplotlib import pyplot as plt
from matplotlib import rc
# import seaborn as sns
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams["mathtext.fontset"] = "cm"
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 20


rc('font', size=BIGGER_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

def run_one_sf_optimizer(benchmark_name:str, dim:int, adaptive_sample_size:bool, 
                         add_noise:bool=True, num_steps:int=500,
                         lr:float=1e-2, num_samples:int=None,
                         base_folder='./sample_size_study/'):
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    obj_fn = get_benchmark(benchmark_name, dim, add_noise=add_noise)
    constraints = obj_fn.get_constraints()
    if num_samples is None:
        num_samples = get_num_samples(dim)
    # if len(constraints) == 0:
    #     constraints = None
    obj = Baseline1(dim, obj_fn, constraints, num_samples=num_samples, 
                    qmc=False, correct_constraint_derivative=True, 
                    adaptive_sample_size=adaptive_sample_size)
    optimizer = Stochastic_Optimizer(obj, natural_gradients=False)
    optimizer.create_optimizer('Adam', lr=lr)
    if len(constraints) > 0:
        optimizer.optimize(num_steps_per_lambda=num_steps)
    else:
        optimizer.optimize(num_steps=num_steps)
    parameter_evolution = np.array([param.detach().numpy() for param in optimizer.stored_results])[1:, :]
    non_noisy_obj_fn = get_benchmark(benchmark_name, dim, add_noise=False)
    # num_func_evolution = np.cumsum(obj.sample_size_list)
    store_array = np.zeros((len(obj.sample_size_list), 2+2*dim))
    store_array[:, 0] = obj.sample_size_list
    store_array[:, 1] = non_noisy_obj_fn(parameter_evolution[:, :dim]).ravel()
    store_array[:, 2:] = parameter_evolution
    file_name = benchmark_name + '_' +str(dim)
    if adaptive_sample_size:
        file_name += '_' + 'adapt'
    else:
        file_name += '_' + str(num_samples)
    np.savetxt(base_folder+file_name, store_array)

def run_one_mf_optimizer(benchmark_name:str, dim:int, adaptive_sample_size:bool, 
                         add_noise:bool=True, num_steps:int=500,
                         lr:float=1e-2, cost_list:int=None,
                         base_folder='./sample_size_study/', log10_eps_kl=-2):
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    obj_fn = get_benchmark(benchmark_name, dim, add_noise=add_noise)
    if cost_list is None or len(cost_list) != 2:
        raise ValueError('Cost list must be of length 2')
    num_samples_lf = get_num_samples(dim)
    num_samples_hf = int(num_samples_lf/cost_list[1])
    constraints = obj_fn.get_constraints()
    obj = MultifidelityObjective(dim, [obj_fn.evaluate_lf, obj_fn.evaluate_hf], 
                                 constraints, qmc=False, adaptive_sample_size=adaptive_sample_size,                   
                                 cost_list=cost_list)
    obj.set_num_samples([num_samples_lf, num_samples_hf])
    optimizer = Stochastic_Optimizer(obj, natural_gradients=False)
    optimizer.create_optimizer('Adam', lr=lr)
    if len(constraints) > 0:
        optimizer.optimize(num_steps_per_lambda=num_steps, eps_kl=10**log10_eps_kl)
    else:
        optimizer.optimize(num_steps=num_steps, eps_kl=10**log10_eps_kl)
    parameter_evolution = np.array([param.detach().numpy() for param in optimizer.stored_results])[1:, :]
    non_noisy_obj_fn = get_benchmark(benchmark_name, dim, add_noise=False)
    store_array = np.zeros((len(obj.sample_size_list), 4+2*dim))
    sample_size_array = np.array(obj.sample_size_list)
    store_array[:, :2] = np.array(obj.sample_size_list)
    norm_cost = np.array(obj.cost_list)/obj.cost_list[-1]
    cost_array = np.cumsum((sample_size_array[:, 0]*norm_cost[0]) + (sample_size_array[:, 1]*norm_cost[1]))
    store_array[:, 2] = cost_array
    store_array[:, 3] = non_noisy_obj_fn(parameter_evolution[:, :dim]).ravel()
    store_array[:, 4:] = parameter_evolution
    file_name = benchmark_name + '_' +str(dim)
    if adaptive_sample_size:
        file_name += '_' + 'mf_adapt' + str(log10_eps_kl)
    else:
        file_name += '_' + 'mf_fixed'
    np.savetxt(base_folder+file_name, store_array)

def plot_sf_num_eval(ax, benchmark_name:str, dim:int, adaptive_sample_size:bool, 
            num_samples:int=None, base_folder='./sample_size_study/', user_label:str=None):
    if num_samples is None:
        num_samples = get_num_samples(dim)
    file_name = benchmark_name + '_' +str(dim)
    if adaptive_sample_size:
        file_name += '_' + 'adapt'
        label = 'Adaptive sample size'+ r'($\varepsilon_{KL}=10^{-2}$)'
    else:
        file_name += '_' + str(num_samples)
        label = '$S=$' + str(num_samples)
    if user_label is not None:
        label = user_label
    store_array = np.loadtxt(base_folder+file_name)
    computation_cost = np.cumsum(store_array[:, 0])
    line, = ax.plot(computation_cost, store_array[:, 1], label=label)
    return line, label 

def plot_mf_num_eval(ax, benchmark_name:str, dim:int, 
                     adaptive_sample_size:bool, log10_eps_kl:float=-2, 
                     base_folder='./sample_size_study/', user_label:str=None):
    file_name = benchmark_name + '_' +str(dim)
    if adaptive_sample_size:
        file_name += '_' + 'mf_adapt'
        label = 'MF adaptive sample size'+ r'($\varepsilon_{KL}=10^{' + str(log10_eps_kl) + r'}$)'
        file_name += str(log10_eps_kl)
    else:
        file_name += '_' + 'mf_fixed'
        s1, s2 = get_num_samples(dim), int(get_num_samples(dim)/4)
        label = 'MF fixed sample size' + '($S_1=' + str(s1) + ', S_2=' + str(s2) + '$)'
    store_array = np.loadtxt(base_folder+file_name)
    computation_cost = store_array[:, 2]
    if user_label is not None:
        label = user_label
    # print(store_array[:10, 3])
    line, = ax.plot(computation_cost, store_array[:, 3], label=label)
    return line, label 

def plot_mf_num_iterations(ax, benchmark_name:str, dim:int, adaptive_sample_size:bool, 
                           log10_eps_kl:float=-2, base_folder='./sample_size_study/', user_label:str=None):
    file_name = benchmark_name + '_' +str(dim)
    if adaptive_sample_size:
        file_name += '_' + 'mf_adapt'
        label = 'MF adaptive sample size' + '($\varepsilon_{KL}=10^{' + str(log10_eps_kl) + '}$)'
        file_name += str(log10_eps_kl)
    else:
        file_name += '_' + 'mf_fixed'
        s1, s2 = get_num_samples(dim), int(get_num_samples(dim)/4)
        label = 'MF fixed sample size' + '($S_1=' + str(s1) + ', S_2=' + str(s2) + '$)'
    try:
        store_array = np.loadtxt(base_folder+file_name)
    except:
        print('File not found')
        return
    if user_label is not None:
        label = user_label
    num_iterations = np.linspace(0, len(store_array[:,1]), len(store_array[:,1]))
    line,  = ax.plot(num_iterations, store_array[:, 3], label=label)
    plt.xlim([0, 1000])
    return line, label 

def plot_sf_num_iterations(ax, benchmark_name:str, dim:int, adaptive_sample_size:bool, 
                           num_samples:int=None, base_folder='./sample_size_study/', user_label=None):
    if num_samples is None:
        num_samples = get_num_samples(dim)
    file_name = benchmark_name + '_' +str(dim)
    if adaptive_sample_size:
        file_name += '_' + 'adapt'
        label = 'Adaptive sample size'+ '($\varepsilon_{KL}=10^{-2}$)'
    else:
        file_name += '_' + str(num_samples)
        label = '$S=$' + str(num_samples)
    if user_label is not None:
        label = user_label
    store_array = np.loadtxt(base_folder+file_name)
    num_iterations = np.linspace(0, len(store_array[:,1]), len(store_array[:,1]))
    line,= ax.plot(num_iterations, store_array[:, 1], label=label)
    return line, label

def run_optimizers(benchmark_name, dim, num_steps):
    run_one_sf_optimizer(benchmark_name, dim, True, num_steps=num_steps)
    run_one_sf_optimizer(benchmark_name, dim, False, num_steps=num_steps, num_samples=2)
    run_one_sf_optimizer(benchmark_name, dim, False, num_steps=num_steps, num_samples=4)
    run_one_sf_optimizer(benchmark_name, dim, False, num_steps=num_steps, num_samples=8)
    run_one_sf_optimizer(benchmark_name, dim, False, num_steps=num_steps, num_samples=16)
    run_one_sf_optimizer(benchmark_name, dim, False, num_steps=num_steps, num_samples=32)
    run_one_sf_optimizer(benchmark_name, dim, False, num_steps=num_steps, num_samples=64)
    run_one_sf_optimizer(benchmark_name, dim, False, num_steps=num_steps, num_samples=128)
    run_one_sf_optimizer(benchmark_name, dim, False, num_steps=num_steps, num_samples=256)

def plot_num_evals(ax, benchmark_name, dim):
    line1, label1= plot_sf_num_eval(ax, benchmark_name, dim, True)
    line2, label2= plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=2)
    line3, label3=plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=4)
    line4, label4=plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=8)
    line5, label5=plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=16)
    line6, label6=plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=32)
    line7, label7=plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=64)
    line8, label8=plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=128)
    line9, label9=plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=256)
    ax.legend()
    ax.set_yscale('log')
    # ax.set_xlabel('Number of function evaluations')
    ax.set_ylabel('Objective value $f(x)$')
    # plt.xlim([0, 20500])
    # plt.ylim([0.02, 34])

def plot_num_iterations(ax, benchmark_name, dim):
    plot_sf_num_iterations(ax, benchmark_name, dim, True)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=2)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=4)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=8)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=16)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=32)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=64)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=128)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=256)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Objective value $f(x)$')
    plt.xlim([0, 1000])

def mf_plots(benchmark_name, dim):
    fig, ax = plt.figure(), plt.gca()
    fig_legend = plt.figure('legend')
    line1, label1 = plot_mf_num_eval(ax, benchmark_name, dim, True, log10_eps_kl=-2)
    line2, label2 = plot_mf_num_eval(ax, benchmark_name, dim, True, log10_eps_kl=-3)
    line3, label3 = plot_mf_num_eval(ax, benchmark_name, dim, False)
    line4, label4 = plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=256, user_label='SF fixed sample size($S= 256$)')
    ax.set_yscale('log')
    ax.set_xlim([0, 50000])
    ax.set_xlabel('Computational cost')
    ax.set_ylabel('Objective value $f(x)$')
    ax.set_ylim([0.01, 10])
    # ax.legend(bbox_to_anchor=(0, 1.02,1, 0.2), loc="lower left", ncol=1, mode='expand')
    fig_legend.legend([line1, line2, line3, line4], [label1, label2, label3, label4], loc='center', ncol=2)
    fig_legend.savefig('ackley_adaptive_mf_legend.pdf', bbox_inches='tight')
    ax.grid()
    fig.savefig('ackley_adaptive_mf_computational_cost.pdf', bbox_inches='tight')
    # plt.show()
    fig, ax = plt.figure(), plt.gca()
    plot_mf_num_iterations(ax, benchmark_name, dim, True,  log10_eps_kl=-2)
    plot_mf_num_iterations(ax, benchmark_name, dim, True,  log10_eps_kl=-3)
    plot_mf_num_iterations(ax, benchmark_name, dim, False)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=256, user_label='SF fixed sample size($S= 256$)')
    ax.set_yscale('log')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Objective value $f(x)$')
    ax.set_ylim([0.01, 10])
    ax.set_xlim([0, 1000])
    ax.grid()
    # ax.legend(bbox_to_anchor=(0, 1.02,1, 0.2), loc="lower left", ncol=1, mode='expand')
    plt.savefig('ackley_adaptive_mf_num_iterations.pdf', bbox_inches='tight')
    # plt.show()

def sf_plots(benchmark_name, dim):
    fig, ax = plt.figure(), plt.gca()
    legend_fig = plt.figure('Legend')
    line1, label1 = plot_sf_num_eval(ax, benchmark_name, dim, True)
    line2, label2 = plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=4)
    line3, label3 = plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=8)
    line4, label4 = plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=16)
    line5, label5 = plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=32)
    line6, label6 = plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=64)
    line7, label7 = plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=128)
    line8, label8 = plot_sf_num_eval(ax, benchmark_name, dim, False, num_samples=256)
    ax.set_yscale('log')
    ax.set_xlim([0, 50000])
    ax.set_xlabel('Number of function evaluations')
    ax.set_ylabel('Objective value $f(x)$')
    ax.set_ylim([0.01, 10])
    ax.grid()
    # ax.legend(bbox_to_anchor=(0, 1.02,1, 0.2), loc="lower left", ncol=2, mode='expand')
    legend_fig.legend([line1, line2, line3, line4, line5, line6, line7, line8], [label1, label2, label3, label4, label5, label6, label7, label8], loc='center', ncol=2)
    fig.savefig('ackley_adaptive_sf_computational_cost.pdf', bbox_inches='tight')
    
    legend_fig.savefig('ackley_adaptive_sf_legend.pdf', bbox_inches='tight')
    # plt.show()
    fig, ax = plt.figure(), plt.gca()
    plot_sf_num_iterations(ax, benchmark_name, dim, True)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=4)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=8)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=16)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=32)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=64)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=128)
    plot_sf_num_iterations(ax, benchmark_name, dim, False, num_samples=256)
    ax.set_yscale('log')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Objective value $f(x)$')
    ax.set_ylim([0.01, 10])
    ax.set_xlim([0, 1000])
    ax.grid()
    # ax.legend(bbox_to_anchor=(0, 1.02,1, 0.2), loc="lower left", ncol=2, mode='expand')
    plt.savefig('ackley_adaptive_sf_num_iterations.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    benchmark_name, dim, num_steps = 'Ackley', 32, 1000
    # run_one_sf_optimizer(benchmark_name, dim, False, num_steps=13*num_steps, num_samples=4)
    # run_one_sf_optimizer(benchmark_name, dim, False, num_steps=7*num_steps, num_samples=8)
    # run_one_sf_optimizer(benchmark_name, dim, False, num_steps=4*num_steps, num_samples=16)
    # run_one_mf_optimizer(benchmark_name, dim, True, log10_eps_kl=-3, 
    #                      cost_list=[1,4], num_steps=2000)
    # run_one_mf_optimizer(benchmark_name, dim, True, log10_eps_kl=-2, 
    #                      cost_list=[1,4], num_steps=2000)
    # run_one_mf_optimizer(benchmark_name, dim, False, 
    #                      cost_list=[1,4], num_steps=num_steps)
    mf_plots(benchmark_name, dim)
    sf_plots(benchmark_name, dim)
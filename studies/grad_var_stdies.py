# to all the necessary imports
import torch
import numpy as np
import matplotlib.pyplot as plt 

# import from local files
from scout_Nd.objective_function import *
# matplotlib to use latex and bm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{bm,amsmath,amsfonts}')
# set font size for journal article
plt.rcParams.update({'font.size': 16})


dim = [4,8,16,32,64,128]
def study_qmc_influence(qmc:bool):
    constraints = [line_constraint]
    obj = NoVarianceReduction(dim, sphere, constraints,qmc=qmc, num_samples=128) # Biased performs better than unbiased
    parameters = [torch.tensor(np.ones(2*dim), requires_grad=True)]
    optimiser = torch.optim.Adam(parameters, lr=1e-1)
    grad_list = []
    for i in range(50):
        val, grad = obj.function_wrapper(parameters[0])
        grad_list.append(grad)
        grad_var = np.var(np.array(grad_list), axis=0)
    print(f"The grad estimator varience with qmc as {qmc} is:", grad_var)
    return grad_var

#study_baseline = True
#if study_baseline:
def study_baseline_influence(dim:int,baseline:bool):
    """ return the grad with size 2*dim
    TODO: need to update it to return var of the grad norm, also for diff dimension at varrying no of spatial points (then take a box plot)
    """
    constraints = [linear_constraint]
    if baseline:
        obj = SFBiasedBaseline(dim, sphere, constraints,qmc =True, num_samples=128)
    else:   
        obj = NoVarianceReduction(dim, sphere, constraints,qmc =False, num_samples=128)
    parameters = [torch.tensor(1*np.ones(2*dim), requires_grad=True)]
    optimiser = torch.optim.Adam(parameters, lr=1e-1)
    grad_list = []
    for i in range(50):
        val, grad = obj.function_wrapper(parameters[0])
        # take l2 norm and divide by the size of grad vector
        grad_list.append(np.linalg.norm(grad)/dim)
    grad_var = np.var(np.array(grad_list), axis=0)
    print(f"The grad estimator varience with dim {dim} and baseline as {baseline} is:", grad_var)
    return grad_var

# run the studies
#grad_qmc = study_qmc_influence(True)
#grad_no_qmc = study_qmc_influence(False)
generate_data = False
if generate_data:
    no_exp = 10
    grad_baseline = np.ndarray(shape=(no_exp,len(dim)))
    grad_no_baseline = np.ndarray(shape=(no_exp,len(dim)))

    for i in range(no_exp):
        for j in range(len(dim)):
            grad_baseline[i,j] = study_baseline_influence(dim[j],True)
            grad_no_baseline[i,j] = study_baseline_influence(dim[j],False)

    # save results
    np.save('grad_baseline.npy', grad_baseline)
    np.save('grad_no_baseline.npy', grad_no_baseline)

do_plot = True
if do_plot:

    def box_plot_grad_comparision(grad_estimate_var_reduction:np.ndarray, grad_estimate_no_var_reduction:np.ndarray):
        assert grad_estimate_var_reduction.shape == grad_estimate_no_var_reduction.shape
        assert len(grad_estimate_var_reduction.shape) == 2
        # create a box plot
        labels = [str(dim[i]) for i in range(grad_estimate_var_reduction.shape[1])]
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))

        b_plt_1 = axes.boxplot(grad_estimate_var_reduction,labels=labels,patch_artist=True)
        b_plt_2 = axes.boxplot(grad_estimate_no_var_reduction,labels=labels,patch_artist=True)
        # TODO: add legend and more sleek box plot.

        #b_plt_1 as filled with blue and the other with green
        for patch in b_plt_1['boxes']:
            patch.set_facecolor('blue')
        for patch in b_plt_2['boxes']:
            patch.set_facecolor('green')



        
        #axes.set_yscale('log')
        axes.set_xlabel('Dimensions')
        axes.set_ylabel(r'Var$||\nabla_{\bm{\theta}}U||/d$')
        # use grid in the plot
        axes.grid(True)
        #plt.tight_layout()
        # save the plot
        fig.savefig('grad_var_box_plot.pdf')

    grad_baseline = np.load('grad_baseline.npy')
    grad_no_baseline = np.load('grad_no_baseline.npy')
    box_plot_grad_comparision(grad_baseline, grad_no_baseline)



#grad_baseline = study_baseline_influence(True)
#grad_no_baseline = study_baseline_influence(False)

# matplotlib based script to create two plots to study the influence of qmc and baseline\
# should use two different subplots.
    
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plot = False
if plot:

    # Plot the first subplot (subplot on the left)
    axes[0].plot(grad_qmc, 'b+--', label='qmc')
    axes[0].plot(grad_no_qmc, 'r+--', label='no qmc')
    axes[0].set_title('QMC vs. No QMC')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Gradient Variance')
    axes[0].legend()

    # Plot the second subplot (subplot on the right)
    axes[1].plot(grad_baseline, 'b+--', label='baseline')
    axes[1].plot(grad_no_baseline, 'r+--', label='no baseline')
    axes[1].set_title('Baseline vs. No Baseline')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Gradient Variance')
    axes[1].legend()

    # Adjust the layout to avoid overlapping titles and axis labels
    plt.tight_layout()

    # Save the figure and show the plot
    plt.savefig('qmc_and_baseline_study.png')
    plt.show()
"""Plot the design-variable / objective-function evolution from saved SCOUT-Nd runs.

Consumes pickled optimisation histories and produces ``evolution_*.pdf`` figures showing
how the iterate trajectory approaches the optimum.
"""

import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
from copy import deepcopy
# import seaborn as sns
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20


rc('font', size=BIGGER_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize


def get_x_star(eval_points, eval_vals, category):
    if category == 1:
        def constraint(x):
            return 1 - x[0] -x[1]
    else:
        def constraint(x):
            return  x[0] + x[1] - 1
    num_points = len(eval_vals)
    x_star = [eval_vals[0]]
    for i in range(1, num_points):
        if np.abs(eval_vals[i]) < np.abs(x_star[-1]) :#and constraint(eval_points[i]) < 0:
            x_star.append(deepcopy(eval_vals[i]))
        else:
            x_star.append(deepcopy(x_star[-1]))
    return np.array(x_star)


def plot_evolution(method, dim, category, label):
    with open(method+'.pickle', 'rb') as handle:
        accumulated_results = pickle.load(handle)
    keys = accumulated_results.keys()
    eval_vals = []
    eval_points = []
    if category == 1:
        fx_star = 0.5
        if method == 'CBO':
            fx_star = -0.5
    else:
        fx_star = 0.
    for key in keys:
        results = accumulated_results[key]
        if results['dim'] == dim and np.abs(results['fx_star'] - fx_star) < 0.0001:
            if np.array(results['eval_vals']).any() < 0:
                print('Something is wrong')
            eval_vals.append(results['eval_vals'])
            eval_points.append(results['eval_points'])
            break
    jumps = 1
    if method == 'CBO':
        x_star = get_x_star(eval_points[0], -1.*np.array(eval_vals[0]), category)
    else:
        x_star = get_x_star(eval_points[0], np.array(eval_vals[0]), category)
    error = np.abs(x_star - fx_star)
    print(method, len(x_star), x_star[-5:])
    plt.plot(error[::jumps], label=label)

def get_error(method, dim, category):
    with open(method+'.pickle', 'rb') as handle:
        accumulated_results = pickle.load(handle)
    keys = accumulated_results.keys()
    vals = []
    truncated_vals = []
    len_vals = []
    if category == 1:
        if method == 'CBO':
            fx_star = -0.5
        else:
            fx_star = 0.5
    else:
        fx_star = 0.
    for key in keys:
        results = accumulated_results[key]
        if results['dim'] == dim and np.abs(results['fx_star'] - fx_star) < 0.1:
            vals.append(results['eval_vals'])
            len_vals.append(len(results['eval_vals']))
    min_len = np.min(len_vals)
    for val in vals:
        truncated_vals.append(val[:min_len])
    truncated_vals = np.array(truncated_vals)
    error = np.abs(truncated_vals - fx_star)
    return np.mean(error, axis=0)

def draw_box_plots(dim, category):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    labels = ['Scout-Nd', 'MF-Scout-Nd', 'COBYLA']
    methods = ['Scout', 'MF_Scout', 'COBYLA']
    error_list = []
    for method in methods:
        error_list.append(get_error(method, dim, category))
    b_plt_1 = axes.boxplot(error_list,labels=labels,patch_artist=True)

if __name__ == '__main__':
    dim = 8
    category = 0
    # draw_box_plots(dim, 1)
    plot_evolution('Scout', dim, category, 'Scout-Nd')
    plot_evolution('MF_Scout', dim, category, 'MF-Scout-Nd')
    plot_evolution('COBYLA', dim, category, 'COBYLA')
    plot_evolution('CBO', dim, category, 'cBO')
    plt.ylabel(r'$|f(x) - f(x^*)|$')
    plt.xlabel('Number of (HF-)function evaluations')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(right=3500)
    plt.ylim(bottom=1e-4)
    plt.grid()
    plt.style.use('tableau-colorblind10')
    # plt.gca().legend(bbox_to_anchor=(0, 1.02,1, 0.2), loc="lower left", ncol=4, mode='expand')
    plt.savefig('evolution_'+str(dim)+'.pdf', bbox_inches='tight')
    plt.show()
"""Data / performance profile plots (More--Wild style) comparing optimisers across benchmarks.

Loads the pickled run results from the various ``run_*.py`` baselines, computes data
profiles at a given accuracy ``eps_f``, and produces ``data_profile_*.pdf``.
"""

import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import rc
# import seaborn as sns
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
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

def get_data_profile(values, dim, fx_star, eps_f):
    d = np.inf
    values = np.array(values).ravel()
    # fx_star = list(fx_star)
    # if len(fx_star) > 1:
    #     fx_star = fx_star[0]
    # print(values.shape, fx_star, eps_f)
    for idx, value in enumerate(values):
        if np.abs(value - fx_star) <= eps_f:
            d = idx / (dim + 1)
            break 
    return d

def plot_data_profile(method, label, eps=0.1, alpha_upper=2000):
    with open(method+'.pickle', 'rb') as handle:
        accumulated_results = pickle.load(handle)
    keys = accumulated_results.keys()
    alphas = np.linspace(0, alpha_upper, 1000)
    d_list = []
    for key in keys:
        results = accumulated_results[key]
        d = []
        temp = get_data_profile(results['eval_vals'], results['dim'], results['fx_star'], eps)
        for alpha in alphas:
            d.append(int(temp < alpha))
        d_list.append(d)
    num_benchmarks = len(list(keys))
    # print(d_list, alphas)
    data_profile = np.sum(np.array(d_list)/num_benchmarks, axis=0)
    plt.plot(alphas, data_profile, label=label)

if __name__=='__main__':
    alpha_upper, eps = 1000, 0.01
    plot_data_profile('Scout', 'Scout-Nd', eps=eps, alpha_upper=alpha_upper)
    plot_data_profile('MF_Scout', 'MF-Scout-Nd',eps=eps, alpha_upper=alpha_upper)
    plot_data_profile('COBYLA', 'COBYLA', eps=eps, alpha_upper=alpha_upper)
    plot_data_profile('CBO', 'cBO', eps=eps, alpha_upper=alpha_upper)
    plot_data_profile('SLSQP', 'SLSQP', eps=eps, alpha_upper=alpha_upper)
    plt.legend()
    plt.ylabel(r'$d_s(\alpha)$')
    plt.xlabel(r'$\alpha$')
    plt.xlim(left=0, right=alpha_upper)
    plt.ylim(bottom=0, top=1.01)
    plt.grid()
    plt.style.use('tableau-colorblind10')
    # plt.gca().legend(bbox_to_anchor=(0, 1.02,1, 0.2), loc="lower left", ncol=4, mode='expand')
    # plt.savefig('data_profile_'+str(eps)+'.pdf', bbox_inches='tight')
    plt.show()
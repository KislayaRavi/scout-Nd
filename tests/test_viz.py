import numpy as np
import torch
from scoutNd.viz import variable_evolution
import os

# Test case 1: Test the initialization of the variable_evolution class
# def test_variable_evolution_init():
#     L_x = np.array([1, 2, 3])
#     f_x = np.array([4, 5, 6])
#     C_x = np.array([[7, 8], [9, 10]])
#     mu = np.array([11, 12, 13])
#     beta = np.array([14, 15, 16])
#     # get the current path of the parent folder of the file
#     file_path = os.getcwd() +'/tests'
#     path = file_path + '/tmp'
#     save_name = 'test'
#     ve = variable_evolution(L_x, f_x, C_x, mu, beta, path, save_name)
#     assert np.array_equal(ve.L_x, L_x)
#     assert np.array_equal(ve.f_x, f_x)
#     assert np.array_equal(ve.C_x, C_x)
#     assert np.array_equal(ve.mu, mu)
#     assert np.array_equal(ve.beta, beta)
#     assert ve.path == path
#     assert ve.save_name == save_name

# # Test case 2: Test the aug_objective method
# def test_variable_evolution_aug_objective():
#     L_x = np.array([1, 2, 3])
#     f_x = np.array([4, 5, 6])
#     C_x = np.array([[7, 8], [9, 10]])
#     mu = np.array([11, 12, 13])
#     beta = np.array([14, 15, 16])
#     file_path = os.getcwd() +'/tests'
#     path = file_path + '/tmp'
#     save_name = 'test'
#     ve = variable_evolution(L_x, f_x, C_x, mu, beta, path, save_name)
#     ve.aug_objective()
#     # TODO: Add assertions to check if the plot is saved correctly

# # Test case 3: Test the obective method
# def test_variable_evolution_objective():
#     L_x = np.array([1, 2, 3])
#     f_x = np.array([4, 5, 6])
#     C_x = np.array([[7, 8], [9, 10]])
#     mu = np.array([11, 12, 13])
#     beta = np.array([14, 15, 16])
#     file_path = os.getcwd() +'/tests'
#     path = file_path + '/tmp'
#     save_name = 'test'
#     ve = variable_evolution(L_x, f_x, C_x, mu, beta, path, save_name)
#     ve.obective()
#     # TODO: Add assertions to check if the plot is saved correctly

# # Test case 4: Test the constraints method
# def test_variable_evolution_constraints():
#     L_x = np.array([1, 2, 3])
#     f_x = np.array([4, 5, 6])
#     C_x = np.array([[7, 8], [9, 10]])
#     mu = np.array([11, 12, 13])
#     beta = np.array([14, 15, 16])
#     file_path = os.getcwd() +'/tests'
#     path = file_path + '/tmp'
#     save_name = 'test'
#     ve = variable_evolution(L_x, f_x, C_x, mu, beta, path, save_name)
#     ve.constraints()
#     # TODO: Add assertions to check if the plot is saved correctly

# # Test case 5: Test the mean method
# def test_variable_evolution_mean():
#     L_x = np.array([1, 2, 3])
#     f_x = np.array([4, 5, 6])
#     C_x = np.array([[7, 8], [9, 10]])
#     mu = np.array([[11, 12, 13],[12,14,15]])
#     beta = np.array([14, 15, 16])
#     file_path = os.getcwd() +'/tests'
#     path = file_path + '/tmp'
#     save_name = 'test'
#     ve = variable_evolution(L_x, f_x, C_x, mu, beta, path, save_name)
#     ve.mean()
#     # TODO: Add assertions to check if the plot is saved correctly

# # Test case 6: Test the variance method
# def test_variable_evolution_variance():
#     L_x = np.array([1, 2, 3])
#     f_x = np.array([4, 5, 6])
#     C_x = np.array([[7, 8], [9, 10]])
#     mu = np.array([11, 12, 13])
#     beta = np.array([[14, 15, 16],[13,14,15]])
#     file_path = os.getcwd() +'/tests'
#     path = file_path + '/tmp'
#     save_name = 'test'
#     ve = variable_evolution(L_x, f_x, C_x, mu, beta, path, save_name)
#     ve.variance()
    # TODO: Add assertions to check if the plot is saved correctly

# Test case 7: Test the plot_all method
def test_variable_evolution_plot_all():
    L_x = np.array([1, 2, 3])
    f_x = np.array([4, 5, 6])
    C_x = np.array([[7, 8], [9, 10]])
    mu = np.array([11, 12, 13])
    beta = np.array([14, 15, 16])
    lambdas = np.array([17, 18, 19])
    file_path = os.getcwd() +'/tests'
    path = file_path + '/tmp'
    save_name = 'test'
    path =None
    save_name = None
    ve = variable_evolution(L_x, f_x, mu, beta, path,save_name,C_x,lambdas)
    ve.plot_all()
    # TODO: Add assertions to check if all the plots are saved correctly

# Run the tests
# test_variable_evolution_init()
# test_variable_evolution_aug_objective()
# test_variable_evolution_objective()
# test_variable_evolution_constraints()
# test_variable_evolution_mean()
# test_variable_evolution_variance()
test_variable_evolution_plot_all()
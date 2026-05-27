"""List CUTEst unconstrained benchmark problems with 2--4 variables (requires the ``pycutest`` package)."""

import numpy as np
import pycutest

if __name__ == '__main__':
    list_probs = pycutest.find_problems(constraints='unconstrained', userN=True, n=[2,4])
    for prob_name in list_probs:
        prob = pycutest.import_problem(prob_name)
        print(prob_name, prob.n, prob.m, prob.is_eq_cons)

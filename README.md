[Atul Agrawal](mailto:atul.agrawal@tum.de) and [Kislaya Ravi](mailto:kislaya.ravi@tum.de)

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
# scout-Nd

We introduce SCOUT-Nd (Stochastic Constrained Optimization for N Dimensions) and MF-SCOUT-Nd (Multi-Fidelity Stochastic Constrained Optimization for N Dimensions) for constrained stochastic optimization involving stochastic black-box physics-based simulators with high-dimensional parametric dependency. The proposed algorithm consists of the following major elements:

1. A non-intrusive method to estimate gradients of black-box physical simulators, with an ability to account for stochasticity in the objective and handle constraints using penalty methods.
2. Strategies to reduce the variance of the gradient estimator.
3. Ability to handle non-convexity.
4. Better optimum and well-behaved convergence properties using natural gradients.
5. Multi-fidelity strategies and adaptive selection of the number of samples for gradient estimation to provide trade-off between computational cost and accuracy.

The scripts to reproduce the studies in the paper will be made available upon publication. 

### Benchmark codes

The scripts used to reproduce the benchmark studies in the paper are in the
[`benchmarking/`](./benchmarking/) directory.

| File | Purpose |
|------|---------|
| `benchmarks.py` | Benchmark function suite (18 functions, HF/LF variants) |
| `common_data.py` | Shared constants: sample-size schedule and seed list |
| `run_scout.py` | Single-fidelity SCOUT-Nd runner |
| `run_mf_scout.py` | Multi-fidelity SCOUT-Nd runner |
| `run_scipy.py` | SciPy baseline (`trust-constr`, etc.) |
| `run_botorch.py` | BoTorch Bayesian-optimisation baseline |
| `run_cbo.py` | Constrained Bayesian-optimisation baseline (`bayes_opt`) |
| `cutest.py` | CUTEst problem listing (requires `pycutest`) |
| `adaptive_sample_plots.py` | Adaptive vs fixed sample size study + figures |
| `derivative_correction_plot.py` | Constraint-derivative correction visualisation |
| `drift_term_plot.py` | Gradient-estimator drift / convergence-rate analysis |
| `evolution_plot.py` | Design-variable / objective evolution plots |
| `moore_plot.py` | Data / performance profile plots (More--Wild style) |

generated PDF figures from the paper are in `benchmarking/`.
### Real world example 
- Windfarm layout optimization [here](https://github.com/atulag0711/windfarm_layout_optimization)
- Pipe shape optimization [code](), [student thesis](https://mediatum.ub.tum.de/doc/1749210/1749210.pdf) 


## Citation
If you use this code, please cite our paper:
```
@article{Agrawal_2025,
doi = {10.1088/2632-2153/ad8e2b},
url = {https://dx.doi.org/10.1088/2632-2153/ad8e2b},
year = {2025},
month = {jan},
publisher = {IOP Publishing},
volume = {6},
number = {1},
pages = {015024},
author = {Agrawal, Atul and Ravi, Kislaya and Koutsourelakis, Phaedon-Stelios and Bungartz, Hans-Joachim},
title = {Stochastic black-box optimization using multi-fidelity score function estimator},
journal = {Machine Learning: Science and Technology}
}
```

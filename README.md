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


## Citation
If you use this code, please cite our paper:
```
@article{agrawal:hal-04659802,
  TITLE = {{Stochastic Black-Box Optimization using Multi-Fidelity Score Function Estimator}},
  AUTHOR = {Agrawal, Atul and Koutsourelakis, Phaedon-Stelios and Ravi, Kislaya and Bungartz, Hans-Joachim},
  URL = {https://hal.science/hal-04659802},
  NOTE = {working paper or preprint},
  YEAR = {2024},
  MONTH = Jul,
  PDF = {https://hal.science/hal-04659802/file/scout_nd_preprint_neurips_formal_ml_scitech.pdf},
  HAL_ID = {hal-04659802},
  HAL_VERSION = {v1},
}
```

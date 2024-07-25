# scout-Nd
We introduce SCOUT-Nd (\underline{S}tochastic \underline{C}onstrained \underline{O}p\underline{t}imization for \underline{$N$} \underline{D}imensions) and MF-SCOUT-Nd (\underline{M}ulti-\underline{F}idelility \underline{S}tochastic \underline{C}onstrained \underline{O}p\underline{t}imization for \underline{$N$} \underline{D}imensions) for constrained stochastic optimization involving stochastic black-box simulators with high-dimensional parametric dependency.  The proposed algorithm consists of the following major elements: (a) A non-intrusive method to estimate gradients of black-box physical simulators, with an ability to account for stochasticity in the objective and handle constraints using penalty methods. (b) Strategies to reduce the variance of the gradient estimator. (c) Ability to handle non-convexity. (d) Better optimum and well-behaved convergence properties using natural gradients. (e) Multi-fidelity strategies and adaptive selection of the number of samples for gradient estimation to provide trade-off between computational cost and accuracy.

To codes to reproduce the studies in the paper will be made available upon publication. 


## Citation
```python
@unpublished{agrawal:hal-04659802,
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

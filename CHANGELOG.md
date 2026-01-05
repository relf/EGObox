# Changes

## Version 0.35.0 - unreleased

## Version 0.34.2 - 05/12/2025

* Add TREGO configuration allowing to choose nb of global/local steps and other parameters by @relf in <https://github.com/relf/EGObox/pull/359>

## Version 0.34.1 - 22/12/2025

* Fix TREGO implementation by @relf in <https://github.com/relf/EGObox/pull/351>

## Version 0.34.0 - 17/11/2025

* **Breaking changes**:
  * Add Egor configuration validation by @relf in <https://github.com/relf/egobox/pull/335>
  * Use Logarithm of Expected Improvement (LogEI) as the default infill strategy by @relf in <https://github.com/relf/egobox/pull/336>
  * Enable middle points picker heuristic by @relf in <https://github.com/relf/egobox/pull/334>
  * Add new surrogate API functions to optimize GP usage in Egor optimizer by @relf in <https://github.com/relf/egobox/pull/332>
* Add Egor examples on rosenbrock, michalewicz and ackley test functions by @relf in <https://github.com/relf/egobox/pull/325>
* Parallelize distance calculations and eliminate allocations in hot paths by @Copilot in <https://github.com/relf/egobox/pull/328>

## Version 0.33.0 - 11/10/2025

This release implements Gaussian process models quality assessment with metrics computations (Q2, PVA, IAE alpha).
See Marrel, A., & Iooss, B. (2024).
[Probabilistic surrogate modeling by Gaussian process: A review on recent insights in estimation and validation.](https://www.sciencedirect.com/science/article/pii/S0951832024001686?casa_token=l-YuUssKYfsAAAAA:WpEZkMh7bf0Ago2rDi0I_E5BydKDTmjluPOP0KfAratwka89IRqu09U1J6OnpG1gc_8FIafRzA)
Reliability Engineering & System Safety, 247, 110094.

## Version 0.32.0 - 22/08/2025

This release removes experimental or deprecated features so **breaking changes** below:

* Remove experimental GBNM internal optimizer by @relf in <https://github.com/relf/egobox/pull/285>
* GP/SGP 'predict()' and 'predict_var()' return a one-dimensional array by @relf in <https://github.com/relf/egobox/pull/286>
* Remove deprecated features by @relf in <https://github.com/relf/egobox/pull/287>

## Version 0.31.0 - 11/07/2025

* Improve `LogEI` criterion computation by @relf in <https://github.com/relf/egobox/pull/281> and <https://github.com/relf/egobox/pull/283>
* Improve hyperparameters theta bounds handling by @relf in <https://github.com/relf/egobox/pull/272>
* Implement `LogCEI` criterion by @relf in <https://github.com/relf/egobox/pull/284>

## Version 0.30.0 - 27/06/2025

* Refactor GP configuration from Egor optimizer by @relf in <https://github.com/relf/egobox/pull/273>
* Improve Egor Python API: `to_specs` is now useless and deprecated by @relf in <https://github.com/relf/egobox/pull/282>
* Add new optim multistart strategy based on picking points in-between training points by @relf in <https://github.com/relf/egobox/pull/279>
* Add pyi file generation using pyo3-stub-gen by @relf in <https://github.com/relf/egobox/pull/276>
* Fix MoE (hard) gradients, py source dir and theta init by @relf in <https://github.com/relf/egobox/pull/278>
* Add link to an [example](<https://github.com/relf/egobox/tree/master/doc#solve-an-analytical-mdo-problem-with-a-hidden-optimum>) of EGObox optimizer usage (MDO course 2025) by @jomorlier in <https://github.com/relf/egobox/pull/280>

## Version 0.29.0 - 14/05/2025

* Add experimental Global Bounded Nelder-Mead infill criterion optimizer  by @relf in <https://github.com/relf/egobox/pull/264>
* Add Rust simple examples by @relf in <https://github.com/relf/egobox/pull/266>
* Adjust bounds for GP/SGP theta hyperparameters optimization by @relf in <https://github.com/relf/egobox/pull/258>
* Make GP max nb of likelihood evaluations in hyperparameter optimization configurable by @relf in <https://github.com/relf/egobox/pull/262>
* Make CoEGO activity matrix more balanced by @relf in <https://github.com/relf/egobox/pull/265>
* Fix MOE smooth derivatives test which fails from time to time by @relf in <https://github.com/relf/egobox/pull/260>
* Refactor infill criterion optimization by @relf in <https://github.com/relf/egobox/pull/259>
* Remove last resort LHS optimization by @relf in <https://github.com/relf/egobox/pull/261>

## Version 0.28.0 - 17/04/2025

This release contains an implementation of the CoEGO [1] algorithm in a CCBO [2] setting where expensive objective evaluations are done only when all partial optimization problems are solved.

1. Zhan, Dawei, et al., [A cooperative approach to efficient global optimization](https://link.springer.com/article/10.1007/s10898-023-01316-6). Journal of Global Optimization 88.2 (2024): 327-357
2. Lisa Pretsch et al., Bayesian optimization of cooperative components for multi-stage aero-structural compressor blade design. 2024

See also some breaking changes description in optimizer options in PR #253

* CoEGO implementation and refactoring by @relf in <https://github.com/relf/egobox/pull/253>
* Prepare release 0.28.0 by @relf in <https://github.com/relf/egobox/pull/254>

## Version 0.27.0 - 13/03/2025

* `ego`: 
  * Implement upper trusted bound criterion for constraint handling by @relf in <https://github.com/relf/egobox/pull/245>
  * Implement constrained infill criterion by @relf in <https://github.com/relf/egobox/pull/246>
* `moe`: Relax number of points required wrt regression spec by @relf in <https://github.com/relf/egobox/pull/244>

## Version 0.26.0 - 14/02/2025

* `ego`: Manage constraints as functions by @relf in <https://github.com/relf/egobox/pull/242>

## Version 0.25.0 - 19/12/2024

* `gp`: **Breaking change** - Change GP training Rust API to use one dimensional array by @relf in <https://github.com/relf/egobox/pull/222>
* `doe`: Fix optimized LHS ESE algorithm  by @relf in <https://github.com/relf/egobox/pull/219>
* `egobox::Gpx`: Exception raised if training output data is not one-dimensional by @relf in <https://github.com/relf/egobox/pull/218>
* `moe`: Refactor `predict_smooth` by @relf in <https://github.com/relf/egobox/pull/221>
* Update to numpy crate 0.22.1 to fix win32 compilation by @relf in <https://github.com/relf/egobox/pull/216>

## Version 0.24.0 - 12/11/2024

* `moe`: Save/Load surrogates in binary format by @relf in <https://github.com/relf/egobox/pull/213>
* Fix badge link by @relf in <https://github.com/relf/egobox/pull/201>
* Upgrade to PyO3 0.22 by @relf in <https://github.com/relf/egobox/pull/203>

## Version 0.23.0 - 01/10/2024

* Add EGObox logo by @relf in <https://github.com/relf/egobox/pull/193>
* `gp`: Add training infos getters at Python level by @relf in <https://github.com/relf/egobox/pull/196>
* `ego`:
  * Add checkpointing and hot start by @relf in <https://github.com/relf/egobox/pull/197>
  * Do not activate TREGO by default by @relf in <https://github.com/relf/egobox/pull/198>
  * Improve hot start API by @relf in <https://github.com/relf/egobox/pull/199>
  
## Version 0.22.0 - 04/09/2024

* `ego`:
  * Save config and optim history by @relf in <https://github.com/relf/egobox/pull/190>
  * Move the DOE save after the algo the iteration (fixes #182) by @relf in <https://github.com/relf/egobox/pull/185>
* Maintainance by @relf in <https://github.com/relf/egobox/pull/183>

## Version 0.21.1 - 31/07/2024

* `gp`: Fix variance gradient computation by @relf in <https://github.com/relf/egobox/pull/177>

## Version 0.21.0 - 09/07/2024

* `ego`:
  * Implement [TREGO algorithm](https://arxiv.org/abs/2101.06808) by @relf in <https://github.com/relf/egobox/pull/173>
  * Fix added point count in TREGO local step by @relf in <https://github.com/relf/egobox/pull/174>
  * Fix WB2S criteria scaling factor and fmin computation by @relf in <https://github.com/relf/egobox/pull/175>
* Prepare release 0.21 by @relf in <https://github.com/relf/egobox/pull/176>

## Version 0.20.0 - 25/06/2024

* `gp`:
  * Save original parameters in trained `gp` models  by @relf in <https://github.com/relf/egobox/pull/166>
  * Implement cross validation metric for surrogates by @relf in <https://github.com/relf/egobox/pull/167>
* `ego`:
  * Make `n_optmod` option available in Python by @relf in <https://github.com/relf/egobox/pull/161>
  * Better `Egor` solver state handling by @relf in <https://github.com/relf/egobox/pull/168>
  * Refactor `ego` module by @relf in <https://github.com/relf/egobox/pull/169>
* Add dependabot cargo ecosystem check by @relf in <https://github.com/relf/egobox/pull/163>

## Version 0.19.0 - 15/05/2024

* `ego`:
  * Allow to reuse surrogate trainings (reuse previous hyperparameters) from a previous iteration by @relf in <https://github.com/relf/egobox/pull/157>
  * Fix hot start mechanism due to bad loaded DOE exploitation by @relf in <https://github.com/relf/egobox/pull/156>
  * Adjust log messages by @relf in <https://github.com/relf/egobox/pull/158>
  * Log Egor configuration by @relf in <https://github.com/relf/egobox/pull/159>
* `gp`:
  * Allow fixed hyperparameters theta for GP and Sparse GP by @relf in <https://github.com/relf/egobox/pull/155>
* `egobox`:
  * Upgrade to pyo3 0.21 by @relf in <https://github.com/relf/egobox/pull/152>
  * Upgrade dependencies by @relf in <https://github.com/relf/egobox/pull/153>

## Version 0.18.1 - 10/04/2024

* Fix GP mixture with kpls option on Griewank test function by @relf in <https://github.com/relf/egobox/pull/150>

## Version 0.18.0 - 08/04/2024

* [Breaking changes] `gp`, `moe`, `egobox` (Python): Rename `predict_derivatives()` as `predict_gradients()` by @relf in <https://github.com/relf/egobox/pull/148>

## Version 0.17.0 - 04/04/2024

* [Breaking changes] `gp` API renaming by @relf in <https://github.com/relf/egobox/pull/145>
  * `predict_values()` is renamed `predict()`
  * `predict_variances()` is renamed `predict_var()`
  * `predict_variance_derivatives()` is renamed `predict_var_derivatives()`
   Derivatives predictions (`predict_derivatives()` and `predict_var_derivatives()`) are made available in Python.
* Refactor Mixture of Experts by @relf in <https://github.com/relf/egobox/pull/146>
  Factorize code between full GP and sparse GP implementations

## Version 0.16.0 - 07/03/2024

* Add `Gpx` accessors by @relf in <https://github.com/relf/egobox/pull/140>
* Fix `LHS` maximin bug by @relf in <https://github.com/relf/egobox/pull/141>
* `doe`: Improve classic, centered and maximin LHS performances by @relf in <https://github.com/relf/egobox/pull/138>
* `doe`: Improve optimized LHS performances (1.25x speedup) by @relf in <https://github.com/relf/egobox/pull/136>
* Rework (mostly internal) API to avoid awkward &Option by @relf in <https://github.com/relf/egobox/pull/134>
* Add Python bindings for all LHS flavours by @relf in <https://github.com/relf/egobox/pull/135>

## Version 0.15.0 - 02/01/2024

* `gp`: Implement sparse gaussian process methods (cf. `SparseGaussianProcess`)
* Python binding: `SparseGpMix`, see doc/tutorial
* GP/SGP API
  * hyperparameter tuning : initial theta guess and bounds can be specified (`theta_init`, `theta_bounds`)
  * `n_start` controls the number of optimization multistart
* In GP/SGP `rayon` is used to make parallel optimization multistart

## Version 0.14.0 - 13/12/2023

* `ego`: Fix ask-and-tell interface `suggest()` method in presence of discrete variable to work
in discrete not in continuous space
A few API breaking changes:
* `EgorConfig::xtypes` not an option anymore
* `EgorSolver::new_with_xtypes()` renamed `new` as `new` with xlimits is removed, use `to_xtypes` to convert `xlimits`
* `EgorConfig::no_discrete` attribute removed, use `EgorConfig::discrete()` method
* `SurrogateBuilder::new_with_xtypes_rng` renamed `new_with_xtypes`

## Version 0.13.0 - 30/11/2023

* `ego`: API refactoring to enable `ask-and-tell` interface
  * Configuration of Egor is factorize out in `EgorConfig`
  * `EgorBuilder` gets a `configure` method to tune the configuration
  * `EgorService` structure represent `Egor` when used as service
  * Python `Egor` API changes:
    * function under optimization is now given via `minimize(fun, max_iters-...)` method
    * new method `suggest(xdoe, ydoe)` allows to ask for x suggestion and tell current function evaluations
    * new method `get_result(xdoe, ydoe)` to get the best evaluation (ie minimum) from given ones

## Version 0.12.0 - 10/11/2023

* `gp` uses pure Rust COBYLA by @relf in <https://github.com/relf/egobox/pull/110,https://github.com/relf/egobox/pull/113>
* `ego` as pure Rust implementation (`nlopt` is now optional) by @relf in <https://github.com/relf/egobox/pull/112>
* `egobox` Python module: Simplify mixed-integer type declaration by @relf in <https://github.com/relf/egobox/pull/115>
* Upgrade dependencies by @relf in <https://github.com/relf/egobox/pull/114>
* Upgrade edition 2021 by @relf in <https://github.com/relf/egobox/pull/109>
* CI maintainance by @relf in <https://github.com/relf/egobox/pull/111>
* Bump actions/checkout from 2 to 4 by @dependabot in <https://github.com/relf/egobox/pull/107>
* Bump actions/setup-python from 2 to 4 by @dependabot in <https://github.com/relf/egobox/pull/108>

## Version 0.11.0 - 20/09/2023

* Automate Python package build and upload on Pypi from Github CI by @relf in <https://github.com/relf/egobox/pull/104>
* Fix FullFactorial when asked nb iof samples is small wrt x dimension
by @relf in <https://github.com/relf/egobox/pull/105>
* Make mixed-integer sampling methods available in Python by @relf in <https://github.com/relf/egobox/pull/106>

## Version 0.10.0 - 22/06/2023

* `gp`, `moe` and `egobox` Python module:
  * Added Gaussian process sampling (#97)
  * Added string representation (#98)

* `egobox` Python module:
  * Change recombination enum to respect Python uppercase convention (#98)

* Notebooks and documentation updates (#97, #98, #99)

## Version 0.9.0 - 02/06/2023

* `ego`:
  * Infill criterion is now a trait object in `EgorSolver` structure (#92)
  * `Egor` and `EgorSolver` API: methods taking argument of type Option\<T\>
  now take argument of type T (#94)
  * `EgorBuilder::min_within_mixed_space()` is now `EgorBuilder::min_within_mixint_space()` (#96)
  * `egobox-ego` library doc updated (#95)

* `egobox` Python module: Upgrade to PyO3 0.18 (#91)

## Version 0.8.2 - 31/03/2023

* `ego`:
  * Fix Egor solver best iter computation (#89)

## Version 0.8.1 - 28/03/2023

* `ego`:
  * Make objective and constraints training in parallel (#86)
  * Lock mopta execution to allow concurrent computations (#84)
  * Fix and adjust infill criterion optimmization retries strategy (#87)
* `moe`:
  * Fix k-fold cross-validation (#85)  

## Version 0.8.0 - 10/03/2023

* `ego`:
  * Renaming `XType`, `XSpec` for consistency (#82)
  * Export history in optimization result (#81)
  * Use nb iter instead of nb eval, rename q_parallel as q_points (#79)
  * Warn when inf or nan detected during obj scaling computation (#78)
  * Parallelize constraint scales computations (#73)
  * Parallelize multistart optimizations (#76)
  * Handle GMM errors during MOE training (#75)
  * Handle possible errors from GMM clustering (#74)
  * Upgrade argmin 0.8.0 (#72)
  * Add mopta08 test case as example (#71)
  * Fix scaling check for infinity (#70)
  * Use kriging surrogate by default (#69)

## Version 0.7.0 - 11/01/2023

* `gp`:
  * Add analytic derivatives computations (#54, #55, #56, #58, #60). All derivatives available for all mean/correlation models are implemented.
  * Refactor `MeanModel` and `CorrelationModel` methods:
    * `apply()` renamed to `value()`
    * `jac()` renamed to `jacobian()`
  * Fix prediction computation when using linear regression (#52)
* `ego`:
  * Refactor `Egor` using [`argmin 0.7.0` solver framework](http://argmin-rs.org) `EgorSolver` can be used with `argmin::Executor` and benefit from observers and checkpointing features (#67)
  * `Egor` use kriging setting by default (i.e. one cluster with constant mean and squared exponential correlation model)
* Add [notebook on Manuau Loa CO2 example](https://github.com/relf/egobox/blob/master/doc/Gpx_MaunaLoaCO2.ipynb) to show `GpMix`/`Gpx` surrogate model usage (#62)
* Use xoshiro instead of isaac random generator (#63)
* Upgrade `ndarray 0.15`, `linfa 0.6.1`, `PyO3 0.17` (#57, #64)

## Version 0.6.0 - 2022-11-14

* `gp`: Kriging derivatives predictions are implemented (#44, #45), derivatives for Gp with linear regression are implemented (#47)
  * `predict_derivatives`: prediction of the output derivatives y wtr the input x
  * `predict_variance_derivatives`: prediction of the derivatives of the output variance wrt the input x
* `moe`: as for `gp`, derivatives methods for smooth and hard predictions are implemented  (#46)
* `ego`: when available derivatives are used to optimize the infill criterion with slsqp (#44)  
* `egobox` Python binding: add `GpMix`/`Gpx` in Python `egobox` module, the Python binding of `egobox-moe::Moe` (#31)

## Version 0.5.0 - 2022-10-07

* Add Egor `minimize` interruption capability (Ctrl+C) from Python (#30)
* Minor performance improvement in moe clustering (#29)
* Improvements following JOSS submission review (#34, #36, #38, #39, #40, #42)

## Version 0.4.0 - 2022-07-09

* Generate Python `egobox` module for Linux (#20)
* Improve `Egor` robustness by adding LHS optimization (#21)
* Improve `moe` with automatic number of clusters determination (#22)
* Use `linfa 0.6.0` making BLAS dependency optional (#23)
* Improve `Egor` by implementing automatic reclustering every 10-points addition (#25)
* Fix `Egor` parallel infill strategy (qEI): bad objectives and constraints gp models updste (#26)

## Version 0.3.0 - 2022-05-05

Improve mixture of experts (#15)

* Implement moe save/load (feature persistent)
* Rename GpSurrogate to Surrogate
* Remove `fit_for_predict`
* Implement `ParamGuard` for `MoeParams`
* Implement `Fit` for `MoeParams`
* Rename `MoeParams` setters

Refactor `moe`/`ego` relation (#16)

* Move `MoeFit` as `SurrogateBuilder` from `moe` to `ego`
* Implement `SurrogateBuilder` for `Moe`
* `Moe` uses `linfa::Fit` trait
* Rename `Evaluator` as `PreProcessor`

Refactor `MixintEgor` (#17)

* Rename `PreProcessor::eval` to `run`
* Implement `linfa::Fit` for `MixintMoeParams`, use `linfa::Dataset`
* Rename `SurrogateParams` to `MoeBuilder`
* Rename `n_parallel` to `q_parallel` (qEI stategy)

## Version 0.2.1 - 2022-04-13

* Improve documentation
* `egobox` Python module: rename egobox `Optimizer` class to `Egor`

## Version 0.2.0 - 2022-03-24

* Add hot start
* Add constraint handling
* Add mixed-integer optimization capability
* Add Python binding with PyO3

## Version 0.1.0 - 2021-11-18

Initial release

* `doe`: `LHS`, `FullFactorial`, `Random sampling`
* `gp`: Gaussian Process models with 3 regression models (constant, linear quadratic) and 4 correlation models (squared exponential, absolute exponential, matern32, matern52)
* `moe`: Mixture of Experts: find the bests mix of gps given a number of clusters
regarding smooth or hard recombination
* `ego`: Contains egor optimizer which is a super EGO algorithm implemented on top of the previous elements.
It implements several infill strategy: EI, WB2, WB2S and use either COBYLA or SLSQP for internal optimization.

# EGObox 0.39.0 release notes

Experimental multi-objective optimization, in Rust `Egor` and in Python with the new `Belfegor` optimizer.
Mono-objective optimization and the Python `Egor` API are unchanged, Rust API changes are additive.

## Multi-objective `Egor` (Rust, #476–#481)

* **API**: `EgorConfig::n_obj()` and `configure_moo(MooConfig)`. The objective function returns
  `[obj_1, ..., obj_n_obj, cstr_1, ...]`. `Egor::run_pareto()` returns a `ParetoResult` (Pareto set/front,
  compromise point, DOE) and `run()` returns the compromise point.
* **Strategies** (`MooConfig::strategy()`):

  | Strategy | Infill criterion | Default |
  |---|---|---|
  | `ParEgo` | scalarized objectives, single surrogate (Knowles 2006) | beyond 3 objectives |
  | `Eim` | Expected Improvement Matrix, Euclidean/maximin/hypervolume aggregation (Zhan 2017) | |
  | `Ehvi` | Expected Hypervolume Improvement in closed form, up to 8 objectives (Emmerich 2006) | 2 or 3 objectives |
  | `QEhvi` | batch EHVI under the joint posterior, batches of at most 4 points (Daulton 2020) | |

* **Hypervolume-based stop**: `MooConfig::hv_stop(tol, n_iters)`.
* **Supported with several objectives**:
  * constraints, function constraints and `cstr_specs`;
  * mixed-integer variables;
  * batches;
  * warm/hot start;
  * ask-and-tell (`EgorServiceBuilder`);
  * failsafe imputation (not with ParEGO).

  TREGO, CoEGO and `target` are not supported.
* **Helpers**: `find_pareto_front_indices()` and `find_compromise_index()` give the front and the compromise
  point of given data. `GaussianProcess::predict_covariance()` and `GpSurrogateExt::predict_covariance()`
  give the posterior covariance between points.

## Python `Belfegor` (#482)

```python
belfegor = egx.Belfegor(xspecs, n_obj=2, moo_config=egx.MooConfig(strategy=egx.MooStrategy.QEHVI, batch=3))
res = belfegor.minimize(fun, max_iters=20)
res.x_pareto, res.y_pareto, res.x_opt, res.y_opt  # Pareto front and compromise point
```

* Options:
  * the `Egor` ones that apply to several objectives;
  * `n_obj`;
  * `moo_config` (`MooConfig`: strategy, batch size, EIM aggregation, `hv_stop`, ParEGO options).
* Methods: `suggest()` (ask-and-tell), `pareto_result()` / `pareto_indices()` (front of given data).
* Examples: `zdt1.py`, and `belfegor_pymoo.py` on pymoo test problems with quality metrics and plots.

## Other changes

* `EgorState::param` / `cost` hold the first evaluated point of the iteration and its value, instead of the
  first proposed point and its prediction (#476).
* Fix the probability of feasibility gradient (`cstr_infill`) with a non-zero constraint tolerance (#477).
* Python deprecations: names deprecated in 0.38.0 still work with a `DeprecationWarning`. Their removal, and
  `Gpx.thetas()` / `variances()` / `likelihoods()` becoming properties, are postponed to a future release.

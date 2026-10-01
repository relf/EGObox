# EGObox 0.38.0 release notes

This release focuses on two things:

* **Faster optimization loops**: surrogates are now updated incrementally between iterations instead of
  being retrained from scratch, and the infill / hyperparameter optimizers move to pure Rust (`basin`).
* **A cleaner Python API**: standard Python exceptions, consistent names and shapes, per-constraint
  tolerances, and results you can feed back to the optimizer. Old names keep working with a
  `DeprecationWarning` and will be removed in the next release.

## Highlights

### Adaptive surrogate update (#454, #461, #462)

* Surrogate models are persisted across iterations. When the clustering is fixed and the new data stays
  consistent with the current model (z-score check), the GP is updated with the new points without
  re-optimizing its hyperparameters; otherwise a full training is triggered.
* TREGO local steps reuse and incrementally update the persisted models instead of retraining all
  surrogates from scratch at each local iteration.
* Redundant retraining at the start of each EGO step is skipped.
* The `persistent` feature is now always on in `egobox-ego` and `egobox-moe`.
* Fixes along the way: derived constraint models updated in the wrong space, stale virtual observations
  with q-batch, duplicated training data with mixed-integer variables.

### Pure Rust optimizers by default (#458, #459, #460)

* [`basin`](https://crates.io/crates/basin) optimizers (COBYLA, SLSQP) replace `argmin` and are used by
  default for infill optimization and GP hyperparameters training.
* The C-ported NLopt crates are still available behind the opt-in features `c-cobyla` and `c-slsqp`.
  The former `basin` feature is gone (it is now the default).
* The optional `nlopt` dependency is removed.

## Python API

### New and improved

* **Standard exceptions** (#466): invalid input or failure raises `ValueError`, `TypeError`, `OSError` or
  `RuntimeError` instead of `PanicException`. Errors raised by your constraint functions are propagated as
  is. `Gpx.save()` / `SparseGpx.save()` raise on failure instead of returning `False`.
* **Results**: `EgorOptim` forwards `x_opt`, `y_opt`, `x_doe`, `y_doe` from its `result` and unpacks as
  `x_opt, y_opt = egor.minimize(...)`.
* **Initial DOE as two arrays**: `Egor(x_doe=..., y_doe=...)`, like `Egor.suggest()`.
* **Per-constraint tolerance**: `CstrSpec.leq(bound, tol=1e-3)` or `{"leq": bound, "tol": 1e-3}`.
  When `cstr_specs` is given, `n_cstr` is inferred and a mismatching `n_cstr` raises `ValueError`.
* **Raw constraint layout in results** (#475): with `cstr_specs`, `OptimResult.y_opt` / `OptimResult.y_doe`
  (Rust and Python `minimize()`) hold the constraint values as returned by your function (`1 + n_cstr`
  columns) instead of the internal `<= 0` form (sign-flipped for `geq`, two columns for `eq` / `between`).
  They can be given back to `best_result()` or `Egor(x_doe=..., y_doe=...)`. DOE files saved in `outdir`
  keep the internal form.
* **`best_index()` / `best_result()`** (#474) interpret raw constraint values with `cstr_specs` and their
  tolerances, as the optimizer does. A `y_doe` not of shape `(ns, 1 + n_cstr)` raises `ValueError`.
* **Function constraints** (#473): `minimize(fcstrs=...)` also accepts a `(g, grad_g)` tuple or a
  `{"fun": g, "jac": grad_g}` dict, with one-argument callables. The `g(x, return_grad)` form stays.
  Constraints remain `g(x) <= 0`: a scipy-like `"type"` key raises `ValueError`, use `fcstr_specs` instead.
* **`Egor(seed=..., verbose=...)`** gives defaults for `minimize()` and `suggest()`; values passed to those
  methods take precedence.
* **GP models**: `Gpx.predict(x, return_std=True)` / `SparseGpx.predict(x, return_std=True)` return
  `(mean, std)`; new `nx` / `ny` properties on `Gpx` and `SparseGpx`. Input shapes are handled
  consistently across `Gpx` and `SparseGpx` (#470).
* **`sampling(xspecs, n_samples, method=Sampling.LHS, seed=None)`**: `method` is optional, LHS by default.
* **Long enum aliases**: `ConstraintStrategy.MEAN_CONSTRAINT` / `UPPER_TRUST_BOUND`,
  `QEiStrategy.KRIGING_BELIEVER` / `KRIGING_BELIEVER_LOWER_BOUND` / `KRIGING_BELIEVER_UPPER_BOUND` /
  `CONSTANT_LIAR_MINIMUM`, `FeasibleInfillStrategy.EFI_PROBABILITY` / `EFI_FEASIBILITY_ENHANCED`.
* Type stubs brought in sync with the runtime (#467), and several silent bugs found during the Python API
  review fixed (#468).

### Deprecations

Deprecated names still work but emit a `DeprecationWarning`. Passing both a deprecated keyword and its
replacement raises `TypeError`.

| Deprecated | Use instead |
|---|---|
| `Egor(n_start=...)` | `Egor(infill_n_start=...)` |
| `GpConfig(n_start=...)`, `GpConfig.n_start`, `gp_config={"n_start": ...}` | `theta_n_start` |
| `GpConfig(max_eval=...)`, `GpConfig.max_eval`, `gp_config={"max_eval": ...}` | `theta_max_eval` |
| `GpMix` / `Gpx.builder(n_start=..., max_eval=...)` | `theta_n_start=...`, `theta_max_eval=...` |
| `SparseGpMix` / `SparseGpx.builder(n_start=...)` | `theta_n_start=...` |
| `SparseGpMix` / `SparseGpx.builder(nz=..., z=...)` | `n_inducing=...`, `inducing=...` |
| `Egor.get_result(x_doe, y_doe)` | `Egor.best_result(x_doe, y_doe)` |
| `Egor.get_result_index(y_doe)` | `Egor.best_index(y_doe)` |
| `Egor(doe=...)` | `Egor(x_doe=..., y_doe=...)` |
| `Egor(cstr_tol=[...])` | per-spec tolerance: `Egor(cstr_specs=[CstrSpec.leq(0.0, tol=...)])`, `minimize(fcstr_specs=[...])` |
| `CstrSpec.btw(lower, upper)`, `{"btw": (lower, upper)}` | `CstrSpec.between(lower, upper)`, `{"between": (lower, upper)}` |
| `QEiConfig(optmod=...)`, `QEiConfig.optmod`, `qei_config={"optmod": ...}` | `optim_every` |
| `TregoConfig(n_gl_steps=...)`, `TregoConfig.n_gl_steps`, `trego={"n_gl_steps": ...}` | `n_global_local_steps` |
| `TregoConfig(d=...)`, `TregoConfig.d`, `trego={"d": ...}` | `radius_bounds` |
| `sampling(method, xspecs, n_samples, seed)` (positional) | `sampling(xspecs, n_samples, method=..., seed=...)` |

### Upcoming breaking changes (next release)

* `Gpx.thetas()`, `Gpx.variances()` and `Gpx.likelihoods()` become read-only properties
  (`gpx.thetas` instead of `gpx.thetas()`).
* The deprecated names above are removed, and `sampling` accepts only the
  `sampling(xspecs, n_samples, method=..., seed=...)` argument order.

## Breaking changes

* Previously deprecated APIs are removed (#456):
  * Rust `EgorConfig::q_batch()`, `qei_strategy()`, `q_optmod()` (use the q-EI configuration),
    `use_max_proba_of_feasibility()`, `disable_middlepicker_multistarter()`, and the deprecated
    re-exports of `SurrogateBuilder` / `XType` (use `egobox_moe::SurrogateBuilder` / `egobox_moe::XType`).
  * Runtime flags `EGOR_DO_NOT_USE_MIDDLEPICKER_MULTISTARTER` and `EGOR_USE_MAX_PROBA_OF_FEASIBILITY`.
  * Python `Egor(outdir=..., warm_start=..., hot_start=...)`: pass them to `minimize()`.
* The run recorder is removed; state recording is kept for debugging and demos (#455).
* Cargo features: `basin` removed (now default), `nlopt` removed, `persistent` always on;
  new opt-in `c-cobyla` / `c-slsqp`.
* Mixed-integer GP mixture refactored (#457): `MixintGpMixture` handles continuous-only problems without
  overhead and is the only mixture used from Python; `load_gp_models` is removed.
* With `cstr_specs`, `OptimResult.y_opt` / `y_doe` now use the raw constraint layout (see above).

## Bug fixes

* Egor no longer stops at iteration 1 when no initial point is feasible: it falls back to the probability
  of feasibility, then to a random point as a last resort (#465).
* Best point no longer `nan` when the failsafe strategy is imputation (#464).
* A function constraint may return a one-element array instead of a scalar.

## Documentation

* Cookbook recipe and example for constrained engineering problems (lessons from an aero-structural MDO
  study), and extended Egor tuning skill with constraint/failure handling and run diagnosis (#463).
* Docs and typo fixes (#469).

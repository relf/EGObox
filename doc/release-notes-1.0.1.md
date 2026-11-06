# EGObox 1.0.1 release notes

Bug-fix release with no breaking change in the Python API; breaking changes are Rust only.

This release makes the optimizer and the surrogate models more robust. It fixes more than twenty defects:
- panics in long runs, TREGO and automatic clustering;
- wrong gradients;
- non-reproducible seeded runs;
- Ctrl-C handling in Python.

It also speeds up GP fits and infill criteria gradients.

* **Python**: the API is unchanged. Read [Compatibility](#compatibility) before upgrading: binary model files
  from 1.0.0 no longer load, and some results differ for a given seed.
* **Rust**: the crates have breaking changes, see [Rust API](#rust-api).

Rust crates:
- `egobox-ego` 0.43.0;
- `egobox-moe` 0.37.0;
- `egobox-gp` 0.37.2;
- `egobox-gpx` 0.6.1;
- `egobox-doe` 0.35.2 is unchanged.

## Compatibility

* **Binary model files.** GP models and `Egor` hot-start checkpoints saved in binary format by 1.0.0 or
  earlier no longer load. GP models now hold a noise ratio (#500).
  * This affects Python `Gpx.load()` and `SparseGpx.load()` on `.bin` files, Rust `GpMixture::load()` and
    `load_gp_models()` on binary files, and hot start from an older checkpoint.
  * JSON model files and configurations still load. Retrain and save the models again, or save them as JSON.
* **Results for a given seed.** These fixes change the optimization path or the fitted models:
  * function constraints are handled in your units by the infill optimization (#524);
  * the CoEGO gradient order is fixed (#509);
  * the clustering is seeded and the GP variance portfolio points are ordered (#513), so these runs are now
    reproducible;
  * mixtures allowing several regression trends can select linear and quadratic ones (#537);
  * sparse GP mixture experts are seeded (#541);
  * the GP variance portfolio infill gradient is fixed (#549);
  * the cluster size checks count points (#553).

  Other runs can also differ slightly. The fused GP predictions change gradients at round-off level, and the
  infill optimizer can amplify that into a different path (#491).
* **New termination status.** A GP training failure during a run no longer panics (#552). In Python it
  ends with `ExitStatus.UNEXPECTED_EXIT` instead of raising `PanicException`. In Rust it ends with
  `TerminationReason::SolverExit(GP_TRAINING_ERROR)`. Both keep the best point found.
  * This happens in long runs: once the optimum is found, infill points pile up next to it until the GP
    correlation matrix is numerically singular.

## For Python users

* **Ctrl-C** (#546):
  * Before: after a `GpMix` or `SparseGpMix` fit, Ctrl-C during `minimize` exited the interpreter, and
    after a `minimize`, Ctrl-C during a fit or in plain Python code was ignored.
  * Now: Ctrl-C stops `minimize` after the current iteration, with `ExitStatus.INTERRUPT` and the points
    evaluated so far. This also holds when Ctrl-C hits inside the objective or a function constraint.
    Elsewhere Ctrl-C raises `KeyboardInterrupt` as usual.
* **Initial DOE with an `ENUM` variable** (#538): `Egor(x_doe=..., y_doe=...)` panicked or misread the DOE.
  `x_doe` holds one column per variable, as returned by `minimize`. `ENUM` columns must hold category
  indices, otherwise a `ValueError` is raised.
* **`gpx` CLI** (#539): it now loads models saved by `Gpx.save()`. JSON files were rejected, and binary
  files lost the mixed-integer input handling.
* **`SparseGpMix(seed=...)`** (#541): it is now reproducible. About half of the seeds drew experts from
  entropy.
* **Faster** (#491, #492, #494):
  * GP fits: 1.3–2.6× with squared and absolute exponential correlations, 1.4–5.5× with Matérn ones;
  * infill criteria gradients: 1.6–1.8×.
* The optimizer and model fixes below apply to Python as well, for example:
  * `trego=True` with `FailsafeStrategy.VIABILITY`;
  * function constraints in `minimize(fcstrs=...)`;
  * `coego_n_coop`;
  * automatic clustering (`GpConfig(n_clusters=0)`);
  * `GpConfig(regr_spec=RegressionSpec.ALL)`.

## Rust API

### Breaking changes

* `egobox-ego` 0.43.0: `SurrogateState`, `TregoState` and `TregoStrategy` gain fields and are now
  `#[non_exhaustive]`. Build them with `Default` and their builder methods instead of struct literals.
* `egobox-moe` 0.37.0: `SurrogateBuilder::set_noise_ratio()` is a new required method for builders
  implemented outside the crate.

### Deprecations

The `EgorSolver` methods that are internal steps of the solver are deprecated (#530):
- `ego_step`, `trego_step`, `select_next_points`, `refresh_infill_data`;
- `eval_obj`, `eval_fcstrs`, `eval_problem_fcstrs`, `eval_*infill_obj*`;
- `mean_cstr`, `upper_trust_bound_cstr`, `strip`, and others.

Use `Egor` or `EgorService` instead. They will be removed in the next breaking release.

### New features

* **Noisy functions:**
  * The full GP can estimate a homoscedastic noise, as the noise-to-signal ratio (`GpParams::noise_ratio()`,
    `GpMixtureParams::noise_ratio()`, `GpConfig::noise_ratio()`) (#500).
  * With noise modeled, `Egor` picks the best point from the posterior means. The infill criteria and the
    TREGO decrease test use estimated values, following STREGO (#504).
  * In ask-and-tell, `suggest()` follows the same rule, and `EgorServiceApi::best_index()` gives the best
    evaluated point (#505).
* **STREGO local phase** (opt-in): TREGO local steps can train the objective GP on the points around the best
  one (`TregoStrategy::local_gp()`) and minimize its posterior mean (`local_mean()`) (#506).
* **Model files:** `egobox_moe::save_models()` and `load_models()` handle several models per file and detect
  the format. `load_gp_models()` and `gpx` use them (#544).
* `egobox_ego::transform_constraints_at()` applies the constraint specs with several objectives (#548).
* `Egor::interrupt_check()` replaces the Ctrl-C handler of a run with your own check (#546).
* `GP_TRAINING_ERROR` names the termination reason for a GP training failure (#552).
* `SurrogateBuilder::set_rng()` is a provided method: `Egor` uses it to seed the clustering (#513).

## Fixes

**Optimizer:**
* **TREGO** (#508):
  * local steps panicked with the viability failsafe strategy;
  * function constraints received internal continuous-space points, with discrete values not folded and
    gradients not mapped back.
* **CoEGO** (#509):
  * gradients were permuted: index order instead of the activity order of the optimized variables;
  * function constraints received only the active components.
* **Function constraints** (#524): global steps stored their values divided by the infill scaling.
  * They are now evaluated after the objective, as in TREGO local steps, and handled in your units by the
    infill optimization.
  * An infill point that slightly violates them, beyond `cstr_tol`, is moved back to the feasible side.
  * Failed points kept wrong penalized values when some selected points were rejected.
* **Infill gradients:**
  * the GP variance portfolio used the unweighted criterion gradient (#549);
  * `cstr_infill` without a constraint surrogate or a feasible point gave a non-null gradient for a
    constant value (#550);
  * the upper trust bound constraint gradient was wrong (#491).
* **Seeded runs** with the GP variance portfolio or several clusters were not reproducible (#513).
* **Long runs** panicked on GP training failure (#552).
* **I/O errors** (#510): `load_gp_models()` returns the decoding error instead of an empty list. Saving models
  and writing output files return errors instead of panicking.

**GP and mixtures of experts:**
* **Expert selection** with several regression trends never chose linear or quadratic ones in dimension 2 or
  more, and could panic on small cross-validation folds (#537).
* **Mixture variance gradients** with smooth recombination panicked or used wrong clusters (#491).
* **Automatic clustering** (#512, #553):
  * it panicked with fewer than 5 training points;
  * cluster size checks counted array elements instead of points: a cluster needs at least 3 points and
    enough for the least demanding allowed trend.
* **Panics fixed:**
  * GP and sparse GP fits on a single training point (#532);
  * sparse GP fits with per-dimension theta bounds (#531);
  * GP fits in single precision (#514);
  * failed saves of GP and sparse GP experts and of `AffinedSurrogate` (#539, #544).
* **Sparse GP:**
  * a fixed theta was ignored with C COBYLA (#534);
  * seeded mixtures were not reproducible, and a non positive definite trial could panic (#541).

**`gpx`:** it loads models saved by Python `Gpx.save()` (#539).

## Internal

The EGO solver, GP, mixture of experts and Python binding code was restructured without behavior change:
- shared steps between global and local iterations;
- one linear algebra layer for the `blas` and pure Rust backends;
- one infill objective for mono- and multi-objective criteria.

Seeded snapshot scenarios and model and checkpoint compatibility tests guard the results (#507).

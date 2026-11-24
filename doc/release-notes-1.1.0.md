# EGObox 1.1.0 release notes

This release makes stopping a run reliable and prompt, and restores the constrained infill behaviour of 0.37.9.

* **Interruption**: Ctrl-C, termination signals and the new Python `InterruptFlag` stop `minimize`, `fit` and
  `suggest` at once, even during a long GP training or infill optimization. The result and the hot start
  checkpoint stay consistent: a resumed run goes on as if it had not been interrupted.
* **Constrained infill** (`cstr_infill`): the criterion weighted by the probability of feasibility is used from
  the first iteration again, as up to 0.37.9 (#564).
* **Python**: the API changes are additive. Read [Compatibility](#compatibility) before upgrading.
* **Rust**: the crates have breaking changes, see [Rust API](#rust-api).

Rust crates:
- `egobox-ego` 0.44.0;
- `egobox-moe` 0.38.0;
- `egobox-gp` 0.38.0;
- `egobox-gpx` 0.7.0;
- `egobox-doe` 0.35.2 is unchanged.

## Compatibility

* **Hot start checkpoints.** `Egor` hot start checkpoints saved by 1.0.1 or earlier no longer load: basin 3
  writes another format (#555) and the solver state holds a pending model update (#571). Start these runs
  again, or warm start them from their `egor_doe.npy` file (`warm_start=True`).
* **GP model files** saved by 1.0.1 still load, in binary and JSON formats.
* **Results for a given seed.**
  * Runs with `cstr_infill=True` and no feasible point in the initial DOE follow another path: see
    [`feasibility_first`](#for-python-users) to get the 1.0.1 behaviour back (#564).
  * The basin 3 SLSQP infill optimizer changes the later points of some runs slightly; benchmarks show no
    significant difference (#555).

## For Python users

* **Ctrl-C at once** (#572):
  * Before: Ctrl-C was acted on only after the GP training or infill optimization in progress, e.g. 13 s
    after Ctrl-C for a `minimize` with a 1500-point DOE, or at the end of a `GpMix` fit.
  * Now: `minimize` stops at once with `ExitStatus.INTERRUPT` and the points evaluated so far. `GpMix.fit`,
    `SparseGpMix.fit` and `suggest` raise `KeyboardInterrupt` at once.
  * An iteration interrupted before its point is evaluated is dropped. Interrupted while the models are
    updated with the evaluated point, the point is kept, and a hot start does the update first (#571).
  * Interrupted within `fun`, the iteration is dropped instead of recording a failed point (#567).
  * A custom SIGINT handler keeps the previous behaviour.
* **`egx.InterruptFlag`** (#573): stops a run from another thread (GUI, supervising thread) or from within
  `fun`, which Ctrl-C cannot do. It has the `threading.Event` methods `set()`, `clear()` and `is_set()`.

  ```python
  import egobox as egx

  flag = egx.InterruptFlag()
  # flag.set() from another thread, or within fun
  res = egor.minimize(fun, max_iters=100, interrupt=flag)
  ```

  * `Egor.minimize` and `Belfegor.minimize` stop as with Ctrl-C and return the result obtained so far with
    `ExitStatus.INTERRUPT`. An evaluation of `fun` in progress is not interrupted; a flag set within `fun`
    keeps the point.
  * `GpMix.fit`, `SparseGpMix.fit`, `Egor.suggest` and `Belfegor.suggest` raise the new `egx.Interrupted`
    exception (an `Exception` subclass).
  * egobox never clears the flag: call `clear()` to reuse it.
* **`feasibility_first`** (#564): new last argument of `Egor` and `Belfegor`, `False` by default. With
  `cstr_infill=True`:
  * `False`: the infill criterion weighted by the probability of feasibility is used from the first
    iteration, as up to 0.37.9. On MOPTA08 (124 variables, 68 constraints) the objective decreases faster.
  * `True`: the probability of feasibility alone drives the search until a feasible point is found, as in
    0.38.0 to 1.0.1.
* **`gpx` command** (#569): Ctrl-C terminates it at once. Before, it was ignored until the end of a fit, the
  model was written anyway, then a traceback was printed.
* **Deprecated build features:** the `c-cobyla` and `c-slsqp` features, given to `maturin` when building
  from source, are deprecated (see [Deprecations](#deprecations)). The wheels on PyPI do not use them.
* **Fixes:**
  * `minimize` interruption (#566): a second Ctrl-C while the interrupted iteration ends lost the result;
    Ctrl-C during the initial DOE raised `RuntimeError` instead of `KeyboardInterrupt`; an exception raised
    by a signal handler (e.g. `SystemExit` from `sys.exit()` in a SIGTERM handler) was lost and the run went
    on. It now stops the run, then is raised.
  * `Gpx.fit()` with an `ENUM` variable panicked (#559). Enum columns hold category indices, checked in
    `fit` and `predict` (`ValueError`).

## Rust API

### Breaking changes

* `egobox-ego` 0.44.0:
  * The `EgorSolver` methods deprecated in 0.43.0 are removed: `ego_step`, `trego_step`,
    `select_next_points`, `refresh_infill_data`, `have_to_recluster`, `make_viability_surrogate`, `eval_obj`,
    `eval_fcstrs`, `eval_problem_fcstrs`, `eval_grad_infill_obj`, `eval_infill_obj_with_cstrs`,
    `eval_grad_infill_obj_with_cstrs`, `mean_cstr`, `upper_trust_bound_cstr` and `strip`. Use `Egor` or
    `EgorService` instead.
  * New `EgoError::Interrupted` variant (#567).
  * `EgorServiceApi::suggest()` returns a `Result`: `EgoError::Interrupted`, or the error of a GP training
    failure instead of a panic (#572).
* `egobox-gp` 0.38.0: new `GpError::Interrupted` variant (#571).
* `egobox-moe` 0.38.0: `find_best_number_of_clusters()` takes a `&Cancellation` argument; `GpError`
  changes (#571).
* `egobox-gpx` 0.7.0: exposes the `egobox-moe` 0.38 types.

### Deprecations

The `c-cobyla` and `c-slsqp` features are deprecated and will be removed in the next breaking release. They
select the C-ported NLopt optimizers (`cobyla` and `slsqp` crates) instead of the Basin ones, used by default
since 0.38.0:
- Basin is the backend that is tested and benchmarked, and Basin COBYLA gives the same results as the C one
  (#555);
- the C optimizers cannot be stopped during an optimization start: an interruption only skips the starts not
  begun yet (#571).

The features are on `egobox-gp`, `egobox-moe`, `egobox-ego` and `egobox-gpx` (`c-cobyla` only), and on the
Python package. Build without them.

The `blas` feature of `egobox-moe` and `egobox-ego` does not enable `egobox-gp/blas`: GP training keeps the
pure Rust backend unless `egobox-gp/blas` is enabled too. The next breaking release (#577):
- forwards `blas` from `egobox-moe` and `egobox-ego` to `egobox-gp`, where the linear algebra backends differ;
- removes `MoeError::LinalgBlasError` (GP LAPACK errors are reported as `MoeError::GpError`) and the
  `ndarray-linalg` features of `egobox-moe` and `egobox-ego`.

### Interruption

* **Signal handling** (#568): the `ctrlc` dependency is replaced by a handler installed only while
  `Egor::run()` or `run_pareto()` executes, when no `interrupt_check` is given.
  * Before: the first run installed a handler for the life of the process: Ctrl-C no longer stopped the
    program afterwards, and a host SIGINT handler was replaced.
  * Now: the previous dispositions are restored when the last run returns. A signal that already has a
    handler, or is ignored (SIGHUP under `nohup`), is left alone.
  * On Unix, SIGINT, SIGTERM, SIGHUP, SIGUSR1 and SIGUSR2 stop the run after the current iteration
    (`TerminationReason::Interrupt`). For example SLURM `--signal=B:USR1@120` stops a run before the job time
    limit, with its hot start checkpoint saved. On Windows, Ctrl-C and Ctrl-Break.
  * A second signal terminates the process at once, also during a long evaluation.
* **Long phases** (#571): the interruption (signal or `Egor::interrupt_check()`) is also checked during GP
  training and infill optimization, from any thread: the check has to be cheap and thread-safe.
  * `egobox_gp::Cancellation`, set with `GpParams::cancellation()` and `SgpParams::cancellation()`: the
    multistart skips the remaining starts and a cancelled fit fails with `GpError::Interrupted`, never a
    degraded model. C COBYLA and SLSQP (`c-cobyla`, `c-slsqp`) can only skip starts not begun.
  * `GpMixtureParams::cancellation()`, `MixintGpMixtureParams::cancellation()` and
    `SurrogateBuilder::set_cancellation()` (provided, no-op by default) propagate it to the experts and the
    clustering.
  * `EgorServiceApi::interrupt_check()` interrupts `suggest()` (#572).
  * `EgoError::is_interruption()` recognizes an interrupted training.
  * `SurrogateState::pending_update` holds a model update interrupted after the evaluation, done first on
    resume. With noisy values the update is not interrupted.
* **Interrupted objective** (#567): an `ObjFn` returning `EgoError::Interrupted` stops the run cleanly, not
  subject to `stop_on_error`. The iteration is dropped and the checkpoint holds the previous state, so a hot
  start does it again exactly. Interrupted during the initial DOE, the run fails with
  `EgoError::Interrupted`.

### Other changes

* `EgorConfig::feasibility_first()`, `false` by default (#564), see [For Python users](#for-python-users).
  It is not serialized: checkpoints keep their layout, and a resumed run takes it from its configuration.
  `state.feasibility` is now set from the initial DOE: before, it stayed false during the first iteration
  even with a feasible DOE point.
* **Atomic writes** (#570): `egobox_moe::write_atomically()` writes a temporary file in the same directory,
  then renames it into place. Models (`save()`, `save_models()`, Egor GP files, `gpx fit`), the files of the
  `Egor` output directory and the `gpx` outputs use it: a killed process no longer leaves a truncated file,
  such as `egor_doe.npy` read by a warm start. It does not protect from a system crash (no `fsync`).
* **basin 3.3** (#555): hot start checkpoints use postcard instead of bincode. Paired benchmarks against basin
  1.13 (Egor defaults, 20 seeds, Rosenbrock, g24, Styblinski-Tang and Ackley up to 20 variables) show no
  significant difference and the same run times.

## Internal

* Dead code removed from the theta bounds: the kernel influence factors never changed the default bounds
  (#562). Behaviour is unchanged.
* Test suite trimmed of trivial tests and runs covered by snapshots (#556), Rust build cache in CI (#557).
* Snapshot fixtures recorded in CI for Windows as for Linux, Linux fixtures updated for the Ubuntu 26.04
  runner (#565).

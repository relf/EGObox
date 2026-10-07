# Multi-objective optimization (MOO) in Egor — gradual implementation plan

> Scope: the Rust crate `egobox-ego` only. The Python bindings are out of scope: they only consume
> the Rust API, which this plan changes additively, so they keep compiling unchanged. Exposing MOO in
> Python is a separate, later plan.
> Code references are as of egobox-ego 0.41.1 (master at `3a4440e` plus the warm-start fix).

## 1. Goal

Let Egor minimize m ≥ 2 expensive objectives, with the existing constraint machinery, and return an
approximation of the Pareto front. Existing mono-objective users must see **no change**: same API,
same results for a given seed, same output files.

Target usage:

```rust
// objective function returns rows [f1, f2, c1, ..., ck]
let res = EgorBuilder::optimize(zdt1)
    .configure(|cfg| cfg.n_obj(2).max_iters(40).seed(42))
    .min_within(&xlimits)?
    .run_pareto()?;            // new; run() keeps working
println!("{}", res.y_pareto);
```

## 2. Compatibility contract

**Frozen (must not change):**
- Signatures and semantics of public items: `EgorFactory`/`EgorBuilder`, `Egor::run`, `EgorConfig`
  builder methods, `EgorServiceFactory`/`EgorServiceApi::suggest`, the public `EgorSolver` methods
  (`new`, `suggest`, `ego_step`, `select_next_points`, `trego_step`, `refresh_infill_data`, `eval_*`,
  `mean_cstr`, ...), and the free functions (`find_best_result_index`, `transform_constraints`,
  `untransform_constraints`, `internal_cstr_mapping`, `transform_function_constraints`, `to_xtypes`,
  `load_gp_models`, ...).
- Structs whose fields are all public get **no new fields**, because that breaks struct literals and
  exhaustive patterns: `OptimResult`, `EgorState` and its sub-states (`DoeState`, `SurrogateState`,
  `TregoState`, `CoegoState`), `InfillObjData`, `RuntimeFlags`, `TregoStrategy`, `RunInfo`.
- Exhaustive enums get **no new variants**: `InfillStrategy`, `ConstraintStrategy`, `InfillOptimizer`,
  `QEiStrategy`, `CstrSpec`, `InternalCstrKind`, `TerminationReason`, `TerminationStatus`,
  `CoegoStatus`, `HotStartMode`, `IterationMode`, `InfillComposition`. (`FailsafeStrategy` and
  `FeasibleInfillStrategy` are `#[non_exhaustive]` and could take new variants.)
- Traits users can implement get no new required methods and no signature changes: `InfillCriterion`,
  `IterationStrategy`, `ActivityStrategy`, `ObjFn`, `CstrFn`.
- Behavior with `n_obj == 1`: identical `x_doe`/`y_doe` for a given seed and backend, and no extra
  RNG draw on the mono path.
- `outdir` files (`egor_config.json`, `egor_initial_doe.npy`, `egor_doe.npy`, `egor_history.npy`,
  `egor_failed_points.npy`, GP files): same names and layouts for mono runs. New data goes only into
  new JSON keys or new files.

**Rules for new code:**
- New public enums are `#[non_exhaustive]`. New public structs either use `pub(crate)` fields plus
  builder methods (like `GpConfig` and `QEiConfig`) or are `#[non_exhaustive]`.
- New fields of serialized internal structs (`ValidEgorConfig`) get `#[serde(default)]`, so older
  JSON configs still deserialize. Hot-start checkpoints are bincode (basin `ExactCheckpoint`), so
  they are already version-specific. Resuming a checkpoint across egobox versions stays unsupported,
  as it is today.
- MOO state is not persisted in `EgorState`. The Pareto set, normalization bounds and reference
  point are recomputed from `state.surrogate.data` each iteration. ParEGO weights are drawn from the
  state RNG, so hot start stays exact.
- Expose the minimum: the MOO criterion abstraction stays crate-private until it is stable.
- Every PR must pass `cargo semver-checks check-release -p egobox-ego --baseline-rev <base>`
  cleanly and keep the mono snapshot tests green (Step 0).

## 3. Code review: where single-objective is assumed

Today the output row is `[obj, cstr_1..cstr_k]`. The rule "column 0 is the objective, models[0] is
the objective model" is implicit in:

| Area | Location | Assumption |
|---|---|---|
| Init | `solver/egor_solver.rs:254-255` | `clusterings`/`theta_inits` sized `n_int_cstr + 1` |
| Init, warm start | `solver/egor_solver.rs:203-207` | warm-start DOE split with y width `1 + n_internal_cstr` |
| Best point | `utils/find_result.rs:10-172` | `cstr_sum` slices `1..`; `cstr_min`/`find_best_result_index` compare column 0 |
| Best point update | `solver/solver_impl.rs:799-815`, `solver/trego.rs:251-265` | incremental `find_best_result_index_from` |
| Constraint specs | `types.rs:222-315` | `internal_cstr_mapping` index 0 = objective; public `transform/untransform_constraints` copy column 0 |
| Model training | `solver/solver_impl.rs:496-598`, `1397-1410` | columns `0..=n_internal_cstr`, model k ↔ column k, k = 0 named "Objective" |
| Model refresh | `solver/solver_impl.rs:1257-1343`, `1420-1508` | `models_up_to_date` only compares row counts; z-score check is all-or-nothing |
| Infill data | `solver/solver_impl.rs:429-431`, `1014-1020` | `models.split_first()`, `fmin = y[[best, 0]]`, single `xbest` |
| Infill criterion | `criteria/mod.rs:33`, `solver/solver_computations.rs:142-213`, `365-599` | one model plus `fmin`; scaling sampled on the single criterion |
| Infill optimization | `solver/solver_infill_optim.rs:32-41`, `98`, `111-197` | `InfillOptProblem.obj_model`; fallback value `current_best.1[0]` |
| Batch (qEI) | `solver/solver_computations.rs:285-313`, `solver/solver_impl.rs:1148-1160` | virtual point = 1 objective + cstrs; `ConstantLiarMinimum` = argmin of column 0 |
| Failsafe | `solver/solver_computations.rs:321-360`, `628-631`; `solver/solver_impl.rs:971-981` | penalized `res[0]` = objective; NaN fill with `1 + n_cstr` columns |
| TREGO | `solver/trego.rs:103-104`, `203`; `solver/iteration_strategy.rs:225` | decrease of column 0 around a single best point |
| State, executor | `solver/egor_state.rs:620-647`, `738-762`; `executor.rs:49-61`, `217-221` | `get_cost`/`get_best_cost` = column 0; target stop |
| Results, files | `egor.rs:370-410`, `449-496` | `y_opt` = best row; history = best row per iteration |
| Ask-and-tell | `solver/solver_impl.rs:54-114` | `1 + n_int_cstr` models; best index on column 0 |

These parts don't depend on the output count and are reused as is: data handling (`update_data`,
`filter_nans`, `usable_data` in `utils/misc.rs`), function constraints (`c_data`), mixed-integer
relaxation, multistarters, PoF utilities (`utils/cstr_pof.rs`), checkpoint and observer plumbing.

## 4. Target design

**Output layout.** Objective function rows become `[f_1..f_m, c_1..c_k]`, with m = `n_obj` and a
default of 1, which is today's layout. Internally `y_data = [f_1..f_m, internal cstrs]`; `c_data` is
unchanged.

**Objective modeling mode** (an internal enum derived from the config):
- `Single` (m = 1): today's code path.
- `Scalarized` (ParEGO, Knowles 2006):
  - Each iteration builds a training view `[s_λ(f̃) | cstrs]`, where `f̃` is the objectives
    normalized with observed min/max.
  - `s_λ = max_j λ_j f̃_j + ρ Σ_j λ_j f̃_j` (augmented Tchebycheff, ρ = 0.05).
  - λ is drawn from a simplex lattice using the state RNG.
  - One GP is trained on `s_λ`, plus the constraint GPs, so the existing pipeline applies unchanged:
    EI/LogEI/WB2, PoF, `cstr_specs`, function constraints, qEI Kriging believer, mixed-integer.
- `PerObjective` (EIM, EHVI): one GP per objective (`models[..m]`) plus the constraint GPs, and a
  multi-objective criterion computed on the current Pareto front.

**"Best" bookkeeping for MOO.** `state.surrogate.best_index` points to a compromise point: the
feasible Pareto point that minimizes the uniform-weight Tchebycheff on normalized objectives, with
ties going to the lowest index. When no point is feasible, it falls back to the least-violation point,
as today. This keeps `EgorState::update`, basin best tracking, `egor_history.npy` and
`x_opt`/`y_opt` meaningful without new state fields. `get_best_cost()` returns f_1 of that point; it
is documented and not used for decisions in MOO mode.

**API (additive):**
- `EgorConfig::n_obj(usize)`.
- `EgorConfig::configure_moo(|moo| ...)` with a `MooConfig` (`pub(crate)` fields, builder for the
  strategy and its parameters).
- `#[non_exhaustive] enum MooStrategy { ParEgo, .. }`, with `Eim(..)` and `Ehvi` added as they are
  implemented. `ParEgo` is the default when m > 1.
- `Egor::run_pareto() -> Result<ParetoResult<f64>>`. `ParetoResult` is `#[non_exhaustive]` with
  `x_pareto`, `y_pareto`, `x_doe`, `y_doe`, `state`, in the raw constraint layout like
  `OptimResult`. For m = 1 the front is the single best point.
- `run()` still returns `OptimResult`. When m > 1, that is the compromise point.
- Helpers `pareto_front_indices(..)` and `hypervolume(..)` get exported once they are stable.
- `InfillCriterion` does not change.

**New module** `crates/ego/src/moo/`: `pareto.rs`, `hypervolume.rs`, `scalarization.rs`,
`criteria/{eim,ehvi}.rs`, `config.rs`, `result.rs`.

## 5. Gradual steps

Each step is one PR or a few PRs. Each keeps CI green and respects the contract in §2.

### Step 0 — Safety net (tests and tooling only) — done
- Add mono snapshot tests in `crates/ego/tests/mono_snapshots.rs`, next to `execution_contract.rs`:
  - Scenarios, all seeded:
    - xsinx with EI, LogEI and WB2.
    - G24 with `n_cstr`, and again with `cstr_specs`.
    - Function constraints.
    - qEI with batch 3, Kriging believer and constant liar.
    - Mixed-integer.
    - Failsafe imputation and viability.
    - TREGO, CoEGO.
    - Warm start, hot-start continuation.
  - Store `[x_doe, y_doe]` from current master as `.npy` fixtures in
    `crates/ego/tests/snapshots/<os>-<backend>/` and assert bit-for-bit equality. Floating point
    results differ across OS libm and the `c-cobyla`/`c-slsqp`/`blas` backends, so fixtures are
    per platform and a scenario without fixture for the current platform is skipped. Fixtures are
    recorded with `EGOBOX_UPDATE_SNAPSHOTS=1` (only `windows-default` so far; record the
    `linux-default` ones to get CI coverage).
  - A legitimate fixture update must be its own, explicitly reviewed commit.
- Add a `cargo semver-checks` job to `.github/workflows/lint.yml`, run on pull requests against the
  base branch (0.41.1 is not published on crates.io, so there is no registry baseline).

### Step 1 — Internal refactor: explicit output layout (no behavior change) — done
- `ValidEgorConfig` gets `pub(crate) n_obj: usize` (default 1, `#[serde(default)]`), not settable
  yet. Helpers: `n_obj()`, `n_obj_models()` (1 for Single and Scalarized, m for PerObjective),
  `n_surrogates()`, `ny_raw() = n_obj + n_cstr`, `ny_internal() = n_obj + n_internal_cstr()`.
- Offset-aware internal variants:
  - `transform_constraints_at(y, n_obj, specs)` and `untransform_constraints_at`.
  - `output_mapping(n_obj_models, specs)`.
  - `cstr_sum_at` and `is_feasible_at` (violation and feasibility with `n_obj` leading columns).

  The current public functions become wrappers with `n_obj = 1`. The best-index functions stay
  mono-objective: they apply as is to the ParEGO training view.
- Model management keeps the "model k ↔ column k" rule: loops run over `n_surrogates()` /
  `models.len()` instead of `0..=n_internal_cstr`. The solver feeds the surrogates a training view
  whose columns match the models (`[s_λ | cstrs]` for ParEGO, `[f_1..f_m | cstrs]` per objective),
  so no explicit column mapping is needed.
- Replace `split_first()` with `split_at(n_obj_models)`. `compute_virtual_point` and
  `compute_penalized_point` take objective-model slices. The NaN fill in `eval_obj` uses `ny_raw()`,
  and the warm-start DOE split in `init_state` uses `nx + ny_internal()`.
- Public `EgorSolver` methods keep their signatures and delegate to the generalized `pub(crate)` code.
- Exit criteria: snapshots bit-identical, semver-checks clean, clippy clean.

### Step 2 — Pareto toolkit (pure functions in `moo/`) — done
- Non-dominated filtering with constrained domination: feasible points dominate infeasible ones,
  then points compare by violation sum using `cstr_tol` over y constraints and `c_data`. Non-finite
  rows are excluded, as in `find_best_result_index`.
- Normalization (ideal and nadir from data), augmented Tchebycheff, and simplex-lattice weights.
  ParEGO uses s = 10 divisions for m = 2 and s = 4 for m = 3.
- Hypervolume: exact sweep for m = 2 and exact recursive slicing along the last objective for
  m ≥ 3, which is cheap for the small fronts of EGO. The reference point is nadir + 10 % of the
  range.
- Compromise point selection.
- Unit tests on analytic fronts (ZDT1/2, DTLZ2 samples) and hand-computed hypervolumes. No solver
  change in this step.

### Step 3 — MOO v1: ParEGO and the public API (experimental) — done
- Expose `EgorConfig::n_obj`, `configure_moo`, `MooConfig`, `MooStrategy::ParEgo`,
  `Egor::run_pareto` and `ParetoResult`.
- `check()` validates `n_obj ≥ 1`. With `n_obj > 1`:
  - It rejects TREGO (any non-standard iteration strategy), CoEGO, `target` and
    `FailsafeStrategy::Imputation` with explicit errors. Imputation is lifted in Step 6.
  - The GP-variance portfolio flag is ignored with a warning, since it can come from an environment
    variable.
  - A runtime guard covers custom `IterationStrategy` impls that return `IterationMode::Local`.
  - The ask-and-tell `EgorServiceBuilder` rejects `n_obj > 1` until Step 6.
- Scalarized path in `ego_step`, for each try of the point-addition loop:
  1. Draw λ from the RNG (MOO path only). Drawing new weights on retry matters: with the same λ,
     a rejected point (too close to the data) tends to be proposed again until the solver stops.
  2. Build the `[s_λ | cstrs]` view (`moo/parego.rs`).
  3. Clear the persisted surrogates, so that `select_next_points` retrains all of them on the view
     (theta warm-started from `theta_inits`). The scalarized targets change at every try, so the
     incremental `update` path must never be used: `models_up_to_date` only compares row counts.
  4. Run `select_next_points` on the view, with the best index of the view.
  5. Evaluate the new points, then `update_data` on the raw data.
  6. Recompute the compromise index from all data; the current cost is the evaluated point.
  7. Skip the end-of-iteration model refresh (models are retrained at next iteration).
- Plumbing: `init_state` (compromise index), untransforming with the objective offset in
  `run()`/`run_pareto()`. A per-iteration `egor_pareto.npy` in `outdir` is not done.
- Tests in `crates/ego/tests/moo.rs`:
  - ZDT1 (2 objectives, hypervolume above 85 % of the true front one, 95.8 % in practice),
    BNH (2 objectives, 2 constraints, feasible front reaching both extremes) and DTLZ2
    (3 objectives, mean distance to the true front).
  - Seeded determinism.
  - Hot-start continuation equals an uninterrupted run.
  - A mixed-integer variant.
  - `cstr_specs` with m = 2.
  - Unsupported configurations, `run()` compromise point, mono-objective `run_pareto()`.
- Example: `crates/ego/examples/zdt1.rs`.

### Step 4 — Per-objective surrogates and EIM
- `PerObjective` mode: `models[..m]` are persisted and updated incrementally per column.
- Crate-private MOO criterion trait providing value and gradient, given the objective models, the
  normalized front and the reference point.
- Generalize `InfillOptProblem` into an infill objective `Single | Multi`, so constraint handling is
  shared: `cstr_infill` PoF/logPoF, metamodel constraints, function constraints, viability, and the
  fallbacks. `Single` runs today's code unchanged.
- EIM, the expected improvement matrix of Zhan et al. (2017), in Euclidean, maximin and hypervolume
  variants. It is built from `ExpectedImprovement` (`criteria/ei.rs`) applied per objective and
  Pareto point, with analytic gradients, and exposed as `MooStrategy::Eim(..)`.
- Scaling: generalize `compute_infill_obj_scale` to take the criterion as a closure.
- qEI: Kriging believer per objective. The front used for the next batch points includes the
  virtual points already chosen.

### Step 5 — EHVI and a hypervolume-based stop
- EHVI: closed form for m = 2; Monte Carlo with common random numbers for m ≥ 3, using
  finite-difference gradients or Cobyla.
- Optional stop when the hypervolume gain stays below a tolerance for k iterations. It is reported
  as the existing `TerminationReason::SolverConverged`.
- Decide whether to publish the MOO criterion trait, typetag-serialized like `InfillCriterion`.

### Step 6 — Feature coverage
- Failsafe imputation for m > 1: per-objective pessimistic prediction, or the worst observed value
  per objective in ParEGO mode.
- Ask-and-tell: `EgorServiceApi::suggest` with `n_obj` columns, plus a front helper.
- ParEGO batch with one λ per batch point. In PerObjective mode, `ConstantLiarMinimum` uses the
  ideal point as the lie.
- TREGO and CoEGO: either define them around the compromise point (trust region center or CoEGO
  context vector) or keep rejecting them.

### Step 7 — Docs and stabilization
- Update the docs that say "1 objective + n_cstr": `lib.rs`, `egor.rs`, `types.rs` (`ObjFn`) and
  `egor_config.rs` (`n_cstr`). Also update the README and CHANGELOG.
- Add ZDT/DTLZ benches to `crates/ego/benches/ego.rs`.
- Drop the "experimental" label once the defaults (strategy, ρ, reference point) are settled; they
  become part of the contract at that point.

## 6. Suggested sequencing

| Milestone | Steps | User visible |
|---|---|---|
| A | 0, 1, 2 | nothing (tests, refactor, utilities) |
| B | 3 | `n_obj`, ParEGO, `run_pareto` (experimental) |
| C | 4, 5 | EIM, EHVI |
| D | 6, 7 | remaining features, stable MOO |

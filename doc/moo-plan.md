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
    `crates/ego/tests/snapshots/<os>-<backend>/` and compare them with a tight tolerance
    (1e-8 + 1e-6 × the column magnitude, as values close to zero such as active constraints
    amplify tiny differences of the evaluated points). Floating point results differ across OS libm and the
    `c-cobyla`/`c-slsqp`/`blas` backends, so fixtures are per platform and a scenario without
    fixture for the current platform is skipped. Exact equality is not usable: on a given OS, math
    library code paths depend on the CPU, and CI runners differ from developer machines by ~1e-9.
    Fixtures are recorded with `EGOBOX_UPDATE_SNAPSHOTS=1`: `windows-default` (local) and
    `linux-default` (recorded on the CI ubuntu runner). The ubuntu stable default job sets
    `EGOBOX_REQUIRE_SNAPSHOTS=1` so that a missing fixture fails.
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
- Exit criteria: snapshots unchanged, semver-checks clean, clippy clean.

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
  1. Take the next λ of the simplex lattice, shuffled once per iteration with the RNG (MOO path
     only). Each retry uses a distinct λ and the retry budget is the lattice size: the optimum of
     a scalarization often lies on a bound, so a rejected point (too close to the data) tends to be
     proposed again with the same λ, and 3 tries made the solver stop early with the C optimizers.
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
  - ZDT1 (2 objectives, hypervolume above 85 % of the true-front hypervolume, 95.8 % in practice),
    BNH (2 objectives, 2 constraints, feasible front reaching both extremes) and DTLZ2
    (3 objectives, mean distance to the true front).
  - Seeded determinism.
  - Hot-start continuation equals an uninterrupted run.
  - A mixed-integer variant.
  - `cstr_specs` with m = 2.
  - Unsupported configurations, `run()` compromise point, mono-objective `run_pareto()`.
- Example: `crates/ego/examples/zdt1.rs`.

### Step 4 — Per-objective surrogates and EIM — done
- `PerObjective` mode (`MooStrategy::Eim`): `models[..m]` (one GP per objective) are persisted and
  updated incrementally per column like the mono-objective ones (training targets do not change).
- EIM, the expected improvement matrix of Zhan et al. (2017), in Euclidean, maximin and hypervolume
  aggregations (`MooConfig::eim_aggregation`), in `moo/eim.rs`. Objectives are normalized with
  their observed bounds; the expected improvements and their gradients are computed in closed form
  from the predicted means/variances and their gradients (one prediction per objective instead of
  one per front point); the reference point of the hypervolume aggregation is the front nadir +
  10 % of its range.
- Rather than a generic criterion trait, `InfillOptProblem` takes an optional EIM criterion: when
  set, the infill objective is `-EIM / scale`, combined with the probability of feasibility as
  for linear criteria, so constraint handling is shared (`cstr_infill` PoF, metamodel constraints,
  function constraints, viability, fallbacks). The mono-objective path is unchanged.
- Scaling: max of EIM over an LHS sample, as for linear mono-objective criteria.
- qEI: Kriging believer per objective; the front used for the next batch points includes the
  virtual points already chosen. `ConstantLiarMinimum` lies with the row of the minimum of the
  first objective.
- `EgorSolver::suggest` uses the compromise point and the per-objective models.
- Feasibility-enhanced infill (EFI) is rejected with EIM.
- `Eim` is a unit variant with the aggregation in `MooConfig`: a tuple variant in `MooStrategy`
  is reported as a breaking change by `cargo semver-checks` (enum discriminants).
- Results (`crates/ego/tests/moo.rs`, 30 iterations): ZDT1 hypervolume 92 % (Euclidean), 89 %
  (maximin), 95 % (hypervolume aggregation) of the true-front hypervolume vs 96 % with ParEGO; DTLZ2 mean
  distance to the true front 0.06 vs 0.23 with ParEGO.

### Step 5 — EHVI and a hypervolume-based stop — done
- `MooStrategy::Ehvi` (`moo/ehvi.rs`), in closed form for any number of objectives instead of a
  closed form for m = 2 and Monte Carlo above: the region below the reference point not dominated
  by the front is decomposed into boxes built on the grid of the front coordinates (for each
  cell of the grid of the first m - 1 objectives, the non-dominated part is a single box
  `]-inf, u_m]` along the last objective, so at most `(n_front + 1)^(m - 1)` boxes; for m = 2,
  the classic `n + 1` stripes of the staircase); the hypervolume improvement of `y` is
  `sum_boxes prod_j (u_j - max(y_j, l_j))^+`, whose expectation factorizes per objective as
  `EI_j(u_j) - EI_j(l_j)` with independent normal predictions. Analytic gradients by the product
  rule. Boxes are built once per infill optimization; it matches a Monte Carlo estimate to
  ~1e-5 for 2 and 3 objectives (unit test). The decomposition work (boxes × front size ×
  objectives) is bounded to 2^26 and the evaluation work (boxes × objectives) to 2^16: beyond
  (front of more than 5792 points for 2 objectives, 146 for 3, 24 for 4, 9 for 5), the region
  dominated by a spread subset of the front (best point of each objective, then farthest point
  sampling) is used, with a warning. EHVI is limited to 8 objectives; EIM is the alternative
  beyond.
- EIM and EHVI share the normalized predictions, normalized front and reference point
  (`moo/criterion.rs`) and are dispatched by a crate-private `MooCriterion` enum in
  `InfillOptProblem`. The criterion is not published as a public trait: two concrete strategies
  are enough for now.
- `MooConfig::hv_stop(tol, n_iters)`: stop when the hypervolume of the constrained front
  increased by less than `tol` (relative) over the last `n_iters` iterations, both fronts being
  measured with the same normalization and reference point and recomputed from the data (the
  previous front uses the data without the last `n_iters * batch` rows: rejected or failed
  points make the window longer, which only delays the stop, as iteration boundaries are not
  kept in the state), reported as `TerminationReason::SolverConverged`. Only feasible points
  count: without feasible point the hypervolume is zero, so the stop never triggers before
  feasibility is reached. Hypervolumes are exact when the recursive computation is affordable
  (`front_size^(m - 1)` up to 2^22), estimated otherwise by Monte Carlo with 2^16 uniform samples
  shared by both fronts (low variance of their difference).
- Batches (`configure_qei`) use the Kriging believer heuristic, as with EIM: each batch point
  maximizes the single-point EHVI wrt the front augmented with the virtual points (predicted
  means) already chosen. The joint expected hypervolume improvement of the batch (qEHVI,
  Daulton et al. 2020) is available as `MooStrategy::QEhvi`, see step 6.
- Results (30 iterations): ZDT1 hypervolume 93.7 % of the true-front hypervolume (91.7 % with
  Kriging believer batches of 3), DTLZ2 mean distance 0.08 to the true front; with `hv_stop(1e-3, 5)` a ZDT1
  run with a budget of 100 iterations stops after 38.

### Step 6 — Feature coverage — partly done
Done (easy wins):
- Ask-and-tell: `EgorServiceBuilder` accepts `n_obj > 1`, `EgorSolver::suggest` handling ParEGO,
  EIM and EHVI. Like in mono-objective, the ask-and-tell loop has no rejection of points too close
  to the data: when the criterion has nothing better, the current best (compromise) point may be
  suggested again.
- Failsafe imputation with per-objective strategies (EIM, EHVI): per-objective pessimistic
  prediction (`compute_penalized_point` with objective-model slices). Failed points stored with
  imputed values (`failed_rows`) are excluded from the Pareto front, the compromise point and the
  hypervolume progress, as imputed points are never the best in mono-objective.
- In PerObjective mode, `ConstantLiarMinimum` uses the ideal point (minimum of each objective
  and constraint column) as the lie.
- qEHVI (Daulton et al. 2020), `MooStrategy::QEhvi`: batches (`configure_qei`) selected by
  sequential greedy optimization. The first point maximizes EHVI, each following point maximizes
  the expected hypervolume improvement it brings over the front augmented with the points
  already selected, under the joint posterior of the objective surrogates at these points and
  `x` (the surrogates are not updated with virtual points within the batch).
  - Joint posterior: `GaussianProcess::predict_covariance` and the provided
    `GpSurrogateExt::predict_covariance` (error by default; GP surrogates, single-cluster
    `GpMixture`, `AffinedSurrogate`, `MixintGpMixture`), hence single-cluster surrogates only.
  - Monte Carlo estimate with 128 standard normal base samples drawn once per batch point from
    the solver RNG (common random numbers: deterministic and reproducible on hot start),
    semi-definite Cholesky factor (null pivots for perfectly correlated points).
  - The improvement of a sample of `x` over the region not dominated by the front nor by the
    samples of the selected points is computed by inclusion–exclusion over the subsets of the
    selected points (`2^k` terms) on the EHVI box decomposition (`BoxDecomposition`, shared with
    EHVI, front reduced to a spread subset beyond a qEHVI evaluation budget of 2^22).
  - Gradients by central finite differences; for discrete variables (snapped by the mixed-integer
    surrogates before prediction), slope between the adjacent valid levels of the current level
    (one-sided at a domain bound, divided by the actual distance), 0 and 1 for enum one-hot
    dimensions. The scaling uses 50 points.
  - The covariance support is checked when the criterion is built: on failure, the batch stops
    with an error logged instead of optimizing a vanishing criterion.
  - Limits: batches of at most 4 points, at most 8 objectives, single-cluster surrogates.
  - Results (ZDT1, batch of 3, 8 iterations): hypervolume 88.9 % of the true-front hypervolume
    (91.7 % with EHVI Kriging believer batches on the same run): no clear gain on this small
    bi-objective case, where the Kriging believer is already a good batch heuristic.

Remaining:
- Failsafe imputation with ParEGO: the surrogates are trained on the scalarized view, so the
  penalized rows (view layout) do not match the raw data layout; e.g. worst observed value per
  objective.
- A front helper for ask-and-tell users (e.g. a public `pareto_front_indices`).
- ParEGO batch with one λ per batch point (retraining the scalarized model inside the batch
  loop).
- TREGO and CoEGO: either define them around the compromise point (trust region center or CoEGO
  context vector) or keep rejecting them.
- qEHVI follow-ups: several clusters (joint covariance across experts), analytic
  (reparameterized) gradients instead of finite differences, larger batches (the
  inclusion–exclusion grows as `2^k`).
- Function constraint values stored for points added by iterations (`c_data`) come from the
  scaled optimizer closures (divided by the function constraint scale), unlike the initial DOE
  ones; the EIM front of qEI virtual points uses them too. Storing raw values is the fix, but it
  has to come with infill points feasible in raw units: the C-ported COBYLA (no constraint
  tolerance) returns boundary points violating the scaled constraints by ~2e-5, i.e. ~5e-4 in
  raw units, which the scaled values accidentally accept within `cstr_tol` (attempted in #477
  and reverted, as C COBYLA runs with function constraints then stalled).

### Step 7 — Docs and stabilization — done (still experimental)
- Docs updated for several objectives: crate docs (`lib.rs`: multi-objective section with a ZDT1
  example, strategies, references), module docs (`egor.rs`, `egor_solver.rs`), `ObjFn`,
  `OptimResult` (compromise point), `EgorState::cost`, `EgorFactory::optimize`, `target`
  (mono-objective only). READMEs mention multi-objective optimization and the `zdt1` example.
- Default strategy when not set: EHVI for 2 or 3 objectives, ParEGO beyond (`MooConfig` stores an
  optional strategy resolved with `n_obj`); LogEI stays the mono-objective default.
- ZDT1 (2 objectives) and DTLZ2 (3 objectives) benches with each strategy in
  `crates/ego/benches/ego.rs` (`moo` group).
- The "experimental" label is kept for a release to gather feedback before freezing the defaults
  (strategy, ρ, reference point margin): they become part of the contract once it is dropped.

## 6. Suggested sequencing

| Milestone | Steps | User visible |
|---|---|---|
| A | 0, 1, 2 | nothing (tests, refactor, utilities) |
| B | 3 | `n_obj`, ParEGO, `run_pareto` (experimental) |
| C | 4, 5 | EIM, EHVI |
| D | 6, 7 | remaining features, stable MOO |

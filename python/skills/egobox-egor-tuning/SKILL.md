---
name: egobox-egor-tuning
description: >
  Use this skill whenever the user is working with the Egor optimizer for
  Bayesian optimization. Triggers on requests to tune Egor parameters, diagnose
  optimization issues, or configure Egor for specific problem types (high-dimensional,
  constrained, parallel, expensive objectives, etc.).
---

# EGOR Tuning Skill

## Goal

Provide practical guidance for selecting and adapting EGOR optimization parameters based on:
- problem dimension
- objective evaluation runtime  
- available optimization budget
- parallel evaluation availability
- convergence progress
- constraint complexity

## Cookbook Reference

For detailed parameterization recipes with complete examples, see the **[EGObox Cookbook](../../website/content/cookbook.md)**.

The cookbook contains 13 practical recipes covering:
- Cheap vs. expensive objectives
- High-dimensional problems (d > 10, d > 50)
- Parallel/batch evaluations
- Stagnation recovery
- Constraint handling (various forms)
- Cheap constraints
- Warm restarts from existing DOE
- Constrained engineering problems with active constraints (e.g. MDO)
- Batch Thompson Sampling in low dimension

---

## Quick Reference

### Dimension Guidelines

| Dimension | Strategy |
|-----------|----------|
| d < 10 | Standard Egor with adequate DOE |
| 10 < d ≲ 15 | Try full GPs first, then KPLS if GP training is too slow |
| d > 15 | Enable KPLS (kpls_dim ≈ d/2) |
| d > 50 | Enable CoEGO with cooperative groups |

### Evaluation Cost Guidelines

| Cost | DOE Size | Iterations |
|------|----------|------------|
| Cheap | Large (3×n_dims) | High (50+) |
| Expensive | Small (n_dims+1) | Moderate (20-30) |
| Expensive, active constraints | Small (n_dims+1) | High (100-200 for d ≈ 10) |

### Convergence Issues

If optimization stagnates:
1. Enable TREGO trust-region framework
2. Switch kernel from SquaredExponential to Matern52
3. Try different infill strategy (WB2, EI instead of LOG_EI)
4. Increase exploration via infill parameters

### Constraints

- Tolerances are absolute (default `1e-4`, or `CstrSpec.leq(b, tol=...)` per constraint): scale constraints to order 1
- `Egor(cstr_tol=...)` was removed in 1.0.0, give a `tol` to each `CstrSpec` instead
- Prefer a narrow band `CstrSpec.between(-eps, eps)` to `CstrSpec.eq(0.0)` for equality constraints
- Keep `cstr_infill=True`, in particular when no initial DOE point is feasible
- With `cstr_infill=True`, the criterion weighted by the probability of feasibility is used from the
  first iteration (default). `feasibility_first=True` searches with the probability of feasibility
  alone until a feasible point is found: it finds one sooner but improves the objective more slowly,
  so use it only when finding a feasible point is the hard part
- Use `InfillOptimizer.SLSQP` when constraints are expected to be active at the optimum
- With `cstr_infill=True` (or without constraints), `InfillOptimizer.LBFGSB` (gradient based,
  bounds only) can be used instead, unless `fcstrs` or `FailsafeStrategy.VIABILITY` are used
  (rejected with a `ValueError`)
- With `InfillStrategy.TS`, constraint surrogates are sampled (SCBO rule): keep `cstr_infill=False`
  (`ValueError` otherwise). When no candidate is feasible for a sample, the least total violation is
  selected, each constraint scaled by its largest predicted magnitude: order 1 constraints are still preferable

### Failures

- `FailsafeStrategy.REJECTION` is the safest choice when failures are rare
- With `FailsafeStrategy.IMPUTATION`, check that imputed values stay plausible (they can drift)
- The objective function must be free of side effects between calls (e.g. a solver restarting
  from the state of a failed previous evaluation)

### Parallel Execution

When parallel evaluations are available:
- Use `QEiConfig` with appropriate batch size
- Batch size ≈ dimension/10 is a good starting point
- Strategy `KB` works well for most cases
- In low dimension (nx ≲ 8), try `infill_strategy=egx.InfillStrategy.TS` with the same `QEiConfig(batch=...)`:
  batch points come from independent posterior samples, with one GP training per iteration

### Thompson Sampling (`InfillStrategy.TS`)

Use it:

- for batches in low dimension (nx ≲ 8), in particular on multimodal objectives
- with an initial DOE spread over the domain (e.g. `n_doe` ≈ 3 × nx): TS follows the GP belief, with
  a tiny DOE it keeps sampling the current basin (2 successes out of 10 seeds on xsinx with 3 points)

Avoid it:

- in high dimension: the candidate set (`min(100 * nx, 1000)` points) gets sparse
- when TREGO, CoEGO, clustered GPs (`n_clusters != 1`), `cstr_infill`, feasible infill strategies or
  `FailsafeStrategy.VIABILITY` are needed: they are rejected with TS (`ValueError`)

Notes:

- `QEiConfig.strategy` / `optim_every`, `infill_optimizer` and `infill_n_start` have no effect with TS
- Samples are drawn without replacement: each iteration adds `batch` distinct points

---

## Diagnosing a Run

Run with `verbose=egx.Verbose.INFO` and an `outdir`, then check in order:

1. **Feasible points**: count the points satisfying all constraints in `egor_doe.npy`.
   Zero or very few feasible points usually means badly scaled constraints or an equality
   constraint that is too strict.
2. **Failed points**: size of `egor_failed_points.npy`. A sudden run of consecutive failures
   points to a stateful objective function rather than to hard regions of the design space.
3. **Best point in the log** (`End iteration ... Best fun(x[i])=[obj, c1, ...]`): check that
   the objective is finite and plausible (a NaN or huge value comes from failed or imputed points).
4. **Infill criterion** (`... max found = ...`) followed by `Reject ... point too close to previous
   ones`: when every new point is rejected, Egor stops early with "Solver converged".

`egor_doe.npy` rows are `[x (nx), objective, internal constraints]`, where the internal
constraints are in `c <= 0` form after expansion of the `cstr_specs` (two columns for each
`eq` or `between` constraint).

---

## Examples

See [examples directory](examples/) for concrete use cases:
- `cheap_function.yaml` - Low-cost objective optimization
- `expensive_function.yaml` - High-cost objective optimization  
- `high_dimension.yaml` - Problems with d > 10
- `parallel.yaml` - Batch/parallel evaluation setup
- `bad_progress.yaml` - Stagnation recovery strategies
- `constrained_engineering.yaml` - Constrained engineering problem with active constraints
- `thompson_sampling.yaml` - Batch Thompson Sampling in low dimension

---

## API Reference

For complete parameter definitions, see the [Python API documentation](../../website/content/python-api.md).

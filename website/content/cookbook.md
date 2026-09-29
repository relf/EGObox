+++
title = "Cookbook"
weight = 40
+++

# Cookbook

This page contains practical Egor parameterization recipes. 

Each recipe includes:

- when to use it
- a suggested configuration
- why it helps

**Notes**

- Start from the closest recipe, then tune one group of parameters at a time.
- Keep random seeds fixed while comparing configurations, and confirm a difference with at least
  two seeds: the seed alone can change the final objective by about 1%.
- Use a distinct `outdir` for each run you compare: with `warm_start=True`, a run silently
  continues from the DOE already saved in its `outdir`.
- Use the `timeout` argument of `minimize()` to stop gracefully before a batch-job time limit.
- Regarding the required `xspecs` parameter (i.e. the input space) used by `Egor`: When using continuous variables only, the simplest form is a list of `[lower, upper]` pairs, e.g. `[[0.0, 1.0], [1.0, 10.0], [-5.0, 5.0]]`. Otherwise, see the [XSpecs in Python API](../python-api#xspecs) for more complex variable types.
- For complete parameter definitions, see the [Python API](../python-api).

## Recipe 1: Cheap Low-Dimensional Objective

Use when:

- dimension is low (for example 2 to 5)
- objective evaluation is cheap
- you can afford more iterations

Suggested setup:

```python
optim = egx.Egor(
    xspecs,
    n_doe=30,
)
res = optim.minimize(fun, max_iters=60, seed=42)
```

Why it helps:

- larger DOE improves global coverage early
- cheap evaluations allow more exploration iterations

## Recipe 2: Expensive Objective

Use when:

- each objective call is expensive (seconds to minutes)
- you need strong information gain at each iteration

Suggested setup:

```python
optim = egx.Egor(
    xspecs,
    n_doe=13,
)
res = optim.minimize(fun, max_iters=20, seed=42)
```

Why it helps:

- smaller DOE reduces initial expensive calls (default xdim + 1)

Note:

- 20 iterations suit unconstrained problems in low dimension. With several constraints active
  at the optimum, plan for many more iterations (see [Recipe 12](#recipe-12-constrained-engineering-problem-with-active-constraints),
  where about 100 to 200 iterations were needed in dimension 11).

## Recipe 3: High Dimension (d > 10)

Use when:

- the number of variables is high

Suggested setup:

```python
gp_cfg = egx.GpConfig(kpls_dim=10)

optim = egx.Egor(
    xspecs,
    gp_config=gp_cfg,
)
res = optim.minimize(fun, max_iters=40, seed=42)
```

Why it helps:

- KPLS reduces effective surrogate complexity
- improves robustness in high-dimensional GP fitting

Rule of thumb:

- around d=20, start with kpls_dim=5
- around d=100, start with kpls_dim=10
- just above d=10 (up to about 15), first try without KPLS: full GPs remain affordable and can
  model constraints much more accurately than their KPLS counterparts

## Recipe 4: Very High Dimension (d > 50)

Use when:

- standard optimization stalls in very high dimension

Suggested setup:

```python
gp_cfg = egx.GpConfig(kpls_dim=10)

optim = egx.Egor(
    xspecs,
    gp_config=gp_cfg,
    coego_n_coop=5,
)
res = optim.minimize(fun, max_iters=60, seed=42)
```

Why it helps:

- CoEGO decomposes search into cooperative component groups
- better scaling behavior for very high dimension

## Recipe 5: Parallel Evaluations Available

Use when:

- you can evaluate multiple points concurrently

Suggested setup:

```python
qei_cfg = egx.QEiConfig(batch=10, strategy=egx.QEiStrategy.KB, optmod=1)

optim = egx.Egor(
    xspecs,
    qei_config=qei_cfg,
)
res = optim.minimize(fun, max_iters=20, seed=42)
```

Why it helps:

- qEI proposes batches of points per iteration
- better wall-clock performance on parallel hardware

Rule of thumb:

- around d=50, try batch=5
- around d=100, try batch=10

## Recipe 6: Stagnation or Poor Progress

Use when:

- best objective value barely improves
- optimizer revisits similar regions

Suggested setup:

```python
gp_cfg = egx.GpConfig(corr_spec=egx.CorrelationSpec.MATERN52)

optim = egx.Egor(
    xspecs,
    gp_config=gp_cfg,
    trego=True,
    infill_strategy=egx.InfillStrategy.WB2,
    infill_optimizer=egx.InfillOptimizer.SLSQP
)
res = optim.minimize(fun, max_iters=40, seed=42)
```

Why it helps:

- TREGO alternates global and local trust-region behavior improves convergence
- Matern52 is often more robust on rougher landscapes
- On bad infill optimization you can try to change the infill optimizer; try `SLSQP`
- Default `LOG_EI` optimization on rough landscapes may be too difficult; try `WB2`, `WB2S` or even `EI` instead.
  The effect is problem dependent: on a constrained 11D problem, `WB2S` did much worse than `LOG_EI`
  (see [Recipe 12](#recipe-12-constrained-engineering-problem-with-active-constraints)), so compare
  with `LOG_EI` before switching.

## Recipe 7: Constraint-Heavy Problems

Use when:

- feasibility dominates the search difficulty

Suggested setup:

```python
optim = egx.Egor(
    xspecs,
    n_cstr=n_cstr,
    cstr_infill=True,
    cstr_strategy=egx.ConstraintStrategy.UTB,
    infill_strategy=egx.InfillStrategy.WB2,
    feasible_infill_strategy=egx.FeasibleInfillStrategy.EFI_FE,
)
res = optim.minimize(fun, max_iters=30, seed=42)
```

Why it helps:

- UTB makes constraint handling more conservative under uncertainty
- EFI_FE increases exploration of feasible regions

### Note

- EFI_FE is not implemented for default infill strategies (LOG_EI), so you
  need to use WB2, EI or WB2S. 

## Recipe 8: Objective May Crash

Use when:

- objective occasionally fails or returns `NaN`
- long runs may be interrupted and should be resumed safely

Suggested setup:

```python
optim = egx.Egor(
    xspecs,
    failsafe_strategy=egx.FailsafeStrategy.VIABILITY,
)

# First run and subsequent restarts use the same outdir.
# hot_start=0 means resume from latest checkpoint if available.
res = optim.minimize(
    fun,
    max_iters=60,
    outdir="run01",
    hot_start=0,
    seed=42,
)
```

Why it helps:

- `FailsafeStrategy.VIABILITY` models failure regions and steers search away from them
- `hot_start` with a stable `outdir` lets you continue from checkpoints instead of restarting from scratch

Alternatives:

- `FailsafeStrategy.REJECTION`: drops failed points (simplest, and the safest choice when failures are rare)
- `FailsafeStrategy.IMPUTATION`: fills failed outputs with surrogate-based estimates.
  Imputed values are fed back into the surrogates and may drift: check that the imputed
  objective values stay in a plausible range.

Note:

- Make `fun` free of side effects between calls. A simulation that restarts from the state left
  by the previous call can turn a single failure into a series of failures (every following
  evaluation starting from a NaN state).

## Recipe 9: Restart From an Existing DOE

Use when:

- you already have a DOE from a previous run
- you want to continue optimization without starting from scratch

Suggested setup:

```python
initial_doe = np.load("run01/egor_initial_doe.npy")

optim = egx.Egor(
    xspecs,
    doe=initial_doe,
)
res = optim.minimize(fun, max_iters=40, seed=42)
```

If the DOE was saved in an output directory from a previous run, you can also
let Egor reload it automatically:

```python
optim = egx.Egor(
    xspecs
)
res = optim.minimize(
    fun,
    max_iters=40,
    outdir="run01",
    warm_start=True,
    seed=42,
)
```

Why it helps:

- reuses already evaluated points instead of recomputing them
- keeps the surrogate and search history aligned with prior work
- makes long optimization runs easier to resume after interruptions

## Recipe 10: Constraints Not of the Form ≤ 0

Use when:

- your constraint functions return values that need different feasibility interpretations
- constraints are equality constraints (e.g., `g(x) = 0`)
- constraints have lower bounds (e.g., `g(x) ≥ bound`)
- constraints have interval bounds (e.g., `lower ≤ g(x) ≤ upper`)

Suggested setup:

```python
import egobox as egx

# Example: constraint must be >= 0 (instead of default <= 0)
optim = egx.Egor(
    xspecs,
    cstr_specs=[egx.CstrSpec.geq(0.0)],  # g(x) >= 0
)
res = optim.minimize(fun, max_iters=40, seed=42)

# Example: equality constraint g(x) = target
optim = egx.Egor(
    xspecs,
    cstr_specs=[egx.CstrSpec.eq(target_value)],
)
res = optim.minimize(fun, max_iters=40, seed=42)

# Example: constraint in interval [lower, upper]
optim = egx.Egor(
    xspecs,
    cstr_specs=[egx.CstrSpec.btw(lower_bound, upper_bound)],
)
res = optim.minimize(fun, max_iters=40, seed=42)
```

Why it helps:

- `cstr_specs` defines the semantics of each constraint beyond the default `≤ 0`
- `CstrSpec.leq(bound)`: constraint `g(x) ≤ bound`
- `CstrSpec.geq(bound)`: constraint `g(x) ≥ bound`
- `CstrSpec.eq(value)`: equality constraint `g(x) = value`
- `CstrSpec.btw(lower, upper)`: interval constraint `lower ≤ g(x) ≤ upper`

Note:

- The constraint function `fun` should return raw values; `cstr_specs` interprets feasibility
- For equality constraints, consider using a small tolerance via `cstr_tol`, or a narrow band
  `CstrSpec.btw(value - eps, value + eps)`, often easier for the optimizer to satisfy
- `CstrSpec.eq` and `CstrSpec.btw` each expand to two internal constraints: `cstr_tol` must have
  one entry per internal constraint
- Tolerances are absolute (default `1e-4`): scale constraints to order 1 so that the tolerance
  is meaningful

## Recipe 11: Cheap Not Metamodelized Constraint

Use when:

- you have constraints that are cheap to evaluate
- constraints depend only on input variables `x` (not on expensive simulations)
- you want to avoid the overhead of surrogate modeling for these constraints

Suggested setup:

```python
import egobox as egx

# Example: constraint g(x) <= 0 that is cheap to evaluate
def cheap_constraint(x):
    # Cheap computation based only on x
    return x[0] ** 2 + x[1] ** 2  # Circle constraint

optim = egx.Egor(
    xspecs,
)
res = optim.minimize(
    fun,
    fcstrs=[cheap_constraint],  # List of cheap constraint functions
    fcstr_specs=[egx.CstrSpec.leq(1.0)],  # Constraint semantics
    max_iters=40,
    seed=42,
)
```

Why it helps:

- `fcstrs` (function constraints) are evaluated directly, not surrogate-modeled
- Avoids the complexity of modeling cheap constraints with GPs
- Reduces optimization overhead when constraints are inexpensive
- `fcstr_specs` defines feasibility interpretation (same as `cstr_specs`)

Note:

- Use `fcstrs` for cheap constraints; use `cstr_specs` for expensive constraints that need surrogate modeling
- The `fun` callable should return `(objective, *constraint_values)` when using surrogate constraints, but for `fcstrs`, constraints are passed separately
- Multiple cheap constraints can be provided as a list to `fcstrs`

## Recipe 12: Constrained Engineering Problem With Active Constraints

Use when:

- the objective and constraints come from a coupled simulation (e.g. an aero-structural MDA)
- constraints are expressed in physical units (kg, Pa, N, ...)
- some constraints are equalities, or are expected to be active at the optimum
- the optimum is likely to lie on variable bounds

Suggested setup:

```python
import numpy as np
import egobox as egx

LW_REF = 100.0  # order of magnitude of the equality constraint (here in kg)
EPS = 1e-3      # accepted band on the scaled equality constraint (here 0.1 kg)

def fun(x):
    # obj, stress margins (>= 0, already of order 1), lift minus weight (in kg)
    obj, stress, lift_minus_weight = simulate(x)
    return np.hstack([obj, stress, lift_minus_weight / LW_REF])

optim = egx.Egor(
    xspecs,
    cstr_specs=[egx.CstrSpec.geq(0.0)] * 3 + [egx.CstrSpec.btw(-EPS, EPS)],
    gp_config=egx.GpConfig(corr_spec=egx.CorrelationSpec.MATERN52),
    infill_strategy=egx.InfillStrategy.LOG_EI,
    infill_optimizer=egx.InfillOptimizer.SLSQP,
    cstr_infill=True,
    trego=egx.TregoConfig(beta=0.8),
    failsafe_strategy=egx.FailsafeStrategy.REJECTION,
)
res = optim.minimize(fun, max_iters=200, outdir="run_s42", seed=42, timeout=3200.0)
```

Why it helps:

- The feasibility tolerance `cstr_tol` is **absolute** (default `1e-4` on every internal constraint):
  scaling each constraint to order 1 makes it meaningful. A constraint in kg with values around
  1000 is almost never considered satisfied.
- A narrow band `CstrSpec.btw(-eps, eps)` is much easier to satisfy than `CstrSpec.eq(0.0)`
  while keeping the equality precise enough in practice.
- The `SLSQP` infill optimizer follows active constraint boundaries more accurately than `COBYLA`.
- `cstr_infill=True` weights the infill criterion by the probability of feasibility, which drives
  the search towards the feasible region when the initial DoE contains no feasible point.
- `REJECTION` is the safest failsafe strategy when simulation failures are rare. In this case
  study, `VIABILITY` (modeling the failure region) did not reduce the number of failures,
  made iterations more expensive and gave a similar or worse objective.
- `timeout` stops the optimization gracefully before a batch-job time limit.

Case study (11 variables, 3 stress constraints, 1 lift = weight equality, 200 iterations, seed 42):

| Setting | Feasible points | Best feasible objective (reference 41.84) |
|---|---|---|
| unscaled equality constraint (in kg) | 0 / 124 | none |
| scaled, `CstrSpec.eq`, `COBYLA` | 5 / 186 | 37.21 |
| scaled, `CstrSpec.btw` band, `COBYLA` | 8 / 206 | 41.24 |
| as above with `WB2S` infill criterion | 8 / 207 | 32.25 |
| scaled, `CstrSpec.btw` band, `SLSQP` | 13 / 210 | 41.81 |

Across seeds, the last setup reached within 0.5% of the gradient-based reference in about
60 to 140 iterations, but not always within 0.1%. With `max_iters=200`, it ended within 0.07%
of the reference on the three seeds tested, for a run time of 1.1 to 1.7 times that of the
gradient-based optimizer (which needs the derivatives of the simulation).

Failure handling on the same problem (200 iterations):

| Seed | `REJECTION`: objective | failed points | `VIABILITY`: objective | failed points |
|---|---|---|---|---|
| 42 | 41.824 | 7 | 41.827 | 15 |
| 1 | 41.828 | 7 | 41.811 | 3 |
| 3 | 41.807 | 34 | 41.700 | 32 |

Note:

- Make `fun` free of side effects between calls: if the simulation restarts from the state
  of a previous call, one diverged (NaN) evaluation can make all the following ones fail.
- Compare configurations with at least two seeds: in this case study the seed alone changed
  the final objective by up to 1.3%.

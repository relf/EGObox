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
- Keep random seeds fixed while comparing configurations.
- For complete parameter definitions, see the [Python API](python-api).

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
    infill_strategy=egx.InfillStrategy.LOG_EI,
)
res = optim.minimize(fun, max_iters=20, seed=42)
```

Why it helps:

- smaller DOE reduces initial expensive calls
- LOG_EI and feasible infill favor informative evaluations

## Recipe 3: High Dimension (d > 10)

Use when:

- the number of variables is high

Suggested setup:

```python
gp_cfg = egx.GpConfig(kpls_dim=10)

optim = egx.Egor(
    xspecs,
    gp_config=gp_cfg,
    n_doe=0,
)
res = optim.minimize(fun, max_iters=40, seed=42)
```

Why it helps:

- KPLS reduces effective surrogate complexity
- improves robustness in high-dimensional GP fitting

Rule of thumb:

- around d=20, start with kpls_dim=5
- around d=100, start with kpls_dim=10

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
    infill_strategy=egx.InfillStrategy.LOG_EI,
)
res = optim.minimize(fun, max_iters=40, seed=42)
```

Why it helps:

- TREGO alternates global and local trust-region behavior
- Matern52 is often more robust on rougher landscapes

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

- `FailsafeStrategy.REJECTION`: drops failed points (simplest)
- `FailsafeStrategy.IMPUTATION`: fills failed outputs with surrogate-based estimates

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



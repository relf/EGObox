---
name: egobox
description: >
  Use this skill whenever the user is working with the EGObox / egobox Python library
  for Bayesian optimization, Gaussian processes, surrogate modeling, or efficient global
  optimization (EGO). Triggers on any mention of: egobox, Egor optimizer, Gpx surrogate,
  Bayesian optimization in Rust/Python, latin hypercube sampling with egobox, mixed-integer
  optimization with egobox, or requests to minimize expensive black-box functions using
  surrogate models. Also use when the user pastes egobox code or imports like
  `import egobox as egx`.
---

# EGObox Skill

EGObox (`egobox`) is a Rust-backed Python library for **Efficient Global Optimization** (EGO / Bayesian Optimization).
It provides two main Python-facing objects: `Egor` (optimizer) and `Gpx` (Gaussian process surrogate),
`Belfegor` being the multi-objective counterpart of `Egor` (experimental).

## Installation

```bash
pip install egobox
```

## Core Concepts

| Concept | Description |
|---|---|
| `Egor` | Bayesian optimizer — iteratively evaluates an expensive black-box function |
| `Belfegor` | Multi-objective Bayesian optimizer — approximates the Pareto front of several objectives |
| `Gpx` | Mixture of Gaussian Processes — surrogate model for regression/prediction |
| `XType` | Variable type enum for mixed-integer spaces |
| `CstrSpec` | Constraint specification — describes the form of each constraint |
| `InfillStrategy` | Criterion used to select next evaluation point |
| DOE | Design of Experiments — initial sampling of the space |

---

## 1. Egor Optimizer (Python API)

### Minimal Example (continuous)

```python
import numpy as np
import egobox as egx


def f_obj(x: np.ndarray) -> np.ndarray:
    return (x - 3.5) * np.sin((x - 3.5) / np.pi)


# Minimize in [0, 25] with 20 function evaluations
optim = egx.Egor([[0.0, 25.0]]).minimize(f_obj, max_iters=20, seed=42)
print(f"f={optim.result.y_opt} at x={optim.result.x_opt}")
```

> **API note (≥ 0.37.0):** `seed`, `outdir`, `warm_start`, `hot_start`, and `verbose` moved
> from the `Egor()` constructor to `minimize()`. `minimize()` now returns an `EgorOptim`
> object with `.result` (the `OptimResult`) and `.status`.
> Since 0.38.0, `Egor(seed=..., verbose=...)` can also give defaults, overridden by the `minimize()` / `suggest()` values.

### Constructor Signature

```python
egx.Egor(
    xspecs,  # list of [lo, hi] bounds (continuous) OR list of XSpec/XType for mixed-integer
    gp_config=None,  # GpConfig — GP kernel/regression options
    n_cstr=0,  # number of ≤ 0 constraints (use cstr_specs instead for other forms)
    cstr_specs=None,  # list of CstrSpec — use when constraints are not plain ≤ 0
    n_doe=0,  # initial DoE size (0 = auto: max(n_vars + 1, 5))
    x_doe=None,  # np.ndarray (n, n_dims) — provide your own initial DoE inputs
    y_doe=None,  # np.ndarray (n, 1 + n_cstr) — and optionally their known outputs
    infill_strategy=egx.InfillStrategy.LOG_EI,
    trego=None,  # egx.TregoConfig() to activate TREGO variant
    # ... other advanced options
)
```

### minimize() Signature

```python
optim = egor.minimize(
    fun,  # objective (+ constraint) function
    max_iters=20,  # iteration budget
    seed=None,  # int — for reproducibility
    outdir=None,  # str path — save intermediate results
    warm_start=False,  # resume from saved outdir doe
    timeout=None,  # float — stop after N seconds
    verbose=None,  # 0=ERROR … 4=TRACE, or egx.Verbose enum
)
```

- `fun(x: np.ndarray) -> np.ndarray` — x shape `(n_samples, n_dims)`, returns `(n_samples, 1 + n_cstr)`
- Returns `EgorOptim` with `.result` (`OptimResult`) and `.status`; result fields are forwarded
  (`optim.x_opt`, `optim.y_opt`, ...) and it unpacks as `x_opt, y_opt = egor.minimize(...)`

### Result Object

```python
optim.result.x_opt  # np.ndarray shape (n_dims,)          — best input found
optim.result.y_opt  # np.ndarray shape (1 + n_cstr,)      — fun(x_opt): [obj, c1, c2, ...]
optim.result.x_doe  # np.ndarray shape (n_evals, n_dims)  — all evaluated x
optim.result.y_doe  # np.ndarray shape (n_evals, 1+n_cstr)— all fun(x) values
```

---

## 2. Constraints

### Simple form: `n_cstr` (constraints already written as ≤ 0)

Use when every constraint is already expressed as `g(x) ≤ 0` (negative = feasible).
The function returns `[objective, c1, c2, ...]` column-wise.

```python
def f(x):
    obj = (x[:, [0]] - 3.5) * np.sin((x[:, [0]] - 3.5) / np.pi)
    cstr = x[:, [0]] - 10.0  # satisfied when x ≤ 10
    return np.hstack([obj, cstr])


optim = egx.Egor([[0.0, 25.0]], n_cstr=1).minimize(f, max_iters=20, seed=42)
```

### Flexible form: `cstr_specs` (any constraint form)

Use `cstr_specs` when constraints are not naturally expressed as `≤ 0`.
Pass a list of `CstrSpec` objects — one per constraint column returned by `fun`.
When `cstr_specs` is given, **`n_cstr` is inferred automatically** and should be omitted (or left at 0).

| Constructor | Meaning | Internal expansion |
|---|---|---|
| `egx.CstrSpec.leq(b)` | `c ≤ b` | 1 internal constraint: `c - b ≤ 0` |
| `egx.CstrSpec.geq(b)` | `c ≥ b` | 1 internal constraint: `b - c ≤ 0` |
| `egx.CstrSpec.eq(v)` | `c = v` | 2 internal constraints: `c - v ≤ 0` and `v - c ≤ 0` |
| `egx.CstrSpec.between(lo, hi)` | `lo ≤ c ≤ hi` | 2 internal constraints: `lo - c ≤ 0` and `c - hi ≤ 0` |

> **Tolerance per constraint**: give each spec its own absolute tolerance (default `1e-4`), e.g.
> `egx.CstrSpec.leq(b, tol=1e-3)` or `{"leq": b, "tol": 1e-3}`; it applies to all its internal constraints.
> `Egor(cstr_tol=[...])` is deprecated since 0.38.0: for plain `≤ 0` constraints use
> `cstr_specs=[egx.CstrSpec.leq(0.0, tol=1e-3)] * n` instead of `n_cstr=n, cstr_tol=[1e-3] * n`.

#### Example — inequality bounds (leq / geq)

```python
import numpy as np
import egobox as egx


def f(x):
    obj = (x[:, [0]] - 3.5) * np.sin((x[:, [0]] - 3.5) / np.pi)
    c1 = x[:, [0]]  # raw value — we want c1 ≤ 20
    c2 = x[:, [0]]  # raw value — we want c2 ≥ 5
    return np.hstack([obj, c1, c2])


optim = egx.Egor(
    [[0.0, 25.0]],
    cstr_specs=[egx.CstrSpec.leq(20.0), egx.CstrSpec.geq(5.0)],
).minimize(f, max_iters=20, seed=42)
print(optim.result.x_opt, optim.result.y_opt)
```

#### Example — equality constraint

```python
def f(x):
    obj = (x[:, [0]] - 3.5) ** 2
    c = x[:, [0]] * x[:, [1]]  # we want c = 10
    return np.hstack([obj, c])


optim = egx.Egor(
    [[0.0, 10.0], [0.0, 10.0]],
    cstr_specs=[egx.CstrSpec.eq(10.0)],  # expands to 2 internal constraints
).minimize(f, max_iters=30, seed=42)
```

#### Example — double-sided (between) constraint

```python
def f(x):
    obj = x[:, [0]] ** 2 + x[:, [1]] ** 2
    c = x[:, [0]] + x[:, [1]]  # we want 2 ≤ c ≤ 4
    return np.hstack([obj, c])


optim = egx.Egor(
    [[0.0, 5.0], [0.0, 5.0]],
    cstr_specs=[egx.CstrSpec.between(2.0, 4.0)],  # expands to 2 internal constraints
).minimize(f, max_iters=30, seed=42)
```

#### Example — mixed constraint types

```python
def f(x):
    obj = x[:, [0]] ** 2 + x[:, [1]] ** 2
    c1 = x[:, [0]] + x[:, [1]]  # want = 3  (equality)
    c2 = x[:, [0]] - x[:, [1]]  # want ≥ 0  (geq)
    return np.hstack([obj, c1, c2])


optim = egx.Egor(
    [[-5.0, 5.0], [-5.0, 5.0]],
    cstr_specs=[egx.CstrSpec.eq(3.0), egx.CstrSpec.geq(0.0)],
).minimize(f, max_iters=30, seed=42)
```

---

## 3. Mixed-Integer Optimization

Use a list of `XSpec` (or `XType`) objects as `xspecs` when any variable is discrete.

```python
import numpy as np
import egobox as egx

xspecs = [
    egx.XSpec(egx.XType.FLOAT, [0.0, 10.0]),  # continuous in [0, 10]
    egx.XSpec(egx.XType.INT, [0, 5]),  # integer in {0,1,2,3,4,5}
    egx.XSpec(egx.XType.ORD, [1.0, 2.5, 5.0]),  # ordinal — one of given values
    egx.XSpec(egx.XType.ENUM, [4]),  # categorical, 4 unordered levels
]


def f_mixed(x: np.ndarray) -> np.ndarray:
    return x[:, [0]] ** 2 + x[:, [1]]


optim = egx.Egor(xspecs).minimize(f_mixed, max_iters=30, seed=42)
```

**XType values:**

| XType | xlimits | Description |
|---|---|---|
| `XType.FLOAT` | `[lo, hi]` | Continuous variable |
| `XType.INT` | `[lo, hi]` | Integer variable |
| `XType.ORD` | `[v1, v2, ...]` | Ordered discrete — one of the listed values |
| `XType.ENUM` | `[n]` or `tags=[...]` | Unordered categorical with n levels |

---

## 4. Infill Strategies & Advanced Options

```python
egx.InfillStrategy.LOG_EI  # Log Expected Improvement (default), Ament et al. 2023
egx.InfillStrategy.EI  # Expected Improvement (classic)
egx.InfillStrategy.WB2  # Watson & Barnes 2nd criterion — balanced
egx.InfillStrategy.WB2S  # Scaled WB2
```

**TREGO variant** (trust-region, good for high-dimensional problems):

```python
optim = egx.Egor(
    [[0.0, 1.0]] * 10,
    trego=egx.TregoConfig(),
).minimize(f_obj, max_iters=50, seed=42)
```

**Warm restart** (continue from saved DOE):

```python
egx.Egor([[0.0, 25.0]]).minimize(f_obj, max_iters=10, outdir="./.run", seed=42)
egx.Egor([[0.0, 25.0]]).minimize(
    f_obj, max_iters=10, outdir="./.run", warm_start=True, seed=42
)
```

---

## 5. Belfegor Multi-objective Optimizer (Python API, experimental)

`Belfegor` takes the `Egor` options which apply to several objectives (no `trego`, `coego_n_coop`, `target`)
plus `n_obj` and `moo_config`. `fun(x)` returns `[obj_1, ..., obj_n_obj, cstr_1, ...]` columns, all objectives
being minimized.

```python
import numpy as np
import egobox as egx


# ZDT1 bi-objective function: Pareto front f2 = 1 - sqrt(f1) for x2 = 0
def zdt1(x: np.ndarray) -> np.ndarray:
    f1 = x[:, 0]
    g = 1.0 + 9.0 * x[:, 1]
    return np.column_stack([f1, g * (1.0 - np.sqrt(f1 / g))])


belfegor = egx.Belfegor([[0.0, 1.0], [0.0, 1.0]], n_obj=2, n_doe=10, seed=42)
res = belfegor.minimize(zdt1, max_iters=20)
print(f"Pareto front of {len(res.y_pareto)} points")  # res.x_pareto, res.y_pareto
print(f"Compromise point f={res.y_opt} at x={res.x_opt}")
```

- `moo_config=egx.MooConfig(strategy=..., batch=..., eim_aggregation=..., hv_stop=(tol, n_iters), rho=..., n_divisions=...)`
  (or a dict): `strategy` is `MooStrategy.EHVI` by default for 2–3 objectives, `MooStrategy.PAREGO` beyond;
  `MooStrategy.EIM`; `MooStrategy.QEHVI` for batches (`batch=2..4`, single-cluster GPs).
- `batch` (points per iteration) is a `MooConfig` option: `Belfegor` has no `qei_config`, batches of
  non-QEHVI strategies use the Kriging believer heuristic.
- `hv_stop=(1e-3, 5)` stops when the front hypervolume improves by less than 0.1 % over 5 iterations
  (`ExitStatus.SOLVER_CONVERGED`).
- `minimize()` returns `BelfegorOptim`: `.result` (`ParetoResult` with `x_pareto`, `y_pareto`, `x_opt`/`y_opt`
  compromise point, `x_doe`, `y_doe`) and `.status`; unpacks as `x_pareto, y_pareto = belfegor.minimize(...)`.
- Ask-and-tell: `belfegor.suggest(x_doe, y_doe)`; front of given data: `belfegor.pareto_result(x_doe, y_doe)`,
  `belfegor.pareto_indices(y_doe)`.

## 6. Gpx Surrogate Model (Python API)

`Gpx` is a mixture of Gaussian Processes (Kriging + MoE). Use it as a standalone surrogate.

```python
import numpy as np
import egobox as egx

xtrain = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
ytrain = np.array([[0.0], [1.0], [1.5], [0.9], [1.0]])

gpx = egx.Gpx.builder().fit(xtrain, ytrain)

xtest = np.linspace(0, 4, 50).reshape(-1, 1)
y_mean = gpx.predict(xtest)  # shape (50,)
y_var = gpx.predict_var(xtest)
```

Inputs are `(n, nx)` arrays; when `nx == 1` a 1D `(n,)` array is also accepted by `fit`, `predict*`,
`sample` and `update`. Outputs `y` may be `(n,)` or `(n, 1)`.

**Builder options:**

```python
gpx = egx.Gpx.builder(
    kpls_dim=3,  # PLS dimension reduction (recommended when n_dims >= 9)
    regr_spec=egx.RegressionSpec.CONSTANT | egx.RegressionSpec.LINEAR,
    corr_spec=egx.CorrelationSpec.MATERN52,
).fit(xtrain, ytrain)
```

**RegressionSpec flags:** `CONSTANT`, `LINEAR`, `QUADRATIC`, `ALL`
**CorrelationSpec flags:** `SQUARED_EXPONENTIAL`, `MATERN32`, `MATERN52`, `ALL`

**Save / Load:**

```python
gpx.save("model.json")
gpx_loaded = egx.Gpx.load("model.json")
```

---

## 7. Sampling (DOE)

```python
xlimits = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
lhs = egx.lhs(xlimits, n_samples=20, seed=42)  # Latin Hypercube, shape (20, 3)
ff = egx.full_factorial(xlimits, n_samples=27)  # Full factorial
rnd = egx.random(xlimits, n_samples=20, seed=42)  # Random
```

---

## Common Pitfalls

- **`f_obj` must handle batched inputs**: x shape is `(n_samples, n_dims)`, not `(n_dims,)`.
- **`n_cstr` vs `cstr_specs`**: use `n_cstr` for plain `≤ 0` constraints; use `cstr_specs` for all other forms. Don't set both at the same time.
- **Constraint tolerances**: give them per spec (`CstrSpec.leq(b, tol=...)`, also in `fcstr_specs`); `Egor(cstr_tol=...)` is deprecated since 0.38.0.
- **`y_opt` includes constraint values**: shape is `(1 + n_cstr,)` — first element is the objective.
- **`seed` goes in `minimize()`** (changed in v0.37.0). Since 0.38.0 `Egor(seed=...)` gives a default, and the `minimize()` / `suggest()` value wins. For `GpMix`, `seed` is a constructor argument.
- **Function constraints** (`minimize(fcstrs=...)`, cheap, not surrogate-modeled) are `g(x) <= 0` and given as `(g, grad_g)`, `{"fun": g, "jac": grad_g}` or `g(x, return_grad)`. Unlike scipy, no `"type"` key: use `fcstr_specs` for other bounds.
- **Two multistarts**: `Egor(infill_n_start=...)` is the infill criterion multistart, `GpConfig(theta_n_start=...)` the GP hyperparameters multistart
  (`theta_max_eval` its likelihood evaluations budget). The former `n_start` / `max_eval` names are deprecated since 0.38.0.
- **Best point of a DOE**: `egor.best_result(x_doe, y_doe)` / `egor.best_index(y_doe)` (`get_result` / `get_result_index` are deprecated since 0.38.0).
- **`cstr_infill` vs `feasible_infill_strategy`**: the former weights the criterion by the probability of feasibility of the `n_cstr` constraints, the latter (`EFI_P`, `EFI_FE`) by the probability of viability, i.e. of `fun` not failing. They are independent.
- **`xtypes` vs `xlimits`**: pass a flat list of `[lo, hi]` for continuous-only; use `XSpec` objects for mixed-integer.
- **Low `n_doe`**: default is `max(n_dims + 1, 5)`. For complex functions, use `n_doe = 3 * n_dims` or more.

---

## Further Resources

- Cookbook: Practical parameterization recipes - [website/content/cookbook.md](../../website/content/cookbook.md)
- Tuning guidance: Egor optimizer tuning heuristics - [egobox-egor-tuning/SKILL.md](../egobox-egor-tuning/SKILL.md)
- Surrogate modeling: Gaussian Process modeling with Egobox - [egobox-gpx/SKILL.md](../egobox-gpx/SKILL.md)
- GitHub: https://github.com/relf/EGObox
- Rust API docs: https://docs.rs/egobox-ego/latest/egobox_ego/
- Tutorial notebooks: https://github.com/relf/EGObox/tree/master/doc
- Paper (JOSS): https://doi.org/10.21105/joss.04737

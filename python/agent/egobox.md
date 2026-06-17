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
It provides two main Python-facing objects: `Egor` (optimizer) and `Gpx` (Gaussian process surrogate).

## Installation

```bash
pip install egobox
```

## Core Concepts

| Concept | Description |
|---|---|
| `Egor` | Bayesian optimizer — iteratively evaluates an expensive black-box function |
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

### Constructor Signature

```python
egx.Egor(
    xspecs,            # list of [lo, hi] bounds (continuous) OR list of XSpec/XType for mixed-integer
    gp_config=None,    # GpConfig — GP kernel/regression options
    n_cstr=0,          # number of ≤ 0 constraints (use cstr_specs instead for other forms)
    cstr_tol=None,     # list of tolerances, one per internal constraint (default 1e-4 each)
    cstr_specs=None,   # list of CstrSpec — use when constraints are not plain ≤ 0
    n_doe=0,           # initial DoE size (0 = auto: ~n_vars + 1)
    doe=None,          # np.ndarray — provide your own initial DoE
    infill_strategy=egx.InfillStrategy.WB2,
    trego=None,        # egx.TregoConfig() to activate TREGO variant
    # ... other advanced options
)
```

### minimize() Signature

```python
optim = egor.minimize(
    fun,               # objective (+ constraint) function
    max_iters=20,      # iteration budget
    seed=None,         # int — for reproducibility
    outdir=None,       # str path — save intermediate results
    warm_start=False,  # resume from saved outdir doe
    timeout=None,      # float — stop after N seconds
    verbose=None,      # 0=ERROR … 4=TRACE, or egx.Verbose enum
)
```

- `fun(x: np.ndarray) -> np.ndarray` — x shape `(n_samples, n_dims)`, returns `(n_samples, 1 + n_cstr)`
- Returns `EgorOptim` with `.result` (`OptimResult`) and `.status`

### Result Object

```python
optim.result.x_opt   # np.ndarray shape (1, n_dims)        — best input found
optim.result.y_opt   # np.ndarray shape (1, 1 + n_cstr)    — fun(x_opt): [obj, c1, c2, ...]
optim.result.x_hist  # np.ndarray shape (n_evals, n_dims)  — all evaluated x
optim.result.y_hist  # np.ndarray shape (n_evals, 1+n_cstr)— all fun(x) values
```

---

## 2. Constraints

### Simple form: `n_cstr` (constraints already written as ≤ 0)

Use when every constraint is already expressed as `g(x) ≤ 0` (negative = feasible).
The function returns `[objective, c1, c2, ...]` column-wise.

```python
def f(x):
    obj  = (x[:, [0]] - 3.5) * np.sin((x[:, [0]] - 3.5) / np.pi)
    cstr = x[:, [0]] - 10.0   # satisfied when x ≤ 10
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
| `egx.CstrSpec.btw(lo, hi)` | `lo ≤ c ≤ hi` | 2 internal constraints: `lo - c ≤ 0` and `c - hi ≤ 0` |

> **`eq` and `btw` expand to two internal constraints each.** If you also pass `cstr_tol`,
> its length must match the total number of *internal* constraints after expansion.

#### Example — inequality bounds (leq / geq)

```python
import numpy as np
import egobox as egx

def f(x):
    obj = (x[:, [0]] - 3.5) * np.sin((x[:, [0]] - 3.5) / np.pi)
    c1  = x[:, [0]]   # raw value — we want c1 ≤ 20
    c2  = x[:, [0]]   # raw value — we want c2 ≥ 5
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
    c   = x[:, [0]] * x[:, [1]]   # we want c = 10
    return np.hstack([obj, c])

optim = egx.Egor(
    [[0.0, 10.0], [0.0, 10.0]],
    cstr_specs=[egx.CstrSpec.eq(10.0)],   # expands to 2 internal constraints
).minimize(f, max_iters=30, seed=42)
```

#### Example — double-sided (between) constraint

```python
def f(x):
    obj = x[:, [0]] ** 2 + x[:, [1]] ** 2
    c   = x[:, [0]] + x[:, [1]]   # we want 2 ≤ c ≤ 4
    return np.hstack([obj, c])

optim = egx.Egor(
    [[0.0, 5.0], [0.0, 5.0]],
    cstr_specs=[egx.CstrSpec.btw(2.0, 4.0)],   # expands to 2 internal constraints
).minimize(f, max_iters=30, seed=42)
```

#### Example — mixed constraint types

```python
def f(x):
    obj = x[:, [0]] ** 2 + x[:, [1]] ** 2
    c1  = x[:, [0]] + x[:, [1]]   # want = 3  (equality)
    c2  = x[:, [0]] - x[:, [1]]   # want ≥ 0  (geq)
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
    egx.XSpec(egx.XType.FLOAT, [0.0, 10.0]),   # continuous in [0, 10]
    egx.XSpec(egx.XType.INT,   [0, 5]),          # integer in {0,1,2,3,4,5}
    egx.XSpec(egx.XType.ORD,   [1.0, 2.5, 5.0]),# ordinal — one of given values
    egx.XSpec(egx.XType.ENUM,  [4]),             # categorical, 4 unordered levels
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

> **Shorthand constructors** (convenience aliases):
> `egx.XType.Float(lo, hi)`, `egx.XType.Int(lo, hi)`, `egx.XType.Ord([...])`, `egx.XType.Enum(n)`

---

## 4. Infill Strategies & Advanced Options

```python
egx.InfillStrategy.WB2     # Watson & Barnes 2 (default) — balanced
egx.InfillStrategy.WB2S    # Scaled WB2
egx.InfillStrategy.EI      # Expected Improvement (classic)
egx.InfillStrategy.LOG_EI  # Log Expected Improvement
```

**TREGO variant** (trust-region, good for high-dimensional problems):

```python
optim = egx.Egor(
    [[0., 1.]] * 10,
    trego=egx.TregoConfig(),
).minimize(f_obj, max_iters=50, seed=42)
```

**Warm restart** (continue from saved DOE):

```python
egx.Egor([[0., 25.]]).minimize(f_obj, max_iters=10, outdir="./.run", seed=42)
egx.Egor([[0., 25.]]).minimize(f_obj, max_iters=10, outdir="./.run", warm_start=True, seed=42)
```

---

## 5. Gpx Surrogate Model (Python API)

`Gpx` is a mixture of Gaussian Processes (Kriging + MoE). Use it as a standalone surrogate.

```python
import numpy as np
import egobox as egx

xtrain = np.array([[0.], [1.], [2.], [3.], [4.]])
ytrain = np.array([[0.], [1.], [1.5], [0.9], [1.0]])

gpx = egx.Gpx.builder().fit(xtrain, ytrain)

xtest = np.linspace(0, 4, 50).reshape(-1, 1)
y_mean = gpx.predict(xtest)
y_var  = gpx.predict_var(xtest)
```

**Builder options:**

```python
gpx = (
    egx.Gpx.builder()
    .kpls(3)             # PLS dimension reduction (recommended when n_dims >= 9)
    .regression_spec(egx.RegressionSpec.CONSTANT | egx.RegressionSpec.LINEAR)
    .correlation_spec(egx.CorrelationSpec.MATERN52)
    .fit(xtrain, ytrain)
)
```

**RegressionSpec flags:** `CONSTANT`, `LINEAR`, `QUADRATIC`, `ALL`
**CorrelationSpec flags:** `SQUARED_EXPONENTIAL`, `MATERN32`, `MATERN52`, `ALL`

**Save / Load:**

```python
gpx.save("model.json")
gpx_loaded = egx.Gpx.load("model.json")
```

---

## 6. Sampling (DOE)

```python
xlimits = np.array([[0., 1.], [0., 1.], [0., 1.]])
lhs = egx.lhs(xlimits, n_samples=20, seed=42)    # Latin Hypercube, shape (20, 3)
ff  = egx.full_factorial(xlimits, n_samples=27)   # Full factorial
rnd = egx.random(xlimits, n_samples=20, seed=42)  # Random
```

---

## Common Pitfalls

- **`f_obj` must handle batched inputs**: x shape is `(n_samples, n_dims)`, not `(n_dims,)`.
- **`n_cstr` vs `cstr_specs`**: use `n_cstr` for plain `≤ 0` constraints; use `cstr_specs` for all other forms. Don't set both at the same time.
- **`eq` / `btw` expand to 2 internal constraints each**: if you pass `cstr_tol`, size it to the total expanded count.
- **`y_opt` includes constraint values**: shape is `(1, 1 + n_cstr)` — first column is the objective.
- **`seed` belongs in `minimize()`**, not in `Egor()` (changed in v0.37.0).
- **`xtypes` vs `xlimits`**: pass a flat list of `[lo, hi]` for continuous-only; use `XSpec` objects for mixed-integer.
- **Low `n_doe`**: default is ~`n_dims + 1`. For complex functions, use `n_doe = 3 * n_dims` or more.

---

## Further Resources

- GitHub: https://github.com/relf/EGObox
- Rust API docs: https://docs.rs/egobox-ego/latest/egobox_ego/
- Tutorial notebooks: https://github.com/relf/EGObox/tree/master/doc
- Paper (JOSS): https://doi.org/10.21105/joss.04737

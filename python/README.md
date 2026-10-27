# EGObox - Efficient Global Optimization toolbox

![pytests](https://github.com/relf/egobox/actions/workflows/pytest.yml/badge.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

`egobox` package is the Python binding of the optimizer named `Egor` (with its multi-objective counterpart `Belfegor` and its multidisciplinary counterpart `Mdegor`) and the surrogate model `Gpx`, mixture of Gaussian processes, from the [EGObox libraries](https://github.com/relf/egobox?tab=readme-ov-file#egobox---efficient-global-optimization-toolbox) written in Rust.

## Installation

```bash
pip install egobox
```

This installation also provides the `gpx` command-line interface:

```bash
gpx --help
```

### Egor optimizer

```python
import numpy as np
import egobox as egx


# Objective function
def f_obj(x: np.ndarray) -> np.ndarray:
    return (x - 3.5) * np.sin((x - 3.5) / (np.pi))


# Minimize f_opt in [0, 25]
optim = egx.Egor([[0.0, 25.0]]).minimize(f_obj, max_iters=20, seed=42)
print(
    f"Optimization f={optim.result.y_opt} at {optim.result.x_opt}"
)  # Optimization f=[-15.12510323] at [18.93525454]
print(
    f"Status {optim.status.exit} in {optim.status.elapsed_time}s"
)  # ExitStatus.SOLVER_CONVERGED in 0.021s
```

### Belfegor multi-objective optimizer (experimental)

`Belfegor` approximates the Pareto front of several objectives, all minimized, the function returning
`[obj_1, ..., obj_n_obj, cstr_1, ...]` columns. It shares the `Egor` options which apply to several
objectives, the multi-objective strategy being set with `MooConfig` (EHVI by default for 2 or 3 objectives,
ParEGO beyond, EIM and qEHVI for batches of points set with `MooConfig(batch=...)`).

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

### Mdegor multidisciplinary design optimizer (experimental)

`Mdegor` minimizes an objective, subject to constraints, computed from coupling variables of
expensive disciplines which depend on each other (EGMDO and C-EGMDO algorithms). Each discipline is
replaced by Gaussian process surrogates enriched only where the optimum is likely to be, which saves
many disciplinary evaluations. The design and coupling variables are declared once by name, and the
disciplines are wired to them by name; `qoi(z, y)` returns the cheap objective and constraints
(`g <= 0`) from the design variables `z` and the couplings `y`, columns in declaration order.
An ask/tell interface (`ask()`, `tell()`) lets the caller run the disciplines.

```python
import numpy as np
import egobox as egx

# Sellar problem: y1 = z1² + z2 + z3 - 0.2 y2, y2 = sqrt(y1) + z1 + z2
design = {"z1": [-10.0, 10.0], "z2": [0.0, 10.0], "z3": [0.0, 10.0]}
couplings = {"y1": [0.0, 25.0], "y2": [0.0, 25.0]}  # initial ranges


def f1(x: np.ndarray) -> np.ndarray:  # x columns: z1, z2, z3, y2
    return (x[:, 0] ** 2 + x[:, 1] + x[:, 2] - 0.2 * x[:, 3])[:, None]


def f2(x: np.ndarray) -> np.ndarray:  # x columns: z1, z2, y1
    return (np.sqrt(np.maximum(x[:, 2], 0)) + x[:, 0] + x[:, 1])[:, None]


def qoi(z: np.ndarray, y: np.ndarray) -> np.ndarray:  # z: z1, z2, z3; y: y1, y2
    f = z[:, 2] ** 2 + z[:, 1] + y[:, 0] + np.exp(-y[:, 1])
    return np.column_stack([f, 3.16 - y[:, 0], y[:, 1] - 24.0])  # [f, g1, g2]


d1 = egx.Discipline("d1", f1, inputs=["z1", "z2", "z3", "y2"], outputs=["y1"])
d2 = egx.Discipline("d2", f2, inputs=["z1", "z2", "y1"], outputs=["y2"])
mdegor = egx.Mdegor(design, couplings, [d1, d2], qoi, n_ineq=2, seed=0)
res = mdegor.minimize(max_iters=10)
print(
    f"Best point {dict(zip(res.design_names, res.z_best))}, f={res.f_best}, "
    f"with {res.n_evaluations} disciplinary evaluations"
)
```

### Gpx surrogate model

```python
import numpy as np
import matplotlib.pyplot as plt
import egobox as egx

# Training
xtrain = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
ytrain = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
gpx = egx.Gpx.builder().fit(xtrain, ytrain)

# Prediction
xtest = np.linspace(0, 4, 100).reshape((-1, 1))
ytest = gpx.predict(xtest)

# Plot
plt.plot(xtest, ytest)
plt.plot(xtrain, ytrain, "o")
plt.show()
```

See the [tutorial notebooks](https://github.com/relf/egobox/tree/master/notebooks/README.md) and [examples folder](https://github.com/relf/egobox/tree/master/python/examples) for more information on the usage of the optimizer and mixture of Gaussian processes surrogate model.
In particular, `examples/belfegor_pymoo.py` runs `Belfegor` on [pymoo](https://pymoo.org) multi-objective test problems
(`pip install pymoo`, then `python belfegor_pymoo.py --list` to list them and `python belfegor_pymoo.py zdt1 --n-var 5`
to approximate the ZDT1 Pareto front, plotted with quality metrics such as hypervolume and IGD).

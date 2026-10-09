# EGObox - Efficient Global Optimization toolbox

![pytests](https://github.com/relf/egobox/actions/workflows/pytest.yml/badge.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

`egobox` package is the Python binding of the optimizer named `Egor` (and its multi-objective counterpart `Belfegor`) and the surrogate model `Gpx`, mixture of Gaussian processes, from the [EGObox libraries](https://github.com/relf/egobox?tab=readme-ov-file#egobox---efficient-global-optimization-toolbox) written in Rust.

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

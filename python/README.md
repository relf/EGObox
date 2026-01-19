# EGObox - Efficient Global Optimization toolbox

![pytests](https://github.com/relf/egobox/actions/workflows/pytest.yml/badge.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04737/status.svg)](https://doi.org/10.21105/joss.04737)

`egobox` package is the Python binding of the optimizer named `Egor` and the surrogate model `Gpx`, mixture of Gaussian processes, from the [EGObox libraries](https://github.com/relf/egobox?tab=readme-ov-file#egobox---efficient-global-optimization-toolbox) written in Rust.

## Installation

```bash
pip install egobox
```

### Egor optimizer

```python
import numpy as np
import egobox as egx

# Objective function
def f_obj(x: np.ndarray) -> np.ndarray:
    return (x - 3.5) * np.sin((x - 3.5) / (np.pi))

# Minimize f_opt in [0, 25]
res = egx.Egor([[0.0, 25.0]], seed=42).minimize(f_obj, max_iters=20)
print(f"Optimization f={res.y_opt} at {res.x_opt}")  # Optimization f=[-15.12510323] at [18.93525454]
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

See the [tutorial notebooks](https://github.com/relf/egobox/tree/master/doc/README.md) and [examples folder](https://github.com/relf/egobox/tree/d9db0248199558f23d966796737d7ffa8f5de589/python/egobox/examples) for more information on the usage of the optimizer and mixture of Gaussian processes surrogate model.

+++
title = "Python API"
weight = 50
+++

# Python API

This page summarizes the Python API signatures from `python/egobox/egobox.pyi` for:

- `Egor` constructor and `Egor.minimize(...)`
- `Belfegor` constructor and `Belfegor.minimize(...)` (multi-objective)
- `Gpx.builder(...)`
- `Gpx` instance and static methods

Default values are copied from the stub.

## Egor constructor

Signature:

```python
Egor(
    xspecs,
    gp_config=GpConfig(),
    n_cstr=0,
    cstr_tol=None,  # deprecated since 0.38.0
    cstr_specs=None,
    infill_n_start=20,
    n_doe=0,
    x_doe=None,
    y_doe=None,
    infill_strategy=InfillStrategy.LOG_EI,
    feasible_infill_strategy=FeasibleInfillStrategy.NONE,
    cstr_infill=False,
    cstr_strategy=ConstraintStrategy.MC,
    qei_config=QEiConfig(),
    infill_optimizer=InfillOptimizer.COBYLA,
    trego=None,
    coego_n_coop=0,
    target=-1.7976931348623157e+308,
    failsafe_strategy=FailsafeStrategy.REJECTION,
    seed=None,
    verbose=None,
)
```

| Name | Type | Default value | Description |
| --- | --- | --- | --- |
| `xspecs` | `[XSpec]` | required | Input variable specifications (`XSpec` list-like), one per dimension. The simplest form is [[lower1, upper1], [lower2, upper2], ...] which supposes a continuous range for each dimension. Otherwise see [XSpecs](#xspecs) for more details. |
| `gp_config` | `GpConfig` | `GpConfig()` | GP configuration used by the optimizer. |
| `n_cstr` | `int` | `0` | Number of surrogate-modeled constraints returned by `fun`. |
| `cstr_tol` | `Optional[Sequence[float]]` | `None` | Deprecated since 0.38.0: give a `tol` to each spec instead, e.g. `cstr_specs=[CstrSpec.leq(0.0, tol=1e-3)]` (and `fcstr_specs` for function constraints). |
| `cstr_specs` | `Optional[Sequence[CstrSpec]]` | `None` | Optional constraint semantics for surrogate constraints. Possible values per item: `CstrSpec.leq(bound)`, `CstrSpec.geq(bound)`, `CstrSpec.eq(value)`, `CstrSpec.between(lower, upper)`, each with an optional `tol`, the absolute feasibility tolerance (default `1e-4`). When given, `n_cstr` is inferred. |
| `infill_n_start` | `int` | `20` | Number of multistart runs for infill optimization (`n_start` is deprecated since 0.38.0). |
| `n_doe` | `int` | `0` | Initial DOE size (auto-computed when `0`, if DOE not provided). |
| `x_doe` | `Optional[NDArray[float64]]` | `None` | Initial DOE inputs `(ns, nx)`, evaluated when `y_doe` is not given (`doe` is deprecated since 0.38.0). |
| `y_doe` | `Optional[NDArray[float64]]` | `None` | Initial DOE outputs `(ns, 1 + n_cstr)` corresponding to `x_doe`. |
| `infill_strategy` | `InfillStrategy` | `InfillStrategy.LOG_EI` | Infill criterion. Possible values: `InfillStrategy.LOG_EI`, `InfillStrategy.EI`, `InfillStrategy.WB2`, `InfillStrategy.WB2S`. |
| `feasible_infill_strategy` | `FeasibleInfillStrategy` | `FeasibleInfillStrategy.NONE` | Feasibility-aware infill mode. Possible values: `FeasibleInfillStrategy.NONE`, `FeasibleInfillStrategy.EFI_P` (alias `EFI_PROBABILITY`), `FeasibleInfillStrategy.EFI_FE` (alias `EFI_FEASIBILITY_ENHANCED`). |
| `cstr_infill` | `bool` | `False` | Enables constrained infill with probability-of-feasibility factor. Possible values: `True`, `False`. |
| `cstr_strategy` | `ConstraintStrategy` | `ConstraintStrategy.MC` | Constraint handling strategy. Possible values: `ConstraintStrategy.MC` (alias `MEAN_CONSTRAINT`), `ConstraintStrategy.UTB` (alias `UPPER_TRUST_BOUND`). |
| `qei_config` | `QEiConfig` | `QEiConfig()` | Parallel/qEI batch configuration. See [QEiConfig](#qeiconfig) for details. |
| `infill_optimizer` | `InfillOptimizer` | `InfillOptimizer.COBYLA` | Internal optimizer for infill criterion. Possible values: `InfillOptimizer.COBYLA`, `InfillOptimizer.SLSQP`. |
| `trego` | `Optional[Any]` | `None` | TREGO toggle/config. Possible values: `None`, `False`, `True` (uses `TregoConfig()` defaults), `TregoConfig(...)`, or a dict with keys `n_global_local_steps`, `radius_bounds`, `alpha`, `beta`, `sigma0`. |
| `coego_n_coop` | `int` | `0` | Number of cooperative groups for CoEGO (high-dimensional mode). |
| `target` | `float` | `-1.7976931348623157e+308` | Known optimum target used as stopping criterion. |
| `failsafe_strategy` | `FailsafeStrategy` | `FailsafeStrategy.REJECTION` | Failure handling for NaN objective values. Possible values: `FailsafeStrategy.REJECTION`, `FailsafeStrategy.IMPUTATION`, `FailsafeStrategy.VIABILITY`. |
| `seed` | `Optional[int]` | `None` | Default RNG seed of `minimize()` and `suggest()`, overridden by their own `seed`. |
| `verbose` | `Optional[Any]` | `None` | Default logging verbosity of `minimize()` and `suggest()`, overridden by the `minimize()` `verbose`. |

### GpConfig

| Name | Type | Default value | Description |
| --- | --- | --- | --- |
| `regr_spec` | `int` | `RegressionSpec.CONSTANT` (`1`) | Regression basis bitflags. Possible flag values: `RegressionSpec.CONSTANT` (`1`), `RegressionSpec.LINEAR` (`2`), `RegressionSpec.QUADRATIC` (`4`), bitwise combinations allowed. |
| `corr_spec` | `int` | `CorrelationSpec.SQUARED_EXPONENTIAL` (`1`) | Correlation kernel bitflags. Possible flag values: `CorrelationSpec.SQUARED_EXPONENTIAL` (`1`), `CorrelationSpec.ABSOLUTE_EXPONENTIAL` (`2`), `CorrelationSpec.MATERN32` (`4`), `CorrelationSpec.MATERN52` (`8`), bitwise combinations allowed. |
| `kpls_dim` | `Optional[int]` | `None` | No PLS projection by default. |
| `n_clusters` | `int` | `1` | Single expert/cluster by default. |
| `recombination` | `Recombination` | `Recombination.HARD` | Expert recombination mode. Possible values: `Recombination.HARD`, `Recombination.SMOOTH`. |
| `theta_init` | `Optional[Sequence[float]]` | `None` | Uses internal default initialization. |
| `theta_bounds` | `Optional[Sequence[Sequence[float]]]` | `None` | Uses internal default bounds. |
| `theta_n_start` | `int` | `10` | GP hyperparameter optimization multistart (`n_start` is deprecated since 0.38.0). |
| `theta_max_eval` | `int` | `50` | Max likelihood evaluations for hyperparameter optimization (`max_eval` is deprecated since 0.38.0). |

### QEiConfig

| Name | Type | Default value | Description |
| --- | --- | --- | --- |
| `batch` | `int` | `1` | One point per iteration (sequential EGO). |
| `strategy` | `QEiStrategy` | `QEiStrategy.KB` | qEI strategy. Possible values: `QEiStrategy.KB`, `QEiStrategy.KBLB`, `QEiStrategy.KBUB`, `QEiStrategy.CLMIN` (long name aliases `KRIGING_BELIEVER`, `KRIGING_BELIEVER_LOWER_BOUND`, `KRIGING_BELIEVER_UPPER_BOUND`, `CONSTANT_LIAR_MINIMUM`). |
| `optim_every` | `int` | `1` | Re-optimize hyperparameters every point (`optmod` is deprecated since 0.38.0). |

### TregoConfig

When `trego=True` in `Egor(...)`, the Python binding activates TREGO with `TregoConfig()` defaults.

| Name | Type | Default value | Description |
| --- | --- | --- | --- |
| `n_global_local_steps` | `tuple[int, int]` | `(1, 4)` | Number of global and local steps `(n_global_steps, n_local_steps)` (`n_gl_steps` is deprecated since 0.38.0). |
| `radius_bounds` | `tuple[float, float]` | `(1e-6, 1.0)` | Trust-region radius bounds `(dmin, dmax)` (`d` is deprecated since 0.38.0). |
| `alpha` | `float` | `1.0` | Acceptance threshold coefficient in `rho(sigma) = alpha * sigma^2`. |
| `beta` | `float` | `0.9` | Trust-region contraction factor. |
| `sigma0` | `float` | `0.1` | Initial trust-region radius. |

## Egor.minimize

Signature:

```python
Egor.minimize(
    fun,
    fcstrs=[],
    fcstr_specs=[],
    max_iters=20,
    run_info=None,
    outdir=None,
    warm_start=False,
    hot_start=None,
    seed=None,
    timeout=None,
    verbose=None,
)
```

| Name | Type | Default value | Description |
| --- | --- | --- | --- |
| `fun` | `typing.Any` | required | Objective/constraint callable evaluated by the optimizer. fun: callable with signature `fun(x: np.ndarray) -> np.ndarray` where `x` is a 2D array of shape `(n_points, n_variables)` and return value is a 2D array with shape `(n_points, 1 + n_constraints)` containing [obj, cstr1, cstr2, ...] evaluations. |
| `fcstrs` | `Sequence[typing.Any]` | `[]` | Optional function constraints `g(x) <= 0` (cheap constraints, not surrogate-modeled). Possible values per item: `(g, grad_g)`, `{"fun": g, "jac": grad_g}`, or `g(x, return_grad)`. The gradient is only used with `InfillOptimizer.SLSQP`. |
| `fcstr_specs` | `Sequence[CstrSpec]` | `[]` | Optional semantics for `fcstrs` constraints. Possible values per item: `CstrSpec.leq(bound)`, `CstrSpec.geq(bound)`, `CstrSpec.eq(value)`, `CstrSpec.between(lower, upper)`. |
| `max_iters` | `int` | `20` | Iteration budget. |
| `run_info` | `Optional[Any]` | `None` | Optional run metadata (`RunInfo`) for checkpoint naming/tracking. |
| `outdir` | `Optional[str]` | `None` | Directory for history/checkpoint artifacts and warm start lookup. |
| `warm_start` | `bool` | `False` | Loads initial DOE from `outdir`. Possible values: `True`, `False`. |
| `hot_start` | `Optional[Any]` | `None` | Resume from checkpoint and optionally extend iteration budget. Possible values: `None`, `True` (interpreted as 0), or non-negative integer. |
| `seed` | `Optional[int]` | `None` | RNG seed for reproducibility. Defaults to the `Egor(seed=...)` value. |
| `timeout` | `Optional[float]` | `None` | Optional time limit in seconds. |
| `verbose` | `Optional[Any]` | `None` | Logging verbosity. Possible values: `None`, integer level, or `Verbose` enum (`ERROR`, `WARNING`, `INFO`, `DEBUG`, `TRACE`). Defaults to the `Egor(verbose=...)` value. |

It returns an `EgorOptim` with `result` (an `OptimResult` holding `x_opt`, `y_opt`, `x_doe`, `y_doe`) and `status`.
The result fields are also available directly on the returned object, which can be unpacked as `(x_opt, y_opt)`:

```python
res = egor.minimize(fun, max_iters=20)
res.x_opt, res.y_opt  # same as res.result.x_opt, res.result.y_opt
x_opt, y_opt = egor.minimize(fun, max_iters=20)
```

`Egor.best_result(x_doe, y_doe)` and `Egor.best_index(y_doe)` give the best point of a given DOE
(`get_result` and `get_result_index` are deprecated since 0.38.0).

## Belfegor (multi-objective, experimental)

`Belfegor` approximates the Pareto front of several objectives, all minimized: `fun(x)` returns
`[obj_1, ..., obj_n_obj, cstr_1, ... cstr_k]` columns (`ny = n_obj + n_cstr`). It shares the `Egor`
options which apply to several objectives (`trego`, `coego_n_coop` and `target` are not available).

Signature:

```python
Belfegor(
    xspecs,
    n_obj=2,
    moo_config=None,
    gp_config=None,
    n_cstr=0,
    cstr_specs=None,
    infill_n_start=None,
    n_doe=0,
    x_doe=None,
    y_doe=None,
    infill_strategy=InfillStrategy.LOG_EI,
    feasible_infill_strategy=FeasibleInfillStrategy.NONE,
    cstr_infill=False,
    cstr_strategy=ConstraintStrategy.MC,
    qei_config=None,
    infill_optimizer=InfillOptimizer.COBYLA,
    failsafe_strategy=FailsafeStrategy.REJECTION,
    seed=None,
    verbose=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_obj` | `int` | `2` | Number of objectives, the first columns returned by `fun`. |
| `moo_config` | `Optional[MooConfig \| dict]` | `None` | Multi-objective configuration, see `MooConfig` below. |

The other parameters are the `Egor` ones. `infill_strategy` and `feasible_infill_strategy` apply to the
scalarized objective of `MooStrategy.PAREGO`. With `MooStrategy.QEHVI`, the batch points
(`qei_config=QEiConfig(batch=...)`, at most 4) are selected by qEHVI.

### MooConfig

```python
MooConfig(
    strategy=None,
    eim_aggregation=EimAggregation.EUCLIDEAN,
    hv_stop=None,
    rho=0.05,
    n_divisions=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | `Optional[MooStrategy]` | `None` | `PAREGO` (scalarization, single surrogate), `EIM` (expected improvement matrix), `EHVI` (expected hypervolume improvement, at most 8 objectives) or `QEHVI` (batch EHVI, single-cluster surrogates). `None` gives `EHVI` for 2 or 3 objectives, `PAREGO` beyond. |
| `eim_aggregation` | `EimAggregation` | `EUCLIDEAN` | Aggregation used by `EIM`: `EUCLIDEAN`, `MAXIMIN` or `HYPERVOLUME`. |
| `hv_stop` | `Optional[tuple[float, int]]` | `None` | `(tol, n_iters)`: stop when the hypervolume of the feasible front improves by less than `tol` (relative) over the last `n_iters` iterations (`ExitStatus.SOLVER_CONVERGED`). |
| `rho` | `float` | `0.05` | ParEGO augmented Tchebycheff coefficient. |
| `n_divisions` | `Optional[int]` | `None` | ParEGO number of divisions of the weight simplex lattice (10 for 2 objectives, 4 for 3, 3 beyond). |

A dict with the same keys is also accepted.

### Belfegor.minimize

`Belfegor.minimize(fun, fcstrs=None, fcstr_specs=None, max_iters=20, run_info=None, outdir=None, warm_start=False, hot_start=None, seed=None, timeout=None, verbose=None, stop_on_error=False)`
takes the `Egor.minimize` parameters and returns a `BelfegorOptim` with `result` (a `ParetoResult` holding
`x_pareto`, `y_pareto`, `x_opt`, `y_opt`, `x_doe`, `y_doe`) and `status`. `x_pareto`/`y_pareto` is the
(constrained) Pareto front approximation and `x_opt`/`y_opt` its compromise point. The result fields are
also available directly on the returned object, which can be unpacked as `(x_pareto, y_pareto)`:

```python
res = belfegor.minimize(fun, max_iters=20)
res.x_pareto, res.y_pareto  # same as res.result.x_pareto, res.result.y_pareto
x_pareto, y_pareto = belfegor.minimize(fun, max_iters=20)
```

`Belfegor.suggest(x_doe, y_doe)` gives the next points to evaluate (ask-and-tell), and
`Belfegor.pareto_result(x_doe, y_doe)` / `Belfegor.pareto_indices(y_doe)` give the Pareto front of a given DOE.

## Gpx.builder

Signature:

```python
Gpx.builder(
    xspecs=None,
    regr_spec=1,
    corr_spec=1,
    kpls_dim=None,
    n_clusters=1,
    recombination=Recombination.HARD,
    theta_init=None,
    theta_bounds=None,
    theta_n_start=10,
    theta_max_eval=50,
    seed=None,
    verbose=None,
)
```

| Name | Type | Default value | Description |
| --- | --- | --- | --- |
| `xspecs` | `Optional[[XSpec]]` | `None` | Optional input variable specification (`XSpec`-like). |
| `regr_spec` | `int` | `1` | Regression basis bitflags (`RegressionSpec`). Possible flag values: `CONSTANT` (`1`), `LINEAR` (`2`), `QUADRATIC` (`4`), bitwise combinations allowed. |
| `corr_spec` | `int` | `1` | Correlation kernel bitflags (`CorrelationSpec`). Possible flag values: `SQUARED_EXPONENTIAL` (`1`), `ABSOLUTE_EXPONENTIAL` (`2`), `MATERN32` (`4`), `MATERN52` (`8`), bitwise combinations allowed. |
| `kpls_dim` | `Optional[int]` | `None` | PLS projection dimension for high-dimensional inputs. |
| `n_clusters` | `int` | `1` | Number of local experts/clusters in the mixture. |
| `recombination` | `Recombination` | `Recombination.HARD` | Expert recombination policy. Possible values: `Recombination.HARD`, `Recombination.SMOOTH`. |
| `theta_init` | `Optional[Sequence[float]]` | `None` | Initial GP hyperparameter guess. |
| `theta_bounds` | `Optional[Sequence[Sequence[float]]]` | `None` | Search bounds for GP hyperparameter optimization. |
| `theta_n_start` | `int` | `10` | Hyperparameter optimization multistart count (`n_start` is deprecated since 0.38.0). |
| `theta_max_eval` | `int` | `50` | Max likelihood evaluations during hyperparameter optimization (`max_eval` is deprecated since 0.38.0). |
| `seed` | `Optional[int]` | `None` | RNG seed for reproducibility. |
| `verbose` | `Optional[Any]` | `None` | Optional logging verbosity. Possible values: `None`, integer level, or `Verbose` enum (`ERROR`, `WARNING`, `INFO`, `DEBUG`, `TRACE`). |

## Gpx methods

```python
def __repr__(self) -> str
def __str__(self) -> str
def save(self, filename: str) -> bool
@staticmethod
def load(filename: str) -> Gpx
def predict(self, x: NDArray[float64], return_std: bool = False) -> NDArray[float64] | tuple[NDArray[float64], NDArray[float64]]
def predict_var(self, x: NDArray[float64]) -> NDArray[float64]
def predict_gradients(self, x: NDArray[float64]) -> NDArray[float64]
def predict_var_gradients(self, x: NDArray[float64]) -> NDArray[float64]
def sample(self, x: NDArray[float64], n_traj: int) -> NDArray[float64]
def dims(self) -> tuple[int, int]
nx: int  # read-only property, number of inputs
ny: int  # read-only property, number of outputs
def training_data(self) -> tuple[NDArray[float64], NDArray[float64]]
def thetas(self) -> NDArray[float64]
def variances(self) -> NDArray[float64]
def likelihoods(self) -> NDArray[float64]
```

## XSpecs

`xspecs` passed to `Egor(...)` or `Gpx.builder(...)` is a sequence of `XSpec`, one per input dimension (`len(xspecs) == nx`).

Note: When using continuous variables only, the simplest form is a list of `[lower, upper]` pairs, e.g. `[[0.0, 1.0], [1.0, 10.0], [-5.0, 5.0]]`. Otherwise, use `XSpec(...)` for more complex variable types.

Signature:

```python
XSpec(xtype: XType, xlimits: Sequence[float] = [], tags: Sequence[str] = [])
```

| Field | Type | Description |
| --- | --- | --- |
| `xtype` | `XType` | Variable kind. Possible values: `XType.FLOAT`, `XType.INT`, `XType.ORD`, `XType.ENUM`. |
| `xlimits` | `Sequence[float]` | Domain encoding, interpreted from `xtype` (see table below). |
| `tags` | `Sequence[str]` | Optional labels for enum categories (documentation labels only; optimization uses category indices). |

`xlimits` meaning by `xtype`:

| `XType` | `xlimits` format | Example |
| --- | --- | --- |
| `FLOAT` | `[lower: float, upper: float]` | `XSpec(XType.FLOAT, [0.0, 1.0])` |
| `INT` | `[lower: int, upper: int]` | `XSpec(XType.INT, [1, 10])` |
| `ORD` | ordered list of allowed values | `XSpec(XType.ORD, [0.1, 0.5, 1.0])` |
| `ENUM` | either `[n_categories]` or explicit `tags=[...]` | `XSpec(XType.ENUM, [3])` or `XSpec(XType.ENUM, tags=["red", "green", "blue"])` |

Full mixed-type example:

```python
xspecs = [
    XSpec(XType.FLOAT, [0.0, 1.0]),
    XSpec(XType.INT, [1, 8]),
    XSpec(XType.ORD, [0.2, 0.5, 1.0]),
    XSpec(XType.ENUM, tags=["small", "medium", "large"]),
]
```
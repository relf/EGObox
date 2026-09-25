# Basin integration

Basin is used by default everywhere: `egobox-ego` always runs its EGO solver
with Basin's executor, optimizes the infill criterion with Basin's pure Rust
COBYLA and SLSQP, and `egobox-gp` trains GP hyperparameters with Basin's COBYLA.

The `c-cobyla` and `c-slsqp` features switch back, respectively, to the `cobyla`
and `slsqp` crates (C-ported NLopt implementations). `c-cobyla` applies to both
GP training and infill optimization:

```sh
cargo run --release -p egobox-ego --features c-cobyla,c-slsqp --example ackley
cargo run --release -p egobox-gpx --features c-cobyla -- --help
cd python
maturin develop --release --features c-cobyla,c-slsqp
```

`c-cobyla` is available on `egobox-gp`, `egobox-moe`, `egobox-ego`,
`egobox-gpx`, and the Python package; `c-slsqp` on `egobox-ego`, `egobox-gpx`,
and the Python package. Both are disabled by default.

## Execution and compatibility

The high-level Rust and Python APIs (`EgorBuilder`, `EgorServiceBuilder`) retain
their existing signatures. `EgorSolver` implements `basin::Solver` and is run
by a private executor module which translates Basin execution results into
EGObox's public state. `egobox-ego` does not depend on Argmin anymore:
`EgorState` exposes its accessors as inherent methods, and termination is
reported with EGObox's own `TerminationStatus` and `TerminationReason` types.

Targets are checked after initialization and before further iterations. A target
of negative infinity disables target stopping. Iteration limits, timeouts,
objective errors, and interrupts retain their existing exit-status meanings.
Elapsed time measures the current invocation, including initialization. Timeout
checks occur after completed iterations and cannot interrupt a running objective
evaluation.

Basin writes solver-aware hot-start checkpoints to `egor_checkpoint.bin`
(former Argmin `egor_checkpoint.json` files are not supported anymore). Exact
continuation requires the same Basin version and concrete solver/problem types.
The checkpoint retains the solver, state, RNG, and evaluation counters. Loading
skips initialization, and `ExtendedIters(n)` adds to the saved iteration budget.
The saved target remains in effect. A target already reached stops immediately,
even when the iteration budget is extended.

Use the existing DOE warm-start mechanism to switch numerical backends or to
restart from an older run. Checkpoint read and write errors are returned to the
caller.

## Numerical settings

Both Basin adapters use EGObox's normalized inequalities, `c(x) <= 0`, including
the existing scaling and tolerance offsets. SLSQP uses analytic objective and
constraint derivatives. COBYLA folds box bounds into its constraint model.
Its trial points may lie outside the box. Acquisition callbacks and sparse GP
likelihoods are projected onto the bounds to keep domain functions and covariance
factorizations valid. Dense GP likelihoods retain their smooth extension outside
the box for COBYLA's interpolation, but only candidates within the bounds can be
returned. All initial points are projected, including sparse GP restarts whose
variance or noise bounds were adjusted after sampling.

The adapters enforce evaluation budgets inside callbacks, including solver
initialization. Gradient callbacks count too because EGObox's combined callback
also evaluates the objective. Successful feasible solver states are returned
directly. A budget stop, or a COBYLA radius stop whose incumbent remains
infeasible, returns the best feasible evaluated candidate when one exists.
Numerical failures retain the existing infinity fallback.

COBYLA checks absolute and relative objective tolerances between completed
radius stages, detected when its published `rho` decreases. The first completed
stage establishes the reference cost, and a later stage must produce a strictly
better feasible cost before an objective tolerance can stop the solve. Checking
every small improvement would stop too early in flat acquisition functions.
Infeasible stages reset the cost comparison. COBYLA uses the configured initial
radius and a final radius of `1e-6`.

SLSQP checks objective tolerances after published feasible steps. It uses
exact-zero native accuracy tests so that an additional absolute accuracy floor
does not override EGObox's objective tolerances. Solver paths and evaluation
counts can differ from the original implementations.

## Reproducing comparisons

See the [recorded comparison](basin-comparison.md) for numerical results,
historical timings, and the investigation of the original stopping policy.

The comparison example emits JSON Lines for `xsinx`, constrained G24, and
three-dimensional Ackley, using both acquisition algorithms. It runs seeds
0, 1, 2, 3, and 42, one warm-up, and ten measured repetitions per case. Each run
has a 30-iteration limit and a documented target in the example. Records include
objective value, constraint violation, observations, iterations, exit reason, and
wall-clock time.

```sh
cargo build --release -p egobox-ego --features c-cobyla,c-slsqp --example backend_comparison
RAYON_NUM_THREADS=2 target/release/examples/backend_comparison 10 > c-optimizers.jsonl
cargo build --release -p egobox-ego --example backend_comparison
RAYON_NUM_THREADS=2 target/release/examples/backend_comparison 10 > basin.jsonl
```

Compile before measuring, keep thread counts fixed, and run the binaries without
competing build jobs. Compare solution quality and work performed alongside time:
a faster run that stops earlier is not necessarily a faster solver. The separate
executor commit can be measured with the same harness to isolate framework
overhead while retaining the original numerical backends.

For a comparison of GP fitting alone, use
`cargo run --release -p egobox-gp --example gp_backend_comparison -- 10`
and repeat with `--features c-cobyla`. This example trains on fixed seeded data
for one-dimensional `xsinx` and three-dimensional Griewank. It reports training
time, likelihood, fitted hyperparameters, and validation RMSE. Prediction time is
excluded from the fitting measurement.

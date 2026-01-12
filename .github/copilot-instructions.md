# EGObox - Efficient Global Optimization Toolbox

## Project Overview

EGObox is a Rust library for Bayesian optimization implementing Efficient Global Optimization (EGO) algorithms with Python bindings via PyO3/maturin. The codebase is organized as a Cargo workspace with multiple interdependent crates plus a Python package.

**Core components (in `crates/`):**
- `doe`: Design of Experiments - sampling methods (LHS, FullFactorial, Random)
- `gp`: Gaussian Process regression with Kriging, PLS dimension reduction, sparse methods
- `moe`: Mixture of Experts using GP models for increased accuracy via weighted local models
- `ego`: The main EGO optimizer with constraint handling and mixed-integer support
- `gpqa`: Quality assurance for GP models

**Python bindings (`python/`)**: PyO3-based Python package wrapping Rust optimizer (`Egor`) and surrogate model (`Gpx`)

## Architecture Patterns

### Builder Pattern for Configuration
The project extensively uses builder pattern for complex configurations:
- `EgorBuilder::optimize(fn).configure(|config| ...).min_within(&xlimits).run()`
- Nested builders: `config.configure_gp(|gp| gp.regression_spec(...).correlation_spec(...))`
- Python mirrors this: `egx.Egor(...).minimize(f, max_iters=20)`

### Feature Flags Philosophy
Features control optional dependencies and capabilities across crates:
- `serializable`: Enable serde-based serialization
- `persistent`: Enable JSON save/load (implies serializable)
- `blas`: Optional BLAS/LAPACK backend via `ndarray-linalg` and Intel MKL for performance
- `nlopt`: Alternative optimizers (Cobyla, Slsqp) via nlopt crate

**Always specify features explicitly in dependencies** - most crates use `default-features = false` for linfa dependencies.

### Mixed-Integer Optimization
The optimizer handles continuous (Float), discrete (Int), ordinal (Ord), and categorical (Enum) variables via `XType` enum. Continuous optimization is the base, with `xlimits` defining continuous bounds, while mixed-integer uses `xtypes` specification.

## Development Workflows

### Testing
```bash
cargo test --release --all       # Run all Rust tests (MUST use --release for reasonable speed)
cd python && pytest              # Run Python tests
```

**Critical**: Always test in release mode - debug builds are prohibitively slow for numerical optimization algorithms.

### Running Examples
Examples demonstrate usage patterns:
```bash
cd crates/ego && cargo run --example xsinx --release
cd crates/gp && cargo run --example kriging --release
cd crates/moe && cargo run --example clustering --release
```

### Python Development
```bash
cd python
maturin develop --release       # Build and install Python package locally
pytest                          # Run Python tests
```

Python binding auto-generated stubs using `pyo3-stub-gen`. The `python/src/` mirrors Rust API with PyO3 wrappers.

### Enabling BLAS (Optional Performance)
```bash
cargo run --example kriging --release --features blas,linfa/intel-mkl-static
```
Enable via `cargo-extra-args = "--features linfa/intel-mkl-static"` in `python/pyproject.toml` `[tool.maturin]`.

## Code Conventions

### Gaussian Process Training Returns Types
GP training returns `GpSurrogate<F, D>` where `D` is dataset Lapack feature:
- `WithLapack`: When `blas` feature enabled
- `WithoutLapack`: Pure Rust implementation

Use `#[cfg(feature = "blas")]` guards when dealing with these types.

### Error Handling
- Custom error types via `thiserror` in each crate (`errors.rs`)
- Public APIs return `Result<T, Error>` with crate-specific error types
- Use `anyhow::Result` for internal/example code

### Parallel Processing
Heavy use of `rayon` for parallelization - operations on training data, optimization multistart, etc. Functions accepting closures should be `Send + Sync` aware.

### Serialization Patterns
When `serializable`/`persistent` features enabled:
```rust
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct MyStruct { ... }
```

For JSON persistence in `ego` and `moe`:
```rust
model.save("model.json")?;  // Requires persistent feature
let model = Model::load("model.json")?;
```

## Integration Points

### EGO → MOE → GP Dependency Chain
- `ego` depends on `moe` for surrogate modeling (mixture of experts)
- `moe` depends on `gp` for individual expert models
- All depend on `doe` for sampling
- Optimizer constructs surrogates via `SurrogateBuilder` trait

### Python ↔ Rust Boundary
- NumPy arrays bridge via `numpy` crate: `PyReadonlyArray2` (input), `IntoPyArray`/`ToPyArray` (output)
- `ndarray::Array2<f64>` is the standard Rust matrix type, view as `ArrayView2<f64>` for functions
- Function signatures: `fn(x: &ArrayView2<f64>) -> Array2<f64>` for objective functions

### External Optimizer Integration
- `argmin` framework powers the optimizer loop in `ego`
- Custom `Solver` implementation in `solver/egor_solver.rs`
- Observers for checkpointing/monitoring via `argmin::core::observers::Observe`

## Key Files

- [crates/ego/src/egor.rs](../crates/ego/src/egor.rs): Main EGO builder and optimizer API
- [crates/ego/src/solver/](../crates/ego/src/solver/): Core solver implementation, config, ask-tell interface
- [crates/gp/src/algorithm.rs](../crates/gp/src/algorithm.rs): GP training algorithm (~1800 lines)
- [crates/moe/src/algorithm.rs](../crates/moe/src/algorithm.rs): Mixture of experts implementation
- [python/src/lib.rs](../python/src/lib.rs): PyO3 module definition
- [python/src/egor.rs](../python/src/egor.rs): Python Egor class wrapping Rust
- [doc/](../doc/): Jupyter notebooks with tutorials

## Testing Notes

- Tests use `serial_test` crate for tests requiring deterministic execution order
- `#[serial]` attribute prevents parallel execution where RNG state matters
- Benchmarks use `criterion` - run with `cargo bench` (in specific crate directories)
- Use `approx::assert_abs_diff_eq!` for floating-point comparisons with epsilon tolerances

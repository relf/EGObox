# Efficient global optimization

[![crates.io](https://img.shields.io/crates/v/egobox-ego)](https://crates.io/crates/egobox-ego)
[![docs](https://docs.rs/egobox-ego/badge.svg)](https://docs.rs/egobox-ego)

`egobox-ego` provides a Rust implementation of efficient global optimization algorithm.

## The big picture

`egobox-ego` is a library crate in the top-level package [egobox](https://github.com/relf/egobox).

## Current state

`egobox-ego` currently implements EGO using `egobox-moe` with the following features:

* Mixture of gausian processes
* Infill criteria: EI, WB2, WB2S, LogEI, CEI, LogCEI
* Handling of negative constraints: actual constraint functions or surrogates  
* Mixed integer optimization available through continuous relaxation
* Trust region EGO algorithm
* CoEGO method with CCBO setting
* Multi-objective optimization (experimental): Pareto front approximation with ParEGO,
  Expected Improvement Matrix (EIM) or Expected Hypervolume Improvement (EHVI) strategies,
  batch EHVI (qEHVI) for batches of points

## Examples

There is some usage examples in the examples/ directory. To run, use:

``` bash
cargo run --release --example ackley
cargo run --release --example zdt1  # multi-objective optimization
```

## License

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0>

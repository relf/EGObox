# Gaussian processes

[![crates.io](https://img.shields.io/crates/v/egobox-gp)](https://crates.io/crates/egobox-gp)
[![docs](https://docs.rs/egobox-gp/badge.svg)](https://docs.rs/egobox-gp)

`egobox-gp` provides a Rust implementation of gaussian process regression.
It is a Rust port of some kriging algorithms of the [SMT](https://smt.readthedocs.io) Python library.

## The big picture

`egobox-gp` is a library crate in the top-level package [egobox](https://github.com/relf/EGObox).

## Current state

`egobox-gp` currently provides a Gaussian Process implementation with the following features:

* Regression model choice: constant, linear or quadratic
* Correlation model (kernel) choice: squared exponential, absolute exponential, matern 3/2, matern 5/2
* Handling of high dimensional problem using PLS (`linfa-pls`)

## Examples

There is an usage example in the examples/ directory. To run, use:

```
$ cargo run --release --example kriging
```

## License

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0


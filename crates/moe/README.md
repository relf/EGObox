# Mixture of experts

[![crates.io](https://img.shields.io/crates/v/egobox-moe)](https://crates.io/crates/egobox-moe)
[![docs](https://docs.rs/egobox-moe/badge.svg)](https://docs.rs/egobox-moe)

`egobox-moe` provides a Rust implementation of mixture of experts algorithm.
It is a Rust port of mixture of expert of the [SMT](https://smt.readthedocs.io) Python library.

## The big picture

`egobox-moe` is a library crate in the top-level package [egobox](https://github.com/relf/egobox).

## Current state

`egobox-moe` currently implements mixture of gaussian processes provided by `egobox-gp`:

* Clustering (`linfa-clustering/gmm`)
* Hard recombination / Smooth recombination
* Gaussian processe model choice: specify regression and correlation allowed models 
* Mixed-interger variables handling through continuous relaxation

## Examples

There is some usage examples in the examples/ directory. To run, use:

```
$ cargo run --release --example clustering
```

## License

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0


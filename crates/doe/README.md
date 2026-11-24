# Design of experiments

[![crates.io](https://img.shields.io/crates/v/egobox-doe)](https://crates.io/crates/egobox-doe)
[![docs](https://docs.rs/egobox-doe/badge.svg)](https://docs.rs/egobox-doe)

`egobox-doe` provides a Rust implementation of some design of experiments building methods.
It is a Rust port of sampling methods of the [SMT](https://smt.readthedocs.io) Python library.

## The big picture

`egobox-doe` is a library crate in the top-level package [egobox](https://github.com/relf/EGObox).

## Current state

`egobox-doe` currently provides an implementation of the following methods:

* Random sampling
* Full-factorial sampling
* Latin hypercube sampling: classic, centered, optimized

## Examples

There is an usage example in the examples/ directory. To run, use:

```
$ cargo run --release --example samplings
```

## License

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0


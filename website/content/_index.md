+++
title = "EGObox"
sort_by = "weight"
+++

# EGObox

Rust toolbox for Efficient Global Optimization method (arguably the most well-known bayesian optimization algorithm) which adresses the gradient-free optimization of expensive objective functions.

**EGObox** is twofold:

1. for end-users: a [Python module](https://pypi.org/project/egobox/), the Python binding of the optimizer named **`Egor`** and the surrogate model **`Gpx`**, mixture of Gaussian processes, written in Rust.
2. for developers: a set of [Rust crates](https://github.com/relf/EGObox/tree/master/crates) useful to use or implement bayesian optimization (EGO-like) algorithms.

## Content

This website focuses on Python interface and contains the following sections:

* [Get started](getting-started): learn to install EGObox
* [Examples](examples): discover usage of EGObox
* [Tutorials](tutorials): interactive tutorials to learn how to use EGObox
* [Cookbook](cookbook): find practical recipes for `Egor` optimizer common tasks
* [Python API](python-api): reference for the Python interface

## Cite

The EGObox toolbox rationale is described in the [JOSS paper](https://doi.org/10.21105/joss.04737):

```
@article{
  Lafage2022, 
  author = {Rémi Lafage}, 
  title = {EGObox, a Rust toolbox for efficient global optimization}, 
  journal = {Journal of Open Source Software} 
  year = {2022}, 
  doi = {10.21105/joss.04737}, 
  url = {https://doi.org/10.21105/joss.04737}, 
  publisher = {The Open Journal}, 
  volume = {7}, 
  number = {78}, 
  pages = {4737}, 
}
```
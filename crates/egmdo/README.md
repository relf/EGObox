# Efficient global multidisciplinary design optimization

`egobox-egmdo` provides a Rust implementation of the EGMDO algorithm (Dubreuil et al., 2020) and of its constrained extension C-EGMDO (Cardoso et al., 2024), where the disciplinary solvers of a multidisciplinary design optimization problem are replaced by Gaussian process surrogates enriched adaptively.

## The big picture

`egobox-egmdo` is a library crate in the top-level package [egobox](https://github.com/relf/egobox).

## Current state

The crate is under development on the `egmdo` branch. It currently provides:

* the definition of MDO problems: design space, disciplines, coupling variables, objective and constraints,
* the EGMDO and C-EGMDO algorithms, for problems with or without constraints, as an ask/tell solver (`EgmdoSolver`) or with given disciplinary solvers (`minimize`),
* the acquisition criteria (EI/CEI, WB2, WB2S, Monte Carlo LogEI) and the enrichment of all disciplines or of the one selected by Sobol indices,
* the disciplinary Gaussian process surrogates,
* the multidisciplinary analysis (MDA) solver, used with the true disciplines, the surrogate means or random samples of the surrogates,
* the polynomial chaos expansions with Sobol indices, and the Karhunen–Loève decomposition and KL-GP random field,
* the test problems of the EGMDO and C-EGMDO papers, with benchmarks in `examples/egmdo_benchmarks.rs`.

Variables are identified by index in this crate. The Python binding, `Mdegor` in the `egobox` package (also in `egobox.egmdo`), declares the design and coupling variables by name and wires the disciplines to them by name.

See `doc/egmdo_integration_plan.md`, `doc/egmdo_spec.md` and `doc/egmdo_benchmarks.md` in the repository for the plan, the specification and the benchmark results.

## References

S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, *Towards an efficient global multidisciplinary design optimization algorithm*, Struct. Multidisc. Optim. 62:1739–1765, 2020. [doi:10.1007/s00158-020-02514-6](https://doi.org/10.1007/s00158-020-02514-6)

I. Cardoso, S. Dubreuil, N. Bartoli, C. Gogu, M. Salaün, *Constrained efficient global multidisciplinary design optimization using adaptive disciplinary surrogate enrichment*, Struct. Multidisc. Optim. 67:23, 2024. [doi:10.1007/s00158-023-03736-0](https://doi.org/10.1007/s00158-023-03736-0)

## License

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0>

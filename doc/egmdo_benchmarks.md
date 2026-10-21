# EGMDO benchmarks

Results of `egobox-egmdo` on the test problems of the EGMDO (Dubreuil et al., 2020) and C-EGMDO (Cardoso et al., 2024) papers, with an `Egor` baseline. Decisions `Dn` refer to `doc/egmdo_spec.md`.

## Reproduction

```text
cargo run -p egobox-egmdo --release --example egmdo_benchmarks -- 20
```

The first argument is the number of seeds (0 to n−1). Problem names can follow it to run a subset: `toy`, `sellar_modified`, `compound_cylinder`, `toy_constrained`, `sellar_constrained`. The full run takes about 20 minutes on a laptop.

## Setup

**Problems and settings.** Each problem uses its paper's setting:

| Problem | Initial disciplinary DoEs | `DoE_UQ` | Iterations | Other |
|---|---|---|---|---|
| 1-D toy | 5 and 4 points | the paper's 4 points | 5 | |
| Modified Sellar | 5 and 5 | 20 (LHS) | 10 | |
| Compound cylinder | 6 and 8 (`2 d_i`) | 20 (LHS) | 20 | MDA relaxation 0.5 (D31) |
| Constrained toy | 5 and 4 | the C-EGMDO paper's 4 points | 5 | |
| Constrained Sellar | 5 and 5 | 20 (LHS) | 10 | `n_MC = 200` |

**EGMDO configurations.**

- *EGMDO paper*: `EgmdoConfig::egmdo()`, i.e. EI, all-discipline enrichment, `ε_P = 1/n_UQ`, `n_MC = 100`, squared exponential GPs.
- *defaults*: `EgmdoConfig::default()`, i.e. C-EGMDO options (WB2, Sobol enrichment, `ε_P = 10⁻³`) and Matérn 5/2 GPs (D34).
- *defaults, WB2S*: the defaults with the adaptive WB2 scale (D35).
- *defaults, CEI*: the defaults with the constrained EI instead of WB2.
- *defaults, LogEI*: the defaults with the Monte Carlo log (C)EI instead of WB2 (D36).

**Egor baseline (MDF).** `Egor` with its default settings minimizes the objective, and the constraints, computed by a true MDA at each point; non-linear Jacobi converges to a `10⁻⁶` relative change. Every call to a discipline inside the MDAs counts as one disciplinary evaluation. The table reports how many evaluations it took to first evaluate an optimal point (true objective within 1%, constraints below `10⁻³`), and the total over the run.

**Criteria.**

- *Converged*: the papers' criterion. For the 2020 problems, `E[Ẑ*]` must be within 5% of the reference (Euclidean relative error, D17). For the 2024 problems, `E[Ŷ_min]` must be within 1%.
- *Best point optimal*: the true objective at the best `DoE_UQ` point (D29) is within 1% of the reference, with the true constraints below `10⁻³`.
- *Disciplinary evaluations*: the mean ± standard deviation of the total true evaluations, all disciplines included, initial DoEs included.

## Results

20 seeds per configuration (0 to 19). Times are wall-clock medians per run on a laptop (release build).

| Problem | Method | Converged (papers' criterion) | Best point optimal | Disciplinary evaluations | Median time (s) |
|---|---|---|---|---|---|
| toy | EGMDO paper | 14/20 | 9/20 | 13.2 ± 4.2 | 0.4 |
| toy | defaults | 19/20 | 17/20 | 14.8 ± 3.8 | 0.4 |
| toy | defaults, WB2S | 19/20 | 18/20 | 15.9 ± 3.8 | 0.5 |
| toy | defaults, LogEI | 19/20 | 15/20 | 15.7 ± 3.7 | 0.5 |
| toy | Egor MDF (20 iterations) | optimum reached 20/20, after 364 evaluations (median) | 20/20 | 1045.0 ± 41.2 | 0.1 |
| sellar_modified | EGMDO paper | 14/20 | 1/20 | 22.8 ± 3.4 | 2.9 |
| sellar_modified | defaults | 18/20 | 18/20 | 20.4 ± 2.2 | 3.0 |
| sellar_modified | defaults, WB2S | 20/20 | 20/20 | 20.3 ± 2.4 | 3.0 |
| sellar_modified | defaults, LogEI | 20/20 | 20/20 | 20.9 ± 2.6 | 3.5 |
| sellar_modified | Egor MDF (40 iterations) | optimum reached 20/20, after 470 evaluations (median) | 20/20 | 902.5 ± 41.6 | 0.3 |
| compound_cylinder | EGMDO paper | 20/20 | 11/20 | 24.4 ± 2.5 | 5.5 |
| compound_cylinder | defaults | 20/20 | 14/20 | 23.8 ± 2.8 | 8.0 |
| compound_cylinder | Egor MDF (40 iterations) | optimum reached 20/20, after 1318 evaluations (median) | 20/20 | 2709.4 ± 43.8 | 0.4 |
| toy_constrained | defaults (WB2) | 17/20 | 12/20 | 22.4 ± 4.7 | 0.7 |
| toy_constrained | defaults, WB2S | 15/20 | 13/20 | 22.4 ± 4.6 | 0.7 |
| toy_constrained | defaults, CEI | 15/20 | 11/20 | 22.4 ± 4.6 | 0.7 |
| toy_constrained | defaults, LogEI | 15/20 | 10/20 | 22.6 ± 4.7 | 0.7 |
| toy_constrained | Egor MDF (20 iterations) | optimum reached 20/20, after 474 evaluations (median) | 20/20 | 1343.8 ± 117.7 | 0.2 |
| sellar_constrained | defaults (WB2) | 19/20 | 17/20 | 24.4 ± 2.3 | 2.9 |
| sellar_constrained | defaults, WB2S | 19/20 | 17/20 | 24.9 ± 2.1 | 3.1 |
| sellar_constrained | defaults, CEI | 11/20 | 7/20 | 21.9 ± 2.9 | 2.6 |
| sellar_constrained | defaults, LogEI | 16/20 | 17/20 | 24.4 ± 3.0 | 2.9 |
| sellar_constrained | Egor MDF (40 iterations) | optimum reached 20/20, after 270 evaluations (median) | 20/20 | 913.7 ± 32.9 | 0.5 |

### Comparison with the papers

The papers report evaluations per discipline; they are summed here.

| Problem | Paper | Here |
|---|---|---|
| Modified Sellar, 10 iterations | 7/10 converged, about 24 evaluations | EGMDO paper setting: 14/20, 22.8; defaults: 18/20, 20.4; WB2S: 20/20, 20.3 |
| Modified Sellar, 15 iterations | 88/100 converged, about 26 evaluations | (not run) |
| Compound cylinder, `2 d_i` DoEs | 8/10 converged, 28 evaluations | 20/20, 24.4 (paper setting) and 23.8 (defaults) |
| Constrained Sellar, WB2 | 96/100 within 1%, 22.8 evaluations | 19/20, 24.4 |
| Constrained Sellar, CEI | 63/100 within 1%, 21.6 evaluations | 11/20, 21.9 |
| Constrained Sellar, MDF + SEGOMOE | 94/100, 259.5 evaluations | Egor MDF: 20/20, optimum after 270 (median) |

## Findings

- **The results match the papers.** The papers' settings converge as often as reported. On constrained Sellar, the gap between WB2 and CEI is the paper's: CEI struggles with the flat zones of the EI.
- **The defaults improve on the 2020 settings.** On every unconstrained problem they converge more often for a similar number of evaluations: toy 19/20 against 14/20, modified Sellar 18/20 against 14/20. Matérn 5/2 GPs are better calibrated with the small initial DoEs (D34), and the C-EGMDO options enrich only the discipline that matters.
- **EGMDO needs 11 to 55 times fewer disciplinary evaluations than `Egor` on the MDF formulation** to reach the optimum. The MDF cost includes every Jacobi iteration of every MDA, typically 10 to 20 iterations. An IDF formulation would lower the `Egor` cost (the C-EGMDO paper reports 56.6 evaluations for IDF-SEGOMOE on constrained Sellar) and is not benchmarked here. `Egor` reaches the optimum in every run, against 85 to 100% of the runs for EGMDO with the defaults. EGMDO trades some robustness for a much smaller number of disciplinary evaluations.
- **WB2S (D35) is the best option on modified Sellar** (20/20). It is equivalent elsewhere: 19/20 on constrained Sellar, and 15/20 against 17/20 on the constrained toy, which is within the noise of 20 runs. WB2 stays the default, as in the C-EGMDO paper.
- **LogEI (D36) removes the flat zones of the (C)EI.** On constrained Sellar, the best point is optimal in 17/20 runs against 7/20 with CEI, and 16/20 runs converge against 11/20, for a similar number of evaluations. On modified Sellar it matches WB2S (20/20). It does not beat WB2 overall (constrained toy 15/20 against 17/20, constrained Sellar 16/20 against 19/20), so WB2 stays the default. LogEI is the better choice when an EI-type criterion is wanted, e.g. without the exploitation term of WB2.
- **WB2S does not remove the mirrored-basin samples of constrained Sellar.** In some runs, part of the final samples of `Ẑ*` fall near `z1 ≈ −1.8`, where the model (19) is less accurate, although the best point is the optimum. Once the disciplinary GPs are enriched, the EI is zero on almost all the design space, so no scale makes WB2S explore that basin. The bias comes from the final estimation with the model (19), not from the acquisition. The "best point optimal" column is the more reliable indicator in that case.
- **The "best point" can be poor while `E[Ẑ*]` is right.** This happens with the 2020 settings on modified Sellar (1/20): the squared exponential GPs give biased PCE means at the `DoE_UQ` points, whereas the distribution of `Ẑ*` is centred on the optimum.

## Why not reuse the EI and LogEI of `egobox-ego`?

`egobox_ego::criteria` implements the analytic EI and LogEI of a Gaussian prediction. They take a `MixtureGpSurrogate`, from which only `N(μ, σ²)` at the point is used, and a scalar current minimum `fmin`. The EGMDO criterion (16, C-19) is `E[(f'_min(Ξ) − Ỹ_obj(z, Ξ, η))⁺]`, which differs in two ways:

- **The random field `Ỹ_obj` is not Gaussian.** Its KL random variables `ζ_k(Ξ)` are degree-3 polynomials of `Ξ`, multiplied by the KL-GP terms. Its mean and variance do not determine it, which is why the paper estimates the EI by Monte Carlo.
- **The current minimum is random and coupled with the field.** `f'_min(Ξ)` is the (feasible) minimum of the `DoE_UQ` PCEs for the same sample `Ξ` as `Ỹ_obj(z, Ξ, η)`. A scalar `fmin` loses this coupling, which lets the criterion tell the disciplinary uncertainty from the KL-GP interpolation uncertainty.

The model is moreover a KL-GP field, not a `MixtureGpSurrogate`, and the `grad` methods of the `InfillCriterion` trait have no Monte Carlo counterpart here. A Gaussian moment-matched variant could call `egobox-ego`'s LogEI, but it would be a different criterion from the papers'. The Monte Carlo LogEI (D36) is the counterpart, from the same paper (qLogEI), that keeps the field and the coupling. `egobox-ego`'s WB2S scaling rule is reused, in D35.

## Linearized random MDA (D37)

Issue #493 proposes alternatives to the rank-one disciplinary GP of the random MDA (9). The first one is implemented: the random MDA is linearized at the mean MDA solution, `y(Ξ) = y_0 + (I − J)⁻¹ diag(σ) Ξ` (D37), and the PCEs are fitted on these samples as before. The comparison below varies the size of the initial disciplinary DoEs, i.e. the level of disciplinary GP uncertainty.

The example also compares the correlations across design points (D38, next section); the tables of this section are its columns and rows for the perfect correlation.

```text
cargo run -p egobox-egmdo --release --example egmdo_uncertainty -- uq 10
cargo run -p egobox-egmdo --release --example egmdo_uncertainty -- opt 10
```

### Uncertainty quantification only

With the same disciplinary GPs (initial DoEs of the given size per discipline, Matérn 5/2) and the same design points (the problem's `DoE_UQ`), the objective PCEs and `P_min` (`P_sol` with constraints) from both random MDAs, over 10 seeds:

- *Objective std*: standard deviation of the objective PCE with the solved random MDA, median over the design points, as the level of uncertainty;
- *Std ratio* and *mean shift*: linearized std over solved std, and difference of the means in solved standard deviations, median [10%, 90%] over the design points;
- *P_min distance*: total variation distance between both `P_min` over the design points, median over the seeds; *same most probable minimizer*: number of seeds where both `P_min` have the same mode;
- *Failed random MDAs*: fraction of the solved random MDAs that do not converge; *fallbacks*: points where the linearization falls back to the solved random MDAs; *time*: per seed, all design points.

| Problem | Initial DoE per discipline | Objective std (solved, median) | Std ratio lin./solved | Mean shift (solved std) | P_min distance (median) | Same most probable minimizer | Failed random MDAs | Fallbacks | Dropped points | Time solved / lin. (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| toy | 3 | 0.460 | 0.99 [0.58, 1.25] | 0.15 [0.03, 0.49] | 0.150 | 8/10 | 2.77% | 0/40 | 0 | 0.041 / 0.003 |
| toy | 4 | 0.523 | 0.99 [0.70, 1.06] | 0.06 [0.00, 0.21] | 0.089 | 8/10 | 0.09% | 0/40 | 0 | 0.024 / 0.002 |
| toy | 6 | 0.099 | 1.00 [0.98, 1.03] | 0.01 [0.00, 0.02] | 0.001 | 10/10 | 0.00% | 0/40 | 0 | 0.003 / 0.000 |
| toy | 10 | 0.044 | 1.00 [0.97, 1.02] | 0.01 [0.00, 0.03] | 0.000 | 10/10 | 0.00% | 0/40 | 0 | 0.006 / 0.000 |
| sellar_modified | 4 | 15.183 | 1.00 [0.95, 1.01] | 0.00 [0.00, 0.05] | 0.031 | 10/10 | 0.00% | 0/200 | 0 | 0.005 / 0.001 |
| sellar_modified | 5 | 11.834 | 1.00 [0.96, 1.03] | 0.00 [0.00, 0.09] | 0.003 | 9/10 | 0.00% | 0/200 | 0 | 0.019 / 0.001 |
| sellar_modified | 8 | 2.861 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.01] | 0.000 | 10/10 | 0.00% | 0/200 | 0 | 0.010 / 0.001 |
| sellar_modified | 12 | 0.733 | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.01] | 0.000 | 9/10 | 0.00% | 0/200 | 0 | 0.019 / 0.001 |
| compound_cylinder | 5 | 0.068 | 1.00 [0.70, 1.10] | 0.02 [0.00, 0.20] | 0.115 | 10/10 | 0.01% | 0/200 | 0 | 0.029 / 0.001 |
| compound_cylinder | 8 | 0.013 | 0.99 [0.78, 1.03] | 0.02 [0.00, 0.11] | 0.013 | 9/10 | 0.12% | 0/200 | 0 | 0.071 / 0.002 |
| compound_cylinder | 12 | 0.007 | 1.00 [0.96, 1.03] | 0.02 [0.00, 0.04] | 0.004 | 10/10 | 0.00% | 0/200 | 0 | 0.056 / 0.002 |
| compound_cylinder | 16 | 0.004 | 1.00 [0.98, 1.01] | 0.01 [0.00, 0.03] | 0.003 | 10/10 | 0.00% | 0/200 | 0 | 0.071 / 0.002 |
| toy_constrained | 3 | 0.450 | 1.00 [0.66, 1.30] | 0.14 [0.01, 0.42] | 0.030 | 7/10 | 2.79% | 0/40 | 0 | 0.036 / 0.002 |
| toy_constrained | 4 | 0.484 | 1.00 [0.71, 1.10] | 0.06 [0.00, 0.38] | 0.035 | 9/10 | 0.04% | 0/40 | 0 | 0.024 / 0.002 |
| toy_constrained | 6 | 0.109 | 1.00 [0.99, 1.01] | 0.01 [0.00, 0.02] | 0.000 | 10/10 | 0.00% | 0/40 | 0 | 0.002 / 0.000 |
| toy_constrained | 10 | 0.065 | 1.00 [0.99, 1.01] | 0.02 [0.01, 0.04] | 0.000 | 10/10 | 0.00% | 0/40 | 0 | 0.005 / 0.000 |
| sellar_constrained | 4 | 15.993 | 1.00 [0.66, 1.28] | 0.00 [0.00, 0.23] | 0.146 | 7/10 | 0.00% | 0/200 | 0 | 0.009 / 0.001 |
| sellar_constrained | 5 | 11.406 | 1.00 [0.82, 1.13] | 0.00 [0.00, 0.16] | 0.079 | 8/10 | 0.00% | 0/200 | 0 | 0.009 / 0.001 |
| sellar_constrained | 8 | 2.718 | 1.00 [0.96, 1.00] | 0.01 [0.00, 0.03] | 0.003 | 10/10 | 0.00% | 0/200 | 0 | 0.011 / 0.001 |
| sellar_constrained | 12 | 0.990 | 1.00 [0.04, 1.00] | 0.01 [0.00, 0.11] | 0.027 | 10/10 | 0.00% | 0/200 | 0 | 0.020 / 0.001 |

### Whole algorithm

The defaults (C-EGMDO options, Matérn 5/2) with the problem settings of the benchmarks above, except the initial DoE sizes, over 10 seeds (0 to 9). *Failed random MDAs* sums, over the runs, the non-converged random MDAs at the final `DoE_UQ`; *solved points* counts the `DoE_UQ` points where the random MDA was solved (for the linearized runs, the fallbacks).

| Problem | Initial DoE per discipline | Random MDA | Converged (papers' criterion) | Best point optimal | Disciplinary evaluations | Failed random MDAs | Solved points (lin. fallbacks) | Median time (s) |
|---|---|---|---|---|---|---|---|---|
| toy | 3 | Solved | 10/10 | 10/10 | 14.4 ± 1.9 | 0 | 90 | 0.66 |
| toy | 3 | Linearized | 10/10 | 10/10 | 14.4 ± 2.3 | 0 | 0 | 0.40 |
| toy | 4 | Solved | 10/10 | 10/10 | 14.7 ± 2.5 | 0 | 90 | 0.48 |
| toy | 4 | Linearized | 9/10 | 7/10 | 13.9 ± 1.8 | 0 | 0 | 0.37 |
| toy | 6 | Solved | 10/10 | 10/10 | 15.2 ± 2.2 | 0 | 90 | 0.38 |
| toy | 6 | Linearized | 10/10 | 10/10 | 14.5 ± 1.1 | 0 | 0 | 0.31 |
| toy | 10 | Solved | 10/10 | 9/10 | 24.6 ± 2.2 | 2 | 90 | 0.49 |
| toy | 10 | Linearized | 9/10 | 9/10 | 23.9 ± 1.7 | 0 | 0 | 0.36 |
| sellar_modified | 4 | Solved | 8/10 | 9/10 | 20.2 ± 2.4 | 7 | 300 | 3.13 |
| sellar_modified | 4 | Linearized | 8/10 | 8/10 | 18.8 ± 2.4 | 9 | 2 | 2.36 |
| sellar_modified | 5 | Solved | 10/10 | 10/10 | 20.1 ± 2.0 | 58 | 300 | 2.92 |
| sellar_modified | 5 | Linearized | 9/10 | 10/10 | 19.2 ± 2.2 | 88 | 1 | 2.48 |
| sellar_modified | 8 | Solved | 8/10 | 9/10 | 21.7 ± 1.7 | 0 | 300 | 2.75 |
| sellar_modified | 8 | Linearized | 8/10 | 9/10 | 21.1 ± 1.4 | 0 | 0 | 2.23 |
| sellar_modified | 12 | Solved | 9/10 | 9/10 | 27.9 ± 1.8 | 3 | 299 | 2.46 |
| sellar_modified | 12 | Linearized | 9/10 | 9/10 | 27.9 ± 2.2 | 124 | 1 | 2.16 |
| compound_cylinder | 5 | Solved | 10/10 | 7/10 | 21.5 ± 3.6 | 1 | 400 | 8.14 |
| compound_cylinder | 5 | Linearized | 10/10 | 8/10 | 22.0 ± 3.5 | 0 | 0 | 6.87 |
| compound_cylinder | 8 | Solved | 10/10 | 8/10 | 25.2 ± 2.4 | 0 | 400 | 8.47 |
| compound_cylinder | 8 | Linearized | 10/10 | 9/10 | 25.6 ± 2.9 | 0 | 0 | 6.42 |
| compound_cylinder | 12 | Solved | 10/10 | 9/10 | 30.7 ± 1.3 | 0 | 399 | 8.05 |
| compound_cylinder | 12 | Linearized | 10/10 | 9/10 | 30.9 ± 1.4 | 0 | 0 | 6.51 |
| compound_cylinder | 16 | Solved | 10/10 | 7/10 | 36.6 ± 1.0 | 0 | 400 | 6.86 |
| compound_cylinder | 16 | Linearized | 10/10 | 10/10 | 36.8 ± 0.9 | 0 | 0 | 5.51 |
| toy_constrained | 3 | Solved | 8/10 | 7/10 | 21.5 ± 1.9 | 0 | 90 | 1.07 |
| toy_constrained | 3 | Linearized | 10/10 | 9/10 | 20.1 ± 2.1 | 0 | 0 | 0.43 |
| toy_constrained | 4 | Solved | 9/10 | 5/10 | 22.0 ± 2.7 | 0 | 90 | 0.82 |
| toy_constrained | 4 | Linearized | 8/10 | 5/10 | 20.4 ± 1.8 | 0 | 0 | 0.43 |
| toy_constrained | 6 | Solved | 9/10 | 8/10 | 21.0 ± 1.3 | 2 | 90 | 0.57 |
| toy_constrained | 6 | Linearized | 9/10 | 9/10 | 20.0 ± 1.1 | 0 | 0 | 0.35 |
| toy_constrained | 10 | Solved | 10/10 | 8/10 | 28.2 ± 0.9 | 0 | 90 | 0.57 |
| toy_constrained | 10 | Linearized | 10/10 | 6/10 | 27.8 ± 0.9 | 0 | 0 | 0.35 |
| sellar_constrained | 4 | Solved | 8/10 | 10/10 | 23.4 ± 1.5 | 0 | 293 | 2.74 |
| sellar_constrained | 4 | Linearized | 8/10 | 10/10 | 23.9 ± 2.3 | 0 | 0 | 2.50 |
| sellar_constrained | 5 | Solved | 9/10 | 9/10 | 23.6 ± 2.2 | 97 | 295 | 2.69 |
| sellar_constrained | 5 | Linearized | 8/10 | 10/10 | 24.2 ± 2.1 | 86 | 3 | 2.27 |
| sellar_constrained | 8 | Solved | 9/10 | 10/10 | 26.5 ± 2.5 | 0 | 293 | 2.61 |
| sellar_constrained | 8 | Linearized | 9/10 | 10/10 | 26.3 ± 2.2 | 0 | 0 | 2.08 |
| sellar_constrained | 12 | Solved | 10/10 | 8/10 | 33.6 ± 2.4 | 13 | 293 | 2.67 |
| sellar_constrained | 12 | Linearized | 10/10 | 9/10 | 33.7 ± 2.2 | 229 | 4 | 2.00 |

### Findings

- **The linearization matches the solved random MDA once the disciplinary GPs are accurate enough:** from 6 points per discipline on the toy problems, 8 on the Sellar problems and 12 on the compound cylinder. The standard deviations then agree within 4% at 80% of the points, the means within 0.05 standard deviation at 90% of the points, and `P_min` within 0.004 in total variation (except constrained Sellar with 12 points, see the outliers below).
- **It departs from the solved random MDA with very small DoEs (3 to 4 points)**, where `σ` is large with respect to the curvature of the GP means: the standard deviation ratio spreads over [0.6, 1.3] on the toy problems and constrained Sellar, and `P_min` differs by 0.03 to 0.15. The most probable minimizer still agrees in 7 to 10 seeds out of 10.
- **The optimization results are equivalent.** Over the 20 problem and DoE size combinations, 184 of 200 linearized runs converge against 187 for the solved random MDA, and the best point is optimal in 176 against 172 runs, for the same number of disciplinary evaluations (within 1.6 evaluations, slightly fewer with the linearization). The differences are within the noise of 10 runs per row; the largest is on the toy problem with 4 points per discipline (best point optimal 7/10 against 10/10).
- **A few outliers remain where the GPs extrapolate.** On constrained Sellar with 12 points per discipline, 10% of the design points have a linearized standard deviation below 4% of the solved one. They are at the edge of the design space (`|z1| ≈ 9`), where `σ_2` reaches 5 to 12 and the objective term `exp(−y_2)` makes the objective distribution extremely heavy-tailed (standard deviations of 10⁹ and more for both random MDAs): the solved random MDA has heavier `y_2` tails. These points are far from the optimum and do not change `P_min`.
- **No random MDA is solved, except at about 1% of the points.** The linearization falls back to the solved random MDAs where the mean MDA does not converge: 0 to 4 points per row out of 90 to 400, on the Sellar problems. These hard points concentrate the remaining failed random MDAs.
- **The scaling of `I − J` matters.** On the compound cylinder, the two coupling variables differ by six orders of magnitude, and the unscaled singular values of `I − J` have a ratio of 10⁻¹² although the coupling loop is well-posed (`J_12 J_21 ≈ −0.7`). The conditioning is tested on `S⁻¹(I − J)S`, `S = diag(|y_0| + σ)` (D37).
- **The uncertainty quantification is 5 to 35 times faster, the whole run 10 to 60% faster.** On these small problems, the random MDAs are a minor part of a run, which is dominated by the acquisition optimization, the KL-GPs and the `P_min` sampling. The gain grows with the number of coupling variables and the cost of the GP predictions.

## Correlation across design points (D38)

The second alternative of issue #493 replaces the perfect correlation of the disciplinary uncertainty across `DoE_UQ` (one `Ξ` shared by every point) by the GP posterior correlation between the disciplinary inputs at the mean MDA solutions (D38). The same example compares both correlations, with the solved and the linearized random MDAs (commands in the previous section).

### Uncertainty quantification only

Same setting as the previous section, with the solved random MDA:

- *GP correlation between points*: posterior correlations `ρ_q(i, l)` between the `DoE_UQ` points, all couplings and pairs, median [10%, 90%];
- *P_min distance*: total variation distance between `P_min` (`P_sol` with constraints) with the GP posterior and with the perfect correlation, median over the seeds;
- *same most probable minimizer*: number of seeds where both `P_min` have the same mode.

| Problem | Initial DoE per discipline | GP correlation between points | P_min distance GP/perfect correlation | Same most probable minimizer GP/perfect |
|---|---|---|---|---|
| toy | 3 | 0.24 [0.05, 1.00] | 0.111 | 9/10 |
| toy | 4 | 0.07 [-0.32, 0.55] | 0.091 | 9/10 |
| toy | 6 | -0.09 [-0.88, 0.72] | 0.040 | 10/10 |
| toy | 10 | -0.03 [-0.50, 0.34] | 0.001 | 10/10 |
| sellar_modified | 4 | 0.05 [-0.30, 0.67] | 0.286 | 9/10 |
| sellar_modified | 5 | 0.03 [-0.39, 0.66] | 0.256 | 9/10 |
| sellar_modified | 8 | 0.01 [-0.53, 0.62] | 0.017 | 10/10 |
| sellar_modified | 12 | -0.01 [-0.63, 0.75] | 0.000 | 10/10 |
| compound_cylinder | 5 | 0.07 [-0.40, 0.75] | 0.226 | 9/10 |
| compound_cylinder | 8 | 0.13 [-0.45, 0.70] | 0.114 | 10/10 |
| compound_cylinder | 12 | 0.07 [-0.41, 0.57] | 0.046 | 10/10 |
| compound_cylinder | 16 | 0.03 [-0.42, 0.50] | 0.029 | 10/10 |
| toy_constrained | 3 | 0.24 [-0.18, 1.00] | 0.148 | 8/10 |
| toy_constrained | 4 | 0.07 [-0.31, 0.62] | 0.105 | 9/10 |
| toy_constrained | 6 | 0.04 [-0.81, 0.80] | 0.000 | 10/10 |
| toy_constrained | 10 | -0.03 [-0.50, 0.43] | 0.000 | 10/10 |
| sellar_constrained | 4 | 0.09 [-0.40, 0.74] | 0.338 | 9/10 |
| sellar_constrained | 5 | 0.05 [-0.40, 0.73] | 0.238 | 8/10 |
| sellar_constrained | 8 | 0.05 [-0.50, 0.74] | 0.100 | 9/10 |
| sellar_constrained | 12 | 0.08 [-0.56, 0.89] | 0.179 | 9/10 |

### Whole algorithm

Same setting as the previous section, with the four combinations of random MDA and correlation. The rows with the perfect correlation are those of the previous section; with the perfect correlation, the results are bit-identical to the implementation before D38.

| Problem | Initial DoE per discipline | Random MDA | Correlation | Converged (papers' criterion) | Best point optimal | Disciplinary evaluations | Failed random MDAs | Solved points (lin. fallbacks) | Median time (s) |
|---|---|---|---|---|---|---|---|---|---|
| toy | 3 | Solved | Perfect | 10/10 | 10/10 | 14.4 ± 1.9 | 0 | 90 | 0.80 |
| toy | 3 | Linearized | Perfect | 10/10 | 10/10 | 14.4 ± 2.3 | 0 | 0 | 0.51 |
| toy | 3 | Solved | GpPosterior | 10/10 | 10/10 | 14.7 ± 3.0 | 1 | 90 | 0.82 |
| toy | 3 | Linearized | GpPosterior | 10/10 | 9/10 | 15.0 ± 2.0 | 0 | 0 | 0.57 |
| toy | 4 | Solved | Perfect | 10/10 | 10/10 | 14.7 ± 2.5 | 0 | 90 | 0.55 |
| toy | 4 | Linearized | Perfect | 9/10 | 7/10 | 13.9 ± 1.8 | 0 | 0 | 0.47 |
| toy | 4 | Solved | GpPosterior | 10/10 | 8/10 | 15.9 ± 2.8 | 2 | 90 | 0.68 |
| toy | 4 | Linearized | GpPosterior | 9/10 | 6/10 | 13.8 ± 1.7 | 0 | 0 | 0.54 |
| toy | 6 | Solved | Perfect | 10/10 | 10/10 | 15.2 ± 2.2 | 0 | 90 | 0.38 |
| toy | 6 | Linearized | Perfect | 10/10 | 10/10 | 14.5 ± 1.1 | 0 | 0 | 0.35 |
| toy | 6 | Solved | GpPosterior | 10/10 | 10/10 | 15.2 ± 2.2 | 0 | 90 | 0.49 |
| toy | 6 | Linearized | GpPosterior | 9/10 | 10/10 | 14.5 ± 1.3 | 0 | 0 | 0.48 |
| toy | 10 | Solved | Perfect | 10/10 | 9/10 | 24.6 ± 2.2 | 2 | 90 | 0.52 |
| toy | 10 | Linearized | Perfect | 9/10 | 9/10 | 23.9 ± 1.7 | 0 | 0 | 0.46 |
| toy | 10 | Solved | GpPosterior | 10/10 | 10/10 | 24.9 ± 2.7 | 2 | 90 | 0.69 |
| toy | 10 | Linearized | GpPosterior | 10/10 | 9/10 | 24.1 ± 2.0 | 0 | 0 | 0.51 |
| sellar_modified | 4 | Solved | Perfect | 8/10 | 9/10 | 20.2 ± 2.4 | 7 | 300 | 3.55 |
| sellar_modified | 4 | Linearized | Perfect | 8/10 | 8/10 | 18.8 ± 2.4 | 9 | 2 | 2.57 |
| sellar_modified | 4 | Solved | GpPosterior | 8/10 | 9/10 | 20.3 ± 2.6 | 15 | 300 | 6.77 |
| sellar_modified | 4 | Linearized | GpPosterior | 9/10 | 9/10 | 18.9 ± 2.7 | 21 | 2 | 7.60 |
| sellar_modified | 5 | Solved | Perfect | 10/10 | 10/10 | 20.1 ± 2.0 | 58 | 300 | 3.66 |
| sellar_modified | 5 | Linearized | Perfect | 9/10 | 10/10 | 19.2 ± 2.2 | 88 | 1 | 3.06 |
| sellar_modified | 5 | Solved | GpPosterior | 10/10 | 10/10 | 19.8 ± 2.0 | 0 | 300 | 7.18 |
| sellar_modified | 5 | Linearized | GpPosterior | 9/10 | 10/10 | 19.4 ± 2.1 | 0 | 0 | 6.63 |
| sellar_modified | 8 | Solved | Perfect | 8/10 | 9/10 | 21.7 ± 1.7 | 0 | 300 | 2.82 |
| sellar_modified | 8 | Linearized | Perfect | 8/10 | 9/10 | 21.1 ± 1.4 | 0 | 0 | 2.25 |
| sellar_modified | 8 | Solved | GpPosterior | 9/10 | 9/10 | 21.7 ± 1.1 | 0 | 300 | 6.70 |
| sellar_modified | 8 | Linearized | GpPosterior | 10/10 | 10/10 | 21.8 ± 2.4 | 0 | 0 | 6.66 |
| sellar_modified | 12 | Solved | Perfect | 9/10 | 9/10 | 27.9 ± 1.8 | 3 | 299 | 2.71 |
| sellar_modified | 12 | Linearized | Perfect | 9/10 | 9/10 | 27.9 ± 2.2 | 124 | 1 | 2.28 |
| sellar_modified | 12 | Solved | GpPosterior | 8/10 | 9/10 | 27.9 ± 1.8 | 3 | 300 | 6.33 |
| sellar_modified | 12 | Linearized | GpPosterior | 9/10 | 9/10 | 27.6 ± 1.4 | 4 | 2 | 6.30 |
| compound_cylinder | 5 | Solved | Perfect | 10/10 | 7/10 | 21.5 ± 3.6 | 1 | 400 | 8.53 |
| compound_cylinder | 5 | Linearized | Perfect | 10/10 | 8/10 | 22.0 ± 3.5 | 0 | 0 | 6.98 |
| compound_cylinder | 5 | Solved | GpPosterior | 10/10 | 8/10 | 21.8 ± 3.2 | 0 | 400 | 16.16 |
| compound_cylinder | 5 | Linearized | GpPosterior | 10/10 | 7/10 | 22.5 ± 3.3 | 0 | 0 | 16.29 |
| compound_cylinder | 8 | Solved | Perfect | 10/10 | 8/10 | 25.2 ± 2.4 | 0 | 400 | 8.13 |
| compound_cylinder | 8 | Linearized | Perfect | 10/10 | 9/10 | 25.6 ± 2.9 | 0 | 0 | 6.39 |
| compound_cylinder | 8 | Solved | GpPosterior | 10/10 | 10/10 | 25.8 ± 2.2 | 0 | 400 | 17.38 |
| compound_cylinder | 8 | Linearized | GpPosterior | 10/10 | 9/10 | 25.6 ± 2.8 | 0 | 0 | 15.45 |
| compound_cylinder | 12 | Solved | Perfect | 10/10 | 9/10 | 30.7 ± 1.3 | 0 | 399 | 7.63 |
| compound_cylinder | 12 | Linearized | Perfect | 10/10 | 9/10 | 30.9 ± 1.4 | 0 | 0 | 6.23 |
| compound_cylinder | 12 | Solved | GpPosterior | 10/10 | 10/10 | 31.0 ± 1.4 | 0 | 400 | 17.92 |
| compound_cylinder | 12 | Linearized | GpPosterior | 10/10 | 9/10 | 30.8 ± 1.3 | 0 | 0 | 16.61 |
| compound_cylinder | 16 | Solved | Perfect | 10/10 | 7/10 | 36.6 ± 1.0 | 0 | 400 | 6.82 |
| compound_cylinder | 16 | Linearized | Perfect | 10/10 | 10/10 | 36.8 ± 0.9 | 0 | 0 | 5.92 |
| compound_cylinder | 16 | Solved | GpPosterior | 10/10 | 9/10 | 36.6 ± 1.0 | 0 | 400 | 17.09 |
| compound_cylinder | 16 | Linearized | GpPosterior | 10/10 | 9/10 | 37.0 ± 0.9 | 0 | 0 | 15.38 |
| toy_constrained | 3 | Solved | Perfect | 8/10 | 7/10 | 21.5 ± 1.9 | 0 | 90 | 1.08 |
| toy_constrained | 3 | Linearized | Perfect | 10/10 | 9/10 | 20.1 ± 2.1 | 0 | 0 | 0.46 |
| toy_constrained | 3 | Solved | GpPosterior | 7/10 | 5/10 | 24.7 ± 9.8 | 0 | 90 | 1.23 |
| toy_constrained | 3 | Linearized | GpPosterior | 9/10 | 8/10 | 20.2 ± 2.0 | 0 | 0 | 0.55 |
| toy_constrained | 4 | Solved | Perfect | 9/10 | 5/10 | 22.0 ± 2.7 | 0 | 90 | 0.87 |
| toy_constrained | 4 | Linearized | Perfect | 8/10 | 5/10 | 20.4 ± 1.8 | 0 | 0 | 0.46 |
| toy_constrained | 4 | Solved | GpPosterior | 9/10 | 5/10 | 21.8 ± 2.9 | 0 | 90 | 0.88 |
| toy_constrained | 4 | Linearized | GpPosterior | 8/10 | 5/10 | 20.4 ± 1.6 | 0 | 0 | 0.49 |
| toy_constrained | 6 | Solved | Perfect | 9/10 | 8/10 | 21.0 ± 1.3 | 2 | 90 | 0.64 |
| toy_constrained | 6 | Linearized | Perfect | 9/10 | 9/10 | 20.0 ± 1.1 | 0 | 0 | 0.35 |
| toy_constrained | 6 | Solved | GpPosterior | 9/10 | 9/10 | 21.7 ± 1.9 | 2 | 90 | 0.71 |
| toy_constrained | 6 | Linearized | GpPosterior | 9/10 | 9/10 | 20.0 ± 1.1 | 0 | 0 | 0.43 |
| toy_constrained | 10 | Solved | Perfect | 10/10 | 8/10 | 28.2 ± 0.9 | 0 | 90 | 0.64 |
| toy_constrained | 10 | Linearized | Perfect | 10/10 | 6/10 | 27.8 ± 0.9 | 0 | 0 | 0.38 |
| toy_constrained | 10 | Solved | GpPosterior | 10/10 | 7/10 | 28.3 ± 0.9 | 0 | 90 | 0.68 |
| toy_constrained | 10 | Linearized | GpPosterior | 10/10 | 6/10 | 27.8 ± 0.9 | 0 | 0 | 0.43 |
| sellar_constrained | 4 | Solved | Perfect | 8/10 | 10/10 | 23.4 ± 1.5 | 0 | 293 | 2.90 |
| sellar_constrained | 4 | Linearized | Perfect | 8/10 | 10/10 | 23.9 ± 2.3 | 0 | 0 | 2.69 |
| sellar_constrained | 4 | Solved | GpPosterior | 9/10 | 10/10 | 23.8 ± 1.8 | 0 | 294 | 4.72 |
| sellar_constrained | 4 | Linearized | GpPosterior | 6/10 | 9/10 | 23.1 ± 1.9 | 0 | 0 | 4.42 |
| sellar_constrained | 5 | Solved | Perfect | 9/10 | 9/10 | 23.6 ± 2.2 | 97 | 295 | 2.86 |
| sellar_constrained | 5 | Linearized | Perfect | 8/10 | 10/10 | 24.2 ± 2.1 | 86 | 3 | 2.37 |
| sellar_constrained | 5 | Solved | GpPosterior | 8/10 | 10/10 | 23.8 ± 1.7 | 6 | 295 | 4.47 |
| sellar_constrained | 5 | Linearized | GpPosterior | 6/10 | 8/10 | 24.8 ± 2.7 | 67 | 4 | 4.12 |
| sellar_constrained | 8 | Solved | Perfect | 9/10 | 10/10 | 26.5 ± 2.5 | 0 | 293 | 2.77 |
| sellar_constrained | 8 | Linearized | Perfect | 9/10 | 10/10 | 26.3 ± 2.2 | 0 | 0 | 2.27 |
| sellar_constrained | 8 | Solved | GpPosterior | 10/10 | 10/10 | 26.0 ± 2.3 | 0 | 296 | 4.11 |
| sellar_constrained | 8 | Linearized | GpPosterior | 10/10 | 10/10 | 26.2 ± 2.2 | 0 | 0 | 4.12 |
| sellar_constrained | 12 | Solved | Perfect | 10/10 | 8/10 | 33.6 ± 2.4 | 13 | 293 | 2.73 |
| sellar_constrained | 12 | Linearized | Perfect | 10/10 | 9/10 | 33.7 ± 2.2 | 229 | 4 | 2.08 |
| sellar_constrained | 12 | Solved | GpPosterior | 10/10 | 9/10 | 34.5 ± 2.1 | 244 | 294 | 4.24 |
| sellar_constrained | 12 | Linearized | GpPosterior | 10/10 | 9/10 | 34.0 ± 2.0 | 242 | 4 | 4.19 |

### Findings

- **The perfect correlation of the papers is far from the GP posterior correlation.** Between `DoE_UQ` points, the median posterior correlation is between −0.09 and 0.24, and the 10–90% range typically spans −0.5 to 0.7, on every problem and for every DoE size. Only with 3 points per discipline on the toy problems do some pairs come close to 1.
- **It changes `P_min` only while the disciplinary uncertainty is large.** With 3 to 5 points per discipline, `P_min` moves by 0.09 to 0.34 in total variation, and the most probable minimizer changes in 1 or 2 seeds out of 10. From 6 points per discipline on the toy problems, 8 on modified Sellar and 12 on the compound cylinder, the distance is below 0.05. On constrained Sellar it stays between 0.1 and 0.18 up to 12 points.
- **The optimization results are equivalent.** Over the 20 problem and DoE size combinations, the four models converge in 183 to 187 runs out of 200, and the best point is optimal in 170 to 177 runs, with the same number of disciplinary evaluations. The largest differences are on constrained Sellar with 4 and 5 points per discipline, where the linearized random MDA with the GP correlation converges in 6/10 runs against 8 or 9/10 for the other models, its best point still being optimal in 9 and 8 runs. They are within the noise of 10 runs per row.
- **The GP correlation costs 1.5 to 2.5 times the run time on the Sellar problems and the compound cylinder** (little on the toy problems). With decorrelated points, the covariance of the objective over `DoE_UQ` has many more significant eigenvalues: on constrained Sellar, the KL keeps 17 to 23 modes against 5 to 7, so 3 times as many KL-GPs are trained and predicted. The random MDAs are then a minor part of the run, and the linearization no longer saves time.
- **The rank-one assumption across design points does not limit the method on these problems.** The perfect correlation keeps the KL small and the results unchanged, so it stays the default. The GP correlation remains available, as the more faithful model of the disciplinary uncertainty, for problems where the uncertainty stays large.

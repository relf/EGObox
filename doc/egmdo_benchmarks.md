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

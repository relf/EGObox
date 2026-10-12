# EGMDO Integration Plan for EGObox

## 1. Goal

Integrate the **Efficient Global Multidisciplinary Design Optimization (EGMDO)** algorithm into EGObox, using the original EGMDO paper (Dubreuil et al., 2020) as the algorithmic reference.

The first implementation will target **continuous, bounded, single-objective, unconstrained MDO problems**, which is the scope of the paper. It will use EGObox's existing numerical and Gaussian-process (GP) capabilities where appropriate, while keeping EGMDO's orchestration separate from the existing `Egor` optimization loop.

The initial integration should not require a dependency on OpenMDAO. Users will provide disciplinary evaluation functions; an optional OpenMDAO adapter can be added to make it convenient to wrap OpenMDAO models.

## 2. Design principles

- **Keep the EGMDO algorithm separate from `Egor`.** EGMDO manages several disciplinary surrogates and a non-Gaussian random objective; it should not be forced into a conventional single-objective infill-criterion interface.
- **Reuse EGObox infrastructure.** Reuse existing GP, design-of-experiments (DoE), linear algebra and optimization functionality where it matches the published algorithm.
- **Keep the core framework-independent.** The Rust core should not depend on OpenMDAO or its Python object model.
- **Use explicit discipline definitions.** The first version should not attempt to infer arbitrary disciplinary structure automatically from an MDO framework.
- **Separate disciplinary evaluations from full MDA evaluations.** These serve different purposes and should be represented explicitly in the API.
- **Follow the paper.** Implement the original EGMDO equations, approximation methods, and enrichment logic rather than substituting a generic EGO loop. Any deviation is documented in the specification.
- **Keep the hot loops in Rust.** Random MDAs are solved thousands of times per iteration; they must run entirely in Rust, without Python callbacks or the GIL.
- **Be reproducible.** Every source of randomness (DoEs, Monte Carlo samples, multistart) is seedable.

## 3. Algorithm summary

The algorithm is specified equation by equation in `doc/egmdo_spec.md`. In short:

1. **Disciplinary GPs.** Each scalar coupling output `y_i = f_i(z, y_c(i))` is modelled by a GP trained on a DoE over `Z × C(i)`, where `C(i)` is a user-provided initial guess of the coupling variables' range.
2. **Simplified random MDA.** Each disciplinary GP is replaced by a perfectly correlated random field `μ_i(z, y) + σ_i(z, y) ξ_i`, with one standard normal `ξ_i` per GP. For a sample of `Ξ = (ξ_1, …)`, the MDA is a deterministic system solved by non-linear Jacobi iterations, pointwise in `z`.
3. **PCE of the random objective.** At each point of a design-space DoE (`DoE_UQ`), the objective `f_obj(z, Y*(z, Ξ))` is approximated by a Hermite polynomial chaos expansion in `Ξ`, fitted by least squares on Monte Carlo MDA solutions.
4. **KL-GP random field.** A Karhunen–Loève decomposition of the PCE coefficient vectors over `DoE_UQ` is computed. The mean and the retained eigenvectors are interpolated in `z` by GPs (KL-GPs), giving a continuous non-Gaussian random field `Ỹ_obj(z, Ξ, η)`.
5. **Step 1: where to reduce uncertainty.** An expected improvement over both uncertainty sources, estimated by Monte Carlo, selects `z_new`, which is added to `DoE_UQ`.
6. **Step 2: disciplinary enrichment (Algorithm 1).** The probability `P_min` that each `DoE_UQ` point is the minimizer, and the coefficient of variation of the objective there, decide whether disciplinary GPs must be enriched. Enrichment evaluates **all** disciplines at `(z, y_c)`, where `y_c` solves the MDA on the GP means.
7. **Stopping.** The paper stops on an iteration budget `n_max`. The final result is the mean KL-GP model (19), from which the distributions of the minimum position `Ẑ*` and value `Ŷ_min` are estimated by Monte Carlo.
8. **No true MDA in the loop.** Only disciplinary evaluations are used. True MDA runs are for validation and benchmarking.

Discipline selection by sensitivity analysis, other EI criteria (e.g. WB2s), other kernels and constraints are listed by the paper as perspectives; they are extensions here (Stage E).

## 4. Proposed architecture

### Rust core

Create a new crate `crates/egmdo` (`egobox-egmdo`). Do not enlarge `egobox-ego`. The workspace picks up `crates/*` automatically.

Modules (indicative):

- `problem`: design space, disciplines, coupling-variable mapping and initial coupling bounds;
- `mda`: non-linear Jacobi fixed-point solver for one sample of `Ξ`, used for random MDAs and mean MDAs (and, in tests, for true MDAs);
- `pce`: multivariate Hermite polynomial basis of total degree `d`, least-squares fit, mean and variance;
- `kl`: Karhunen–Loève decomposition of PCE coefficient vectors and KL-GP interpolation;
- `criteria`: Monte Carlo expected improvement (16), `P_min` (18), coefficient of variation;
- `solver`: Algorithm 1, Algorithm 2, ask/tell state machine, final estimation of `Ẑ*` and `Ŷ_min`.

Reuse:

- `egobox-gp`: `GaussianProcess` with constant mean and squared exponential kernel; `predict` and `predict_var` give `μ_i` and `σ_i`. No new GP capability is required.
- `egobox-doe`: seeded `Lhs` for `DoE_fi` and `DoE_UQ`.
- `linfa-linalg`: symmetric eigendecomposition for the KL step (already used by `egobox-gp`).
- `egobox-ego`: the COBYLA multistart optimizer used for EI maximization. The `optimizers` module is currently private (`crates/ego/src/lib.rs`), so reusing it means making it public. That is an addition to the public API, not a breaking change. Alternatively, `egmdo` can call the `cobyla` crate directly.

New numerical code (not available in EGObox): Hermite PCE with least-squares regression, KL decomposition of the PCE vector, Monte Carlo EI.

### Ask/tell orchestration

The Rust core owns the algorithm state and runs all surrogate computations. It does not call disciplines itself. Instead, it exposes an ask/tell interface:

- `ask()` returns the next batch of disciplinary evaluations to perform. During initialization it is the initial `DoE_fi` of every discipline; during enrichment it is one point `(z, y_c(i))` per discipline;
- the caller evaluates the true disciplines and passes the outputs back with `tell(...)`;
- `minimize()` is a convenience wrapper that runs this loop with user-provided callbacks.

This follows the existing `suggest` pattern (`crates/ego/src/solver/egor_service.rs`, `python/src/egor.rs`). Python callbacks never run inside Rust loops, and framework adapters (OpenMDAO, GEMSEO) only have to answer requests.

### Python API

Expose a user-facing API under `egobox.egmdo`, with an optional adapter namespace such as `egobox.egmdo.openmdao`.

The initial API accepts explicit discipline specifications, Python evaluation functions and a cheap objective function, and exposes both `minimize()` and ask/tell.

### Optional OpenMDAO adapter

The adapter should be a thin layer that answers enrichment requests by running individual OpenMDAO components, and runs the full model for validation. The EGMDO core must remain independent of OpenMDAO.

Do not require automatic introspection of arbitrary OpenMDAO models in the first version.

## 5. Discipline evaluation model

A discipline `f_i(z, y_c(i))` takes the design variables (or a subset) and the coupling variables it receives, and returns one or more coupling outputs.

Represent each discipline explicitly with at least:

- a stable name or identifier;
- the design variables it uses;
- the coupling variables it receives and the ones it produces, with their dimensions;
- **initial bounds `C(i)` for the coupling variables it receives**. They are required to sample the initial disciplinary DoE over `Z × C(i)`. They are only an initial guess: enrichment points may fall outside them;
- an evaluation callback.

EGObox GPs have a single scalar output. **One GP is trained per scalar coupling output**, and each has its own random variable `ξ`. The stochastic dimension `n_ξ` is therefore the total number of scalar coupling outputs; with the paper's scalar couplings it equals `n_d`. The PCE size grows as `(n_ξ + d)! / (n_ξ! d!)`, which limits the practical number of coupling outputs (documented as a limitation).

Random samples can push coupling variables outside the physical domain of a discipline (e.g. `sqrt(y1)` in Sellar). Test problems must be written to tolerate this, and MDA failures are counted and reported.

### Objective

The objective `f_obj(z, y)` is a **cheap, explicit function** of the design and coupling variables, as in the paper. It is evaluated on the random couplings and is not modelled by a GP. An objective computed by an expensive discipline is a later extension.

An illustrative Python API (not implemented yet):

```python
from egobox.egmdo import Egmdo, Discipline

d1 = Discipline(
    name="d1",
    fun=compute_y1,                # (z, y2) -> y1
    inputs=["z", "y2"],
    outputs=["y1"],
    coupling_bounds={"y2": [1.0, 50.0]},
)

d2 = Discipline(
    name="d2",
    fun=compute_y2,                # (z, y1) -> y2
    inputs=["z", "y1"],
    outputs=["y2"],
    coupling_bounds={"y1": [-5.0, 24.0]},
)

optimizer = Egmdo(
    disciplines=[d1, d2],
    objective=compute_objective,   # cheap function of (z, y1, y2)
    xlimits=[[0.0, 10.0], [-10.0, 10.0], [0.0, 10.0]],
    seed=42,
)

# Convenience loop
result = optimizer.minimize(max_iters=10)

# Or ask/tell
request = optimizer.ask()          # disciplinary evaluations to perform
outputs = evaluate(request)        # caller-side evaluation
optimizer.tell(request, outputs)
```

This snippet is a design sketch, not a commitment to the final constructor signatures.

### Three distinct evaluation modes

1. **True disciplinary evaluation**
   - Calls the original discipline function at `(z, y_c(i))`.
   - Supplies training data for the corresponding GPs.
   - The only expensive evaluation inside the optimization loop.

2. **True MDA evaluation**
   - Solves the coupled system at a design point with the true disciplines.
   - Used for validation and benchmarking only.

3. **Surrogate MDA evaluation**, in two forms:
   - **random MDA**: the GPs are replaced by `μ_i + σ_i ξ_i` for one sample of `Ξ`; used to build the PCE at each `DoE_UQ` point;
   - **mean MDA**: the GPs are replaced by their means; used to compute the coupling values `y_c` of an enrichment point.

   Non-converged surrogate MDAs are counted, excluded as specified in `doc/egmdo_spec.md`, and reported.

### Computational cost

- Building the PCE at one design point costs `n_MC` random MDAs (100 in the paper). Each disciplinary enrichment rebuilds the PCE at all `n_UQ` points (20 to 35 in the paper), i.e. a few thousand Jacobi solves, each requiring GP predictions.
- EI evaluations are cheap: KL-GP predictions at `z` plus polynomial evaluations. They are estimated with fixed Monte Carlo samples (common random numbers), so that the criterion is deterministic during its optimization.
- The final estimation of `Ẑ*` and `Ŷ_min` requires `n_sim` global minimizations of the cheap model (19).

These loops run in Rust and are parallelized with rayon over design points and samples.

## 6. Algorithm implementation stages

### Stage A — Technical specification

Write `doc/egmdo_spec.md` from the paper: equations, algorithms, default settings, mapping to Rust modules, test cases and reference results, and the decisions the paper leaves open.

### Stage B — Core algorithm

Scope:

- continuous bounded design variables;
- one scalar, cheap objective;
- two or more explicitly declared disciplines with scalar or vector coupling outputs;
- one GP per scalar coupling output;
- random MDA by non-linear Jacobi;
- Hermite PCE, KL decomposition, KL-GPs;
- Monte Carlo EI, Algorithm 1 and Algorithm 2;
- final estimation of `Ẑ*` and `Ŷ_min` distributions;
- ask/tell interface and `minimize()` wrapper;
- per-discipline evaluation accounting.

Defer constraints, mixed-integer variables, multi-objective optimization, and parallel true evaluations.

### Stage C — Benchmarks

Use the paper's test cases, which come with reference results:

- **1-D toy problem** (section 2): small enough for unit tests of each step, with the paper's figures as qualitative references;
- **modified Sellar problem** (section 4.1, unconstrained): reference `z* ≈ (0, 2.634, 0)`, `f* ≈ −2.808`, local minimum at `(0, −2.595, 0)`; the paper reports 88% convergence and 13 evaluations per discipline on average over 100 runs (15 iterations);
- **compound cylinder** (section 4.2): reference `z* ≈ (7, 9, 12, 6.37e−3)`, `f* ≈ 2.33e−2`.

Tests should verify:

- variable mapping and callback dimensions;
- convergence of true, random and mean MDAs;
- PCE moments against Monte Carlo moments;
- KL truncation and KL-GP interpolation at `DoE_UQ` points;
- `P_min`, coefficient of variation and EI on small hand-checked cases (EI is zero at `DoE_UQ` points);
- enrichment logic of Algorithm 1;
- reproducibility for a given seed;
- reference results within documented tolerances.

### Stage D — Add the optional OpenMDAO adapter

Provide a documented way to answer EGMDO enrichment requests with OpenMDAO disciplinary evaluations.

Requirements:

- explicitly map named OpenMDAO variables to discipline inputs and outputs;
- support individual discipline evaluation where the model permits it;
- support a full coupled model run for validation and true MDA evaluations;
- keep OpenMDAO imports and dependencies out of the Rust core;
- report clear errors for missing variables, dimension mismatches, failed component runs, and failed MDA convergence.

Do not assume `Problem.run_model()` alone provides the disciplinary observations required by EGMDO. Full-model execution and individual component execution must be handled as separate capabilities.

### Stage E — Profile and extend

After correctness is established:

- profile GP fitting, random MDAs, PCE, KL, EI optimization and true engineering evaluations separately;
- investigate incremental GP updates if they provide a measurable benefit;
- add parallel discipline evaluations;
- enrich only the disciplines responsible for most of the variance of the minimum (sensitivity analysis, a perspective of the paper);
- try other EI-type criteria (e.g. WB2s) and other kernels, which the paper suggests;
- add a stopping criterion based on the EI value;
- add constraints, validated on the classic constrained Sellar problem;
- consider objectives computed by expensive disciplines;
- consider more general variable types and additional framework adapters.

## 7. Initial scope and deferred features

| Feature | Initial version |
|---|---|
| Design variables | Continuous and bounded |
| Objective | One scalar, cheap, explicit objective |
| Disciplines | Two or more |
| Discipline outputs | One or more scalar coupling outputs |
| Coupling | Explicit mappings, initial coupling bounds |
| Surrogates | One GP per scalar coupling output |
| Uncertainty model | Simplified GP, one standard normal per GP |
| Random objective | Hermite PCE + KL decomposition + KL-GPs |
| Enrichment | All disciplines at each enrichment point |
| Stopping | Iteration budget |
| Orchestration | Ask/tell, plus `minimize()` wrapper |
| True evaluations | Serial |
| Surrogate computations | Parallel, in Rust |
| Constraints | Deferred |
| Discipline selection | Deferred |
| Multi-objective optimization | Deferred |
| Mixed-integer variables | Deferred |
| OpenMDAO dependency in core | None |
| OpenMDAO adapter | Optional, after core interface is stable |

## 8. API and implementation decisions to settle

Before finalizing the public API, decide:

1. How input and output variables are represented, including names, dimensions, and vector-valued outputs.
2. How design and coupling variables are mapped to each discipline's input vector, and how initial coupling bounds are declared.
3. The exact ask/tell contract: request and response types, how pending requests are tracked, and how failed evaluations are reported.
4. How Python exceptions in callbacks reach the user (follow the existing pattern in `python/src/errors.rs`).
5. How non-converged random MDAs are handled in PCE regression.
6. Which internal `egobox-ego` utilities are made public, or whether `egmdo` depends on `cobyla` directly.

## 9. Acceptance criteria

The initial implementation is ready for a first release when:

- the algorithm matches the paper's formulation, as documented in `doc/egmdo_spec.md`;
- the 1-D toy, modified Sellar and compound cylinder problems run end to end;
- on modified Sellar, results over many seeds are comparable to the paper's (convergence rate and number of evaluations per discipline), with the comparison documented;
- EGMDO needs fewer disciplinary evaluations than `Egor` applied to the true MDA objective (median and spread over fixed seeds);
- runs are reproducible for a given seed;
- tests cover variable mapping, MDA, PCE, KL, KL-GPs, EI, `P_min`, enrichment and stopping behavior;
- Python users can provide discipline callbacks, or use ask/tell, without installing OpenMDAO;
- the optional OpenMDAO integration does not leak framework dependencies into the Rust core;
- API and documentation clearly distinguish disciplinary evaluations, true MDA evaluations, and surrogate MDA evaluations.

## 10. References

- S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, *Towards an efficient global multidisciplinary design optimization algorithm*, Structural and Multidisciplinary Optimization 62:1739–1765, 2020, [doi:10.1007/s00158-020-02514-6](https://doi.org/10.1007/s00158-020-02514-6)
- S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, J. Mas Colomer, *Extreme value oriented random field discretization based on an hybrid polynomial chaos expansion – kriging approach*, Computer Methods in Applied Mechanics and Engineering 332:540–571, 2018 (KL-GP random field and EI estimation)
- OpenMDAO documentation: [https://openmdao.org/](https://openmdao.org/)
- GEMSEO (possible future adapter, not the first target): [https://gemseo.org/](https://gemseo.org/)

## 11. Immediate next steps

1. Write `doc/egmdo_spec.md`.
2. Create `crates/egmdo` with the problem definition, the MDA solver and the paper's test problems.
3. Implement the PCE, KL and KL-GP random field.
4. Implement the criteria and Algorithms 1 and 2.
5. Reproduce the paper's results and compare with `Egor` on the true MDA objective.
6. Add the Python bindings, then the OpenMDAO adapter.

## 12. Delivery

### Branching workflow

- `egmdo` is the integration branch, created from `master`. Nothing is merged into `master` until the acceptance criteria of section 9 are met.
- Each PR has its own branch created from `egmdo` and targets `egmdo`. Branches are named `egmdo-<topic>`. (`egmdo/<topic>` is not possible because Git cannot have both a branch `egmdo` and branches under `egmdo/`.)
- `master` is merged into `egmdo` periodically (merge, not rebase, since `egmdo` is shared) to keep it up to date with the rest of EGObox.
- The final integration is a single PR from `egmdo` to `master`.

### PR sequence

| # | Branch | Content | Stage |
|---|---|---|---|
| 1 | `egmdo-plan` | This integration plan | — |
| 2 | `egmdo-spec` | `doc/egmdo_spec.md` | A |
| 3 | `egmdo-mda` | `crates/egmdo` skeleton: problem definition, disciplinary GPs, MDA solver (true, random, mean), test problems | B |
| 4 | `egmdo-uq` | Hermite PCE, KL decomposition, KL-GP random field | B |
| 5 | `egmdo-algorithm` | EI, `P_min`, Algorithms 1 and 2, final estimation, ask/tell and `minimize()` | B |
| 6 | `egmdo-benchmarks` | Paper test cases at scale, robustness study, comparison with `Egor` on the true MDA | C |
| 7 | `egmdo-python` | `egobox.egmdo` Python bindings, type stubs, tests and documentation | B/C |
| 8 | `egmdo-openmdao` | Optional `egobox.egmdo.openmdao` adapter | D |

Stage E items get their own PRs once the first release is merged into `master`.

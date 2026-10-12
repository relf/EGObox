# EGMDO Integration Plan for EGObox

## 1. Goal

Integrate the **Efficient Global Multidisciplinary Design Optimization (EGMDO)** algorithm into EGObox, using the original EGMDO paper as the algorithmic reference.

The first implementation will target **continuous, bounded, single-objective, unconstrained MDO problems**. It will use EGObox's existing numerical and Gaussian-process (GP) capabilities where appropriate, while keeping EGMDO's orchestration separate from the existing `Egor` optimization loop.

The initial integration should not require a dependency on OpenMDAO. Users will provide disciplinary evaluation functions; an optional OpenMDAO adapter can be added to make it convenient to wrap OpenMDAO models.

## 2. Design principles

- **Keep the EGMDO algorithm separate from `Egor`.** EGMDO manages multiple disciplinary surrogates and uncertainty propagated through a coupled system; it should not be forced into a conventional single-objective infill-criterion interface.
- **Reuse EGObox infrastructure.** Reuse existing GP, design-of-experiments (DoE), and numerical optimization functionality where it matches the published algorithm.
- **Keep the core framework-independent.** The Rust core should not depend on OpenMDAO or its Python object model.
- **Use explicit discipline definitions.** The first version should not attempt to infer arbitrary disciplinary structure automatically from an MDO framework.
- **Separate disciplinary evaluations from full MDA evaluations.** These serve different purposes and should be represented explicitly in the API.
- **Follow the paper.** Implement the original EGMDO equations, approximation methods, and enrichment logic rather than substituting a generic EGO loop.
- **Keep the hot loop in Rust.** Surrogate MDAs are solved many times per iteration; they must run entirely in Rust, without Python callbacks or the GIL.
- **Be reproducible.** Every source of randomness (DoE, GP realizations, acquisition multistart) is seedable.

## 3. Working hypothesis of the algorithm

The following summary is written from memory of the paper and is meant to steer Stage A. **Every item must be verified against the paper** before it is treated as specified.

1. **Disciplinary surrogates.** Each scalar coupling output of each discipline is modelled by a GP. Its inputs are the shared design variables, the discipline's local variables, and the coupling variables it receives.
2. **Random MDA.** Replacing the disciplines by GP realizations makes the coupling variables, and therefore the objective, random functions of the design variables. Each realization is solved with a fixed-point MDA.
3. **Functional GP realizations.** Since the MDA decides where each realization is evaluated, realizations must be functions that can be evaluated at any point, built from a fixed random vector (the paper uses a truncated expansion of the GPs; check which one).
4. **Random objective approximation.** The mean and variance of the random objective are estimated with the paper's method (sampling and/or polynomial-chaos-type approximation over the expansion's random variables).
5. **Design point selection.** An expected-improvement-type criterion on this approximation selects the next design point `z*`.
6. **Discipline selection.** The discipline to enrich is the one whose GPs contribute most to the objective's variance at `z*` (Sobol'-type sensitivity indices).
7. **Enrichment.** Only the selected discipline is evaluated, at `z*` and the coupling values given by the surrogate MDA at `z*` (the mean, or as specified in the paper).
8. **Stopping.** Evaluation budget, an EI threshold, and/or a threshold on the objective's variance at the current optimum.
9. **True MDA evaluations.** The algorithm enriches with *disciplinary* evaluations only. Full MDA runs on the true disciplines are used for final validation and benchmarking, not inside the loop.

## 4. Proposed architecture

### Rust core

Create a new crate `crates/egmdo` (`egobox-egmdo`). Do not enlarge `egobox-ego`. The workspace picks up `crates/*` automatically.

Responsibilities:

- manage the design space and disciplinary models;
- manage training data and GP models for each coupling output;
- generate functional GP realizations and solve the surrogate MDA for each one;
- implement the random-objective approximation required by EGMDO;
- implement the paper's enrichment criterion and acquisition optimization;
- choose which discipline to enrich and at which point, as specified by the algorithm;
- handle stopping conditions and return optimization results.

Reuse and required changes:

- `egobox-gp`: GP training and prediction. **Missing:** functional realizations (see Stage A′). The current `sample`/`sample_eig`/`sample_chol` (`crates/gp/src/algorithm.rs`) draw joint values only at a set of points chosen in advance and use an unseeded RNG, so they cannot be used inside a fixed-point MDA.
- `egobox-doe`: initial designs and sampling.
- `egobox-ego`: numerical optimizers (multistart, SLSQP, Cobyla). The `optimizers` module is currently private (`crates/ego/src/lib.rs`), so reusing it means making it public. That is an addition to the public API, not a breaking change.

### Ask/tell orchestration

The Rust core owns the algorithm state and runs the surrogate MDAs. It does not call disciplines itself. Instead, it exposes an ask/tell interface:

- `ask()` returns enrichment requests of the form `(discipline_id, input_vector)`;
- the caller evaluates the true discipline and passes the outputs back with `tell(discipline_id, input_vector, outputs)`;
- `minimize()` is a convenience wrapper that runs this loop with user-provided callbacks.

This follows the existing `suggest` pattern (`crates/ego/src/solver/egor_service.rs`, `python/src/egor.rs`). Python callbacks never run inside Rust loops, and framework adapters (OpenMDAO, GEMSEO) only have to answer requests.

### Python API

Expose a user-facing API under `egobox.egmdo`, with an optional adapter namespace such as `egobox.egmdo.openmdao`.

The initial API should accept explicit discipline specifications and Python evaluation functions, and expose both `minimize()` and ask/tell. Keep the API illustrative until the algorithm's exact data requirements have been mapped from the paper.

### Optional OpenMDAO adapter

The adapter should be a thin layer that answers enrichment requests by running individual OpenMDAO components, and runs the full model for validation. The EGMDO core must remain independent of OpenMDAO.

Do not require automatic introspection of arbitrary OpenMDAO models in the first version.

## 5. Discipline evaluation model

A discipline is **not necessarily a scalar objective function**. It is an evaluation function that accepts its declared inputs and returns one or more outputs. Some outputs may be coupling variables consumed by other disciplines.

Represent each discipline explicitly with at least:

- a stable name or identifier;
- input variables and their dimensions;
- output variables and their dimensions;
- bounds or domain information for the inputs where applicable, **including coupling variables** (surrogate realizations can leave the physical domain, e.g. `sqrt(y1)` in Sellar);
- an evaluation callback;
- the mapping between its inputs and global design/coupling variables.

EGObox GPs have a single scalar output. **One GP is trained per scalar coupling output component**, not one per discipline. A discipline with a vector coupling output therefore owns several GPs, all trained on the same evaluations.

### Objective

In the first version, the objective is assumed to be a **cheap, explicit function** of the design and coupling variables. It is evaluated on the (random) couplings and is not itself modelled by a GP. An objective that is the output of an expensive discipline is a later extension; it then needs its own GP like any other discipline output.

An illustrative Python API (not implemented yet):

```python
from egobox.egmdo import Egmdo, Discipline

discipline_1 = Discipline(
    name="discipline_1",
    fun=compute_discipline_1,
    # Explicit input-variable mapping and bounds
)

discipline_2 = Discipline(
    name="discipline_2",
    fun=compute_discipline_2,
    # Explicit input-variable mapping and bounds
)

optimizer = Egmdo(
    disciplines=[discipline_1, discipline_2],
    objective=compute_objective,  # cheap function of (design, couplings)
    # Global design variables and bounds, seed
)

# Convenience loop
result = optimizer.minimize()

# Or ask/tell
request = optimizer.ask()          # (discipline name, input vector)
outputs = evaluate(request)        # caller-side evaluation
optimizer.tell(request, outputs)
```

This snippet is a design sketch, not a commitment to the final constructor signatures. In particular, each discipline's inputs can include global design variables, local variables, and coupling variables.

### Three distinct evaluation modes

1. **True disciplinary evaluation**
   - Calls the original discipline function with the specified inputs.
   - Returns the discipline's outputs.
   - Supplies training observations and new data for the corresponding GPs.
   - May be expensive. This is the only expensive evaluation inside the optimization loop.

2. **True MDA evaluation**
   - Solves the coupled system at a specified global design point using the true disciplinary functions or the external framework.
   - Evaluates the true system objective and any quantities required for validation.
   - Used for initial checks, final validation and benchmarking, not for enrichment (to be confirmed in Stage A).

3. **Surrogate MDA evaluation**
   - Solves the coupled system for one functional GP realization at a time.
   - Supports the random-objective approximation and the enrichment calculations.
   - Must preserve the uncertainty treatment specified in the EGMDO paper; it must not be treated as an ordinary GP of the final objective without validation.
   - Must report non-converged realizations explicitly: count them, exclude or handle them as the paper specifies, and expose the count in the results and logs.

The precise role and frequency of each evaluation mode must be derived from the original algorithm before implementation.

### Computational cost

Each iteration requires roughly `n_realizations × n_acquisition_candidates` surrogate fixed-point solves, plus GP retraining for the enriched discipline. The surrogate MDA loop must:

- run entirely in Rust, without holding the GIL;
- be parallelized over realizations and candidate points (rayon);
- use a configurable fixed-point solver (Gauss–Seidel or Jacobi, with optional relaxation, tolerance and iteration limit).

## 6. Algorithm implementation stages

### Stage A — Translate the paper into a technical specification

Before coding, write `doc/egmdo_spec.md`:

- document the algorithm's mathematical steps and equations;
- confirm or correct each item of the working hypothesis in section 3;
- identify the GP realization method, the random-objective approximation, and the uncertainty-propagation method;
- document the acquisition/enrichment criterion and how new evaluations are assigned to disciplines;
- record all assumptions, required distributions, and numerical approximations;
- for each equation, give the Rust function or reusable EGObox capability it maps to, and the test that checks it;
- identify capabilities EGObox does not yet provide.

Do not begin with a generic EI loop and assume it reproduces EGMDO.

### Stage A′ — GP functional realizations in `egobox-gp`

Add a way to draw a GP realization as a function that can be evaluated at arbitrary points, using the method specified by the paper (e.g. a truncated Karhunen–Loève/Nyström expansion), with a seedable RNG.

Tests:

- the empirical mean and variance of many realizations match `predict` and `predict_var`;
- a realization interpolates (up to noise) the training data;
- draws are reproducible for a given seed;
- evaluating a realization at the same point twice gives the same value.

This capability is independent of EGMDO and could also serve Thompson-sampling-type criteria in `Egor`.

### Stage B — Implement the minimal unconstrained algorithm

Scope:

- continuous bounded design variables;
- one scalar, cheap objective;
- two or more explicitly declared disciplines;
- explicit coupling-variable mappings;
- one GP per scalar coupling output;
- serial true evaluations, parallel surrogate MDAs;
- original EGMDO uncertainty-propagation and enrichment method;
- ask/tell interface and `minimize()` wrapper;
- basic stopping criteria and per-discipline evaluation accounting.

Defer constraints, mixed-integer variables, multi-objective optimization, and parallel true evaluations until the basic algorithm is validated.

### Stage C — Build the benchmarks

Use the standard two-discipline Sellar problem as the first end-to-end benchmark.

The classic Sellar problem is **constrained** (`g1 = 1 − y1/3.16 ≤ 0`, `g2 = y2/24 − 1 ≤ 0`); its well-known optimum `f ≈ 3.18339` at `z = (1.9776, 0)`, `x = 0` is a constrained one and does not apply to the first, unconstrained version. Therefore:

- Stage C runs the **unconstrained** Sellar problem;
- its reference optimum is computed by a direct optimizer on the true MDA and recorded with the tolerances used;
- the constrained Sellar problem becomes the reference benchmark of the constraints extension (Stage E).

After Sellar, add a problem with more couplings or a higher dimension (e.g. the analytical cases of the paper). Defer SSBJ-type problems.

Tests should verify:

- discipline callbacks receive inputs of the expected dimensions;
- discipline outputs are mapped to the correct coupling variables;
- the true and surrogate MDAs converge, and non-converged realizations are reported;
- the initial designs and GP training data are constructed correctly;
- GP predictions and uncertainty propagation behave sensibly;
- enrichment selects the intended points and discipline evaluations;
- the objective improves and the algorithm terminates;
- results are consistent with the reference optimum within justified tolerances.

Where possible, test the realization, uncertainty-propagation and enrichment calculations independently of the full optimization loop.

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

- profile GP fitting, GP realizations, surrogate MDAs, random-objective approximation, acquisition optimization, and true engineering evaluations separately;
- cache repeated evaluations when safe;
- investigate incremental GP updates if they provide a measurable benefit;
- add parallel discipline evaluations only where dependency relationships permit;
- add constraints through the relevant constrained EGMDO extension, validated on the constrained Sellar problem;
- consider objectives computed by expensive disciplines;
- consider more general variable types and additional framework adapters.

## 7. Initial scope and deferred features

| Feature | Initial version |
|---|---|
| Design variables | Continuous and bounded |
| Objective | One scalar, cheap, explicit objective |
| Disciplines | Two or more |
| Discipline outputs | One or more outputs |
| Coupling | Explicit mappings |
| Surrogates | One GP per scalar coupling output |
| GP realizations | Functional, seedable (new in `egobox-gp`) |
| Orchestration | Ask/tell, plus `minimize()` wrapper |
| True evaluations | Serial |
| Surrogate MDAs | Parallel, in Rust |
| Constraints | Deferred |
| Multi-objective optimization | Deferred |
| Mixed-integer variables | Deferred |
| OpenMDAO dependency in core | None |
| OpenMDAO adapter | Optional, after core interface is stable |

## 8. API and implementation decisions to settle

Before finalizing the public API, decide:

1. How input and output variables are represented, including names, dimensions, and vector-valued outputs.
2. How global design, local, and coupling variables are mapped to each discipline's input vector, and how coupling-variable bounds are declared.
3. The exact ask/tell contract: request and response types, how pending requests are tracked, and how failed evaluations are reported.
4. How Python exceptions in callbacks reach the user (follow the existing pattern in `python/src/errors.rs`).
5. Which evaluations the published EGMDO algorithm requires at each iteration.
6. How to represent the MDA solver, its settings, and convergence failures (true and surrogate).
7. Which internal `egobox-ego` utilities need to be made public, as additions that do not break existing APIs.

## 9. Acceptance criteria

The initial implementation is ready for a first release when:

- the algorithm matches the original paper's formulation, as documented in `doc/egmdo_spec.md`;
- the unconstrained Sellar benchmark and one larger benchmark run end to end;
- the reference results are reproduced within documented tolerances;
- over several fixed-seed runs, EGMDO reaches the reference optimum with fewer **disciplinary evaluations** than `Egor` applied to the true MDA objective (median and spread reported);
- runs are reproducible for a given seed;
- tests cover variable mapping, GP training data, GP realizations, uncertainty propagation, enrichment, and stopping behavior;
- Python users can provide discipline callbacks, or use ask/tell, without installing OpenMDAO;
- the optional OpenMDAO integration does not leak framework dependencies into the Rust core;
- API and documentation clearly distinguish disciplinary evaluations, true MDA evaluations, and surrogate MDA evaluations.

## 10. References

- Original EGMDO paper: S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, [Towards an Efficient Global Multidisciplinary Design Optimization algorithm](https://www.researchgate.net/publication/343657447_Towards_an_Efficient_Global_Multidisciplinary_Design_Optimization_algorithm), Structural and Multidisciplinary Optimization, 2020
- OpenMDAO documentation: [https://openmdao.org/](https://openmdao.org/)
- GEMSEO (possible future adapter, not the first target): [https://gemseo.org/](https://gemseo.org/)

## 11. Immediate next steps

1. Read the original EGMDO paper, confirm or correct section 3, and write `doc/egmdo_spec.md`.
2. Implement and test functional GP realizations in `egobox-gp` (Stage A′).
3. Specify the discipline, variable-mapping, ask/tell and callback types.
4. Create `crates/egmdo`, implement the mathematical core, and test it on the unconstrained Sellar problem.
5. Compare against `Egor` on the true MDA objective.
6. Add the OpenMDAO adapter after the core evaluation contract is stable.

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
| 2 | `egmdo-spec` | `doc/egmdo_spec.md`: equations, confirmed algorithm, mapping to Rust functions and tests | A |
| 3 | `egmdo-gp-realizations` | Functional, seedable GP realizations in `egobox-gp` | A′ |
| 4 | `egmdo-mda` | `crates/egmdo` skeleton: variables, disciplines, coupling mappings, fixed-point MDA solver (true and surrogate), Sellar problem | B |
| 5 | `egmdo-algorithm` | Random-objective approximation, acquisition, discipline selection, enrichment, stopping, ask/tell and `minimize()` | B |
| 6 | `egmdo-benchmarks` | Unconstrained Sellar reference, second benchmark, comparison with `Egor` on the true MDA | C |
| 7 | `egmdo-python` | `egobox.egmdo` Python bindings, type stubs, tests and documentation | B/C |
| 8 | `egmdo-openmdao` | Optional `egobox.egmdo.openmdao` adapter | D |

PRs 2 and 3 are independent and can proceed in parallel. Stage E items get their own PRs once the first release is merged into `master`.

# EGMDO Integration Plan for EGObox

## 1. Goal

Integrate the **Efficient Global Multidisciplinary Design Optimization (EGMDO)** algorithm into EGObox, together with its constrained extension **C-EGMDO**, using the two papers as algorithmic references:

- EGMDO (Dubreuil et al., 2020): unconstrained problems;
- C-EGMDO (Cardoso et al., 2024): inequality and equality constraints, the WB2 acquisition criterion, and enrichment of a single discipline selected by Sobol sensitivity indices.

The first release targets **continuous, bounded, single-objective MDO problems, with or without constraints**. It is built in two steps: the unconstrained EGMDO first, then the C-EGMDO extensions on top of it. The problem definition and data structures are designed for constraints from the start. It will use EGObox's existing numerical and Gaussian-process (GP) capabilities where appropriate, while keeping EGMDO's orchestration separate from the existing `Egor` optimization loop.

EGObox does not depend on any MDO framework. Users provide disciplinary evaluation functions, or answer the ask/tell requests themselves. Adapters to MDO frameworks such as OpenMDAO or GEMSEO are not part of EGObox; they can be built outside it on top of ask/tell.

## 2. Design principles

- **Keep the EGMDO algorithm separate from `Egor`.** EGMDO manages several disciplinary surrogates and a non-Gaussian random objective; it should not be forced into a conventional single-objective infill-criterion interface.
- **Reuse EGObox infrastructure.** Reuse existing GP, design-of-experiments (DoE), linear algebra and optimization functionality where it matches the published algorithm.
- **Stay framework-independent.** Neither the Rust core nor the Python package depends on an MDO framework.
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

### C-EGMDO extensions

C-EGMDO (Cardoso et al., 2024, specified in part II of `doc/egmdo_spec.md`) keeps the structure above and changes the following steps:

1. **Random constraints.** Constraints `g_m(z, y) ≤ 0` and `h_r(z, y) = 0` are cheap functions of the design and coupling variables, like the objective. They become random through the couplings, and get a PCE at each `DoE_UQ` point, computed from the same random MDAs as the objective.
2. **Constraint models.** Constraints are not modelled by KL-GPs. Instead, the **mean** of each constraint PCE is interpolated in `z` by a GP, as in SEGO.
3. **Step 1: constrained acquisition.** The constrained EI (CEI) improves on `f'_min(Ξ)`, the minimum of the objective PCE over the `DoE_UQ` points that are feasible for that sample. If no point is feasible, the least-violating point is used instead. CEI, or the less multimodal **WB2** criterion (`CEI − m_0(z)`), is maximized under the GP-mean constraints.
4. **Step 2: constrained enrichment.** `P_min` is replaced by `P_sol`, the probability of being the feasible minimum. If `P_sol` is below a threshold `ε_P` everywhere, it falls back to the probability of being the least-violating point. The coefficient of variation becomes the maximum over the objective and the constraints.
5. **Single-discipline enrichment.** Instead of all disciplines, only the discipline with the highest first-order Sobol index is enriched. The indices are computed analytically from the PCE coefficients of the most uncertain quantity. This option applies to unconstrained problems too.
6. **Stopping.** Iteration budget, as in EGMDO.

The papers list other kernels, an EI-based stopping criterion, costly constraints independent of the couplings, and high-dimensional couplings as perspectives. These are extensions here (Stage E).

## 4. Proposed architecture

### Rust core

Create a new crate `crates/egmdo` (`egobox-egmdo`). Do not enlarge `egobox-ego`. The workspace picks up `crates/*` automatically.

Modules (indicative):

- `problem`: design space, disciplines, coupling-variable mapping, initial coupling bounds, objective and constraints;
- `mda`: non-linear Jacobi fixed-point solver for one sample of `Ξ`, used for random MDAs and mean MDAs (and, in tests, for true MDAs);
- `pce`: multivariate Hermite polynomial basis of total degree `d`, least-squares fit of the objective and constraint PCEs from the same MDA samples, mean, variance and first-order Sobol indices;
- `kl`: Karhunen–Loève decomposition of the objective PCE coefficient vectors and KL-GP interpolation; GP interpolation of the constraint PCE means;
- `criteria`: Monte Carlo EI and CEI, WB2, `P_min`/`P_sol`, constraint violation, coefficient of variation;
- `solver`: enrichment (EGMDO Algorithm 1, C-EGMDO Algorithm 2), main loop, ask/tell state machine, final estimation of `Ẑ*` and `Ŷ_min`.

The unconstrained EGMDO is the special case of C-EGMDO with no constraints, all-discipline enrichment and the EI criterion. A single solver, configured by options, implements both rather than two parallel code paths.

Reuse:

- `egobox-gp`: `GaussianProcess` with constant mean and a Matérn 5/2 kernel by default, or the papers' squared exponential (spec decision D34); `predict` and `predict_var` give `μ_i` and `σ_i`. No new GP capability is required.
- `egobox-doe`: seeded `Lhs` for `DoE_fi` and `DoE_UQ`.
- `linfa-linalg`: symmetric eigendecomposition for the KL step and QR least squares for the PCE.
- `egobox_gp::basin_optimizer`: the COBYLA adapter shared with `Egor` (constraints `c ≤ 0`), used for the multistart acquisition maximization and the final estimation. It already supports the constraints of the constrained acquisition, so `egobox-ego` does not need to make its private `optimizers` module public.
- `Egor` conventions: inequality constraints `g ≤ 0` and a constraint tolerance, so that EGMDO and `Egor` problems are written the same way.

New numerical code (not available in EGObox): Hermite PCE with least-squares regression and PCE-based Sobol indices, KL decomposition of the PCE vector, Monte Carlo EI/CEI.

### Ask/tell orchestration

The Rust core owns the algorithm state and runs all surrogate computations. It does not call disciplines itself. Instead, it exposes an ask/tell interface:

- `ask()` returns the next batch of disciplinary evaluations to perform. During initialization it is the initial `DoE_fi` of every discipline; during enrichment it is one point `(z, y_c(i))` per enriched discipline (all of them, or only the one selected by Sobol indices);
- the caller evaluates the true disciplines and passes the outputs back with `tell(...)`;
- `minimize()` is a convenience wrapper that runs this loop with user-provided callbacks.

This follows the existing `suggest` pattern (`crates/ego/src/solver/egor_service.rs`, `python/src/egor.rs`). Python callbacks never run inside Rust loops, and MDO frameworks (OpenMDAO, GEMSEO), outside EGObox, only have to answer requests.

### Python API

Expose a user-facing API under `egobox.egmdo`.

The initial API accepts explicit discipline specifications, Python evaluation functions, a cheap objective function and optional cheap constraint functions, and exposes both `minimize()` and ask/tell.

### MDO frameworks

Adapters to MDO frameworks are not part of EGObox. A framework drives EGMDO through ask/tell: it answers each request by running the requested discipline at the given inputs, and runs its own full model for validation.

## 5. Discipline evaluation model

A discipline `f_i(z, y_c(i))` takes the design variables (or a subset) and the coupling variables it receives, and returns one or more coupling outputs.

Represent each discipline explicitly with at least:

- a stable name or identifier;
- the design variables it uses;
- the coupling variables it receives and the ones it produces;
- **initial bounds `C(i)` for the coupling variables it receives**. They are required to sample the initial disciplinary DoE over `Z × C(i)`. They are only an initial guess: enrichment points may fall outside them. In Python, each coupling variable is declared once, by name, with its initial range, and `C(i)` is assembled from the ranges of the couplings the discipline receives;
- an evaluation callback.

EGObox GPs have a single scalar output. **One GP is trained per scalar coupling output**, and each has its own random variable `ξ`. The stochastic dimension `n_ξ` is therefore the total number of scalar coupling outputs; with the paper's scalar couplings it equals `n_d`. The PCE size grows as `(n_ξ + d)! / (n_ξ! d!)`, which limits the practical number of coupling outputs (documented as a limitation).

Random samples can push coupling variables outside the physical domain of a discipline (e.g. `sqrt(y1)` in Sellar). Test problems must be written to tolerate this, and MDA failures are counted and reported.

### Objective and constraints

The objective `f_obj(z, y)` is a **cheap, explicit function** of the design and coupling variables, as in both papers. It is evaluated on the random couplings and is not modelled by a GP. An objective computed by an expensive discipline is a later extension.

Constraints follow the same model: inequality constraints `g_m(z, y) ≤ 0` and equality constraints `h_r(z, y) = 0` are cheap, explicit functions of the design and coupling variables. A constraint that depends only on `z` is a special case: it is deterministic, so its feasibility is the same for every random sample. The sign convention is the `Egor` one (`g ≤ 0`); C-EGMDO's equations use `g ≥ 0` and are translated in the specification. Equality constraints are satisfied within a tolerance. Costly constraints that do not depend on the couplings need their own surrogate; they are an extension.

The Python facade is `Emdegor` ("Efficient MultiDisciplinary Egor"), named after `Egor` and `Belfegor`; the Rust crate keeps the algorithm's name. The Python API (`egobox.egmdo`, also exported by `egobox`) declares the design and coupling variables once, by name, with their bounds; the disciplines are wired to them by name. The Rust core identifies variables by index, and the Python layer resolves the names into indices:

```python
from egobox.egmdo import Discipline, Emdegor

# Classic constrained Sellar problem, as in C-EGMDO
design = {"z1": [-10.0, 10.0], "z2": [0.0, 10.0], "z3": [0.0, 10.0]}  # bounds, order = z columns
couplings = {"y1": [0.0, 25.0], "y2": [0.0, 25.0]}  # initial ranges C(i), order = y columns

# Inputs in any order: it is the column order of x given to the discipline
d1 = Discipline("d1", compute_y1, inputs=["z1", "z2", "z3", "y2"], outputs=["y1"])
d2 = Discipline("d2", compute_y2, inputs=["z1", "z2", "y1"], outputs=["y2"])

# qoi(z, y) returns [f, g1, g2] for a batch of points, g <= 0
optimizer = Emdegor(design, couplings, [d1, d2], qoi, n_ineq=2, seed=42)

# Convenience loop, calling Discipline.fun
result = optimizer.minimize(max_iters=10)

# Or ask/tell, the caller running the disciplines
while (requests := optimizer.ask()) is not None:
    optimizer.tell([evaluate(r.discipline, r.x) for r in requests])
result = optimizer.result()
```

Every coupling variable is computed by exactly one discipline and has a single initial range. The names are checked when `Emdegor` is built (unknown, duplicated or uncomputed variables, a discipline receiving its own output).

### Three distinct evaluation modes

1. **True disciplinary evaluation**
   - Calls the original discipline function at `(z, y_c(i))`.
   - Supplies training data for the corresponding GPs.
   - The only expensive evaluation inside the optimization loop.

2. **True MDA evaluation**
   - Solves the coupled system at a design point with the true disciplines.
   - Used for validation and benchmarking only.

3. **Surrogate MDA evaluation**, in two forms:
   - **random MDA**: the GPs are replaced by `μ_i + σ_i ξ_i` for one sample of `Ξ`; used to build the objective and constraint PCEs at each `DoE_UQ` point;
   - **mean MDA**: the GPs are replaced by their means; used to compute the coupling values `y_c` of an enrichment point.

   Non-converged surrogate MDAs are counted, excluded as specified in `doc/egmdo_spec.md`, and reported.

### Computational cost

- Building the PCEs at one design point costs `n_MC` random MDAs (100 to 500 in the papers). The objective and all constraints share the same MDA samples. Each disciplinary enrichment rebuilds the PCEs at all `n_UQ` points (15 to 35 in the papers), i.e. a few thousand Jacobi solves, each requiring GP predictions.
- EI/CEI evaluations are cheap: KL-GP predictions at `z` plus polynomial evaluations. They are estimated with fixed Monte Carlo samples (common random numbers), so that the criterion is deterministic during its optimization.
- Sobol indices come analytically from the PCE coefficients and cost nothing extra.
- The final estimation of `Ẑ*` and `Ŷ_min` requires `n_sim` global minimizations of the cheap model (19).

These loops run in Rust and are parallelized with rayon over design points and samples.

## 6. Algorithm implementation stages

### Stage A — Technical specification

Write `doc/egmdo_spec.md` from both papers: equations, algorithms, default settings, mapping to Rust modules, test cases and reference results, and the decisions the papers leave open. Part I covers EGMDO and part II covers the C-EGMDO extensions.

### Stage B — Core algorithm (EGMDO)

Scope:

- continuous bounded design variables;
- one scalar, cheap objective; the problem definition already accepts constraints (used in Stage B′);
- two or more explicitly declared disciplines with scalar or vector coupling outputs;
- one GP per scalar coupling output;
- random MDA by non-linear Jacobi;
- Hermite PCE, KL decomposition, KL-GPs;
- Monte Carlo EI, Algorithm 1 and Algorithm 2;
- final estimation of `Ẑ*` and `Ŷ_min` distributions;
- ask/tell interface and `minimize()` wrapper;
- per-discipline evaluation accounting.

Defer mixed-integer variables, multi-objective optimization, and parallel true evaluations.

### Stage B′ — C-EGMDO extensions

Scope:

- PCEs of the constraints at `DoE_UQ` points, from the same random MDAs as the objective;
- GP interpolation of the constraint PCE means;
- constrained EI (CEI) with the feasible minimum `f'_min(Ξ)` and the constraint-violation fallback;
- WB2 acquisition criterion, available for unconstrained problems too;
- acquisition maximization under the GP-mean constraints;
- `P_sol`, its constraint-violation fallback, and the maximum coefficient of variation over objective and constraints (C-EGMDO Algorithm 2);
- single-discipline enrichment by first-order Sobol indices from the PCE, available for unconstrained problems too;
- rejection of acquisition points that duplicate a `DoE_UQ` point (next best multistart point).

### Stage C — Benchmarks

Use the papers' test cases, which come with reference results:

- **1-D toy problem** (EGMDO section 2, C-EGMDO section 2.1): small enough for unit tests of each step, with the papers' figures as qualitative references. C-EGMDO adds the constraint `g1 = −0.15 y1 + 0.9 ≥ 0`, which moves the optimum to `x ≈ −2.42`;
- **modified Sellar problem** (EGMDO section 4.1, unconstrained): reference `z* ≈ (0, 2.634, 0)`, `f* ≈ −2.808`, local minimum at `(0, −2.595, 0)`; the paper reports 88% convergence and 13 evaluations per discipline on average over 100 runs (15 iterations);
- **classic constrained Sellar problem** (C-EGMDO section 4.1): reference `z* ≈ (1.9776, 0, 0)`, `f* ≈ 3.1833`; with WB2, the paper reports 96% convergence over 100 runs and 22.8 disciplinary evaluations in total on average (vs. 56.6 for IDF-SEGOMOE);
- **compound cylinder** (EGMDO section 4.2): reference `z* ≈ (7, 9, 12, 6.37e−3)`, `f* ≈ 2.33e−2`.

The aeroelastic wing of C-EGMDO (section 4.2) relies on Code Aster and an in-house VLM solver; it is not reproducible here.

The results are in `doc/egmdo_benchmarks.md`, reproduced by the `egmdo_benchmarks` example of `egobox-egmdo`. They include an `Egor` baseline on the MDF formulation, and the WB2S (spec D35) and Monte Carlo LogEI (spec D36) options. LogEI removes the flat zones of the CEI but does not beat WB2 overall, so WB2 stays the default. On constrained Sellar, some samples of the final minimum position `Ẑ*` fall in the mirrored basin `z1 ≈ −1.8`, although the best `DoE_UQ` point is the optimum. WB2S does not change this: the EI is zero almost everywhere once the disciplinary GPs are enriched, so the bias comes from the final estimation with the model (19). Improving that estimation is an extension.

Tests should verify:

- variable mapping and callback dimensions;
- convergence of true, random and mean MDAs;
- PCE moments against Monte Carlo moments;
- KL truncation and KL-GP interpolation at `DoE_UQ` points;
- `P_min`, coefficient of variation and EI on small hand-checked cases (EI is zero at `DoE_UQ` points);
- CEI, WB2, `P_sol`, constraint violation and Sobol indices on small hand-checked cases (Sobol indices sum to at most 1);
- enrichment logic of both algorithms;
- reproducibility for a given seed;
- reference results within documented tolerances.

### Stage D — Framework adapters (out of scope)

Adapters to MDO frameworks such as OpenMDAO are not part of EGObox (section 4). Ask/tell is the integration point.

### Stage E — Profile and extend

After correctness is established:

- profile GP fitting, random MDAs, PCE, KL, acquisition optimization and true engineering evaluations separately;
- investigate incremental GP updates if they provide a measurable benefit;
- add parallel discipline evaluations;
- try other kernels and other criteria (e.g. upper trust bound constraint relaxation), which the papers suggest. C-EGMDO tested constraint relaxation by quantiles of the constraint PCEs, which was less robust than the mean, so it is not in the first release;
- add a stopping criterion based on the maximum EI value;
- improve the final estimation of `Ẑ*` and `Ŷ_min` (model (19)) away from the enriched regions, e.g. by restricting it to the basin of the best `DoE_UQ` point or by enriching where its samples disagree;
- benchmark against `Egor` on the IDF formulation;
- support costly constraints that do not depend on the couplings, with their own surrogate;
- consider objectives and constraints computed by expensive disciplines;
- consider high-dimensional couplings with reduced-order models (Berthelin et al., 2022) or KPLS;
- consider more general variable types.

## 7. Initial scope and deferred features

| Feature | Initial version |
|---|---|
| Design variables | Continuous and bounded |
| Objective | One scalar, cheap, explicit objective |
| Disciplines | Two or more |
| Discipline outputs | One or more scalar coupling outputs |
| Coupling | Explicit mappings, initial coupling bounds |
| Surrogates | One GP per scalar coupling output |
| Uncertainty model | Simplified GP, one standard normal per GP; optionally linearized at the mean MDA solution (D37) |
| Random objective | Hermite PCE + KL decomposition + KL-GPs |
| Constraints | Cheap inequality and equality constraints of `(z, y)`, PCE + GP of the mean |
| Acquisition | EI/CEI or WB2 |
| Enrichment | All disciplines, or the one with the highest Sobol index |
| Stopping | Iteration budget |
| Orchestration | Ask/tell, plus `minimize()` wrapper |
| True evaluations | Serial |
| Surrogate computations | Parallel, in Rust |
| Costly constraints independent of the couplings | Deferred |
| Constraint relaxation | Deferred |
| Multi-objective optimization | Deferred |
| Mixed-integer variables | Deferred |
| MDO framework dependency | None, in the Rust core and the Python package |
| MDO framework adapters (OpenMDAO, GEMSEO) | Not part of EGObox; ask/tell is the integration point |

## 8. API and implementation decisions to settle

All settled:

1. Settled: variables are scalars, identified by name in Python and by index in the Rust core; vector outputs are several coupling variables of one discipline.
2. Settled: in Python, a discipline input follows the order of its `inputs` names, and the initial coupling ranges are declared once with the coupling variables; the Rust core uses `[z_i, y_c(i)]`, with initial bounds per coupling input, and the Python layer reorders the columns.
3. Settled: `ask()` returns the requests `(discipline, x)` of one step (initial DoEs, then one enrichment point per enriched discipline), `tell()` takes their outputs in the same order; failed evaluations are raised as errors.
4. Settled: as `Egor`, Python exceptions in the `qoi` callback abort the run and are re-raised; exceptions in the disciplines are raised directly. The `qoi` callback is called from several threads, the solver running without the GIL.
5. Settled: excluded from the regression (spec D5), with relaxation retries (D33) and point dropping (D32).
6. Settled: no `egobox-ego` utility is made public; the acquisition uses `egobox_gp::basin_optimizer`.
7. How constraints are declared (functions of `(z, y)` returning a vector, or one function per constraint), and the equality tolerance.
8. The option names and defaults that select between EGMDO and C-EGMDO behaviors (acquisition criterion, enrichment strategy, `ε_P`).

## 9. Acceptance criteria

The initial implementation is ready for a first release when:

- the algorithms match the papers' formulations, as documented in `doc/egmdo_spec.md`;
- the 1-D toy (unconstrained and constrained), modified Sellar, constrained Sellar and compound cylinder problems run end to end;
- on modified Sellar (EGMDO settings) and constrained Sellar (C-EGMDO settings), results over many seeds are comparable to the papers' (convergence rate and number of evaluations per discipline), with the comparison documented;
- EGMDO and C-EGMDO need fewer disciplinary evaluations than `Egor` applied to the true MDA objective and constraints (median and spread over fixed seeds);
- runs are reproducible for a given seed;
- tests cover variable mapping, MDA, PCE, Sobol indices, KL, KL-GPs, EI/CEI/WB2, `P_min`/`P_sol`, constraint violation, enrichment and stopping behavior;
- Python users can provide discipline callbacks, or use ask/tell, without any MDO framework;
- API and documentation clearly distinguish disciplinary evaluations, true MDA evaluations, and surrogate MDA evaluations.

## 10. References

- S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, *Towards an efficient global multidisciplinary design optimization algorithm*, Structural and Multidisciplinary Optimization 62:1739–1765, 2020, [doi:10.1007/s00158-020-02514-6](https://doi.org/10.1007/s00158-020-02514-6)
- I. Cardoso, S. Dubreuil, N. Bartoli, C. Gogu, M. Salaün, *Constrained efficient global multidisciplinary design optimization using adaptive disciplinary surrogate enrichment*, Structural and Multidisciplinary Optimization 67:23, 2024, [doi:10.1007/s00158-023-03736-0](https://doi.org/10.1007/s00158-023-03736-0)
- S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, J. Mas Colomer, *Extreme value oriented random field discretization based on an hybrid polynomial chaos expansion – kriging approach*, Computer Methods in Applied Mechanics and Engineering 332:540–571, 2018 (KL-GP random field and EI estimation)
- B. Sudret, *Global sensitivity analysis using polynomial chaos expansions*, Reliability Engineering & System Safety 93(7):964–979, 2008 (Sobol indices from PCE coefficients)
- G. Berthelin, S. Dubreuil, M. Salaün, N. Bartoli, C. Gogu, *Disciplinary proper orthogonal decomposition and interpolation for the resolution of parameterized multidisciplinary analysis*, International Journal for Numerical Methods in Engineering 123(15):3594–3626, 2022 (Sobol-based disciplinary enrichment, high-dimensional couplings)

## 11. Immediate next steps

1. Write `doc/egmdo_spec.md` (parts I and II).
2. Create `crates/egmdo` with the problem definition (including constraints), the MDA solver and the papers' test problems.
3. Implement the PCE, KL and KL-GP random field.
4. Implement the EGMDO criteria and algorithms.
5. Add the C-EGMDO extensions.
6. Reproduce the papers' results and compare with `Egor` on the true MDA.
7. Add the Python bindings.

## 12. Delivery

### Branching workflow

- `egmdo` is the integration branch, created from `master`. Nothing is merged into `master` until the acceptance criteria of section 9 are met.
- Each PR has its own branch created from `egmdo` and targets `egmdo`. Branches are named `egmdo-<topic>`. (`egmdo/<topic>` is not possible because Git cannot have both a branch `egmdo` and branches under `egmdo/`.)
- `egmdo` is rebased on `master` periodically, to keep it up to date with the rest of EGObox; the open `egmdo-<topic>` branches are then rebased on the new `egmdo`.
- The final integration is a single PR from `egmdo` to `master`.

### PR sequence

| # | Branch | Content | Stage |
|---|---|---|---|
| 1 | `egmdo-plan` | This integration plan | — |
| 2 | `egmdo-spec` | `doc/egmdo_spec.md` (EGMDO and C-EGMDO) | A |
| 3 | `egmdo-mda` | `crates/egmdo` skeleton: problem definition with constraints, disciplinary GPs, MDA solver (true, random, mean), test problems | B |
| 4 | `egmdo-uq` | Hermite PCE (objective and constraints) with Sobol indices, KL decomposition, KL-GP random field | B |
| 5 | `egmdo-algorithm` | EI, `P_min`, EGMDO enrichment and main loop, final estimation, ask/tell and `minimize()` | B |
| 6 | `egmdo-constraints` | C-EGMDO: constraint GPs, CEI, WB2, `P_sol`, constraint violation, Sobol enrichment | B′ |
| 7 | `egmdo-benchmarks` | Papers' test cases at scale, robustness studies, comparison with `Egor` on the true MDA | C |
| 8 | `egmdo-python` | `egobox.egmdo` Python bindings, type stubs, tests and documentation | B/C |
| 9 | — | Dropped: MDO framework adapters (OpenMDAO) are not part of EGObox | — |
| 10 | `egmdo-linearized-mda` | Linearized random MDA (D37, issue #493), compared with the solved random MDA for several initial DoE sizes | E |

Stage E items get their own PRs once the first release is merged into `master`, except the alternatives to the rank-one disciplinary GP of issue #493, studied with its author on `egmdo`:

1. the linearized random MDA (PR 10);
2. the correlation across design points from the GP posterior covariance, in place of the shared `Ξ`, with the solved or the linearized random MDA;
3. a local KL expansion of the disciplinary GPs around the mean MDA solution, if the correlation across design points matters;
4. a hybrid that solves the random MDAs while the uncertainty is large and linearizes them afterwards.

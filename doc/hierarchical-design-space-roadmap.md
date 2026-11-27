# Hierarchical design spaces in Egor: PR roadmap

## Context
Saves et al. 2026, "Modeling hierarchical spaces: a review and unified
framework for surrogate-based architecture design", Struct Multidisc Optim
69:65 (doi:10.1007/s00158-026-04249-2), describes design spaces where some
variables exist only for some values of other variables, and a GP kernel for
such spaces, implemented in SMT. This roadmap splits the work needed for Egor
to optimize over these spaces into PRs that can each be reviewed and tested on
their own. Egor's current behaviour doesn't change until PR 4.

`DesignSpace` and hierarchical sampling go in `egobox-moe`, next to `XType`
(`crates/moe/src/xtypes.rs`) and `MixintSampling`
(`crates/moe/src/mixint/sampling.rs`). `egobox-doe` stays continuous-only and
is not changed.

## Branch and PR workflow
The work happens on a reference branch, `hierarchical`, as EGMDO does with
`egmdo`:
- `hierarchical` starts from `master`. Each PR in this roadmap targets
  `hierarchical`, is squash-merged into it, and has a title starting with
  `Hierarchical:`.
- PR 0 opens the branch:
  - it adds this roadmap;
  - it adds `hierarchical` to the `pull_request` and `push` branches of
    `.github/workflows/test.yml` and `.github/workflows/pytest.yml`, so that
    PRs to the branch run the tests (`egmdo` has the same change).

  Snapshot fixtures for new tests are then recorded by CI on these PRs too.
- `hierarchical` is rebased onto `master` from time to time and force-pushed
  with `--force-with-lease`, as `egmdo` is. Open PRs are rebased onto it
  afterwards.
- `hierarchical` is merged into `master` when the feature is ready. PR 4 is
  the first point where Egor can use a design space.

## Terms (from the paper)
- At a given point, a variable is either *included* or *excluded*, depending
  on the values of other variables. The word "active" is avoided because it
  already refers to CoEGO's `activity_strategy`.
- Roles (Def. 3) are flags that can be combined:
  - *meta*: decrees other variables;
  - *decreed*: included only for some values of its meta variables;
  - *partially decreed*: always included, but the values it can take depend on
    other variables;
  - *neutral*: none of the above.

  For example, `P_input` (Fig. 6) is both meta and partially decreed, and
  `c_i` (Fig. 8) is partially decreed by `w_s` and decreed by `w_c`.
- Each variable has a type (`XType`: Float, Int, Ord or Enum) and a role, and
  the two are independent: `DesignSpace` holds the `xtypes` and the role
  graph.
- The role graph (Def. 1) contains:
  - decree arcs;
  - bound-limiting arcs (partial decrees);
  - intermediate AND/OR nodes (Fig. 7);
  - incompatibility edges between values;
  - order edges between variables.

## Overall approach
The optimizer keeps working in a fixed-size space that contains every
variable, using today's `xtypes` relaxation. The hierarchy enters at three
places:

1. **Correction**: `x -> (x_corrected, included)`, which extends the
   mixed-integer cast (`cast_to_discrete_values`,
   `crates/moe/src/mixint/space.rs`). It goes through the variables in
   topological order:
   - an excluded variable is set to a fixed default value;
   - an included variable is cast to the nearest value its parents allow:
     the nearest allowed value for Int and Ord, and for Enum, the allowed
     level with the largest relaxed one-hot component.

   Casting and correcting in a single pass avoids rounding twice. For example,
   a relaxed 6.4 with allowed values {4, 8, 12} goes to 8, whereas rounding
   first gives 6, halfway between 4 and 8. Users call the correction in
   folded space; Egor calls it in its unfolded space, where the relaxed values
   are still available. Egor already casts discrete values in a few central
   places, so the correction goes there (PR 4).
2. **Surrogate**, in two stages:
   - *Imputation*: the current mixint GP, trained on corrected points. The
     mixed-integer kernel has no value for "excluded":
     - an excluded Enum imputed with its first level looks identical to that
       level;
     - a Float imputed with its lower bound looks like an included one at
       that bound.

     The paper argues (end of §3.2) that imputation cannot represent
     incompatibilities, so this stage only serves as a baseline.
   - *Algebraic kernel* (Eqs. 6–8), computed through an input embedding.
     App. C shows that the algebraic distance equals the Euclidean distance
     between embedded points:
     - an included variable maps to `[(1-u²)/(1+u²), 2u/(1+u²)]`, where `u`
       is its value scaled to [0, 1] by its bounds;
     - an excluded variable maps to `[0, 0]`.

     With one θ per variable, a squared exponential (SE) kernel applied to
     the embedding is then a hierarchical kernel built on the distances of
     Eq. 6. The variable's type sets the distance between included values,
     and the embedding adds an "excluded" value at distance 1 from all of
     them.
3. **Infill**:
   - start points come from the hierarchical sampler, and the criterion is
     evaluated at corrected points;
   - then (PR 5), each local search stays within one configuration of the
     meta variables, reusing CoEGO's mechanism for optimizing only some of the
     components.

## PRs

### 1. `DesignSpace` model (new `crates/moe/src/hierarchy.rs`, no behaviour change)
- Variables are named, preferably, with `DesignSpace::named(&[(name, xtype)])`.
  Otherwise `DesignSpace::new(xtypes)` names the variables `x0`, `x1`, ...
  after their index, so every variable has a name. Names appear in error
  messages next to the index, e.g. `variable 3 (r_11)`, or alone for a
  default name, e.g. `variable x3`.
- `decreed_var(decreed, meta, values)` is SMT's `declare_decreed_var`, renamed
  for brevity: `decreed` is included when `meta` takes one of `values`.
  - It takes each variable as a name or an index (`Var`, from `&str` or
    `usize`). There is one method, not separate named and indexed versions.
  - `decreed_var_when(decreed, condition)` takes an AND/OR condition, whose
    leaves `ds.is_in(name, values)` builds by name.
  - Conditions store indices, so names are labels only.
- A condition type that combines value sets of discrete parents with AND/OR,
  for intermediate nodes. Thresholds on continuous parents come in PR 9.
- No graph crate. The role graph is stored as the domain model: `xtypes`, one
  condition per decreed variable (a small `In` / `And` / `Or` tree, whose
  inner nodes are the intermediate nodes), and the value constraints. A
  variable's parents are the variables named in its condition and in value
  constraints oriented towards it.
- Roles derived from the arcs, as flags. Check that the graph is acyclic and
  compute a topological order (Kahn's algorithm, which detects cycles too).
- `correct_get_included(x)` goes through the variables in topological order.
  For each variable, it decides from the already-corrected parents whether
  the variable is included. An excluded variable gets a fixed default (lower
  bound or first level); an included one is cast as `cast_to_discrete_values`
  does.
- Two entry points: folded space for users, and Egor's unfolded space, where
  the relaxed values are kept until each variable is cast.
- Serde support, names included, so that adding them later doesn't change
  the postcard format of saved surrogates (PR 7).
- Tests:
  - motors and propellers (Fig. 1);
  - an AND/OR condition, as in Fig. 7 but without the continuous threshold.

### 2. Value constraints and partially decreed discrete variables
- `forbid(var1, vals1, var2, vals2)` (SMT's `add_value_constraint`) forbids a
  combination of values when both variables are included, and records it as
  a `ValueConstraint`. The variable that
  comes later in the topological order of the decrees (variable index breaks
  ties) is partially decreed by the other, whichever argument it is, and
  correction changes it. Constraint arcs then all go forward in that order,
  so they cannot create a cycle. An example is DRAGON's motor count, whose
  allowed values depend on the core count (§4.2). Like `decreed_var`, it
  takes variables by name or by index.
- Roles: `Role::PARTIALLY_DECREED` for the restricted variable, `Role::META`
  for the one restricting it (Def. 3).
- Order relations and bounds between discrete variables are finite sets of
  forbidden pairs, so they are value constraints too, e.g. `l ∈ [m+1, 4]`
  when `l` and `m` are integers.
- Correction casts each variable, in topological order, to the nearest value
  allowed by the variables before it, with the tie-breaks of the
  mixed-integer cast; Int values are also clamped to their bounds. If a
  variable has no allowed value left, it backtracks with a depth-first
  search, trying the nearest allowed values first.
- Values that no condition or constraint mentions all have the same effect on
  the other variables: the search only tries the nearest one, and the count
  groups them, so large Int ranges stay cheap.
- Count configurations (`n_valid`, `n_declared`). A declaration that leaves
  no valid configuration fails and leaves the space unchanged.
- Tests:
  - the MLP of Table 1 with the constraints of Fig. 14 (no 3 layers with Adam,
    no 40 or 45 units with ASGD) gives `n_valid = 207` and
    `n_declared = 2250`, as in Table 6;
  - DRAGON's motor counts: 17 architectures (Table 3). Table 3 lists 4
    cores for architectures 15 to 17, but Table 4 gives cores in {2, 4, 6},
    and motors in multiples of 4, 8 or 12 give the 17.

### 3. Hierarchical sampling
- A decorator around an `egobox_doe::SamplingMethod`, like `MixintSampling`.
- Take one point per valid configuration (more if more points are requested),
  sampling the included continuous variables, then subsample to the requested
  size (§4.4).
- Enumerate only the discrete variables that shape the configuration: meta
  variables and the variables in value constraints. Other discrete
  variables, such as the MLP's activation function, are sampled as
  `MixintSampling` does today. The counts are multiplied by their number of
  values (MLP: 69 configurations × 3 activation functions = 207).
- Above the cap, sample the whole space and correct the points.
- Sampling can be restricted to a given configuration of the meta variables,
  which PR 5 needs.
- Log `n_valid` / `n_declared`.

### 4. Egor integration with correction only (imputation baseline)
- `EgorConfig::design_space(..)`, which also sets `xtypes`, and
  `min_within_design_space(..)` next to `min_within_mixint_space`
  (`crates/ego/src/egor.rs`, `crates/ego/src/solver/egor_service.rs`).
- Correction goes where Egor casts discrete values today:
  - `projected` (`solver_computations.rs`), which is used for infill
    candidates and duplicate checks (`solver_impl.rs`), Thompson candidates
    and portfolio members (`point_selection.rs`), and TREGO points
    (`trego.rs`);
  - the `to_discrete_space` calls for evaluations and function constraints
    (`fcstr_input` in `solver_computations.rs`), ask-and-tell suggestions
    (`egor_service.rs`) and results (`egor.rs`).

  Their `config.discrete()` guard must also pass when a design space is set.
- Correction also applies to:
  - the initial DoE, drawn with the PR 3 sampler;
  - DoE, warm-start and hot-start data given by the user;
  - the infill criterion, which is evaluated at the corrected point.

  Gradients along excluded variables are set to zero, as Enum one-hot
  columns already are (`unfold_fcstr_gradient`).
- Every sampling site switches to the PR 3 sampler:
  - Thompson sampling candidates (`candidate_set`, `thompson.rs`);
  - the LHS fill of `MiddlePickerMultiStarter` and the TREGO
    `LocalLhsMultiStarter` (`multistart.rs`).

  A midpoint between training points from two configurations mixes those
  configurations, so midpoints are corrected as well.
- Duplicate checks (`usable_data` in `crates/ego/src/utils/misc.rs`,
  `distinct_indices` in `crates/ego/src/utils/portfolio.rs`) compare
  `projected` points. Two points that differ only in excluded variables
  therefore count as the same point, with no further change.
- `OptimResult` has public fields and no `#[non_exhaustive]`, so adding a
  field would be a breaking change. Instead, users call
  `DesignSpace::correct_get_included(&x_opt)` to get the inclusion mask.
- `check()` rejects CoEGO (`activity_strategy`) when a design space is set,
  until the optional PR 11.
- Tests: hierarchical Goldstein (Pelamatti), Zaefferer–Horn, and an MLP-like
  problem with value constraints. Snapshot fixtures for new tests are taken
  from CI artifacts, not recorded locally.

### 5. Infill within each configuration (reusing CoEGO's activity)
- CoEGO's infill loop already optimizes only some of the components and keeps
  the others fixed at the values of a reference point
  (`optimize_infill_criterion` in `solver_infill_optim.rs`):
  - full points are rebuilt with `coego::set_active_x`;
  - gradients are restricted with `coego::get_active_grad`;
  - bounds and start points come from the multistarter, for the active
    components only.

  This applies to the criterion, the constraint models and the function
  constraints alike. Here a group is one configuration of the meta variables:
  - its active components are the unfolded columns of the included variables
    that are not meta;
  - the meta values and the defaults of the excluded variables stay fixed.

  Fixed variables aren't optimizer variables, so no bounds of zero width are
  needed. The padded activity matrix (index `nx` as padding, `coego::strip`)
  already supports groups of different sizes.
- Differences from CoEGO:
  - Anchor: each group has its own anchor point (its configuration), whereas
    CoEGO uses `infill_data.xbest` for every group.
  - Results: CoEGO chains its groups, each one starting from the previous
    optimum, and returns the last one. Configurations are alternatives
    instead: each group is a separate local search, and the best criterion
    value wins.
  - Training: in CoEGO, the activity also restricts θ tuning
    (`ThetaTuning::Partial`, `adjust_theta_tuning` in `solver_impl.rs`). Here
    the groups only reach the infill optimizer, and all θ are trained.
  - Bounds and starts: each group has its own bounds, narrowed by partial
    decrees (e.g. at most 35 units with ASGD), and its start points are
    sampled within the configuration (PR 3). This means the multistarter
    takes the configuration, not just the active indices.
- A single loop can serve both: a list of `(active, anchor, bounds)` groups
  with a "chain" mode for CoEGO and a "best of" mode for configurations.
- There is one group per valid meta configuration, 5 for the MLP (ASGD with
  1 to 3 layers, Adam with 1 or 2). When there are many, each iteration
  searches a subset: the configurations of the best points plus a few random
  ones.
- TREGO's local step uses the same activity (`trego.rs`), so it searches
  within the configuration of the current best point.
- Tests: compare with PR 4 and with the per-configuration reference (see
  Verification) on the same problems.

### 6. `egobox-gp`: tied thetas and normalization opt-out (independent, can start any time)
- Allow a fixed 0/1 grouping matrix to be used as `w_star` instead of
  `pls_rotation`. SE computes `Σ_j d_j² Σ_l (θ_l w_jl)²`
  (`crates/gp/src/correlation_models.rs`), so the matrix ties the θ of each
  group together.
- Add an explicit flag for this mode. Otherwise `kpls_dim()` would report
  PLS, since it infers PLS from `w_star.ncols() < nx`
  (`crates/gp/src/algorithm.rs`), and the surrogate's `Display` would print
  `_PLS(h)` (`crates/moe/src/surrogates.rs`).
- Allow some columns to skip per-column scaling (`NormalizedData`,
  `crates/gp/src/utils.rs`). Scaling the two embedding columns of a variable
  separately breaks the distance equality of App. C; centering doesn't.
- The sparse GP recomputes `pls_rotation`
  (`crates/gp/src/sparse_algorithm.rs`), so it is not covered.
- Tests: an identity grouping gives the same GP as today.

### 7. Hierarchical surrogate: the algebraic kernel
- A new surrogate type next to `MixintGpMixture`, holding a `GpMixture` and
  the `DesignSpace`, with its own `#[typetag::serde]` name. Adding a field to
  `MixintGpMixture` instead would prevent loading existing `.bin` models:
  postcard (`crates/moe/src/persistence.rs`) doesn't store field names, and
  `#[serde(default)]` only helps with JSON.
- Embedding of decreed variables, done where `cast_to_model_space` and
  `to_input_gradients` are today (`crates/moe/src/mixint/surrogate.rs`):
  - Float, Int and Ord: the circle point when included, `[0, 0]` when
    excluded, with `u` scaled by the bounds rather than by data statistics;
  - Enum: one-hot when included, all zeros when excluded. An included level
    is at distance 1 from the excluded value and at √2 from another level,
    which satisfies Eq. 6's condition δ ≥ max/2;
  - neutral and meta variables are handled as they are today.
- One θ per variable through the PR 6 grouping, as in HIER (Table 8). With a
  design space, all one-hot columns of an Enum share one θ; without one,
  Enum variables keep one θ per level.
- Gradients:
  - included variables get the derivative of the circle map;
  - excluded variables get zero;
  - meta variables keep their own gradient, since the inclusion mask is
    piecewise constant.
- Kernel form: SE on the embedding gives `exp(-½ Σ θ_i² d_i²)`, while Eq. 10
  writes `exp(-d)`. Check which form SMT computes (its default correlation is
  `squar_exp`) and test against that.
- Abs-exp and Matérn apply θ column by column. On the two embedding columns,
  the distance between an included and an excluded value then depends on the
  included value, which breaks Eq. 6. `check()` therefore allows only SE,
  until those kernels get a per-variable Euclidean norm.
- `check()` also rejects `kpls_dim`, any `n_clusters` other than 1 (including
  `Auto`), and `GpType::SparseGp`.
- Egor uses this surrogate when a design space is set. Thompson sampling
  (`sample()`) goes through the same embedding. `gpx` can save, load and
  predict with it.
- Tests:
  - the kernel matches a direct implementation of Eqs. 6–8;
  - the correlation matrix stays positive definite when points have
    different included variables;
  - optionally, reference values from SMT;
  - a comparison with the PR 4 baseline on the PR 4 problems.

### 8. Python bindings and docs
- `egx.DesignSpace(xspecs, names=None)` with `decreed_var` and `forbid`,
  taking variables as `int | str`, and a
  `design_space` argument for `Egor` and `Gpx`. `names=None` gives the
  default names `x0`, `x1`, ...
- Regenerated `.pyi` stubs, updated skills in `python/skills/`, and an MLP
  notebook.

### 9. Continuous dependencies
- Supported cases:
  - order relations involving a continuous variable, e.g.
    `Pmin < Pinput < Pmax`;
  - bounds of a continuous variable that depend on another variable;
  - decree thresholds on continuous parents (Fig. 7's "reserve too small").
- Sampling follows the topological order: parents first, then children within
  the bounds their parents allow. Enumeration splits a continuous parent at
  its thresholds.
- In PR 5's groups, a continuous meta variable stays active, with bounds
  restricted to the interval of its group.
- During infill, these become cheap known constraints, handled by
  `infill_fcstrs` and `repair_on_segment` (`crates/ego/src/solver/`).

### 10. (optional) Interop and visualisation
- A pure-Python adapter from SMT's `DesignSpace` / adsg-core.
- DOT export of the role graph, with variable names as labels.

### 11. (optional) CoEGO within configurations
Worth doing only for hierarchical problems with enough continuous variables
per configuration for CoEGO to pay off.
- Two levels of groups in `optimize_infill_criterion`
  (`solver_infill_optim.rs`), building on PR 5's `(active, anchor, bounds)`
  groups: configurations are searched as alternatives (best of), and each
  one chains CoEGO groups, as CoEGO does today.
- CoEGO groups are drawn for each configuration among its active variables
  (included, not meta), with a group size relative to their number, so that
  no group consists mostly of fixed variables.
- Each chain starts from the best point within its configuration, not from
  the global `xbest`.
- θ tuning: `ThetaTuning::Partial` (`adjust_theta_tuning`,
  `solver_impl.rs`) indexes θ by unfolded column.
  - With PR 4's imputation surrogate it works as is.
  - With PR 7's surrogate (one θ per variable, two embedding columns per
    Float, Int or Ord variable), the activity is translated from columns to
    variables first.
- Cost is about (configurations searched) × (groups per configuration)
  local searches per iteration. PR 5's subset of configurations is then
  required, and the number of groups per configuration may have to shrink.
- Remove the `check()` rejection of `activity_strategy` with a design space
  (PR 4).
- Tests: compare with PR 5 alone on a high-dimensional hierarchical problem,
  e.g. a MOPTA-like problem with a few architecture choices added.

Dependencies:
- 1 → 2 → 3 → 4 → 5;
- 6 can start any time; 7 needs 4 and 6;
- 8 comes after 4 for `Egor` and after 7 for `Gpx`;
- 9 comes after 5, and 10 comes last;
- 11 needs 5, and 7 for the θ mapping.

## Out of scope
- The arc kernel, SMT's other hierarchical kernel, which learns an extra
  parameter ρ.
- KPLS with a hierarchy, which the paper leaves as future work.
- Using a design space with more than one cluster or with sparse GPs.
  CoEGO's groups are covered by the optional PR 11.

## Verification
- Rust:
  - `cargo test -p egobox-moe` for PRs 1–3 and 7;
  - `cargo test -p egobox-gp` for PR 6;
  - `cargo test -p egobox-ego` for PRs 4, 5, 7 and 9.
- Exact checks against the paper: the MLP counts (Table 6), kernel values
  (Eqs. 6–8) and the number of θ (Table 8).
- Per-configuration reference, buildable before PR 1 with today's Egor: for
  each valid meta configuration, the included variables form a fixed-size
  mixed-integer problem, solved with `min_within_mixint_space`. It gives:
  - reference optima for the test problems (best over configurations, with a
    large budget);
  - a baseline at equal total budget, split across configurations, for
    PRs 4, 5 and 7. Hierarchical Egor should win when configurations are
    many or the budget is tight, since one model shares information between
    configurations through the variables they have in common.

  Enumerate meta configurations only (5 for the MLP), not every discrete
  configuration (207): the remaining discrete variables stay in each
  sub-problem.
- Python (PR 8), run from `python/`:
  - `maturin develop --release`
  - `cargo run --bin stub_gen`
  - `python -m pytest tests/`
  - `python -m mypy.stubtest egobox.egobox --allowlist stubtest_allowlist.txt`

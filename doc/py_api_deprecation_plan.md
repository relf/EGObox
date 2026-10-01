# egobox Python API — deprecation and breaking-change plan

## Context

[`py_api_and_ux_review.md`](py_api_and_ux_review.md) has steps 1–4 done (errors, stubs, silent bugs, docs/shape parity). Step 5 is left:
renames/aliases (item 4) and the structural points (item 5) that need new names or break behaviour.
Goal: ship these in **0.38.0** (already "unreleased" in CHANGELOG) as **additive + `DeprecationWarning`** wherever
a transition path exists, and list the few that can only be hard breaks. Old names are removed in the release
after 0.38 (0.39 / 1.0). Work is split into three PR-sized tiers, ordered by user impact vs. cost.

## Shared mechanism (do first, reused by every tier)

The package is a pure PyO3 extension (`python/egobox/__init__.py` just re-exports), so warnings are raised from Rust.

New `python/src/deprecation.rs`:
- `warn_deprecated(py, old: &str, new: &str) -> PyResult<()>` — `PyErr::warn(py, &py.get_type::<PyDeprecationWarning>(), msg, 1)`
  with message `"`{old}` is deprecated since 0.38.0 and will be removed in a future release, use `{new}` instead"`.
  Check the stacklevel so the warning points at the user's line (native frames don't count).
- `resolve_renamed<T>(py, old_name, old: Option<T>, new_name, new: Option<T>, default: T) -> PyResult<T>` —
  `TypeError` if both given, warn if only old given, else new or default.
- Dict-form helper for the `FromPyObject` dict branches (`GpConfig`, `QEiConfig`, `TregoConfig`, `CstrSpec`):
  accept new keys, warn on old keys, error on both.

Patterns:
- **Renamed kwarg**: signature carries both `old=None, new=None` → `resolve_renamed`. Docstring: new name documented,
  old one listed under "Deprecated". Stub shows both (acceptable until removal).
- **Renamed config field**: store under the new name; add `#[getter]/#[setter]` for the old name that warn.
- **Renamed method**: new method holds the body; old one calls `warn_deprecated` then delegates.
- **Enum alias (no deprecation)**: `#[classattr]` in a `#[pymethods]` block on the enum returning the variant
  (check `gen_stub_pymethods` emits it in the stub; otherwise add to `stubtest_allowlist.txt`).

Tests: new `python/tests/test_deprecations.py` — for each item, `pytest.warns(DeprecationWarning)` on the old name,
same result as the new name, `pytest.raises(TypeError)` when both are given, and no warning on the new name
(run with `-W error::DeprecationWarning` for the new-name path). Extend `tests/test_stubs.py` expectations.

---

## HIGH — frequent friction, cheap, non-breaking — ✅ done

**Implemented:** `python/src/deprecation.rs` (`warn_deprecated`, `resolve_renamed`, `resolve_renamed_key`) and
items 1–4 below. Deprecated kwargs are keyword-only (after `*`) at the end of the signatures, so positional callers
get the new name at the old position. Tests in `python/tests/test_deprecations.py` and `tests/test_stubs.py`.

1. **Result forwarding on `EgorOptim`** (`python/src/types.rs`, `EgorOptim`)
   - Add read-only getters `x_opt`, `y_opt`, `x_doe`, `y_doe` delegating to `self.result`.
   - Add `__iter__` yielding `(x_opt, y_opt)` so `x, y = egor.minimize(...)` works.
   - `res.result` stays, no warning (it is still the natural home of `OptimResult`).
2. **`n_start` ambiguity** (`egor.rs` `Egor::new`, `gp_config.rs`, `gp_mix.rs`, `sparse_gp_mix.rs`)
   - Naming rule: a family prefix on the existing name, so both renames follow one scheme.
   - `Egor(n_start)` → `infill_n_start` (deprecated old kwarg). It groups with `infill_strategy` and `infill_optimizer`.
   - `GpConfig` / `GpMix` / `Gpx.builder` / `SparseGpMix` / `SparseGpx.builder`: `n_start` → `theta_n_start`,
     `max_eval` → `theta_max_eval` (kwargs, config field getters/setters, dict keys). This fits the existing
     `theta_init` / `theta_bounds` prefix.
3. **`get_` prefix** (`egor.rs`): `best_result(x_doe, y_doe)` and `best_index(y_doe)`; `get_result` /
   `get_result_index` warn and delegate.
4. **Update in-repo callers** in the same PR, so examples and tests don't emit warnings: `python/examples/*.py`,
   `python/tests/*.py`, `python/README.md`, `README.md`, `website/content/{python-api,cookbook}.md`,
   `python/skills/egobox*/SKILL.md`, notebooks if they use the old names.

## MEDIUM — clearer names and ergonomics, still transitional — ✅ done (except item 8, moved to LOW)

**Implemented:** items 5–7 and 9–12 below. Tests in `python/tests/test_deprecations.py` for the renames, and in the
`test_egor.py`, `test_gpmix.py`, `test_sgpmix.py` and `test_stubs.py` files for the additions. Differences from the plan:
- Enum aliases are `#[classattr]`s returning the variant, so they are equal (`==`) to it but not the same object (`is`).
  The stub declares them as `ALIAS: Enum`.
- `Egor(doe=..., x_doe=...)` (the old name mixed with a new one) raises `TypeError`, like other renamed kwargs.
  `y_doe` without `x_doe`, an `x_doe` without nx columns, or a row count mismatch raise `ValueError`.
- `TregoConfig` dict keys are handled only by `TregoConfig` extraction. The duplicate parsing in `egor.rs`
  `parse_trego_config` was removed.
- `sampling`: the old order is supported only when positional, `sampling(M, X, N[, seed])`. With
  `sampling(M, X, n_samples=N)`, Python binds `n_samples` twice before the function runs, so that call raises
  `TypeError`. Calls using only keywords are unaffected. The stub shows `method: Sampling | None = None`
  (None meaning LHS), as a pyo3 signature default cannot be a Python enum instance.
- `CstrSpec.tol` lives in the Python binding only (the Rust `egobox_ego::CstrSpec` is unchanged). `Egor` builds the full
  internal `cstr_tol` vector (`Egor::internal_cstr_tol`) from `cstr_tol`, padded with the default, then overwritten by
  the specs' tolerances. `best_index` / `best_result` keep using `cstr_tol` only.

5. **Long enum aliases** (`types.rs`), additive with no deprecation, short names stay:
   `ConstraintStrategy.MEAN_CONSTRAINT` / `UPPER_TRUST_BOUND`, `QEiStrategy.KRIGING_BELIEVER` /
   `KRIGING_BELIEVER_LOWER_BOUND` / `..._UPPER_BOUND` / `CONSTANT_LIAR_MINIMUM`, `FeasibleInfillStrategy` long form
   of `EFI_FE` etc. `repr` keeps the canonical short name.
6. **Abbreviation renames** (deprecated old names, same patterns as above):
   - `CstrSpec.btw` → `CstrSpec.between` (also the `"btw"` dict key → `"between"`).
   - `QEiConfig.optmod` → `optim_every`.
   - `TregoConfig.n_gl_steps` → `n_global_local_steps`, `d` → `radius_bounds`.
   - `SparseGpMix` / `SparseGpx.builder`: `nz` → `n_inducing`, `z` → `inducing`.
7. **`Egor(doe=...)` split** (`egor.rs`): add `x_doe=` / `y_doe=` (same names as `suggest` and `OptimResult`, rather
   than the review's `doe_x` / `doe_y`); `doe` warns. `ValueError` if `doe` is mixed with `x_doe`/`y_doe`, or if
   `y_doe` is given without `x_doe`. `x_doe` alone keeps the current "x only" behaviour of `doe` with nx columns (check
   what `apply_config` does today).
8. **`GpMix(gp_config=...)`**: deferred, see item 19 in LOW.
9. **`Gpx.predict(x, return_std=False)`** and same on `SparseGpx` (`gp_mix.rs`, `sparse_gp_mix.rs`): additive;
   `return_std=True` returns `(mean, std)` with std = `sqrt(predict_var)`. Stub return type is a union/overload.
10. **`nx` / `ny` properties** on `Gpx` / `SparseGpx`: additive. `dims()` kept for now, not deprecated (low churn).
11. **`sampling(xspecs, n_samples, method=Sampling.LHS, seed=None)`** (`sampling.rs`): the first argument is either
    a `Sampling`/int (old order) or a domain (new order), so type-dispatch on it and warn for the old order. `lhs`
    becomes a thin alias. The stub shows the new signature only.
12. **Per-constraint tolerance** (`types.rs` `CstrSpec`, `egor.rs` `cstr_tol()`): `CstrSpec.leq(bound, tol=None)` etc.
    When `cstr_specs` / `fcstr_specs` are given, `n_cstr` is inferred (a `ValueError` if it contradicts), and
    `tol` on a spec takes precedence over `cstr_tol`. This fixes the case where `cstr_tol` can't cover the
    expanded function constraints. `cstr_tol` is not deprecated yet.

## LOW — design-heavy or only possible as hard breaks

**Implemented for 0.38:** items 13, 14, 20 and 21 (follow-up PRs). Item 17 is dropped. Items 15, 18 and 19 are left
for later, and item 16 goes with the removal release. Differences from the plan:
- `fcstrs` dicts accept only the `"fun"` and `"jac"` keys, both required. A scipy `"type"` key raises `ValueError`, as
  scipy `"ineq"` means `g(x) >= 0` while egobox constraints are `g(x) <= 0` (`fcstr_specs` gives other bounds).
  Tuples must be `(g, grad_g)`. The gradient is only called when the infill optimizer needs it (SLSQP).
- `Egor(verbose=...)` also initializes the logger in `suggest()`, which has no `verbose` argument.

13. ✅ **scipy-like `fcstrs`** (`egor.rs` `minimize`): also accept `(g, grad_g)` tuples and `{"fun": g, "jac": grad_g}`
    dicts. The current `g(x, return_grad)` form stays without a warning until the new one has proven itself.
14. ✅ **`seed` / `verbose` placement**: add `Egor(seed=, verbose=)` as the defaults used by `minimize`/`suggest` (the
    call-time value wins). No deprecation, which avoids churn for a minor gain.
15. **`RegressionSpec` / `CorrelationSpec` as `enum.IntFlag`**: create them at module init in `lib.rs` with the
    `enum.IntFlag` functional API and register them in place of the pyclasses; keep the `u8` extraction. They stay
    int-compatible, so this is non-breaking at runtime, but the stubs must be hand-written (outside pyo3-stub-gen) and
    `isinstance` checks against the old pyclass change.
16. **`thetas()` / `variances()` / `likelihoods()` as properties**: not possible with a warning transition (a
    property returning an ndarray can't also be callable). Hard break, planned for the removal release only.
    Listed in CHANGELOG as "upcoming".
17. ❌ **Full `CstrConfig`** grouping (`n_cstr`, `cstr_tol`, `cstr_specs`, `cstr_infill`, `cstr_strategy`): dropped.
    With item 12, `n_cstr` is inferred from `cstr_specs` and each spec carries its tolerance, so the constraint
    settings are already grouped in the specs, and a config class would add a rename for all constrained users.
    Moving these settings to `minimize()` is rejected too: they describe the `y_doe` layout (`ny = 1 + n_cstr`),
    also needed by `Egor(x_doe=, y_doe=)`, `suggest`, `best_index` and `best_result`. `fcstrs` / `fcstr_specs` sit
    in `minimize()` because they are callables, like `fun`. The one misplaced setting is `cstr_tol`, see item 20.
18. **`TypedDict`s for dict forms**: hand-maintained stub section; optional.
19. **`GpMix(gp_config=...)`** (was item 8, deferred from MEDIUM): the idea was
    `GpMix(xspecs=None, gp_config=None, seed=None, verbose=None)`, with the flat GP kwargs deprecated.
    It is on hold because a one-argument `GpMix(GpConfig(...))` is awkward: unlike `Egor`, which has many other
    parameters, the `GpMix` args are the `GpConfig` args. It would also touch almost every `GpMix` user. If it is
    revisited, `gp_config=` would be added without a warning first, with `TypeError` when it is mixed with flat kwargs.
20. ✅ **Deprecate `Egor(cstr_tol=...)`** (`egor.rs`), follow-up PR. Differences from the plan: `best_index` /
    `best_result` raise `ValueError` when `y_doe` is not `(ns, 1 + n_cstr)` (raw layout, as `suggest`); the warning
    names `CstrSpec(..., tol=...)` as replacement; `notebooks/Egor_Tutorial.ipynb` was migrated too. The
    `minimize()` output layout left open here is fixed by item 21. `cstr_tol` is given to the constructor but must
    also cover the function constraints passed later to `minimize()` (after `eq` / `between` expansion), which the
    constructor can't know. Since item 12 a spec `tol` does the same job next to its own constraint, so:
    - `Egor::new` warns when `cstr_tol` is given (not a rename, so call `warn_deprecated` directly rather than
      `resolve_renamed`), and still uses the value. The argument keeps its positional slot. The docstring moves it
      under **Deprecated**, and the `cstr_tol` sentences are removed from the `cstr_specs` / `fcstr_specs` docs.
    - Migration: `Egor(n_cstr=2, cstr_tol=[1e-3] * 2)` → `Egor(cstr_specs=[CstrSpec.leq(0.0, tol=1e-3)] * 2)`,
      and for function constraints `minimize(fcstrs=fs, fcstr_specs=[CstrSpec.leq(0.0, tol=1e-3)] * len(fs))`.
    - Fix `best_index` / `best_result` first, otherwise the migration changes results. They use
      `Egor::cstr_tol(0)`, which ignores spec `tol`, and they judge raw `y_doe` columns as `c <= 0` even when
      `cstr_specs` is set (wrong for `geq` / `eq` / `between`). Apply `egobox_ego::transform_constraints` when
      specs are set, as the core solver does (`solver_impl.rs`), then use `internal_cstr_tol(&[], 0)` with the
      default tolerance as fallback. `best_result` still returns the raw `y_doe` row. `Egor::cstr_tol()` goes away.
    - In-repo callers: the 7 `cstr_tol=` uses in `python/tests/test_egor.py`, `python/examples/g24.py` and
      `g24_fcstr.py` move to spec `tol`. `test_deprecations.py` gets the warning test and a check that the value
      is still used. Add a `best_index` test with `CstrSpec.geq(b, tol=...)`.
    - Docs: CHANGELOG deprecations table and a `best_index` / `best_result` fix bullet; `cookbook.md` (equality
      and scaling tips), `python-api.md`, both skills `SKILL.md`, and `py_api_and_ux_review.md` (constraint
      settings finding marked resolved).
    - Rust core `EgorConfig::cstr_tol` is unchanged. The bindings keep feeding it the merged vector from
      `internal_cstr_tol`. Removal of the Python `cstr_tol` goes with the removal release.
21. ✅ **Item 20 leftover: `minimize()` results in the raw constraint layout** (core fix, 0.38, behaviour change).
    With `cstr_specs`, `OptimResult.y_doe` / `y_opt` hold the internal constraints (`geq` sign-flipped and offset,
    `eq` / `between` expanded to two columns), whereas `EgorConfig::doe` / `Egor(y_doe=)`, `suggest`,
    `best_index` and `best_result` take the raw `(ns, 1 + n_cstr)` layout. So a `minimize()` output can be fed
    neither to `best_result` nor back to `Egor(x_doe=, y_doe=)`. Same discrepancy in the Rust API.
    Options reviewed:
    - Additive raw fields (`y_doe_raw`, ...) or a helper: no break, but two layouts to explain, and the
      default one stays the wrong one.
    - `best_index` / `best_result` also accepting the internal layout: the shape is ambiguous when all specs are
      `leq` / `geq`, so it needs a `layout=` kwarg; it does not fix `Egor(y_doe=)` nor the Rust API.
    - Untransform in the Python bindings only: leaves the Rust `OptimResult` inconsistent with `EgorConfig::doe`.
    - ✔ Untransform in the core `Egor::run`: one place, Rust and Python both consistent.

    `cstr_specs` shipped in 0.37.2, so this is a breaking change for its users; it is listed as a fix in CHANGELOG
    (no warning transition is possible on a returned array). Plan:
    - `crates/ego/src/types.rs`: add `pub fn untransform_constraints(y: &Array2<f64>, specs: &[CstrSpec]) ->
      Array2<f64>`, the inverse of `transform_constraints`: column 0 copied, then for each spec take its *first*
      internal column `v = scale * raw + offset` (from `CstrSpec::terms()`) and return `raw = (v - offset) / scale`
      (the second column of `eq` / `between` is derived from the first). Shape check `ncols == 1 +
      n_internal_cstrs(specs)`. Values match the evaluated ones up to rounding (`(c - z) + z`). Unit tests: round
      trip on `leq` / `geq` / `eq` / `between`, doc example next to the `transform_constraints` one.
    - `crates/ego/src/egor.rs` `Egor::run`: when `config.cstr_specs` is set, untransform `y_data` and the
      `get_full_best_cost()` row (as a 1-row array) before building `OptimResult`, in both the continuous and
      discrete branches (factor it before the `if`). `x_opt` and the best-point selection are unchanged.
      `OptimResult.state` keeps the internal layout (solver state), as do `egor_doe.npy` /
      `egor_initial_doe.npy` (warm start reads them as already transformed, `egor_solver.rs`) and the
      `egor_history.npy` best costs; document this on `OptimResult` and on the `outdir` doc.
    - Function constraints (`fcstrs`) are not in `y_doe`, nothing changes for them.
    - Rust test `test_egor_g24_metamodelized_geq_constraints` currently asserts the internal layout
      (`f_g24(doe)` vs `res.y_doe`): compare against `f_g24_geq(doe)` instead. Add a test that
      `res.y_doe` / `res.y_opt` equal `fobj(res.x_doe)` / `fobj(x_opt)` with `Eq` / `Btw` specs, and that
      `res.y_doe` can be passed back as `EgorConfig::doe` (with `x_doe`) and gives the same best point.
    - Python: `minimize()` needs no code change. Docstrings of `minimize` (`y_opt`, `y_doe` are `(…, 1 + n_cstr)`
      raw values) and `OptimResult` in `egor.rs` / `egobox.pyi` (regenerate stubs).
      `test_sphere_with_eq_cstr` reads `y_opt[1]`, `y_opt[2]` as violations: check `y_opt[1] ≈ 1.0` and
      `y_opt.shape == (2,)`. New test: with `CstrSpec.geq` / `eq`, `egor.best_result(res.x_doe, res.y_doe)`
      returns `res.x_opt` / `res.y_opt`, and `Egor(cstr_specs=..., x_doe=res.x_doe, y_doe=res.y_doe)` is accepted.
    - Docs: CHANGELOG 0.38 bullet ("with `cstr_specs`, `OptimResult.y_doe` / `y_opt` hold the raw constraint
      values, as returned by the objective function, instead of the internal `<= 0` form"); remove the "Left open"
      sentence in item 20 and in `py_api_and_ux_review.md`; check `python-api.md`, `cookbook.md` and both skills
      `SKILL.md` for examples reading constraint columns of `y_opt` / `y_doe`.

## Removal release (after 0.38)

Delete the deprecated kwargs, getters, methods, dict keys and the old `sampling` order (then `sampling` can take
typed arguments again); do item 16. `test_deprecations.py` becomes "old names raise `TypeError` / `AttributeError`".

## Docs

- ✅ CHANGELOG 0.38.0: a "Deprecations" section listing old → new, and an "Upcoming breaking changes" section (item 16).
- Update `doc/py_api_and_ux_review.md`: the status line and the step 5 entry, plus a "Fix applied" per tier.
- Regenerate the stub with `stub_gen`.

## Verification (per tier PR)

- Build and install the extension locally.
- Regenerate `python/egobox/egobox.pyi`; run `python -m mypy.stubtest egobox.egobox --allowlist stubtest_allowlist.txt`.
- `pytest python/tests` (including the new `test_deprecations.py` and `test_stubs.py`), then
  `pytest -W error::DeprecationWarning python/tests --ignore=python/tests/test_deprecations.py` to prove
  in-repo code uses no deprecated name.
- Run `python/tests/test_examples.py` (the examples) and `ruff`/`cargo clippy` on `python/src`.

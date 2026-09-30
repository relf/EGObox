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

13. **scipy-like `fcstrs`** (`egor.rs` `minimize`): also accept `(g, grad_g)` tuples and `{"fun": g, "jac": grad_g}`
    dicts. The current `g(x, return_grad)` form stays without a warning until the new one has proven itself.
14. **`seed` / `verbose` placement**: add `Egor(seed=, verbose=)` as the defaults used by `minimize`/`suggest` (the
    call-time value wins). No deprecation, which avoids churn for a minor gain.
15. **`RegressionSpec` / `CorrelationSpec` as `enum.IntFlag`**: create them at module init in `lib.rs` with the
    `enum.IntFlag` functional API and register them in place of the pyclasses; keep the `u8` extraction. They stay
    int-compatible, so this is non-breaking at runtime, but the stubs must be hand-written (outside pyo3-stub-gen) and
    `isinstance` checks against the old pyclass change.
16. **`thetas()` / `variances()` / `likelihoods()` as properties**: not possible with a warning transition (a
    property returning an ndarray can't also be callable). Hard break, planned for the removal release only.
    Listed in CHANGELOG as "upcoming".
17. **Full `CstrConfig`** grouping (`n_cstr`, `cstr_tol`, `cstr_specs`, `cstr_infill`, `cstr_strategy`): revisit
    after item 12. Per-spec tolerance may make it unnecessary.
18. **`TypedDict`s for dict forms**: hand-maintained stub section; optional.
19. **`GpMix(gp_config=...)`** (was item 8, deferred from MEDIUM): the idea was
    `GpMix(xspecs=None, gp_config=None, seed=None, verbose=None)`, with the flat GP kwargs deprecated.
    It is on hold because a one-argument `GpMix(GpConfig(...))` is awkward: unlike `Egor`, which has many other
    parameters, the `GpMix` args are the `GpConfig` args. It would also touch almost every `GpMix` user. If it is
    revisited, `gp_config=` would be added without a warning first, with `TypeError` when it is mixed with flat kwargs.

## Removal release (after 0.38)

Delete the deprecated kwargs, getters, methods, dict keys and the old `sampling` order (then `sampling` can take
typed arguments again); do item 16. `test_deprecations.py` becomes "old names raise `TypeError` / `AttributeError`".

## Docs

- CHANGELOG 0.38.0: a "Deprecations" section listing old → new, and an "Upcoming breaking changes" section (item 16).
- Update `doc/py_api_and_ux_review.md`: the status line and the step 5 entry, plus a "Fix applied" per tier.
- Regenerate the stub with `stub_gen`.

## Verification (per tier PR)

- Build and install the extension locally.
- Regenerate `python/egobox/egobox.pyi`; run `python -m mypy.stubtest egobox.egobox --allowlist stubtest_allowlist.txt`.
- `pytest python/tests` (including the new `test_deprecations.py` and `test_stubs.py`), then
  `pytest -W error::DeprecationWarning python/tests --ignore=python/tests/test_deprecations.py` to prove
  in-repo code uses no deprecated name.
- Run `python/tests/test_examples.py` (the examples) and `ruff`/`cargo clippy` on `python/src`.

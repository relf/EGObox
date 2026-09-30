# egobox Python API — review and UX suggestions

This review covers the public Python API of `egobox` (v0.37.9), as defined by the PyO3 bindings in
`python/src/*.rs` and the generated stub file `python/egobox/egobox.pyi`. The runtime findings were
checked against a local build.

The core design is sound:

- builder → `fit` → trained model (`GpMix` / `Gpx`, `SparseGpMix` / `SparseGpx`);
- `Egor(...).minimize()` for the full loop, and `suggest()` for ask-and-tell use;
- the recent grouping of options into `GpConfig`, `QEiConfig` and `TregoConfig`.

The main problems are error handling, the stubs, and naming drift. Findings are ordered by user impact.

> **Status:** items 1 (errors), 2 (stubs), 3 (silent behaviour) and 6 (doc errors) are implemented.
> Item 4 (naming) is addressed in the docs only: renames are left to the breaking release. Item 5 is open.

---

## 1. Errors bypass `except Exception` — ✅ addressed

Bad input usually raised `pyo3_runtime.PanicException`. That class subclasses `BaseException`, not
`Exception`, so `try: ... except Exception:` did not catch it, and the messages were Rust internals:

```
Egor([])                    -> PanicException: domain argument cannot be empty
Gpx.builder().fit(x, y2d)   -> PanicException: Bad training output data
gpx.save("noext")           -> PanicException: called `Option::unwrap()` on a `None` value
```

Sources of the panics:

- `domain::parse`: empty or malformed `xspecs`, and index panics on short `xlimits`.
- `GpMix.fit` / `SparseGpMix.fit`: training-data shape checks.
- `Gpx.save` / `Gpx.load` and their `SparseGpx` equivalents: `extension().unwrap()`, and `unwrap()` on I/O and parse errors.
- Every `predict*`, `sample` and `update` call: `.unwrap()` / `.expect()`.
- `Egor`:
  - `.expect("Egor configured")` and `.expect(...)` on `run()`;
  - `unwrap()` when casting the value returned by `fun`;
  - `panic!` when an `fcstrs` callback fails.
- `from_bits(..).unwrap()` on invalid `regr_spec` / `corr_spec`.

**Fix applied:**
- These paths now raise `ValueError`, `TypeError`, `OSError` or `RuntimeError`.
- An exception raised inside an `fcstrs` callback is re-raised unchanged.
- `save()` raises on failure and still returns `True` on success.

## 2. The stubs are incomplete or out of sync with the runtime — ✅ addressed

IDE help and type checking get these wrong:

- **Missing constructors.** `GpConfig`, `QEiConfig` and `TregoConfig` have no `#[gen_stub_pymethods]`
  on their `#[new]`. Their `__init__` signatures and defaults are absent from `egobox.pyi`.
- **Wrong member names.** The stub lists `SparseMethod.Fitc` / `Vfe`, but the runtime names are `FITC` / `VFE`.
- **`Any` where the types are known.** `xspecs`, `trego`, `hot_start`, `verbose` and `run_info` are
  all typed `typing.Any`. Suggested types:
  - `xspecs`: `Domain = Sequence[XSpec] | Sequence[Sequence[float]] | NDArray`;
  - `trego`: `TregoConfig | bool | dict | None`;
  - `verbose`: `Verbose | int | None`.
- **Undocumented dict forms.** `gp_config`, `qei_config`, `trego`, `run_info` and `CstrSpec` also accept
  dicts. This is useful but invisible; expose it in the type hints, e.g. `GpConfig | GpConfigDict` with a `TypedDict`.
- **Unhelpful repr.** Classes print as `<builtins.GpConfig object at …>`. Add
  `#[pyclass(module = "egobox")]`, and a `__repr__` on the configs, `OptimResult`, `EgorOptim` and `RunStatus`.

**Fix applied:**
- `GpConfig`, `QEiConfig` and `TregoConfig` constructors, with their defaults, are in the stub.
- The `SparseMethod` members are `FITC` / `VFE` in the stub. The fix was attribute order: `#[gen_stub_pyclass_enum]`
  must come before `#[pyclass(rename_all = ...)]`.
- `typing.Any` is gone from public signatures, except as the value type in `dict[str, Any]`. The stub types come from
  `#[gen_stub(override_type(...))]`: the domain union for `xspecs`, `TregoConfig | bool | dict | None` for `trego`,
  `Verbose | int | None` for `verbose`, and so on, plus callable types for `fun` and `fcstrs`.
- `gp_config`, `qei_config`, `fcstrs` and `fcstr_specs` now default to `None`, so the stub shows no
  unrepresentable default and no mutable `[]`. Passing `None` explicitly now works too.
- All classes report `__module__ == "egobox"`. The configs, `RunInfo`, `RunStatus`, `OptimResult` and `EgorOptim`
  now have a constructor-style `__repr__`.
- Checks: `tests/test_stubs.py` compares the stub's enum members and parameter names with the runtime, and forbids
  `typing.Any`. `python -m mypy.stubtest egobox.egobox --allowlist stubtest_allowlist.txt` passes; the allowlist
  covers only pyo3 artefacts.
- Possible follow-up: `TypedDict`s for the dict forms. pyo3-stub-gen cannot generate them, so they would need a
  hand-maintained stub.

## 3. Silent or misleading behaviour — ✅ addressed

- **`max_eval` does nothing in `GpMix`.** It is accepted but never passed on in `GpMix.fit`. Only `Egor`
  forwards it, in `Egor::apply_config`.
- **Result shapes are documented wrong.** `OptimResult.x_opt` / `y_opt` are 1D, `(nx,)` and `(ny,)`.
  The docs say `array[1, nx]`, and `y_opt` is also described as having `nx` components.
- **Two defaults for the run name.** `RunInfo()` defaults `fname` to `"fobj"`, but
  `minimize(run_info=None)` uses `"objective_function"`. Pick one.
- **`target=-1.797e308` shows up as a sentinel in the signature.** `target: float | None = None` would be clearer.

**Fix applied:**
- `GpMix.fit` now forwards `max_eval`. It caps the likelihood evaluations of each hyperparameter optimization start,
  whose budget is `clamp(10 * nx, 25, max_eval)`: it only has an effect when `max_eval < 10 * nx`.
- `x_opt` / `y_opt` are documented as `array[nx]` / `array[ny]`, with `ny = 1 + n_cstr`, in `minimize`,
  `get_result` and the `egobox` skill. `suggest` is documented as returning `array[batch, nx]`, one row per qEI point.
- `RunInfo()`, the `run_info` dict form and `minimize(run_info=None)` all default `fname` to `"objective_function"`,
  the Rust core default.
- `Egor(target=None)` is the default and means no target. Passing a float works as before.

## 4. Naming consistency — ✅ addressed in docs

| Issue | Where | Suggestion |
|---|---|---|
| `n_start` means two different things | `Egor(n_start=20)` is infill multistart; `GpConfig(n_start=10)` is hyperparameter multistart | `n_infill_starts` / `n_theta_starts`, or keep the names and cross-reference them in the docs |
| `max_iters` vs `max_eval` | `minimize` / `GpConfig` | Fine in themselves, but give GP options a `theta_` prefix for consistency (`theta_init`, `theta_bounds`, `theta_n_start`, `theta_max_eval`) |
| Heavy abbreviations | `cstr`, `fcstr`, `btw`, `optmod`, `n_gl_steps`, `d`, `nz`, `z`, `coego_n_coop`, `regr_spec` | Keep the short names as aliases, but consider `CstrSpec.between`, `QEiConfig.optim_every`, `TregoConfig.n_global_local_steps` / `radius_bounds`, `SparseGpMix.n_inducing` / `inducing` |
| `get_` prefix | `Egor.get_result`, `get_result_index` | `best_result(x, y)` / `best_index(y)` |
| `seed` and `verbose` live in different places | `Egor.minimize(seed, verbose)` vs `GpMix(seed, verbose)` | Choose one convention (constructor or call) for both classes |
| `doe`/`n_doe` vs `x_doe`/`y_doe` | `Egor.__init__` vs `suggest` | Both are fine, but `doe` packs x and y into one matrix. Consider `Egor(doe_x=..., doe_y=...)` |
| Enum casing in docstrings | `Recombination.Smooth`; `ConstraintStrategy.MeanValue` / `UpperTrustedBound` in the `Egor` docstring | The real names are `SMOOTH`, `MC`, `UTB` |
| Cryptic enum names | `ConstraintStrategy.MC/UTB`, `QEiStrategy.KBLB/CLMIN`, `FeasibleInfillStrategy.EFI_FE` | Add long aliases such as `MEAN_CONSTRAINT`, `UPPER_TRUST_BOUND`, `KRIGING_BELIEVER` |
| Unclear sampling names | `Sampling.LHS` vs `LHS_MAXIMIN` / `LHS_CLASSIC` | Document what plain `LHS` is (optimized?). `lhs(xspecs, n)` and `sampling(method, xspecs, n)` also order their arguments differently; `sampling(xspecs, n, method=Sampling.LHS)` would make `lhs` a thin alias |

**Fix applied (docs only, no new names):**
- `Egor(n_start)` and `GpConfig(n_start)` docs point to each other. `GpConfig(max_eval)` and `minimize(max_iters)` too.
- `Egor.minimize` and `GpMix` docs say where `seed` and `verbose` go for each class.
- `Egor(doe)` doc points to the separate `x_doe` / `y_doe` of `suggest`.
- Docstrings use the real enum names (`SMOOTH`, `HARD`, `MC`, `UTB`). Each member of `ConstraintStrategy`,
  `QEiStrategy`, `FeasibleInfillStrategy` and `Sampling` says what its abbreviation stands for.
- `Sampling.LHS` is documented as the optimized (ESE) LHS, and `lhs(xspecs, n)` as `sampling(Sampling.LHS, xspecs, n)`.
- Abbreviations are spelled out in parameter docs: `btw` (between), `optmod`, `n_gl_steps`, `d`, `nz`, `z`.
- Aliases and renames (long enum names, `CstrSpec.between`, `best_result`, `theta_` prefix, `doe_x` / `doe_y`,
  `sampling` argument order) are left to the breaking release (see Suggested order, step 5).

## 5. Structural and ergonomic points

- **`RegressionSpec` / `CorrelationSpec` are bare int holders.** Making them `enum.IntFlag` would give a
  readable repr, `|` composition and type safety while staying int-compatible.
- **Constraint settings are spread across constructor and call.**
  - `n_cstr`, `cstr_tol`, `cstr_specs`, `cstr_infill` and `cstr_strategy` are set on `Egor`.
  - `fcstrs` and `fcstr_specs` are passed to `minimize`.
  - Yet `cstr_tol` must also cover the expanded function constraints, which the constructor can't know.
    `n_cstr` is redundant once `cstr_specs` is set.
  - Suggested fix: a `CstrConfig` object, or `CstrSpec(kind, bound, tol=...)` so that each constraint carries its own tolerance.
- **Overlapping constraint options.** `cstr_infill` (a probability-of-feasibility factor) and
  `FeasibleInfillStrategy.EFI_P` look similar from the outside. The docs should say how they interact.
- **The `fcstrs` callback is unusual.** Its signature is `g(x, return_grad) -> float | ndarray`. A
  scipy-like form, `fcstrs=[(g, grad_g)]` or a `{"fun": g, "jac": grad_g}` dict, would feel more familiar.
- **Nested result access.** `res.result.x_opt` is one level deeper than users expect. Forward `x_opt`,
  `y_opt`, `x_doe` and `y_doe` directly on `EgorOptim`, and eventually support unpacking: `x, y = res`.
- **Two ways to build a model.** `GpMix(...)` and `Gpx.builder(...)` both exist, and their two
  12-parameter signatures are maintained by hand. Consider `GpMix(xspecs, gp_config=GpConfig(...), seed=...)`
  so that it matches `Egor`.
- **Inconsistent shape handling in `Gpx`:**
  - `fit` accepts 1D `x` and `(n, 1)` `y`;
  - `predict` requires 2D `x`;
  - `update` requires 1D `y` and rejects `(n, 1)`.

  Accept the same shapes everywhere. An sklearn-style `predict(x, return_std=True)` would also help.
- **`dims()` returns a tuple.** Properties `nx` / `ny` would be more idiomatic. The same applies to
  `thetas()`, `variances()` and `likelihoods()`, which are read-only data.
- **`SparseGpx` lags behind `Gpx`.** It has no `dims`, `training_data` or `update`. Its docstring:
  - documents `n_clusters` and `recombination`, which are not parameters;
  - leaves out `nz`, `z`, `theta_init` and `theta_bounds`;
  - refers to a nonexistent `GpSparse`.
- **Docstring style is mixed.** Some use Rust-style `# Parameters` blocks, others numpydoc
  `Parameters\n----------` blocks. Pick numpydoc so that Sphinx and IDEs render them.

## 6. Doc typos and errors (quick fixes) — ✅ addressed

- **Wrong copy-paste.** The `QEiStrategy` docstring says it is "for handling constraints". The
  `QEiConfig` doc refers to `q_optmod`, but the field is `optmod`.
- **Citations to check** in `InfillStrategy`:
  - WB2 is the Watson & Barnes criterion, but the docstring says "Warnes and Barnes (2020)".
  - LogEI is Ament et al. 2023, *Unexpected Improvements to Expected Improvement*. The docstring gives 2020 and a different title.
- **Spelling:** "responsability", "hyperpameters", "objecctive", "documention", and "peek at the same point twice" (should be "picked").

**Fix applied:**
- `QEiStrategy` is described as the qEI batch selection strategy; `QEiConfig` refers to `optmod`.
- WB2 cites Watson & Barnes (1995), LogEI cites Ament et al. (2023).
- Spelling fixed ("objecctive" was already gone).
- `SparseGpx.builder` refers to `SparseGpMix`, whose doc lists its real parameters (`theta_init`, `theta_bounds`,
  `nz`, `z`) and no longer `n_clusters` / `recombination`.
- The `Egor`, `GpMix`, `Gpx`, `SparseGpMix`, `SparseGpx`, `sampling` and `lhs` docstrings use the numpydoc layout,
  like the config classes (item 5, last point).
- `Egor` doc explains that `cstr_infill` and `feasible_infill_strategy` are independent (item 5, third point):
  the first weights the criterion by the probability of feasibility of the `n_cstr` constraints, the second by
  the probability of viability, i.e. of `fun` not failing.
- The `egobox` skill no longer documents nonexistent `XType.Float(lo, hi)` constructors, and gives `LOG_EI`
  as the default infill strategy.

---

## Suggested order

1. ✅ **Errors:** replace panics with Python exceptions.
2. ✅ **Stubs:** add `gen_stub_pymethods` to the configs, use real type hints, and set `module="egobox"` with `__repr__`s.
3. ✅ **Silent bugs:** `GpMix` ignoring `max_eval`, the result-shape docs, and the `RunInfo` default.
4. **Additive UX:** aliases, `IntFlag` specs and result forwarding. None of these break existing code.
5. **Breaking renames:** do these behind deprecation warnings in one release, grouping the constraint settings at the same time.

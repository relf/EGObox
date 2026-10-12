# EGMDO Technical Specification

This document specifies the EGMDO algorithm as it will be implemented in `egobox-egmdo`. It is derived from:

> S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, *Towards an efficient global multidisciplinary design optimization algorithm*, Struct. Multidisc. Optim. 62:1739–1765, 2020. [doi:10.1007/s00158-020-02514-6](https://doi.org/10.1007/s00158-020-02514-6)

Equation numbers `(n)` refer to the paper. Where the paper is silent or ambiguous, the choice made here is marked **Decision**, and collected in section 12. The integration plan is in `doc/egmdo_integration_plan.md`.

## 1. Problem

Find

```text
z* = argmin_{z ∈ Z} f_obj(z, y*_c(obj)(z))                                  (3)
```

where:

- `z ∈ Z ⊂ R^n` are the design variables, `Z` a box;
- `y*(z) = {y*_1(z), …, y*_nd(z)}` solves the multidisciplinary analysis (MDA)

  ```text
  y_i = f_i(z, y_c(i)),   i = 1, …, n_d                                      (4)
  ```

- `f_i` is the solver of discipline `i` (expensive), `c(i)` the indexes of the coupling variables it receives (`i ∉ c(i)`), and `c(obj)` the coupling variables used by the objective;
- `f_obj` is cheap and explicit;
- (4) is assumed to have at least one feedback loop and a unique solution for every `z ∈ Z`.

The paper writes `f_i(z, y_c(i))` with the full `z`. **Decision D1:** a discipline may use a subset of the design variables (as in the compound cylinder, section 11.3); its GP inputs are then that subset and its coupling inputs.

Each discipline `i` also has a box `C(i)` for its coupling inputs `y_c(i)`. It is a user-provided initial guess, used only to sample the initial disciplinary DoE. Enrichment points may fall outside it.

## 2. Disciplinary GPs

Each scalar coupling output is modelled by its own GP. The paper's examples have scalar couplings, so it writes one GP per discipline. **Decision D2:** a discipline with an output of dimension `m_i` owns `m_i` independent GPs, trained on the same inputs. In the following, `q = 1, …, n_ξ` indexes all scalar coupling GPs, with `n_ξ = Σ_i m_i`.

```text
Ỹ_q(z, y_c(i)) = μ_q(z, y_c(i)) + ε_q(z, y_c(i))                             (5)
```

GP settings (paper, "Replication of results"):

| Setting | Paper | EGObox |
|---|---|---|
| Trend | Constant | `ConstantMean` |
| Kernel | Squared exponential | `SquaredExponentialCorr`: `exp(-½ Σ (θ_l d_l)²)` |
| Input scaling | Centred and reduced | Built in (`xt_norm`) |
| Hyperparameters | Maximum likelihood, length scales in `[0.3, 100]` | `ThetaTuning::Full` with `θ = 1/ℓ ∈ [0.01, 3.33]` |

Predictions used by the algorithm: the mean `μ_q` (`predict`) and the standard deviation `σ_q = sqrt(predict_var)`. No sampling or gradient capability is needed.

Initial training data: `DoE_fi`, an LHS of size `n_fi` over `Z_i × C(i)`, evaluated with the true `f_i`.

## 3. Simplified random MDA

The paper replaces each GP by a perfectly correlated random field with a single standard normal variable:

```text
Ỹ'_q(z, y_c(i)) = μ_q(z, y_c(i)) + σ_q(z, y_c(i)) ξ_q,   ξ_q ~ N(0, 1)       (9)
```

with `Ξ = (ξ_1, …, ξ_nξ)` independent. For a given `z` and a given sample `Ξ^(j)`, (9) is a deterministic non-linear system, solved pointwise in `z`.

**Solver: non-linear Jacobi.** All disciplines are updated simultaneously from the previous iterate:

```text
y_q^(k+1) = μ_q(z, y_c(i)^(k)) + σ_q(z, y_c(i)^(k)) ξ_q,   q = 1, …, n_ξ
```

Convergence (paper): the mean relative change of the coupling variables between two iterations is below `10⁻⁶`:

```text
(1/n_ξ) Σ_q |y_q^(k+1) − y_q^(k)| / max(|y_q^(k)|, ε_abs) < 10⁻⁶
```

- **Decision D3:** `ε_abs = 1e-12` guards against division by zero.
- **Decision D4:** the starting point is the centre of the `C(i)` boxes; the iteration limit is 500 (configurable).
- **Decision D5:** a sample whose MDA does not converge (iteration limit, non-finite value) is excluded from the PCE regression sample of section 4, and counted. If fewer than `max(P, n_MC / 2)` samples remain, the PCE at that point is flagged as unreliable and reported; the run continues.

The same solver gives:

- the **mean MDA**, with `Ξ = 0`, used for enrichment (section 8);
- the **true MDA**, with the true `f_i` in place of the GPs, used only for validation and benchmarks.

Surrogate MDAs only call GPs, never the true disciplines.

## 4. Polynomial chaos expansion of the objective

At a design point `z`, the random objective `Y_obj(z, Ξ) = f_obj(z, Y*_c(obj)(z, Ξ))` (11) is approximated by

```text
Ŷ_obj(z, Ξ) = Σ_{j=1..P} a_j(z) H_j(Ξ)                                       (12)
```

- `H_j` are the **orthonormal** multivariate Hermite polynomials (probabilists' Hermite, normalized): `H_α(Ξ) = Π_q He_{α_q}(ξ_q) / sqrt(α_q!)`, with multi-indices `α` of total degree `|α| ≤ d`. `H_1 = 1`.
- `P = (n_ξ + d)! / (n_ξ! d!)`. Default `d = 3`. With `n_ξ = 2`, `P = 10`.
- Coefficients by ordinary least squares on `n_MC` samples `Ξ^(k) ~ N(0, I)` (paper: `n_MC = 100`). With `Ψ_kj = H_j(Ξ^(k))` and `Y_k = f_obj(z, y*(z, Ξ^(k)))`, solve `min ||Ψ a − Y||²` by QR (`linfa_linalg::qr::LeastSquaresQr`).
- Orthonormality gives `E[Ŷ_obj(z)] = a_1(z)` and `Var[Ŷ_obj(z)] = Σ_{j≥2} a_j(z)²`.

**Decision D6:** the regression sample `{Ξ^(k)}` is drawn once per run and shared by all design points (common random numbers). The paper does not say whether it redraws the sample per point; sharing it removes sampling noise between points and makes runs reproducible.

**Decision D7:** `n_MC` must be at least `P + 1`; the default is `max(100, 2P)`.

The ordering of multi-indices is fixed: graded, then reverse lexicographic. It only matters for tests and serialization.

## 5. Karhunen–Loève decomposition and KL-GP random field

Let `DoE_UQ = {z^(1), …, z^(n_UQ)}` be the design points where a PCE is available. For each `j`, gather the coefficient vector `a_j = (a_j(z^(1)), …, a_j(z^(n_UQ)))ᵀ`.

```text
μ_Ŷ = a_1,          K_Ŷ = Σ_{j=2..P} a_j a_jᵀ          (n_UQ × n_UQ)
```

With `(λ_k, φ̂_k)` the eigenpairs of `K_Ŷ` in decreasing order (`eigh`, `linfa-linalg`), the discretized random field is

```text
Ŷ_obj(Ξ) = μ_Ŷ + Σ_{k=1..M} ( Σ_{j=2..P} a_jᵀ φ̂_k H_j(Ξ) ) φ̂_k               (13)
```

- `M` is the smallest number of modes with `Σ_{k≤M} λ_k / Σ_k λ_k > 1 − 10⁻⁶` (paper). **Decision D8:** if `Σ_k λ_k = 0` (no disciplinary uncertainty left), `M = 0`.
- Write `b_kj = a_jᵀ φ̂_k` and `ζ_k(Ξ) = Σ_{j≥2} b_kj H_j(Ξ)`.

**KL-GPs.** The mean vector and each eigenvector are interpolated in `z` by GPs with the same settings as section 2, trained on `DoE_UQ`:

- `μ̃(z)`: GP of `(z^(i), μ_Ŷ[i])`, with mean `m_0(z)` and standard deviation `s_0(z)`;
- `φ̃_k(z)`: GP of `(z^(i), φ̂_k[i])`, with mean `m_k(z)` and standard deviation `s_k(z)`, `k = 1, …, M`.

The continuous random field is

```text
Ỹ_obj(z, Ξ, η) = m_0(z) + s_0(z) η_0 + Σ_{k=1..M} ζ_k(Ξ) (m_k(z) + s_k(z) η_k)   (14)
```

with `η = (η_0, …, η_M) ~ N(0, I)` independent of `Ξ`. `Ξ` models the disciplinary GP uncertainty; `η` models the KL-GP interpolation uncertainty. As in (9), each KL-GP is represented pointwise by one standard normal variable.

Eigenvectors are defined up to a sign. A sign flip changes both `ζ_k` and `φ̃_k`, so (14) is unaffected. Tests must not depend on the sign.

The final model of the objective uses the KL-GP means only:

```text
Ŷ_obj(z, Ξ) ≈ m_0(z) + Σ_{k=1..M} ζ_k(Ξ) m_k(z)                              (19)
```

## 6. Step 1: where to reduce uncertainty (expected improvement)

The minimum of the discretized random field over `DoE_UQ` is

```text
Ŷ_min(Ξ) = min_{i=1..n_UQ} Ŷ_obj(z^(i), Ξ)                                    (15)
```

evaluated with the PCE (12) at each point. The expected improvement over both uncertainty sources is

```text
EI(z) = E[ (Ŷ_min(Ξ) − Ỹ_obj(z, Ξ, η))⁺ ]                                     (16)
z_new = argmax_{z ∈ Z} EI(z)                                                  (17)
```

EI is not available analytically (the field is not Gaussian) and is estimated by Monte Carlo over `(Ξ, η)`.

- **Decision D9:** `N_EI` samples `(Ξ^(s), η^(s))` are drawn once per iteration and reused for every evaluation of `EI(z)` during its maximization (common random numbers). `EI` is then a deterministic, piecewise smooth function of `z`. `Ŷ_min(Ξ^(s))` is computed once per iteration. Default `N_EI = 2000` (the paper defers to Dubreuil et al. 2018 and gives no value).
- Optimization (paper): COBYLA on `Z` mapped to `[0, 1]^n`, `rhobeg = 0.5`, 20 starting points drawn uniformly; keep the best result.

In theory `EI(z^(i)) = 0` at `DoE_UQ` points. In practice it is small but not zero, because (15) uses the full PCE while (14) uses the truncated KL. This is expected (paper, Fig. 9).

`z_new` is added to `DoE_UQ` and its PCE is computed.

## 7. Step 2: where to enrich the disciplinary GPs (Algorithm 1)

**Probability of being the minimizer** of the discrete problem:

```text
P_min(z^(i)) ≈ (1/n_P) Σ_k 1[ argmin_l Ŷ_obj(z^(l), Ξ^(k)) = i ]              (18)
```

estimated with `n_P` fresh samples of `Ξ` on the PCEs, which are cheap. **Decision D10:** default `n_P = 10⁴`; ties are broken by the lowest index.

**Coefficient of variation** at a design point, from the PCE moments:

```text
cv(z) = sqrt(Var[Ŷ_obj(z)]) / |E[Ŷ_obj(z)]|
```

and `cv(z) = sqrt(Var[Ŷ_obj(z)])` when `E[Ŷ_obj(z)] = 0`. **Decision D11:** the paper writes `E[·]` without absolute value. With a negative mean (the modified Sellar optimum is −2.808), the ratio would always be negative and enrichment would never happen. The absolute value is clearly intended. **Decision D12:** "`E = 0`" is tested as `|E| < 1e-12`.

**Algorithm 1** (default `ε_cv = 0.01`):

```text
compute PCE at every z ∈ DoE_UQ, then P_min
loop
    Z_can ← { z^(i) ∈ DoE_UQ : P_min(z^(i)) ≥ 1/n_UQ }, sorted by decreasing P_min
    added ← false
    for z in Z_can
        if cv(z) ≤ ε_cv: continue
        enrich the disciplinary GPs at z (section 8)
        recompute the PCE at every point of DoE_UQ, then P_min
        added ← true
        break
    if not added: stop
```

It stops when every candidate (`P_min ≥ 1/n_UQ`) has `cv ≤ ε_cv`, i.e. when `Ŷ_min` restricted to `DoE_UQ` is almost a Dirac distribution.

The paper has no safeguard on the number of enrichments.

- **Decision D13:** an optional budget of disciplinary evaluations (per run) stops Algorithm 1 and Algorithm 2 when reached.
- **Decision D14:** if an enrichment point duplicates an existing training point of a discipline (within `1e-10` in normalized inputs), it is not added. The design point is then skipped for the rest of this Algorithm 1 call, so that the loop cannot repeat forever.

## 8. Enrichment of the disciplinary GPs

To enrich at a design point `z_e` (paper, section 3.4.3):

1. Solve the **mean MDA** (`Ξ = 0`) at `z_e`; this gives `y_c`.
2. For **every** discipline `i`, evaluate the true solver `f_i(z_e, y_c(i))`.
3. Add each point to `DoE_fi` and retrain the GPs of every discipline.

All disciplines are enriched each time; each enrichment therefore costs one evaluation per discipline. Choosing only the disciplines that drive the uncertainty is a perspective of the paper (Stage E of the plan).

In ask/tell mode, step 2 is the request returned by `ask()`: one input point per discipline, all of which must be answered by `tell()` before the algorithm resumes.

## 9. EGMDO (Algorithm 2)

```text
initialization
    for each discipline i: DoE_fi ← LHS(n_fi) over Z_i × C(i); evaluate f_i; train GPs
    DoE_UQ ← LHS(n_UQ0) over Z
    compute the PCE at every z ∈ DoE_UQ
for n_iter = 1..n_max
    KL decomposition of the PCE coefficient vectors over DoE_UQ           (13)
    KL-GPs of the mean and of the M eigenvectors                          (14)
    z_new ← argmax EI(z)                                                  (16, 17)
    DoE_UQ ← DoE_UQ ∪ {z_new}; compute the PCE at z_new
    Algorithm 1 (may enrich the disciplinary GPs)
final
    KL decomposition and KL-GPs on the final DoE_UQ
    return the model (19) and the estimates of section 10
```

As in the paper's flowchart (Fig. 17), Algorithm 1 is not run before the first EI step.

Both initial DoEs can also be given by the user instead of sampled (the 1-D toy problem uses a fixed `DoE_UQ`). For a user-given `DoE_fi`, the outputs can be provided too, to reuse existing evaluations.

**Stopping.** The paper stops on `n_max` only and monitors the maximum EI. **Decision D15:** the first version stops on `n_max` or on the evaluation budget (D13). A stopping criterion on the EI value is an extension; the maximum EI of every iteration is recorded in the results.

## 10. Results

Using the model (19), draw `n_sim` samples `Ξ^(s)` (paper: 100) and compute

```text
Ẑ*(Ξ^(s)) = argmin_{z ∈ Z} Ŷ_obj(z, Ξ^(s))                                    (21)
Ŷ_min(Ξ^(s)) = min_{z ∈ Z} Ŷ_obj(z, Ξ^(s))                                    (20)
```

**Decision D16:** each minimization is a multistart COBYLA started from the best points of `DoE_UQ` for this sample and from random points (default: 5 + 5 starts).

The result reports:

- mean, standard deviation and coefficient of variation of each component of `Ẑ*` and of `Ŷ_min`, and the samples themselves (for histograms);
- the number of true evaluations per discipline (initial DoE and added points);
- per iteration: `z_new`, maximum EI, `M`, number of enrichments, number of non-converged random MDAs;
- the final disciplinary GPs, `DoE_UQ`, PCE coefficients and KL-GPs, so the model (19) can be evaluated afterwards.

The paper considers a run converged when the relative error between `E[Ẑ*]` and the reference is below 5%. **Decision D17:** the relative error is `||E[Ẑ*] − z_ref|| / ||z_ref||` (Euclidean norm); several reference components are 0, so a per-component relative error is not usable.

## 11. Test problems

### 11.1 1-D toy problem (paper, section 2)

```text
f_obj(z, y1, y2) = cos((y1 + exp(−y2)) / π) + z / 20
y1 = f1(z, y2) = z² − cos(y2 / 2)
y2 = f2(z, y1) = z + y1
Z = [−5, 5],  C(1) = [0, 25] (for y2),  C(2) = [0, 25] (for y1)
```

- Reference: `z* ≈ −3.0`, `f_obj(z*) ≈ −1.15`.
- Paper setting: `n_f1 = 5`, `n_f2 = 4`, `DoE_UQ = {−4.3, −2.5, 0.5, 3.7}`, 5 iterations.
- Paper results:
  - the first EI step gives `z_new ≈ −3.5`;
  - Algorithm 1 then adds 5 points (2 at −3.5, 2 at −2.5, 1 at 3.7);
  - the next 4 iterations add `z ≈ −2.95, 5, 1.89, −5` without enrichment.
- Use: unit tests of each step, and qualitative comparison with Figs. 3–16.

### 11.2 Modified Sellar problem (paper, section 4.1)

Unconstrained variant of Sellar et al. (1996):

```text
f_obj(z, y1, y2) = z1 + z3² + y1 + exp(−y2) + 10 cos(z2)
y1 = f1(z, y2) = z1 + z2² + z3 − 0.2 y2
y2 = f2(z, y1) = sqrt(y1) + z1 + z2
Z = [0, 10] × [−10, 10] × [0, 10],  C(1) = [−5, 24] (for y2),  C(2) = [1, 50] (for y1)
```

- Reference (MDF + SLSQP): `z* ≈ (0, 2.634, 0)`, `f* ≈ −2.808`. Local minimum: `(0, −2.595, 0)`, `f ≈ −0.809`.
- Paper setting: `n_f1 = n_f2 = 5`, `n_UQ0 = 20`, `d = 3`, `n_MC = 100`, `n_max = 10`, `ε_cv = 0.01`.
- Paper results:
  - single run: 6 points added, i.e. 11 evaluations per discipline, `M = 6`, `E[Ẑ*] = (≤1e-12, 2.634, ≤1e-12)`, `E[Ŷ_min] = −2.798`;
  - 10 runs: 7 converged, 12 evaluations per discipline on average;
  - 100 runs with `n_max = 15`: 88 converged, 13 evaluations per discipline on average, `E[Ŷ_min] = −2.788`.
- Paper baselines (Table 6, convergence rate / evaluations per discipline): MDF-EGO-EI 96% / 296, MDF-EGO-WB2s 100% / 212, IDF-EGO-WB2s 100% / 44.

**Decision D18:** `sqrt(y1)` is not defined for `y1 < 0`, which an enrichment point computed by the mean MDA could request. The test problem uses `sqrt(max(y1, 0))`. Surrogate MDAs are not affected since they never call `f2`.

### 11.3 Compound cylinder (paper, section 4.2)

Two coupled cylinders under internal pressure `P0`. The design variables are `z = (a, b, c, δ) ∈ [5, 7] × [9, 11] × [12, 15] × [6e−3, 1e−2]`. The constants are `E = 3e7`, `ρ = 0.3` (Poisson ratio) and `P0 = 40000`. The couplings are the contact pressure `p` and the radial deformation `δ1`:

```text
δ1 = f1(a, b, p)     = (p b / E) ((a² + b²)/(b² − a²) − ρ)
p  = f2(b, c, δ, δ1) = (δ − δ1) E / b / ((b² + c²)/(c² − b²) + ρ)               (22)
```

Discipline 1 has 3 inputs `(a, b, p)` and discipline 2 has 4 inputs `(b, c, δ, δ1)` (paper), hence decision D1. `C(1) = [1000, 9000]` (for `p`) and `C(2) = [0, 6e−3]` (for `δ1`).

Tangential stress and objective:

```text
σ_a  = −2 p b²/(b² − a²) + (a² + c²) P0/(c² − a²)
σ_b− = −p (b² + a²)/(b² − a²) + a² P0 (b² + c²)/(b² (c² − a²))
σ_b+ =  p (b² + c²)/(c² − b²) + a² P0 (b² + c²)/(b² (c² − a²))
σ_c  =  2 b² p/(c² − b²) + 2 a² P0/(c² − a²)
B1 = (σ_a − σ_b−) a² b²/(a² + b²),   A1 = σ_a − B1/a²
B2 = (σ_b+ − σ_c) b² c²/(b² + c²),   A2 = σ_b+ − B2/b²
σ_θ(r) = A1 + B1/r²  for r ∈ [a, b],   A2 + B2/r²  for r ∈ (b, c]           (23)
f_obj(z, p) = sqrt(Var_r[σ_θ]) / E_r[σ_θ]                                     (24)
```

The paper's typesetting of (23) contains typos: it prints `B/r` instead of `B/r²` and `A2 = σ_b+ − B2/2²`. The reading above was checked numerically:

- it reproduces the four stress values of Fig. 25 (configuration `a = 6, b = 10, c = 13, δ = 8e−3`): `σ(a) = 4.92e4`, `σ(b⁻) = 3.58e4`, `σ(b⁺) = 4.47e4`, `σ(c) = 4.17e4`;
- with `E_r` and `Var_r` taken over `r` uniform on `[a, c]`, it gives `f_obj = 2.336e−2` at the paper's reference `z* ≈ (7, 9, 12, 6.37e−3)` (paper: `2.33e−2`);
- MDF + SLSQP with this reading finds `z ≈ (7, 9.11, 12, 6.5e−3)`, `f = 2.319e−2`;
- the paper's starting point `z0 = (6, 8, 11, 8e−3)` lies outside `Z` and is a typo; at the Fig. 25 configuration `f_obj = 7.40e−2` (paper: `7.00e−2`).

**Decision D19:** the benchmark reference for this problem is recomputed with our implementation of (22)–(24) and documented with the tolerance used. `E_r` and `Var_r` are computed with a fine uniform grid on `[a, c]`.

- Paper setting: `n_fi ∈ {d_i + 1, 2d_i, 3d_i, 4d_i}` (`d_1 = 3`, `d_2 = 4`), `n_UQ0 = 20`, `d = 3`, `n_MC = 100`, `n_max = 20`, `ε_cv = 0.01`.
- Paper results (10 runs per DoE size, Table 7): 6/10 runs converged with `d_i + 1` and 8 to 9 with larger DoEs; 3 to 8 points added on average.

## 12. Decisions summary

| Id | Topic | Decision |
|---|---|---|
| D1 | Discipline inputs | A discipline may use a subset of the design variables |
| D2 | Vector couplings | One GP and one `ξ` per scalar coupling output |
| D3 | MDA convergence | `ε_abs = 1e-12` in the relative change |
| D4 | MDA start / limit | Centre of the `C(i)` boxes; 500 iterations |
| D5 | Failed random MDA | Excluded from the regression, counted and reported |
| D6 | PCE regression sample | Drawn once per run, shared by all design points |
| D7 | Regression size | `n_MC ≥ P + 1`, default `max(100, 2P)` |
| D8 | No uncertainty left | `M = 0` |
| D9 | EI estimation | Fixed samples per iteration, `N_EI = 2000` |
| D10 | `P_min` estimation | `n_P = 10⁴` fresh samples, ties to lowest index |
| D11 | Coefficient of variation | Absolute value of the mean |
| D12 | Zero mean | `|E| < 1e-12` |
| D13 | Budget | Optional budget of disciplinary evaluations |
| D14 | Duplicate enrichment | Not added; point skipped for the current Algorithm 1 call |
| D15 | Stopping | `n_max` or budget; maximum EI recorded |
| D16 | Final minimizations | Multistart COBYLA, 5 best `DoE_UQ` points + 5 random |
| D17 | Convergence metric | Euclidean relative error of `E[Ẑ*]`, threshold 5% |
| D18 | Sellar `sqrt` | `sqrt(max(y1, 0))` in the test problem |
| D19 | Cylinder reference | Recomputed with our implementation |

## 13. Default settings

| Parameter | Symbol | Default | Source |
|---|---|---|---|
| Initial disciplinary DoE size | `n_fi` | user-defined (`d_i + 1` minimum advised) | paper |
| Initial `DoE_UQ` size | `n_UQ0` | 20 | paper |
| PCE total degree | `d` | 3 | paper |
| PCE regression sample | `n_MC` | `max(100, 2P)` | paper, D7 |
| KL energy threshold | | `1 − 10⁻⁶` | paper |
| Enrichment threshold | `ε_cv` | 0.01 | paper |
| Max iterations | `n_max` | 10 | paper (10 to 20) |
| MDA tolerance | | `10⁻⁶` | paper |
| MDA iteration limit | | 500 | D4 |
| EI Monte Carlo samples | `N_EI` | 2000 | D9 |
| EI optimizer | | COBYLA, `rhobeg = 0.5` on `[0, 1]^n`, 20 starts | paper |
| `P_min` samples | `n_P` | 10⁴ | D10 |
| Final estimation samples | `n_sim` | 100 | paper |
| GP length-scale range | `ℓ` | `[0.3, 100]`, i.e. `θ ∈ [0.01, 3.33]` | paper |
| Seed | | user-defined, `None` for entropy | plan |

## 14. Mapping to the implementation

| Section | Module (`crates/egmdo`) | EGObox reuse | Tests |
|---|---|---|---|
| 1, 2 | `problem`, `disciplines` | `egobox-gp`, `egobox-doe::Lhs` | input mapping, DoE sizes and bounds, GP settings |
| 3 | `mda` | — | toy and Sellar true MDA vs analytical/fixed-point solutions; `Ξ = 0` equals mean MDA; failure counting |
| 4 | `pce` | `linfa-linalg` QR | orthonormality of the basis (Monte Carlo), exact recovery of polynomial targets, moments vs Monte Carlo |
| 5 | `kl` | `egobox-gp`, `linfa-linalg` `eigh` | energy truncation, reconstruction of `Ŷ_obj` at `DoE_UQ`, KL-GP interpolation, sign invariance |
| 6, 7 | `criteria` | COBYLA multistart (`egobox-ego` or `cobyla`) | EI ≈ 0 at `DoE_UQ`, EI ≥ 0, determinism with fixed samples, `P_min` sums to 1, `cv` with negative and zero mean |
| 7, 8, 9 | `solver` | — | Algorithm 1 stops when all candidates have `cv ≤ ε_cv`; one evaluation per discipline per enrichment; ask/tell sequence; reproducibility for a seed |
| 10 | `solver`, `result` | — | statistics of `Ẑ*` and `Ŷ_min` on a problem with a known minimum |
| 11 | `problems` (tests and examples) | — | end-to-end runs and reference results |

## 15. Known limitations

- The PCE size `P` grows combinatorially with the number of scalar couplings `n_ξ` (e.g. `P = 286` for `n_ξ = 10`, `d = 3`), and so does the number of random MDAs per design point. The method targets problems with few scalar couplings.
- The method relies on (4) having a unique solution and on Jacobi converging; strongly coupled problems may need relaxation, which is an extension.
- The coupling boxes `C(i)` must be guessed by the user. A poor guess degrades the initial disciplinary GPs, but enrichment corrects it.
- As in the paper, EI optimization is noisy and multimodal; the common random numbers of D9 make it deterministic but do not make it easy.

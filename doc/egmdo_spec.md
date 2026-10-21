# EGMDO Technical Specification

This document specifies the EGMDO algorithm and its constrained extension C-EGMDO as they will be implemented in `egobox-egmdo`. It is derived from:

> S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, *Towards an efficient global multidisciplinary design optimization algorithm*, Struct. Multidisc. Optim. 62:1739–1765, 2020. [doi:10.1007/s00158-020-02514-6](https://doi.org/10.1007/s00158-020-02514-6)

> I. Cardoso, S. Dubreuil, N. Bartoli, C. Gogu, M. Salaün, *Constrained efficient global multidisciplinary design optimization using adaptive disciplinary surrogate enrichment*, Struct. Multidisc. Optim. 67:23, 2024. [doi:10.1007/s00158-023-03736-0](https://doi.org/10.1007/s00158-023-03736-0)

Part I (sections 1–15) specifies EGMDO, and part II (sections 16–23) the C-EGMDO extensions. Equation numbers `(n)` refer to the 2020 paper and `(C-n)` to the 2024 paper. Where the papers are silent or ambiguous, the choice made here is marked **Decision**, and collected in sections 12 and 22. The integration plan is in `doc/egmdo_integration_plan.md`.

# Part I — EGMDO

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
| Kernel | Squared exponential | Matérn 5/2 by default (`Matern52Corr`), squared exponential (`SquaredExponentialCorr`: `exp(-½ Σ (θ_l d_l)²)`) as an option (D34) |
| Input scaling | Centred and reduced | Built in (`xt_norm`) |
| Hyperparameters | Maximum likelihood, length scales in `[0.3, 100]` | `ThetaTuning::Full` with `θ = 1/ℓ ∈ [0.01, 3.33]` |

Predictions used by the algorithm: the mean `μ_q` (`predict`) and the standard deviation `σ_q = sqrt(predict_var)`. No sampling capability is needed. The gradient of the mean (`predict_gradients`) is used only by the linearized random MDA (D37).

**Decision D34 (kernel).** The default kernel is Matérn 5/2, for the disciplinary GPs and the KL-GPs; the papers' squared exponential is an option (`GpSettings::paper()`). Both kernels use `θ` as an inverse length scale, with the same bounds. With the small initial DoEs of the method, maximum likelihood often puts a squared exponential length scale at its upper bound, and the GP becomes overconfident. Its standard deviation then underestimates the error, so the coefficient of variation stays below `ε_CV` and Algorithm 1 does not enrich where it should. On the 1-D toy problem:

- with 4 to 5 training points, 5–31% of the errors fall within 2σ for most squared exponential fits, against 43–100% with Matérn 5/2, for a similar RMSE;
- in the paper's setting (initial DoEs of 5 and 4 points, the paper's `DoE_UQ`, 5 iterations, seeds 0 to 19), 17/20 runs converge with Matérn 5/2 against 13/20 with the squared exponential, for about one more true evaluation per run.

The C-EGMDO paper lists the choice of kernel as a likely improvement. Benchmarks reproducing the papers use the squared exponential.

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
- **Decision D31:** an optional relaxation `y^(k+1) = (1 − ω) y^(k) + ω F(y^(k))`, `ω ∈ ]0, 1]`. The default `ω = 1` is the paper's plain Jacobi method. Plain Jacobi diverges on part of the compound cylinder's design space: its iteration factor reaches −1.55 at `a = 7, b = 9, c = 15`. With `ω = 0.5`, every corner of `Z` contracts, so this test problem uses `ω = 0.5`.
- **Decision D33:** the resolutions which do not converge are retried from their starting point with stronger relaxation, `ω = 0.5` then `0.25` (only the factors smaller than the base `ω` are used). Random MDAs fail mostly near the edges of the design space: there the couplings leave the training boxes `C(i)`, the GPs extrapolate with large `σ`, and some realizations `μ + σ ξ` have iteration factors larger than 1 in magnitude, so plain Jacobi cycles. The paper's plain Jacobi is still used whenever it converges.
- **Decision D5:** a sample whose MDA does not converge (iteration limit, non-finite value), even after the retries of D33, is excluded from the PCE regression sample of section 4, and counted. If fewer than `max(P, n_MC / 2)` samples remain, the PCE at that point is flagged as unreliable and reported; the run continues.
- **Decision D32:** if no more than `P` random MDAs converge at a `DoE_UQ` point, its PCE cannot be fitted: the point is dropped from `DoE_UQ` with a warning, and counted in the iteration history. A proposed point `z_new` in that case is not added. The run fails only if fewer than 2 `DoE_UQ` points remain.

**Decision D37 (linearized random MDA).** As an option (`RandomMda::Linearized`, issue #493), the random MDA is linearized at the solution `y_0` of the mean MDA. With `J_qp = ∂μ_q/∂y_p` the Jacobian of the GP means with respect to the coupling variables and `σ_q` the GP standard deviations, both at `(z, y_0)`:

```text
y(z, Ξ) = y_0 + (I − J)⁻¹ diag(σ) Ξ
```

- This is the first-order expansion of (9) in `Ξ`. The variation of `σ`, or of the GP error, along `y` is a second-order term, so the full GP posterior has the same expansion: the linearization does not depend on the rank-one assumption at a given `z`.
- The quantities of interest are evaluated on these samples and their PCEs fitted as in section 4, on the same samples `Ξ^(k)` (D6). The rest of the method is unchanged.
- One mean MDA and the GP mean gradients replace the `n_MC` random MDAs at each design point. No random MDA is solved, so D5, D32 and D33 do not apply.
- The point falls back to the solved random MDAs where the mean MDA does not converge, or where `I − J` is ill-conditioned: the ratio of the extreme singular values of `S⁻¹(I − J)S` is below `10⁻⁸`, with `S = diag(|y_0| + σ)` so that the test does not depend on the units of the coupling variables (on the compound cylinder they differ by 10⁶).
- The approximation holds when `σ` is small with respect to the curvature of the GP means. It misses the non-Gaussian coupling variables of large uncertainties, as at the beginning of a run with small initial DoEs.

The default is the papers' solved random MDA.

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

**Decision D7:** `n_MC` must be at least `P + 1`; the default is `max(200, 2P)` (EGMDO used 100, C-EGMDO 200 to 500).

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

**Decision D38 (correlation across design points).** In (9), the variable `ξ_q` is shared by every design point, so the disciplinary uncertainty is perfectly correlated over `DoE_UQ`. As an option (`Correlation::GpPosterior`, issue #493), each point `z_i` has its own variables `u_q(i) ~ N(0, 1)`. The correlation of `u_q(i)` and `u_q(l)` is `ρ_q(i, l)`, the posterior correlation of GP `q` between its inputs at the mean MDA solutions of both points (the mean of the converged random MDAs where the mean MDA does not converge). The variables of different couplings stay independent.

- The PCE of each point is fitted in its own variables, on the same samples as before (D6). Their marginal law is unchanged, and so are the coefficients of variation and the Sobol indices (sections 7 and 20).
- The covariance of the PCEs over `DoE_UQ`, which replaces `Σ_j a_j a_jᵀ` in (13), is given by Mehler's formula for orthonormal Hermite polynomials:

  ```text
  K_il = Σ_{α≠0} a_α(i) a_α(l) Π_q ρ_q(i, l)^α_q
  ```

- Joint samples are `u_q = W_q R_q^{1/2}`, with `W_q` independent standard normal samples and `R_q^{1/2}` the symmetric square root of the correlation matrix (eigenvalues below `10⁻¹²` times the largest one set to zero). They replace the samples of `Ξ` in `Ŷ_min` (15), `P_min` (18), `f'_min` (C-20), `P_sol` (C-25) and the final estimation. Each PCE is evaluated at the variables of its own point.
- The KL random variables are the projections `ζ_k = φ̂_kᵀ (Ŷ − μ_Ŷ)` of the sampled vector. With shared variables, this is `Σ_j b_kj H_j(Ξ)`. The KL-GP field (14) and the criteria are unchanged.
- The correlations are recomputed whenever `DoE_UQ` or the disciplinary GPs change.
- With `ρ = 1` everywhere, this is the papers' model. `Correlation::Perfect` keeps its implementation, and its results are bit-identical.

The default is the papers' perfect correlation.

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

All disciplines are enriched each time; each enrichment therefore costs one evaluation per discipline. Choosing only the discipline that drives the uncertainty is a perspective of the 2020 paper, developed by C-EGMDO (section 20).

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

- Reference (MDF + SLSQP): `z* ≈ (0, 2.634, 0)`, `f* ≈ −2.808`. Local minimum: `(0, −2.595, 0)`, `f ≈ −0.809`. Recomputed: `z* = (0, 2.63449586, 0)`, `f* = −2.80852222`.
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
- with `E_r` and `Var_r` taken over `r` uniform on `[a, c]`, it gives `f_obj = 2.336362e−2` at the paper's reference `z* ≈ (7, 9, 12, 6.37e−3)` (paper: `2.33e−2`);
- the paper's starting point `z0 = (6, 8, 11, 8e−3)` lies outside `Z` and is a typo; at the Fig. 25 configuration `f_obj = 7.396061e−2` (paper: `7.00e−2`).

**Decision D19:** `E_r` and `Var_r` are computed exactly: on each segment, the integrals of `A + B/r²` and of its square have closed forms. The benchmark reference is recomputed with this objective (MDF + multistart SLSQP): `z* = (7, 9.10539021, 12, 6.53453308e−3)`, `f* = 2.31770023e−2`.

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
| D7 | Regression size | `n_MC ≥ P + 1`, default `max(200, 2P)` |
| D8 | No uncertainty left | `M = 0` |
| D9 | EI estimation | Fixed samples per iteration, `N_EI = 2000` |
| D10 | `P_min` estimation | `n_P = 10⁴` fresh samples, ties to lowest index |
| D11 | Coefficient of variation | Absolute value of the mean |
| D12 | Zero mean | `\|E\| < 1e-12` |
| D13 | Budget | Optional budget of disciplinary evaluations |
| D14 | Duplicate enrichment | Not added; point skipped for the current Algorithm 1 call |
| D15 | Stopping | `n_max` or budget; maximum EI recorded |
| D16 | Final minimizations | Multistart COBYLA, 5 best `DoE_UQ` points + 5 random |
| D17 | Convergence metric | Euclidean relative error of `E[Ẑ*]`, threshold 5% |
| D18 | Sellar `sqrt` | `sqrt(max(y1, 0))` in the test problem |
| D19 | Cylinder reference | Exact radial moments; reference recomputed |
| D31 | MDA relaxation | Optional `ω`, default 1 (plain Jacobi); 0.5 for the compound cylinder |
| D32 | Failing `DoE_UQ` point | Dropped when no more than `P` random MDAs converge; error below 2 points |
| D33 | MDA retries | Failed resolutions retried with `ω = 0.5`, then `0.25` |
| D34 | GP kernel | Matérn 5/2 by default; squared exponential (papers) as an option |
| D35 | WB2S | Adaptive scale `100 \|m_0\| / EI_max` at the maximum of the EI |
| D36 | LogEI | Smoothed Monte Carlo log (C)EI (qLogEI), temperature `10⁻⁶ σ` of the objective over `DoE_UQ` |
| D37 | Linearized random MDA | Optional `y_0 + (I − J)⁻¹ diag(σ) Ξ` at the mean MDA solution; fallback to the solved random MDAs; solved by default |
| D38 | Correlation across design points | Optional GP posterior correlation of the variables of each point, at the mean MDA solutions; Mehler covariance for the KL; perfect by default |

## 13. Default settings

| Parameter | Symbol | Default | Source |
|---|---|---|---|
| Initial disciplinary DoE size | `n_fi` | user-defined (`d_i + 1` minimum advised) | paper |
| Initial `DoE_UQ` size | `n_UQ0` | 20 | paper |
| PCE total degree | `d` | 3 | paper |
| PCE regression sample | `n_MC` | `max(200, 2P)` | papers, D7 |
| KL energy threshold | | `1 − 10⁻⁶` | paper |
| Enrichment threshold | `ε_cv` | 0.01 | paper |
| Max iterations | `n_max` | 10 | paper (10 to 20) |
| MDA tolerance | | `10⁻⁶` | paper |
| MDA iteration limit | | 500 | D4 |
| MDA relaxation | `ω` | 1 | D31 |
| MDA retry relaxations | | 0.5, 0.25 | D33 |
| GP kernel | | Matérn 5/2 | D34 |
| Random MDA | | solved | D37 |
| Correlation across design points | | perfect | D38 |
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
| 3 (D37) | `uq` | `linfa-linalg` SVD, `predict_gradients` | coupling Jacobian vs finite differences of the mean MDA map; solved random MDA equals `y_0 + t L Ξ` to first order in `t`; PCE moments close to the solved random MDA; fallback |
| 4 | `pce` | `linfa-linalg` QR | orthonormality of the basis (Monte Carlo), exact recovery of polynomial targets, moments vs Monte Carlo |
| 5 | `kl` | `egobox-gp`, `linfa-linalg` `eigh` | energy truncation, reconstruction of `Ŷ_obj` at `DoE_UQ`, KL-GP interpolation, sign invariance |
| 5 (D38) | `uq`, `kl`, `solver` | `predict_covariance`, `linfa-linalg` `eigh` | correlation matrices (unit diagonal, symmetric, PSD, larger for close points); correlation of the joint samples; Mehler covariance equals `Σ a_j a_jᵀ` for `ρ = 1` and the covariance of the joint samples; projected `ζ` equals the PCE form with shared variables; toy problem converges; `Perfect` bit-identical |
| 6, 7 | `criteria` | multistart COBYLA of `egobox_gp::basin_optimizer`, as `Egor` | EI ≈ 0 at `DoE_UQ`, EI ≥ 0, determinism with fixed samples, `P_min` sums to 1, `cv` with negative and zero mean |
| 7, 8, 9 | `solver` | — | Algorithm 1 stops when all candidates have `cv ≤ ε_cv`; one evaluation per discipline per enrichment; ask/tell sequence; reproducibility for a seed |
| 10 | `solver`, `result` | — | statistics of `Ẑ*` and `Ŷ_min` on a problem with a known minimum |
| 11 | `problems` (tests and examples) | — | end-to-end runs and reference results |

## 15. Known limitations

- The PCE size `P` grows combinatorially with the number of scalar couplings `n_ξ` (e.g. `P = 286` for `n_ξ = 10`, `d = 3`), and so does the number of random MDAs per design point. The method targets problems with few scalar couplings.
- The method relies on (4) having a unique solution and on Jacobi converging. Relaxation (D31) extends convergence to problems with oscillating iterations, as the compound cylinder; strongly non-linear couplings may need other solvers (e.g. Newton), which is an extension.
- The coupling boxes `C(i)` must be guessed by the user. A poor guess degrades the initial disciplinary GPs, but enrichment corrects it.
- As in the paper, EI optimization is noisy and multimodal; the common random numbers of D9 make it deterministic but do not make it easy.

# Part II — C-EGMDO

C-EGMDO keeps part I unchanged except for the acquisition step (section 6), the enrichment step (sections 7 and 8) and the results (section 10). The unconstrained EGMDO is the special case with no constraints, all-discipline enrichment, the EI criterion and `ε_P = 1/n_UQ`. The implementation has a single solver configured by options (section 23).

## 16. Constrained problem

```text
min_{z ∈ Z} f_obj(z, y*_c(obj)(z))
s.t. g_m(z, y*_c(gm)(z)) ≤ 0,   m = 1, …, n_g
     h_r(z, y*_c(hr)(z)) = 0,   r = 1, …, n_h                                 (C-16)
```

Like the objective, constraints are cheap, explicit functions of the design variables and of some coupling variables. They become random through the couplings.

**Decision D20 (sign convention).** C-EGMDO's equations (C-20, C-22, C-23) treat `g ≥ 0` as feasible, but its Sellar problem (C-28) is written with `g ≤ 0`. This specification uses `g ≤ 0`, as `Egor` does, and rewrites the paper's equations accordingly: feasibility is `g ≤ 0`, and the violation of an inequality is `max(g, 0)`.

**Decision D21 (tolerance).** As in `Egor` (`cstr_tol`, default `1e-4`), each constraint has a tolerance `τ`. Feasibility is `g_m ≤ τ_m` and `|h_r| ≤ τ_r`. Without a tolerance, an equality constraint evaluated on a PCE sample would never be satisfied exactly.

## 17. Random constraints

At each `DoE_UQ` point, the quantities of interest

```text
QoI(z, Ξ) = { f_obj^PCE(z, Ξ), g_m^PCE(z, Ξ), h_r^PCE(z, Ξ) }                    (C-18)
```

all get a PCE (section 4). They are computed from the **same** random MDA samples, so the regression matrix `Ψ` is shared and factorized once per point, with one right-hand side per quantity.

**Constraint models.** Constraints are not modelled by KL-GPs. Instead, the PCE mean `a_1^{g_m}(z^(i))` of each constraint is interpolated over `DoE_UQ` by a GP, with the same settings as section 2. Its mean `μ_gm(z)` is the constraint model used by the acquisition step (as in SEGO). The same applies to `μ_hr(z)`.

C-EGMDO also tested relaxing the constraints by interpolating a quantile of the constraint PCEs instead of the mean. It was less robust on Sellar (88 and 74 successful runs out of 100 for the 95% and 99% quantiles, against 96 for the mean). It is not part of this specification.

**Decision D22 (deterministic constraints).** A constraint can be declared as depending on `z` only (C-EGMDO section 3.4). It is then evaluated directly: it needs no PCE and no GP, and its feasibility is the same for every sample `Ξ`. Costly constraints of that kind, which would need their own surrogate, are an extension.

## 18. Step 1: constrained acquisition

**Feasible minimum.** For a sample `Ξ`, the current minimum is taken over the `DoE_UQ` points that are feasible for this sample:

```text
f'_min(Ξ) = min { f_obj^PCE(z^(i), Ξ) : g_m^PCE(z^(i), Ξ) ≤ τ_m, |h_r^PCE(z^(i), Ξ)| ≤ τ_r }   (C-20)
```

If no `DoE_UQ` point is feasible for this sample, `f'_min(Ξ)` is the objective at the point with the smallest constraint violation

```text
θ(z^(i), Ξ) = Σ_m max(g_m^PCE(z^(i), Ξ), 0) + Σ_r |h_r^PCE(z^(i), Ξ)|          (C-22)
```

Ties are broken by the lowest objective value. Without constraints, `f'_min(Ξ)` is `Ŷ_min(Ξ)` of (15).

**Criteria.**

```text
CEI(z) = E[ (f'_min(Ξ) − Ỹ_obj(z, Ξ, η))⁺ ]                                    (C-19)
WB2(z) = CEI(z) − m_0(z)                                                        (C-24)
```

`m_0(z)` is the mean of the KL-GP of the mean vector (section 5), i.e. the expectation of the objective model (19) over `Ξ`. CEI is estimated by Monte Carlo with the same fixed samples as EI (D9). WB2 is less multimodal than CEI and easier to maximize: on Sellar, it converged in 96 runs out of 100 against 63 for CEI. **Decision D23:** WB2 takes an optional scale on CEI (`scale · CEI − m_0`, default 1), as `Egor`'s `WB2Criterion` does. A scale other than 1 (WB2S) is not studied by the paper.

**Decision D35 (WB2S).** `Acquisition::Wb2s` adapts the scale at each iteration, as `Egor`'s WB2S (Bartoli et al. 2019): `scale = 100 |m_0(z_EI)| / EI_max`, where `EI_max` is the maximum of the EI under the constraint models, reached at `z_EI`. The scale is 1 when `EI_max` is negligible (below `100 ε`). `Egor` computes `EI_max` over the starting points of the infill optimization. Here the EI is zero almost everywhere once the disciplinary GPs are enriched, so the EI is maximized first; `z_EI` is then added to the starting points of the WB2 optimization. This costs one more multistart optimization per iteration.

**Decision D36 (LogEI).** `Acquisition::LogEi` maximizes the logarithm of a smoothed (C)EI, as the Monte Carlo qLogEI of Ament et al. (2023): `LogEI(z) = log((1/N) Σ_s τ softplus(Δ_s(z)/τ))` with `Δ_s(z) = f'_min(Ξ_s) − Ỹ_obj(z, Ξ_s, η_s)`, computed with a log-sum-exp over the same fixed samples as the EI (D9). Where some samples improve, it is close to `log EI`. Where none does, the EI estimate is exactly 0, and its flat zones stall the optimizer (C-EGMDO, section 4.1.1); the LogEI stays finite there, about `max_s Δ_s/τ`, and ranks the points by how close they come to improving. The analytic LogEI assumes a Gaussian prediction and does not apply to the non-Gaussian field (14). The temperature is `τ = 10⁻⁶ × σ`, where `σ` is the standard deviation of the objective PCE means over `DoE_UQ` (or their mean absolute value if it is 0), since the objective is not standardized.

**Optimization.**

```text
z_new = argmax_{z ∈ Z} CEI(z) or WB2(z)   s.t.  μ_gm(z) ≤ τ_m,  |μ_hr(z)| ≤ τ_r     (C-23)
```

The paper maximizes the criterion with a gradient-based multistart solver (SLSQP). **Decision D24:** multistart COBYLA, as in EGMDO. It handles the constraints natively and needs no gradient of a Monte Carlo estimate. Each equality constraint is passed as two inequalities.

**Duplicates (C-EGMDO, Remark 1).** A proposed point that is too close to an existing `DoE_UQ` point makes the KL-GP interpolation ill-conditioned. In that case, the next best multistart result is used. **Decision D25:** points are compared in `Z` normalized to `[0, 1]^n`, with a distance threshold of `1e-6`. If all multistart results are duplicates, no point is added at this iteration, and this is recorded. The same rule applies to the unconstrained EI.

## 19. Step 2: constrained enrichment (C-EGMDO Algorithm 2)

**Probability of being the solution.**

```text
P_sol(z^(i)) = P( f'_min(Ξ) = f_obj^PCE(z^(i), Ξ) )                             (C-25)
```

estimated with `n_P` samples as in (18).

**Decision D26.** The paper defines `f'_min` with a least-violation fallback (C-20, C-22). It also says that, with equality constraints, all `P_sol` values may be close to zero, and then falls back to the probability of being the least-violating point. These two statements are only compatible if, in `P_sol`, samples with no feasible point count for no point. If such samples counted for the least-violating point, the `P_sol` values would always sum to 1 and the fallback would never trigger. Hence:

- in `P_sol`, a sample counts for the point that realizes its feasible minimum; samples with no feasible point count for none;
- in CEI (section 18), the least-violation fallback of `f'_min` applies;
- if `P_sol(z^(i)) < ε_P` for every point, `P_sol` is replaced by `P( argmin_l θ(z^(l), Ξ) = i )` (ties to the lowest objective value).

**Coefficient of variation.** At each point, it is the largest one over the quantities of interest:

```text
CV(z^(i)) = max_j  sqrt(Var[QoI_j(z^(i))]) / |E[QoI_j(z^(i))]|                   (C-26)
```

with the absolute value of D11. **Decision D27:** for a constraint, the denominator is `max(|E|, τ / ε_CV)`. An active constraint has a mean close to zero, so its CV can stay large even when its standard deviation is negligible. With this floor, an active constraint counts as certain when its standard deviation is below its tolerance `τ`.

The floor was first set to `τ`, which requires a standard deviation below `ε_CV · τ = 10⁻⁶`; GP interpolation rarely reaches it. On the constrained Sellar problem (C-EGMDO setting, seeds 0 to 9), that floor led to 19.9 and 11.4 evaluations per discipline, against 12.8 and 10.0 with `τ / ε_CV` (paper: 13.0 and 9.8). Raising `τ` instead also loosens feasibility, and the best point degrades. With `τ / ε_CV`, over seeds 0 to 19 and Matérn 5/2 GPs, 19/20 runs converge within 1% with 24.4 evaluations in total on average (paper: 96/100 and 22.8). On the constrained toy problem, 17/20 runs converge, against 9/20 with the squared exponential kernel.

**Algorithm** (C-EGMDO defaults: `ε_CV = 0.01`, `ε_P = 10⁻³`):

```text
compute the PCEs of all QoI at every z ∈ DoE_UQ, then P_sol (with fallback) and CV
while ∃ z^(i) with P_sol(z^(i)) ≥ ε_P and CV(z^(i)) ≥ ε_CV
    X_enrich ← { z^(i) : P_sol(z^(i)) ≥ ε_P }, sorted by decreasing P_sol
    added ← false
    for z in X_enrich
        if CV(z) ≤ ε_CV: continue
        select the discipline(s) to enrich at z (section 20)
        enrich them (section 8, restricted to the selected disciplines)
        recompute the PCEs at every point of DoE_UQ, then P_sol (with fallback) and CV
        added ← true
        break
    if not added: stop
```

The safeguards D13 (evaluation budget) and D14 (duplicate disciplinary points) apply.

## 20. Discipline selection by Sobol indices

In EGMDO, every discipline is enriched each time. C-EGMDO enriches only the discipline that contributes most to the uncertainty (C-EGMDO section 3.3, after Berthelin et al. 2022):

1. At the enrichment point `z`, choose the quantity of interest with the largest CV (objective or constraint).
2. Compute the first-order Sobol index of each discipline for this quantity:

   ```text
   S_i(z) = Var[ E[QoI(z) | ξ_i] ] / Var[QoI(z)]                                 (C-27)
   ```

   With the orthonormal PCE of section 4, this is analytical (Sudret 2008). For a single variable `ξ_q`, `S_q = Σ_{α : supp(α) = {q}} a_α² / Σ_{α ≠ 0} a_α²`.
3. Enrich only the discipline with the largest index. Ties go to the lowest discipline index.

**Decision D28 (vector couplings).** In the paper each discipline has one random variable. With D2, discipline `i` owns a group `u_i` of variables `ξ_q`, one per scalar output. Its index is the closed index of the group, `S_{u_i} = Σ_{α ≠ 0, supp(α) ⊆ u_i} a_α² / Σ_{α ≠ 0} a_α²`. This is `Var[E[QoI | ξ_q, q ∈ u_i]] / Var[QoI]`, and it reduces to (C-27) for a scalar coupling.

Enriching one discipline at `z` follows section 8: solve the mean MDA at `z`, evaluate only the selected `f_i` at `(z, y_c(i))`, and retrain only its GPs. In ask/tell mode, the request then contains a single disciplinary evaluation.

The enrichment strategy is an option: `All` (EGMDO) or `Sobol` (C-EGMDO). Both apply to unconstrained and constrained problems.

## 21. Main loop and results

The main loop is section 9, with:

- CEI or WB2 under the constraint models in place of the EI (section 18);
- C-EGMDO Algorithm 2 (section 19) in place of Algorithm 1;
- GPs of the constraint PCE means rebuilt whenever `DoE_UQ` or the PCEs change.

Stopping: iteration budget `n_max`, as in EGMDO (C-EGMDO section 3.5). The maximum value of the criterion is recorded at each iteration.

**Decision D29 (results).** C-EGMDO's Algorithm 1 returns the best feasible `DoE_UQ` point. The paper's tables also report the mean and coefficient of variation of `z*`, as EGMDO does. Both are reported:

- **best point:** the `DoE_UQ` point with the lowest objective PCE mean among the points whose constraint PCE means are feasible. If none is, it is the point with the lowest mean violation;
- **distributions:** `Ẑ*` and `Ŷ_min` as in section 10, each sample minimizing (19) under the constraint models `μ_g ≤ τ`, `|μ_h| ≤ τ`.

The paper considers a run successful when the relative error on the objective is below 1% (Sellar), or below 5% on the design variables (aeroelastic wing). Benchmarks report the criterion they use.

## 22. Test problems and decisions (part II)

### 22.1 Constrained 1-D toy problem (C-EGMDO section 2.1)

The toy problem of section 11.1 with

```text
g1(z, y1) = 0.15 y1 − 0.9 ≤ 0
```

The paper writes `−0.15 y1 + 0.9 ≥ 0`; see D20.

- Reference: the paper gives `z* ≈ −2.42`. The constraint is active at the optimum (`y1 = 6`): `z* = −2.40296525`, `f_obj = −0.46096568`. The unconstrained minimum `z ≈ −3.00`, `f ≈ −1.150` is infeasible.
- Paper setting (Figs. 4–10):
  - `DoE_UQ = {−3.6, −2, 1.2, 3.8}`, `C(1) = C(2) = [0, 25]`, `n_MC = 200`;
  - `ε_P = 0.1`, `ε_CV = 0.001`, `n_max = 5`.
- Paper results: disciplinary enrichments happen only at `z = −2`, `−2.5` and `−2.42`.

### 22.2 Classic constrained Sellar problem (C-EGMDO section 4.1)

```text
f_obj(z, y1, y2) = z3² + z2 + y1 + exp(−y2)
g1 = 3.16 − y1 ≤ 0,   g2 = y2 − 24 ≤ 0
y1 = f1(z, y2) = z1² + z2 + z3 − 0.2 y2
y2 = f2(z, y1) = sqrt(y1) + z1 + z2
Z = [−10, 10] × [0, 10] × [0, 10],  C(1) = [0, 25] (for y2),  C(2) = [0, 25] (for y1)
```

**Decision D30.** The paper prints `z3³` in the objective; the classic Sellar problem has the square of the local variable. The paper's Table 1 confirms the square. With `z3²`, MDF + SLSQP gives:

- the global optimum `(1.9776, 0, 0)`, `f = 3.1834`;
- the local optimum `(−1.7172, 0.1384, 0.1128)`, `f = 4.1308`.

Both match Table 1. With `z3³`, the local optimum would be `f = 4.1019`.

- Reference: `z* ≈ (1.9776, 0, 0)`, `f* ≈ 3.1833`, `g1` active.
- Paper setting: `n_f1 = n_f2 = 5`, `n_UQ0 = 20`, `n_MC = 200`, `d = 3`, `ε_CV = 0.01`, `ε_P = 10⁻³`, `n_max = 10`, Sobol enrichment.
- Paper results over 100 runs, a run being successful when the relative error on `f` is below 1%:
  - CEI: 63 successful runs, `n1 = 5 + 7.1`, `n2 = 5 + 4.5` evaluations;
  - WB2: 96 successful runs, `n1 = 5 + 8.0`, `n2 = 5 + 4.8` evaluations, i.e. 22.8 disciplinary evaluations in total on average.
- Paper baselines (Table 6, successful runs / total disciplinary evaluations):
  - MDF: SLSQP 43 / 136.2, COBYLA 55 / 334.2, SEGOMOE 94 / 259.5;
  - IDF: SLSQP 59 / 103.1, COBYLA 90 / 210.9, SEGOMOE 95 / 56.6.

`sqrt(y1)` is defined at the initial DoE (`C(2) = [0, 25]`); D18 applies to enrichment points.

### 22.3 Aeroelastic wing (C-EGMDO section 4.2)

This problem couples a Code Aster beam model with an in-house VLM solver, through 5 + 5 parameterized coupling variables. It cannot be reproduced here and is not a benchmark. It is the only published case with 10 scalar couplings (`P = 286` for `d = 3`, `n_MC = 500` in the paper), which documents the limitation of section 15.

### 22.4 Decisions summary (part II)

| Id | Topic | Decision |
|---|---|---|
| D20 | Sign convention | `g ≤ 0` feasible, as in `Egor`; paper equations rewritten |
| D21 | Tolerance | Per-constraint tolerance `τ`, default `1e-4` |
| D22 | Constraints of `z` only | Evaluated directly, no PCE or GP |
| D23 | WB2 scale | Optional scale on CEI, default 1 |
| D24 | Acquisition optimizer | Multistart COBYLA with the constraint models (`egobox_gp::basin_optimizer`); a local optimum is feasible up to `1e-6`, COBYLA's final radius |
| D25 | Duplicate `DoE_UQ` point | Next best multistart point; threshold `1e-6` in `[0, 1]^n` |
| D26 | `P_sol` | Samples without a feasible point count for none; least-violation fallback when all `P_sol < ε_P` |
| D27 | Constraint CV | Denominator `max(\|E\|, τ / ε_CV)`: an active constraint is certain when its std is below `τ` |
| D28 | Sobol index of a discipline | Closed index of its group of `ξ` |
| D29 | Results | Best feasible `DoE_UQ` point and constrained distributions of `Ẑ*`, `Ŷ_min` |
| D30 | Sellar objective | `z3²` (the paper's `z3³` is a typo) |

## 23. Options and defaults

The library defaults follow C-EGMDO, the more recent and more robust variant, with Matérn 5/2 GPs (D34). `EgmdoConfig::egmdo()` gives the settings of the 2020 paper.

| Option | Values | Default | EGMDO (2020) | C-EGMDO (2024) |
|---|---|---|---|---|
| Acquisition | `Ei`, `Wb2 { scale }`, `Wb2s` (D35), `LogEi` (D36) | `Wb2 { scale: 1 }` | `Ei` | `Wb2` |
| Enrichment | `All`, `Sobol` | `Sobol` | `All` | `Sobol` |
| `ε_P` | > 0 | `10⁻³` | `1/n_UQ` | `10⁻³` |
| `ε_CV` | > 0 | 0.01 | 0.01 | 0.01 |
| GP kernel | `Matern52`, `SquaredExponential` | `Matern52` (D34) | `SquaredExponential` | `SquaredExponential` |
| Random MDA | `Solved`, `Linearized` (D37) | `Solved` | `Solved` | `Solved` |
| Correlation across design points | `Perfect`, `GpPosterior` (D38) | `Perfect` | `Perfect` | `Perfect` |
| `n_MC` | `≥ P + 1` | `max(200, 2P)` | 100 | 200 to 500 |
| Constraint tolerance `τ` | ≥ 0 per constraint | `1e-4` | — | — |

All other defaults are those of section 13.

Mapping to the implementation, in addition to section 14:

| Section | Module | Tests |
|---|---|---|
| 16, 17 | `problem`, `pce`, `kl` | constraint PCEs share the MDA samples; GP of constraint means interpolates `DoE_UQ` values; deterministic constraints bypass PCE |
| 18 | `criteria` | `f'_min` with and without feasible points; CEI equals EI without constraints; WB2 = CEI − `m_0`; duplicate rejection |
| 19 | `solver` | `P_sol` sums to the fraction of samples with a feasible point; fallback to least violation; constraint CV floor |
| 20 | `pce`, `solver` | Sobol indices of known polynomials (e.g. `ξ_1 + 2ξ_2` gives 0.2 and 0.8); group index; only the selected discipline is requested in ask/tell |
| 22 | `problems` | constrained toy and Sellar end to end, references of 22.1 and 22.2 |

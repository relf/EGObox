use crate::ThetaTuning;
use crate::correlation_models::CorrelationModel;
use crate::errors::{GpError, Result};
use crate::linalg::{UPLO, cholesky, solve_triangular};
use crate::optimization::{
    CobylaParams, broadcast_bounds, broadcast_theta0, multistart_minimize, prepare_multistart,
};
use crate::pairwise_parameters::{
    PairwiseGpParams, PairwiseLikelihood, PairwiseValidParams, PreferenceComparison,
    PreferenceOutcome,
};
use crate::utils::{DiffMatrix, NormalizedData, check_training_size, differences};
use linfa::ParamGuard;
use linfa::prelude::{Float, PredictInplace};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::fmt;

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const SQRT_2PI: f64 = 2.506_628_274_631_000_2;

/// Standard normal probability density function `phi(z)`.
#[inline]
fn std_norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / SQRT_2PI
}

/// Standard normal cumulative distribution function `Phi(z)` (Abramowitz & Stegun 7.1.26).
#[inline]
fn std_norm_cdf(z: f64) -> f64 {
    if z < -8.5 {
        return 0.0;
    }
    if z > 8.5 {
        return 1.0;
    }
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let erf = 1.0
        - ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    0.5 * (1.0 + sign * erf)
}

/// Numerically stable `(log Phi(z), phi(z) / Phi(z))` across all real `z`
/// using Laplace's continued fraction in the far left tail (`z < -5`).
#[inline]
fn log_ndtr_and_hazard(z: f64) -> (f64, f64) {
    if z >= -5.0 {
        let cdf = std_norm_cdf(z).max(1e-300);
        let pdf = std_norm_pdf(z);
        (cdf.ln(), pdf / cdf)
    } else {
        let u = -z;
        // Continued fraction for phi(u) / Phi(-u)
        let mut cf = u;
        for k in (1..=12).rev() {
            cf = u + (k as f64) / cf;
        }
        let hazard = cf;
        let log_pdf = -0.5 * z * z - SQRT_2PI.ln();
        let log_cdf = log_pdf - hazard.ln();
        (log_cdf, hazard)
    }
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline]
fn log_sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
enum DualRowKind {
    Inequality { offset: f64 },
    GaussianTie { variance: f64 },
}

/// Sparse linear functional `d_r = sum_k terms[k].1 * u[terms[k].0]` over latent utilities.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
struct DualObservationRow {
    terms: Vec<(usize, f64)>,
    kind: DualRowKind,
}

impl DualObservationRow {
    fn eval_loglik_grad_hess(&self, d_val: f64, likelihood: PairwiseLikelihood) -> (f64, f64, f64) {
        match self.kind {
            DualRowKind::Inequality { offset } => match likelihood {
                PairwiseLikelihood::Probit => {
                    let z = (d_val + offset) / SQRT_2;
                    let (log_p, hazard) = log_ndtr_and_hazard(z);
                    let grad = hazard / SQRT_2;
                    let neg_hess = (0.5 * hazard * (hazard + z)).max(1e-10);
                    (log_p, grad, neg_hess)
                }
                PairwiseLikelihood::Logit => {
                    let z = d_val + offset;
                    let p = sigmoid_f64(z);
                    let log_p = log_sigmoid_f64(z);
                    let grad = 1.0 - p;
                    let neg_hess = (p * (1.0 - p)).max(1e-10);
                    (log_p, grad, neg_hess)
                }
            },
            DualRowKind::GaussianTie { variance } => {
                let log_p = -0.5 * (d_val * d_val) / variance
                    - 0.5 * (2.0 * std::f64::consts::PI * variance).ln();
                let grad = -d_val / variance;
                let neg_hess = (1.0 / variance).max(1e-10);
                (log_p, grad, neg_hess)
            }
        }
    }
}

struct DualLaplaceSolution {
    alpha: Array1<f64>,
    sqrt_w: Array1<f64>,
    lb_chol: Array2<f64>,
    log_marginal_likelihood: f64,
}

/// Preferential Gaussian Process model learning a latent utility function `u(x)` from
/// pairwise comparisons (`a > b`, ties `a ~= b`, and joint rejections) via a dual-space
/// Newton-Raphson Laplace approximation (Chu & Ghahramani, 2005; Rasmussen & Williams, Alg. 3.1/3.2).
///
/// Because inference operates in dual observation space `(m x m)` rather than `(n x n)` item
/// space, fitting and evaluating analytic Expected Utility of the Best Option ([`Self::eubo`])
/// scales cleanly both to continuous design spaces and to large discrete candidate pools.
///
/// # Example
///
/// ```
/// use egobox_gp::{PreferentialKriging, PreferenceComparison};
/// use ndarray::{array, Array};
///
/// // Hidden utility maximized near x = 0.7
/// let xtrain = array![[0.0], [0.35], [0.7], [1.0]];
/// let comparisons = vec![
///     PreferenceComparison::prefer(2, 0), // u(0.7) > u(0.0)
///     PreferenceComparison::prefer(2, 1), // u(0.7) > u(0.35)
///     PreferenceComparison::prefer(2, 3), // u(0.7) > u(1.0)
/// ];
///
/// let model = PreferentialKriging::<f64>::params()
///     .fit(&xtrain, &comparisons)
///     .expect("preferential GP fitted");
///
/// let utilities = model.predict(&xtrain).unwrap();
/// assert!(utilities[2] > utilities[0] && utilities[2] > utilities[3]);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "F: Serialize, Corr: Serialize",
        deserialize = "F: Deserialize<'de>, Corr: Deserialize<'de>"
    ))
)]
pub struct PairwiseGaussianProcess<F: Float, Corr: CorrelationModel<F>> {
    corr: Corr,
    theta: Array1<F>,
    log_marginal_likelihood: F,
    xt_norm: NormalizedData<F>,
    obs_rows: Vec<DualObservationRow>,
    alpha: Array1<F>,
    sqrt_w: Array1<F>,
    lb_chol: Array2<F>,
    params: PairwiseValidParams<F, Corr>,
}

impl<F: Float, Corr: CorrelationModel<F>> fmt::Display for PairwiseGaussianProcess<F, Corr> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PairwiseGP(corr={}, likelihood={:?}, theta={}, log_evidence={})",
            self.corr, self.params.likelihood, self.theta, self.log_marginal_likelihood
        )
    }
}

impl<F: Float, Corr: CorrelationModel<F>> PairwiseGpParams<F, Corr> {
    /// Fit a [`PairwiseGaussianProcess`] given `(n, d)` training points and pairwise comparisons.
    pub fn fit(
        &self,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        comparisons: &[PreferenceComparison],
    ) -> Result<PairwiseGaussianProcess<F, Corr>> {
        self.check_ref()?.fit(xtrain, comparisons)
    }
}

impl<F: Float, Corr: CorrelationModel<F>> PairwiseValidParams<F, Corr> {
    /// Fit a [`PairwiseGaussianProcess`] from validated parameters.
    pub fn fit(
        &self,
        xtrain: &ArrayBase<impl Data<Elem = F>, Ix2>,
        comparisons: &[PreferenceComparison],
    ) -> Result<PairwiseGaussianProcess<F, Corr>> {
        check_training_size(xtrain)?;
        let n = xtrain.nrows();
        let nx = xtrain.ncols();

        let obs_rows = expand_comparisons(
            comparisons,
            n,
            self.tie_variance.to_f64().unwrap_or(0.15),
            self.reject_threshold.to_f64().unwrap_or(-0.75),
        )?;

        let xt_norm = NormalizedData::new(xtrain);
        let xtrain_f64 = xt_norm.data.mapv(|v| v.to_f64().unwrap_or(0.0));
        let diff_mat = DiffMatrix::new(&xtrain_f64);
        let eye_weights = Array2::<f64>::eye(nx);
        let nugget_f64 = self.nugget.to_f64().unwrap_or(1e-5);

        let theta0 = broadcast_theta0(self.theta_tuning.init(), nx);
        let theta0_f64 = theta0.mapv(|v| v.to_f64().unwrap_or(0.5));

        // Reinterpret Corr on f64 coordinates by evaluating kernel via F conversions
        let eval_for_theta =
            |theta_f64: &Array1<f64>| -> Result<(Array2<f64>, DualLaplaceSolution)> {
                let k_mat = compute_kernel_matrix_f64(
                    &self.corr,
                    &diff_mat,
                    theta_f64,
                    &eye_weights,
                    nugget_f64,
                );
                let sol = solve_dual_laplace(&k_mat, &obs_rows, self.likelihood)?;
                Ok((k_mat, sol))
            };

        let best_theta_f64 = match &self.theta_tuning {
            ThetaTuning::Fixed(_) => theta0_f64,
            ThetaTuning::Full { init: _, bounds } => {
                if obs_rows.is_empty() {
                    theta0_f64
                } else {
                    let b_vec = broadcast_bounds(bounds, nx);
                    let b_f64: Vec<(f64, f64)> = b_vec
                        .iter()
                        .map(|(lo, up)| (lo.to_f64().unwrap_or(1e-2), up.to_f64().unwrap_or(10.0)))
                        .collect();
                    let (inits, log_bounds) = prepare_multistart(self.n_start, &theta0_f64, &b_f64);
                    let cobyla = CobylaParams {
                        maxeval: self.max_eval,
                        ..CobylaParams::default()
                    };
                    let (best_neg_ev, opt_log_theta) = multistart_minimize(
                        |u_log10: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()| -> f64 {
                            let cand = Array1::from_iter(u_log10.iter().map(|&v| 10f64.powf(v)));
                            // Gentle log-normal prior regularizer keeps theta well-conditioned on small m
                            let reg: f64 = u_log10.iter().map(|&v| 0.02 * v * v).sum();
                            match eval_for_theta(&cand) {
                                Ok((_, sol)) => -sol.log_marginal_likelihood + reg,
                                Err(_) => f64::INFINITY,
                            }
                        },
                        &inits,
                        &log_bounds,
                        cobyla,
                        &self.cancellation,
                    );
                    if self.cancellation.is_cancelled() {
                        return Err(GpError::Interrupted);
                    }
                    if best_neg_ev.is_finite() {
                        opt_log_theta.mapv(|v| 10f64.powf(v))
                    } else {
                        theta0_f64
                    }
                }
            }
            ThetaTuning::Partial {
                init: _,
                bounds,
                active,
            } => {
                if obs_rows.is_empty() || active.is_empty() {
                    theta0_f64
                } else {
                    let full_bounds = broadcast_bounds(bounds, nx);
                    let active_init = Array1::from_iter(active.iter().map(|&i| theta0_f64[i]));
                    let active_bounds: Vec<(f64, f64)> = active
                        .iter()
                        .map(|&i| {
                            (
                                full_bounds[i].0.to_f64().unwrap_or(1e-2),
                                full_bounds[i].1.to_f64().unwrap_or(10.0),
                            )
                        })
                        .collect();
                    let (inits, log_bounds) =
                        prepare_multistart(self.n_start, &active_init, &active_bounds);
                    let cobyla = CobylaParams {
                        maxeval: self.max_eval,
                        ..CobylaParams::default()
                    };
                    let (best_neg_ev, opt_log_active) = multistart_minimize(
                        |u_log10: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()| -> f64 {
                            let mut cand = theta0_f64.clone();
                            for (pos, &dim_idx) in active.iter().enumerate() {
                                cand[dim_idx] = 10f64.powf(u_log10[pos]);
                            }
                            let reg: f64 = u_log10.iter().map(|&v| 0.02 * v * v).sum();
                            match eval_for_theta(&cand) {
                                Ok((_, sol)) => -sol.log_marginal_likelihood + reg,
                                Err(_) => f64::INFINITY,
                            }
                        },
                        &inits,
                        &log_bounds,
                        cobyla,
                        &self.cancellation,
                    );
                    if self.cancellation.is_cancelled() {
                        return Err(GpError::Interrupted);
                    }
                    if best_neg_ev.is_finite() {
                        let mut merged = theta0_f64.clone();
                        for (pos, &dim_idx) in active.iter().enumerate() {
                            merged[dim_idx] = 10f64.powf(opt_log_active[pos]);
                        }
                        merged
                    } else {
                        theta0_f64
                    }
                }
            }
        };

        let (_, final_sol) = eval_for_theta(&best_theta_f64)?;

        Ok(PairwiseGaussianProcess {
            corr: self.corr,
            theta: best_theta_f64.mapv(F::cast),
            log_marginal_likelihood: F::cast(final_sol.log_marginal_likelihood),
            xt_norm,
            obs_rows,
            alpha: final_sol.alpha.mapv(F::cast),
            sqrt_w: final_sol.sqrt_w.mapv(F::cast),
            lb_chol: final_sol.lb_chol.mapv(F::cast),
            params: self.clone(),
        })
    }
}

impl<F: Float, Corr: CorrelationModel<F>> PairwiseGaussianProcess<F, Corr> {
    /// Construct a hyperparameter builder [`PairwiseGpParams`] for the given correlation kernel.
    pub fn params(corr: Corr) -> PairwiseGpParams<F, Corr> {
        PairwiseGpParams::new(corr)
    }

    /// Fitted autocorrelation parameter `theta`.
    pub fn theta(&self) -> &Array1<F> {
        &self.theta
    }

    /// Laplace-approximated log marginal likelihood (log model evidence).
    pub fn log_marginal_likelihood(&self) -> F {
        self.log_marginal_likelihood
    }

    /// Input dimensionality of the model.
    pub fn dims(&self) -> usize {
        self.xt_norm.ncols()
    }

    /// Predict posterior mean latent utility `u(x)` at query points `x` of shape `(q, d)`.
    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array1<F>> {
        self.validate_query_dims(x)?;
        let q = x.nrows();
        if self.obs_rows.is_empty() {
            return Ok(Array1::zeros(q));
        }

        let k_star_x = self.cross_kernel_to_train(x); // (q, n)
        let mk_star = self.apply_incidence_transpose(&k_star_x); // (q, m_eff)
        let out = mk_star.dot(&self.alpha);
        Ok(out)
    }

    /// Predict marginal posterior utility variance `Var[u(x)]` at query points `x` of shape `(q, d)`.
    pub fn predict_var(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array1<F>> {
        self.validate_query_dims(x)?;
        let q = x.nrows();
        let diag_prior = F::one() + self.params.nugget;
        if self.obs_rows.is_empty() {
            return Ok(Array1::from_elem(q, diag_prior));
        }

        let k_star_x = self.cross_kernel_to_train(x); // (q, n)
        let v_mat = self.compute_whitened_projection(&k_star_x)?; // (m_eff, q)

        let eps = F::cast(1e-9);
        let mut vars = Array1::zeros(q);
        for i in 0..q {
            let col = v_mat.column(i);
            let reduction = col.dot(&col);
            vars[i] = (diag_prior - reduction).max(eps);
        }
        Ok(vars)
    }

    /// Predict full joint posterior covariance matrix `Cov[u(X), u(X)]` of shape `(q, q)`.
    pub fn predict_cov(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array2<F>> {
        self.validate_query_dims(x)?;
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;
        let q = xnorm.nrows();
        let nx = xnorm.ncols();
        let eye_w = Array2::<F>::eye(nx);

        let mut k_qq = Array2::<F>::zeros((q, q));
        for i in 0..q {
            let r_col = self.corr.rval(&xnorm.row(i), &xnorm, &self.theta, &eye_w);
            for j in 0..q {
                k_qq[[i, j]] = r_col[[j, 0]];
            }
            k_qq[[i, i]] += self.params.nugget;
        }

        if self.obs_rows.is_empty() {
            return Ok(k_qq);
        }

        let k_star_x = self.cross_kernel_to_train(x); // (q, n)
        let v_mat = self.compute_whitened_projection(&k_star_x)?; // (m_eff, q)
        let reduction = v_mat.t().dot(&v_mat);
        let mut cov = k_qq - reduction;
        let eps = F::cast(1e-9);
        for i in 0..q {
            if cov[[i, i]] < eps {
                cov[[i, i]] = eps;
            }
        }
        Ok(cov)
    }

    /// Evaluate the analytic **Expected Utility of the Best Option (EUBO)**
    /// `E[max(u(x_a), u(x_b))]` for a candidate duel `(x_a, x_b)` (Lin et al., 2022; Astudillo et al., 2023).
    pub fn eubo(
        &self,
        x_a: &ArrayBase<impl Data<Elem = F>, Ix1>,
        x_b: &ArrayBase<impl Data<Elem = F>, Ix1>,
    ) -> Result<F> {
        if x_a.len() != self.dims() || x_b.len() != self.dims() {
            return Err(GpError::InvalidValueError(format!(
                "Candidate dimension mismatch: expected {}, got ({}, {})",
                self.dims(),
                x_a.len(),
                x_b.len()
            )));
        }
        let mut pair = Array2::<F>::zeros((2, self.dims()));
        pair.row_mut(0).assign(x_a);
        pair.row_mut(1).assign(x_b);

        let mean = self.predict(&pair)?;
        let cov = self.predict_cov(&pair)?;
        Ok(F::cast(analytic_eubo_scalar(
            mean[0].to_f64().unwrap_or(0.0),
            mean[1].to_f64().unwrap_or(0.0),
            cov[[0, 0]].to_f64().unwrap_or(1.0),
            cov[[1, 1]].to_f64().unwrap_or(1.0),
            cov[[0, 1]].to_f64().unwrap_or(0.0),
        )))
    }

    /// Select the highest-EUBO candidate pair `(idx_a, idx_b, eubo_value)` from a discrete
    /// `(n_cand, d)` candidate pool while optionally skipping previously compared index pairs.
    pub fn select_next_pair_eubo(
        &self,
        candidates: &ArrayBase<impl Data<Elem = F>, Ix2>,
        excluded_pairs: &[(usize, usize)],
    ) -> Result<(usize, usize, F)> {
        if candidates.nrows() < 2 {
            return Err(GpError::InvalidValueError(
                "At least 2 candidates are required to select a comparison pair".to_string(),
            ));
        }
        let mean = self.predict(candidates)?;
        let cov = self.predict_cov(candidates)?;
        let n_cand = candidates.nrows();

        let is_excluded = |i: usize, j: usize| -> bool {
            excluded_pairs
                .iter()
                .any(|&(a, b)| (a == i && b == j) || (a == j && b == i))
        };

        let mut best_pair = (0usize, 1usize);
        let mut best_val = f64::NEG_INFINITY;

        for i in 0..n_cand {
            let mu_i = mean[i].to_f64().unwrap_or(0.0);
            let var_i = cov[[i, i]].to_f64().unwrap_or(1.0);
            for j in (i + 1)..n_cand {
                if is_excluded(i, j) {
                    continue;
                }
                let mu_j = mean[j].to_f64().unwrap_or(0.0);
                let var_j = cov[[j, j]].to_f64().unwrap_or(1.0);
                let cov_ij = cov[[i, j]].to_f64().unwrap_or(0.0);
                let val = analytic_eubo_scalar(mu_i, mu_j, var_i, var_j, cov_ij);
                if val > best_val {
                    best_val = val;
                    best_pair = if mu_i >= mu_j { (i, j) } else { (j, i) };
                }
            }
        }

        if !best_val.is_finite() {
            return Err(GpError::InvalidValueError(
                "All candidate pairs were excluded".to_string(),
            ));
        }
        Ok((best_pair.0, best_pair.1, F::cast(best_val)))
    }

    fn validate_query_dims(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<()> {
        if x.ncols() != self.dims() {
            return Err(GpError::InvalidValueError(format!(
                "Query input dimension mismatch: model expects {}, got {}",
                self.dims(),
                x.ncols()
            )));
        }
        Ok(())
    }

    fn cross_kernel_to_train(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;
        let q = xnorm.nrows();
        let n = self.xt_norm.data.nrows();
        let eye_w = Array2::<F>::eye(self.dims());
        let mut k_star = Array2::<F>::zeros((q, n));
        for i in 0..q {
            let d = differences(&xnorm.row(i), &self.xt_norm.data);
            let r_col = self.corr.rval_from_distances(&d, &self.theta, &eye_w);
            for j in 0..n {
                k_star[[i, j]] = r_col[[j, 0]];
            }
        }
        k_star
    }

    /// Computes `K_{*X} * M^T` of shape `(q, m_eff)`.
    fn apply_incidence_transpose(&self, k_star_x: &Array2<F>) -> Array2<F> {
        let q = k_star_x.nrows();
        let m_eff = self.obs_rows.len();
        let mut out = Array2::<F>::zeros((q, m_eff));
        for (r, row) in self.obs_rows.iter().enumerate() {
            for &(idx, coeff) in &row.terms {
                let c = F::cast(coeff);
                for qi in 0..q {
                    out[[qi, r]] += k_star_x[[qi, idx]] * c;
                }
            }
        }
        out
    }

    /// Solves `L_B * V = W^{1/2} * M * K_{X*}` returning `V` of shape `(m_eff, q)`.
    fn compute_whitened_projection(&self, k_star_x: &Array2<F>) -> Result<Array2<F>> {
        let mk_star = self.apply_incidence_transpose(k_star_x); // (q, m_eff)
        let m_eff = self.obs_rows.len();
        let q = k_star_x.nrows();
        let mut rhs = Array2::<F>::zeros((m_eff, q));
        for r in 0..m_eff {
            let sw = self.sqrt_w[r];
            for qi in 0..q {
                rhs[[r, qi]] = sw * mk_star[[qi, r]];
            }
        }
        solve_triangular(&self.lb_chol, rhs, UPLO::Lower)
    }
}

impl<F: Float, Corr: CorrelationModel<F>, D: Data<Elem = F>>
    PredictInplace<ArrayBase<D, Ix2>, Array1<F>> for PairwiseGaussianProcess<F, Corr>
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<F>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );
        let vals = self.predict(x).expect("PairwiseGP prediction failed");
        y.assign(&vals);
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        Array1::zeros(x.nrows())
    }
}

#[inline]
fn analytic_eubo_scalar(mu_a: f64, mu_b: f64, var_a: f64, var_b: f64, cov_ab: f64) -> f64 {
    let delta = mu_a - mu_b;
    let diff_var = (var_a + var_b - 2.0 * cov_ab).max(1e-12);
    let sigma = diff_var.sqrt();
    let z = delta / sigma;
    mu_b + delta * std_norm_cdf(z) + sigma * std_norm_pdf(z)
}

fn expand_comparisons(
    comparisons: &[PreferenceComparison],
    n_points: usize,
    tie_variance: f64,
    reject_threshold: f64,
) -> Result<Vec<DualObservationRow>> {
    let mut rows = Vec::with_capacity(comparisons.len());
    for comp in comparisons {
        if comp.first >= n_points || comp.second >= n_points {
            return Err(GpError::InvalidValueError(format!(
                "Comparison index ({}, {}) out of bounds for {} training samples",
                comp.first, comp.second, n_points
            )));
        }
        match comp.outcome {
            PreferenceOutcome::Strict => {
                if comp.first == comp.second {
                    return Err(GpError::InvalidValueError(format!(
                        "Cannot strictly prefer item {} over itself",
                        comp.first
                    )));
                }
                rows.push(DualObservationRow {
                    terms: vec![(comp.first, 1.0), (comp.second, -1.0)],
                    kind: DualRowKind::Inequality { offset: 0.0 },
                });
            }
            PreferenceOutcome::Tie => {
                if comp.first != comp.second {
                    rows.push(DualObservationRow {
                        terms: vec![(comp.first, 1.0), (comp.second, -1.0)],
                        kind: DualRowKind::GaussianTie {
                            variance: tie_variance,
                        },
                    });
                }
            }
            PreferenceOutcome::RejectBoth => {
                // Penalize u[first] > reject_threshold and u[second] > reject_threshold
                rows.push(DualObservationRow {
                    terms: vec![(comp.first, -1.0)],
                    kind: DualRowKind::Inequality {
                        offset: reject_threshold,
                    },
                });
                if comp.second != comp.first {
                    rows.push(DualObservationRow {
                        terms: vec![(comp.second, -1.0)],
                        kind: DualRowKind::Inequality {
                            offset: reject_threshold,
                        },
                    });
                }
            }
        }
    }
    Ok(rows)
}

fn compute_kernel_matrix_f64<F: Float, Corr: CorrelationModel<F>>(
    corr: &Corr,
    diff_mat: &DiffMatrix<f64>,
    theta_f64: &Array1<f64>,
    eye_weights: &Array2<f64>,
    nugget: f64,
) -> Array2<f64> {
    let n = diff_mat.n_obs;
    let d_f = diff_mat.d.mapv(F::cast);
    let theta_f = theta_f64.mapv(F::cast);
    let w_f = eye_weights.mapv(F::cast);
    let r_vals = corr.rval_from_distances(&d_f, &theta_f, &w_f);

    let mut k_mat = Array2::<f64>::eye(n) * (1.0 + nugget);
    for (row_idx, pair) in diff_mat.d_indices.rows().into_iter().enumerate() {
        let i = pair[0];
        let j = pair[1];
        let val = r_vals[[row_idx, 0]].to_f64().unwrap_or(0.0);
        k_mat[[i, j]] = val;
        k_mat[[j, i]] = val;
    }
    k_mat
}

/// Exact Dual-Space Newton-Raphson Laplace Approximation (Rasmussen & Williams, Alg. 3.1):
/// Solves for `alpha` in R^{m_eff} where `d = K_d * alpha` and `K_d = M * K * M^T`.
fn solve_dual_laplace(
    k_mat: &Array2<f64>,
    obs_rows: &[DualObservationRow],
    likelihood: PairwiseLikelihood,
) -> Result<DualLaplaceSolution> {
    let m = obs_rows.len();
    if m == 0 {
        return Ok(DualLaplaceSolution {
            alpha: Array1::zeros(0),
            sqrt_w: Array1::zeros(0),
            lb_chol: Array2::zeros((0, 0)),
            log_marginal_likelihood: 0.0,
        });
    }

    let n = k_mat.nrows();
    // Form M * K (m x n) and K_d = M * K * M^T (m x m)
    let mut mk = Array2::<f64>::zeros((m, n));
    for (r, row) in obs_rows.iter().enumerate() {
        for &(idx, coeff) in &row.terms {
            for j in 0..n {
                mk[[r, j]] += coeff * k_mat[[idx, j]];
            }
        }
    }

    let mut kd = Array2::<f64>::zeros((m, m));
    for r in 0..m {
        for c in r..m {
            let mut sum = 0.0;
            for &(idx, coeff) in &obs_rows[c].terms {
                sum += mk[[r, idx]] * coeff;
            }
            if r == c {
                sum += 1e-8;
            }
            kd[[r, c]] = sum;
            kd[[c, r]] = sum;
        }
    }

    let mut alpha = Array1::<f64>::zeros(m);
    let mut d_vec = Array1::<f64>::zeros(m);
    let mut grad = Array1::<f64>::zeros(m);
    let mut w_diag = Array1::<f64>::zeros(m);

    let eval_psi = |a_cand: &Array1<f64>,
                    d_cand: &Array1<f64>,
                    g_out: &mut Array1<f64>,
                    w_out: &mut Array1<f64>|
     -> f64 {
        let mut ll = 0.0;
        for r in 0..m {
            let (lp, g, w) = obs_rows[r].eval_loglik_grad_hess(d_cand[r], likelihood);
            ll += lp;
            g_out[r] = g;
            w_out[r] = w;
        }
        ll - 0.5 * a_cand.dot(d_cand)
    };

    let mut psi = eval_psi(&alpha, &d_vec, &mut grad, &mut w_diag);

    for _step in 0..30 {
        let max_res = (&grad - &alpha)
            .iter()
            .fold(0.0f64, |acc, &v| acc.max(v.abs()));
        if max_res < 1e-10 {
            break;
        }

        let sqrt_w = w_diag.mapv(f64::sqrt);
        let mut b_mat = Array2::<f64>::eye(m);
        for r in 0..m {
            for c in 0..m {
                b_mat[[r, c]] += sqrt_w[r] * kd[[r, c]] * sqrt_w[c];
            }
        }
        let lb = cholesky(b_mat)?;

        let b_rhs = &w_diag * &d_vec + &grad;
        let kd_b = kd.dot(&b_rhs);
        let scaled_rhs = (&sqrt_w * &kd_b).insert_axis(Axis(1));
        let y_mid = solve_triangular(&lb, scaled_rhs, UPLO::Lower)?;
        let z_sol = solve_triangular(&lb.t(), y_mid, UPLO::Upper)?.remove_axis(Axis(1));
        let alpha_target = &b_rhs - &(&sqrt_w * &z_sol);

        let mut step_size = 1.0;
        let mut accepted = false;
        let mut next_grad = Array1::<f64>::zeros(m);
        let mut next_w = Array1::<f64>::zeros(m);

        for _ls in 0..10 {
            let alpha_try = &alpha * (1.0 - step_size) + &alpha_target * step_size;
            let d_try = kd.dot(&alpha_try);
            let psi_try = eval_psi(&alpha_try, &d_try, &mut next_grad, &mut next_w);
            if psi_try >= psi - 1e-12 {
                alpha = alpha_try;
                d_vec = d_try;
                grad = next_grad;
                w_diag = next_w;
                psi = psi_try;
                accepted = true;
                break;
            }
            step_size *= 0.5;
        }
        if !accepted {
            break;
        }
    }

    let final_psi = eval_psi(&alpha, &d_vec, &mut grad, &mut w_diag);
    let sqrt_w = w_diag.mapv(f64::sqrt);
    let mut b_mat = Array2::<f64>::eye(m);
    for r in 0..m {
        for c in 0..m {
            b_mat[[r, c]] += sqrt_w[r] * kd[[r, c]] * sqrt_w[c];
        }
    }
    let lb_chol = cholesky(b_mat)?;
    let half_log_det_b: f64 = (0..m).map(|i| lb_chol[[i, i]].ln()).sum();
    let log_marginal_likelihood = final_psi - half_log_det_b;

    Ok(DualLaplaceSolution {
        alpha,
        sqrt_w,
        lb_chol,
        log_marginal_likelihood,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PreferentialKriging;
    use crate::correlation_models::Matern52Corr;
    use approx::assert_abs_diff_eq;
    use linfa::prelude::Predict;
    use ndarray::{Array, array};

    #[test]
    fn test_pairwise_gp_probit_and_logit_ranking_recovery() {
        let xtrain = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        // Latent peak at index 3 (x = 0.75)
        let comps = vec![
            PreferenceComparison::prefer(3, 0),
            PreferenceComparison::prefer(3, 1),
            PreferenceComparison::prefer(3, 2),
            PreferenceComparison::prefer(3, 4),
            PreferenceComparison::prefer(2, 0),
            PreferenceComparison::prefer(1, 0),
        ];

        for likelihood in [PairwiseLikelihood::Probit, PairwiseLikelihood::Logit] {
            let gp = PreferentialKriging::<f64>::params()
                .likelihood(likelihood)
                .n_start(3)
                .fit(&xtrain, &comps)
                .expect("PairwiseGP fit should succeed");

            let pred = gp.predict(&xtrain).unwrap();
            assert!(pred[3] > pred[2]);
            assert!(pred[2] > pred[0]);
            assert!(pred[3] > pred[4]);

            // Predict trait compatibility with linfa::prelude::Predict
            let linfa_pred: Array1<f64> = Predict::predict(&gp, &xtrain);
            assert_abs_diff_eq!(pred, linfa_pred, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pairwise_gp_ties_and_reject_both() {
        let xtrain = array![[0.0, 0.0], [0.5, 0.5], [0.52, 0.48], [1.0, 1.0]];
        let comps = vec![
            PreferenceComparison::prefer(1, 0),
            PreferenceComparison::tie(1, 2),
            PreferenceComparison::reject_both(0, 3),
        ];

        let gp = PairwiseGaussianProcess::<f64, Matern52Corr>::params(Matern52Corr::default())
            .theta_tuning(ThetaTuning::Fixed(array![0.6, 0.6]))
            .fit(&xtrain, &comps)
            .expect("fit with tie & reject_both should succeed");

        let u = gp.predict(&xtrain).unwrap();
        // Tied items 1 and 2 should have close posterior utilities and strictly beat rejected 0 & 3
        assert!((u[1] - u[2]).abs() < 0.25);
        assert!(u[1] > u[0] && u[1] > u[3]);

        let cov = gp.predict_cov(&xtrain).unwrap();
        for i in 0..4 {
            assert!(cov[[i, i]] > 0.0);
            for j in 0..4 {
                assert_abs_diff_eq!(cov[[i, j]], cov[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_preferential_bo_loop_with_analytic_eubo() {
        // Discrete 1D candidate pool with hidden quadratic optimum at x* = 0.68
        let candidates = Array::linspace(0.0, 1.0, 21).insert_axis(Axis(1));
        let true_utility = |x: f64| -(x - 0.68).powi(2);

        let mut compared: Vec<(usize, usize)> = Vec::new();
        let mut observations: Vec<PreferenceComparison> = Vec::new();

        // Start from cold prior (0 observations)
        let mut model = PreferentialKriging::<f64>::params()
            .theta_tuning(ThetaTuning::Fixed(array![0.4]))
            .fit(&candidates, &observations)
            .unwrap();

        for _iter in 0..7 {
            let (i, j, eubo_val) = model
                .select_next_pair_eubo(&candidates, &compared)
                .expect("EUBO pair selected");
            assert!(eubo_val.is_finite());
            compared.push((i, j));

            let ui = true_utility(candidates[[i, 0]]);
            let uj = true_utility(candidates[[j, 0]]);
            let obs = if ui >= uj {
                PreferenceComparison::prefer(i, j)
            } else {
                PreferenceComparison::prefer(j, i)
            };
            observations.push(obs);

            model = PreferentialKriging::<f64>::params()
                .n_start(2)
                .max_eval(80)
                .fit(&candidates, &observations)
                .unwrap();
        }

        let post_u = model.predict(&candidates).unwrap();
        let best_idx = post_u
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let found_x = candidates[[best_idx, 0]];
        assert!(
            (found_x - 0.68).abs() <= 0.11,
            "Expected optimum near 0.68 within 7 EUBO duels, got {found_x}"
        );
    }

    #[test]
    fn test_from_pairs_validation() {
        let pairs = array![[2usize, 0usize], [1usize, 0usize]];
        let parsed = PreferenceComparison::from_pairs(&pairs).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], PreferenceComparison::prefer(2, 0));

        let bad = Array2::<usize>::zeros((2, 3));
        assert!(PreferenceComparison::from_pairs(&bad).is_err());
    }
}

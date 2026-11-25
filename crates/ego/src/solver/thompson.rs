//! Exact discrete Thompson Sampling: joint posterior samples of the surrogates on a finite
//! candidate set, the minimizer of a sample being the candidate point.

use crate::utils::{cholesky_psd, compute_cstr_scales};

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_moe::{MixtureGpSurrogate, MoeError};
use ndarray::{Array1, Array2, ArrayView1, Axis, concatenate};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::{Normal, StandardNormal};
use rand_xoshiro::Xoshiro256Plus;

/// Maximum default number of global candidate points
const MAX_DEFAULT_GLOBAL_CANDIDATES: usize = 1000;
/// Default number of global candidate points per input dimension
const DEFAULT_GLOBAL_CANDIDATES_PER_DIM: usize = 100;
/// Ratio of local candidates (around the best point) to global candidates
const LOCAL_CANDIDATES_RATIO: f64 = 0.2;
/// Standard deviation of local candidates perturbations, relative to the domain ranges
const LOCAL_CANDIDATES_STD: f64 = 0.05;

/// Default number of global candidate points for `nx` input dimensions
pub(crate) fn default_n_candidates(nx: usize) -> usize {
    (DEFAULT_GLOBAL_CANDIDATES_PER_DIM * nx).min(MAX_DEFAULT_GLOBAL_CANDIDATES)
}

/// Candidate points: `n_global` LHS points of the domain `xlimits` and `n_global * 0.2`
/// gaussian perturbations of `xbest` (standard deviation 5% of the ranges, clipped to the domain)
/// for the set to be dense around the current best point
pub(crate) fn candidate_set(
    xlimits: &Array2<f64>,
    xbest: ArrayView1<f64>,
    n_global: usize,
    rng: &mut Xoshiro256Plus,
) -> Array2<f64> {
    let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
    let global = Lhs::new(xlimits)
        .kind(LhsKind::Classic)
        .with_rng(sub_rng)
        .sample(n_global.max(1));
    let n_local = ((n_global as f64) * LOCAL_CANDIDATES_RATIO).ceil() as usize;
    let mut local = Array2::zeros((n_local, xlimits.nrows()));
    for (j, mut col) in local.columns_mut().into_iter().enumerate() {
        let (lower, upper) = (xlimits[[j, 0]], xlimits[[j, 1]]);
        let std = LOCAL_CANDIDATES_STD * (upper - lower);
        if std > 0. {
            let normal = Normal::new(xbest[j], std).unwrap();
            col.assign(&Array1::random_using(n_local, normal, rng).mapv(|v| v.clamp(lower, upper)));
        } else {
            col.fill(xbest[j]);
        }
    }
    concatenate![Axis(0), global, local]
}

/// Joint posterior distribution of a surrogate on fixed points, ready to be sampled
pub(crate) struct JointSampler {
    mean: Array1<f64>,
    /// Lower Cholesky factor of the posterior covariance
    factor: Array2<f64>,
}

impl JointSampler {
    /// Posterior mean and covariance of `model` at `x` points
    pub(crate) fn new(model: &dyn MixtureGpSurrogate, x: &Array2<f64>) -> Result<Self, MoeError> {
        let mean = model.predict(&x.view())?;
        let cov = model.predict_covariance(&x.view())?;
        Ok(JointSampler {
            mean,
            factor: cholesky_psd(&cov),
        })
    }

    /// A joint posterior sample at the points, drawn from `rng`
    pub(crate) fn sample(&self, rng: &mut Xoshiro256Plus) -> Array1<f64> {
        let z = Array1::random_using(self.mean.len(), StandardNormal, rng);
        &self.mean + &self.factor.dot(&z)
    }
}

/// Constraint scales for the sampled violations: the largest magnitude of the mean prediction
/// of each constraint model at the candidate points `x` (see [`compute_cstr_scales`]),
/// 1 when it is null or not finite
pub(crate) fn violation_scales(
    cstr_models: &[Box<dyn MixtureGpSurrogate>],
    x: &Array2<f64>,
) -> Vec<f64> {
    compute_cstr_scales(&x.view(), cstr_models)
        .iter()
        .map(|&s| if s.is_finite() && s > 0. { s } else { 1. })
        .collect()
}

/// Index of the point selected from an objective sample `f` and constraint samples `g`
/// (`g_k <= tols[k]` is feasible), Scalable Constrained Bayesian Optimization (SCBO) rule:
/// the minimizer of `f` among the points feasible for all sampled constraints, or the
/// minimizer of the total sampled violation when no point is feasible, the violation of
/// each constraint being divided by its scale `scales[k]` (see [`violation_scales`]).
pub(crate) fn sampled_argmin(
    f: &Array1<f64>,
    g: &[Array1<f64>],
    tols: &[f64],
    scales: &[f64],
) -> usize {
    let violation = |i: usize| -> f64 {
        g.iter()
            .zip(tols)
            .zip(scales)
            .map(|((gk, tol), scale)| (gk[i] - tol).max(0.) / scale)
            .sum()
    };
    let argmin = |key: &dyn Fn(usize) -> f64, rows: &mut dyn Iterator<Item = usize>| {
        rows.fold(None, |best: Option<usize>, i| match best {
            Some(b) if key(b) <= key(i) => Some(b),
            _ => Some(i),
        })
    };
    let mut feasible = (0..f.len()).filter(|&i| violation(i) == 0.);
    argmin(&|i| f[i], &mut feasible)
        .or_else(|| argmin(&violation, &mut (0..f.len())))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_moe::GpMixture;
    use linfa::prelude::{Dataset, Fit};
    use ndarray::array;

    fn xsinx_model() -> GpMixture {
        let xt = array![[0.], [7.], [12.], [25.]];
        let yt = xt.mapv(|x: f64| (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).sin());
        GpMixture::params()
            .fit(&Dataset::new(xt, yt.column(0).to_owned()))
            .expect("GP fitted")
    }

    #[test]
    fn test_joint_samples_moments() {
        let model = xsinx_model();
        let x = array![[2.], [3.], [18.]];
        let sampler = JointSampler::new(&model, &x).unwrap();
        let cov = <GpMixture as egobox_moe::GpSurrogateExt>::predict_covariance(&model, &x.view())
            .unwrap();
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let n = 20000;
        let mut samples = Array2::zeros((n, 3));
        for mut row in samples.rows_mut() {
            row.assign(&sampler.sample(&mut rng));
        }
        let mean = samples.mean_axis(Axis(0)).unwrap();
        let centered = &samples - &mean;
        let emp_cov = centered.t().dot(&centered) / (n as f64 - 1.);
        let scale = cov.diag().iter().fold(0., |m: f64, v| m.max(*v));
        assert_abs_diff_eq!(mean, sampler.mean, epsilon = 0.05 * scale.sqrt());
        assert_abs_diff_eq!(emp_cov, cov, epsilon = 0.05 * scale);
    }

    #[test]
    fn test_samples_are_reproducible() {
        let model = xsinx_model();
        let sampler = JointSampler::new(&model, &array![[2.], [18.]]).unwrap();
        let mut rng1 = Xoshiro256Plus::seed_from_u64(1);
        let mut rng2 = Xoshiro256Plus::seed_from_u64(1);
        assert_eq!(sampler.sample(&mut rng1), sampler.sample(&mut rng2));
    }

    #[test]
    fn test_sampled_argmin() {
        let f = array![3., 1., 2., 0.];
        // no constraint
        assert_eq!(sampled_argmin(&f, &[], &[], &[]), 3);
        // best feasible point
        let g = [array![0., 0., 0., 1.]];
        assert_eq!(sampled_argmin(&f, &g, &[1e-4], &[1.]), 1);
        // no feasible point: least total violation
        let g = [array![2., 3., 0.5, 1.], array![0., 0., 1., 1.]];
        assert_eq!(sampled_argmin(&f, &g, &[0., 0.], &[1., 1.]), 2);
    }

    #[test]
    fn test_sampled_argmin_scaled_violation() {
        let f = array![0., 1.];
        // a constraint of order 1000 and a constraint of order 1, none feasible
        let g = [array![1000., 0.], array![0., 1.5]];
        let tols = [0., 0.];
        // raw violations 1000 vs 1.5: the large constraint dominates
        assert_eq!(sampled_argmin(&f, &g, &tols, &[1., 1.]), 1);
        // scaled violations 1 vs 1.5
        assert_eq!(sampled_argmin(&f, &g, &tols, &[1000., 1.]), 0);
        // feasibility does not depend on scales
        let g = [array![0., 1000.], array![0., 0.]];
        assert_eq!(sampled_argmin(&f, &g, &tols, &[1000., 1.]), 0);
    }

    #[test]
    fn test_candidate_set() {
        let xlimits = array![[0., 25.], [-1., 1.]];
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let xc = candidate_set(&xlimits, array![24.9, 0.].view(), 50, &mut rng);
        assert_eq!(xc.nrows(), 60);
        for row in xc.rows() {
            for (v, lim) in row.iter().zip(xlimits.rows()) {
                assert!(*v >= lim[0] && *v <= lim[1]);
            }
        }
        // local points around the best point
        let local = xc.slice(ndarray::s![50.., ..]);
        assert!(
            local
                .column(0)
                .iter()
                .all(|v| (v - 24.9).abs() < 25. * 0.05 * 6.)
        );
    }
}

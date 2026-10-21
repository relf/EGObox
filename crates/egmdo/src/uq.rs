//! Uncertainty quantification of the quantities of interest at design points.
//!
//! At a design point `z`, the random MDA is solved for samples `Ξ^(k)` of the disciplinary
//! GP uncertainty, the objective and constraints are evaluated on the solutions, and their
//! PCEs are fitted by least squares (specification, sections 3, 4 and 17).
//!
//! The random MDA is either solved for every sample, as in the papers, or linearized at the
//! solution of the mean MDA (decision D37).
//!
//! Across design points, the disciplinary GP uncertainty is either perfectly correlated, one
//! sample `Ξ` being shared by every point as in the papers, or correlated as the GP posteriors
//! (decision D38): [JointSamples] gives the joint samples of the `DoE_UQ` points in both cases.

use crate::errors::{EgmdoError, Result};
use crate::mda::{MdaSettings, SurrogateCouplings, solve_one, solve_random};
use crate::pce::{HermiteBasis, Pce, normal_samples};
use crate::problem::MdoProblem;
use crate::surrogates::DisciplinaryModels;
use linfa_linalg::eigh::EighInto;
use linfa_linalg::svd::SVD;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// The smallest ratio of the extreme singular values of `I - J`, scaled by the magnitudes of
/// the coupling variables, for which the random MDA is linearized (decision D37)
pub const LINEARIZATION_CONDITION: f64 = 1e-8;

/// How the random MDA is solved at a design point
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RandomMda {
    /// The simplified random MDA (9) is solved by Jacobi iterations for every sample,
    /// as in the EGMDO and C-EGMDO papers
    #[default]
    Solved,
    /// The random MDA is linearized at the solution `y_0` of the mean MDA (decision D37):
    /// `y(Ξ) = y_0 + (I - J)^-1 diag(σ) Ξ`, where `J` is the Jacobian of the GP means
    /// with respect to the coupling variables and `σ` the GP standard deviations, at `y_0`.
    /// The quantities of interest are evaluated on these samples, without any MDA.
    Linearized,
}

/// How the disciplinary GP uncertainty is correlated across design points
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Correlation {
    /// Perfectly correlated: one sample `Ξ` is shared by every design point, as in the EGMDO
    /// and C-EGMDO papers
    #[default]
    Perfect,
    /// Correlated as the GP posteriors (decision D38): the standard normal variable of
    /// coupling `q` at design point `z_i` has the correlation `ρ_q(i, l)` with the one at
    /// `z_l`, the posterior correlation of GP `q` between its inputs at the mean MDA
    /// solutions of both points
    GpPosterior,
}

/// Settings of the uncertainty quantification
#[derive(Clone, Debug)]
pub struct UqSettings {
    /// Total degree `d` of the PCEs
    pub degree: usize,
    /// Number of random MDAs `n_MC` per design point; `None` gives `max(200, 2P)` (decision D7)
    pub n_mc: Option<usize>,
    /// The KL decomposition keeps the modes explaining more than `1 - kl_threshold`
    /// of the variance
    pub kl_threshold: f64,
    /// How the random MDA is solved
    pub random_mda: RandomMda,
    /// How the disciplinary GP uncertainty is correlated across design points
    pub correlation: Correlation,
}

impl Default for UqSettings {
    fn default() -> Self {
        UqSettings {
            degree: 3,
            n_mc: None,
            kl_threshold: 1e-6,
            random_mda: RandomMda::default(),
            correlation: Correlation::default(),
        }
    }
}

impl UqSettings {
    /// The PCE basis for `n_couplings` random variables
    pub fn basis(&self, n_couplings: usize) -> HermiteBasis {
        HermiteBasis::new(n_couplings, self.degree)
    }

    /// The number of random MDAs per design point for a basis of `p` polynomials
    pub fn n_mc(&self, p: usize) -> usize {
        self.n_mc.unwrap_or_else(|| 200.max(2 * p))
    }

    /// Draws the regression samples of `Ξ`, shared by all design points (decision D6)
    pub fn regression_samples(&self, basis: &HermiteBasis, seed: u64) -> Array2<f64> {
        normal_samples(self.n_mc(basis.size()), basis.dim(), seed)
    }
}

/// The PCEs of the quantities of interest at a design point
#[derive(Clone, Debug)]
pub struct PointUq {
    /// PCEs of `[f_obj, g..., h...]`
    pub pce: Pce,
    /// Number of random MDAs which did not converge, excluded from the regression
    pub n_failed: usize,
    /// Whether at least `max(P, n_MC / 2)` random MDAs converged (decision D5)
    pub reliable: bool,
    /// Whether the random MDA was linearized at this point (decision D37)
    pub linearized: bool,
    /// The coupling variables of the mean MDA, or the mean of the converged random MDAs when
    /// the mean MDA does not converge
    pub y_mean: Array1<f64>,
}

/// Computes the PCEs of the quantities of interest at design point `z` from the random MDAs
/// of the samples `xi` `(n_MC, n_couplings)`, solved or linearized as given by `random_mda`.
///
/// Non-converged random MDAs are excluded from the regression (decision D5). It fails if no
/// more than `P` of them converged. The linearization falls back to the solved random MDA
/// where the mean MDA does not converge or `I - J` is ill-conditioned (decision D37).
pub fn point_uq(
    problem: &MdoProblem,
    models: &DisciplinaryModels,
    z: &ArrayView1<f64>,
    xi: &ArrayView2<f64>,
    basis: &HermiteBasis,
    mda: &MdaSettings,
    random_mda: RandomMda,
) -> Result<PointUq> {
    let (y0, status) = solve_one(&SurrogateCouplings::mean(problem, models), z, mda)?;
    if !status.is_converged() {
        log::warn!("Mean MDA not converged at z = {z} ({status:?})");
    }
    if random_mda == RandomMda::Linearized {
        let y = if status.is_converged() {
            linearize(problem, models, z, &y0, xi)?
        } else {
            None
        };
        match y {
            Some(y) => {
                let zs = z
                    .broadcast((y.nrows(), z.len()))
                    .expect("broadcast design point");
                let qoi = problem.eval_qoi(&zs, &y.view())?;
                return Ok(PointUq {
                    pce: Pce::fit(basis, xi, &qoi.view())?,
                    n_failed: 0,
                    reliable: true,
                    linearized: true,
                    y_mean: y0,
                });
            }
            None => log::warn!("Random MDA not linearized at z = {z}: solved instead"),
        }
    }
    let couplings = SurrogateCouplings::random(problem, models, xi.view())?;
    let sol = solve_random(&couplings, z, mda)?;
    let converged = sol.converged();
    if converged.len() <= basis.size() {
        return Err(EgmdoError::MdaFailure(format!(
            "only {} of {} random MDAs converged at z = {z}, more than {} are needed",
            converged.len(),
            xi.nrows(),
            basis.size()
        )));
    }
    let y = sol.y.select(Axis(0), &converged);
    let zs = z
        .broadcast((converged.len(), z.len()))
        .expect("broadcast design point");
    let qoi = problem.eval_qoi(&zs, &y.view())?;
    let pce = Pce::fit(basis, &xi.select(Axis(0), &converged).view(), &qoi.view())?;
    let n_failed = sol.n_failed();
    if n_failed > 0 {
        log::warn!(
            "{n_failed} of {} random MDAs did not converge at z = {z}",
            xi.nrows()
        );
    }
    Ok(PointUq {
        pce,
        n_failed,
        reliable: converged.len() >= basis.size().max(xi.nrows() / 2),
        linearized: false,
        y_mean: if status.is_converged() {
            y0
        } else {
            y.mean_axis(Axis(0)).expect("converged random MDAs")
        },
    })
}

/// The Jacobian `J` `(n_couplings, n_couplings)` of the GP means with respect to the coupling
/// variables, `J_qp = ∂μ_q / ∂y_p`, and the GP standard deviations `σ`, at design point `z`
/// and coupling state `y`
pub fn coupling_jacobian(
    problem: &MdoProblem,
    models: &DisciplinaryModels,
    z: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    let n = problem.n_couplings();
    let mut jac = Array2::zeros((n, n));
    let mut sigma = Array1::zeros(n);
    let y = y.insert_axis(Axis(0));
    for (i, d) in problem.disciplines().iter().enumerate() {
        let x = d.inputs(z, &y);
        let (_, std) = models.predict(i, &x)?;
        let grads = models.predict_gradients(i, &x);
        let offset = d.design_input_indices().len();
        for (k, &q) in d.coupling_output_indices().iter().enumerate() {
            sigma[q] = std[[0, k]];
            for (m, &p) in d.coupling_input_indices().iter().enumerate() {
                jac[[q, p]] = grads[k][[0, offset + m]];
            }
        }
    }
    Ok((jac, sigma))
}

/// The coupling variables `y_0 + (I - J)^-1 diag(σ) Ξ` of the linearized random MDA at design
/// point `z` for the samples `xi`, one row per sample, or `None` if the mean MDA does not
/// converge or `I - J` is ill-conditioned (decision D37)
pub fn linearized_couplings(
    problem: &MdoProblem,
    models: &DisciplinaryModels,
    z: &ArrayView1<f64>,
    xi: &ArrayView2<f64>,
    mda: &MdaSettings,
) -> Result<Option<Array2<f64>>> {
    let (y0, status) = solve_one(&SurrogateCouplings::mean(problem, models), z, mda)?;
    if !status.is_converged() {
        log::warn!("Mean MDA not converged at z = {z} ({status:?})");
        return Ok(None);
    }
    linearize(problem, models, z, &y0, xi)
}

/// The linearized random MDA at design point `z` around the mean MDA solution `y0`, or `None`
/// if `I - J` is ill-conditioned
fn linearize(
    problem: &MdoProblem,
    models: &DisciplinaryModels,
    z: &ArrayView1<f64>,
    y0: &Array1<f64>,
    xi: &ArrayView2<f64>,
) -> Result<Option<Array2<f64>>> {
    let (jac, sigma) = coupling_jacobian(problem, models, z, &y0.view())?;
    // The coupling variables can have very different magnitudes: the conditioning is that of
    // B = S^-1 (I - J) S with S = diag(|y_0| + σ), which does not depend on their units.
    let scale = (y0.mapv(f64::abs) + &sigma).mapv(|v| if v > 0. { v } else { 1. });
    let scale_col = scale.view().insert_axis(Axis(1));
    let b = (Array2::eye(jac.nrows()) - jac) * &scale / scale_col;
    let Ok((Some(u), s, Some(vt))) = b.svd(true, true) else {
        log::warn!("SVD of I - J failed at z = {z}");
        return Ok(None);
    };
    let (s_min, s_max) = s
        .iter()
        .fold((f64::INFINITY, 0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if s_min.is_nan() || s_min <= LINEARIZATION_CONDITION * s_max {
        log::warn!("I - J is ill-conditioned at z = {z} (singular values {s} after scaling)");
        return Ok(None);
    }
    // B^-1 = V diag(1/s) U^T, (I - J)^-1 = S B^-1 S^-1, then L = (I - J)^-1 diag(σ)
    let b_inv = vt.t().dot(&(&u.t() / &s.insert_axis(Axis(1))));
    let l = b_inv * scale_col / &scale * &sigma;
    Ok(Some(xi.dot(&l.t()) + y0))
}

/// Computes [point_uq] at every design point of `z` `(n, nx)`, in parallel.
pub fn doe_uq(
    problem: &MdoProblem,
    models: &DisciplinaryModels,
    z: &ArrayView2<f64>,
    xi: &ArrayView2<f64>,
    basis: &HermiteBasis,
    mda: &MdaSettings,
    random_mda: RandomMda,
) -> Result<Vec<PointUq>> {
    (0..z.nrows())
        .into_par_iter()
        .map(|i| point_uq(problem, models, &z.row(i), xi, basis, mda, random_mda))
        .collect()
}

/// The PCE coefficients of quantity `k` at every design point, as a `(n_points, P)` matrix
pub fn coefficients(uq: &[PointUq], k: usize) -> Array2<f64> {
    let p = uq.first().map_or(0, |u| u.pce.basis().size());
    let mut coefs = Array2::zeros((uq.len(), p));
    for (i, u) in uq.iter().enumerate() {
        coefs.row_mut(i).assign(&u.pce.coefs_of(k));
    }
    coefs
}

/// The posterior correlation matrices `R_q` `(n_points, n_points)` of the coupling GPs across
/// the design points `z` `(n_points, nx)`, one per coupling variable `q` (decision D38).
///
/// `R_q[i, l]` is the posterior correlation of GP `q` between its inputs at the mean MDA
/// solutions of `z_i` and `z_l`, given by `uq`. A point where the GP variance is zero
/// (a training point) is uncorrelated with the others.
pub fn coupling_correlations(
    problem: &MdoProblem,
    models: &DisciplinaryModels,
    z: &Array2<f64>,
    uq: &[PointUq],
) -> Result<Vec<Array2<f64>>> {
    let n = z.nrows();
    let mut correlations = vec![Array2::eye(n); problem.n_couplings()];
    for (i, d) in problem.disciplines().iter().enumerate() {
        let mut x = Array2::zeros((n, d.input_dim()));
        for (k, u) in uq.iter().enumerate() {
            let y = u.y_mean.view().insert_axis(Axis(0));
            x.row_mut(k).assign(&d.inputs(&z.row(k), &y).row(0));
        }
        for (cov, &q) in models
            .predict_covariance(i, &x)?
            .into_iter()
            .zip(d.coupling_output_indices())
        {
            let std = cov.diag().mapv(|v| v.max(0.).sqrt());
            correlations[q] = Array2::from_shape_fn((n, n), |(k, l)| {
                if k == l {
                    1.
                } else if std[k] > 0. && std[l] > 0. {
                    (cov[[k, l]] / (std[k] * std[l])).clamp(-1., 1.)
                } else {
                    0.
                }
            });
        }
    }
    Ok(correlations)
}

/// The symmetric square root of a correlation matrix. Its eigenvalues below `1e-12` times the
/// largest one are round-off and set to zero, so that a perfect correlation gives identical
/// samples.
fn correlation_sqrt(r: &Array2<f64>) -> Result<Array2<f64>> {
    let (values, vectors) = r.clone().eigh_into().map_err(|e| {
        EgmdoError::InvalidData(format!("eigendecomposition of a correlation failed: {e}"))
    })?;
    let tol = 1e-12 * values.iter().fold(0f64, |m, &v| m.max(v));
    let root = values.mapv(|v| if v > tol { v.sqrt() } else { 0. });
    Ok((&vectors * &root).dot(&vectors.t()))
}

/// Joint samples of the standard normal variables of the disciplinary GP uncertainty at the
/// `DoE_UQ` points
#[derive(Clone, Debug)]
pub enum JointSamples {
    /// One sample `Ξ` `(n, n_couplings)` shared by every point (perfect correlation)
    Shared(Array2<f64>),
    /// Per point, the samples `(n, n_couplings)` of its own variables (decision D38)
    PerPoint(Vec<Array2<f64>>),
}

impl JointSamples {
    /// `n` samples with the correlation `correlation`, for the design points `z` with the
    /// uncertainty quantifications `uq`
    pub fn draw(
        correlation: Correlation,
        problem: &MdoProblem,
        models: &DisciplinaryModels,
        z: &Array2<f64>,
        uq: &[PointUq],
        n: usize,
        seed: u64,
    ) -> Result<Self> {
        match correlation {
            Correlation::Perfect => Ok(JointSamples::Shared(normal_samples(
                n,
                problem.n_couplings(),
                seed,
            ))),
            Correlation::GpPosterior => {
                let roots = coupling_correlations(problem, models, z, uq)?
                    .iter()
                    .map(correlation_sqrt)
                    .collect::<Result<Vec<_>>>()?;
                Ok(JointSamples::correlated(&roots, n, seed))
            }
        }
    }

    /// `n` samples whose variable `q` has the correlation `roots[q]²` across the points
    pub fn correlated(roots: &[Array2<f64>], n: usize, seed: u64) -> Self {
        let n_points = roots.first().map_or(0, |r| r.nrows());
        let w = normal_samples(n, roots.len() * n_points, seed);
        let mut samples = vec![Array2::zeros((n, roots.len())); n_points];
        for (q, root) in roots.iter().enumerate() {
            // Correlated samples over the points: u_q = W_q R_q^(1/2)
            let u = w
                .slice(ndarray::s![.., q * n_points..(q + 1) * n_points])
                .dot(root);
            for (i, s) in samples.iter_mut().enumerate() {
                s.column_mut(q).assign(&u.column(i));
            }
        }
        JointSamples::PerPoint(samples)
    }

    /// The number of samples
    pub fn n_samples(&self) -> usize {
        match self {
            JointSamples::Shared(xi) => xi.nrows(),
            JointSamples::PerPoint(s) => s.first().map_or(0, |s| s.nrows()),
        }
    }

    /// The samples `(n, n_couplings)` of the variables of point `i`
    pub fn point(&self, i: usize) -> ArrayView2<'_, f64> {
        match self {
            JointSamples::Shared(xi) => xi.view(),
            JointSamples::PerPoint(s) => s[i].view(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::mda::Couplings;
    use crate::problems;
    use crate::surrogates::{GpSettings, initial_doe};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    pub(crate) fn toy_models() -> (problems::TestProblem, DisciplinaryModels) {
        // Paper setting: 5 and 4 initial disciplinary points
        let tp = problems::toy();
        let data = [5, 4]
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let x = initial_doe(&tp.problem, i, n, 10 + i as u64);
                let y = tp.eval_discipline(i, &x);
                (x, y)
            })
            .collect();
        let models = DisciplinaryModels::fit(&tp.problem, data, GpSettings::default()).unwrap();
        (tp, models)
    }

    #[test]
    fn test_n_mc() {
        let settings = UqSettings::default();
        assert_eq!(settings.n_mc(10), 200);
        assert_eq!(settings.n_mc(286), 572);
        let settings = UqSettings {
            n_mc: Some(100),
            ..UqSettings::default()
        };
        assert_eq!(settings.n_mc(10), 100);
    }

    #[test]
    fn test_pce_moments_match_monte_carlo() {
        let (tp, models) = toy_models();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 0);
        for z in [array![-2.5], array![0.5]] {
            let uq = point_uq(
                &tp.problem,
                &models,
                &z.view(),
                &xi.view(),
                &basis,
                &tp.mda,
                RandomMda::Solved,
            )
            .unwrap();
            assert!(uq.reliable);

            // Reference moments from a large independent Monte Carlo sample
            let xi_mc = normal_samples(20_000, 2, 7);
            let couplings = SurrogateCouplings::random(&tp.problem, &models, xi_mc.view()).unwrap();
            let sol = solve_random(&couplings, &z.view(), &tp.mda).unwrap();
            let y = sol.y.select(Axis(0), &sol.converged());
            let zs = z.broadcast((y.nrows(), 1)).unwrap();
            let f = tp
                .problem
                .eval_qoi(&zs, &y.view())
                .unwrap()
                .column(0)
                .to_owned();
            let (mean_mc, std_mc) = (f.mean().unwrap(), f.std(0.));

            let (mean, var) = (uq.pce.mean()[0], uq.pce.variance()[0]);
            assert_abs_diff_eq!(mean, mean_mc, epsilon = 0.1 * std_mc);
            assert!(
                (var.sqrt() / std_mc - 1.).abs() < 0.2,
                "z = {z}: PCE std {} vs Monte Carlo std {std_mc}",
                var.sqrt()
            );
        }
    }

    #[test]
    fn test_doe_uq_with_constraints() {
        let tp = problems::sellar_constrained();
        let data = (0..2)
            .map(|i| {
                let x = initial_doe(&tp.problem, i, 5, 20 + i as u64);
                let y = tp.eval_discipline(i, &x);
                (x, y)
            })
            .collect();
        let models = DisciplinaryModels::fit(&tp.problem, data, GpSettings::default()).unwrap();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 1);
        let z = array![[1., 1., 1.], [2., 0., 0.], [-3., 5., 5.]];
        let uq = doe_uq(
            &tp.problem,
            &models,
            &z.view(),
            &xi.view(),
            &basis,
            &tp.mda,
            RandomMda::Solved,
        )
        .unwrap();
        assert_eq!(uq.len(), 3);
        for u in uq.iter() {
            assert_eq!(u.pce.n_quantities(), 3, "objective and two constraints");
        }
        let coefs = coefficients(&uq, 1);
        assert_eq!(coefs.dim(), (3, 10));
        assert_abs_diff_eq!(coefs[[2, 0]], uq[2].pce.mean()[1]);

        // Reproducible, and independent of the parallel scheduling
        let again = doe_uq(
            &tp.problem,
            &models,
            &z.view(),
            &xi.view(),
            &basis,
            &tp.mda,
            RandomMda::Solved,
        )
        .unwrap();
        assert_eq!(coefficients(&again, 0), coefficients(&uq, 0));
    }

    #[test]
    fn test_failed_random_mdas() {
        let (tp, models) = toy_models();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 0);
        let mda = MdaSettings {
            max_iters: 1,
            ..MdaSettings::default()
        };
        assert!(
            point_uq(
                &tp.problem,
                &models,
                &array![0.5].view(),
                &xi.view(),
                &basis,
                &mda,
                RandomMda::Solved
            )
            .is_err()
        );
    }

    #[test]
    fn test_coupling_jacobian_matches_finite_differences() {
        let (tp, models) = toy_models();
        let mean = SurrogateCouplings::mean(&tp.problem, &models);
        for z in [array![-2.5], array![0.5]] {
            let (y0, status) = solve_one(&mean, &z.view(), &tp.mda).unwrap();
            assert!(status.is_converged());
            let (jac, sigma) =
                coupling_jacobian(&tp.problem, &models, &z.view(), &y0.view()).unwrap();
            let h = 1e-6;
            for p in 0..2 {
                let mut y = ndarray::stack![Axis(0), y0, y0];
                y[[0, p]] += h;
                y[[1, p]] -= h;
                let f = mean.update(&z.view(), &y.view(), &[0, 1]).unwrap();
                let fd = (&f.row(0) - &f.row(1)) / (2. * h);
                for q in 0..2 {
                    assert_abs_diff_eq!(jac[[q, p]], fd[q], epsilon = 1e-5 * (1. + fd[q].abs()));
                }
            }
            // A discipline does not receive its own output
            assert_eq!(jac[[0, 0]], 0.);
            assert_eq!(jac[[1, 1]], 0.);
            let (_, std) = models
                .predict(
                    0,
                    &tp.problem.disciplines()[0].inputs(&z.view(), &y0.view().insert_axis(Axis(0))),
                )
                .unwrap();
            assert_abs_diff_eq!(sigma[0], std[[0, 0]]);
        }
    }

    #[test]
    fn test_linearization_is_first_order() {
        // For small samples t Ξ, the solved random MDA is y_0 + t L Ξ + O(t²)
        let (tp, models) = toy_models();
        let xi = normal_samples(5, 2, 3);
        let z = array![-2.5];
        let y_lin = linearized_couplings(&tp.problem, &models, &z.view(), &xi.view(), &tp.mda)
            .unwrap()
            .unwrap();
        let y0 = linearized_couplings(
            &tp.problem,
            &models,
            &z.view(),
            &Array2::zeros((1, 2)).view(),
            &tp.mda,
        )
        .unwrap()
        .unwrap()
        .row(0)
        .to_owned();
        let mda = MdaSettings {
            tol: 1e-12,
            ..tp.mda.clone()
        };
        for t in [1e-2, 1e-3] {
            let xi_t = &xi * t;
            let couplings = SurrogateCouplings::random(&tp.problem, &models, xi_t.view()).unwrap();
            let sol = solve_random(&couplings, &z.view(), &mda).unwrap();
            assert_eq!(sol.n_failed(), 0);
            let first_order = (&y_lin - &y0) * t + &y0;
            let err = (&sol.y - &first_order)
                .mapv(f64::abs)
                .fold(0f64, |m, &v| m.max(v));
            let step = (&y_lin - &y0).mapv(f64::abs).fold(0f64, |m, &v| m.max(v)) * t;
            assert!(err < 0.05 * step, "t = {t}: error {err} for a step {step}");
        }
    }

    #[test]
    fn test_linearized_pce() {
        let (tp, models) = toy_models();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 0);
        for z in [array![-2.5], array![0.5]] {
            let solved = point_uq(
                &tp.problem,
                &models,
                &z.view(),
                &xi.view(),
                &basis,
                &tp.mda,
                RandomMda::Solved,
            )
            .unwrap();
            let lin = point_uq(
                &tp.problem,
                &models,
                &z.view(),
                &xi.view(),
                &basis,
                &tp.mda,
                RandomMda::Linearized,
            )
            .unwrap();
            assert!(lin.linearized && !solved.linearized);
            assert!(lin.reliable);
            assert_eq!(lin.n_failed, 0);
            // Same order of magnitude as the solved random MDA; the difference is the
            // nonlinearity of the random MDA
            let (s_lin, s_sol) = (
                lin.pce.variance()[0].sqrt(),
                solved.pce.variance()[0].sqrt(),
            );
            assert!(
                (s_lin / s_sol - 1.).abs() < 0.5,
                "z = {z}: std {s_lin} linearized vs {s_sol} solved"
            );
            assert_abs_diff_eq!(
                lin.pce.mean()[0],
                solved.pce.mean()[0],
                epsilon = 2. * s_sol
            );
        }
    }

    #[test]
    fn test_linearized_falls_back_to_solved() {
        // The mean MDA does not converge in one iteration: the random MDA is solved, and fails
        let (tp, models) = toy_models();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 0);
        let mda = MdaSettings {
            max_iters: 1,
            ..MdaSettings::default()
        };
        assert!(
            linearized_couplings(&tp.problem, &models, &array![0.5].view(), &xi.view(), &mda)
                .unwrap()
                .is_none()
        );
        assert!(
            point_uq(
                &tp.problem,
                &models,
                &array![0.5].view(),
                &xi.view(),
                &basis,
                &mda,
                RandomMda::Linearized
            )
            .is_err()
        );
    }

    fn toy_doe_uq(z: &Array2<f64>) -> (problems::TestProblem, DisciplinaryModels, Vec<PointUq>) {
        let (tp, models) = toy_models();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 0);
        let uq = doe_uq(
            &tp.problem,
            &models,
            &z.view(),
            &xi.view(),
            &basis,
            &tp.mda,
            RandomMda::Solved,
        )
        .unwrap();
        (tp, models, uq)
    }

    #[test]
    fn test_coupling_correlations() {
        let z = array![[-4.3], [-2.5], [0.5], [0.6], [3.7]];
        let (tp, models, uq) = toy_doe_uq(&z);
        for u in uq.iter() {
            assert_eq!(u.y_mean.len(), 2);
        }
        let correlations = coupling_correlations(&tp.problem, &models, &z, &uq).unwrap();
        assert_eq!(correlations.len(), 2);
        for r in correlations.iter() {
            assert_eq!(r.dim(), (5, 5));
            for i in 0..5 {
                assert_eq!(r[[i, i]], 1.);
                for l in 0..5 {
                    assert_abs_diff_eq!(r[[i, l]], r[[l, i]], epsilon = 1e-12);
                    assert!(r[[i, l]].abs() <= 1.);
                }
            }
            let (values, _) = r.clone().eigh_into().unwrap();
            assert!(values.iter().all(|&v| v > -1e-8), "PSD: {values}");
            // Close points are more correlated than distant ones
            assert!(r[[2, 3]] > r[[2, 4]].abs(), "{r}");
            assert!(r[[2, 3]] > 0.9, "{r}");
        }
    }

    #[test]
    fn test_correlated_joint_samples() {
        let r = array![[1., 0.8, 0.], [0.8, 1., -0.5], [0., -0.5, 1.]];
        let roots = vec![correlation_sqrt(&r).unwrap(), Array2::eye(3)];
        let samples = JointSamples::correlated(&roots, 50_000, 3);
        assert_eq!(samples.n_samples(), 50_000);
        for q in 0..2 {
            for i in 0..3 {
                let ui = samples.point(i).column(q).to_owned();
                assert_abs_diff_eq!(ui.mean().unwrap(), 0., epsilon = 0.02);
                assert_abs_diff_eq!(ui.std(0.), 1., epsilon = 0.02);
                for l in 0..3 {
                    let ul = samples.point(l).column(q).to_owned();
                    let rho = (&ui * &ul).mean().unwrap();
                    let expected = if q == 0 {
                        r[[i, l]]
                    } else {
                        (i == l) as u8 as f64
                    };
                    assert_abs_diff_eq!(rho, expected, epsilon = 0.02);
                }
            }
            // Variables of different couplings are independent
            let p0 = samples.point(0);
            let (u0, u1) = (p0.column(0), p0.column(1));
            assert_abs_diff_eq!((&u0 * &u1).mean().unwrap(), 0., epsilon = 0.02);
        }

        // Perfect correlation: every point has the same samples
        let ones = Array2::from_elem((3, 3), 1.);
        let samples = JointSamples::correlated(&[correlation_sqrt(&ones).unwrap()], 100, 4);
        for i in 1..3 {
            assert_abs_diff_eq!(samples.point(i), samples.point(0), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mehler_covariance_matches_joint_samples() {
        use crate::criteria::qoi_joint_samples;
        use crate::kl::mehler_covariance;
        let z = array![[-4.3], [-2.5], [0.5], [3.7]];
        let (tp, models, uq) = toy_doe_uq(&z);
        let coefs = coefficients(&uq, 0);
        let basis = uq[0].pce.basis();

        // All correlations 1: the covariance of shared variables
        let ones = vec![Array2::from_elem((4, 4), 1.); 2];
        let a = coefs.slice(ndarray::s![.., 1..]);
        assert_abs_diff_eq!(
            mehler_covariance(&coefs, basis, &ones),
            a.dot(&a.t()),
            epsilon = 1e-10
        );

        // GP correlations: Mehler's formula is the covariance of the joint samples
        let samples = JointSamples::draw(
            Correlation::GpPosterior,
            &tp.problem,
            &models,
            &z,
            &uq,
            100_000,
            5,
        )
        .unwrap();
        let correlations = coupling_correlations(&tp.problem, &models, &z, &uq).unwrap();
        let cov = mehler_covariance(&coefs, basis, &correlations);
        let values = qoi_joint_samples(&uq, &samples);
        let scale = cov.diag().iter().fold(0f64, |m, &v| m.max(v));
        for i in 0..4 {
            for l in 0..4 {
                let (vi, vl) = (values[i].column(0), values[l].column(0));
                let empirical = ((&vi - vi.mean().unwrap()) * (&vl - vl.mean().unwrap()))
                    .mean()
                    .unwrap();
                assert_abs_diff_eq!(empirical, cov[[i, l]], epsilon = 0.03 * scale);
            }
        }
    }

    #[test]
    fn test_zeta_from_values_with_shared_samples() {
        use crate::criteria::qoi_samples;
        use crate::kl::KlDecomposition;
        let z = array![[-4.3], [-2.5], [0.5], [3.7]];
        let (_, _, uq) = toy_doe_uq(&z);
        let kl = KlDecomposition::new(&coefficients(&uq, 0), 1e-6).unwrap();
        let xi = normal_samples(50, 2, 9);
        let values = qoi_samples(&uq, &xi.view());
        let mut objective = Array2::zeros((4, 50));
        for (i, v) in values.iter().enumerate() {
            objective.row_mut(i).assign(&v.column(0));
        }
        let zeta = kl.zeta(uq[0].pce.basis(), &xi.view());
        let scale = zeta.iter().fold(1f64, |m, &v| m.max(v.abs()));
        assert_abs_diff_eq!(
            kl.zeta_from_values(&objective),
            zeta,
            epsilon = 1e-10 * scale
        );
    }
}

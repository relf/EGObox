//! Uncertainty quantification of the quantities of interest at design points.
//!
//! At a design point `z`, the random MDA is solved for samples `Ξ^(k)` of the disciplinary
//! GP uncertainty, the objective and constraints are evaluated on the solutions, and their
//! PCEs are fitted by least squares (specification, sections 3, 4 and 17).

use crate::errors::{EgmdoError, Result};
use crate::mda::{MdaSettings, SurrogateCouplings, solve_random};
use crate::pce::{HermiteBasis, Pce, normal_samples};
use crate::problem::MdoProblem;
use crate::surrogates::DisciplinaryModels;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

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
}

impl Default for UqSettings {
    fn default() -> Self {
        UqSettings {
            degree: 3,
            n_mc: None,
            kl_threshold: 1e-6,
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
}

/// Computes the PCEs of the quantities of interest at design point `z` from the random MDAs
/// of the samples `xi` `(n_MC, n_couplings)`.
///
/// Non-converged random MDAs are excluded from the regression (decision D5). It fails if no
/// more than `P` of them converged.
pub fn point_uq(
    problem: &MdoProblem,
    models: &DisciplinaryModels,
    z: &ArrayView1<f64>,
    xi: &ArrayView2<f64>,
    basis: &HermiteBasis,
    mda: &MdaSettings,
) -> Result<PointUq> {
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
    })
}

/// Computes [point_uq] at every design point of `z` `(n, nx)`, in parallel.
pub fn doe_uq(
    problem: &MdoProblem,
    models: &DisciplinaryModels,
    z: &ArrayView2<f64>,
    xi: &ArrayView2<f64>,
    basis: &HermiteBasis,
    mda: &MdaSettings,
) -> Result<Vec<PointUq>> {
    (0..z.nrows())
        .into_par_iter()
        .map(|i| point_uq(problem, models, &z.row(i), xi, basis, mda))
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
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
            let uq =
                point_uq(&tp.problem, &models, &z.view(), &xi.view(), &basis, &tp.mda).unwrap();
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
        let uq = doe_uq(&tp.problem, &models, &z.view(), &xi.view(), &basis, &tp.mda).unwrap();
        assert_eq!(uq.len(), 3);
        for u in uq.iter() {
            assert_eq!(u.pce.n_quantities(), 3, "objective and two constraints");
        }
        let coefs = coefficients(&uq, 1);
        assert_eq!(coefs.dim(), (3, 10));
        assert_abs_diff_eq!(coefs[[2, 0]], uq[2].pce.mean()[1]);

        // Reproducible, and independent of the parallel scheduling
        let again = doe_uq(&tp.problem, &models, &z.view(), &xi.view(), &basis, &tp.mda).unwrap();
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
                &mda
            )
            .is_err()
        );
    }
}

//! Continuous models of the constraints (specification, section 17).
//!
//! The mean of the PCE of each constraint at the `DoE_UQ` points is interpolated over the
//! design space by a GP, as in SEGO ([ConstraintModels]). Constraints declared as depending on
//! the design variables only are evaluated directly (decision D22). As an option, the
//! constraints of the acquisition come instead from the MDA solved on the means of the
//! disciplinary GPs ([MeanMdaQoi], decision D40).

use crate::criteria::Feasibility;
use crate::errors::{EgmdoError, Result};
use crate::mda::{Couplings, MdaSettings, SurrogateCouplings, solve_one};
use crate::problem::MdoProblem;
use crate::surrogates::{DisciplinaryGp, DisciplinaryModels, GpSettings};
use crate::uq::PointUq;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::sync::Mutex;
use std::thread::ThreadId;

/// The value of a constraint `c_k(z) <= 0` of the acquisition where the mean MDA does not
/// converge (D40)
pub const NON_CONVERGED_VIOLATION: f64 = 1e10;

/// The constraint `j` and its sign for inequality `k` of the acquisition optimization: one
/// per inequality constraint `g`, then two per equality constraint `h` (`h` and `-h`)
fn inequality(n_ineq: usize, k: usize) -> (usize, f64) {
    if k < n_ineq {
        (k, 1.)
    } else {
        let r = (k - n_ineq) / 2;
        let sign = if (k - n_ineq).is_multiple_of(2) {
            1.
        } else {
            -1.
        };
        (n_ineq + r, sign)
    }
}

/// The models of the constraints `[g..., h...]` of a problem
#[derive(Clone)]
pub struct ConstraintModels {
    problem: MdoProblem,
    /// One GP per constraint, `None` for the deterministic ones
    gps: Vec<Option<DisciplinaryGp>>,
}

impl ConstraintModels {
    /// Interpolates the constraint PCE means of `uq` over the `DoE_UQ` points `z_uq` `(n, nx)`
    pub fn fit(
        problem: &MdoProblem,
        z_uq: &Array2<f64>,
        uq: &[PointUq],
        settings: &GpSettings,
    ) -> Result<Self> {
        let n_constraints = problem.n_ineq() + problem.n_eq();
        if let Some(&k) = problem
            .deterministic_constraint_indices()
            .iter()
            .find(|&&k| k >= n_constraints)
        {
            return Err(EgmdoError::InvalidProblem(format!(
                "deterministic constraint {k} does not exist ({n_constraints} constraints)"
            )));
        }
        let gps = (0..n_constraints)
            .map(|k| {
                if problem.is_deterministic_constraint(k) {
                    Ok(None)
                } else {
                    let means: Array1<f64> = uq.iter().map(|u| u.pce.mean()[1 + k]).collect();
                    settings.train(z_uq, &means).map(Some)
                }
            })
            .collect::<Result<_>>()?;
        Ok(ConstraintModels {
            problem: problem.clone(),
            gps,
        })
    }

    /// The number of constraints
    pub fn n_constraints(&self) -> usize {
        self.gps.len()
    }

    /// The modelled constraints `[g..., h...]` at design points `z` `(n, nx)`
    pub fn predict(&self, z: &Array2<f64>) -> Result<Array2<f64>> {
        let mut values = Array2::zeros((z.nrows(), self.gps.len()));
        if !self.problem.deterministic_constraint_indices().is_empty() {
            let exact = self.problem.eval_deterministic_constraints(&z.view())?;
            for &k in self.problem.deterministic_constraint_indices() {
                values.column_mut(k).assign(&exact.column(k));
            }
        }
        for (k, gp) in self.gps.iter().enumerate() {
            if let Some(gp) = gp {
                values.column_mut(k).assign(&gp.predict(z)?);
            }
        }
        Ok(values)
    }

    /// The modelled constraints `[g..., h...]` at design point `z` with the standard deviations
    /// of their GPs (0 for the deterministic constraints)
    pub fn predict_with_std(&self, z: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let z = z.view().insert_axis(ndarray::Axis(0)).to_owned();
        let mean = self.predict(&z)?.row(0).to_owned();
        let mut std = Array1::zeros(self.gps.len());
        for (k, gp) in self.gps.iter().enumerate() {
            if let Some(gp) = gp {
                std[k] = gp.predict_var(&z)?[0].max(0.).sqrt();
            }
        }
        Ok((mean, std))
    }

    /// The logarithm of the probability of feasibility of the modelled constraints at design
    /// point `z` (D39), the deterministic ones excluded
    pub fn log_probability_of_feasibility(
        &self,
        feasibility: &Feasibility,
        z: &Array1<f64>,
    ) -> Result<f64> {
        let (mean, std) = self.predict_with_std(z)?;
        Ok(feasibility
            .log_probability_of_feasibility(&mean.view(), &std.view(), |k| self.gps[k].is_none()))
    }

    /// The constraints `c_k(z) <= 0` of the deterministic constraints only, as
    /// [ConstraintModels::as_inequalities]
    pub fn deterministic_inequalities<'a>(
        &'a self,
        feasibility: &'a Feasibility,
    ) -> (usize, impl Fn(usize, &Array1<f64>) -> f64 + Sync + 'a) {
        let (_, all) = self.as_inequalities(feasibility);
        let n_ineq = feasibility.n_ineq();
        // The indices, among the inequalities of as_inequalities, of the deterministic ones
        let mut indices = vec![];
        let mut k = 0;
        for (j, gp) in self.gps.iter().enumerate() {
            let n = if j < n_ineq { 1 } else { 2 };
            if gp.is_none() {
                indices.extend(k..k + n);
            }
            k += n;
        }
        (indices.len(), move |i: usize, z: &Array1<f64>| {
            all(indices[i], z)
        })
    }

    /// The constraints of the acquisition optimization `c_k(z) <= 0` (C-23): `g - τ` for each
    /// inequality constraint, and `h - τ`, `-h - τ` for each equality constraint.
    /// Returns their number and a function `(k, z) -> c_k(z)`.
    pub fn as_inequalities<'a>(
        &'a self,
        feasibility: &'a Feasibility,
    ) -> (usize, impl Fn(usize, &Array1<f64>) -> f64 + Sync + 'a) {
        let n_ineq = feasibility.n_ineq();
        let n = n_ineq + 2 * (self.gps.len() - n_ineq);
        let constraint = move |k: usize, z: &Array1<f64>| {
            let (j, sign) = inequality(n_ineq, k);
            let z = z.view().insert_axis(ndarray::Axis(0)).to_owned();
            let value = match &self.gps[j] {
                Some(gp) => gp.predict(&z).map(|v| v[0]),
                None => self
                    .problem
                    .eval_deterministic_constraints(&z.view())
                    .map(|c| c[[0, j]]),
            };
            value.map_or(f64::INFINITY, |v| sign * v - feasibility.tolerances()[j])
        };
        (n, constraint)
    }
}

/// The last design point of each thread, with its mean MDA quantities of interest
type QoiCache = Mutex<HashMap<ThreadId, (Array1<f64>, Option<Array1<f64>>)>>;

/// The quantities of interest `[f_obj, g..., h...]` of the MDA solved on the means of the
/// disciplinary GPs at a design point (the mean MDA), used as the constraints of the
/// acquisition with [AcquisitionConstraints::MeanMda](crate::AcquisitionConstraints) (D40).
///
/// The optimizer queries the constraints one by one at the same point, from several threads:
/// the last point of each thread is cached, so that one MDA is solved per point.
pub struct MeanMdaQoi<'a> {
    couplings: SurrogateCouplings<'a>,
    settings: &'a MdaSettings,
    cache: QoiCache,
}

impl<'a> MeanMdaQoi<'a> {
    /// The mean MDA of `problem` with the disciplinary GPs `models`
    pub fn new(
        problem: &'a MdoProblem,
        models: &'a DisciplinaryModels,
        settings: &'a MdaSettings,
    ) -> Self {
        MeanMdaQoi {
            couplings: SurrogateCouplings::mean(problem, models),
            settings,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The quantities of interest `[f_obj, g..., h...]` of the mean MDA at design point `z`,
    /// or `None` if the mean MDA does not converge there
    pub fn qoi(&self, z: &Array1<f64>) -> Result<Option<Array1<f64>>> {
        let thread = std::thread::current().id();
        if let Some((zc, qc)) = self.cache.lock().expect("cache").get(&thread)
            && zc == z
        {
            return Ok(qc.clone());
        }
        let (y, status) = solve_one(&self.couplings, &z.view(), self.settings)?;
        let qoi = if status.is_converged() {
            let problem = self.couplings.problem();
            let q = problem.eval_qoi(
                &z.view().insert_axis(Axis(0)),
                &y.view().insert_axis(Axis(0)),
            )?;
            Some(q.row(0).to_owned())
        } else {
            None
        };
        self.cache
            .lock()
            .expect("cache")
            .insert(thread, (z.clone(), qoi.clone()));
        Ok(qoi)
    }

    /// The constraints of the acquisition optimization `c_k(z) <= 0` from the mean MDA, as
    /// [ConstraintModels::as_inequalities]; [NON_CONVERGED_VIOLATION] where the mean MDA does
    /// not converge
    pub fn as_inequalities<'b>(
        &'b self,
        feasibility: &'b Feasibility,
    ) -> (usize, impl Fn(usize, &Array1<f64>) -> f64 + Sync + 'b) {
        let n_ineq = feasibility.n_ineq();
        let n_constraints = feasibility.n_constraints();
        let n = n_ineq + 2 * (n_constraints - n_ineq);
        let constraint = move |k: usize, z: &Array1<f64>| {
            let (j, sign) = inequality(n_ineq, k);
            match self.qoi(z) {
                Ok(Some(q)) => sign * q[1 + j] - feasibility.tolerances()[j],
                _ => NON_CONVERGED_VIOLATION,
            }
        };
        (n, constraint)
    }

    /// The violation of the constraints of the mean MDA beyond their tolerances,
    /// `Σ max(g - τ, 0) + Σ max(|h| - τ, 0)`, minimized by the feasibility-first acquisition
    /// with these constraints (D39, D40); [NON_CONVERGED_VIOLATION] where the mean MDA does not
    /// converge
    pub fn violation(&self, feasibility: &Feasibility, z: &Array1<f64>) -> f64 {
        match self.qoi(z) {
            Ok(Some(q)) => {
                let n_ineq = feasibility.n_ineq();
                (0..feasibility.n_constraints())
                    .map(|j| {
                        let c = if j < n_ineq { q[1 + j] } else { q[1 + j].abs() };
                        (c - feasibility.tolerances()[j]).max(0.)
                    })
                    .sum()
            }
            _ => NON_CONVERGED_VIOLATION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems;
    use crate::surrogates::{DisciplinaryModels, initial_doe};
    use crate::uq::{RandomMda, UqSettings, doe_uq};
    use approx::assert_abs_diff_eq;
    use ndarray::{Axis, array, concatenate};

    #[test]
    fn test_constraint_models() {
        // Constrained Sellar with an extra equality constraint depending on z only: z1 - z2 = 0
        let tp = problems::sellar_constrained();
        let qoi = |z: &ndarray::ArrayView2<f64>, y: &ndarray::ArrayView2<f64>| {
            let base = problems::sellar_constrained()
                .problem
                .eval_qoi(z, y)
                .unwrap();
            let h = (&z.column(0) - &z.column(1)).insert_axis(Axis(1));
            concatenate![Axis(1), base, h]
        };
        let problem = MdoProblem::new(tp.problem.xlimits(), tp.problem.disciplines().to_vec(), qoi)
            .unwrap()
            .constraints(2, 1)
            .deterministic_constraints(&[2]);

        let data = (0..2)
            .map(|i| {
                let x = initial_doe(&problem, i, 8, i as u64);
                let y = tp.eval_discipline(i, &x);
                (x, y)
            })
            .collect();
        let models = DisciplinaryModels::fit(&problem, data, GpSettings::default()).unwrap();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 0);
        let z_uq = array![[1., 1., 1.], [2., 0., 0.], [-3., 5., 5.], [0., 8., 2.]];
        let uq = doe_uq(
            &problem,
            &models,
            &z_uq.view(),
            &xi.view(),
            &basis,
            &tp.mda,
            RandomMda::Solved,
        )
        .unwrap();

        let cm = ConstraintModels::fit(&problem, &z_uq, &uq, &GpSettings::default()).unwrap();
        assert_eq!(cm.n_constraints(), 3);
        // The GPs interpolate the PCE means at DoE_UQ; the deterministic constraint is exact
        let pred = cm.predict(&z_uq).unwrap();
        for (i, u) in uq.iter().enumerate() {
            assert_abs_diff_eq!(pred[[i, 0]], u.pce.mean()[1], epsilon = 1e-6);
            assert_abs_diff_eq!(pred[[i, 1]], u.pce.mean()[2], epsilon = 1e-6);
        }
        let z = array![[0.5, 3., 1.], [-2., 7., 4.]];
        assert_abs_diff_eq!(cm.predict(&z).unwrap().column(2), array![-2.5, -9.]);

        // Acquisition constraints: g1, g2, then h - τ and -h - τ
        let feas = Feasibility::new(2, array![1e-4, 1e-4, 1e-2], 0.01);
        let (n, c) = cm.as_inequalities(&feas);
        assert_eq!(n, 4);
        let z0 = array![0.5, 3., 1.];
        let pred = cm
            .predict(&z0.view().insert_axis(Axis(0)).to_owned())
            .unwrap();
        assert_abs_diff_eq!(c(0, &z0), pred[[0, 0]] - 1e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(c(2, &z0), -2.5 - 1e-2, epsilon = 1e-12);
        assert_abs_diff_eq!(c(3, &z0), 2.5 - 1e-2, epsilon = 1e-12);

        // Mean MDA constraints (D40): the quantities of interest of the mean MDA
        let mean_mda = MeanMdaQoi::new(&problem, &models, &tp.mda);
        let (y, status) = solve_one(
            &SurrogateCouplings::mean(&problem, &models),
            &z0.view(),
            &tp.mda,
        )
        .unwrap();
        assert!(status.is_converged());
        let expected = problem
            .eval_qoi(
                &z0.view().insert_axis(Axis(0)),
                &y.view().insert_axis(Axis(0)),
            )
            .unwrap()
            .row(0)
            .to_owned();
        let q = mean_mda.qoi(&z0).unwrap().unwrap();
        assert_abs_diff_eq!(q, expected, epsilon = 1e-12);
        // Cached for the next constraints at the same point, recomputed at another one
        assert_eq!(mean_mda.qoi(&z0).unwrap().unwrap(), q);
        let z1 = array![-2., 7., 4.];
        assert_ne!(mean_mda.qoi(&z1).unwrap().unwrap(), q);
        let (n, c) = mean_mda.as_inequalities(&feas);
        assert_eq!(n, 4);
        assert_abs_diff_eq!(c(0, &z0), q[1] - 1e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(c(1, &z0), q[2] - 1e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(c(2, &z0), -2.5 - 1e-2, epsilon = 1e-12);
        assert_abs_diff_eq!(c(3, &z0), 2.5 - 1e-2, epsilon = 1e-12);
        let violation = (q[1] - 1e-4).max(0.) + (q[2] - 1e-4).max(0.) + (2.5 - 1e-2);
        assert_abs_diff_eq!(mean_mda.violation(&feas, &z0), violation, epsilon = 1e-12);

        // Unknown deterministic constraint
        let bad = problem.clone().deterministic_constraints(&[3]);
        assert!(ConstraintModels::fit(&bad, &z_uq, &uq, &GpSettings::default()).is_err());
    }
}

//! Continuous models of the constraints (specification, section 17).
//!
//! The mean of the PCE of each constraint at the `DoE_UQ` points is interpolated over the
//! design space by a GP, as in SEGO. Constraints declared as depending on the design variables
//! only are evaluated directly (decision D22).

use crate::criteria::Feasibility;
use crate::errors::{EgmdoError, Result};
use crate::problem::MdoProblem;
use crate::surrogates::{DisciplinaryGp, GpSettings};
use crate::uq::PointUq;
use ndarray::{Array1, Array2};

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
            let (j, sign) = if k < n_ineq {
                (k, 1.)
            } else {
                let r = (k - n_ineq) / 2;
                (
                    n_ineq + r,
                    if (k - n_ineq).is_multiple_of(2) {
                        1.
                    } else {
                        -1.
                    },
                )
            };
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

        // Unknown deterministic constraint
        let bad = problem.clone().deterministic_constraints(&[3]);
        assert!(ConstraintModels::fit(&bad, &z_uq, &uq, &GpSettings::default()).is_err());
    }
}

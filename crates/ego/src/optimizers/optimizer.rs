//! # Infill Criterion Optimizer
//!
//! This module provides optimization backends for the infill criterion.
//!
//! ## Design
//!
//! The optimizer follows the Strategy pattern with the [`InfillOptimizerTrait`] trait
//! defining the interface for optimization backends. This allows:
//!
//! - **Open/Closed Principle**: New optimizers can be added by implementing the trait
//! - **Dependency Inversion**: Code depends on the trait abstraction, not concrete types
//!
//! ## Available Backends
//!
//! - [`Algorithm::Slsqp`] - Sequential Least Squares Programming (gradient-based)
//! - [`Algorithm::Cobyla`] - Constrained Optimization BY Linear Approximations (derivative-free)
//!
//! The `basin` feature selects Basin's pure Rust implementations. Otherwise,
//! the `slsqp` and `cobyla` crates are used as the default.
//!
//! ## Usage
//!
//! The optimizer is used internally by [`EgorSolver`](crate::EgorSolver) to optimize
//! the infill criterion at each iteration.

use crate::InfillObjData;
use ndarray::{Array1, Array2, ArrayView1, arr1};

use crate::types::UserFn;
pub(crate) trait OptFn<U>: UserFn<U> + Sync {}
impl<T, U> OptFn<U> for T where T: UserFn<U> + Sync {}

#[cfg(not(feature = "basin"))]
use cobyla::RhoBeg;

#[derive(Copy, Clone, Debug)]
pub enum Algorithm {
    Cobyla,
    Slsqp,
}

pub const INFILL_MAX_EVAL_DEFAULT: usize = 2000;

/// Facade for various optimization algorithms
pub(crate) struct Optimizer<'a> {
    algo: Algorithm,
    fun: &'a (dyn OptFn<InfillObjData<f64>> + Sync),
    cons: Vec<&'a (dyn OptFn<InfillObjData<f64>> + Sync)>,
    cstr_tol: Option<Array1<f64>>,
    bounds: Array2<f64>,
    user_data: &'a InfillObjData<f64>,
    max_eval: usize,
    xinit: Option<Array1<f64>>,
    ftol_abs: Option<f64>,
    ftol_rel: Option<f64>,
}

impl<'a> Optimizer<'a> {
    pub fn new(
        algo: Algorithm,
        fun: &'a (dyn OptFn<InfillObjData<f64>> + Sync),
        cons: &[&'a (dyn OptFn<InfillObjData<f64>> + Sync)],
        user_data: &'a InfillObjData<f64>,
        bounds: &Array2<f64>,
    ) -> Self {
        Optimizer {
            algo,
            fun,
            cons: cons.to_vec(),
            cstr_tol: None,
            bounds: bounds.clone(),
            user_data,
            max_eval: INFILL_MAX_EVAL_DEFAULT,
            xinit: None,
            ftol_abs: None,
            ftol_rel: None,
        }
    }

    pub fn ftol_abs(&mut self, ftol_abs: f64) -> &mut Self {
        self.ftol_abs = Some(ftol_abs);
        self
    }

    pub fn ftol_rel(&mut self, ftol_rel: f64) -> &mut Self {
        self.ftol_rel = Some(ftol_rel);
        self
    }

    pub fn max_eval(&mut self, max_eval: usize) -> &mut Self {
        self.max_eval = max_eval;
        self
    }

    pub fn xinit(&mut self, xinit: &ArrayView1<f64>) -> &mut Self {
        self.xinit = Some(xinit.to_owned());
        self
    }

    #[cfg(feature = "basin")]
    pub fn minimize(&self) -> (f64, Array1<f64>) {
        use egobox_gp::basin_optimizer::{Algorithm as BasinAlgorithm, Settings, minimize};
        use std::cell::RefCell;
        let user_data = RefCell::new(self.user_data.clone());
        let bounds: Vec<_> = self.bounds.outer_iter().map(|r| (r[0], r[1])).collect();
        let xinit = self.xinit.as_ref().expect("initial point").to_vec();
        let cstr_tol = self
            .cstr_tol
            .clone()
            .unwrap_or(Array1::zeros(self.cons.len()));
        let objective =
            |x: &[f64], g: Option<&mut [f64]>| (self.fun)(x, g, &mut user_data.borrow_mut());
        let constraint = |i: usize, x: &[f64], g: Option<&mut [f64]>| {
            let mut data = user_data.borrow_mut();
            let scale = data.scale_cstr.as_ref().expect("constraint scaling")[i];
            (self.cons[i])(x, g, &mut data) - cstr_tol[i] / scale
        };
        let algorithm = match self.algo {
            Algorithm::Cobyla => BasinAlgorithm::Cobyla,
            Algorithm::Slsqp => BasinAlgorithm::Slsqp,
        };
        let (value, point) = minimize(
            algorithm,
            &objective,
            &constraint,
            Settings {
                xinit: &xinit,
                bounds: &bounds,
                n_constraints: self.cons.len(),
                max_evals: self.max_eval,
                ftol_abs: self.ftol_abs.unwrap_or(0.),
                ftol_rel: self.ftol_rel.unwrap_or(0.),
                initial_radius: 0.5,
                bounded_evaluations: true,
            },
        );
        (value, arr1(&point))
    }

    #[cfg(not(feature = "basin"))]
    pub fn minimize(&self) -> (f64, Array1<f64>) {
        let cstr_tol = self
            .cstr_tol
            .clone()
            .unwrap_or(Array1::zeros(self.cons.len()));
        match self.algo {
            Algorithm::Cobyla => {
                let xinit = self.xinit.clone().unwrap().to_vec();
                let bounds: Vec<_> = self
                    .bounds
                    .outer_iter()
                    .map(|row| (row[0], row[1]))
                    .collect();
                let cstrs: Vec<_> = self
                    .cons
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let cstr_tol = cstr_tol[i];
                        move |x: &[f64], u: &mut InfillObjData<f64>| {
                            let scale_cstr = u.scale_cstr.as_ref().expect("constraint scaling")[i];
                            -(*f)(x, None, u) + cstr_tol / scale_cstr
                        }
                    })
                    .collect();
                let res = cobyla::minimize(
                    |x: &[f64], u: &mut InfillObjData<f64>| (self.fun)(x, None, u),
                    &xinit,
                    &bounds,
                    &cstrs,
                    self.user_data.clone(),
                    self.max_eval,
                    RhoBeg::All(0.5),
                    Some(cobyla::StopTols {
                        ftol_rel: self.ftol_rel.unwrap_or(0.0),
                        ftol_abs: self.ftol_abs.unwrap_or(0.0),
                        ..cobyla::StopTols::default()
                    }),
                );
                match res {
                    Ok((_, x_opt, y_opt)) => (y_opt, arr1(&x_opt)),
                    Err((_, x_opt, _)) => (f64::INFINITY, arr1(&x_opt)),
                }
            }
            Algorithm::Slsqp => {
                let xinit = self.xinit.clone().unwrap().to_vec();
                let bounds: Vec<_> = self
                    .bounds
                    .outer_iter()
                    .map(|row| (row[0], row[1]))
                    .collect();
                let cstrs: Vec<_> = self
                    .cons
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let cstr_tol = cstr_tol[i];
                        move |x: &[f64], g: Option<&mut [f64]>, u: &mut InfillObjData<f64>| {
                            let scale_cstr = u.scale_cstr.as_ref().expect("constraint scaling")[i];
                            (*f)(x, g, u) - cstr_tol / scale_cstr
                        }
                    })
                    .collect();
                let res = slsqp::minimize(
                    self.fun,
                    &xinit,
                    &bounds,
                    &cstrs,
                    self.user_data.clone(),
                    self.max_eval,
                    Some(slsqp::StopTols {
                        ftol_rel: self.ftol_rel.unwrap_or(0.0),
                        ftol_abs: self.ftol_abs.unwrap_or(0.0),
                        ..slsqp::StopTols::default()
                    }),
                );
                match res {
                    Ok((_, x_opt, y_opt)) => (y_opt, arr1(&x_opt)),
                    Err((_, x_opt, _)) => (f64::INFINITY, arr1(&x_opt)),
                }
            }
        }
    }
}

#[cfg(all(test, feature = "basin"))]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn scaled_constraint_tolerance_keeps_its_physical_meaning() {
        let _ = env_logger::try_init();
        let data = InfillObjData {
            scale_cstr: Some(array![10.]),
            ..Default::default()
        };
        let objective = |x: &[f64], g: Option<&mut [f64]>, _: &mut InfillObjData<f64>| {
            if let Some(g) = g {
                g[0] = -1.;
            }
            -x[0]
        };
        let constraint = |x: &[f64], g: Option<&mut [f64]>, data: &mut InfillObjData<f64>| {
            let scale = data.scale_cstr.as_ref().unwrap()[0];
            if let Some(g) = g {
                g[0] = 1. / scale;
            }
            (x[0] - 1.) / scale
        };
        for algorithm in [Algorithm::Cobyla, Algorithm::Slsqp] {
            let mut optimizer = Optimizer::new(
                algorithm,
                &objective,
                &[&constraint],
                &data,
                &array![[0., 2.]],
            );
            optimizer.cstr_tol = Some(array![0.2]);
            let (value, point) = optimizer
                .xinit(&array![0.].view())
                .ftol_abs(1e-10)
                .ftol_rel(1e-10)
                .minimize();
            assert!((point[0] - 1.2).abs() < 1e-5, "{algorithm:?}: {point}");
            assert!((value + 1.2).abs() < 1e-5);
        }
    }
}

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
//! - [`Algorithm::Ipopt`] - Interior Point OPTimizer (gradient-based, requires the `pounce` feature)
//!
//! When the `nlopt` feature is enabled, [`Algorithm::Slsqp`] and [`Algorithm::Cobyla`] use the
//! NLopt library. Otherwise, pure-Rust implementations from the `slsqp` and `cobyla` crates are
//! used.
//!
//! ## Usage
//!
//! The optimizer is used internally by [`EgorSolver`](crate::EgorSolver) to optimize
//! the infill criterion at each iteration.

use crate::InfillObjData;
use ndarray::{Array1, Array2, ArrayView1, arr1};

#[cfg(not(feature = "nlopt"))]
use crate::types::UserFn;
#[cfg(not(feature = "nlopt"))]
pub(crate) trait OptFn<U>: UserFn<U> + Sync {}
#[cfg(not(feature = "nlopt"))]
impl<T, U> OptFn<U> for T where T: UserFn<U> + Sync {}

#[cfg(not(feature = "nlopt"))]
use cobyla::RhoBeg;

#[cfg(feature = "nlopt")]
pub(crate) trait OptFn<U>: nlopt::ObjFn<U> + Sync {}
#[cfg(feature = "nlopt")]
impl<T, U> OptFn<U> for T where T: nlopt::ObjFn<U> + Sync {}

#[derive(Copy, Clone, Debug)]
pub enum Algorithm {
    Cobyla,
    Slsqp,
    /// Interior-point method, delegated to the `pounce` crate.
    /// Only usable when the `pounce` feature is enabled.
    Ipopt,
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

    fn pounce_minimize(&self, cstr_tol: &Array1<f64>) -> (f64, Array1<f64>) {
        use pounce_rs::builder::Nlp;

        let m = self.cons.len();

        let lo: Vec<f64> = self.bounds.column(0).to_vec();
        let hi: Vec<f64> = self.bounds.column(1).to_vec();
        let x0: Vec<f64> = self
            .xinit
            .clone()
            .expect("xinit is required to run the pounce/IPOPT infill optimizer")
            .to_vec();
        debug_assert_eq!(x0.len(), self.bounds.nrows());

        // Same feasibility convention already used elsewhere in this file:
        // a constraint function `c` is feasible at x when `c(x) <= tol/scale`.
        // Ipopt/pounce express constraints as bounds `g_l <= g(x) <= g_u`, so
        // this maps directly onto `g(x) = c(x)`, `g_u = tol/scale`, `g_l = -inf`
        // -- no sign flip needed (unlike the Cobyla/Slsqp branches below, whose
        // underlying crates use their own, different constraint conventions).
        const IPOPT_INF: f64 = 2.0e19;
        let g_hi: Vec<f64> = (0..m)
            .map(|i| {
                let scale_cstr = self
                    .user_data
                    .scale_cstr
                    .as_ref()
                    .expect("constraint scaling")[i];
                cstr_tol[i] / scale_cstr
            })
            .collect();
        let g_lo = vec![-IPOPT_INF; m];

        // SAFETY: `pounce::builder::Nlp::new` requires `Problem: 'static`
        // because it drives the solve through an internal
        // `Rc<RefCell<dyn TNLP + 'static>>` adapter. `self.fun` / `self.cons`
        // only need to live for the duration of this function call: `solve()`
        // below runs the interior-point iterations to completion (or failure)
        // synchronously and returns a `Solution` that owns its data, so no
        // reference derived from `fun`/`cons` is read after `solve()` returns,
        // and nothing here spawns a thread or otherwise lets the adapter
        // outlive this stack frame. Extending the borrow's lifetime marker to
        // `'static` is therefore sound in this single call, even though the
        // real borrow is only valid for `'a`.
        let fun: &'static (dyn OptFn<InfillObjData<f64>> + Sync) =
            unsafe { std::mem::transmute(self.fun) };
        let cons: &'static [&'static (dyn OptFn<InfillObjData<f64>> + Sync)] =
            unsafe { std::mem::transmute(self.cons.as_slice()) };

        let problem = PounceInfillProblem {
            fun,
            cons,
            user_data: std::cell::RefCell::new(self.user_data.clone()),
        };

        // Divide max_eval by 10 to get a more reasonable number of iterations for Ipopt
        let max_iter = i32::try_from(self.max_eval / 10).unwrap_or(i32::MAX);
        // ftol_abs/ftol_rel (Cobyla/Slsqp objective-change stopping criteria)
        // do not map 1:1 onto Ipopt's KKT-error-based `tol`; use the tighter
        // of the two as a best-effort floor, falling back to Ipopt's own
        // default when neither was set.
        let tol = match (self.ftol_rel, self.ftol_abs) {
            (None, None) => 1e-8,
            (a, b) => a
                .into_iter()
                .chain(b)
                .fold(f64::INFINITY, f64::min)
                .max(1e-12),
        };

        let mut builder = Nlp::new(problem)
            .var_bounds(&lo, &hi)
            .x0(&x0)
            .option_str("hessian_approximation", "limited-memory")
            .option_int("print_level", 0)
            .option_str("sb", "yes")
            .option_num("tol", tol)
            .option_int("max_iter", max_iter);
        if m > 0 {
            builder = builder.constraint_bounds(&g_lo, &g_hi);
        }

        match builder.try_solve() {
            Ok(sol) if sol.success => (sol.objective, arr1(&sol.x)),
            Ok(sol) if !sol.x.is_empty() => (f64::INFINITY, arr1(&sol.x)),
            _ => (f64::INFINITY, arr1(&x0)),
        }
    }

    #[cfg(feature = "nlopt")]
    fn nlopt_minimize(&self, algo: nlopt::Algorithm, cstr_tol: Array1<f64>) -> (f64, Array1<f64>) {
        use nlopt::*;
        let mut optimizer = Nlopt::new(
            algo,
            self.bounds.nrows(),
            self.fun,
            Target::Minimize,
            self.user_data.clone(),
        );
        let lower = self.bounds.column(0).to_owned();
        optimizer
            .set_lower_bounds(lower.as_slice().unwrap())
            .unwrap();
        let upper = self.bounds.column(1).to_owned();
        optimizer
            .set_upper_bounds(upper.as_slice().unwrap())
            .unwrap();
        optimizer.set_maxeval(self.max_eval as u32).unwrap();
        optimizer
            .set_ftol_rel(self.ftol_rel.unwrap_or(0.0))
            .unwrap();
        optimizer
            .set_ftol_abs(self.ftol_abs.unwrap_or(0.0))
            .unwrap();
        self.cons.iter().enumerate().for_each(|(i, cstr)| {
            let scale_cstr = self
                .user_data
                .scale_cstr
                .as_ref()
                .expect("constraint scaling")[i];
            optimizer
                .add_inequality_constraint(cstr, self.user_data.clone(), cstr_tol[i] / scale_cstr)
                .unwrap();
        });

        let mut x_opt = self.xinit.clone().unwrap().to_vec();
        match optimizer.optimize(&mut x_opt) {
            Ok((_, opt)) => (opt, arr1(&x_opt)),
            Err((_err, _code)) => {
                // debug!("Nlopt Err: {:?} (y_opt={})", err, code);
                (f64::INFINITY, arr1(&x_opt))
            }
        }
    }

    pub fn minimize(&self) -> (f64, Array1<f64>) {
        let cstr_tol = self
            .cstr_tol
            .clone()
            .unwrap_or(Array1::zeros(self.cons.len()));
        match self.algo {
            Algorithm::Ipopt => self.pounce_minimize(&cstr_tol),
            Algorithm::Cobyla => {
                #[cfg(feature = "nlopt")]
                {
                    self.nlopt_minimize(nlopt::Algorithm::Cobyla, cstr_tol)
                }

                #[cfg(not(feature = "nlopt"))]
                {
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
                                let scale_cstr =
                                    u.scale_cstr.as_ref().expect("constraint scaling")[i];
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
            }
            Algorithm::Slsqp => {
                #[cfg(feature = "nlopt")]
                {
                    self.nlopt_minimize(nlopt::Algorithm::Slsqp, cstr_tol)
                }
                #[cfg(not(feature = "nlopt"))]
                {
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
                                let scale_cstr =
                                    u.scale_cstr.as_ref().expect("constraint scaling")[i];
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
}

/// Adapts the closure-based `fun`/`cons` used by [`Optimizer`] to the
/// [`pounce::builder::Problem`] trait expected by `pounce`'s [`pounce::builder::Nlp`]
/// builder, so [`Algorithm::Ipopt`] can reuse exactly the same objective and
/// constraint closures as the Cobyla/Slsqp branches above.
///
/// `user_data` is read-only from the point of view of a single `Nlp::solve()`
/// call (each per-call evaluation works off its own clone, exactly as the
/// Cobyla/Slsqp branches already do via `self.user_data.clone()`); it is
/// wrapped in a `RefCell` purely so `objective`/`gradient`/`constraints`/
/// `jacobian` can be implemented with the `&self` receiver `pounce::builder::Problem`
/// requires.
struct PounceInfillProblem<'a> {
    fun: &'a (dyn OptFn<InfillObjData<f64>> + Sync),
    cons: &'a [&'a (dyn OptFn<InfillObjData<f64>> + Sync)],
    user_data: std::cell::RefCell<InfillObjData<f64>>,
}

impl<'a> pounce_rs::builder::Problem for PounceInfillProblem<'a> {
    fn objective(&self, x: &[f64]) -> f64 {
        let mut u = self.user_data.borrow().clone();
        (self.fun)(x, None, &mut u)
    }

    fn n_constraints(&self) -> usize {
        self.cons.len()
    }

    fn constraints(&self, x: &[f64], out: &mut [f64]) {
        for (i, c) in self.cons.iter().enumerate() {
            let mut u = self.user_data.borrow().clone();
            out[i] = (*c)(x, None, &mut u);
        }
    }

    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> bool {
        let mut u = self.user_data.borrow().clone();
        (self.fun)(x, Some(grad), &mut u);
        true
    }

    fn jacobian(&self, x: &[f64], jac: &mut [f64]) -> bool {
        let n = x.len();
        for (i, c) in self.cons.iter().enumerate() {
            let mut u = self.user_data.borrow().clone();
            let row = &mut jac[i * n..(i + 1) * n];
            (*c)(x, Some(row), &mut u);
        }
        true
    }
}

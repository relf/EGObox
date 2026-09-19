//! Shared numerical adapter for GP fitting and EGO acquisition optimization.
//!
//! This module is an implementation detail shared by the EGObox crates.
#![allow(missing_docs)]

use basin::{
    Cobyla, CobylaState, ConstraintJacobian, CostFunction, DenseMatrix, FoldedConstraints,
    Gradient, NonlinearConstraints, Slsqp, SlsqpState, State, TerminationReason,
};
use std::borrow::Cow;
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    Cobyla,
    Slsqp,
}

pub struct Settings<'a> {
    pub xinit: &'a [f64],
    pub bounds: &'a [(f64, f64)],
    pub n_constraints: usize,
    pub max_evals: usize,
    pub ftol_abs: f64,
    pub ftol_rel: f64,
    pub initial_radius: f64,
    pub bounded_evaluations: bool,
}

#[derive(Debug)]
enum EvaluationError {
    Budget,
    NonFinite,
}

type Objective<'a> = dyn Fn(&[f64], Option<&mut [f64]>) -> f64 + 'a;
type Constraint<'a> = dyn Fn(usize, &[f64], Option<&mut [f64]>) -> f64 + 'a;

struct Problem<'a> {
    algorithm: Algorithm,
    objective: &'a Objective<'a>,
    constraint: &'a Constraint<'a>,
    settings: &'a Settings<'a>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    evaluations: Cell<usize>,
    // COBYLA's published incumbent can remain infeasible at a radius stop, and
    // a callback budget can expire before publication. Retain feasible trials
    // independently so either ordinary stop can return usable work.
    best: RefCell<Option<(Vec<f64>, f64)>>,
}

impl Problem<'_> {
    fn bounded<'a>(&self, x: &'a [f64]) -> Cow<'a, [f64]> {
        if x.iter()
            .zip(self.settings.bounds)
            .all(|(x, (lo, hi))| *x >= *lo && *x <= *hi)
        {
            Cow::Borrowed(x)
        } else {
            Cow::Owned(
                x.iter()
                    .zip(self.settings.bounds)
                    .map(|(x, (lo, hi))| x.clamp(*lo, *hi))
                    .collect(),
            )
        }
    }

    fn callback_point<'a>(&self, x: &'a [f64]) -> Cow<'a, [f64]> {
        if self.settings.bounded_evaluations {
            self.bounded(x)
        } else {
            Cow::Borrowed(x)
        }
    }

    fn feasible(&self, x: &[f64]) -> bool {
        if !x
            .iter()
            .zip(self.settings.bounds)
            .all(|(x, (lo, hi))| *x >= *lo - 1e-8 && *x <= *hi + 1e-8)
        {
            return false;
        }
        let x = self.bounded(x);
        (0..self.settings.n_constraints).all(|i| {
            let value = (self.constraint)(i, &x, None);
            value.is_finite() && value <= 1e-8
        })
    }

    fn evaluate(&self, x: &[f64], gradient: Option<&mut [f64]>) -> Result<f64, EvaluationError> {
        if self.evaluations.get() >= self.settings.max_evals {
            return Err(EvaluationError::Budget);
        }
        self.evaluations.set(self.evaluations.get() + 1);
        // Folded COBYLA bounds constrain the solution, not every trial point.
        // Project callbacks with a restricted domain. Smooth GP likelihoods
        // can retain their extension outside the box for interpolation.
        let x = self.callback_point(x);
        let value = (self.objective)(&x, gradient);
        if !value.is_finite() {
            return Err(EvaluationError::NonFinite);
        }
        if self.feasible(&x)
            && self
                .best
                .borrow()
                .as_ref()
                .is_none_or(|(_, best)| value < *best)
        {
            *self.best.borrow_mut() = Some((x.to_vec(), value));
        }
        Ok(value)
    }
}

impl CostFunction for &Problem<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = EvaluationError;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        self.evaluate(x, None)
    }
}

impl Gradient for &Problem<'_> {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        let mut gradient = vec![0.; x.len()];
        self.evaluate(x, Some(&mut gradient))?;
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(EvaluationError::NonFinite);
        }
        for ((g, x), (lo, hi)) in gradient.iter_mut().zip(x).zip(self.settings.bounds) {
            if self.settings.bounded_evaluations && (*x < *lo || *x > *hi) {
                *g = 0.;
            }
        }
        Ok(gradient)
    }
}

impl NonlinearConstraints for &Problem<'_> {
    type Matrix = DenseMatrix;
    fn num_nonlinear_constraints(&self) -> usize {
        self.settings.n_constraints
    }
    fn nonlinear_constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        let x = self.callback_point(x);
        let values: Vec<_> = (0..self.settings.n_constraints)
            .map(|i| (self.constraint)(i, &x, None))
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(EvaluationError::NonFinite);
        }
        Ok(values)
    }
    fn lower(&self) -> Option<&Vec<f64>> {
        Some(&self.lower)
    }
    fn upper(&self) -> Option<&Vec<f64>> {
        Some(&self.upper)
    }
}

impl ConstraintJacobian for &Problem<'_> {
    fn constraint_jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Self::Error> {
        let bounded = self.callback_point(x);
        let mut values = vec![0.; self.settings.n_constraints * x.len()];
        for (i, row) in values.chunks_mut(x.len()).enumerate() {
            (self.constraint)(i, &bounded, Some(row));
            for ((g, x), (lo, hi)) in row.iter_mut().zip(x).zip(self.settings.bounds) {
                if self.settings.bounded_evaluations && (*x < *lo || *x > *hi) {
                    *g = 0.;
                }
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(EvaluationError::NonFinite);
        }
        Ok(DenseMatrix::from_row_slice(
            self.settings.n_constraints,
            x.len(),
            &values,
        ))
    }
}

fn cost_stop<S: State<Param = Vec<f64>, Float = f64>>(
    problem: &Problem<'_>,
    previous: &mut Option<f64>,
    state: &S,
) -> Option<TerminationReason> {
    let current = state.cost();
    if !current.is_finite() || !problem.feasible(state.param()) {
        // Objective progress cannot be compared across infeasible stages.
        *previous = None;
        return None;
    }
    let stop = previous.is_some_and(|cost| {
        // Like the default COBYLA backend, require strict improvement so a
        // repeated incumbent does not prevent refinement at a smaller radius.
        (matches!(problem.algorithm, Algorithm::Slsqp) || current < cost)
            && ((problem.settings.ftol_abs > 0.
                && (cost - current).abs() < problem.settings.ftol_abs)
                || (problem.settings.ftol_rel > 0.
                    && (cost - current).abs()
                        < problem.settings.ftol_rel * (cost.abs() + current.abs()) / 2.))
    });
    *previous = Some(current);
    stop.then_some(TerminationReason::CostTolerance)
}

pub fn minimize(
    algorithm: Algorithm,
    objective: &Objective<'_>,
    constraint: &Constraint<'_>,
    settings: Settings<'_>,
) -> (f64, Vec<f64>) {
    let problem = Problem {
        algorithm,
        objective,
        constraint,
        lower: settings.bounds.iter().map(|b| b.0).collect(),
        upper: settings.bounds.iter().map(|b| b.1).collect(),
        settings: &settings,
        evaluations: Cell::new(0),
        best: RefCell::new(None),
    };
    let xinit = problem.bounded(settings.xinit).into_owned();
    let outcome = match algorithm {
        Algorithm::Cobyla => {
            let solver = Cobyla::new()
                .with_initial_radius(settings.initial_radius)
                .with_final_radius(1e-6);
            let mut wrapped = basin::Problem::new(FoldedConstraints::new(&problem));
            let mut solver = solver;
            let mut control = basin::RunControl::new().max_iter(u64::MAX);
            let state = CobylaState::new(xinit);
            let mut radius = None;
            run(
                &problem,
                &mut wrapped,
                &mut solver,
                state,
                &mut control,
                |state: &CobylaState<Vec<f64>>| {
                    // Small improvements can precede exploration of a flat
                    // region. Compare completed radius stages, not individual
                    // steps; initialization only starts the first stage.
                    radius
                        .replace(state.rho())
                        .is_some_and(|previous| state.rho() < previous)
                },
            )
        }
        Algorithm::Slsqp => {
            let mut wrapped = basin::Problem::new(&problem);
            // EGObox supplies objective-change tolerances. An additional native
            // absolute accuracy floor would stop small acquisition values early.
            let mut solver = Slsqp::new().with_absolute_accuracy_tolerance(0.0);
            let mut control = basin::RunControl::new().max_iter(u64::MAX);
            let state = SlsqpState::new(xinit);
            run(
                &problem,
                &mut wrapped,
                &mut solver,
                state,
                &mut control,
                |_| true,
            )
        }
    };
    log::debug!(
        "Basin {algorithm:?}: {outcome:?}, callbacks={}",
        problem.evaluations.get()
    );
    if let Ok((x, value, reason)) = &outcome
        && !reason.is_failure()
        && value.is_finite()
        && problem.feasible(x)
    {
        return (*value, problem.bounded(x).into_owned());
    }
    let stopped_normally = match outcome {
        Ok((_, _, reason)) => !reason.is_failure(),
        Err(EvaluationError::Budget) => true,
        Err(EvaluationError::NonFinite) => false,
    };
    if stopped_normally {
        problem
            .best
            .into_inner()
            .map_or((f64::INFINITY, settings.xinit.to_vec()), |(x, f)| (f, x))
    } else {
        (f64::INFINITY, settings.xinit.to_vec())
    }
}

// The solver chooses when objective progress is meaningful to check. The
// callback enforces the hard budget independently of convergence checks.
fn run<P, S, So, Check>(
    problem: &Problem<'_>,
    wrapped: &mut basin::Problem<P>,
    solver: &mut So,
    state: S,
    control: &mut basin::RunControl<S>,
    checkpoint: Check,
) -> Result<(Vec<f64>, f64, TerminationReason), EvaluationError>
where
    S: State<Param = Vec<f64>, Float = f64> + basin::CountsMirror,
    So: basin::Solver<P, S, Error = EvaluationError>,
    Check: FnMut(&S) -> bool,
{
    // An observer cannot stop a borrowed run. A solver adapter lets the native
    // executor own counting and iteration while retaining local stop history.
    struct WithTolerance<'a, 'b, So, Check> {
        solver: &'a mut So,
        problem: &'a Problem<'b>,
        previous: Option<f64>,
        checkpoint: Check,
    }
    impl<P, S, So, Check> basin::Solver<P, S> for WithTolerance<'_, '_, So, Check>
    where
        S: State<Param = Vec<f64>, Float = f64>,
        So: basin::Solver<P, S, Error = EvaluationError>,
        Check: FnMut(&S) -> bool,
    {
        type Error = EvaluationError;
        fn init(&mut self, p: &mut basin::Problem<P>, s: S) -> Result<S, Self::Error> {
            self.solver.init(p, s)
        }
        fn next_iter(
            &mut self,
            p: &mut basin::Problem<P>,
            s: S,
        ) -> Result<(S, Option<TerminationReason>), Self::Error> {
            self.solver.next_iter(p, s)
        }
        fn check_convergence(&mut self, p: &basin::Problem<P>, s: &S) -> Option<TerminationReason> {
            self.solver.check_convergence(p, s).or_else(|| {
                if (self.checkpoint)(s) {
                    cost_stop(self.problem, &mut self.previous, s)
                } else {
                    None
                }
            })
        }
    }
    let mut adapter = WithTolerance {
        solver,
        problem,
        previous: None,
        checkpoint,
    };
    let result =
        basin::core::executor::run_loop_with_control(wrapped, state, &mut adapter, control)?;
    Ok((
        result.state.param().clone(),
        result.state.cost(),
        result.reason,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings<'a>(xinit: &'a [f64], bounds: &'a [(f64, f64)]) -> Settings<'a> {
        Settings {
            xinit,
            bounds,
            n_constraints: 0,
            max_evals: 200,
            ftol_abs: 1e-10,
            ftol_rel: 1e-10,
            initial_radius: 0.5,
            bounded_evaluations: true,
        }
    }

    #[test]
    fn bounded_quadratic_for_both_solvers() {
        for algorithm in [Algorithm::Cobyla, Algorithm::Slsqp] {
            let (value, x) = minimize(
                algorithm,
                &|x, g| {
                    if let Some(g) = g {
                        g[0] = 2. * (x[0] - 3.);
                    }
                    (x[0] - 3.).powi(2)
                },
                &|_, _, _| unreachable!(),
                settings(&[0.], &[(-1., 2.)]),
            );
            assert!((value - 1.).abs() < 1e-5, "{value} at {x:?}");
            assert!((x[0] - 2.).abs() < 1e-5);
        }
    }

    #[test]
    fn cobyla_explores_a_flat_start_before_applying_objective_tolerances() {
        for (offset, ftol_abs, ftol_rel) in [(0., 1e-4, 1e-4), (100., 0., 1e-4)] {
            let mut config = settings(&[0.], &[(-5., 5.)]);
            config.max_evals = 600;
            config.ftol_abs = ftol_abs;
            config.ftol_rel = ftol_rel;
            let (value, x) = minimize(
                Algorithm::Cobyla,
                &|x, _| offset - (-(x[0] - 4.1).powi(2)).exp(),
                &|_, _, _| unreachable!(),
                config,
            );
            assert!(value < offset - 0.99, "{value} at {x:?}, offset={offset}");
            assert!((x[0] - 4.1).abs() < 0.1, "{x:?}, offset={offset}");
        }
    }

    #[test]
    fn cobyla_objective_tolerance_limits_refinement_after_exploration() {
        let evaluations = [1e-4, 0.].map(|tolerance| {
            let calls = Cell::new(0);
            let mut config = settings(&[0.], &[(-5., 5.)]);
            config.max_evals = 600;
            config.ftol_abs = tolerance;
            config.ftol_rel = tolerance;
            let (value, x) = minimize(
                Algorithm::Cobyla,
                &|x, _| {
                    calls.set(calls.get() + 1);
                    -(-(x[0] - 4.1234567).powi(2)).exp()
                },
                &|_, _, _| unreachable!(),
                config,
            );
            assert!((value + 1.).abs() < 1e-4, "{value} at {x:?}");
            calls.get()
        });
        assert!(
            evaluations[0] < evaluations[1],
            "Objective tolerances must still limit refinement: {evaluations:?}"
        );
    }

    #[test]
    fn slsqp_respects_relative_tolerance_for_small_objectives() {
        let mut config = settings(&[0.], &[(-2., 2.)]);
        config.ftol_abs = 0.;
        let (_, x) = minimize(
            Algorithm::Slsqp,
            &|x, g| {
                if let Some(g) = g {
                    g[0] = 2e-8 * (x[0] - 1.);
                }
                1e-8 * (x[0] - 1.).powi(2)
            },
            &|_, _, _| unreachable!(),
            config,
        );
        assert!((x[0] - 1.).abs() < 1e-4, "{x:?}");
    }

    #[test]
    fn initial_points_respect_adjusted_box_bounds() {
        for algorithm in [Algorithm::Cobyla, Algorithm::Slsqp] {
            let (value, x) = minimize(
                algorithm,
                &|x, g| {
                    assert!((-1.0..=1.0).contains(&x[0]));
                    if let Some(g) = g {
                        g[0] = 2. * (x[0] - 0.5);
                    }
                    (x[0] - 0.5).powi(2)
                },
                &|_, _, _| unreachable!(),
                settings(&[1000.], &[(-1., 1.)]),
            );
            assert!(value < 1e-8, "{algorithm:?}: {value} at {x:?}");
        }
    }

    #[test]
    fn negative_constraints_and_analytic_jacobian() {
        for algorithm in [Algorithm::Cobyla, Algorithm::Slsqp] {
            let mut config = settings(&[0., 0.], &[(-2., 2.), (-2., 2.)]);
            config.n_constraints = 1;
            let (value, x) = minimize(
                algorithm,
                &|x, g| {
                    if let Some(g) = g {
                        g.copy_from_slice(&[-1., -1.]);
                    }
                    -x[0] - x[1]
                },
                &|_, x, g| {
                    if let Some(g) = g {
                        g.copy_from_slice(&[2. * x[0], 2. * x[1]]);
                    }
                    x[0] * x[0] + x[1] * x[1] - 1.
                },
                config,
            );
            assert!((value + 2f64.sqrt()).abs() < 2e-3, "{value} at {x:?}");
            assert!(x[0] * x[0] + x[1] * x[1] <= 1. + 1e-8);
        }
    }

    #[test]
    fn hard_budget_includes_initialization_and_gradient_callbacks() {
        for algorithm in [Algorithm::Cobyla, Algorithm::Slsqp] {
            for budget in [0, 1, 2, 3] {
                let calls = Cell::new(0);
                let mut config = settings(&[0., 0.], &[(-2., 2.), (-2., 2.)]);
                config.max_evals = budget;
                let (value, _) = minimize(
                    algorithm,
                    &|x, g| {
                        calls.set(calls.get() + 1);
                        if let Some(g) = g {
                            g.copy_from_slice(&[2. * (x[0] - 1.), 2. * (x[1] - 1.)]);
                        }
                        (x[0] - 1.).powi(2) + (x[1] - 1.).powi(2)
                    },
                    &|_, _, _| unreachable!(),
                    config,
                );
                assert!(calls.get() <= budget);
                assert_eq!(value.is_finite(), budget > 0);
            }
        }
    }

    #[test]
    fn nonfinite_objective_is_not_a_successful_candidate() {
        for algorithm in [Algorithm::Cobyla, Algorithm::Slsqp] {
            let (value, _) = minimize(
                algorithm,
                &|_, _| f64::NAN,
                &|_, _, _| unreachable!(),
                settings(&[0.], &[(-1., 1.)]),
            );
            assert_eq!(value, f64::INFINITY);
        }
    }

    #[test]
    fn nonfinite_constraint_is_not_a_feasible_budget_candidate() {
        for algorithm in [Algorithm::Cobyla, Algorithm::Slsqp] {
            let mut config = settings(&[0.], &[(-1., 1.)]);
            config.max_evals = 1;
            config.n_constraints = 1;
            let (value, _) = minimize(algorithm, &|_, _| 1., &|_, _, _| f64::NEG_INFINITY, config);
            assert_eq!(value, f64::INFINITY);
        }
    }
}

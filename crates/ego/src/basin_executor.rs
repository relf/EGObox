//! Basin execution with the existing public Argmin state and solver interfaces.

use crate::egor::OptimizationObserver;
use crate::{
    CHECKPOINT_FILE, Constraints, CstrFn, CstrSpec, EgorSolver, EgorState, HotStartMode, ObjFn,
    ProblemFunc, SurrogateBuilder,
};
use argmin::core::{State as _, TerminationReason as LegacyReason, TerminationStatus};
use basin::{CountsMirror, EvalCounts, Executor, State, TerminationReason};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::cell::RefCell;
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};
use web_time::Instant;

impl<O: ObjFn, C: CstrFn> basin::CostFunction for ProblemFunc<O, C> {
    type Param = Array2<f64>;
    type Output = Array2<f64>;
    type Error = argmin::core::Error;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        argmin::core::CostFunction::cost(self, x)
    }
}

// The bridge keeps the algorithm shared and charges batch evaluations to Basin.
// Constraints are copied before borrowing the problem mutably for evaluations.
struct ProblemBridge<'a, P, C> {
    problem: RefCell<&'a mut basin::Problem<P>>,
    constraints: Vec<C>,
    specs: Option<Vec<CstrSpec>>,
}

impl<P, C: CstrFn> Constraints<C> for ProblemBridge<'_, P, C> {
    fn constraints(&self) -> &[C] {
        &self.constraints
    }
    fn constraint_specs(&self) -> Option<&[CstrSpec]> {
        self.specs.as_deref()
    }
}

impl<P, C> argmin::core::CostFunction for ProblemBridge<'_, P, C>
where
    P: basin::CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = argmin::core::Error>,
{
    type Param = Array2<f64>;
    type Output = Array2<f64>;
    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        self.problem.borrow_mut().cost(x)
    }
}

fn bridge<P: Constraints<C>, C: CstrFn>(
    problem: &mut basin::Problem<P>,
) -> argmin::core::Problem<ProblemBridge<'_, P, C>> {
    let constraints = problem.inner().constraints().to_vec();
    let specs = problem.inner().constraint_specs().map(<[_]>::to_vec);
    argmin::core::Problem::new(ProblemBridge {
        problem: RefCell::new(problem),
        constraints,
        specs,
    })
}

#[derive(Serialize, Deserialize)]
pub(crate) struct BasinState {
    inner: EgorState<f64>,
    selected: Option<usize>,
    best_evals: u64,
    #[serde(skip)]
    started: Option<Instant>,
}

impl State for BasinState {
    type Param = Array1<f64>;
    type Float = f64;
    fn iter(&self) -> u64 {
        self.inner.iter
    }
    fn increment_iter(&mut self) {
        // Argmin updates the best point before incrementing its iteration index.
        self.inner.update();
        self.inner.increment_iter();
    }
    fn cost_evals(&self) -> u64 {
        *self.inner.counts.get("cost_count").unwrap_or(&0)
    }
    fn param(&self) -> &Self::Param {
        self.inner
            .get_param()
            .or_else(|| self.inner.get_best_param())
            .expect("initialized parameter")
    }
    fn cost(&self) -> f64 {
        if self.inner.cost.is_some() {
            self.inner.get_cost()
        } else {
            self.inner.get_best_cost()
        }
    }
    fn best_param(&self) -> &Self::Param {
        self.inner.get_best_param().expect("initialized best point")
    }
    fn best_cost(&self) -> f64 {
        self.inner.get_best_cost()
    }
    fn best_iter(&self) -> u64 {
        self.inner.last_best_iter
    }
    fn best_cost_evals(&self) -> u64 {
        self.best_evals
    }
    fn update_best(&mut self) {
        if self.selected != self.inner.surrogate.best_index {
            self.selected = self.inner.surrogate.best_index;
            self.best_evals = self.cost_evals();
        }
    }
    fn reset_best(&mut self) {
        self.selected = None;
        self.best_evals = 0;
    }
}

impl CountsMirror for BasinState {
    fn mirror(&mut self, counts: &EvalCounts) {
        self.inner
            .counts
            .insert("cost_count".into(), counts.cost_evals);
        self.inner.time = self.started.map(|start| start.elapsed());
    }
}

impl<P, SB, C> basin::Solver<P, BasinState> for EgorSolver<SB, C>
where
    P: basin::CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = argmin::core::Error>
        + Constraints<C>,
    C: CstrFn,
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
{
    type Error = argmin::core::Error;
    fn init(
        &mut self,
        problem: &mut basin::Problem<P>,
        mut state: BasinState,
    ) -> Result<BasinState, Self::Error> {
        state.inner = argmin::core::Solver::init(self, &mut bridge(problem), state.inner)?.0;
        state.inner.update();
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut basin::Problem<P>,
        mut state: BasinState,
    ) -> Result<(BasinState, Option<TerminationReason>), Self::Error> {
        state.inner = argmin::core::Solver::next_iter(self, &mut bridge(problem), state.inner)?.0;
        // EGO's clean stops consume an iteration under its existing contract.
        // Publish that iteration, then let the execution hook read its status.
        Ok((state, None))
    }
}

static INTERRUPTS: AtomicU64 = AtomicU64::new(0);
static SIGNAL_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();

fn install_signal_handler() -> Result<(), argmin::core::Error> {
    SIGNAL_HANDLER
        .get_or_init(|| {
            match ctrlc::set_handler(|| {
                INTERRUPTS.fetch_add(1, Ordering::Relaxed);
            }) {
                Ok(()) | Err(ctrlc::Error::MultipleHandlers) => Ok(()),
                Err(error) => Err(error.to_string()),
            }
        })
        .clone()
        .map_err(argmin::core::Error::msg)
}

struct Observer {
    inner: OptimizationObserver,
    error: std::rc::Rc<RefCell<Option<argmin::core::Error>>>,
}

impl basin::Observe<BasinState> for Observer {
    fn observe_iter(&mut self, state: &BasinState) {
        let mut snapshot = state.inner.clone();
        snapshot.iter = snapshot.iter.saturating_sub(1);
        if let Err(error) = argmin::core::observers::Observe::observe_iter(
            &mut self.inner,
            &snapshot,
            &argmin::core::KV::new(),
        ) {
            *self.error.borrow_mut() = Some(error);
        }
    }
}

pub(crate) fn run<O, C, SB>(
    problem: ProblemFunc<O, C>,
    solver: EgorSolver<SB, C>,
) -> crate::Result<EgorState<f64>>
where
    O: ObjFn,
    C: CstrFn,
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
{
    use basin::{ExactCheckpoint, ExactCheckpointWriter, ObserverMode, read_exact_checkpoint};
    install_signal_handler()?;
    let interrupt = INTERRUPTS.load(Ordering::Relaxed);
    let config = solver.config.clone();
    let path = std::path::Path::new(config.outdir.as_deref().unwrap_or(".checkpoints"))
        .join(CHECKPOINT_FILE);
    let checkpoint = config.hot_start != HotStartMode::Disabled;
    let started = Instant::now();
    let (exec, start_iter) = if checkpoint && path.exists() {
        let saved: ExactCheckpoint<EgorSolver<SB, C>, BasinState> = read_exact_checkpoint(&path)?;
        let (solver, mut state, counts) = saved.into_parts();
        if let HotStartMode::ExtendedIters(n) = config.hot_start {
            state.inner.max_iters = state.inner.max_iters.checked_add(n).ok_or_else(|| {
                crate::EgoError::InvalidConfigError("extended iteration budget overflows".into())
            })?;
        }
        // A saved clean stop describes the previous invocation, not this one.
        state.inner.termination_status = TerminationStatus::NotTerminated;
        state.started = Some(started);
        let iter = state.inner.iter;
        (
            Executor::resume_from_checkpoint(
                problem,
                ExactCheckpoint::from_parts(solver, state, counts),
            ),
            iter,
        )
    } else {
        let state = BasinState {
            inner: EgorState::new(),
            selected: None,
            best_evals: 0,
            started: Some(started),
        };
        (Executor::new(problem, solver, state), 0)
    };
    let observer_error = std::rc::Rc::new(RefCell::new(None));
    let errors = observer_error.clone();
    let timeout = config.timeout.map(web_time::Duration::from_secs_f64);
    let mut exec = exec.max_iter(u64::MAX).stop_when(move |state| {
        if INTERRUPTS.load(Ordering::Relaxed) != interrupt {
            return Some(TerminationReason::Cancelled);
        }
        if errors.borrow().is_some() {
            return Some(TerminationReason::UserRequested);
        }
        // Match the legacy timeout boundary, including initialization in the clock.
        if state.inner.iter > start_iter && timeout.is_some_and(|limit| started.elapsed() > limit) {
            return Some(TerminationReason::MaxTime);
        }
        if state.inner.terminated() {
            return Some(TerminationReason::UserRequested);
        }
        if state.inner.iter >= state.inner.max_iters {
            return Some(TerminationReason::MaxIter);
        }
        if state.inner.target_cost > f64::NEG_INFINITY
            && state.best_cost() <= state.inner.target_cost
        {
            return Some(TerminationReason::TargetCost);
        }
        None
    });
    let writer_status = if checkpoint {
        std::fs::create_dir_all(path.parent().expect("checkpoint directory"))?;
        let writer = ExactCheckpointWriter::new(path);
        let status = writer.status();
        exec = exec.checkpoint_with(writer, ObserverMode::Always);
        Some(status)
    } else {
        None
    };
    if let Some(outdir) = config.outdir {
        exec = exec.observe_with(
            Observer {
                inner: OptimizationObserver::new(outdir),
                error: observer_error.clone(),
            },
            ObserverMode::Always,
        );
    }
    let result = exec.run()?;
    if let Some(error) = observer_error.borrow_mut().take() {
        return Err(error.into());
    }
    if let Some(error) = writer_status.and_then(|status| status.last_error()) {
        return Err(std::io::Error::other(format!("checkpoint write failed: {error:?}")).into());
    }
    let mut state = result.state.inner;
    state.time = Some(started.elapsed());
    let reason = match result.reason {
        TerminationReason::MaxIter => LegacyReason::MaxItersReached,
        TerminationReason::TargetCost => LegacyReason::TargetCostReached,
        TerminationReason::MaxTime => LegacyReason::Timeout,
        TerminationReason::Cancelled => LegacyReason::Interrupt,
        TerminationReason::UserRequested if state.terminated() => return Ok(state),
        reason => {
            return Err(argmin::core::Error::msg(format!(
                "unexpected Basin executor stop: {reason:?}"
            ))
            .into());
        }
    };
    state.termination_status = TerminationStatus::Terminated(reason);
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn initialized_state_exposes_the_best_doe_point() {
        let mut inner = EgorState::new();
        inner.best_param = Some(array![2.]);
        inner.best_cost = Some(array![-1.]);
        let state = BasinState {
            inner,
            selected: Some(0),
            best_evals: 1,
            started: None,
        };
        assert_eq!(state.param(), &array![2.]);
        assert_eq!(state.cost(), -1.);
    }
}

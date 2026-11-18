//! Execution of the [`EgorSolver`] with the basin executor.

use crate::outputs::{
    DOE_FILE, FAILED_POINTS_FILE, HISTORY_FILE, STATE_FILE_PREFIX, save_npy, save_text,
};
use crate::{
    CHECKPOINT_FILE, Constraints, CstrFn, EgoError, EgorSolver, EgorState, HotStartMode, ObjFn,
    ProblemFunc, SurrogateBuilder, TerminationReason as EgorReason, TerminationStatus,
};
use basin::{ApplicationStop, CountsMirror, EvalCounts, Executor, State, Termination};
use egobox_gp::Cancellation;
use log::info;
use ndarray::{Array1, Array2, Axis, concatenate};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::cell::RefCell;
use std::path::PathBuf;
use web_time::Instant;

/// Basin state wrapping [`EgorState`] with the best point bookkeeping
/// required by the basin executor.
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
        // Best point is updated before incrementing the iteration index.
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
    P: basin::CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = EgoError>
        + Constraints<C>,
    C: CstrFn,
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
{
    type Error = EgoError;
    fn init(
        &mut self,
        problem: &mut basin::Problem<P>,
        mut state: BasinState,
    ) -> Result<BasinState, Self::Error> {
        state.inner = self.init_state(problem, state.inner)?;
        state.inner.update();
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut basin::Problem<P>,
        mut state: BasinState,
    ) -> Result<basin::SolverStep<BasinState>, Self::Error> {
        let previous = state.inner.clone();
        match self.next_state(problem, std::mem::replace(&mut state.inner, previous)) {
            Ok(inner) => {
                state.inner = inner;
                // EGO's clean stops consume an iteration under its existing contract.
                // Publish that iteration, then let the execution hook read its status.
                Ok(basin::SolverStep::completed(state))
            }
            // Interrupted objective evaluation: the unfinished iteration is dropped, the
            // previous one is published and checkpointed so that a hot start does it again
            Err(EgoError::Interrupted) => Ok(basin::SolverStep::stopped(
                state,
                Termination::Application(ApplicationStop::new(STOP_INTERRUPT)),
            )),
            Err(err) => Err(err),
        }
    }
}

// Application stop codes raised by the execution hook and mapped back to
// Egor termination reasons once the basin executor returns.
const STOP_INTERRUPT: &str = "egor.interrupt";
const STOP_OBSERVER_ERROR: &str = "egor.observer_error";
const STOP_TIMEOUT: &str = "egor.timeout";
const STOP_SOLVER: &str = "egor.solver";
const STOP_MAX_ITERS: &str = "egor.max_iters";
const STOP_TARGET_COST: &str = "egor.target_cost";

// The optimization observer collects best costs ans params
// during the optimization execution allowing to get optimization history
// saved as a numpy array for further analysis
// Note: the observer is activated only when outdir is specified
#[derive(Default)]
pub(crate) struct OptimizationObserver {
    pub dir: PathBuf,
    pub best_params: Option<Array2<f64>>,
    pub best_costs: Option<Array2<f64>>,
    pub state_recording: bool,
}

impl OptimizationObserver {
    pub(crate) fn new(dir: String, state_recording: bool) -> Self {
        Self {
            dir: PathBuf::from(dir),
            best_params: None,
            best_costs: None,
            state_recording,
        }
    }
}

impl OptimizationObserver {
    /// Save optimization history, current doe and optionally state in output directory
    pub(crate) fn observe_iter(&mut self, state: &EgorState<f64>) -> crate::Result<()> {
        if let Some((xdata, ydata, cdata)) = &state.surrogate.data {
            let doe = concatenate![Axis(1), xdata.view(), ydata.view(), cdata.view()];
            if !self.dir.exists() {
                std::fs::create_dir_all(&self.dir)?
            }

            let filepath = self.dir.join(DOE_FILE);
            info!(">>> Save doe shape {:?} in {:?}", doe.shape(), filepath);
            save_npy(filepath, &doe)?;

            if let Some(failed_points) = state.surrogate.x_fail.as_ref() {
                let filepath = self.dir.join(FAILED_POINTS_FILE);
                info!(
                    ">>> Save failed points {:?} in {:?}",
                    failed_points.shape(),
                    filepath
                );
                save_npy(filepath, failed_points)?;
            }

            if self.best_params.is_none() {
                // Have to initialize best params and full best costs
                let mut state = state.clone();
                state.update();
                let bp = state.get_best_param().unwrap().to_owned();
                self.best_params = Some(bp.insert_axis(Axis(0)));
                let bc = state.get_full_best_cost().unwrap().to_owned();
                self.best_costs = Some(bc.insert_axis(Axis(0)));
            } else {
                let bp = state.get_best_param().unwrap().to_owned();
                let bp = bp.insert_axis(Axis(0));
                self.best_params =
                    Some(concatenate![Axis(0), self.best_params.take().unwrap(), bp]);
                let bc = state.get_full_best_cost().unwrap().to_owned();
                let bc = bc.insert_axis(Axis(0));
                self.best_costs = Some(concatenate![Axis(0), self.best_costs.take().unwrap(), bc]);
            };
            let hist = concatenate![
                Axis(1),
                self.best_costs.clone().unwrap(),
                self.best_params.clone().unwrap(),
            ];

            let filepath = std::path::Path::new(&self.dir).join(HISTORY_FILE);
            info!(">>> Save history {:?} in {:?}", hist.shape(), filepath);
            save_npy(filepath, &hist)?;

            // Save EgorState if EGOR_USE_STATE_RECORDING equals "WITH_ITER_STATE"
            if self.state_recording {
                let state_filename = format!("{}_{:04}.json", STATE_FILE_PREFIX, state.iter);
                let state_filepath = std::path::Path::new(&self.dir).join(state_filename);
                // Do not embed the surrogate GP models in the per-iteration state
                // dump: they are by far the largest part of the state (GP training
                // data and Cholesky factorizations) and are saved separately as GP
                // files by the run recorder. Hot start checkpoints still serialize
                // them for exact continuation.
                let json = serde_json::to_value(state)
                    .map(|mut value| {
                        if let Some(surrogate) = value.get_mut("surrogate")
                            && let Some(map) = surrogate.as_object_mut()
                        {
                            map.remove("models");
                        }
                        value
                    })
                    .and_then(|value| serde_json::to_string_pretty(&value));
                match json {
                    Ok(json) => {
                        if let Err(e) = save_text(&state_filepath, &json) {
                            log::warn!("Failed to save EgorState to {:?}: {}", state_filepath, e);
                        } else {
                            log::debug!(">>> Saved EgorState to {:?}", state_filepath);
                        }
                    }
                    Err(_) => log::warn!("Failed to serialize EgorState"),
                }
            }
        }
        Ok(())
    }
}

/// A function telling whether the optimization has to be interrupted,
/// used instead of signal handling (see [crate::Egor::interrupt_check])
pub(crate) type InterruptCheck = std::sync::Arc<dyn Fn() -> bool + Send + Sync>;

struct Observer {
    inner: OptimizationObserver,
    error: std::rc::Rc<RefCell<Option<EgoError>>>,
}

impl basin::Observe<BasinState> for Observer {
    fn observe_iter(&mut self, state: &BasinState) {
        let mut snapshot = state.inner.clone();
        snapshot.iter = snapshot.iter.saturating_sub(1);
        if let Err(error) = self.inner.observe_iter(&snapshot) {
            *self.error.borrow_mut() = Some(error);
        }
    }
}

/// Run the given solver on the given problem with the basin executor
/// handling hot start, timeout, interruption and history observation.
pub(crate) fn run<O, C, SB>(
    problem: ProblemFunc<O, C>,
    mut solver: EgorSolver<SB, C>,
    interrupt_check: Option<InterruptCheck>,
) -> crate::Result<EgorState<f64>>
where
    O: ObjFn,
    C: CstrFn,
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
{
    use basin::{ExactCheckpoint, ExactCheckpointWriter, ObserverMode, read_exact_checkpoint};
    // Signal handlers installed for the duration of the run, unless a check is given
    let _signals = interrupt_check.is_none().then(crate::signals::install);
    let signaled = interrupt_check.is_none();
    let check: InterruptCheck = interrupt_check.unwrap_or_else(|| {
        let interrupts = crate::signals::interrupts();
        std::sync::Arc::new(move || crate::signals::interrupts() != interrupts)
    });
    // The check also cuts long computations short (training, infill optimization)
    let cancellation = {
        let check = check.clone();
        Cancellation::new(move || check())
    };
    solver.cancellation = cancellation.clone();
    let interrupted = move || {
        let interrupted = check();
        if interrupted && signaled {
            log::warn!(
                "Optimization interrupted by {}",
                crate::signals::last_signal()
            );
        }
        interrupted
    };
    let config = solver.config.clone();
    let path = std::path::Path::new(config.outdir.as_deref().unwrap_or(".checkpoints"))
        .join(CHECKPOINT_FILE);
    let checkpoint = config.hot_start != HotStartMode::Disabled;
    let started = Instant::now();
    let (exec, start_iter) = if checkpoint && path.exists() {
        let saved: ExactCheckpoint<EgorSolver<SB, C>, BasinState> = read_exact_checkpoint(&path)?;
        let (mut solver, mut state, counts) = saved.into_parts();
        // Not saved in checkpoints, taken from the configuration of this run
        solver.config.feasibility_first = config.feasibility_first;
        solver.cancellation = cancellation;
        if let HotStartMode::ExtendedIters(n) = config.hot_start {
            state.inner.max_iters = state.inner.max_iters.checked_add(n).ok_or_else(|| {
                EgoError::InvalidConfigError("extended iteration budget overflows".into())
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
        let stop = if interrupted() {
            STOP_INTERRUPT
        } else if errors.borrow().is_some() {
            STOP_OBSERVER_ERROR
        } else if state.inner.iter > start_iter
            && timeout.is_some_and(|limit| started.elapsed() > limit)
        {
            // Timeout boundary includes initialization in the clock.
            STOP_TIMEOUT
        } else if state.inner.terminated() {
            STOP_SOLVER
        } else if state.inner.iter >= state.inner.max_iters {
            STOP_MAX_ITERS
        } else if state.inner.target_cost > f64::NEG_INFINITY
            && state.best_cost() <= state.inner.target_cost
        {
            STOP_TARGET_COST
        } else {
            return None;
        };
        Some(ApplicationStop::new(stop))
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
                inner: OptimizationObserver::new(outdir, config.runtime_flags.use_state_recording),
                error: observer_error.clone(),
            },
            ObserverMode::Always,
        );
    }
    let result = exec.run()?;
    if let Some(error) = observer_error.borrow_mut().take() {
        return Err(error);
    }
    if let Some(error) = writer_status.and_then(|status| status.last_error()) {
        return Err(std::io::Error::other(format!("checkpoint write failed: {error:?}")).into());
    }
    let mut state = result.state.inner;
    state.time = Some(started.elapsed());
    let reason = match &result.report.termination {
        Termination::Application(stop) => match stop.code.as_str() {
            STOP_MAX_ITERS => EgorReason::MaxItersReached,
            STOP_TARGET_COST => EgorReason::TargetCostReached,
            STOP_TIMEOUT => EgorReason::Timeout,
            STOP_INTERRUPT => EgorReason::Interrupt,
            STOP_SOLVER if state.terminated() => return Ok(state),
            _ => {
                return Err(EgoError::InternalError(format!(
                    "unexpected basin executor stop: {:?}",
                    result.report
                )));
            }
        },
        termination => {
            return Err(EgoError::InternalError(format!(
                "unexpected basin executor stop: {termination:?}"
            )));
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

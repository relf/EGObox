//! Termination status and reasons of an EGO optimization run.

use serde::{Deserialize, Serialize};

/// Reasons for the termination of the optimization.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerminationReason {
    /// Reached maximum number of iterations
    MaxItersReached,
    /// Reached target cost function value
    TargetCostReached,
    /// Algorithm manually interrupted with Ctrl+C
    Interrupt,
    /// Algorithm cannot add new points: considered as converged
    SolverConverged,
    /// Timeout reached
    Timeout,
    /// Solver exit with given reason
    SolverExit(String),
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminationReason::MaxItersReached => write!(f, "Maximum number of iterations reached"),
            TerminationReason::TargetCostReached => write!(f, "Target cost value reached"),
            TerminationReason::Interrupt => write!(f, "Interrupt"),
            TerminationReason::SolverConverged => write!(f, "Solver converged"),
            TerminationReason::Timeout => write!(f, "Timeout reached"),
            TerminationReason::SolverExit(reason) => write!(f, "Solver exit: {reason}"),
        }
    }
}

/// Status of the optimization execution.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerminationStatus {
    /// Optimization is still running
    #[default]
    NotTerminated,
    /// Optimization terminated with the given reason
    Terminated(TerminationReason),
}

impl TerminationStatus {
    /// Returns `true` if the optimization is terminated.
    pub fn terminated(&self) -> bool {
        matches!(self, TerminationStatus::Terminated(_))
    }
}

impl std::fmt::Display for TerminationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminationStatus::NotTerminated => write!(f, "Running"),
            TerminationStatus::Terminated(reason) => write!(f, "{reason}"),
        }
    }
}

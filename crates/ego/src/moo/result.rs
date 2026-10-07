//! Multi-objective optimization result

use crate::EgorState;
use linfa::Float;
use ndarray::Array2;

/// Multi-objective optimization result returned by [`crate::Egor::run_pareto`]
///
/// `y_pareto` and `y_doe` hold the objectives and the raw constraint values as returned by
/// the objective function, i.e. `n_obj + n_cstr` columns, like [`crate::OptimResult`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ParetoResult<F: Float> {
    /// Pareto set: x values of the points of the (constrained) Pareto front,
    /// in evaluation order. When no point is feasible, the point with the smallest
    /// constraint violation.
    pub x_pareto: Array2<F>,
    /// Pareto front: y values of the points of the Pareto set
    pub y_pareto: Array2<F>,
    /// History of successive x values
    pub x_doe: Array2<F>,
    /// History of successive y values (e.g f(x_doe))
    pub y_doe: Array2<F>,
    /// EgorSolver final state
    pub state: EgorState<F>,
}

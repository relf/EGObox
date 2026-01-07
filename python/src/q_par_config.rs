use crate::types::*;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// Configuration for parallel (qEI) infill criterion evaluation.
///
/// The q-parallel configuration allows for evaluating multiple points
/// in parallel during each EGO iteration, which can significantly speed up
/// optimization when function evaluations can be performed in parallel.
///
/// Parameters
/// ----------
/// q_points : int
///     Number of points to evaluate in parallel at each iteration.
///     When set to 1, standard sequential EGO is used.
/// q_infill_strategy : QInfillStrategy
///     Strategy for selecting multiple points:
///     * KB (Kriging Believer): Uses the GP mean prediction as a pseudo-observation
///     * KBLB (Kriging Believer Lower Bound): Uses GP mean - std as pseudo-observation
///     * KBUB (Kriging Believer Upper Bound): Uses GP mean + std as pseudo-observation
///     * CLMIN (Constant Liar Minimum): Uses the current best value as pseudo-observation
/// q_optmod : int
///     Interval between two hyperparameter optimizations when computing q points.
///     For example, with q_optmod=2, hyperparameters are optimized every 2 points.
#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct QParConfig {
    /// Number of points to evaluate in parallel
    #[pyo3(get, set)]
    pub q_points: usize,

    /// Strategy for selecting multiple points in parallel
    #[pyo3(get, set)]
    pub q_infill_strategy: QInfillStrategy,

    /// Interval between hyperparameter optimizations
    #[pyo3(get, set)]
    pub q_optmod: usize,
}

impl Default for QParConfig {
    fn default() -> Self {
        QParConfig::new(1, QInfillStrategy::Kb, 1)
    }
}

#[pymethods]
impl QParConfig {
    /// Create a new parallel evaluation configuration.
    ///
    /// Parameters
    /// ----------
    /// q_points : int, optional
    ///     Number of points to evaluate in parallel (default: 1)
    /// q_infill_strategy : QInfillStrategy, optional
    ///     Strategy for parallel point selection (default: QInfillStrategy.KB)
    /// q_optmod : int, optional
    ///     Interval between hyperparameter optimizations (default: 1)
    ///
    /// Returns
    /// -------
    /// QParConfig
    ///     A new parallel evaluation configuration object
    #[new]
    #[pyo3(signature = (
        q_points=QParConfig::default().q_points,
        q_infill_strategy=QParConfig::default().q_infill_strategy,
        q_optmod=QParConfig::default().q_optmod,
    ))]
    pub fn new(q_points: usize, q_infill_strategy: QInfillStrategy, q_optmod: usize) -> Self {
        QParConfig {
            q_points,
            q_infill_strategy,
            q_optmod,
        }
    }
}

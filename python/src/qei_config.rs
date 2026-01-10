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
/// q_batch : int
///     Number of points to evaluate in parallel at each iteration.
///     When set to 1, standard sequential EGO is used.
/// q_ei_strategy : QEiStrategy
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
pub(crate) struct QEiConfig {
    /// Number of points to evaluate in parallel
    #[pyo3(get, set)]
    pub batch: usize,

    /// Strategy for selecting multiple points in parallel
    #[pyo3(get, set)]
    pub strategy: QEiStrategy,

    /// Interval between hyperparameter optimizations
    #[pyo3(get, set)]
    pub optmod: usize,
}

impl Default for QEiConfig {
    fn default() -> Self {
        QEiConfig::new(1, QEiStrategy::Kb, 1)
    }
}

#[pymethods]
impl QEiConfig {
    /// Create a new parallel evaluation configuration.
    ///
    /// Parameters
    /// ----------
    /// batch : int, optional
    ///     Number of points to evaluate in parallel (default: 1)
    /// strategy : QEiStrategy, optional
    ///     Strategy for parallel point selection (default: QEiStrategy.KB)
    /// optmod : int, optional
    ///     Interval between hyperparameter optimizations (default: 1)
    ///
    /// Returns
    /// -------
    /// QEiConfig
    ///     A new parallel evaluation configuration object
    #[new]
    #[pyo3(signature = (
        batch=QEiConfig::default().batch,
        strategy=QEiConfig::default().strategy,
        optmod=QEiConfig::default().optmod,
    ))]
    pub fn new(batch: usize, strategy: QEiStrategy, optmod: usize) -> Self {
        QEiConfig {
            batch,
            strategy,
            optmod,
        }
    }
}

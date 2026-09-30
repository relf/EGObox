use crate::types::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Configuration for parallel (qEI) infill criterion evaluation.
///
/// The q-parallel configuration allows for evaluating multiple points
/// in parallel during each EGO iteration, which can significantly speed up
/// optimization when function evaluations can be performed in parallel.
///
/// Parameters
/// ----------
///
/// batch : int
///     Number of points to evaluate in parallel at each iteration.
///     When set to 1, standard sequential EGO is used.
///
/// strategy : QEiStrategy
///     Strategy for selecting multiple points:
///     * KB (Kriging Believer): Uses the GP mean prediction as a pseudo-observation
///     * KBLB (Kriging Believer Lower Bound): Uses GP mean - std as pseudo-observation
///     * KBUB (Kriging Believer Upper Bound): Uses GP mean + std as pseudo-observation
///     * CLMIN (Constant Liar Minimum): Uses the current best value as pseudo-observation
///
/// optmod : int
///     Optimization modulo: interval between two GP hyperparameter optimizations
///     when computing the q points of a batch. For example, with optmod=2,
///     hyperparameters are optimized every 2 points, otherwise they are kept as is.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
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

impl<'a, 'py> FromPyObject<'a, 'py> for QEiConfig {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(cfg) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(cfg.clone());
        }

        let dict = obj.cast::<PyDict>()?;
        let mut cfg = QEiConfig::default();

        for key_any in dict.keys().iter() {
            let key = key_any.extract::<String>()?;
            match key.as_str() {
                "batch" => cfg.batch = dict.get_item("batch")?.unwrap().extract()?,
                "strategy" => cfg.strategy = dict.get_item("strategy")?.unwrap().extract()?,
                "optmod" => cfg.optmod = dict.get_item("optmod")?.unwrap().extract()?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown qei_config key '{key}'"
                    )));
                }
            }
        }

        Ok(cfg)
    }
}

impl Default for QEiConfig {
    fn default() -> Self {
        QEiConfig::new(1, QEiStrategy::Kb, 1)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl QEiConfig {
    /// Create a new parallel evaluation configuration.
    ///
    /// Parameters
    /// ----------
    ///
    /// batch : int, optional
    ///     Number of points to evaluate in parallel (default: 1)
    ///
    /// strategy : QEiStrategy, optional
    ///     Strategy for parallel point selection (default: QEiStrategy.KB)
    ///
    /// optmod : int, optional
    ///     Interval between hyperparameter optimizations (default: 1)
    ///
    /// Returns
    /// -------
    ///
    /// QEiConfig
    ///     A new parallel evaluation configuration object
    ///
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

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "QEiConfig",
            &[
                ("batch", self.batch.into_pyobject(py)?.into_any()),
                ("strategy", self.strategy.into_pyobject(py)?.into_any()),
                ("optmod", self.optmod.into_pyobject(py)?.into_any()),
            ],
        )
    }
}

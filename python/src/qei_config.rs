use crate::deprecation::{resolve_renamed, resolve_renamed_key, warn_deprecated};
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
/// optim_every : int
///     Interval between two GP hyperparameter optimizations when computing the q points of a batch.
///     For example, with optim_every=2, hyperparameters are optimized every 2 points,
///     otherwise they are kept as is.
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
    pub optim_every: usize,
}

/// Deprecated qEI configuration names (old, new)
const QEI_CONFIG_RENAMED: [(&str, &str); 1] = [("optmod", "optim_every")];

impl<'a, 'py> FromPyObject<'a, 'py> for QEiConfig {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(cfg) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(cfg.clone());
        }

        let dict = obj.cast::<PyDict>()?;
        let mut cfg = QEiConfig::default();

        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            match resolve_renamed_key(&dict, &key, &QEI_CONFIG_RENAMED)? {
                "batch" => cfg.batch = value.extract()?,
                "strategy" => cfg.strategy = value.extract()?,
                "optim_every" => cfg.optim_every = value.extract()?,
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
        QEiConfig {
            batch: 1,
            strategy: QEiStrategy::Kb,
            optim_every: 1,
        }
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
    /// optim_every : int, optional
    ///     Interval between hyperparameter optimizations (default: 1)
    ///
    /// Deprecated
    /// ----------
    /// optmod : int, optional
    ///     Deprecated since 0.38.0, use `optim_every` instead.
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
        optim_every=None,
        *,
        optmod=None,
    ))]
    pub fn new(
        py: Python,
        batch: usize,
        strategy: QEiStrategy,
        optim_every: Option<usize>,
        optmod: Option<usize>,
    ) -> PyResult<Self> {
        let optim_every = resolve_renamed(
            py,
            "optmod",
            optmod,
            "optim_every",
            optim_every,
            QEiConfig::default().optim_every,
        )?;
        Ok(QEiConfig {
            batch,
            strategy,
            optim_every,
        })
    }

    /// Deprecated since 0.38.0, use `optim_every` instead.
    #[getter(optmod)]
    fn get_optmod(&self, py: Python) -> PyResult<usize> {
        warn_deprecated(py, "QEiConfig.optmod", "QEiConfig.optim_every")?;
        Ok(self.optim_every)
    }

    #[setter(optmod)]
    fn set_optmod(&mut self, value: usize) -> PyResult<()> {
        // no `py` argument as pyo3-stub-gen does not support it on setters
        Python::attach(|py| warn_deprecated(py, "QEiConfig.optmod", "QEiConfig.optim_every"))?;
        self.optim_every = value;
        Ok(())
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "QEiConfig",
            &[
                ("batch", self.batch.into_pyobject(py)?.into_any()),
                ("strategy", self.strategy.into_pyobject(py)?.into_any()),
                (
                    "optim_every",
                    self.optim_every.into_pyobject(py)?.into_any(),
                ),
            ],
        )
    }
}

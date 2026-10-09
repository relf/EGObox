use crate::types::repr_kwargs;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

/// MooStrategy specifies how several objectives are optimized by Belfegor
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MooStrategy {
    /// ParEGO (Knowles 2006): at each iteration, objectives normalized with their observed bounds
    /// are aggregated with an augmented Tchebycheff function using a weight vector randomly drawn
    /// from a simplex lattice, then the mono-objective EGO machinery is applied to that
    /// scalarized objective (single surrogate)
    Parego = 1,
    /// Expected Improvement Matrix (Zhan et al. 2017): one surrogate per objective, the expected
    /// improvements of each objective over each point of the Pareto front are aggregated into
    /// an infill criterion (see MooConfig eim_aggregation)
    Eim = 2,
    /// Expected Hypervolume Improvement (Emmerich et al. 2006): one surrogate per objective,
    /// expected improvement of the hypervolume dominated by the Pareto front (at most 8 objectives)
    Ehvi = 3,
    /// Batch Expected Hypervolume Improvement (Daulton et al. 2020) for batches of points
    /// (see QEiConfig batch): EHVI for the first point of a batch, then expected hypervolume
    /// improvement of each following point given the points already selected, under the joint
    /// posterior of the surrogates (single-cluster surrogates, batches of at most 4 points)
    Qehvi = 4,
}

impl<'a, 'py> FromPyObject<'a, 'py> for MooStrategy {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Parego),
            Ok(2) => Ok(Self::Eim),
            Ok(3) => Ok(Self::Ehvi),
            Ok(4) => Ok(Self::Qehvi),
            Ok(v) => Err(PyValueError::new_err(format!(
                "moo strategy integer value must be in [1, 4], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "moo strategy must be a MooStrategy enum or an integer in [1, 4]",
            )),
        }
    }
}

impl From<MooStrategy> for egobox_ego::MooStrategy {
    fn from(value: MooStrategy) -> Self {
        match value {
            MooStrategy::Parego => egobox_ego::MooStrategy::ParEgo,
            MooStrategy::Eim => egobox_ego::MooStrategy::Eim,
            MooStrategy::Ehvi => egobox_ego::MooStrategy::Ehvi,
            MooStrategy::Qehvi => egobox_ego::MooStrategy::QEhvi,
        }
    }
}

/// EimAggregation specifies how the expected improvement matrix is aggregated
/// by the MooStrategy.EIM strategy
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EimAggregation {
    /// Minimum over front points of the Euclidean norm of the expected improvements
    Euclidean = 1,
    /// Minimum over front points of the maximum expected improvement
    Maximin = 2,
    /// Minimum over front points of the expected hypervolume improvement of the
    /// hyper-rectangle dominated by the point improved by the expected improvements
    Hypervolume = 3,
}

impl<'a, 'py> FromPyObject<'a, 'py> for EimAggregation {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Euclidean),
            Ok(2) => Ok(Self::Maximin),
            Ok(3) => Ok(Self::Hypervolume),
            Ok(v) => Err(PyValueError::new_err(format!(
                "eim aggregation integer value must be in [1, 3], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "eim aggregation must be an EimAggregation enum or an integer in [1, 3]",
            )),
        }
    }
}

impl From<EimAggregation> for egobox_ego::EimAggregation {
    fn from(value: EimAggregation) -> Self {
        match value {
            EimAggregation::Euclidean => egobox_ego::EimAggregation::Euclidean,
            EimAggregation::Maximin => egobox_ego::EimAggregation::Maximin,
            EimAggregation::Hypervolume => egobox_ego::EimAggregation::Hypervolume,
        }
    }
}

/// Default ParEGO augmented Tchebycheff coefficient
const DEFAULT_RHO: f64 = 0.05;

/// Multi-objective optimization configuration used by Belfegor.
///
/// Parameters
/// ----------
///
/// strategy : MooStrategy, optional
///     Strategy used to optimize the objectives. When None (default), MooStrategy.EHVI is used
///     for 2 or 3 objectives and MooStrategy.PAREGO beyond.
///
/// eim_aggregation : EimAggregation
///     Aggregation of the expected improvement matrix used by MooStrategy.EIM
///     (default: EimAggregation.EUCLIDEAN).
///
/// hv_stop : tuple of (float, int), optional
///     Hypervolume-based stop (tol, n_iters): the optimization stops when the hypervolume of the
///     (feasible) Pareto front improves by less than the relative tolerance tol over the last
///     n_iters iterations, reported as ExitStatus.SOLVER_CONVERGED. When None (default),
///     the optimization runs till max_iters.
///
/// rho : float
///     Coefficient of the augmented Tchebycheff function used by MooStrategy.PAREGO
///     (default: 0.05).
///
/// n_divisions : int, optional
///     Number of divisions of the weight simplex lattice used by MooStrategy.PAREGO.
///     When None (default), 10 for 2 objectives, 4 for 3 objectives and 3 beyond.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Clone, Debug)]
pub(crate) struct MooConfig {
    /// Strategy used to optimize the objectives (None for the default depending on the number of objectives)
    #[pyo3(get, set)]
    pub strategy: Option<MooStrategy>,

    /// Aggregation of the expected improvement matrix used by MooStrategy.EIM
    #[pyo3(get, set)]
    pub eim_aggregation: EimAggregation,

    /// Hypervolume-based stop (relative tolerance, number of iterations)
    #[pyo3(get, set)]
    pub hv_stop: Option<(f64, usize)>,

    /// ParEGO augmented Tchebycheff coefficient
    #[pyo3(get, set)]
    pub rho: f64,

    /// ParEGO number of divisions of the weight simplex lattice (None for the default)
    #[pyo3(get, set)]
    pub n_divisions: Option<usize>,
}

impl Default for MooConfig {
    fn default() -> Self {
        MooConfig {
            strategy: None,
            eim_aggregation: EimAggregation::Euclidean,
            hv_stop: None,
            rho: DEFAULT_RHO,
            n_divisions: None,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for MooConfig {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(cfg) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(cfg.clone());
        }

        let dict = obj.cast::<PyDict>()?;
        let mut cfg = MooConfig::default();

        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "strategy" => cfg.strategy = value.extract()?,
                "eim_aggregation" => cfg.eim_aggregation = value.extract()?,
                "hv_stop" => cfg.hv_stop = value.extract()?,
                "rho" => cfg.rho = value.extract()?,
                "n_divisions" => cfg.n_divisions = value.extract()?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown moo_config key '{key}'"
                    )));
                }
            }
        }

        Ok(cfg)
    }
}

impl From<&MooConfig> for egobox_ego::MooConfig {
    fn from(value: &MooConfig) -> Self {
        let mut config = egobox_ego::MooConfig::default()
            .eim_aggregation(value.eim_aggregation.into())
            .rho(value.rho);
        if let Some(strategy) = value.strategy {
            config = config.strategy(strategy.into());
        }
        if let Some((tol, n_iters)) = value.hv_stop {
            config = config.hv_stop(tol, n_iters);
        }
        if let Some(n_divisions) = value.n_divisions {
            config = config.n_divisions(n_divisions);
        }
        config
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl MooConfig {
    /// Create a new multi-objective optimization configuration.
    ///
    /// Parameters
    /// ----------
    ///
    /// strategy : MooStrategy, optional
    ///     Strategy used to optimize the objectives (default: None, i.e. EHVI for 2 or 3
    ///     objectives, PAREGO beyond)
    ///
    /// eim_aggregation : EimAggregation, optional
    ///     Aggregation used by MooStrategy.EIM (default: EimAggregation.EUCLIDEAN)
    ///
    /// hv_stop : tuple of (float, int), optional
    ///     Hypervolume-based stop (tol, n_iters) (default: None, no stop)
    ///
    /// rho : float, optional
    ///     ParEGO augmented Tchebycheff coefficient (default: 0.05)
    ///
    /// n_divisions : int, optional
    ///     ParEGO number of divisions of the weight simplex lattice (default: None)
    ///
    /// Returns
    /// -------
    ///
    /// MooConfig
    ///     A new multi-objective optimization configuration object
    ///
    #[new]
    #[pyo3(signature = (
        strategy=None,
        eim_aggregation=MooConfig::default().eim_aggregation,
        hv_stop=None,
        rho=DEFAULT_RHO,
        n_divisions=None,
    ))]
    pub fn new(
        strategy: Option<MooStrategy>,
        eim_aggregation: EimAggregation,
        hv_stop: Option<(f64, usize)>,
        rho: f64,
        n_divisions: Option<usize>,
    ) -> Self {
        MooConfig {
            strategy,
            eim_aggregation,
            hv_stop,
            rho,
            n_divisions,
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "MooConfig",
            &[
                ("strategy", self.strategy.into_pyobject(py)?.into_any()),
                (
                    "eim_aggregation",
                    self.eim_aggregation.into_pyobject(py)?.into_any(),
                ),
                ("hv_stop", self.hv_stop.into_pyobject(py)?.into_any()),
                ("rho", self.rho.into_pyobject(py)?.into_any()),
                (
                    "n_divisions",
                    self.n_divisions.into_pyobject(py)?.into_any(),
                ),
            ],
        )
    }
}

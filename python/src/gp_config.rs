use crate::types::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// GP configuration used by `Egor` and `GpMix`
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Clone, Debug)]
pub(crate) struct GpConfig {
    /// (RegressionSpec flags, an int in [1, 7])
    ///   Specification of regression models used in mixture.
    ///   Can be RegressionSpec.CONSTANT (1), RegressionSpec.LINEAR (2), RegressionSpec.QUADRATIC (4) or
    ///   any bit-wise union of these values (e.g. RegressionSpec.CONSTANT | RegressionSpec.LINEAR)
    #[pyo3(get, set)]
    pub regr_spec: u8,

    /// (CorrelationSpec flags, an int in [1, 15])
    ///   Specification of correlation models used in mixture.
    ///   Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
    ///   CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
    ///   any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
    #[pyo3(get, set)]
    pub corr_spec: u8,

    /// (0 < int < nx where nx is the dimension of inputs x)
    ///   Number of components to be used when PLS projection is used (a.k.a KPLS method).
    ///   This is used to address high-dimensional problems typically when nx > 9.
    #[pyo3(get, set)]
    pub kpls_dim: Option<usize>,

    /// (int)
    ///   Number of clusters used by the mixture of surrogate experts (default is 1).
    ///   When set to 0, the number of cluster is determined automatically and refreshed every
    ///   10-points addition (should say 'tentative addition' because addition may fail for some points
    ///   but it is counted anyway).
    ///   When set to negative number -n, the number of clusters is determined automatically in [1, n]
    ///   this is used to limit the number of trials hence the execution time.
    #[pyo3(get, set)]
    pub n_clusters: isize,

    /// (Recombination.Smooth or Recombination.Hard (default))
    ///   Specify how the various experts predictions are recombined
    ///   * Smooth: prediction is a combination of experts prediction wrt their responsabilities,
    ///   the heaviside factor which controls steepness of the change between experts regions is optimized
    ///   to get best mixture quality.
    ///   * Hard: prediction is taken from the expert with highest responsability
    ///   resulting in a model with discontinuities.
    #[pyo3(get, set)]
    pub recombination: Recombination,

    /// ([nx] where nx is the dimension of inputs x)
    ///   Initial guess for GP theta hyperparameters.
    ///   When None the default is 1e-1 for all components
    #[pyo3(get, set)]
    pub theta_init: Option<Vec<f64>>,

    /// ([[lower_1, upper_1], ..., [lower_nx, upper_nx]] where nx is the dimension of inputs x)
    ///   Space search when optimizing theta GP hyperparameters
    ///   When None the default is [1e-2, 1e1] for all components.
    ///   Note: `Egor` may adapt these bounds automatically for high-dimensional inputs.
    #[pyo3(get, set)]
    pub theta_bounds: Option<Vec<Vec<f64>>>,

    /// (int >= 0)
    ///   Number of internal GP hyperpameters optimization restart (multistart)
    ///   When zero, optimization is disabled and theta init value is used as is.
    #[pyo3(get, set)]
    pub n_start: usize,

    /// (int >= 0)
    ///   Max number of likelihood evaluations during GP hyperparameters optimization
    #[pyo3(get, set)]
    pub max_eval: usize,
}

impl<'a, 'py> FromPyObject<'a, 'py> for GpConfig {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(cfg) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(cfg.clone());
        }

        let dict = obj.cast::<PyDict>()?;
        let mut cfg = GpConfig::default();

        for key_any in dict.keys().iter() {
            let key = key_any.extract::<String>()?;
            match key.as_str() {
                "regr_spec" => cfg.regr_spec = dict.get_item("regr_spec")?.unwrap().extract()?,
                "corr_spec" => cfg.corr_spec = dict.get_item("corr_spec")?.unwrap().extract()?,
                "kpls_dim" => cfg.kpls_dim = dict.get_item("kpls_dim")?.unwrap().extract()?,
                "n_clusters" => cfg.n_clusters = dict.get_item("n_clusters")?.unwrap().extract()?,
                "recombination" => {
                    cfg.recombination = dict.get_item("recombination")?.unwrap().extract()?
                }
                "theta_init" => cfg.theta_init = dict.get_item("theta_init")?.unwrap().extract()?,
                "theta_bounds" => {
                    cfg.theta_bounds = dict.get_item("theta_bounds")?.unwrap().extract()?
                }
                "n_start" => cfg.n_start = dict.get_item("n_start")?.unwrap().extract()?,
                "max_eval" => cfg.max_eval = dict.get_item("max_eval")?.unwrap().extract()?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown gp_config key '{key}'"
                    )));
                }
            }
        }

        Ok(cfg)
    }
}

/// Check correlation spec flags, an int in [1, 15]
pub(crate) fn validate_corr_spec(corr_spec: u8) -> PyResult<()> {
    if egobox_moe::CorrelationSpec::from_bits(corr_spec).is_none_or(|spec| spec.is_empty()) {
        return Err(PyValueError::new_err(format!(
            "corr_spec should be a union of CorrelationSpec flags (an int in [1, 15]), got {corr_spec}"
        )));
    }
    Ok(())
}

/// Check theta hyperparameters initial values and bounds
pub(crate) fn validate_theta(
    theta_init: Option<&Vec<f64>>,
    theta_bounds: Option<&Vec<Vec<f64>>>,
) -> PyResult<()> {
    if theta_init.is_some_and(|init| init.is_empty()) {
        return Err(PyValueError::new_err("theta_init should not be empty"));
    }
    if let Some(bounds) = theta_bounds {
        if bounds.is_empty() {
            return Err(PyValueError::new_err("theta_bounds should not be empty"));
        }
        for (i, b) in bounds.iter().enumerate() {
            if !matches!(b[..], [lower, upper] if lower < upper) {
                return Err(PyValueError::new_err(format!(
                    "theta_bounds[{i}] should be [lower, upper] with lower < upper, got {b:?}"
                )));
            }
        }
    }
    Ok(())
}

impl GpConfig {
    /// Check the configuration consistency
    pub(crate) fn validate(&self) -> PyResult<()> {
        if egobox_moe::RegressionSpec::from_bits(self.regr_spec).is_none_or(|spec| spec.is_empty())
        {
            return Err(PyValueError::new_err(format!(
                "regr_spec should be a union of RegressionSpec flags (an int in [1, 7]), got {}",
                self.regr_spec
            )));
        }
        validate_corr_spec(self.corr_spec)?;
        validate_theta(self.theta_init.as_ref(), self.theta_bounds.as_ref())
    }
}

impl Default for GpConfig {
    fn default() -> Self {
        GpConfig::new(
            RegressionSpec::CONSTANT,
            CorrelationSpec::SQUARED_EXPONENTIAL,
            None,
            1,
            Recombination::Hard,
            None,
            None,
            egobox_ego::EGO_GP_OPTIM_N_START,
            egobox_ego::EGO_GP_OPTIM_MAX_EVAL,
        )
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl GpConfig {
    /// Create a new GP configuration.
    ///
    /// Parameters
    /// ----------
    /// regr_spec : int, optional
    ///     RegressionSpec flags, an int in [1, 7] (default: RegressionSpec.CONSTANT)
    /// corr_spec : int, optional
    ///     CorrelationSpec flags, an int in [1, 15] (default: CorrelationSpec.SQUARED_EXPONENTIAL)
    /// kpls_dim : int, optional
    ///     Number of PLS components, 0 < kpls_dim < nx (default: None, no PLS reduction)
    /// n_clusters : int, optional
    ///     Number of clusters of the mixture of experts; 0 or -n for automatic selection (default: 1)
    /// recombination : Recombination, optional
    ///     How the experts predictions are recombined (default: Recombination.HARD)
    /// theta_init : list of float, optional
    ///     Initial guess for GP theta hyperparameters (default: None, 1e-1 for all components)
    /// theta_bounds : list of [float, float], optional
    ///     Search space of GP theta hyperparameters (default: None, [1e-2, 1e1] for all components)
    /// n_start : int, optional
    ///     Number of GP hyperparameters optimization restarts, 0 to disable optimization (default: 10)
    /// max_eval : int, optional
    ///     Max number of likelihood evaluations during GP hyperparameters optimization (default: 50)
    ///
    /// Returns
    /// -------
    /// GpConfig
    ///     A new GP configuration object
    #[new]
    #[pyo3(signature = (
        regr_spec=GpConfig::default().regr_spec,
        corr_spec=GpConfig::default().corr_spec,
        kpls_dim=GpConfig::default().kpls_dim,
        n_clusters=GpConfig::default().n_clusters,
        recombination=GpConfig::default().recombination,
        theta_init=GpConfig::default().theta_init,
        theta_bounds=GpConfig::default().theta_bounds,
        n_start=GpConfig::default().n_start,
        max_eval=GpConfig::default().max_eval,
))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        regr_spec: u8,
        corr_spec: u8,
        kpls_dim: Option<usize>,
        n_clusters: isize,
        recombination: Recombination,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        n_start: usize,
        max_eval: usize,
    ) -> Self {
        GpConfig {
            regr_spec,
            corr_spec,
            kpls_dim,
            n_clusters,
            recombination,
            theta_init,
            theta_bounds,
            n_start,
            max_eval,
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "GpConfig",
            &[
                ("regr_spec", self.regr_spec.into_pyobject(py)?.into_any()),
                ("corr_spec", self.corr_spec.into_pyobject(py)?.into_any()),
                ("kpls_dim", self.kpls_dim.into_pyobject(py)?.into_any()),
                ("n_clusters", self.n_clusters.into_pyobject(py)?.into_any()),
                (
                    "recombination",
                    self.recombination.clone().into_pyobject(py)?.into_any(),
                ),
                (
                    "theta_init",
                    self.theta_init.clone().into_pyobject(py)?.into_any(),
                ),
                (
                    "theta_bounds",
                    self.theta_bounds.clone().into_pyobject(py)?.into_any(),
                ),
                ("n_start", self.n_start.into_pyobject(py)?.into_any()),
                ("max_eval", self.max_eval.into_pyobject(py)?.into_any()),
            ],
        )
    }
}

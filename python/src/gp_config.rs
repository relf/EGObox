use crate::deprecation::{resolve_renamed, resolve_renamed_key, warn_deprecated};
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

    /// (Recombination.SMOOTH or Recombination.HARD (default))
    ///   Specify how the various experts predictions are recombined
    ///   * SMOOTH: prediction is a combination of experts prediction wrt their responsibilities,
    ///   the heaviside factor which controls steepness of the change between experts regions is optimized
    ///   to get best mixture quality.
    ///   * HARD: prediction is taken from the expert with highest responsibility
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
    ///   Number of internal GP hyperparameters optimization restarts (multistart).
    ///   When zero, optimization is disabled and theta init value is used as is.
    ///   Not to be confused with `Egor(infill_n_start=...)`, the infill criterion optimization multistart.
    #[pyo3(get, set)]
    pub theta_n_start: usize,

    /// (int >= 0)
    ///   Max number of likelihood evaluations of each GP hyperparameters optimization start.
    ///   This is an upper limit: each start gets clamp(10 * nx, 25, theta_max_eval) evaluations.
    ///   Not to be confused with `Egor.minimize(max_iters=...)`, the optimization iteration budget.
    #[pyo3(get, set)]
    pub theta_max_eval: usize,
}

/// Deprecated GP configuration names (old, new)
pub(crate) const GP_CONFIG_RENAMED: [(&str, &str); 2] =
    [("n_start", "theta_n_start"), ("max_eval", "theta_max_eval")];

impl<'a, 'py> FromPyObject<'a, 'py> for GpConfig {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(cfg) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(cfg.clone());
        }

        let dict = obj.cast::<PyDict>()?;
        let mut cfg = GpConfig::default();

        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            match resolve_renamed_key(&dict, &key, &GP_CONFIG_RENAMED)? {
                "regr_spec" => cfg.regr_spec = value.extract()?,
                "corr_spec" => cfg.corr_spec = value.extract()?,
                "kpls_dim" => cfg.kpls_dim = value.extract()?,
                "n_clusters" => cfg.n_clusters = value.extract()?,
                "recombination" => cfg.recombination = value.extract()?,
                "theta_init" => cfg.theta_init = value.extract()?,
                "theta_bounds" => cfg.theta_bounds = value.extract()?,
                "theta_n_start" => cfg.theta_n_start = value.extract()?,
                "theta_max_eval" => cfg.theta_max_eval = value.extract()?,
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_values(
        regr_spec: u8,
        corr_spec: u8,
        kpls_dim: Option<usize>,
        n_clusters: isize,
        recombination: Recombination,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        theta_n_start: usize,
        theta_max_eval: usize,
    ) -> Self {
        GpConfig {
            regr_spec,
            corr_spec,
            kpls_dim,
            n_clusters,
            recombination,
            theta_init,
            theta_bounds,
            theta_n_start,
            theta_max_eval,
        }
    }

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
        GpConfig::from_values(
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
    /// theta_n_start : int, optional
    ///     Number of GP hyperparameters optimization restarts, 0 to disable optimization (default: 10).
    ///     Not to be confused with `Egor(infill_n_start=...)`, the infill criterion optimization multistart.
    /// theta_max_eval : int, optional
    ///     Max number of likelihood evaluations of each GP hyperparameters optimization start (default: 50).
    ///     Upper limit: each start gets clamp(10 * nx, 25, theta_max_eval) evaluations.
    ///
    /// Deprecated
    /// ----------
    /// n_start : int, optional
    ///     Deprecated since 0.38.0, use `theta_n_start` instead.
    /// max_eval : int, optional
    ///     Deprecated since 0.38.0, use `theta_max_eval` instead.
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
        theta_n_start=None,
        theta_max_eval=None,
        *,
        n_start=None,
        max_eval=None,
))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python,
        regr_spec: u8,
        corr_spec: u8,
        kpls_dim: Option<usize>,
        n_clusters: isize,
        recombination: Recombination,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        theta_n_start: Option<usize>,
        theta_max_eval: Option<usize>,
        n_start: Option<usize>,
        max_eval: Option<usize>,
    ) -> PyResult<Self> {
        let default = GpConfig::default();
        Ok(GpConfig::from_values(
            regr_spec,
            corr_spec,
            kpls_dim,
            n_clusters,
            recombination,
            theta_init,
            theta_bounds,
            resolve_renamed(
                py,
                "n_start",
                n_start,
                "theta_n_start",
                theta_n_start,
                default.theta_n_start,
            )?,
            resolve_renamed(
                py,
                "max_eval",
                max_eval,
                "theta_max_eval",
                theta_max_eval,
                default.theta_max_eval,
            )?,
        ))
    }

    /// Deprecated since 0.38.0, use `theta_n_start` instead.
    #[getter(n_start)]
    fn get_n_start(&self, py: Python) -> PyResult<usize> {
        warn_deprecated(py, "GpConfig.n_start", "GpConfig.theta_n_start")?;
        Ok(self.theta_n_start)
    }

    #[setter(n_start)]
    fn set_n_start(&mut self, value: usize) -> PyResult<()> {
        // no `py` argument as pyo3-stub-gen does not support it on setters
        Python::attach(|py| warn_deprecated(py, "GpConfig.n_start", "GpConfig.theta_n_start"))?;
        self.theta_n_start = value;
        Ok(())
    }

    /// Deprecated since 0.38.0, use `theta_max_eval` instead.
    #[getter(max_eval)]
    fn get_max_eval(&self, py: Python) -> PyResult<usize> {
        warn_deprecated(py, "GpConfig.max_eval", "GpConfig.theta_max_eval")?;
        Ok(self.theta_max_eval)
    }

    #[setter(max_eval)]
    fn set_max_eval(&mut self, value: usize) -> PyResult<()> {
        // no `py` argument as pyo3-stub-gen does not support it on setters
        Python::attach(|py| warn_deprecated(py, "GpConfig.max_eval", "GpConfig.theta_max_eval"))?;
        self.theta_max_eval = value;
        Ok(())
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
                (
                    "theta_n_start",
                    self.theta_n_start.into_pyobject(py)?.into_any(),
                ),
                (
                    "theta_max_eval",
                    self.theta_max_eval.into_pyobject(py)?.into_any(),
                ),
            ],
        )
    }
}

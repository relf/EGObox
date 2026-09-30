use crate::deprecation::{resolve_renamed, resolve_renamed_key, warn_deprecated};
use crate::types::repr_kwargs;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// TREGO configuration specification which can be either
/// a boolean to activate/deactivate the TREGO strategy
/// or a full TregoConfig object.
pub enum TregoConfigSpec {
    Activated(bool),
    Custom(TregoConfig),
}

/// Trust region configuration for EGO optimization.
///
/// The TREGO algorithm enhances the Efficient Global Optimization (EGO)
/// by incorporating a trust region strategy to improve local convergence.
///
/// Parameters
/// ----------
/// n_global_local_steps : (int, int)
///     Number of global and local steps: a tuple specifying the number of
///     global and local optimization steps as (n_global_steps, n_local_steps).
/// radius_bounds : tuple of float
///     Trust region radius bounds as (dmin, dmax). The trust region radius
///     is constrained between these values.
/// alpha : float
///     Factor used within the trust region acceptance criteria defined as:
///     rho(sigma) = alpha * sigma * sigma
/// beta : float
///     Trust region contraction factor in ]0., 1.[
/// sigma0 : float
///     Initial trust region radius.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Clone, Debug)]
pub(crate) struct TregoConfig {
    /// Number of global and local optimization steps (n_global_steps, n_local_steps)
    #[pyo3(get, set)]
    pub n_global_local_steps: (usize, usize),

    /// Trust region radius bounds (dmin, dmax) with 0 < dmin < dmax
    #[pyo3(get, set)]
    pub radius_bounds: (f64, f64),

    /// Threshold ratio for iteration acceptance used in trust region criteria
    /// rho(sigma) = alpha * sigma * sigma
    #[pyo3(get, set)]
    pub alpha: f64,

    /// Trust region contraction factor
    #[pyo3(get, set)]
    pub beta: f64,

    /// Initial trust region radius
    #[pyo3(get, set)]
    pub sigma0: f64,
}

/// Deprecated TREGO configuration names (old, new)
const TREGO_CONFIG_RENAMED: [(&str, &str); 2] = [
    ("n_gl_steps", "n_global_local_steps"),
    ("d", "radius_bounds"),
];

impl<'a, 'py> FromPyObject<'a, 'py> for TregoConfig {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(cfg) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(cfg.clone());
        }

        let dict = obj.cast::<PyDict>()?;
        let mut cfg = TregoConfig::default();

        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            match resolve_renamed_key(&dict, &key, &TREGO_CONFIG_RENAMED)? {
                "n_global_local_steps" => cfg.n_global_local_steps = value.extract()?,
                "radius_bounds" => cfg.radius_bounds = value.extract()?,
                "alpha" => cfg.alpha = value.extract()?,
                "beta" => cfg.beta = value.extract()?,
                "sigma0" => cfg.sigma0 = value.extract()?,
                _ => return Err(PyValueError::new_err(format!("unknown trego key '{key}'"))),
            }
        }

        Ok(cfg)
    }
}

impl Default for TregoConfig {
    fn default() -> Self {
        TregoConfig {
            n_global_local_steps: (1, 4),
            radius_bounds: (1e-6, 1.),
            alpha: 1.0,
            beta: 0.9,
            sigma0: 1e-1,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl TregoConfig {
    /// Create a new TReGO configuration.
    ///
    /// Parameters
    /// ----------
    /// n_global_local_steps : (int, int), optional
    ///     Number of global/local steps (default: (1, 4))
    /// radius_bounds : tuple of float, optional
    ///     Trust region radius bounds (default: (1e-6, 1.0))
    /// alpha : float, optional
    ///     Threshold ratio for iteration acceptance (default: 1.0)
    /// beta : float, optional
    ///     Trust region contraction factor (default: 0.9)
    /// sigma0 : float, optional
    ///     Initial trust region radius (default: 0.1)
    ///
    /// Deprecated
    /// ----------
    /// n_gl_steps : (int, int), optional
    ///     Deprecated since 0.38.0, use `n_global_local_steps` instead.
    /// d : tuple of float, optional
    ///     Deprecated since 0.38.0, use `radius_bounds` instead.
    ///
    /// Returns
    /// -------
    /// TregoConfig
    ///     A new TREGO configuration object
    #[new]
    #[pyo3(signature = (
        n_global_local_steps=None,
        radius_bounds=None,
        alpha=TregoConfig::default().alpha,
        beta=TregoConfig::default().beta,
        sigma0=TregoConfig::default().sigma0,
        *,
        n_gl_steps=None,
        d=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python,
        n_global_local_steps: Option<(usize, usize)>,
        radius_bounds: Option<(f64, f64)>,
        alpha: f64,
        beta: f64,
        sigma0: f64,
        n_gl_steps: Option<(usize, usize)>,
        d: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let default = TregoConfig::default();
        Ok(TregoConfig {
            n_global_local_steps: resolve_renamed(
                py,
                "n_gl_steps",
                n_gl_steps,
                "n_global_local_steps",
                n_global_local_steps,
                default.n_global_local_steps,
            )?,
            radius_bounds: resolve_renamed(
                py,
                "d",
                d,
                "radius_bounds",
                radius_bounds,
                default.radius_bounds,
            )?,
            alpha,
            beta,
            sigma0,
        })
    }

    /// Deprecated since 0.38.0, use `n_global_local_steps` instead.
    #[getter(n_gl_steps)]
    fn get_n_gl_steps(&self, py: Python) -> PyResult<(usize, usize)> {
        warn_deprecated(
            py,
            "TregoConfig.n_gl_steps",
            "TregoConfig.n_global_local_steps",
        )?;
        Ok(self.n_global_local_steps)
    }

    #[setter(n_gl_steps)]
    fn set_n_gl_steps(&mut self, value: (usize, usize)) -> PyResult<()> {
        // no `py` argument as pyo3-stub-gen does not support it on setters
        Python::attach(|py| {
            warn_deprecated(
                py,
                "TregoConfig.n_gl_steps",
                "TregoConfig.n_global_local_steps",
            )
        })?;
        self.n_global_local_steps = value;
        Ok(())
    }

    /// Deprecated since 0.38.0, use `radius_bounds` instead.
    #[getter(d)]
    fn get_d(&self, py: Python) -> PyResult<(f64, f64)> {
        warn_deprecated(py, "TregoConfig.d", "TregoConfig.radius_bounds")?;
        Ok(self.radius_bounds)
    }

    #[setter(d)]
    fn set_d(&mut self, value: (f64, f64)) -> PyResult<()> {
        // no `py` argument as pyo3-stub-gen does not support it on setters
        Python::attach(|py| warn_deprecated(py, "TregoConfig.d", "TregoConfig.radius_bounds"))?;
        self.radius_bounds = value;
        Ok(())
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "TregoConfig",
            &[
                (
                    "n_global_local_steps",
                    self.n_global_local_steps.into_pyobject(py)?.into_any(),
                ),
                (
                    "radius_bounds",
                    self.radius_bounds.into_pyobject(py)?.into_any(),
                ),
                ("alpha", self.alpha.into_pyobject(py)?.into_any()),
                ("beta", self.beta.into_pyobject(py)?.into_any()),
                ("sigma0", self.sigma0.into_pyobject(py)?.into_any()),
            ],
        )
    }
}

impl From<TregoConfig> for egobox_ego::TregoStrategy {
    fn from(config: TregoConfig) -> Self {
        egobox_ego::TregoStrategy::default()
            .n_gl_steps(config.n_global_local_steps)
            .d(config.radius_bounds)
            .alpha(config.alpha)
            .beta(config.beta)
            .sigma0(config.sigma0)
    }
}

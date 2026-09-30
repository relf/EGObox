use crate::types::repr_kwargs;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// TREGO configuration specification which can be either
/// a boolean to activate/deactivate the TREGO strategy
/// or a full TregoConfig object.
#[derive(FromPyObject)]
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
/// n_gl_steps : (int, int)
///     Number of global and local steps (gl): a tuple specifying the number of
///     global and local optimization steps as (n_global_steps, n_local_steps).
/// d : tuple of float
///     Trust region distance (radius) bounds as (dmin, dmax). The trust region radius
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
    pub n_gl_steps: (usize, usize),

    /// Trust region size bounds (dmin, dmax) with 0 < dmin < dmax
    #[pyo3(get, set)]
    pub d: (f64, f64),

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

impl<'a, 'py> FromPyObject<'a, 'py> for TregoConfig {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(cfg) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(cfg.clone());
        }

        let dict = obj.cast::<PyDict>()?;
        let mut cfg = TregoConfig::default();

        for key_any in dict.keys().iter() {
            let key = key_any.extract::<String>()?;
            match key.as_str() {
                "n_gl_steps" => cfg.n_gl_steps = dict.get_item("n_gl_steps")?.unwrap().extract()?,
                "d" => cfg.d = dict.get_item("d")?.unwrap().extract()?,
                "alpha" => cfg.alpha = dict.get_item("alpha")?.unwrap().extract()?,
                "beta" => cfg.beta = dict.get_item("beta")?.unwrap().extract()?,
                "sigma0" => cfg.sigma0 = dict.get_item("sigma0")?.unwrap().extract()?,
                _ => return Err(PyValueError::new_err(format!("unknown trego key '{key}'"))),
            }
        }

        Ok(cfg)
    }
}

impl Default for TregoConfig {
    fn default() -> Self {
        TregoConfig::new((1, 4), (1e-6, 1.), 1.0, 0.9, 1e-1)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl TregoConfig {
    /// Create a new TReGO configuration.
    ///
    /// Parameters
    /// ----------
    /// n_gl_steps : (int, int), optional
    ///     Number of global/local steps (default: (1, 4))
    /// d : tuple of float, optional
    ///     Trust region size bounds (default: (1e-6, 1.0))
    /// alpha : float, optional
    ///     Threshold ratio for iteration acceptance (default: 1.0)
    /// beta : float, optional
    ///     Trust region contraction factor (default: 0.9)
    /// sigma0 : float, optional
    ///     Initial trust region radius (default: 0.1)
    ///
    /// Returns
    /// -------
    /// TregoConfig
    ///     A new TREGO configuration object
    #[new]
    #[pyo3(signature = (
        n_gl_steps=TregoConfig::default().n_gl_steps,
        d=TregoConfig::default().d,
        alpha=TregoConfig::default().alpha,
        beta=TregoConfig::default().beta,
        sigma0=TregoConfig::default().sigma0,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_gl_steps: (usize, usize),
        d: (f64, f64),
        alpha: f64,
        beta: f64,
        sigma0: f64,
    ) -> Self {
        TregoConfig {
            n_gl_steps,
            d,
            alpha,
            beta,
            sigma0,
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "TregoConfig",
            &[
                ("n_gl_steps", self.n_gl_steps.into_pyobject(py)?.into_any()),
                ("d", self.d.into_pyobject(py)?.into_any()),
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
            .n_gl_steps(config.n_gl_steps)
            .d(config.d)
            .alpha(config.alpha)
            .beta(config.beta)
            .sigma0(config.sigma0)
    }
}

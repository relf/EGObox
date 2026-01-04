use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// Trust region configuration for EGO optimization.
///
/// The TREGO algorithm enhances the Efficient Global Optimization (EGO)
/// by incorporating a trust region strategy to improve local convergence.
///
/// Parameters
/// ----------
/// n_gl_steps : (int, int)
///     A tuple specifying the number of global and local optimization steps as
///    (n_global_steps, n_local_steps).
/// d : tuple of float
///     Trust region size bounds as (min, max). The trust region radius
///     is constrained between these values.
/// alpha : float
///     Factor used within the trust region acceptance criteria defined as:
///     rho(sigma) = alpha * sigma * sigma
/// beta : float
///     Trust region contraction factor in ]0., 1.[
/// sigma0 : float
///     Initial trust region radius.
#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct TregoConfig {
    /// Number of global optimization steps
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

impl Default for TregoConfig {
    fn default() -> Self {
        TregoConfig::new((1, 4), (1e-6, 1.), 1.0, 0.9, 1e-1)
    }
}

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
}

impl From<TregoConfig> for egobox_ego::TregoConfig {
    fn from(config: TregoConfig) -> Self {
        egobox_ego::TregoConfig::default()
            .activated(true)
            .n_gl_steps(config.n_gl_steps)
            .d(config.d)
            .alpha(config.alpha)
            .beta(config.beta)
            .sigma0(config.sigma0)
    }
}

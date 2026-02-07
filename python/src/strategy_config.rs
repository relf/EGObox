use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

// =============================================================================
// TregoStrategy - Trust Region EGO iteration strategy
// =============================================================================

/// TREGO strategy specification which can be either
/// a boolean to activate/deactivate the TREGO strategy
/// or a full TregoStrategy object.
#[derive(FromPyObject)]
pub enum TregoStrategySpec {
    Activated(bool),
    Custom(TregoStrategy),
}

/// Trust region strategy for EGO optimization (TREGO).
///
/// The TREGO algorithm enhances the Efficient Global Optimization (EGO)
/// by incorporating a trust region strategy to improve local convergence.
/// It alternates between global EGO steps and local trust region steps.
///
/// This class is a Python wrapper around the Rust ``TregoStrategy`` type.
/// The deprecated alias ``TregoConfig`` is still available for backward compatibility.
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
///
/// Examples
/// --------
/// >>> import egobox as egx
/// >>> # Default TREGO
/// >>> egor = egx.Egor(xlimits, trego=True)
/// >>> # Custom TREGO configuration
/// >>> egor = egx.Egor(xlimits, trego=egx.TregoStrategy(n_gl_steps=(2, 3), beta=0.8))
/// >>> # Backward-compatible alias still works
/// >>> egor = egx.Egor(xlimits, trego=egx.TregoConfig(n_gl_steps=(4, 1)))
#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct TregoStrategy {
    /// Number of global/local optimization steps as (n_global, n_local)
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

impl Default for TregoStrategy {
    fn default() -> Self {
        TregoStrategy::new((1, 4), (1e-6, 1.), 1.0, 0.9, 1e-1)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl TregoStrategy {
    /// Create a new TREGO strategy configuration.
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
    /// TregoStrategy
    ///     A new TREGO strategy object
    #[new]
    #[pyo3(signature = (
        n_gl_steps=TregoStrategy::default().n_gl_steps,
        d=TregoStrategy::default().d,
        alpha=TregoStrategy::default().alpha,
        beta=TregoStrategy::default().beta,
        sigma0=TregoStrategy::default().sigma0,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_gl_steps: (usize, usize),
        d: (f64, f64),
        alpha: f64,
        beta: f64,
        sigma0: f64,
    ) -> Self {
        TregoStrategy {
            n_gl_steps,
            d,
            alpha,
            beta,
            sigma0,
        }
    }
}

impl From<TregoStrategy> for egobox_ego::TregoStrategy {
    fn from(config: TregoStrategy) -> Self {
        egobox_ego::TregoStrategy::default()
            .n_gl_steps(config.n_gl_steps)
            .d(config.d)
            .alpha(config.alpha)
            .beta(config.beta)
            .sigma0(config.sigma0)
    }
}

// =============================================================================
// CooperativeActivity - CoEGO variable grouping strategy
// =============================================================================

/// Cooperative EGO (CoEGO) activity strategy for high-dimensional problems.
///
/// Randomly decomposes the design space into ``n_coop`` groups of variables.
/// At each iteration, only one group is optimized while others are held fixed.
/// This reduces the effective dimensionality of the surrogate models.
///
/// This class is a Python wrapper around the Rust ``CooperativeActivity`` type.
/// The deprecated ``coego_n_coop`` integer parameter on ``Egor`` is still
/// accepted for backward compatibility, but using ``CooperativeActivity``
/// is preferred for clarity and forward compatibility.
///
/// Parameters
/// ----------
/// n_coop : int
///     Number of cooperative groups to split variables into.
///     Should ideally be a divisor of the number of design variables (nx).
///     If not a divisor, the remainder variables are distributed across groups.
///
/// Examples
/// --------
/// >>> import egobox as egx
/// >>> # New style: using CooperativeActivity strategy
/// >>> egor = egx.Egor(xlimits, activity=egx.CooperativeActivity(n_coop=5))
/// >>> # Deprecated style: still works
/// >>> egor = egx.Egor(xlimits, coego_n_coop=5)
#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct CooperativeActivity {
    /// Number of cooperative groups to split variables into
    #[pyo3(get, set)]
    pub n_coop: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl CooperativeActivity {
    /// Create a new CooperativeActivity strategy.
    ///
    /// Parameters
    /// ----------
    /// n_coop : int
    ///     Number of cooperative groups
    ///
    /// Returns
    /// -------
    /// CooperativeActivity
    ///     A new cooperative activity strategy object
    #[new]
    #[pyo3(signature = (n_coop))]
    pub fn new(n_coop: usize) -> Self {
        CooperativeActivity { n_coop }
    }
}

impl From<CooperativeActivity> for egobox_ego::CooperativeActivity {
    fn from(config: CooperativeActivity) -> Self {
        egobox_ego::CooperativeActivity::new(config.n_coop)
    }
}

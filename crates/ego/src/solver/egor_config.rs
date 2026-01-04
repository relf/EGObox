//! Egor optimizer configuration.
use crate::{HotStartMode, criteria::*, errors::Result, types::*};
use egobox_gp::ThetaTuning;
use egobox_moe::NbClusters;
use egobox_moe::Recombination;
use egobox_moe::{CorrelationSpec, RegressionSpec};
use ndarray::Array1;
use ndarray::Array2;

use serde::{Deserialize, Serialize};

/// Default number of starts for multistart approach used for optimization
pub const EGO_GP_OPTIM_N_START: usize = 10;
/// Default number of likelihood evaluation during one internal optimization
pub const EGO_GP_OPTIM_MAX_EVAL: usize = 50;

/// GP configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpConfig {
    /// Regression specification for GP models used by mixture of experts (see [egobox_moe])
    pub(crate) regression_spec: RegressionSpec,
    /// Correlation specification for GP models used by mixture of experts (see [egobox_moe])
    pub(crate) correlation_spec: CorrelationSpec,
    /// Optional dimension reduction (see [egobox_moe])
    pub(crate) kpls_dim: Option<usize>,
    /// Number of clusters used by mixture of experts (see [egobox_moe])
    /// When set to Auto the clusters are computes automatically and refreshed
    /// every 10-points (tentative) additions
    pub(crate) n_clusters: NbClusters,
    /// The mode of recombination to get the output prediction from experts prediction
    pub(crate) recombination: Recombination<f64>,
    /// Parameter tuning hint of the autocorrelation model
    pub(crate) theta_tuning: ThetaTuning<f64>,
    /// Number of starts for multistart approach used for optimization
    pub(crate) n_start: usize,
    /// Number of likelihood evaluation during one internal optimization
    pub(crate) max_eval: usize,
}

impl Default for GpConfig {
    fn default() -> Self {
        GpConfig {
            regression_spec: RegressionSpec::CONSTANT,
            correlation_spec: CorrelationSpec::SQUAREDEXPONENTIAL,
            kpls_dim: None,
            n_clusters: NbClusters::default(),
            recombination: Recombination::Smooth(Some(1.)),
            theta_tuning: ThetaTuning::default(),
            n_start: EGO_GP_OPTIM_N_START,
            max_eval: EGO_GP_OPTIM_MAX_EVAL,
        }
    }
}

impl GpConfig {
    /// Sets the allowed regression models used in gaussian processes.
    pub fn regression_spec(mut self, regression_spec: RegressionSpec) -> Self {
        self.regression_spec = regression_spec;
        self
    }

    /// Sets the allowed correlation models used in gaussian processes.
    pub fn correlation_spec(mut self, correlation_spec: CorrelationSpec) -> Self {
        self.correlation_spec = correlation_spec;
        self
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.kpls_dim = kpls_dim;
        self
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    ///
    /// This is used to address high-dimensional problems typically when `nx` > 9 wher `nx` is the dimension of `x`.
    pub fn kpls(mut self, kpls_dim: usize) -> Self {
        self.kpls_dim = Some(kpls_dim);
        self
    }

    /// Removes any PLS dimension reduction usage
    pub fn no_kpls(mut self) -> Self {
        self.kpls_dim = None;
        self
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    pub fn n_clusters(mut self, n_clusters: NbClusters) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Sets the mode of recombination to get the output prediction from experts prediction
    pub fn recombination(mut self, recombination: Recombination<f64>) -> Self {
        self.recombination = recombination;
        self
    }

    /// Sets the parameter tuning hint of the autocorrelation model
    pub fn theta_tuning(mut self, theta_tuning: ThetaTuning<f64>) -> Self {
        self.theta_tuning = theta_tuning;
        self
    }

    /// Sets the number of starts for multistart approach used for optimization
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.n_start = n_start;
        self
    }

    /// Sets the number of likelihood evaluation during one internal optimization
    pub fn max_eval(mut self, max_eval: usize) -> Self {
        self.max_eval = max_eval;
        self
    }
}

/// A structure to handle TREGO method parameterization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TregoConfig {
    /// Whether the TReGO algorithm is activated
    pub(crate) activated: bool,
    /// Number of global and local optimization steps.
    /// number of steps should be strictly positive
    pub(crate) n_gl_steps: (usize, usize),
    /// Trust region size bounds (min, max)
    pub(crate) d: (f64, f64),
    /// Threshold ratio for iteration acceptance used in trust region criteria
    /// rho(sigma) = alpha * sigma * sigma
    pub(crate) alpha: f64,
    /// Trust region contraction factor in ]0, 1.[.
    pub(crate) beta: f64,
    /// Initial trust region radius
    pub(crate) sigma0: f64,
}

impl Default for TregoConfig {
    fn default() -> Self {
        TregoConfig {
            activated: false,
            n_gl_steps: (1, 4),
            d: (1e-6, 1.),
            alpha: 1.0,
            beta: 0.9,
            sigma0: 1e-1,
        }
    }
}

impl TregoConfig {
    /// Sets whether TReGO is activated
    pub fn activated(mut self, activated: bool) -> Self {
        self.activated = activated;
        self
    }

    /// Sets the number of global optimization steps
    pub fn n_gl_steps(mut self, n_gl_steps: (usize, usize)) -> Self {
        self.n_gl_steps = n_gl_steps;
        self
    }

    /// Sets the trust region size bounds (min, max)
    pub fn d(mut self, d: (f64, f64)) -> Self {
        self.d = d;
        self
    }

    /// Sets the threshold ratio for iteration acceptance
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the trust region contraction factor
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Sets the initial trust region radius
    pub fn sigma0(mut self, sigma0: f64) -> Self {
        self.sigma0 = sigma0;
        self
    }
}

/// An enum to specify CoEGO status and component number
pub enum CoegoStatus {
    /// Do not use CoEGO algorithm
    Disabled,
    /// Apply CoEGO algorithm with a specified number of groups of components
    /// meaning at most nx / n_coop components will be optimized at a time
    Enabled(usize),
}

/// A structure to handle CoEGO method parameterization
/// CoEGO variant is intended to be used for high dimensional problems
/// with dim > 100
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CoegoConfig {
    pub(crate) activated: bool,
    pub(crate) n_coop: usize,
}

impl Default for CoegoConfig {
    fn default() -> Self {
        CoegoConfig {
            activated: false,
            n_coop: 5,
        }
    }
}

/// Max number of iterations of EGO algorithm (aka iteration budget)
pub const EGO_DEFAULT_MAX_ITERS: usize = 20;
/// Number of restart for optimization of the infill criterion (aka multistart)
pub const EGO_DEFAULT_N_START: usize = 20;

/// Valid Egor optimizer configuration
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ValidEgorConfig {
    /// Max number of function iterations allocated to find the optimum (aka iteration budget)
    /// Note 1 : The number of cost function evaluations is deduced using the following formula (n_doe + max_iters)
    /// Note 2 : When q_points > 1, the number of cost function evaluations is (n_doe + max_iters * q_points)
    /// is is an upper bounds as some points may be rejected as being to close to previous ones.   
    pub(crate) max_iters: usize,
    /// Number of starts for multistart approach used for optimization
    pub(crate) n_start: usize,
    /// Number of initial doe drawn using Latin hypercube sampling
    /// Note: n_doe > 0; otherwise n_doe = max(xdim + 1, 5)
    pub(crate) n_doe: usize,
    /// Number of Constraints
    /// Note: dim function ouput = 1 objective + n_cstr constraints
    pub(crate) n_cstr: usize,
    /// Optional constraints violation tolerance meaning cstr < cstr_tol is considered valid
    pub(crate) cstr_tol: Option<Array1<f64>>,
    /// Initial doe can be either \[x\] with x inputs only or an evaluated doe \[x, y\]
    /// Note: x dimension is determined using `xlimits.nrows()`
    pub(crate) doe: Option<Array2<f64>>,
    /// Multipoint strategy used to get several points to be evaluated at each iteration
    pub(crate) q_ei: QEiStrategy,
    /// Interval between two hyperparameters optimizations (as iteration number modulo)
    /// hyperparameters are optimized or re-used from an iteration to another when getting q points
    pub(crate) q_optmod: usize,
    /// Number of points returned by EGO iteration (aka qEI Multipoint strategy)
    /// Actually as some point determination may fail (at most q_points are returned)
    pub(crate) q_points: usize,
    /// General configuration for GP models used by the optimizer
    pub(crate) gp: GpConfig,
    /// Criterion to select next point to evaluate
    pub(crate) infill_criterion: Box<dyn InfillCriterion>,
    /// The optimizer used to optimize infill criterium
    pub(crate) infill_optimizer: InfillOptimizer,
    /// Specification of a target objective value which is used to stop the algorithm once reached
    pub(crate) target: f64,
    /// Directory to save intermediate results: inital doe + evalutions at each iteration
    pub(crate) outdir: Option<String>,
    /// If true use `outdir` to retrieve and start from previous results
    pub(crate) warm_start: bool,
    /// If some enable checkpointing allowing to restart for given ext_iters number of iteration from last checkpointed iteration
    pub(crate) hot_start: HotStartMode,
    /// List of x types allowing the handling of discrete input variables
    pub(crate) xtypes: Vec<XType>,
    /// A random generator seed used to get reproductible results.
    pub(crate) seed: Option<u64>,
    /// TREGO parameterization
    pub(crate) trego_config: TregoConfig,
    /// CoEGO  parameterization
    pub(crate) coego: CoegoConfig,
    /// Constrained infill criterion activation
    pub(crate) cstr_infill: bool,
    /// Constraints criterion
    pub(crate) cstr_strategy: ConstraintStrategy,
}

impl Default for ValidEgorConfig {
    fn default() -> Self {
        ValidEgorConfig {
            max_iters: EGO_DEFAULT_MAX_ITERS,
            n_start: EGO_DEFAULT_N_START,
            n_doe: 0,
            n_cstr: 0,
            cstr_tol: None,
            doe: None,
            q_ei: QEiStrategy::KrigingBeliever,
            q_optmod: 1,
            q_points: 1,
            gp: GpConfig::default(),
            infill_criterion: Box::new(LOG_EI),
            infill_optimizer: InfillOptimizer::Slsqp,
            target: f64::MIN,
            outdir: None,
            warm_start: false,
            hot_start: HotStartMode::Disabled,
            xtypes: vec![],
            seed: None,
            trego_config: TregoConfig::default(),
            coego: CoegoConfig::default(),
            cstr_infill: false,
            cstr_strategy: ConstraintStrategy::MeanConstraint,
        }
    }
}

impl ValidEgorConfig {
    /// Check whether we are in a discrete optimization context
    pub fn discrete(&self) -> bool {
        crate::utils::discrete(&self.xtypes)
    }
}

/// Egor optimizer configuration builder
#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct EgorConfig(ValidEgorConfig);

impl EgorConfig {
    /// Sets the infill criterion
    pub fn infill_criterion(mut self, infill_criterion: Box<dyn InfillCriterion>) -> Self {
        self.0.infill_criterion = infill_criterion;
        self
    }

    /// Sets max number of iterations to optimize the objective function
    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.0.max_iters = max_iters;
        self
    }

    /// Sets the number of runs of infill strategy optimizations (best result taken)
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.0.n_start = n_start;
        self
    }

    /// Number of samples of initial LHS sampling (used when DOE not provided by the user)
    ///
    /// When 0 a number of points is computed automatically regarding the number of input variables
    /// of the function under optimization.
    pub fn n_doe(mut self, n_doe: usize) -> Self {
        self.0.n_doe = n_doe;
        self
    }

    /// Sets the number of constraint functions
    pub fn n_cstr(mut self, n_cstr: usize) -> Self {
        self.0.n_cstr = n_cstr;
        self
    }

    /// Sets the tolerance on constraints violation (`cstr < tol`)
    pub fn cstr_tol(mut self, tol: Array1<f64>) -> Self {
        self.0.cstr_tol = Some(tol);
        self
    }

    /// Sets an initial DOE \['ns', `nt`\] containing `ns` samples.
    ///
    /// Either `nt` = `nx` then only `x` input values are specified and `ns` evals are done to get y ouput doe values,
    /// or `nt = nx + ny` then `x = doe\[:, :nx\]` and `y = doe\[:, nx:\]` are specified
    pub fn doe(mut self, doe: &Array2<f64>) -> Self {
        self.0.doe = Some(doe.to_owned());
        self
    }

    /// Removes any previously specified initial doe to get the default doe usage
    pub fn default_doe(mut self) -> Self {
        self.0.doe = None;
        self
    }

    /// Sets the parallel infill strategy
    ///
    /// Parallel infill criterion to get virtual next promising points in order to allow
    /// n parallel evaluations of the function under optimization.
    pub fn qei_strategy(mut self, q_ei: QEiStrategy) -> Self {
        self.0.q_ei = q_ei;
        self
    }

    /// Sets the number of iteration interval between two hyperparameter optimization
    /// when computing q points to be evaluated in parallel
    pub fn q_optmod(mut self, q_optmod: usize) -> Self {
        self.0.q_optmod = q_optmod;
        self
    }

    /// Sets Number of parallel evaluations of the function under optimization
    pub fn q_points(mut self, q_points: usize) -> Self {
        self.0.q_points = q_points;
        self
    }
    /// Sets the infill strategy
    pub fn infill_strategy(mut self, infill: InfillStrategy) -> Self {
        self.0.infill_criterion = match infill {
            InfillStrategy::EI => Box::new(EI),
            InfillStrategy::LogEI => Box::new(LOG_EI),
            InfillStrategy::WB2 => Box::new(WB2),
            InfillStrategy::WB2S => Box::new(WB2S),
        };
        self
    }

    /// Sets the infill optimizer
    pub fn infill_optimizer(mut self, optimizer: InfillOptimizer) -> Self {
        self.0.infill_optimizer = optimizer;
        self
    }

    /// Sets the configuration of the GPs
    pub fn configure_gp<F: FnOnce(GpConfig) -> GpConfig>(mut self, init: F) -> Self {
        self.0.gp = init(self.0.gp);
        self
    }

    /// Sets a known target minimum to be used as a stopping criterion.
    pub fn target(mut self, target: f64) -> Self {
        self.0.target = target;
        self
    }

    /// Sets a directory to write optimization history and used as search path for warm start doe
    pub fn outdir(mut self, outdir: impl Into<String>) -> Self {
        self.0.outdir = Some(outdir.into());
        self
    }
    /// Do not write optimization history
    pub fn no_outdir(mut self) -> Self {
        self.0.outdir = None;
        self
    }

    /// Whether we start by loading last DOE saved in `outdir` as initial DOE
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.0.warm_start = warm_start;
        self
    }

    /// Whether checkpointing is enabled allowing hot start from previous checkpointed iteration if any
    pub fn hot_start(mut self, hot_start: HotStartMode) -> Self {
        self.0.hot_start = hot_start;
        self
    }

    /// Allow to specify a seed for random number generator to allow
    /// reproducible runs.
    pub fn seed(mut self, seed: u64) -> Self {
        self.0.seed = Some(seed);
        self
    }

    /// Define design space with given x types
    pub fn xtypes(mut self, xtypes: &[XType]) -> Self {
        self.0.xtypes = xtypes.into();
        self
    }

    /// Activate or deactivate TREGO method
    pub fn trego(mut self, activated: bool) -> Self {
        self.0.trego_config.activated = activated;
        self
    }

    /// Configure TREGO parameters. TREGO is automatically activated by this method
    pub fn configure_trego<F: FnOnce(TregoConfig) -> TregoConfig>(mut self, init: F) -> Self {
        self.0.trego_config = init(self.0.trego_config);
        self.0.trego_config.activated = true;
        self
    }

    /// Activate CoEGO method
    pub fn coego(mut self, status: CoegoStatus) -> Self {
        match status {
            CoegoStatus::Disabled => self.0.coego.activated = false,
            CoegoStatus::Enabled(n) => {
                self.0.coego.activated = true;
                self.0.coego.n_coop = n;
            }
        }
        self
    }

    /// Activate constrained infill criterion
    pub fn cstr_infill(mut self, activated: bool) -> Self {
        self.0.cstr_infill = activated;
        self
    }

    /// Sets the infill strategy
    pub fn cstr_strategy(mut self, cstr_strategy: ConstraintStrategy) -> Self {
        self.0.cstr_strategy = cstr_strategy;
        self
    }

    /// Checks and wraps an EgorConfig
    pub fn check(self) -> Result<ValidEgorConfig> {
        let config = self.0;
        // Check cstr_tol length if any
        if config.n_cstr > 0
            && config.cstr_tol.is_some()
            && config.cstr_tol.as_ref().unwrap().len() != config.n_cstr
        {
            return Err(crate::EgoError::InvalidConfigError(format!(
                "EgorConfig invalid: cstr_tol length ({}) does not match n_cstr ({})",
                config.cstr_tol.as_ref().unwrap().len(),
                config.n_cstr
            )));
        }

        // Check exclusicve use of coego and gp_config kpls
        if config.coego.activated && config.gp.kpls_dim.is_some() {
            return Err(crate::EgoError::InvalidConfigError(
                "EgorConfig invalid: CoEGO and KPLS cannot be used together".to_string(),
            ));
        }

        Ok(config)
    }
}

use crate::correlation_models::CorrelationModel;
use crate::errors::{GpError, Result};
use crate::mean_models::RegressionModel;
use crate::{GP_COBYLA_MAX_EVAL, GP_COBYLA_MIN_EVAL, GP_OPTIM_N_START};
use linfa::{Float, ParamGuard};

use ndarray::{Array1, array};
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// An enum to represent a n-dim hyper parameter tuning
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub enum ThetaTuning<F: Float> {
    /// Constant parameter (ie given not estimated)
    Fixed(Array1<F>),
    /// Parameter is optimized between given bounds (lower, upper) starting from the inital guess
    Full {
        /// Initial guess for the parameter
        init: Array1<F>,
        /// Bounds for the parameter array(lower, upper)
        bounds: Array1<(F, F)>,
    },
    /// Parameter is partially optimized on specified active components
    Partial {
        /// Initial guess for the parameter
        init: Array1<F>,
        /// Bounds for the parameter array(lower, upper)
        bounds: Array1<(F, F)>,
        /// Active components for the parameter optimization
        active: Vec<usize>,
    },
}

impl<F: Float> Default for ThetaTuning<F> {
    fn default() -> Self {
        ThetaTuning::Full {
            init: array![F::cast(ThetaTuning::<F>::DEFAULT_INIT)],
            bounds: array![(
                F::cast(ThetaTuning::<F>::DEFAULT_BOUNDS.0),
                F::cast(ThetaTuning::<F>::DEFAULT_BOUNDS.1),
            )],
        }
    }
}

impl<F: Float> ThetaTuning<F> {
    /// Default initial theta value
    pub const DEFAULT_INIT: f64 = 1e-1;
    /// Default bounds for theta values
    pub const DEFAULT_BOUNDS: (f64, f64) = (1e-2, 1e1);

    /// Get initial theta value
    pub fn init(&self) -> &Array1<F> {
        match self {
            ThetaTuning::Full { init, bounds: _ } => init,
            ThetaTuning::Partial {
                init,
                active: _,
                bounds: _,
            } => init,
            ThetaTuning::Fixed(init) => init,
        }
    }

    /// Get bounds for theta value
    pub fn bounds(&self) -> Option<&Array1<(F, F)>> {
        match self {
            ThetaTuning::Full { init: _, bounds } => Some(bounds),
            ThetaTuning::Partial {
                init: _,
                active: _,
                bounds,
            } => Some(bounds),
            ThetaTuning::Fixed(_) => None,
        }
    }

    /// Convert to fixed theta tuning with the same initial theta value
    pub fn to_fixed(&self) -> Self {
        ThetaTuning::Fixed(self.init().clone())
    }

    /// Check if theta tuning is fixed
    pub fn is_fixed(&self) -> bool {
        matches!(self, ThetaTuning::Fixed(_))
    }
}

/// A set of validated GP parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "F: Serialize, Mean: Serialize, Corr: Serialize",
        deserialize = "F: Deserialize<'de>, Mean: Deserialize<'de>, Corr: Deserialize<'de>"
    ))
)]
pub struct GpValidParams<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> {
    /// Parameter tuning hint of the autocorrelation model
    pub(crate) theta_tuning: ThetaTuning<F>,
    /// Regression model representing the mean(x)
    pub(crate) mean: Mean,
    /// Correlation model representing the spatial correlation between errors at e(x) and e(x')
    pub(crate) corr: Corr,
    /// Optionally apply dimension reduction (KPLS) or not
    pub(crate) kpls_dim: Option<usize>,
    /// Number of internal likelihood optimization restart
    pub(crate) n_start: usize,
    /// Max number of internal likelihood evaluation during optimization
    pub(crate) max_eval: usize,
    /// Parameter to improve numerical stability
    pub(crate) nugget: F,
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> Default
    for GpValidParams<F, Mean, Corr>
{
    fn default() -> GpValidParams<F, Mean, Corr> {
        GpValidParams {
            theta_tuning: ThetaTuning::default(),
            mean: Mean::default(),
            corr: Corr::default(),
            kpls_dim: None,
            n_start: GP_OPTIM_N_START,
            max_eval: GP_COBYLA_MAX_EVAL,
            nugget: F::cast(100.0) * F::epsilon(),
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GpValidParams<F, Mean, Corr> {
    /// Get mean model  
    pub fn mean(&self) -> &Mean {
        &self.mean
    }

    /// Get correlation corr k(x, x')
    pub fn corr(&self) -> &Corr {
        &self.corr
    }

    /// Get starting theta value for optimization
    pub fn theta_tuning(&self) -> &ThetaTuning<F> {
        &self.theta_tuning
    }

    /// Get number of components used by PLS
    pub fn kpls_dim(&self) -> Option<&usize> {
        self.kpls_dim.as_ref()
    }

    /// Get the number of internal optimization restart
    pub fn n_start(&self) -> usize {
        self.n_start
    }

    /// Get the max number of internal likelihood evaluations during one optimization
    pub fn max_eval(&self) -> usize {
        self.max_eval
    }

    /// Get number of components used by PLS
    pub fn nugget(&self) -> F {
        self.nugget
    }
}

#[derive(Clone, Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [GP algorithm](struct.GaussianProcess.html).
pub struct GpParams<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>>(
    GpValidParams<F, Mean, Corr>,
);

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GpParams<F, Mean, Corr> {
    /// A constructor for GP parameters given mean and correlation models
    pub fn new(mean: Mean, corr: Corr) -> GpParams<F, Mean, Corr> {
        Self(GpValidParams {
            mean,
            corr,
            ..Default::default()
        })
    }

    /// A constructor for GP parameters from validated parameters
    pub fn new_from_valid(params: &GpValidParams<F, Mean, Corr>) -> Self {
        Self(params.clone())
    }

    /// Set mean model.
    pub fn mean(mut self, mean: Mean) -> Self {
        self.0.mean = mean;
        self
    }

    /// Set correlation model.
    pub fn corr(mut self, corr: Corr) -> Self {
        self.0.corr = corr;
        self
    }

    /// Set the number of PLS components.
    /// Should be 0 < n < pb size (i.e. x dimension)
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.kpls_dim = kpls_dim;
        self
    }

    /// Set value for theta hyper parameter.
    ///
    /// When theta is optimized, the internal optimization is started from `theta_init`.
    /// When theta is fixed, this set theta constant value.
    pub fn theta_init(mut self, theta_init: Array1<F>) -> Self {
        self.0.theta_tuning = match self.0.theta_tuning {
            ThetaTuning::Full { init: _, bounds } => ThetaTuning::Full {
                init: theta_init,
                bounds,
            },
            ThetaTuning::Partial {
                init: _,
                active: _,
                bounds,
            } => ThetaTuning::Full {
                init: theta_init,
                bounds,
            },
            ThetaTuning::Fixed(_) => ThetaTuning::Fixed(theta_init),
        };
        self
    }

    /// Set theta hyper parameter search space.
    ///
    /// This function is no-op when theta tuning is fixed
    pub fn theta_bounds(mut self, theta_bounds: Array1<(F, F)>) -> Self {
        self.0.theta_tuning = match self.0.theta_tuning {
            ThetaTuning::Full { init, bounds: _ } => ThetaTuning::Full {
                init,
                bounds: theta_bounds,
            },
            ThetaTuning::Partial {
                init,
                active: _,
                bounds: _,
            } => ThetaTuning::Full {
                init,
                bounds: theta_bounds,
            },
            ThetaTuning::Fixed(f) => ThetaTuning::Fixed(f),
        };
        self
    }

    /// Set theta hyper parameter tuning
    pub fn theta_tuning(mut self, theta_tuning: ThetaTuning<F>) -> Self {
        self.0.theta_tuning = theta_tuning;
        self
    }

    /// Set the number of internal GP hyperparameter theta optimization restarts
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.0.n_start = n_start;
        self
    }

    /// Set the max number of internal likelihood evaluations during one optimization
    /// Given max_eval has to be greater than [crate::GP_COBYLA_MIN_EVAL] otherwise
    /// max_eval is set to [crate::GP_COBYLA_MAX_EVAL].
    pub fn max_eval(mut self, max_eval: usize) -> Self {
        self.0.max_eval = GP_COBYLA_MIN_EVAL.max(max_eval);
        self
    }

    /// Set nugget.
    ///
    /// Nugget is used to improve numerical stability
    pub fn nugget(mut self, nugget: F) -> Self {
        self.0.nugget = nugget;
        self
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>>
    From<GpValidParams<F, Mean, Corr>> for GpParams<F, Mean, Corr>
{
    fn from(valid: GpValidParams<F, Mean, Corr>) -> Self {
        GpParams(valid.clone())
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> ParamGuard
    for GpParams<F, Mean, Corr>
{
    type Checked = GpValidParams<F, Mean, Corr>;
    type Error = GpError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if let Some(d) = self.0.kpls_dim {
            if d == 0 {
                return Err(GpError::InvalidValueError(
                    "`kpls_dim` canot be 0!".to_string(),
                ));
            }
            let theta = self.0.theta_tuning().init();
            if theta.len() > 1 && d > theta.len() {
                return Err(GpError::InvalidValueError(format!(
                    "Dimension reduction ({}) should be smaller than expected
                        training input size infered from given initial theta length ({})",
                    d,
                    theta.len()
                )));
            };
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

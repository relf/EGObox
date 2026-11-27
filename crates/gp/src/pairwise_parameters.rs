use crate::correlation_models::{CorrelationModel, SquaredExponentialCorr};
use crate::errors::{GpError, Result};
use crate::{Cancellation, GP_COBYLA_MAX_EVAL, GP_COBYLA_MIN_EVAL, GP_OPTIM_N_START, ThetaTuning};
use linfa::{Float, ParamGuard};
use ndarray::{Array1, ArrayBase, Data, Ix2, array};
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

/// Human or decision-maker outcome for a pairwise query between two candidates.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum PreferenceOutcome {
    /// Strict preference: the `first` item has higher latent utility than `second`.
    #[default]
    Strict,
    /// Indifference / tie: both items have approximately equal latent utility.
    Tie,
    /// Joint rejection: neither item satisfies the decision-maker's utility threshold.
    RejectBoth,
}

/// A pairwise preference observation over training point indices `(first, second)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct PreferenceComparison {
    /// Index of the first (preferred, when [`PreferenceOutcome::Strict`]) training sample.
    pub first: usize,
    /// Index of the second (less preferred, when [`PreferenceOutcome::Strict`]) training sample.
    pub second: usize,
    /// Observed comparison outcome.
    pub outcome: PreferenceOutcome,
}

impl PreferenceComparison {
    /// Construct a strict preference `winner > loser`.
    pub fn prefer(winner: usize, loser: usize) -> Self {
        Self {
            first: winner,
            second: loser,
            outcome: PreferenceOutcome::Strict,
        }
    }

    /// Construct a tie / indifference observation `first ~= second`.
    pub fn tie(first: usize, second: usize) -> Self {
        Self {
            first,
            second,
            outcome: PreferenceOutcome::Tie,
        }
    }

    /// Construct a joint rejection observation where both `first` and `second` fall below threshold.
    pub fn reject_both(first: usize, second: usize) -> Self {
        Self {
            first,
            second,
            outcome: PreferenceOutcome::RejectBoth,
        }
    }

    /// Convert a `(m, 2)` array of `[winner_idx, loser_idx]` rows into strict [`PreferenceComparison`]s.
    pub fn from_pairs(pairs: &ArrayBase<impl Data<Elem = usize>, Ix2>) -> Result<Vec<Self>> {
        if pairs.ncols() != 2 {
            return Err(GpError::InvalidValueError(format!(
                "Comparison pairs array must have shape (m, 2), got ({}, {})",
                pairs.nrows(),
                pairs.ncols()
            )));
        }
        Ok(pairs
            .rows()
            .into_iter()
            .map(|row| Self::prefer(row[0], row[1]))
            .collect())
    }
}

/// Observation likelihood family for preferential Gaussian Process inference.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum PairwiseLikelihood {
    /// Thurstone-Mosteller probit likelihood `Phi((u(x_a) - u(x_b)) / sqrt(2))` (Chu & Ghahramani, 2005).
    #[default]
    Probit,
    /// Bradley-Terry logistic likelihood `1 / (1 + exp(-(u(x_a) - u(x_b))))`.
    Logit,
}

/// Validated hyperparameters for [`PairwiseGaussianProcess`](crate::PairwiseGaussianProcess).
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "F: Serialize, Corr: Serialize",
        deserialize = "F: Deserialize<'de>, Corr: Deserialize<'de>"
    ))
)]
pub struct PairwiseValidParams<F: Float, Corr: CorrelationModel<F>> {
    pub(crate) corr: Corr,
    pub(crate) theta_tuning: ThetaTuning<F>,
    pub(crate) likelihood: PairwiseLikelihood,
    pub(crate) n_start: usize,
    pub(crate) max_eval: usize,
    pub(crate) nugget: F,
    pub(crate) tie_variance: F,
    pub(crate) reject_threshold: F,
    #[cfg_attr(feature = "serializable", serde(skip))]
    pub(crate) cancellation: Cancellation,
}

impl<F: Float, Corr: CorrelationModel<F>> Default for PairwiseValidParams<F, Corr> {
    fn default() -> Self {
        Self {
            corr: Corr::default(),
            theta_tuning: ThetaTuning::Full {
                init: array![F::cast(0.5)],
                bounds: array![(F::cast(1e-2), F::cast(1e1))],
            },
            likelihood: PairwiseLikelihood::default(),
            n_start: GP_OPTIM_N_START,
            max_eval: GP_COBYLA_MAX_EVAL,
            nugget: F::cast(1e-5),
            tie_variance: F::cast(0.15),
            reject_threshold: F::cast(-0.75),
            cancellation: Cancellation::default(),
        }
    }
}

impl<F: Float, Corr: CorrelationModel<F>> PairwiseValidParams<F, Corr> {
    /// Correlation kernel `k(x, x')`.
    pub fn corr(&self) -> &Corr {
        &self.corr
    }

    /// Autocorrelation lengthscale parameter tuning configuration.
    pub fn theta_tuning(&self) -> &ThetaTuning<F> {
        &self.theta_tuning
    }

    /// Pairwise choice likelihood family (`Probit` or `Logit`).
    pub fn likelihood(&self) -> PairwiseLikelihood {
        self.likelihood
    }

    /// Number of multistart restarts for marginal Laplace evidence optimization.
    pub fn n_start(&self) -> usize {
        self.n_start
    }

    /// Maximum COBYLA evaluations per restart during hyperparameter optimization.
    pub fn max_eval(&self) -> usize {
        self.max_eval
    }

    /// Diagonal nugget added to kernel matrices for numerical stability.
    pub fn nugget(&self) -> F {
        self.nugget
    }

    /// Observation noise variance used for [`PreferenceOutcome::Tie`] observations.
    pub fn tie_variance(&self) -> F {
        self.tie_variance
    }

    /// Latent utility upper-bound threshold used for [`PreferenceOutcome::RejectBoth`] observations.
    pub fn reject_threshold(&self) -> F {
        self.reject_threshold
    }

    /// Cooperative cancellation token for hyperparameter optimization.
    pub fn cancellation(&self) -> &Cancellation {
        &self.cancellation
    }
}

/// Hyperparameter builder for [`PairwiseGaussianProcess`](crate::PairwiseGaussianProcess).
#[derive(Clone, Debug)]
pub struct PairwiseGpParams<F: Float, Corr: CorrelationModel<F>>(PairwiseValidParams<F, Corr>);

/// Preferential Kriging shortcut using a squared-exponential correlation kernel.
pub type PreferentialKriging<F> = PairwiseGpParams<F, SquaredExponentialCorr>;

impl<F: Float> PreferentialKriging<F> {
    /// Construct default [`PairwiseGpParams`] with a [`SquaredExponentialCorr`] kernel.
    pub fn params() -> PairwiseGpParams<F, SquaredExponentialCorr> {
        PairwiseGpParams::new(SquaredExponentialCorr())
    }
}

impl<F: Float, Corr: CorrelationModel<F>> Default for PairwiseGpParams<F, Corr> {
    fn default() -> Self {
        Self(PairwiseValidParams::default())
    }
}

impl<F: Float, Corr: CorrelationModel<F>> PairwiseGpParams<F, Corr> {
    /// Create a new preferential GP parameter builder for the given correlation kernel.
    pub fn new(corr: Corr) -> Self {
        Self(PairwiseValidParams {
            corr,
            ..Default::default()
        })
    }

    /// Construct builder from already validated parameters.
    pub fn new_from_valid(params: &PairwiseValidParams<F, Corr>) -> Self {
        Self(params.clone())
    }

    /// Set the correlation kernel.
    pub fn corr(mut self, corr: Corr) -> Self {
        self.0.corr = corr;
        self
    }

    /// Set the pairwise observation likelihood (`Probit` or `Logit`).
    pub fn likelihood(mut self, likelihood: PairwiseLikelihood) -> Self {
        self.0.likelihood = likelihood;
        self
    }

    /// Set initial `theta` value for lengthscale optimization (or constant value when fixed).
    pub fn theta_init(mut self, theta_init: Array1<F>) -> Self {
        self.0.theta_tuning = match self.0.theta_tuning {
            ThetaTuning::Full { init: _, bounds } => ThetaTuning::Full {
                init: theta_init,
                bounds,
            },
            ThetaTuning::Partial {
                init: _,
                bounds,
                active,
            } => ThetaTuning::Partial {
                init: theta_init,
                bounds,
                active,
            },
            ThetaTuning::Fixed(_) => ThetaTuning::Fixed(theta_init),
        };
        self
    }

    /// Set `(lower, upper)` bounds for `theta` optimization.
    pub fn theta_bounds(mut self, theta_bounds: Array1<(F, F)>) -> Self {
        self.0.theta_tuning = match self.0.theta_tuning {
            ThetaTuning::Full { init, bounds: _ } => ThetaTuning::Full {
                init,
                bounds: theta_bounds,
            },
            ThetaTuning::Partial {
                init,
                bounds: _,
                active,
            } => ThetaTuning::Partial {
                init,
                bounds: theta_bounds,
                active,
            },
            ThetaTuning::Fixed(init) => ThetaTuning::Full {
                init,
                bounds: theta_bounds,
            },
        };
        self
    }

    /// Set explicit [`ThetaTuning`] strategy (`Fixed`, `Full`, or `Partial`).
    pub fn theta_tuning(mut self, theta_tuning: ThetaTuning<F>) -> Self {
        self.0.theta_tuning = theta_tuning;
        self
    }

    /// Set number of multistart restarts during Laplace marginal likelihood optimization.
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.0.n_start = n_start;
        self
    }

    /// Set maximum number of COBYLA evaluations per multistart run.
    pub fn max_eval(mut self, max_eval: usize) -> Self {
        self.0.max_eval = GP_COBYLA_MIN_EVAL.max(max_eval);
        self
    }

    /// Set diagonal nugget for kernel positive-definiteness.
    pub fn nugget(mut self, nugget: F) -> Self {
        self.0.nugget = nugget;
        self
    }

    /// Set Gaussian observation variance for [`PreferenceOutcome::Tie`] feedback.
    pub fn tie_variance(mut self, tie_variance: F) -> Self {
        self.0.tie_variance = tie_variance;
        self
    }

    /// Set upper-bound utility threshold for [`PreferenceOutcome::RejectBoth`] feedback.
    pub fn reject_threshold(mut self, reject_threshold: F) -> Self {
        self.0.reject_threshold = reject_threshold;
        self
    }

    /// Attach a [`Cancellation`] token for cooperative early stopping.
    pub fn cancellation(mut self, cancellation: Cancellation) -> Self {
        self.0.cancellation = cancellation;
        self
    }
}

impl<F: Float, Corr: CorrelationModel<F>> ParamGuard for PairwiseGpParams<F, Corr> {
    type Checked = PairwiseValidParams<F, Corr>;
    type Error = GpError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if let Some(bounds) = self.0.theta_tuning.bounds() {
            for &(lo, up) in bounds.iter() {
                if lo <= F::zero() || lo > up {
                    return Err(GpError::InvalidValueError(format!(
                        "Invalid theta bounds ({lo:?}, {up:?}): expected 0 < lower <= upper"
                    )));
                }
            }
        }
        for &val in self.0.theta_tuning.init().iter() {
            if val <= F::zero() {
                return Err(GpError::InvalidValueError(format!(
                    "Initial theta values must be strictly positive, got {val:?}"
                )));
            }
        }
        if self.0.nugget < F::zero() {
            return Err(GpError::InvalidValueError(
                "Nugget must be non-negative".to_string(),
            ));
        }
        if self.0.tie_variance <= F::zero() {
            return Err(GpError::InvalidValueError(
                "Tie variance must be strictly positive".to_string(),
            ));
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

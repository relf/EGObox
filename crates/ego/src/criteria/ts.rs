use crate::criteria::{InfillCriterion, PosteriorMean};
use egobox_moe::MixtureGpSurrogate;
use ndarray::Array1;

use serde::{Deserialize, Serialize};

/// Thompson Sampling (TS) infill strategy, exact on a discrete candidate set.
///
/// At each iteration, `Egor` draws joint samples of the objective surrogate posterior on
/// a candidate set (a LHS of the domain completed with points around the current best point,
/// see [`ThompsonSampling::candidates`]), the minimizer of each sample being a candidate
/// point. The posterior covariance of the candidate set is factorized once per iteration,
/// samples are drawn from the optimizer random generator, so that runs are reproducible given
/// a seed and consistent with hot start. Several samples are drawn per iteration
/// (see [`ThompsonSampling::draws`]) and the points to evaluate (as many as the qEI batch
/// size) are selected in draw order, skipping candidates close to a previous one or to an
/// evaluated point.
///
/// Constraint surrogates are sampled as well (Scalable Constrained Bayesian Optimization rule):
/// the candidate minimizes the objective sample among the points feasible for the constraint
/// samples, or minimizes the total sampled violation when none is feasible. Points violating
/// function constraints are removed from the candidate set.
///
/// TS follows the GP posterior belief: with few points, a GP confident that unexplored
/// regions are high keeps sampling the current basin where EI would explore, hence an
/// initial DOE spread over the domain is recommended.
///
/// Requires single-cluster surrogates (joint posterior of one GP). Unlike the other criteria,
/// no infill criterion optimization is run: this criterion value is the opposite of the
/// posterior mean, only used by the scaling computations common to all criteria.
///
/// [`crate::InfillStrategy::TS`] uses the default settings, other ones are given with
/// [`crate::EgorConfig::infill_criterion`]:
///
/// ```no_run
/// # use egobox_ego::{EgorConfig, criteria::ThompsonSampling};
/// let config = EgorConfig::default()
///     .infill_criterion(Box::new(ThompsonSampling::default().draws(20).candidates(500)));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThompsonSampling {
    draws: Option<usize>,
    candidates: Option<usize>,
}

/// Thompson Sampling infill strategy with default settings
pub const TS: ThompsonSampling = ThompsonSampling {
    draws: None,
    candidates: None,
};

/// Default minimum number of Thompson samples per iteration
const DEFAULT_MIN_DRAWS: usize = 10;
/// Default number of Thompson samples per point of the batch
const DEFAULT_DRAWS_PER_POINT: usize = 4;

impl ThompsonSampling {
    /// Sets the number of posterior samples drawn per iteration, at least the qEI batch size,
    /// `max(10, 4 * batch)` by default. The minimizer of each sample is a candidate point,
    /// then the points to evaluate (as many as the qEI batch size) are selected from these
    /// candidates in draw order: one per cluster of candidates first, then the next ones not
    /// duplicating a selected or evaluated point. The first usable sample is always selected,
    /// the extra samples replace the ones landing on evaluated points or on the same point as
    /// a previous sample, and diversify batches. Samples are cheap: the posterior covariance
    /// is factorized once per iteration.
    pub fn draws(mut self, n: usize) -> Self {
        self.draws = Some(n);
        self
    }

    /// Sets the number of global candidate points the samples are drawn on,
    /// `min(100 * nx, 1000)` by default: a LHS of the domain, completed with 20% of points
    /// around the current best point. The posterior covariance of these points is computed
    /// and factorized at each iteration (cubic cost).
    pub fn candidates(mut self, n: usize) -> Self {
        self.candidates = Some(n);
        self
    }

    /// Number of samples drawn per iteration for a batch of `batch` points
    pub(crate) fn n_draws(&self, batch: usize) -> usize {
        self.draws
            .unwrap_or((DEFAULT_DRAWS_PER_POINT * batch.max(1)).max(DEFAULT_MIN_DRAWS))
    }

    /// Number of global candidate points for `nx` input dimensions
    pub(crate) fn n_candidates(&self, nx: usize) -> usize {
        self.candidates
            .unwrap_or_else(|| crate::solver::default_ts_candidates(nx))
    }
}

#[typetag::serde]
impl InfillCriterion for ThompsonSampling {
    fn name(&self) -> &'static str {
        "TS"
    }

    fn thompson_sampling(&self) -> Option<&ThompsonSampling> {
        Some(self)
    }

    /// Opposite of the posterior mean
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        viability_model: Option<&dyn MixtureGpSurrogate>,
        alpha: Option<f64>,
        sigma_weight: Option<f64>,
        scale: Option<f64>,
    ) -> f64 {
        PosteriorMean.value(
            x,
            obj_model,
            fmin,
            viability_model,
            alpha,
            sigma_weight,
            scale,
        )
    }

    /// Derivatives wrt x components of the opposite of the posterior mean
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        viability_model: Option<&dyn MixtureGpSurrogate>,
        alpha: Option<f64>,
        sigma_weight: Option<f64>,
        scale: Option<f64>,
    ) -> Array1<f64> {
        PosteriorMean.grad(
            x,
            obj_model,
            fmin,
            viability_model,
            alpha,
            sigma_weight,
            scale,
        )
    }
}

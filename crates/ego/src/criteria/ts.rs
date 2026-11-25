use crate::criteria::{InfillCriterion, PosteriorMean};
use egobox_moe::MixtureGpSurrogate;
use ndarray::Array1;

use serde::{Deserialize, Serialize};

/// Thompson Sampling (TS) infill strategy, exact on a discrete candidate set.
///
/// At each iteration, `Egor` draws joint samples of the objective surrogate posterior on
/// a candidate set (a LHS of the domain completed with points around the current best point,
/// see [`crate::EgorConfig::ts_candidates`]), the minimizer of each sample being a candidate
/// point. The posterior covariance of the candidate set is factorized once per iteration,
/// samples are drawn from the optimizer random generator, so that runs are reproducible given
/// a seed and consistent with hot start. Several samples are drawn per iteration
/// (see [`crate::EgorConfig::ts_draws`]) and the points to evaluate (as many as the qEI batch
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
#[derive(Clone, Serialize, Deserialize)]
pub struct ThompsonSampling;

/// Thompson Sampling infill strategy
pub const TS: ThompsonSampling = ThompsonSampling;

#[typetag::serde]
impl InfillCriterion for ThompsonSampling {
    fn name(&self) -> &'static str {
        "TS"
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

//! Expert candidates of a mixture: one per (regression, correlation) pair.
//!
//! A cluster expert is chosen among the candidates allowed by the
//! [RegressionSpec] and [CorrelationSpec] of the mixture, using the
//! cross-validation error of each candidate.

use crate::surrogates::*;
use crate::types::{CorrelationSpec, RegressionSpec};
use egobox_gp::{
    GaussianProcess, Inducings, ParamTuning, SparseGaussianProcess, correlation_models::*,
    mean_models::*,
};
use linfa::Dataset;
use linfa::dataset::Records;
use log::{debug, trace};
use ndarray::{Axis, Ix1};
use paste::paste;
use std::fmt;

#[cfg(not(feature = "blas"))]
use linfa_linalg::norm::*;
#[cfg(feature = "blas")]
use ndarray_linalg::Norm;

/// Regression model of an expert
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Regression {
    Constant,
    Linear,
    Quadratic,
}

impl Regression {
    /// All models, in candidate order
    const ALL: [Regression; 3] = [Self::Constant, Self::Linear, Self::Quadratic];

    fn spec(self) -> RegressionSpec {
        match self {
            Self::Constant => RegressionSpec::CONSTANT,
            Self::Linear => RegressionSpec::LINEAR,
            Self::Quadratic => RegressionSpec::QUADRATIC,
        }
    }
}

/// Correlation model of an expert
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Correlation {
    SquaredExponential,
    AbsoluteExponential,
    Matern32,
    Matern52,
}

impl Correlation {
    /// All models, in candidate order
    const ALL: [Correlation; 4] = [
        Self::SquaredExponential,
        Self::AbsoluteExponential,
        Self::Matern32,
        Self::Matern52,
    ];

    fn spec(self) -> CorrelationSpec {
        match self {
            Self::SquaredExponential => CorrelationSpec::SQUAREDEXPONENTIAL,
            Self::AbsoluteExponential => CorrelationSpec::ABSOLUTEEXPONENTIAL,
            Self::Matern32 => CorrelationSpec::MATERN32,
            Self::Matern52 => CorrelationSpec::MATERN52,
        }
    }
}

/// An expert candidate: a regression and a correlation model
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Expert {
    pub regr: Regression,
    pub corr: Correlation,
}

/// Name used in logs and errors, e.g. `Linear_Matern32`
impl fmt::Display for Expert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}_{:?}", self.regr, self.corr)
    }
}

/// The experts allowed by the given specs, regression models first then
/// correlation models, each in [Regression::ALL] and [Correlation::ALL] order.
pub(crate) fn allowed_experts(regr: RegressionSpec, corr: CorrelationSpec) -> Vec<Expert> {
    Regression::ALL
        .into_iter()
        .filter(|r| regr.contains(r.spec()))
        .flat_map(|r| {
            Correlation::ALL
                .into_iter()
                .filter(|c| corr.contains(c.spec()))
                .map(move |c| Expert { regr: r, corr: c })
        })
        .collect()
}

/// Default GP parameters of the given expert
pub(crate) fn gp_expert_params(expert: Expert) -> Box<dyn GpSurrogateParams> {
    // Each arm names the models once, so the pattern and the built type agree
    macro_rules! gp_params {
        ($(($regr:ident, $corr:ident)),+ $(,)?) => {
            match (expert.regr, expert.corr) {
                $(
                    // block tail: the macro output carries an attribute
                    (Regression::$regr, Correlation::$corr) => {
                        make_surrogate_params!($regr, $corr)
                    }
                )+
            }
        };
    }
    gp_params!(
        (Constant, SquaredExponential),
        (Constant, AbsoluteExponential),
        (Constant, Matern32),
        (Constant, Matern52),
        (Linear, SquaredExponential),
        (Linear, AbsoluteExponential),
        (Linear, Matern32),
        (Linear, Matern52),
        (Quadratic, SquaredExponential),
        (Quadratic, AbsoluteExponential),
        (Quadratic, Matern32),
        (Quadratic, Matern52),
    )
}

/// Default sparse GP parameters of the given expert,
/// `None` when its regression model is not constant (unsupported by sparse GPs)
pub(crate) fn sgp_expert_params(
    expert: Expert,
    inducings: Inducings<f64>,
) -> Option<Box<dyn SgpSurrogateParams>> {
    macro_rules! sgp_params {
        ($($corr:ident),+ $(,)?) => {
            match (expert.regr, expert.corr) {
                $(
                    (Regression::Constant, Correlation::$corr) => {
                        Some(make_sgp_surrogate_params!($corr, inducings))
                    }
                )+
                (Regression::Linear | Regression::Quadratic, _) => None,
            }
        };
    }
    sgp_params!(SquaredExponential, AbsoluteExponential, Matern32, Matern52)
}

/// Cross-validation error of the expert (mean of the fold errors) on the dataset.
///
/// The error is infinite when the dataset is too small for the regression model,
/// or when a fold GP cannot be trained.
pub(crate) fn cross_validation_error(
    expert: Expert,
    kpls_dim: Option<usize>,
    noise_ratio: &ParamTuning<f64>,
    dataset: &mut Dataset<f64, f64, Ix1>,
) -> f64 {
    debug!(
        "Surrogate {expert} on dataset size = {}",
        dataset.nsamples()
    );
    let mut params = gp_expert_params(expert);
    params.kpls_dim(kpls_dim);
    params.noise_ratio(noise_ratio.clone());

    let input_dim = dataset.records().ncols();
    let n_fold = std::cmp::min(dataset.nsamples(), 5);
    trace!("Cross validation N fold = {n_fold}");
    let not_enough_points = match expert.regr {
        Regression::Constant => false,
        Regression::Linear => dataset.nsamples() < 3 * input_dim,
        Regression::Quadratic => dataset.nsamples() < 4 * input_dim,
    };
    if not_enough_points {
        return f64::INFINITY; // not enough points => huge error
    }

    let mut errors = Vec::new();
    for (gp, valid) in dataset.iter_fold(n_fold, |train| {
        params.train(
            &train.records().view(),
            &train.targets().view().insert_axis(Axis(1)),
        )
    }) {
        // A fit failure (e.g. not enough training points for the regression
        // terms of the fold) disqualifies the surrogate => huge error
        let error = match gp.and_then(|gp| gp.predict(&valid.records().view())) {
            Ok(pred) => (valid.targets() - pred).norm_l2(),
            Err(err) => {
                trace!("GP training failed: {err}");
                f64::INFINITY
            }
        };
        trace!("Prediction error = {error}");
        errors.push(error);
    }
    let mean_err = errors.iter().sum::<f64>() / errors.len() as f64;
    trace!("-> mean error = {mean_err}");
    mean_err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allowed_experts_order() {
        let experts = allowed_experts(
            RegressionSpec::CONSTANT | RegressionSpec::QUADRATIC,
            CorrelationSpec::MATERN52 | CorrelationSpec::SQUAREDEXPONENTIAL,
        );
        let names: Vec<String> = experts.iter().map(|e| e.to_string()).collect();
        assert_eq!(
            names,
            [
                "Constant_SquaredExponential",
                "Constant_Matern52",
                "Quadratic_SquaredExponential",
                "Quadratic_Matern52"
            ]
        );
        assert_eq!(
            allowed_experts(RegressionSpec::ALL, CorrelationSpec::ALL).len(),
            12
        );
    }

    #[test]
    fn test_sgp_expert_params_constant_only() {
        for expert in allowed_experts(RegressionSpec::ALL, CorrelationSpec::ALL) {
            let params = sgp_expert_params(expert, Inducings::default());
            assert_eq!(
                params.is_some(),
                expert.regr == Regression::Constant,
                "{expert}"
            );
        }
    }
}

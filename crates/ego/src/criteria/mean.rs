use crate::criteria::InfillCriterion;
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView};

use serde::{Deserialize, Serialize};

/// Posterior mean criterion: maximizing it minimizes the posterior mean of the objective
/// surrogate, i.e. pure exploitation of the surrogate (used by TREGO local steps,
/// see [`crate::TregoStrategy::local_mean`])
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct PosteriorMean;

#[typetag::serde]
impl InfillCriterion for PosteriorMean {
    fn name(&self) -> &'static str {
        "PosteriorMean"
    }

    /// Opposite of the posterior mean of `obj_model` at given `x` point
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
        _viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        _sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> f64 {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        obj_model
            .predict(&pt)
            .map_or(f64::NEG_INFINITY, |mu| -mu[0])
    }

    /// Derivatives wrt x components of the opposite of the posterior mean
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
        _viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        _sigma_weight: Option<f64>,
        _scale: Option<f64>,
    ) -> Array1<f64> {
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        obj_model
            .predict_gradients(&pt)
            .map_or(Array1::zeros(x.len()), |grad| -grad.row(0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_doe::{Lhs, SamplingMethod};
    use egobox_moe::GpMixture;
    use linfa::prelude::{Dataset, Fit};
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_posterior_mean_value_and_grad() {
        let xt = Lhs::new(&array![[-2., 2.], [-2., 2.]])
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(15);
        let yt = xt.map_axis(ndarray::Axis(1), |x| x[0] * x[0] + f64::sin(x[1]));
        let model = GpMixture::params()
            .fit(&Dataset::new(xt, yt))
            .expect("GP fitted");

        let x = [0.3, -0.7];
        let mu = model.predict(&array![[0.3, -0.7]].view()).unwrap()[0];
        let value = PosteriorMean.value(&x, &model, 0., None, None, None, None);
        assert_abs_diff_eq!(value, -mu);

        let grad = PosteriorMean.grad(&x, &model, 0., None, None, None, None);
        let e = 1e-6;
        for k in 0..2 {
            let (mut xp, mut xm) = (x, x);
            xp[k] += e;
            xm[k] -= e;
            let fd = (PosteriorMean.value(&xp, &model, 0., None, None, None, None)
                - PosteriorMean.value(&xm, &model, 0., None, None, None, None))
                / (2. * e);
            assert_abs_diff_eq!(grad[k], fd, epsilon = 1e-4 * (1. + fd.abs()));
        }
    }
}

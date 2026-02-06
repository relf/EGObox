//! A module for metrics to evaluate Gaussian Process models performances.
//!
//! It implements metrics from the following paper:
//!
//! Marrel, Amandine, and Bertrand Iooss.
//! "Probabilistic surrogate modeling by Gaussian process: A review on recent insights in estimation and validation."
//! Reliability Engineering & System Safety 247 (2024): 110094.

use linfa::dataset::Dataset;
use linfa::{
    Float, ParamGuard,
    traits::{Fit, Predict, PredictInplace},
};
use ndarray::{Array1, Array2};

use crate::{
    GaussianProcess, GpError, GpParams, SgpParams, SparseGaussianProcess, correlation_models,
    mean_models,
};

/// A trait for Q2 predictive coefficient cross validation score
pub trait PredictScore<F, ER, P, O>
where
    F: Float,
    ER: std::error::Error + From<linfa::error::Error>,
    P: Fit<Array2<F>, Array1<F>, ER, Object = O> + ParamGuard,
    O: PredictInplace<Array2<F>, Array1<F>>,
{
    /// Return the training data (xt, yt)
    fn training_data(&self) -> &(Array2<F>, Array1<F>);

    /// Return the model parameters
    fn params(&self) -> P;

    /// Compute quality metric Q2 with kfold cross validation
    fn q2_score(&self, kfold: usize) -> F {
        let (xt, yt) = self.training_data();
        let dataset = Dataset::new(xt.to_owned(), yt.to_owned());
        let yt_mean = yt.mean().unwrap();
        // Predictive Residual Sum of Squares
        let mut press = F::zero();
        // Total Sum of Squares
        let mut tss = F::zero();
        for (train, valid) in dataset.fold(kfold).into_iter() {
            let params = self.params();
            let model: O = params
                .fit(&train)
                .expect("cross-validation: sub model fitted");
            let pred = model.predict(valid.records());
            press += (valid.targets() - pred).mapv(|v| v * v).sum();
            tss += (valid.targets() - yt_mean).mapv(|v| v * v).sum();
        }
        F::one() - press / tss
    }

    /// Q2 predictive coefficient with Leave-One-Out Cross-Validation
    fn looq2_score(&self) -> F {
        self.q2_score(self.training_data().0.nrows())
    }
}

impl<F, Mean, Corr> PredictScore<F, GpError, GpParams<F, Mean, Corr>, Self>
    for GaussianProcess<F, Mean, Corr>
where
    F: Float,
    Mean: mean_models::RegressionModel<F>,
    Corr: correlation_models::CorrelationModel<F>,
{
    fn training_data(&self) -> &(Array2<F>, Array1<F>) {
        &self.training_data
    }

    fn params(&self) -> GpParams<F, Mean, Corr> {
        GpParams::from(self.params.clone())
    }
}

impl<F, Corr> PredictScore<F, GpError, SgpParams<F, Corr>, Self> for SparseGaussianProcess<F, Corr>
where
    F: Float,
    Corr: correlation_models::CorrelationModel<F>,
{
    fn training_data(&self) -> &(Array2<F>, Array1<F>) {
        &self.training_data
    }

    fn params(&self) -> SgpParams<F, Corr> {
        SgpParams::from(self.params.clone())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Inducings, SparseKriging};
    use approx::assert_abs_diff_eq;
    use egobox_doe::{Lhs, SamplingMethod};
    use ndarray::{Array, Array1, ArrayBase, Axis, Data, Ix2, Zip, array};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use rand_xoshiro::Xoshiro256Plus;

    fn griewank(x: &Array2<f64>) -> Array1<f64> {
        let dim = x.ncols();
        let d = Array1::linspace(1., dim as f64, dim).mapv(|v| v.sqrt());
        let mut y = Array1::zeros((x.nrows(),));
        Zip::from(&mut y).and(x.rows()).for_each(|y, x| {
            let s = x.mapv(|v| v * v).sum() / 4000.;
            let p = (x.to_owned() / &d)
                .mapv(|v| v.cos())
                .fold(1., |acc, x| acc * x);
            *y = s - p + 1.;
        });
        y
    }

    #[test]
    fn test_q2_gp_griewank() {
        let dims = [5]; // , 10, 60];
        let nts = [100]; // , 300, 500];
        let lim = array![[-600., 600.]];

        (0..dims.len()).for_each(|i| {
            let dim = dims[i];
            let nt = nts[i];
            let xlimits = lim.broadcast((dim, 2)).unwrap();

            let rng = Xoshiro256Plus::seed_from_u64(42);
            let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
            let yt = griewank(&xt);

            let gp = GaussianProcess::<
                f64,
                mean_models::ConstantMean,
                correlation_models::SquaredExponentialCorr,
            >::params(
                mean_models::ConstantMean::default(),
                correlation_models::SquaredExponentialCorr::default(),
            )
            .kpls_dim(Some(3))
            .fit(&Dataset::new(xt, yt))
            .expect("GP fit error");

            assert_abs_diff_eq!(gp.looq2_score(), 1., epsilon = 1e-2);
            assert_abs_diff_eq!(gp.q2_score(10), 1., epsilon = 1e-2);
        });
    }

    const PI: f64 = std::f64::consts::PI;

    fn f_obj(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        x.mapv(|v| (3. * PI * v).sin() + 0.3 * (9. * PI * v).cos() + 0.5 * (7. * PI * v).sin())
    }

    fn make_test_data(
        nt: usize,
        eta2: f64,
        rng: &mut Xoshiro256Plus,
    ) -> (Array2<f64>, Array1<f64>) {
        let normal = Normal::new(0., eta2.sqrt()).unwrap();
        let gaussian_noise = Array::<f64, _>::random_using((nt, 1), normal, rng);
        let xt = 2. * Array::<f64, _>::random_using((nt, 1), Uniform::new(0., 1.), rng) - 1.;
        let yt = (f_obj(&xt) + gaussian_noise).remove_axis(Axis(1));
        (xt, yt)
    }

    #[test]
    fn test_q2_sgp() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        // Generate training data
        let nt = 200;
        // Variance of the gaussian noise on our training data
        let eta2: f64 = 0.01;
        let (xt, yt) = make_test_data(nt, eta2, &mut rng);
        let n_inducings = 30;
        let sgp = SparseKriging::params(Inducings::Randomized(n_inducings))
            .seed(Some(42))
            .fit(&Dataset::new(xt.clone(), yt.clone()))
            .expect("GP fitted");

        assert_abs_diff_eq!(sgp.looq2_score(), 1., epsilon = 2e-2);
        assert_abs_diff_eq!(sgp.q2_score(10), 1., epsilon = 2e-2);
    }
}

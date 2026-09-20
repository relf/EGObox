use linfa::dataset::Dataset;
use linfa::prelude::Records;
use linfa::{ParamGuard, traits::Fit};
use ndarray::{Array1, Array2, ArrayView2, Ix1, Zip};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::GpSurrogate;

/// Data structure to hold data for IAE alpha plot
#[derive(Clone, Debug, Default)]
pub struct IaeAlphaPlotData {
    /// Alpha values (e.g. 0.02, 0.05, ..., 0.98)
    pub alphas: Vec<f64>,
    /// Empirical coverage for each alpha
    pub deltas: Vec<f64>,
}

/// A trait for cross validation score
pub trait GpMetrics<ER, P, O>
where
    ER: std::error::Error + From<linfa::error::Error>,
    P: Fit<Array2<f64>, Array1<f64>, ER, Object = O> + ParamGuard,
    O: GpSurrogate,
{
    /// Return the training data (xt, yt)
    fn training_data(&self) -> &(Array2<f64>, Array1<f64>);

    /// Return the model parameters
    fn params(&self) -> P;

    /// Compute quality metric Q2 with kfold cross validation
    fn q2_k_score(&self, kfold: usize) -> f64 {
        let (xt, yt) = self.training_data();
        let dataset = Dataset::new(xt.to_owned(), yt.to_owned());
        let yt_mean = yt.mean().unwrap();
        // Predictive Residual Sum of Squares
        let mut press = 0.;
        // Total Sum of Squares
        let mut tss = 0.;
        for (train, valid) in dataset.fold(kfold).into_iter() {
            let params = self.params();
            let model: O = params
                .fit(&train)
                .expect("cross-validation: sub model fitted");
            let pred = model.predict(&valid.records().view()).unwrap();
            press += (valid.targets() - pred).mapv(|v| v * v).sum();
            tss += (valid.targets() - yt_mean).mapv(|v| v * v).sum();
        }
        1. - press / tss
    }

    /// Q2 predictive coefficient with Leave-One-Out Cross-Validation
    fn q2_score(&self) -> f64 {
        self.q2_k_score(self.training_data().0.nrows())
    }

    /// Predictive variance adequacy
    fn pva_k_score(&self, kfold: usize) -> f64 {
        let (xt, yt) = self.training_data();
        let dataset = Dataset::new(xt.to_owned(), yt.to_owned());
        // Total Sum of Squares
        let mut varss = 0.;
        // Number of fold
        let mut n = 0usize;
        for (train, valid) in dataset.fold(kfold).into_iter() {
            let params = self.params();
            let model: O = params
                .fit(&train)
                .expect("cross-validation: sub model fitted");
            let (pred, var) = model.predict_valvar(&valid.records().view()).unwrap();
            varss += ((valid.targets() - &pred).mapv(|v| v * v) / var).sum();
            n += valid.nsamples();
        }
        (varss / n as f64).ln().abs()
    }

    /// Predictive variance adequacy with Leave-One-Out Cross-Validation
    fn pva_score(&self) -> f64 {
        self.pva_k_score(self.training_data().0.nrows())
    }

    /// Compute integrated absolute error on alpha
    fn iae_alpha_k_score(&self, kfold: usize, plot_data: Option<&mut IaeAlphaPlotData>) -> f64 {
        let (xt, yt) = self.training_data();
        let dataset = Dataset::new(xt.to_owned(), yt.to_owned());

        let n_alpha = 20;
        let alphas = Array1::linspace(0.02, 0.98, n_alpha);

        let iaes = dataset
            .fold(kfold)
            .into_iter()
            .map(|(train, valid)| {
                let params = self.params();
                let model: O = params
                    .fit(&train)
                    .expect("cross-validation: sub model fitted");
                iae_alpha(&model, &valid, &alphas)
            })
            .collect::<Vec<_>>();

        let score = iaes.iter().fold(0., |acc, infos| acc + infos.0) / iaes.len() as f64;

        let n_iaes = iaes.len() as f64;
        let deltas = iaes
            .into_iter()
            .fold(Array1::<f64>::zeros(n_alpha), |acc, infos| acc + infos.1)
            / n_iaes;

        // println!("Alpha | Empirical coverage | Target coverage | Delta");
        // println!("---------------------------------------------------");
        // let mut score2 = 0.;
        // for i in 0..n_alpha {
        //     let alpha = alphas[i];
        //     let delta = deltas[i];
        //     score2 += (delta - (1. - alpha)).abs();
        //     println!(
        //         "{:5.2}% |       {:5.2}%      |     {:5.2}%    | {:5.2}%",
        //         alpha * 100.,
        //         delta * 100.,
        //         (1. - alpha) * 100.,
        //         (delta - (1. - alpha)).abs() * 100.
        //     );
        // }

        // let _score2 = score2 * 100. / n_alpha as f64;
        //println!("IAE (computed) = {:.2}%", score2);

        // return data for plotting if requested
        if let Some(data) = plot_data {
            *data = IaeAlphaPlotData {
                alphas: alphas.to_vec(),
                deltas: deltas.to_vec(),
            };
        }

        score
    }

    /// Integrated absolute error on alpha with Leave-One-Out Cross-Validation
    fn iae_alpha_score(&self, plot_data: Option<&mut IaeAlphaPlotData>) -> f64 {
        self.iae_alpha_k_score(self.training_data().0.nrows(), plot_data)
    }
}

fn iae_alpha(
    model: &dyn GpSurrogate,
    valid: &linfa::Dataset<f64, f64, Ix1>,
    alphas: &Array1<f64>,
) -> (f64, Array1<f64>) {
    let x = valid.records();
    let y = valid.targets();

    let (pred, var) = model.predict_valvar(&x.view()).unwrap();
    let sigma = var.sqrt();

    let n_test = x.nrows();
    let n_alpha = alphas.len();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let norm_ppf: Vec<f64> = alphas
        .iter()
        .map(|&a| normal.inverse_cdf(1.0 - a / 2.0))
        .collect();

    let mut ci_inf = Array2::<f64>::zeros((n_test, n_alpha));
    let mut ci_sup = Array2::<f64>::zeros((n_test, n_alpha));

    for (i_alpha, &q) in norm_ppf.iter().enumerate() {
        Zip::from(&pred)
            .and(&sigma)
            .and(ci_inf.column_mut(i_alpha))
            .and(ci_sup.column_mut(i_alpha))
            .for_each(|&mean, &sigm, inf, sup| {
                let offset = sigm * q;
                *inf = mean - offset;
                *sup = mean + offset;
            });
    }

    // Empirical coverage using Zip
    let targets = y.to_owned().into_shape_with_order((n_test, 1)).unwrap();
    let targets = targets.broadcast((n_test, n_alpha)).unwrap();
    let deltas = empirical_coverage(targets.view(), ci_inf.view(), ci_sup.view());

    // Compute IAE
    let iae = Zip::from(&deltas)
        .and(alphas)
        .fold(0.0, |acc, &delta, &alpha| {
            acc + (delta - (1.0 - alpha)).abs()
        })
        / n_alpha as f64;

    (iae, deltas)
}

/// Compute empirical coverage: fraction of `y` values inside CI for each alpha.
fn empirical_coverage(
    y: ArrayView2<f64>,
    ci_alpha_inf: ArrayView2<f64>,
    ci_alpha_sup: ArrayView2<f64>,
) -> Array1<f64> {
    let (n_test, n_alpha) = ci_alpha_inf.dim();
    let mut deltas = Array1::<f64>::zeros(n_alpha);

    // For each alpha (column), count coverage using Zip over rows
    for i in 0..n_alpha {
        let mut count = 0usize;
        Zip::from(y.column(i))
            .and(ci_alpha_inf.column(i))
            .and(ci_alpha_sup.column(i))
            .for_each(|&y_val, &low, &high| {
                if y_val >= low && y_val <= high {
                    count += 1;
                }
            });
        deltas[i] = count as f64 / n_test as f64;
    }
    deltas
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::GpMixtureParams;
    use crate::MixtureGpSurrogate;
    use approx::assert_abs_diff_eq;
    use egobox_doe::{Lhs, SamplingMethod};
    use egobox_gp::ThetaTuning;
    use ndarray::{Array1, array};
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    fn x_squared(x: &Array2<f64>) -> Array1<f64> {
        x.mapv(|v| v * v).sum_axis(ndarray::Axis(1))
    }

    #[test]
    fn test_gpqa_griewank() {
        let dims = [2];
        let nts = [20];
        let lim = array![[-10., 10.]];

        (0..dims.len()).for_each(|i| {
            let dim = dims[i];
            let nt = nts[i];
            let xlimits = lim.broadcast((dim, 2)).unwrap();

            let rng = Xoshiro256Plus::seed_from_u64(42);
            let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
            let yt = x_squared(&xt);

            let moe = GpMixtureParams::default()
                .fit(&Dataset::new(xt, yt))
                .expect("GP fit error");

            assert_abs_diff_eq!(moe.q2_k_score(10), 1., epsilon = 1e-3);
            assert_abs_diff_eq!(moe.q2_score(), 1., epsilon = 1e-3);
            assert_abs_diff_eq!(moe.pva_k_score(10), 0., epsilon = 2e-1);
            assert_abs_diff_eq!(moe.pva_score(), 0., epsilon = 2e-1);
        });
    }

    fn iooss_function(x: &Array2<f64>) -> Array1<f64> {
        x.map_axis(ndarray::Axis(1), |row| {
            (row[0].exp() - row[1]) / 5.0
                + row[1].powi(6) / 3.0
                + 4.0 * (row[1].powi(4) - row[1].powi(2))
                + 7.0 * row[0].powi(2) / 10.0
                + row[0].powi(4)
                + 3.0 / (4.0 * (row[0].powi(2) + row[1].powi(2)) + 1.0)
        })
    }

    #[test]
    fn test_gpqa_iooss() {
        let xlimits = array![[-1., 1.], [-1., 1.]];
        let nts = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50];

        (0..nts.len()).for_each(|i| {
            let nt = nts[i];

            let rng = Xoshiro256Plus::seed_from_u64(42);
            let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
            let yt = iooss_function(&xt);

            let moe = GpMixtureParams::default()
                .fit(&Dataset::new(xt, yt))
                .expect("GP fit error");

            println!(
                "i={:2}: Q2 = {:.6}, PVA = {:.6}",
                i,
                moe.q2_k_score(10),
                moe.pva_k_score(10)
            );

            if i == 5 {
                assert_abs_diff_eq!(moe.q2_k_score(10), 0.9, epsilon = 1e-1);
                assert_abs_diff_eq!(moe.q2_score(), 0.95, epsilon = 2e-1);
                assert_abs_diff_eq!(moe.pva_k_score(10), 0.5, epsilon = 1e-1);
                assert_abs_diff_eq!(moe.pva_score(), 0.3, epsilon = 1e-1);
            }
        });
    }

    #[test]
    fn test_empirical_coverage_includes_interval_boundaries() {
        let targets = array![[0.], [1.], [2.], [3.]];
        let lower = array![0., 1., 4.];
        let upper = array![3., 2., 5.];
        let shape = (targets.nrows(), lower.len());

        let coverage = empirical_coverage(
            targets.broadcast(shape).unwrap(),
            lower.broadcast(shape).unwrap(),
            upper.broadcast(shape).unwrap(),
        );

        assert_eq!(coverage, array![1., 0.5, 0.]);
    }

    #[test]
    fn test_iae_alpha() {
        let xlimits = array![[-5., 10.], [0., 15.]];
        let nt = 50;

        let rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let yt = iooss_function(&xt);

        let moe = GpMixtureParams::default()
            // Keep the calibration fixture independent of hyperparameter search.
            .theta_tunings(&[ThetaTuning::Fixed(array![0.07, 0.24])])
            .fit(&Dataset::new(xt, yt))
            .expect("GP fit error");

        let iae = moe.iae_alpha_score(None);
        println!("IAE = {:.6}", iae);
        // Coverage changes discretely when numerical differences in GP fits
        // move predictions across interval boundaries.
        assert_abs_diff_eq!(iae, 0.361507, epsilon = 1e-2);
    }

    fn rescaled_branin(x: &Array2<f64>) -> Array1<f64> {
        x.map_axis(ndarray::Axis(1), |row| {
            let x1 = row[0] * 15.0 - 5.0;
            let x2 = row[1] * 15.0;
            (x2 - (5.1 / (4.0 * std::f64::consts::PI * std::f64::consts::PI)) * x1 * x1
                + (5.0 / std::f64::consts::PI) * x1
                - 6.0)
                .powi(2)
                + 10.0 * (1.0 - 1.0 / (8.0 * std::f64::consts::PI)) * (x1).cos()
                + 10.0
        }) / 51.95
    }

    #[test]
    /// Test IAE on Branin function with fixed thetas
    /// cf. Marrel et al., Probabilistic surrogate modeling by Gaussian process: a review on
    /// recent insights in estimation and validation, 2024
    fn test_iae_alpha_branin() {
        let xlimits = array![[0., 1.], [0., 1.]];
        let nt = 30;

        let rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(&xlimits).with_rng(rng.clone()).sample(nt);
        let yt = rescaled_branin(&xt);

        let moe = GpMixtureParams::default()
            .correlation_spec(crate::types::CorrelationSpec::MATERN52)
            .fit(&Dataset::new(xt.clone(), yt.clone()))
            .expect("GP fit error");

        println!(
            "trained thetas = {}",
            moe.experts().first().unwrap().theta()
        );

        let theta_tuning = ThetaTuning::Fixed(array![0.893, 1.25]);
        let moe = GpMixtureParams::default()
            .correlation_spec(crate::types::CorrelationSpec::MATERN52)
            .theta_tunings(&[theta_tuning])
            .fit(&Dataset::new(xt, yt))
            .expect("GP fit error");

        println!("fixed thetas = {}", moe.experts().first().unwrap().theta());

        let q2 = moe.q2_score();
        println!("Q2 = {:.6}", q2);

        let xt = Lhs::new(&xlimits).sample(1000);
        let yt = rescaled_branin(&xt);
        let valid = Dataset::new(xt, yt);

        let alphas = Array1::linspace(0.02, 0.98, 20);
        let iae = iae_alpha(&moe, &valid, &alphas).0;
        println!("IAE = {:.6}", iae);

        assert_abs_diff_eq!(iae, 0.1, epsilon = 5e-2);
    }
}

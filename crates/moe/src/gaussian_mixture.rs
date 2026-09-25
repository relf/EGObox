// Extract relevant part of Gaussian Mixture Model from `linfa_clustering/gmm`
// to allow the specification of the heaviside factor used to tune the
// smoothness of the mixture smooth recombination
#![allow(dead_code)]
use crate::Result;
use linfa::{Float, traits::*};
#[cfg(feature = "blas")]
use linfa::{dataset::WithLapack, dataset::WithoutLapack};
#[cfg(not(feature = "blas"))]
use linfa_linalg::{cholesky::*, triangular::*};
use ndarray::{Array, Array1, Array2, Array3, ArrayBase, Axis, Data, Ix1, Ix2, Ix3, Zip, s};
#[cfg(feature = "blas")]
use ndarray_linalg::{cholesky::*, triangular::*};
use ndarray_stats::QuantileExt;

use serde::{Deserialize, Serialize};

/// Gaussian mixture is a set of n weigthed multivariate normal distributions of dimension nx
/// This structure is derived from `linfa::GaussianMixtureModel` clustering method
/// to handle the resulting multivariate normals and related computations in one go.
/// Moreover an `heaviside factor` is handled in case of smooth Recombination
/// to control the smoothness between clusters and can be adjusted afterwards
///
/// Note: distribution means are handle in a (n, nx) matrix whie covariances
/// are handled in a (n, nx, nx) ndarray

#[derive(Debug, Serialize, Deserialize)]
pub struct GaussianMixture<F: Float> {
    /// weights vector (n,) of each cluster
    weights: Array1<F>,
    /// means (n, nx) matrix of the multivariate normal distributions
    means: Array2<F>,
    /// covariances (n, nx) ndarray of the multivariate normal distributions
    covariances: Array3<F>,
    /// precisions (n, nx) ndarray of the multivariate normal distributions
    precisions: Array3<F>,
    /// lower cholesky precisions (n, nx) ndarray of the multivariate normal distributions
    precisions_chol: Array3<F>,
    /// factor controlling the smoothness (used when smooth recombinatoin is used)
    heaviside_factor: F,
    /// determinants of the cholesky decomposition matrices of the precision matrices
    log_det: Array1<F>,
}

impl<F: Float> Clone for GaussianMixture<F> {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.to_owned(),
            means: self.means.to_owned(),
            covariances: self.covariances.to_owned(),
            precisions: self.precisions.to_owned(),
            precisions_chol: self.precisions_chol.to_owned(),
            heaviside_factor: self.heaviside_factor,
            log_det: self.log_det.to_owned(),
        }
    }
}

impl<F: Float> GaussianMixture<F> {
    /// Create a new GaussianMixture from the weights, means and covariances of the multivariate
    /// normal distributions.
    pub fn new(
        weights: Array1<F>,
        means: Array2<F>,
        covariances: Array3<F>,
    ) -> Result<GaussianMixture<F>> {
        let precisions_chol = Self::compute_precisions_cholesky(&covariances)?;
        let precisions = Self::compute_precisions(&precisions_chol);
        let log_det = Self::compute_log_det(&precisions_chol, F::one());
        Ok(GaussianMixture {
            weights,
            means,
            covariances,
            precisions,
            precisions_chol,
            heaviside_factor: F::one(),
            log_det,
        })
    }

    /// Number of clusters corresponding to the number of multivariate normal distributions
    /// used in the mixture
    pub fn n_clusters(&self) -> usize {
        self.means.nrows()
    }

    /// Return the weights of each cluster
    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    /// Return the means of each cluster (i.e. the means of each multivariate normal distribution)
    pub fn means(&self) -> &Array2<F> {
        &self.means
    }

    /// Return the covariances of each cluster (i.e. the covariances of each multivariate normal distribution)
    pub fn covariances(&self) -> &Array3<F> {
        &self.covariances
    }

    /// Setter for heaviside factor which change the transition between
    /// clusters in case of smooth recombination
    pub fn heaviside_factor(mut self, heaviside_factor: F) -> Self {
        self.heaviside_factor = heaviside_factor;
        // refresh log of precision matrix determinant
        self.log_det = Self::compute_log_det(&self.precisions_chol, self.heaviside_factor);
        self
    }

    /// Compute the probability of each n x points given as a (n, nx) matrix to belong to a given cluster.
    pub fn predict_probas<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        if self.n_clusters() == 1 {
            Array::from_elem((x.nrows(), 1), F::one())
        } else {
            let (_, log_resp) = self.compute_log_prob_resp(x);
            log_resp.mapv(|v| v.exp())
        }
    }

    /// Compute the derivatives of the probability at the x point given as a (nx,) vector
    /// to belong to a given cluster among the n clusters.
    /// Returns a (n, nx) matrix where the ith row is the derivatives wrt to the nx components valued at x
    /// of the responsability (the probability of being part of) of the ith cluster (ie the ith mvn distribution)
    pub fn predict_single_probas_derivatives<D: Data<Elem = F>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Array2<F> {
        let v = self.weights.to_owned().dot(&self.pdfs(x));
        let precs = &self.precisions / self.heaviside_factor;
        let mut deriv = Array2::zeros((self.means.nrows(), self.means.ncols()));
        Zip::from(deriv.rows_mut())
            .and(self.means.rows())
            .and(precs.outer_iter())
            .for_each(|mut der, mu, prec| {
                der.assign(&(&x.to_owned() - &mu).dot(&prec));
            });
        let vprime =
            deriv.to_owned() * &(-self.weights.to_owned() * self.pdfs(x)).insert_axis(Axis(1));
        let vprime = vprime.sum_axis(Axis(0));

        let u = (self.weights.to_owned() * self.pdfs(x))
            .to_owned()
            .insert_axis(Axis(1));
        let uprime = -(deriv.to_owned() * &u.to_owned());
        let v2 = v * v;
        (uprime.mapv(|up| up * v)
            - u.to_owned() * vprime.broadcast((u.nrows(), vprime.len())).unwrap())
        .mapv(|w| w / v2)
    }

    /// Compute the derivatives of the probability of a set of x points given as a (m, nx) vector
    /// to belong to a given cluster among the n clusters.
    /// Returns (m, n, nx) ndarray where the mth element is the derivatives wrt to x valued at x
    /// of the responsability (the probability of being part of) of the ith cluster (ie the ith mvn distribution)
    pub fn predict_probas_derivatives<D: Data<Elem = F>>(
        &self,
        x: &ArrayBase<D, Ix2>,
    ) -> Array3<F> {
        let mut prob = Array3::zeros((x.nrows(), self.means.nrows(), x.ncols()));
        Zip::from(prob.outer_iter_mut())
            .and(x.rows())
            .for_each(|mut p, xi| {
                let pred_prob = self.predict_single_probas_derivatives(&xi);
                p.assign(&pred_prob);
            });
        prob
    }

    /// Compute the density functions at x for the n multivariate normal distributions
    /// Returns the pdf values as a (n,) vector
    pub fn pdfs<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix1>) -> Array1<F> {
        let xx = x.to_owned().insert_axis(Axis(0));
        self.compute_log_gaussian_prob(&xx).row(0).mapv(|v| v.exp())
    }

    /// Compute precision matrices cholesky decomposiotions given the covariance matrices of
    /// the n multivariate normal distributions specified as a (n, nx, nx) ndarray where
    /// nx is the multivariate dimension.
    fn compute_precisions_cholesky<D: Data<Elem = F>>(
        covariances: &ArrayBase<D, Ix3>,
    ) -> Result<Array3<F>> {
        let n_clusters = covariances.shape()[0];
        let n_features = covariances.shape()[1];
        let mut precisions_chol = Array::zeros((n_clusters, n_features, n_features));
        for (k, covariance) in covariances.outer_iter().enumerate() {
            #[cfg(feature = "blas")]
            let sol = {
                let cov_chol = covariance.with_lapack().cholesky(UPLO::Lower)?;
                cov_chol
                    .solve_triangular(UPLO::Lower, Diag::NonUnit, &Array::eye(n_features))?
                    .without_lapack()
            };
            #[cfg(not(feature = "blas"))]
            let sol = {
                let cov_chol = covariance.cholesky()?;
                cov_chol.solve_triangular(&Array::eye(n_features), UPLO::Lower)?
            };
            precisions_chol.slice_mut(s![k, .., ..]).assign(&sol.t());
        }
        Ok(precisions_chol)
    }

    /// Compute precision matrices of the multivariate normal distributions
    fn compute_precisions<D: Data<Elem = F>>(precisions_chol: &ArrayBase<D, Ix3>) -> Array3<F> {
        let mut precisions = Array3::zeros(precisions_chol.dim());
        for (k, prec_chol) in precisions_chol.outer_iter().enumerate() {
            precisions
                .slice_mut(s![k, .., ..])
                .assign(&prec_chol.dot(&prec_chol.t()));
        }
        precisions
    }

    /// Compute the log of the determinant of the precision matrix decompositions (of the mvn distributions)
    /// taking into account the `heaviside factor`.
    /// Returns the vector of log determinants
    fn compute_log_det(precisions_chol: &Array3<F>, heaviside_factor: F) -> Array1<F> {
        let factor =
            ndarray_rand::rand_distr::num_traits::Float::powf(heaviside_factor, F::cast(-0.5));
        let precs = precisions_chol * factor;
        let n_features = precisions_chol.shape()[1];
        Self::compute_log_det_cholesky(&precs, n_features)
    }

    // Compute weighted log probabilities per component (log P(X)) and responsibilities
    // for each sample in X with respect to the current state of the model.
    fn compute_log_prob_resp<D: Data<Elem = F>>(
        &self,
        x: &ArrayBase<D, Ix2>,
    ) -> (Array1<F>, Array2<F>) {
        let weighted_log_prob = self.compute_log_gaussian_prob(x) + self.weights().mapv(|v| v.ln());
        let log_prob_norm = weighted_log_prob
            .mapv(|v| {
                if v <= F::cast(f64::MIN_10_EXP) {
                    F::zero()
                } else {
                    v.exp()
                }
            })
            .sum_axis(Axis(1))
            .mapv(|v| {
                if v.abs() < F::epsilon() {
                    F::zero()
                } else {
                    v.ln()
                }
            });
        let log_resp = weighted_log_prob - log_prob_norm.to_owned().insert_axis(Axis(1));
        (log_prob_norm, log_resp)
    }

    // Compute the log Likelihood
    // log(P(X|Mean, Precision)) = -0.5*(d*ln(2*PI)-ln(det(Precision)+(X-Mean)^t.Precision.(X-Mean))
    fn compute_log_gaussian_prob<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let means = self.means();
        let n_clusters = means.nrows();
        let factor =
            ndarray_rand::rand_distr::num_traits::Float::powf(self.heaviside_factor, F::cast(-0.5));
        let precs = &self.precisions_chol * factor;
        // The determinant of the precision matrix from the Cholesky decomposition
        // corresponds to the negative half of the determinant of the full precision
        // matrix.
        // In short: det(precision_chol) = - det(precision) / 2
        //let log_det = Self::compute_log_det_cholesky(&precs, n_features);
        let mut log_prob: Array2<F> = Array::zeros((n_samples, n_clusters));
        Zip::indexed(means.rows())
            .and(precs.outer_iter())
            .for_each(|k, mu, prec_chol| {
                let diff = (&x.to_owned() - &mu).dot(&prec_chol);
                log_prob
                    .slice_mut(s![.., k])
                    .assign(&diff.mapv(|v| v * v).sum_axis(Axis(1)))
            });
        let cst = F::cast(n_features as f64 * f64::ln(2. * std::f64::consts::PI));
        let minus_half = F::cast(-0.5);
        log_prob.mapv(|v| minus_half * (v + cst)) + &self.log_det
    }

    /// Compute the determinant of the cholesky decompositions
    /// (i.e. the product of diagonal eleùments) for each multivariate normal distriutions
    fn compute_log_det_cholesky<D: Data<Elem = F>>(
        matrix_chol: &ArrayBase<D, Ix3>,
        n_features: usize,
    ) -> Array1<F> {
        let n_clusters = matrix_chol.shape()[0];
        let log_diags = &matrix_chol
            .to_owned()
            .into_shape_with_order((n_clusters, n_features * n_features))
            .unwrap()
            .slice(s![.., ..; n_features+1])
            .to_owned()
            .mapv(|v| v.ln());
        log_diags.sum_axis(Axis(1))
    }
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<usize>>
    for GaussianMixture<F>
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, targets: &mut Array1<usize>) {
        assert_eq!(
            x.nrows(),
            targets.len(),
            "The number of data points must match the number of output targets."
        );

        let (_, log_resp) = self.compute_log_prob_resp(x);
        *targets = log_resp
            .mapv(F::exp)
            .map_axis(Axis(1), |row| row.argmax().unwrap_or(0));
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<usize> {
        Array1::zeros(x.nrows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use ndarray::{Array, Array2, array};

    #[test]
    fn test_gmx() {
        let weights = array![0.5, 0.5];
        let means = array![[0., 0.], [4., 4.]];
        let covs = array![[[3., 0.], [0., 3.]], [[3., 0.], [0., 3.]]];
        let gmix = GaussianMixture::new(weights, means, covs)
            .expect("Gaussian mixture creation failed")
            .heaviside_factor(0.99);
        let mut obs = Array2::from_elem((11, 2), 0.);
        Zip::from(obs.rows_mut())
            .and(&Array::linspace(0., 4., 11))
            .for_each(|mut o, &v| o.assign(&array![v, v]));
        let _preds = gmix.predict(&obs);
        println!("preds =  {_preds:?}");
        let probas = gmix.predict_probas(&obs);
        println!("probas =  {probas:?}");
    }

    #[test]
    fn test_gmx_one_cluster() {
        let weights = array![1.0];
        let means = array![[4., 4.]];
        let covs = array![[[3., 0.], [0., 3.]]];
        let gmix = GaussianMixture::new(weights, means, covs)
            .expect("Gaussian mixture creation failed")
            .heaviside_factor(1.0);
        let mut obs = Array2::from_elem((11, 2), 0.);
        Zip::from(obs.rows_mut())
            .and(&Array::linspace(0., 4., 11))
            .for_each(|mut o, &v| o.assign(&array![v, v]));
        let _preds = gmix.predict(&obs);
        assert_abs_diff_eq!(_preds, Array::from_elem((11,), 0));
        let probas = gmix.predict_probas(&obs);
        assert_abs_diff_eq!(probas, Array::from_elem((11, 1), 1.0));
    }

    fn test_case(
        means: Array2<f64>,
        covariances: Array3<f64>,
        expected: Array1<f64>,
        x: Array1<f64>,
    ) {
        let part = 1. / means.nrows() as f64;
        let weights = Array::from_elem((means.len(),), part);
        let mvn = GaussianMixture::new(weights, means, covariances).unwrap();
        assert_abs_diff_eq!(expected, mvn.pdfs(&x));
    }

    #[test]
    fn test_pdfs() {
        test_case(
            array![[0., 0.]],
            array![[[1., 0.], [0., 1.]]],
            array![0.05854983152431917],
            array![1., 1.],
        );
        test_case(
            array![[0., 0.]],
            array![[[1., 0.], [0., 1.]]],
            array![0.013064233284684921],
            array![1., 2.],
        );
        test_case(
            array![[0.5, -0.2]],
            array![[[2.0, 0.3], [0.3, 0.5]]],
            array![0.00014842259203296995],
            array![-1., 2.],
        )
    }
}

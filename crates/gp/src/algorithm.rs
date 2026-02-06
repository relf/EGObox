use crate::errors::{GpError, Result};
use crate::mean_models::*;
use crate::optimization::{CobylaParams, optimize_params, prepare_multistart};
use crate::parameters::{GpParams, GpValidParams};
use crate::utils::{DiffMatrix, NormalizedData, pairwise_differences};
use crate::{ThetaTuning, correlation_models::*};

use linfa::dataset::{WithLapack, WithoutLapack};
use linfa::prelude::{Dataset, DatasetBase, Fit, Float, PredictInplace};

#[cfg(not(feature = "blas"))]
use linfa_linalg::{cholesky::*, eigh::*, qr::*, svd::*, triangular::*};
#[cfg(feature = "blas")]
use log::warn;
#[cfg(feature = "blas")]
use ndarray_linalg::{cholesky::*, eigh::*, qr::*, svd::*, triangular::*};

use linfa_pls::PlsRegression;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_stats::QuantileExt;

use log::debug;
use rayon::prelude::*;
#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Instant;

/// Default number of multistart for hyperparameters optimization
pub const GP_OPTIM_N_START: usize = 10;
/// Minimum of function evaluations for COBYLA optimizer
pub const GP_COBYLA_MIN_EVAL: usize = 25;
/// Maximum of function evaluations for COBYLA optimizer
pub const GP_COBYLA_MAX_EVAL: usize = 1000;

/// Internal parameters computed Gp during training
/// used later on in prediction computations
#[derive(Default, Debug)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(deserialize = "F: Deserialize<'de>"))
)]
pub(crate) struct GpInnerParams<F: Float> {
    /// Gaussian process variance
    sigma2: F,
    /// Generalized least-squares regression weights for Universal Kriging or given beta0 for Ordinary Kriging
    beta: Array2<F>,
    /// Gaussian Process weights
    gamma: Array2<F>,
    /// Cholesky decomposition of the correlation matrix \[R\]
    r_chol: Array2<F>,
    /// Solution of the linear equation system : \[R\] x Ft = y
    ft: Array2<F>,
    /// R upper triangle matrix of QR decomposition of the matrix Ft
    ft_qr_r: Array2<F>,
}

impl<F: Float> Clone for GpInnerParams<F> {
    fn clone(&self) -> Self {
        Self {
            sigma2: self.sigma2.to_owned(),
            beta: self.beta.to_owned(),
            gamma: self.gamma.to_owned(),
            r_chol: self.r_chol.to_owned(),
            ft: self.ft.to_owned(),
            ft_qr_r: self.ft_qr_r.to_owned(),
        }
    }
}

/// A GP regression is an interpolation method where the
/// interpolated values are modeled by a Gaussian process with a mean and
/// governed by a prior covariance kernel, which depends on some
/// parameters to be determined.
///
/// The interpolated output is modeled as stochastic process as follows:
///
/// `Y(x) = mu(x) + Z(x)`
///
/// where:
/// * `mu(x)` is the trend i.e. the mean of the gaussian process
/// * `Z(x)` the realization of stochastic gaussian process ~ `Normal(0, sigma^2)`
///
/// which in turn is written as:
///
/// `Y(x) = betas.regr(x) + sigma^2*corr(x, x')`
///
/// where:
/// * `betas` is a vector of linear regression parameters to be determined
/// * `regr(x)` a vector of polynomial basis functions
/// * `sigma^2` is the process variance
/// * `corr(x, x')` is a correlation function which depends on `distance(x, x')`
///   and a set of unknown parameters `thetas` to be determined.
///
/// # Implementation
///
/// * Based on [ndarray](https://github.com/rust-ndarray/ndarray)
///   and [linfa](https://github.com/rust-ml/linfa) and strive to follow [linfa guidelines](https://github.com/rust-ml/linfa/blob/master/CONTRIBUTE.md)
/// * GP mean model can be constant, linear or quadratic
/// * GP correlation model can be build the following kernels: squared exponential, absolute exponential, matern 3/2, matern 5/2    
///   cf. [SMT Kriging](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/krg.html)
/// * For high dimensional problems, the classic GP algorithm does not perform well as
///   it depends on the inversion of a correlation (n, n) matrix which is an O(n3) operation.
///   To work around this problem the library implements dimension reduction using
///   Partial Least Squares method upon Kriging method also known as KPLS algorithm (see Reference)
/// * GP models can be saved and loaded using [serde](https://serde.rs/).
///   See `serializable` feature section below.
///
/// # Features
///
/// ## serializable
///
/// The `serializable` feature enables the serialization of GP models using the [`serde crate`](https://serde.rs/).
///
/// ## blas
///
/// The `blas` feature enables the use of BLAS/LAPACK linear algebra backend available with [`ndarray-linalg`](https://github.com/rust-ndarray/ndarray-linalg).
///
/// # Example
///
/// ```no_run
/// use egobox_gp::{correlation_models::*, mean_models::*, GaussianProcess};
/// use linfa::prelude::*;
/// use ndarray::{arr2, concatenate, Array, Array1, Array2, Axis};
///
/// // one-dimensional test function to approximate
/// fn xsinx(x: &Array2<f64>) -> Array1<f64> {
///     ((x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())).remove_axis(Axis(1))
/// }
///
/// // training data
/// let xt = arr2(&[[0.0], [5.0], [10.0], [15.0], [18.0], [20.0], [25.0]]);
/// let yt = xsinx(&xt);
///
/// // GP with constant mean model and squared exponential correlation model
/// // i.e. Oridinary Kriging model
/// let kriging = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
///                 ConstantMean::default(),
///                 SquaredExponentialCorr::default())
///                 .fit(&Dataset::new(xt, yt))
///                 .expect("Kriging trained");
///
/// // Use trained model for making predictions
/// let xtest = Array::linspace(0., 25., 26).insert_axis(Axis(1));
/// let ytest = xsinx(&xtest);
///
/// let ypred = kriging.predict(&xtest).expect("Kriging prediction");
/// let yvariances = kriging.predict_var(&xtest).expect("Kriging prediction");  
///```
///
/// # Reference:
///
/// Mohamed Amine Bouhlel, John T. Hwang, Nathalie Bartoli, Rémi Lafage, Joseph Morlier, Joaquim R.R.A. Martins,
/// [A Python surrogate modeling framework with derivatives](https://doi.org/10.1016/j.advengsoft.2019.03.005),
/// Advances in Engineering Software, Volume 135, 2019, 102662, ISSN 0965-9978.
///
/// Bouhlel, Mohamed Amine, et al. [Improving kriging surrogates of high-dimensional design
/// models by Partial Least Squares dimension reduction](https://hal.archives-ouvertes.fr/hal-01232938/document)
/// Structural and Multidisciplinary Optimization 53.5 (2016): 935-952.
///
#[derive(Debug)]
#[cfg_attr(
    feature = "serializable",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "F: Serialize, Mean: Serialize, Corr: Serialize",
        deserialize = "F: Deserialize<'de>, Mean: Deserialize<'de>, Corr: Deserialize<'de>"
    ))
)]
pub struct GaussianProcess<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> {
    /// Parameter of the autocorrelation model equal to the inverse of length scale
    theta: Array1<F>,
    /// Reduced likelihood value (result from internal optimization)
    /// Maybe used to compare different trained models
    likelihood: F,
    /// Gaussian process internal fitted params
    inner_params: GpInnerParams<F>,
    /// Weights in case of KPLS dimension reduction coming from PLS regression (orig_dim, kpls_dim)
    w_star: Array2<F>,
    /// Training inputs
    xt_norm: NormalizedData<F>,
    /// Training outputs
    yt_norm: NormalizedData<F>,
    /// Training dataset (input, output)
    pub(crate) training_data: (Array2<F>, Array1<F>),
    /// Parameters used to fit this model
    pub(crate) params: GpValidParams<F, Mean, Corr>,
}

pub(crate) enum GpSamplingMethod {
    Cholesky,
    EigenValues,
}

/// Kriging as GP special case when using constant mean and squared exponential correlation
pub type Kriging<F> = GpParams<F, ConstantMean, SquaredExponentialCorr>;

impl<F: Float> Kriging<F> {
    /// Kriging parameters constructor
    pub fn params() -> GpParams<F, ConstantMean, SquaredExponentialCorr> {
        GpParams::new(ConstantMean(), SquaredExponentialCorr())
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> Clone
    for GaussianProcess<F, Mean, Corr>
{
    fn clone(&self) -> Self {
        Self {
            theta: self.theta.to_owned(),
            likelihood: self.likelihood,
            inner_params: self.inner_params.clone(),
            w_star: self.w_star.to_owned(),
            xt_norm: self.xt_norm.clone(),
            yt_norm: self.yt_norm.clone(),
            training_data: self.training_data.clone(),
            params: self.params.clone(),
        }
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> fmt::Display
    for GaussianProcess<F, Mean, Corr>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "GP(mean={}, corr={}, theta={}, variance={}, likelihood={})",
            self.params.mean,
            self.params.corr,
            self.theta,
            self.inner_params.sigma2,
            self.likelihood,
        )
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>> GaussianProcess<F, Mean, Corr> {
    /// Gp parameters contructor
    pub fn params<NewMean: RegressionModel<F>, NewCorr: CorrelationModel<F>>(
        mean: NewMean,
        corr: NewCorr,
    ) -> GpParams<F, NewMean, NewCorr> {
        GpParams::new(mean, corr)
    }

    /// Predict output values at n given `x` points of nx components specified as a (n, nx) matrix.
    /// Returns n scalar output values as a vector (n,).
    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array1<F>> {
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;
        // Compute the mean term at x
        let f = self.params.mean.coefs(&xnorm);
        // Compute the correlation term at x
        let corr = self._compute_correlation(&xnorm);
        // Scaled predictor
        let y_ = &f.dot(&self.inner_params.beta) + &corr.dot(&self.inner_params.gamma);
        // Predictor
        Ok((&y_ * &self.yt_norm.std + &self.yt_norm.mean).remove_axis(Axis(1)))
    }

    /// Predict variance values at n given `x` points of nx components specified as a (n, nx) matrix.
    /// Returns n variance values as (n,) column vector.
    pub fn predict_var(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Result<Array1<F>> {
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;
        let corr = self._compute_correlation(&xnorm);
        let (rt, u) = self._compute_rt_u(&xnorm, &corr);

        let mut mse = Array::ones(rt.ncols()) - rt.mapv(|v| v * v).sum_axis(Axis(0))
            + u.mapv(|v: F| v * v).sum_axis(Axis(0));
        mse.mapv_inplace(|v| self.inner_params.sigma2 * v);

        // Mean Squared Error might be slightly negative depending on
        // machine precision: set to zero in that case
        Ok(mse.mapv(|v| if v < F::zero() { F::zero() } else { F::cast(v) }))
    }

    /// Predict both output values and variance at n given `x` points of nx components
    pub fn predict_valvar(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Result<(Array1<F>, Array1<F>)> {
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;
        // Compute the mean term at x
        let f = self.params.mean.coefs(&xnorm);
        // Compute the correlation term at x
        let corr = self._compute_correlation(&xnorm);
        // Scaled predictor
        let y_ = &f.dot(&self.inner_params.beta) + &corr.dot(&self.inner_params.gamma);
        // Predictor
        let yp = (&y_ * &self.yt_norm.std + &self.yt_norm.mean).remove_axis(Axis(1));

        let (rt, u) = self._compute_rt_u(&xnorm, &corr);

        let mut mse = Array::ones(rt.ncols()) - rt.mapv(|v| v * v).sum_axis(Axis(0))
            + u.mapv(|v: F| v * v).sum_axis(Axis(0));
        mse.mapv_inplace(|v| self.inner_params.sigma2 * v);

        // Mean Squared Error might be slightly negative depending on
        // machine precision: set to zero in that case
        let vmse = mse.mapv(|v| if v < F::zero() { F::zero() } else { F::cast(v) });

        Ok((yp, vmse))
    }

    /// Compute covariance matrix given x points specified as a (n, nx) matrix
    fn _compute_covariance(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;
        let corr = self._compute_correlation(&xnorm);
        let (rt, u) = self._compute_rt_u(&xnorm, &corr);

        let cross_dx = pairwise_differences(&xnorm, &xnorm);
        let k = self
            .params
            .corr
            .rval_from_distances(&cross_dx, &self.theta, &self.w_star);
        let k = k
            .into_shape_with_order((xnorm.nrows(), xnorm.nrows()))
            .unwrap();

        // let cov_matrix =
        //     &array![self.inner_params.sigma2] * (k - rt.t().to_owned().dot(&rt) + u.t().dot(&u));
        let mut cov_matrix = k - rt.t().to_owned().dot(&rt) + u.t().dot(&u);
        cov_matrix.mapv_inplace(|v| self.inner_params.sigma2 * v);
        cov_matrix
    }

    /// Compute `rt` and `u` matrices and return normalized x as well
    /// This method factorizes computations done to get variances and covariance matrix
    fn _compute_rt_u(
        &self,
        xnorm: &ArrayBase<impl Data<Elem = F>, Ix2>,
        corr: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array2<F>, Array2<F>) {
        let inners = &self.inner_params;

        let corr_t = corr.t().to_owned();
        #[cfg(feature = "blas")]
        let rt = inners
            .r_chol
            .to_owned()
            .with_lapack()
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &corr_t.with_lapack())
            .unwrap()
            .without_lapack();
        #[cfg(not(feature = "blas"))]
        let rt = inners
            .r_chol
            .solve_triangular(&corr_t, UPLO::Lower)
            .unwrap();

        let rhs = inners.ft.t().dot(&rt) - self.params.mean.coefs(xnorm).t();
        #[cfg(feature = "blas")]
        let u = inners
            .ft_qr_r
            .to_owned()
            .t()
            .with_lapack()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &rhs.with_lapack())
            .unwrap()
            .without_lapack();
        #[cfg(not(feature = "blas"))]
        let u = inners
            .ft_qr_r
            .t()
            .solve_triangular(&rhs, UPLO::Lower)
            .unwrap();
        (rt, u)
    }

    /// Compute correlation matrix given x points specified as a (n, nx) matrix
    fn _compute_correlation(&self, xnorm: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        // Get pairwise componentwise L1-distances to the input training set
        let dx = pairwise_differences(xnorm, &self.xt_norm.data);
        // Compute the correlation function
        let r = self
            .params
            .corr
            .rval_from_distances(&dx, &self.theta, &self.w_star);
        let n_obs = xnorm.nrows();
        let nt = self.xt_norm.data.nrows();
        r.into_shape_with_order((n_obs, nt)).unwrap().to_owned()
    }

    /// Sample the gaussian process for `n_traj` trajectories using cholesky decomposition
    pub fn sample_chol(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>, n_traj: usize) -> Array2<F> {
        self._sample(x, n_traj, GpSamplingMethod::Cholesky)
    }

    /// Sample the gaussian process for `n_traj` trajectories using eigenvalues decomposition
    pub fn sample_eig(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>, n_traj: usize) -> Array2<F> {
        self._sample(x, n_traj, GpSamplingMethod::EigenValues)
    }

    /// Sample the gaussian process for `n_traj` trajectories using eigenvalues decomposition (alias of `sample_eig`)
    pub fn sample(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>, n_traj: usize) -> Array2<F> {
        self.sample_eig(x, n_traj)
    }

    /// Sample the gaussian process for `n_traj` trajectories using either
    /// cholesky or eigenvalues decomposition to compute the decomposition of the conditioned covariance matrix.
    /// The later one is recommended as cholesky decomposition suffer from occurence of ill-conditioned matrices
    /// when the number of x locations increase.
    fn _sample(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        n_traj: usize,
        method: GpSamplingMethod,
    ) -> Array2<F> {
        let mean = self.predict(x).unwrap();
        let cov = self._compute_covariance(x);
        sample(x, mean.insert_axis(Axis(1)), cov, n_traj, method)
    }

    /// Retrieve optimized hyperparameters theta
    pub fn theta(&self) -> &Array1<F> {
        &self.theta
    }

    /// Estimated variance
    pub fn variance(&self) -> F {
        self.inner_params.sigma2
    }

    /// Retrieve reduced likelihood value
    pub fn likelihood(&self) -> F {
        self.likelihood
    }

    /// Retrieve number of PLS components 1 <= n <= x dimension
    pub fn kpls_dim(&self) -> Option<usize> {
        if self.w_star.ncols() < self.xt_norm.ncols() {
            Some(self.w_star.ncols())
        } else {
            None
        }
    }

    /// Retrieve input and output dimensions
    pub fn dims(&self) -> (usize, usize) {
        (self.xt_norm.ncols(), self.yt_norm.ncols())
    }

    /// Predict derivatives of the output prediction
    /// wrt the kxth component at a set of n points `x` specified as a (n, nx) matrix where x has nx components.
    pub fn predict_kth_derivatives(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        kx: usize,
    ) -> Array1<F> {
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;
        let corr = self._compute_correlation(&xnorm);

        let beta = &self.inner_params.beta;
        let gamma = &self.inner_params.gamma;

        let df_dx_kx = if self.inner_params.beta.nrows() <= 1 + self.xt_norm.data.ncols() {
            // for constant or linear: df/dx = cst ([0] or [1]) for all x, so takes use x[0] to get the constant
            let df = self.params.mean.jac(&x.row(0));
            let df_dx = df.t().row(kx).dot(beta);
            df_dx.broadcast((x.nrows(), 1)).unwrap().to_owned()
        } else {
            // for quadratic df/dx really depends on x
            let mut dfdx = Array2::zeros((x.nrows(), 1));
            Zip::from(dfdx.rows_mut())
                .and(xnorm.rows())
                .for_each(|mut dfxi, xi| {
                    let df = self.params.mean.jac(&xi);
                    let df_dx = (df.t().row(kx)).dot(beta);
                    dfxi.assign(&df_dx);
                });
            dfdx
        };

        let nr = x.nrows();
        let nc = self.xt_norm.data.nrows();
        let d_dx_1 = &xnorm
            .column(kx)
            .to_owned()
            .into_shape_with_order((nr, 1))
            .unwrap()
            .broadcast((nr, nc))
            .unwrap()
            .to_owned();

        let d_dx_2 = self
            .xt_norm
            .data
            .column(kx)
            .to_owned()
            .as_standard_layout()
            .into_shape_with_order((1, nc))
            .unwrap()
            .to_owned();

        let d_dx = d_dx_1 - d_dx_2;

        // Get pairwise componentwise L1-distances to the input training set
        let theta = &self.theta.to_owned();
        let d_dx_corr = d_dx * corr;

        // (df(xnew)/dx).beta + (dr(xnew)/dx).R^-1(ytrain - f.beta)
        // gamma = R^-1(ytrain - f.beta)
        // Warning: squared exponential only
        let res = (df_dx_kx - d_dx_corr.dot(gamma).map(|v| F::cast(2.) * theta[kx] * *v))
            * self.yt_norm.std[0]
            / self.xt_norm.std[kx];
        res.column(0).to_owned()
    }

    /// Predict derivatives at a set of point `x` specified as a (n, nx) matrix where x has nx components.
    /// Returns a (n, nx) matrix containing output derivatives at x wrt each nx components
    pub fn predict_gradients(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let mut drv = Array2::<F>::zeros((x.nrows(), self.xt_norm.data.ncols()));
        Zip::from(drv.rows_mut())
            .and(x.rows())
            .for_each(|mut row, xi| {
                let pred = self.predict_jacobian(&xi);
                row.assign(&pred.column(0));
            });
        drv
    }

    /// Predict gradient at a given x point
    /// Note: output is one dimensional, named jacobian as result is given as a one-column matrix  
    fn predict_jacobian(&self, x: &ArrayBase<impl Data<Elem = F>, Ix1>) -> Array2<F> {
        let xx = x.to_owned().insert_axis(Axis(0));
        let mut jac = Array2::zeros((xx.ncols(), 1));

        let xnorm = (xx - &self.xt_norm.mean) / &self.xt_norm.std;

        let beta = &self.inner_params.beta;
        let gamma = &self.inner_params.gamma;

        let df = self.params.mean.jac(&xnorm.row(0));
        let df_dx = df.t().dot(beta);

        let dr = self
            .params
            .corr
            .jac(&xnorm.row(0), &self.xt_norm.data, &self.theta, &self.w_star);

        let dr_dx = df_dx + dr.t().dot(gamma);
        Zip::from(jac.rows_mut())
            .and(dr_dx.rows())
            .and(&self.xt_norm.std)
            .for_each(|mut jc, dr_i, std_i| {
                let jc_i = dr_i.map(|v| *v * self.yt_norm.std[0] / *std_i);
                jc.assign(&jc_i)
            });

        jac
    }

    /// Predict variance derivatives at a point `x` specified as a (nx,) vector where x has nx components.
    /// Returns a (nx,) vector containing variance derivatives at `x` wrt each nx components
    #[cfg(not(feature = "blas"))]
    pub fn predict_var_gradients_single(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
    ) -> Array1<F> {
        let x = &(x.to_owned().insert_axis(Axis(0)));
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;
        let sigma2 = self.inner_params.sigma2;
        let r_chol = &self.inner_params.r_chol;

        let (r, dr) = self.params.corr.rval_with_jac(
            &xnorm.row(0),
            &self.xt_norm.data,
            &self.theta,
            &self.w_star,
        );

        // rho1 = Rc^-1 . r(x, X)
        let rho1 = r_chol.solve_triangular(&r, UPLO::Lower).unwrap();

        // inv_kr = Rc^t^-1 . Rc^-1 . r(x, X) = R^-1 . r(x, X)
        let inv_kr = r_chol.t().solve_triangular(&rho1, UPLO::Upper).unwrap();

        // p1 = ((dr(x, X)/dx)^t . R^-1 . r(x, X))^t = ((R^-1 . r(x, X))^t . dr(x, X)/dx) = r(x, X)^t . R^-1 . dr(x, X)/dx = p2
        // let p1 = dr.t().dot(&inv_kr).t().to_owned();

        // p2 = ((R^-1 . r(x, X))^t . dr(x, X)/dx)^t = dr(x, X)/dx)^t . R^-1 . r(x, X) = p1
        let p2 = inv_kr.t().dot(&dr);

        let f_x = self.params.mean.coefs(&xnorm).t().to_owned();
        let f_mean = self.params.mean.coefs(&self.xt_norm.data);

        // rho2 = Rc^-1 . F(X)
        let rho2 = r_chol.solve_triangular(&f_mean, UPLO::Lower).unwrap();
        // inv_kf = Rc^-1^t . Rc^-1 . F(X) = R^-1 . F(X)
        let inv_kf = r_chol.t().solve_triangular(&rho2, UPLO::Upper).unwrap();

        // A = f(x)^t - r(x, X)^t . R^-1 . F(X)   -> (1 x m)
        let a_mat = f_x.t().to_owned() - r.t().dot(&inv_kf);

        // B = F(X)^t . R^-1 . F(X)
        let b_mat = f_mean.t().dot(&inv_kf);
        // rho3 = Bc
        let rho3 = b_mat.cholesky().unwrap();
        // inv_bat = Bc^-1 . A^t
        let inv_bat = rho3.solve_triangular(&a_mat.t(), UPLO::Lower).unwrap();
        // D = Bc^t-1 . Bc^-1 . A^t = B^-1 . A^t
        let d_mat = rho3.t().solve_triangular(&inv_bat, UPLO::Upper).unwrap();

        let df = self.params.mean.jac(&xnorm.row(0));

        // dA/dx = df(x)/dx^t - dr(x, X)/dx^t . R^-1 . F
        let d_a = df.t().to_owned() - dr.t().dot(&inv_kf);

        // p3 = (dA/dx . B^-1 . A^t)^t = A . B^-1 . dA/dx^t
        // let p3 = d_a.dot(&d_mat).t().to_owned();

        // p4 = (B^-1 . A)^t . dA/dx^t = A^t . B^-1 . dA/dx^t = p3
        let p4 = d_mat.t().dot(&d_a.t());
        let two = F::cast(2.);
        let prime = (p4 - p2).mapv(|v| two * v);

        let x_std = &self.xt_norm.std;
        let dvar = (prime / x_std).mapv(|v| v * sigma2);
        dvar.row(0).into_owned()
    }

    /// See non blas version
    #[cfg(feature = "blas")]
    pub fn predict_var_gradients_single(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix1>,
    ) -> Array1<F> {
        let x = &(x.to_owned().insert_axis(Axis(0)));
        let xnorm = (x - &self.xt_norm.mean) / &self.xt_norm.std;

        let dx = pairwise_differences(&xnorm, &self.xt_norm.data);

        let sigma2 = self.inner_params.sigma2;
        let r_chol = &self.inner_params.r_chol.to_owned().with_lapack();

        let r = self
            .params
            .corr
            .rval_from_distances(&dx, &self.theta, &self.w_star)
            .with_lapack();
        let dr = self
            .params
            .corr
            .jac(&xnorm.row(0), &self.xt_norm.data, &self.theta, &self.w_star)
            .with_lapack();

        let rho1 = r_chol
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &r)
            .unwrap();
        let inv_kr = r_chol
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &rho1)
            .unwrap();

        // let p1 = dr.t().dot(&inv_kr).t().to_owned();

        let p2 = inv_kr.t().dot(&dr);

        let f_x = self.params.mean.coefs(x).t().to_owned();
        let f_mean = self.params.mean.coefs(&self.xt_norm.data).with_lapack();

        let rho2 = r_chol
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &f_mean)
            .unwrap();
        let inv_kf = r_chol
            .t()
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &rho2)
            .unwrap();

        let a_mat = f_x.t().to_owned().with_lapack() - r.t().dot(&inv_kf);

        let b_mat = f_mean.t().dot(&inv_kf);

        let d_mat = match b_mat.cholesky(UPLO::Lower) {
            Ok(rho3) => {
                let inv_bat = rho3
                    .solve_triangular(UPLO::Upper, Diag::NonUnit, &a_mat.t().to_owned())
                    .unwrap();
                rho3.t()
                    .solve_triangular(UPLO::Upper, Diag::NonUnit, &inv_bat)
                    .unwrap()
            }
            Err(_) => {
                warn!("Cholesky decomposition error during variance dervivatives computation");
                Array2::zeros((b_mat.nrows(), b_mat.ncols()))
            }
        };

        let df = self.params.mean.jac(&xnorm.row(0)).with_lapack();

        let d_a = df.t().to_owned() - dr.t().dot(&inv_kf);
        // let p3 = d_a.dot(&d_mat).t();
        let p4 = d_mat.t().dot(&d_a.t());

        let two = F::cast(2.);
        let prime_t = (p4 - p2).without_lapack().mapv(|v| two * v);

        let x_std = &self.xt_norm.std;
        let dvar = (prime_t / x_std).mapv(|v| v * sigma2);
        dvar.row(0).into_owned()
    }

    /// Predict variance derivatives at a set of points `x` specified as a (n, nx) matrix where x has nx components.
    /// Returns a (n, nx) matrix containing variance derivatives at `x` wrt each nx components
    pub fn predict_var_gradients(&self, x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
        let mut derivs = Array::zeros((x.nrows(), x.ncols()));
        Zip::from(derivs.rows_mut())
            .and(x.rows())
            .for_each(|mut der, x| der.assign(&self.predict_var_gradients_single(&x)));
        derivs
    }

    /// Predict both value and variance gradients at a set of points `x` specified as a (n, nx) matrix
    /// where x has nx components.
    pub fn predict_valvar_gradients(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array2<F>, Array2<F>) {
        let mut val_derivs = Array::zeros((x.nrows(), x.ncols()));
        let mut var_derivs = Array::zeros((x.nrows(), x.ncols()));
        Zip::from(val_derivs.rows_mut())
            .and(var_derivs.rows_mut())
            .and(x.rows())
            .for_each(|mut val_der, mut var_der, x| {
                val_der.assign(&self.predict_jacobian(&x).column(0));
                var_der.assign(&self.predict_var_gradients_single(&x));
            });
        (val_derivs, var_derivs)
    }
}

impl<F, D, Mean, Corr> PredictInplace<ArrayBase<D, Ix2>, Array1<F>>
    for GaussianProcess<F, Mean, Corr>
where
    F: Float,
    D: Data<Elem = F>,
    Mean: RegressionModel<F>,
    Corr: CorrelationModel<F>,
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<F>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let values = self.predict(x).expect("GP Prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        Array1::zeros((x.nrows(),))
    }
}

/// Gausssian Process adaptator to implement `linfa::Predict` trait for variance prediction.
#[allow(dead_code)]
pub struct GpVariancePredictor<'a, F, Mean, Corr>(&'a GaussianProcess<F, Mean, Corr>)
where
    F: Float,
    Mean: RegressionModel<F>,
    Corr: CorrelationModel<F>;

impl<F, D, Mean, Corr> PredictInplace<ArrayBase<D, Ix2>, Array1<F>>
    for GpVariancePredictor<'_, F, Mean, Corr>
where
    F: Float,
    D: Data<Elem = F>,
    Mean: RegressionModel<F>,
    Corr: CorrelationModel<F>,
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<F>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let values = self.0.predict_var(x).expect("GP Prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        Array1::zeros(x.nrows())
    }
}

impl<F: Float, Mean: RegressionModel<F>, Corr: CorrelationModel<F>, D: Data<Elem = F>>
    Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>, GpError> for GpValidParams<F, Mean, Corr>
{
    type Object = GaussianProcess<F, Mean, Corr>;

    /// Fit GP parameters using maximum likelihood
    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
    ) -> Result<Self::Object> {
        let x = dataset.records();
        let y = dataset.targets().to_owned().insert_axis(Axis(1));

        if let Some(d) = self.kpls_dim()
            && *d > x.ncols()
        {
            return Err(GpError::InvalidValueError(format!(
                "Dimension reduction {} should be smaller than actual \
                    training input dimensions {}",
                d,
                x.ncols()
            )));
        }

        let dim = if let Some(n_components) = self.kpls_dim() {
            *n_components
        } else {
            x.ncols()
        };

        let (x, y, active, init) = match self.theta_tuning() {
            ThetaTuning::Fixed(init) | ThetaTuning::Full { init, bounds: _ } => (
                x.to_owned(),
                y.to_owned(),
                (0..dim).collect::<Vec<_>>(),
                init,
            ),
            ThetaTuning::Partial {
                init,
                bounds: _,
                active,
            } => (x.to_owned(), y.to_owned(), active.to_vec(), init),
        };
        // Initial guess for theta
        let theta0_dim = init.len();
        let theta0 = if theta0_dim == 1 {
            Array1::from_elem(dim, init[0])
        } else if theta0_dim == dim {
            init.to_owned()
        } else {
            panic!(
                "Initial guess for theta should be either 1-dim or dim of xtrain (w_star.ncols()), got {theta0_dim}"
            )
        };

        let xtrain = NormalizedData::new(&x);
        let ytrain = NormalizedData::new(&y);

        let mut w_star = Array2::eye(x.ncols());
        if let Some(n_components) = self.kpls_dim() {
            let ds = Dataset::new(x.to_owned(), y.to_owned());
            w_star = PlsRegression::params(*n_components).fit(&ds).map_or_else(
                |e| match e {
                    linfa_pls::PlsError::PowerMethodConstantResidualError() => {
                        Ok(Array2::zeros((x.ncols(), *n_components)))
                    }
                    err => Err(err),
                },
                |v| Ok(v.rotations().0.to_owned()),
            )?;
        };
        let x_distances = DiffMatrix::new(&xtrain.data);
        let sums = x_distances
            .d
            .mapv(|v| num_traits::float::Float::abs(v))
            .sum_axis(Axis(1));
        if *sums.min().unwrap() == F::zero() {
            println!(
                "Warning: multiple x input features have the same value (at least same row twice)."
            );
        }
        let fx = self.mean().coefs(&xtrain.data);

        let opt_params = match self.theta_tuning() {
            ThetaTuning::Fixed(init) => {
                // Easy path no optimization
                init.to_owned()
            }
            ThetaTuning::Full { init: _, bounds }
            | ThetaTuning::Partial {
                init: _,
                bounds,
                active: _,
            } => {
                let base: f64 = 10.;
                let objfn = |x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()| -> f64 {
                    let mut theta = theta0.to_owned();
                    let xarr = x.iter().map(|v| base.powf(*v)).collect::<Vec<_>>();
                    std::iter::zip(active.clone(), xarr).for_each(|(i, xi)| theta[i] = F::cast(xi));

                    for v in theta.iter() {
                        // check theta as optimizer may return nan values
                        if v.is_nan() {
                            // shortcut return worst value wrt to rlf minimization
                            return f64::INFINITY;
                        }
                    }
                    let rxx = self
                        .corr()
                        .rval_from_distances(&x_distances.d, &theta, &w_star);
                    match reduced_likelihood(&fx, rxx, &x_distances, &ytrain, self.nugget()) {
                        Ok(r) => unsafe { -(*(&r.0 as *const F as *const f64)) },
                        Err(_) => f64::INFINITY,
                    }
                };

                // Multistart: user theta0 + 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10.
                // let bounds = vec![(F::cast(-6.), F::cast(2.)); theta0.len()];
                let bounds_dim = bounds.len();
                let bounds = if bounds_dim == 1 {
                    vec![bounds[0]; w_star.ncols()]
                } else if bounds_dim == w_star.ncols() {
                    bounds.to_vec()
                } else {
                    panic!(
                        "Bounds for theta should be either 1-dim or dim of xtrain ({}), got {}",
                        w_star.ncols(),
                        bounds_dim
                    )
                };

                // Select init params and bounds wrt to activity
                let active_bounds = bounds
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| active.contains(i))
                    .map(|(_, &b)| b)
                    .collect::<Vec<_>>();
                let (theta_inits, bounds) = prepare_multistart(
                    self.n_start(),
                    &theta0.select(Axis(0), &active),
                    &active_bounds,
                );
                debug!("Optimize with multistart theta = {theta_inits:?} and bounds = {bounds:?}");
                let now = Instant::now();
                let opt_params = (0..theta_inits.nrows())
                    .into_par_iter()
                    .map(|i| {
                        optimize_params(
                            objfn,
                            &theta_inits.row(i).to_owned(),
                            &bounds,
                            CobylaParams {
                                maxeval: (10 * theta_inits.ncols())
                                    .clamp(GP_COBYLA_MIN_EVAL, self.max_eval()),
                                ..CobylaParams::default()
                            },
                        )
                    })
                    .reduce(
                        || (f64::INFINITY, Array::ones((theta_inits.ncols(),))),
                        |a, b| if b.0 < a.0 { b } else { a },
                    );
                debug!("elapsed optim = {:?}", now.elapsed().as_millis());
                opt_params.1.mapv(|v| F::cast(base.powf(v)))
            }
        };

        // In case of partial optimization we set only active components
        let opt_params = match self.theta_tuning() {
            ThetaTuning::Fixed(_) | ThetaTuning::Full { init: _, bounds: _ } => opt_params,
            ThetaTuning::Partial {
                init,
                bounds: _,
                active,
            } => {
                let mut opt_theta = init.to_owned();
                std::iter::zip(active.clone(), opt_params)
                    .for_each(|(i, xi)| opt_theta[i] = F::cast(xi));
                opt_theta
            }
        };

        let rxx = self
            .corr()
            .rval_from_distances(&x_distances.d, &opt_params, &w_star);
        let (lkh, inner_params) =
            reduced_likelihood(&fx, rxx, &x_distances, &ytrain, self.nugget())?;
        Ok(GaussianProcess {
            theta: opt_params,
            likelihood: lkh,
            inner_params,
            w_star,
            xt_norm: xtrain,
            yt_norm: ytrain,
            training_data: (x.to_owned(), y.to_owned().remove_axis(Axis(1))),
            params: self.clone(),
        })
    }
}

/// Compute reduced likelihood function
/// fx: mean factors term at x samples,
/// rxx: correlation factors at x samples,
/// x_distances: pairwise distances between x samples
/// ytrain: normalized output training values
/// nugget: factor to improve numerical stability  
#[cfg(not(feature = "blas"))]
fn reduced_likelihood<F: Float>(
    fx: &ArrayBase<impl Data<Elem = F>, Ix2>,
    rxx: ArrayBase<impl Data<Elem = F>, Ix2>,
    x_distances: &DiffMatrix<F>,
    ytrain: &NormalizedData<F>,
    nugget: F,
) -> Result<(F, GpInnerParams<F>)> {
    // Set up R
    let mut r_mx: Array2<F> = Array2::<F>::eye(x_distances.n_obs).mapv(|v| v + v * nugget);
    for (i, ij) in x_distances.d_indices.outer_iter().enumerate() {
        r_mx[[ij[0], ij[1]]] = rxx[[i, 0]];
        r_mx[[ij[1], ij[0]]] = rxx[[i, 0]];
    }
    let fxl = fx;
    // R cholesky decomposition
    let r_chol = r_mx.cholesky()?;
    // Solve generalized least squared problem
    let ft = r_chol.solve_triangular(fxl, UPLO::Lower)?;
    let (ft_qr_q, ft_qr_r) = ft.qr().unwrap().into_decomp();

    // Check whether we have an ill-conditionned problem
    let (_, sv_qr_r, _) = ft_qr_r.svd(false, false).unwrap();
    let cond_ft = sv_qr_r[sv_qr_r.len() - 1] / sv_qr_r[0];
    if F::cast(cond_ft) < F::cast(1e-10) {
        let (_, sv_f, _) = &fxl.svd(false, false).unwrap();
        let cond_fx = sv_f[0] / sv_f[sv_f.len() - 1];
        if F::cast(cond_fx) > F::cast(1e15) {
            return Err(GpError::LikelihoodComputationError(
                "F is too ill conditioned. Poor combination \
                of regression model and observations."
                    .to_string(),
            ));
        } else {
            // ft is too ill conditioned, get out (try different theta)
            return Err(GpError::LikelihoodComputationError(
                "ft is too ill conditioned, try another theta again".to_string(),
            ));
        }
    }
    let yt = r_chol.solve_triangular(&ytrain.data, UPLO::Lower)?;

    let beta = ft_qr_r.solve_triangular_into(ft_qr_q.t().dot(&yt), UPLO::Upper)?;
    let rho = yt - ft.dot(&beta);
    let rho_sqr = rho.mapv(|v| v * v).sum_axis(Axis(0));

    let gamma = r_chol.t().solve_triangular_into(rho, UPLO::Upper)?;
    // The determinant of R is equal to the squared product of
    // the diagonal elements of its Cholesky decomposition r_chol
    let n_obs: F = F::cast(x_distances.n_obs);

    let logdet = r_chol.diag().mapv(|v: F| v.log10()).sum() * F::cast(2.) / n_obs;

    // Reduced likelihood
    let sigma2 = rho_sqr / n_obs;
    let reduced_likelihood = -n_obs * (sigma2.sum().log10() + logdet);

    Ok((
        reduced_likelihood,
        GpInnerParams {
            sigma2: sigma2[0] * ytrain.std[0] * ytrain.std[0],
            beta,
            gamma,
            r_chol,
            ft,
            ft_qr_r,
        },
    ))
}

/// See non blas version
#[cfg(feature = "blas")]
fn reduced_likelihood<F: Float>(
    fx: &ArrayBase<impl Data<Elem = F>, Ix2>,
    rxx: ArrayBase<impl Data<Elem = F>, Ix2>,
    x_distances: &DiffMatrix<F>,
    ytrain: &NormalizedData<F>,
    nugget: F,
) -> Result<(F, GpInnerParams<F>)> {
    // Set up R
    let mut r_mx: Array2<F> = Array2::<F>::eye(x_distances.n_obs).mapv(|v| v + v * nugget);
    for (i, ij) in x_distances.d_indices.outer_iter().enumerate() {
        r_mx[[ij[0], ij[1]]] = rxx[[i, 0]];
        r_mx[[ij[1], ij[0]]] = rxx[[i, 0]];
    }

    let fxl = fx.to_owned().with_lapack();

    // R cholesky decomposition
    let r_chol = r_mx.with_lapack().cholesky(UPLO::Lower)?;

    // Solve generalized least squared problem
    let ft = r_chol.solve_triangular(UPLO::Lower, Diag::NonUnit, &fxl)?;
    let (ft_qr_q, ft_qr_r) = ft.qr().unwrap();

    // Check whether we have an ill-conditionned problem
    let (_, sv_qr_r, _) = ft_qr_r.svd(false, false).unwrap();
    let cond_ft = sv_qr_r[sv_qr_r.len() - 1] / sv_qr_r[0];
    if F::cast(cond_ft) < F::cast(1e-10) {
        let (_, sv_f, _) = &fxl.svd(false, false).unwrap();
        let cond_fx = sv_f[0] / sv_f[sv_f.len() - 1];
        if F::cast(cond_fx) > F::cast(1e15) {
            return Err(GpError::LikelihoodComputationError(
                "F is too ill conditioned. Poor combination \
                of regression model and observations."
                    .to_string(),
            ));
        } else {
            // ft is too ill conditioned, get out (try different theta)
            return Err(GpError::LikelihoodComputationError(
                "ft is too ill conditioned, try another theta again".to_string(),
            ));
        }
    }

    let yt = r_chol.solve_triangular(
        UPLO::Lower,
        Diag::NonUnit,
        &ytrain.data.to_owned().with_lapack(),
    )?;

    let beta = ft_qr_r.solve_triangular_into(UPLO::Upper, Diag::NonUnit, ft_qr_q.t().dot(&yt))?;

    let rho = yt - ft.dot(&beta);
    let rho_sqr = rho.mapv(|v| v * v).sum_axis(Axis(0));
    let rho_sqr = rho_sqr.without_lapack();

    let gamma = r_chol
        .t()
        .solve_triangular_into(UPLO::Upper, Diag::NonUnit, rho)?;

    // The determinant of R is equal to the squared product of
    // the diagonal elements of its Cholesky decomposition r_chol
    let n_obs: F = F::cast(x_distances.n_obs);

    let logdet = r_chol
        .to_owned()
        .without_lapack()
        .diag()
        .mapv(|v: F| v.log10())
        .sum()
        * F::cast(2.)
        / n_obs;

    // Reduced likelihood
    let sigma2: Array1<F> = rho_sqr / n_obs;
    let reduced_likelihood = -n_obs * (sigma2.sum().log10() + logdet);
    Ok((
        reduced_likelihood,
        GpInnerParams {
            sigma2: sigma2[0] * ytrain.std[0] * ytrain.std[0],
            beta: beta.without_lapack(),
            gamma: gamma.without_lapack(),
            r_chol: r_chol.without_lapack(),
            ft: ft.without_lapack(),
            ft_qr_r: ft_qr_r.without_lapack(),
        },
    ))
}

/// Sample the gaussian process for `n_traj` trajectories using either
/// cholesky or eigenvalues decomposition to compute the decomposition of the conditioned covariance matrix.
/// `cov_x` is the covariance matrix at the given x points [n, nx]
/// The later one is recommended as cholesky decomposition suffer from occurence of ill-conditioned matrices
/// when the number of x locations increase.
pub(crate) fn sample<F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    mean_x: Array2<F>,
    cov_x: Array2<F>,
    n_traj: usize,
    method: GpSamplingMethod,
) -> Array2<F> {
    let n_eval = x.nrows();
    let c = match method {
        GpSamplingMethod::Cholesky => {
            #[cfg(not(feature = "blas"))]
            let c = cov_x.with_lapack().cholesky().unwrap();
            #[cfg(feature = "blas")]
            let c = cov_x.with_lapack().cholesky(UPLO::Lower).unwrap();
            c
        }
        GpSamplingMethod::EigenValues => {
            #[cfg(feature = "blas")]
            let (v, w) = cov_x.with_lapack().eigh(UPLO::Lower).unwrap();
            #[cfg(not(feature = "blas"))]
            let (v, w) = cov_x.with_lapack().eigh_into().unwrap();
            let v = v.mapv(F::cast);
            let v = v.mapv(|x| {
                // We lower bound the float value at 1e-9
                if x < F::cast(1e-9) {
                    return F::zero();
                }
                x.sqrt()
            });
            let d = Array2::from_diag(&v).with_lapack();
            #[cfg(feature = "blas")]
            let c = w.dot(&d);
            #[cfg(not(feature = "blas"))]
            let c = w.dot(&d);
            c
        }
    }
    .without_lapack();
    let normal = Normal::new(0., 1.).unwrap();
    let ary = Array::random((n_eval, n_traj), normal).mapv(|v| F::cast(v));
    mean_x.to_owned() + c.dot(&ary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
    use argmin_testfunctions::rosenbrock;
    use egobox_doe::{Lhs, LhsKind, SamplingMethod};
    use linfa::prelude::Predict;
    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::Norm;
    use ndarray::{Array, Zip, arr1, arr2, array};
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;
    use ndarray_npy::write_npy;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_stats::DeviationExt;
    use paste::paste;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_constant_function() {
        let dim = 3;
        let lim = array![[0., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let nt = 5;
        let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let yt = Array::from_vec(vec![3.1; nt]);
        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .theta_init(array![0.1])
        .kpls_dim(Some(1))
        .fit(&Dataset::new(xt, yt))
        .expect("GP fit error");
        let rng = Xoshiro256Plus::seed_from_u64(43);
        let xtest = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let ytest = gp.predict(&xtest).expect("prediction error");
        assert_abs_diff_eq!(Array::from_elem((nt,), 3.1), ytest, epsilon = 1e-6);
    }

    macro_rules! test_gp {
        ($regr:ident, $corr:ident) => {
            paste! {

                #[test]
                fn [<test_gp_ $regr:snake _ $corr:snake >]() {
                    let xt = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
                    let xplot = Array::linspace(0., 4., 100).insert_axis(Axis(1));
                    let yt = array![0.0, 1.0, 1.5, 0.9, 1.0];
                    let gp = GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                        [<$regr Mean>]::default(),
                        [<$corr Corr>]::default(),
                    )
                    .theta_init(array![0.1])
                    .fit(&Dataset::new(xt, yt))
                    .expect("GP fit error");
                    let yvals = gp
                        .predict(&arr2(&[[1.0], [3.5]]))
                        .expect("prediction error");
                    let expected_y = arr1(&[1.0, 0.9]);
                    assert_abs_diff_eq!(expected_y, yvals, epsilon = 0.5);

                    let gpr_vals = gp.predict(&xplot).unwrap();

                    let yvars = gp
                        .predict_var(&arr2(&[[1.0], [3.5]]))
                        .expect("prediction error");
                    let expected_vars = arr1(&[0., 0.1]);
                    assert_abs_diff_eq!(expected_vars, yvars, epsilon = 0.5);

                    let gpr_vars = gp.predict_var(&xplot).unwrap();

                    let test_dir = "target/tests";
                    std::fs::create_dir_all(test_dir).ok();

                    let xplot_file = stringify!([<gp_x_ $regr:snake _ $corr:snake >]);
                    let file_path = format!("{}/{}.npy", test_dir, xplot_file);
                    write_npy(file_path, &xplot).expect("x saved");

                    let gp_vals_file = stringify!([<gp_vals_ $regr:snake _ $corr:snake >]);
                    let file_path = format!("{}/{}.npy", test_dir, gp_vals_file);
                    write_npy(file_path, &gpr_vals).expect("gp vals saved");

                    let gp_vars_file = stringify!([<gp_vars_ $regr:snake _ $corr:snake >]);
                    let file_path = format!("{}/{}.npy", test_dir, gp_vars_file);
                    write_npy(file_path, &gpr_vars).expect("gp vars saved");
                }
            }
        };
    }

    test_gp!(Constant, SquaredExponential);
    test_gp!(Constant, AbsoluteExponential);
    test_gp!(Constant, Matern32);
    test_gp!(Constant, Matern52);

    test_gp!(Linear, SquaredExponential);
    test_gp!(Linear, AbsoluteExponential);
    test_gp!(Linear, Matern32);
    test_gp!(Linear, Matern52);

    test_gp!(Quadratic, SquaredExponential);
    test_gp!(Quadratic, AbsoluteExponential);
    test_gp!(Quadratic, Matern32);
    test_gp!(Quadratic, Matern52);

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
    fn test_griewank() {
        let x = array![[1., 1., 1., 1., 1.], [2., 2., 2., 2., 2.]];
        assert_abs_diff_eq!(array![0.72890641, 1.01387135], griewank(&x), epsilon = 1e-8);
    }

    #[test]
    fn test_kpls_griewank() {
        let dims = [5]; // , 10, 60];
        let nts = [100]; // , 300, 500];
        let lim = array![[-600., 600.]];

        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();

        (0..dims.len()).for_each(|i| {
            let dim = dims[i];
            let nt = nts[i];
            let xlimits = lim.broadcast((dim, 2)).unwrap();

            let prefix = "griewank";
            let xfilename = format!("{test_dir}/{prefix}_xt_{nt}x{dim}.npy");
            let yfilename = format!("{test_dir}/{prefix}_yt_{nt}x1.npy");

            let rng = Xoshiro256Plus::seed_from_u64(42);
            let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
            write_npy(xfilename, &xt).expect("cannot save xt");
            let yt = griewank(&xt);
            write_npy(yfilename, &yt).expect("cannot save yt");

            let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
                ConstantMean::default(),
                SquaredExponentialCorr::default(),
            )
            .kpls_dim(Some(3))
            .fit(&Dataset::new(xt, yt))
            .expect("GP fit error");

            let rng = Xoshiro256Plus::seed_from_u64(0);
            let xtest = Lhs::new(&xlimits).with_rng(rng).sample(100);
            //let xtest = Array2::ones((1, dim));
            let ytest = gp.predict(&xtest).expect("prediction error");
            let ytrue = griewank(&xtest);

            let nrmse = (ytrue.to_owned() - &ytest).norm_l2() / ytrue.norm_l2();
            println!(
                "diff={}  ytrue={} nrsme={}",
                (ytrue.to_owned() - &ytest).norm_l2(),
                ytrue.norm_l2(),
                nrmse
            );
            assert_abs_diff_eq!(nrmse, 0., epsilon = 1e-2);
        });
    }

    fn tensor_product_exp(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<f64> {
        x.mapv(|v| v.exp()).map_axis(Axis(1), |row| row.product())
    }

    #[test]
    fn test_kpls_tp_exp() {
        let dim = 3;
        let nt = 300;
        let lim = array![[-1., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let yt = tensor_product_exp(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .kpls_dim(Some(1))
        .fit(&Dataset::new(xt, yt))
        .expect("GP training");

        let xv = Lhs::new(&xlimits).sample(100);
        let yv = tensor_product_exp(&xv);

        let ytest = gp.predict(&xv).unwrap();
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 2e-2);
    }

    fn rosenb(x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<f64> {
        let mut y: Array1<f64> = Array1::zeros((x.nrows(),));
        Zip::from(&mut y).and(x.rows()).par_for_each(|yi, xi| {
            *yi = rosenbrock(&xi.to_vec());
        });
        y
    }

    #[test]
    fn test_kpls_rosenb() {
        let dim = 20;
        let nt = 30;
        let lim = array![[-1., 1.]];
        let xlimits = lim.broadcast((dim, 2)).unwrap();
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
        let yt = rosenb(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, Matern32Corr>::params(
            ConstantMean::default(),
            Matern52Corr::default(),
        )
        .kpls_dim(Some(1))
        .fit(&Dataset::new(xt.to_owned(), yt))
        .expect("GP training");

        let rng2 = Xoshiro256Plus::seed_from_u64(41);
        let xv = Lhs::new(&xlimits).with_rng(rng2).sample(300);
        let yv = rosenb(&xv);

        let ytest = gp.predict(&xv).expect("GP prediction");
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 4e-1);

        let var = GpVariancePredictor(&gp).predict(&xt);
        assert_abs_diff_eq!(var, Array1::zeros(nt), epsilon = 2e-1);
    }

    fn sphere(x: &Array2<f64>) -> Array1<f64> {
        (x * x).sum_axis(Axis(1))
    }

    fn dsphere(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| 2. * v)
    }

    fn norm1(x: &Array2<f64>) -> Array1<f64> {
        x.mapv(|v| v.abs()).sum_axis(Axis(1)).to_owned()
    }

    fn dnorm1(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0. { 1. } else { -1. })
    }

    macro_rules! test_gp_derivatives {
        ($regr:ident, $corr:ident, $func:ident, $limit:expr_2021, $nt:expr_2021) => {
            paste! {

                #[test]
                fn [<test_gp_derivatives_ $regr:snake _ $corr:snake>]() {
                    let mut rng = Xoshiro256Plus::seed_from_u64(42);
                    let xt = egobox_doe::Lhs::new(&array![[-$limit, $limit], [-$limit, $limit]])
                    .kind(egobox_doe::LhsKind::CenteredMaximin)
                    .with_rng(rng.clone())
                    .sample($nt);

                    let yt = [<$func>](&xt);
                    let gp = GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                        [<$regr Mean>]::default(),
                        [<$corr Corr>]::default(),
                    )
                    .fit(&Dataset::new(xt, yt))
                    .expect("GP fitting");

                    let x = Array::random_using((2,), Uniform::new(-$limit, $limit), &mut rng);
                    //let x = array![3., 5.];
                    let xa: f64 = x[0];
                    let xb: f64 = x[1];
                    let e = 1e-5;

                    let x = array![
                        [xa, xb],
                        [xa + e, xb],
                        [xa - e, xb],
                        [xa, xb + e],
                        [xa, xb - e]
                    ];

                    let y_pred = gp.predict(&x).unwrap();
                    println!("value at [{},{}] = {}", xa, xb, y_pred);
                    let y_deriv = gp.predict_gradients(&x);
                    println!("deriv at [{},{}] = {}", xa, xb, y_deriv);
                    let true_deriv = [<d $func>](&array![[xa, xb]]);
                    println!("true deriv at [{},{}] = {}", xa, xb, true_deriv);
                    println!("jacob = at [{},{}] = {}", xa, xb, gp.predict_jacobian(&array![xa, xb]));

                    let diff_g = (y_pred[1] - y_pred[2]) / (2. * e);
                    let diff_d = (y_pred[3] - y_pred[4]) / (2. * e);

                    // test only if fdiff is not largely wrong
                    if (diff_g-true_deriv[[0, 0]]).abs() < 10. {
                        assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
                    }
                    if (diff_d-true_deriv[[0, 1]]).abs() < 10. {
                        assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);
                    }
                }
            }
        };
    }

    test_gp_derivatives!(Constant, SquaredExponential, sphere, 10., 10);
    test_gp_derivatives!(Linear, SquaredExponential, sphere, 10., 10);
    test_gp_derivatives!(Quadratic, SquaredExponential, sphere, 10., 10);
    test_gp_derivatives!(Constant, AbsoluteExponential, sphere, 10., 10);
    test_gp_derivatives!(Linear, AbsoluteExponential, norm1, 10., 16);
    test_gp_derivatives!(Quadratic, AbsoluteExponential, norm1, 10., 16);
    test_gp_derivatives!(Constant, Matern32, norm1, 10., 16);
    test_gp_derivatives!(Linear, Matern32, norm1, 10., 16);
    test_gp_derivatives!(Quadratic, Matern32, sphere, 10., 16);
    test_gp_derivatives!(Constant, Matern52, norm1, 10., 16);
    test_gp_derivatives!(Linear, Matern52, norm1, 10., 16);
    test_gp_derivatives!(Quadratic, Matern52, sphere, 10., 10);

    #[allow(unused_macros)]
    macro_rules! test_gp_variance_derivatives {
        ($regr:ident, $corr:ident, $func:ident, $limit:expr_2021, $nt:expr_2021) => {
            paste! {

                #[test]
                fn [<test_gp_variance_derivatives_ $regr:snake _ $corr:snake _ $func:snake>]() {
                    let mut rng = Xoshiro256Plus::seed_from_u64(42);
                    let xt = egobox_doe::Lhs::new(&array![[-$limit, $limit], [-$limit, $limit]]).with_rng(rng.clone()).sample($nt);
                    let yt = [<$func>](&xt);
                    println!(stringify!(<$func>));

                    let gp = GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                        [<$regr Mean>]::default(),
                        [<$corr Corr>]::default(),
                    )
                    .fit(&Dataset::new(xt, yt))
                    .expect("GP fitting");

                    for _ in 0..10 {
                        let x = Array::random_using((2,), Uniform::new(-$limit, $limit), &mut rng);
                        let xa: f64 = x[0];
                        let xb: f64 = x[1];
                        let e = 1e-5;

                        let x = array![
                            [xa, xb],
                            [xa + e, xb],
                            [xa - e, xb],
                            [xa, xb + e],
                            [xa, xb - e]
                        ];
                        println!("****************************************");
                        let y_pred = gp.predict(&x).unwrap();
                        println!("value at [{},{}] = {}", xa, xb, y_pred);
                        let y_deriv = gp.predict_gradients(&x);
                        println!("deriv at [{},{}] = {}", xa, xb, y_deriv);
                        let y_pred = gp.predict_var(&x).unwrap();
                        println!("variance at [{},{}] = {}", xa, xb, y_pred);
                        let y_deriv = gp.predict_var_gradients(&x);
                        println!("variance deriv at [{},{}] = {}", xa, xb, y_deriv);

                        let diff_g = (y_pred[1] - y_pred[2]) / (2. * e);
                        let diff_d = (y_pred[3] - y_pred[4]) / (2. * e);

                        assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
                        assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);
                    }
                }
            }
        };
    }

    test_gp_variance_derivatives!(Constant, SquaredExponential, sphere, 10., 100);
    test_gp_variance_derivatives!(Linear, SquaredExponential, sphere, 10., 100);
    test_gp_variance_derivatives!(Quadratic, SquaredExponential, sphere, 10., 100);
    // FIXME: exclude as it fails on testing-features CI: blas, nlopt...
    #[cfg(not(feature = "nlopt"))]
    test_gp_variance_derivatives!(Constant, AbsoluteExponential, norm1, 10., 100);
    test_gp_variance_derivatives!(Linear, AbsoluteExponential, norm1, 1., 50);
    test_gp_variance_derivatives!(Quadratic, AbsoluteExponential, sphere, 10., 100);
    test_gp_variance_derivatives!(Constant, Matern32, sphere, 10., 100);
    test_gp_variance_derivatives!(Linear, Matern32, norm1, 1., 50);
    test_gp_variance_derivatives!(Quadratic, Matern32, sphere, 10., 100);
    test_gp_variance_derivatives!(Constant, Matern52, sphere, 10., 100);
    test_gp_variance_derivatives!(Linear, Matern52, norm1, 1., 50);
    test_gp_variance_derivatives!(Quadratic, Matern52, sphere, 10., 100);

    #[test]
    fn test_variance_derivatives() {
        let xt = egobox_doe::FullFactorial::new(&array![[-10., 10.], [-10., 10.]]).sample(10);
        let yt = sphere(&xt);

        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .fit(&Dataset::new(xt, yt))
        .expect("GP fitting");

        for _ in 0..20 {
            let mut rng = Xoshiro256Plus::seed_from_u64(42);
            let x = Array::random_using((2,), Uniform::new(-10., 10.), &mut rng);
            let xa: f64 = x[0];
            let xb: f64 = x[1];
            let e = 1e-5;

            let x = array![
                [xa, xb],
                [xa + e, xb],
                [xa - e, xb],
                [xa, xb + e],
                [xa, xb - e]
            ];
            let y_pred = gp.predict_var(&x).unwrap();
            println!("variance at [{xa},{xb}] = {y_pred}");
            let y_deriv = gp.predict_var_gradients(&x);
            println!("variance deriv at [{xa},{xb}] = {y_deriv}");

            let diff_g = (y_pred[1] - y_pred[2]) / (2. * e);
            let diff_d = (y_pred[3] - y_pred[4]) / (2. * e);

            if y_pred[0].abs() > 1e-1 && y_pred[0].abs() > 1e-1 {
                // do not test with fdiff when variance or deriv is too small
                assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
            }
            if y_pred[0].abs() > 1e-1 && y_pred[0].abs() > 1e-1 {
                // do not test with fdiff when variance or deriv  is too small
                assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);
            }
        }
    }

    #[test]
    fn test_fixed_theta() {
        let xt = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let yt = array![0.0, 1.0, 1.5, 0.9, 1.0];
        let gp = Kriging::params()
            .fit(&Dataset::new(xt.clone(), yt.clone()))
            .expect("GP fit error");
        let default = ThetaTuning::default();
        assert_abs_diff_ne!(*gp.theta(), default.init());
        let expected = gp.theta();

        let gp = Kriging::params()
            .theta_tuning(ThetaTuning::Fixed(expected.clone()))
            .fit(&Dataset::new(xt, yt))
            .expect("GP fit error");
        assert_abs_diff_eq!(*gp.theta(), expected);
    }

    fn x2sinx(x: &Array2<f64>) -> Array1<f64> {
        ((x * x) * (x).mapv(|v| v.sin())).remove_axis(Axis(1))
    }

    #[test]
    fn test_sampling() {
        let xdoe = array![[-8.5], [-4.0], [-3.0], [-1.0], [4.0], [7.5]];
        let ydoe = x2sinx(&xdoe);
        let krg = Kriging::<f64>::params()
            .fit(&Dataset::new(xdoe, ydoe))
            .expect("Kriging training");
        let n_plot = 35;
        let n_traj = 10;
        let (x_min, x_max) = (-10., 10.);
        let x = Array::linspace(x_min, x_max, n_plot)
            .into_shape_with_order((n_plot, 1))
            .unwrap();
        let trajs = krg.sample(&x, n_traj);
        assert_eq!(&[n_plot, n_traj], trajs.shape())
    }

    #[test]
    fn test_sampling_eigen() {
        let xdoe = array![[-8.5], [-4.0], [-3.0], [-1.0], [4.0], [7.5]];
        let ydoe = x2sinx(&xdoe);
        let krg = Kriging::<f64>::params()
            .fit(&Dataset::new(xdoe, ydoe))
            .expect("Kriging training");
        let n_plot = 500;
        let n_traj = 10;
        let (x_min, x_max) = (-10., 10.);
        let x = Array::linspace(x_min, x_max, n_plot)
            .into_shape_with_order((n_plot, 1))
            .unwrap();
        let trajs = krg.sample_eig(&x, n_traj);
        assert_eq!(&[n_plot, n_traj], trajs.shape());
        assert!(!trajs.fold(false, |acc, v| acc || v.is_nan())); // check no nans
    }

    fn assert_rel_or_abs_error(y_deriv: f64, fdiff: f64) {
        println!("analytic deriv = {y_deriv}, fdiff = {fdiff}");
        if fdiff.abs() < 1. {
            let atol = 1.;
            println!("Check absolute error: abs({y_deriv}) should be < {atol}");
            assert_abs_diff_eq!(y_deriv, 0.0, epsilon = atol); // check absolute when close to zero
        } else {
            let rtol = 6e-1;
            let rel_error = (y_deriv - fdiff).abs() / fdiff.abs(); // check relative
            println!("Check relative error: {rel_error} should be < {rtol}");
            assert_abs_diff_eq!(rel_error, 0.0, epsilon = rtol);
        }
    }

    fn sin_linear(x: &Array2<f64>) -> Array2<f64> {
        // sin + linear trend
        let x1 = x.column(0).to_owned().mapv(|v| v.sin());
        let x2 = x.column(0).mapv(|v| 2. * v) + x.column(1).mapv(|v| 5. * v);
        (x1 + x2)
            .mapv(|v| v + 10.)
            .into_shape_with_order((x.nrows(), 1))
            .unwrap()
    }

    #[test]
    fn test_bug_var_derivatives() {
        let _xt = egobox_doe::Lhs::new(&array![[-5., 10.], [-5., 10.]])
            .kind(LhsKind::Centered)
            .sample(12);
        let _yt = sin_linear(&_xt);

        let xt = array![
            [6.875, -4.375],
            [-3.125, 1.875],
            [1.875, -1.875],
            [-4.375, 3.125],
            [8.125, 9.375],
            [4.375, 4.375],
            [0.625, 0.625],
            [9.375, 6.875],
            [5.625, 8.125],
            [-0.625, -3.125],
            [3.125, 5.625],
            [-1.875, -0.625]
        ];
        let yt = array![
            2.43286801,
            13.10840811,
            5.32908578,
            17.81862219,
            74.08849877,
            39.68137781,
            14.96009727,
            63.17475741,
            61.26331775,
            -7.46009727,
            44.39159189,
            2.17091422,
        ];

        let gp = GaussianProcess::<f64, ConstantMean, SquaredExponentialCorr>::params(
            ConstantMean::default(),
            SquaredExponentialCorr::default(),
        )
        .theta_tuning(ThetaTuning::Fixed(array![
            f64::sqrt(2. * 0.0437386),
            f64::sqrt(2. * 0.00697978)
        ]))
        .fit(&Dataset::new(xt, yt))
        .expect("GP fitting");

        let e = 5e-6;
        let xa = -1.3;
        let xb = 2.5;
        let x = array![
            [xa, xb],
            [xa + e, xb],
            [xa - e, xb],
            [xa, xb + e],
            [xa, xb - e]
        ];
        let y_pred = gp.predict_var(&x).unwrap();
        let y_deriv = gp.predict_var_gradients(&array![[xa, xb]]);
        let diff_g = (y_pred[1] - y_pred[2]) / (2. * e);
        let diff_d = (y_pred[3] - y_pred[4]) / (2. * e);

        assert_abs_diff_eq!(y_deriv[[0, 0]], diff_g, epsilon = 1e-5);
        assert_abs_diff_eq!(y_deriv[[0, 1]], diff_d, epsilon = 1e-5);
    }
}

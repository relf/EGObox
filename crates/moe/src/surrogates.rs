use crate::errors::Result;
#[cfg(feature = "persistent")]
use crate::types::GpFileFormat;
use egobox_gp::{
    GaussianProcess, GpParams, SgpParams, SparseGaussianProcess, SparseMethod, ThetaTuning,
    correlation_models::*, mean_models::*,
};
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use paste::paste;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "persistent")]
use crate::MoeError;
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;
/// A trait for Gp surrogate parameters to build surrogate.
pub trait GpSurrogateParams {
    /// Set theta
    fn theta_tuning(&mut self, theta_tuning: ThetaTuning<f64>);
    /// Set the number of PLS components
    fn kpls_dim(&mut self, kpls_dim: Option<usize>);
    /// Set the number of internal optimization restarts
    fn n_start(&mut self, n_start: usize);
    /// Set the max number of internal likelihood evaluations per optimization
    fn max_eval(&mut self, max_eval: usize);
    /// Set the nugget parameter to improve numerical stability
    fn nugget(&mut self, nugget: f64);
    /// Train the surrogate
    fn train(&self, x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Result<Box<dyn FullGpSurrogate>>;
}

/// A trait for sparse GP surrogate parameters to build surrogate.
pub trait SgpSurrogateParams: GpSurrogateParams {
    /// Set the sparse method
    fn sparse_method(&mut self, method: SparseMethod);
    /// Set random generator seed
    fn seed(&mut self, seed: Option<u64>);
}

/// A trait for a base GP surrogate
#[cfg_attr(feature = "serializable", typetag::serde(tag = "type_gp"))]
pub trait GpSurrogate: std::fmt::Display + Sync + Send {
    /// Returns input/output dims
    fn dims(&self) -> (usize, usize);
    /// Predict output values at n points given as a vector (n,)..
    fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>>;
    /// Predict variance values at n points given as (n, xdim) matrix.
    fn predict_var(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>>;
    /// Predict both output values and variance at n given `x` points of nx components
    fn predict_valvar(&self, x: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)>;
    /// Save model in given file.
    #[cfg(feature = "persistent")]
    fn save(&self, path: &str, format: GpFileFormat) -> Result<()>;
}

/// A trait for a GP surrogate with derivatives predictions and sampling
#[cfg_attr(feature = "serializable", typetag::serde(tag = "type_gpext"))]
pub trait GpSurrogateExt {
    /// Predict derivatives at n points and return (n, xdim) matrix
    /// where each column is the partial derivatives wrt the ith component
    fn predict_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    /// Predict derivatives of the variance at n points and return (n, xdim) matrix
    /// where each column is the partial derivatives wrt the ith component
    fn predict_var_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>>;
    /// Predict both output values and derivatives of the variance at n points
    /// return ((n, ), (n, xdim)) where first array is the predicted values
    /// and second array is the derivatives of the variance wrt each input dimension
    fn predict_valvar_gradients(&self, x: &ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>)>;
    /// Sample trajectories
    fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> Result<Array2<f64>>;
}

/// A trait for a GP surrogate.
#[cfg_attr(feature = "serializable", typetag::serde(tag = "type_gpparam"))]
pub trait GpParameterized {
    /// Get hyperparameters
    fn theta(&self) -> &Array1<f64>;
    /// Get process variance
    fn variance(&self) -> f64;
    /// Get noise variance (0 for full GP)
    fn noise_variance(&self) -> f64;
    /// Get log-likelihood
    fn likelihood(&self) -> f64;
}

/// A trait for a GP surrogate with update support.
#[cfg_attr(feature = "serializable", typetag::serde(tag = "type_fullgp"))]
#[dyn_clonable::clonable]
pub trait FullGpSurrogate:
    Clone + Sync + Send + GpParameterized + GpSurrogate + GpSurrogateExt
{
    /// Update the GP with new data points efficiently.
    fn update(
        &self,
        x_new: &ndarray::ArrayView2<f64>,
        y_new: &ndarray::ArrayView2<f64>,
    ) -> crate::errors::Result<Box<dyn FullGpSurrogate>>;
}

/// A trait for a Sparse GP surrogate.
#[cfg_attr(feature = "serializable", typetag::serde(tag = "type_sgp"))]
pub trait SgpSurrogate: FullGpSurrogate {}

/// A macro to declare GP surrogate using regression model and correlation model names.
///
/// Regression model is either `Constant`, `Linear` or `Quadratic`.
/// Correlation model is either `SquaredExponential`, `AbsoluteExponential`, `Matern32` or `Matern52`.
macro_rules! declare_surrogate {
    ($regr:ident, $corr:ident) => {
        paste! {

            #[doc(hidden)]
            #[doc = "GP surrogate parameters with `" $regr "` regression model and `" $corr "` correlation model. \n\nSee [GpParams](egobox_gp::GpParams)"]
            #[derive(Clone, Debug)]
            pub struct [<Gp $regr $corr SurrogateParams>](
                GpParams<f64, [<$regr Mean>], [<$corr Corr>]>,
            );

            impl [<Gp $regr $corr SurrogateParams>] {
                /// Constructor
                pub fn new(gp_params: GpParams<f64, [<$regr Mean>], [<$corr Corr>]>) -> [<Gp $regr $corr SurrogateParams>] {
                    [<Gp $regr $corr SurrogateParams>](gp_params)
                }
            }

            impl GpSurrogateParams for [<Gp $regr $corr SurrogateParams>] {
                fn theta_tuning(&mut self, theta_tuning: ThetaTuning<f64>) {
                    self.0 = self.0.clone().theta_tuning(theta_tuning);
                }

                fn kpls_dim(&mut self, kpls_dim: Option<usize>) {
                    self.0 = self.0.clone().kpls_dim(kpls_dim);
                }

                fn n_start(&mut self, n_start: usize) {
                    self.0 = self.0.clone().n_start(n_start);
                }

                fn max_eval(&mut self, max_eval: usize) {
                    self.0 = self.0.clone().max_eval(max_eval);
                }

                fn nugget(&mut self, nugget: f64) {
                    self.0 = self.0.clone().nugget(nugget);
                }

                fn train(
                    &self,
                    x: &ArrayView2<f64>,
                    y: &ArrayView2<f64>,
                ) -> Result<Box<dyn FullGpSurrogate>> {
                    Ok(Box::new([<Gp $regr $corr Surrogate>](
                        self.0.clone().fit(&Dataset::new(x.to_owned(), y.to_owned().remove_axis(Axis(1))))?,
                    )))
                }
            }

            #[doc = "GP surrogate with `" $regr "` regression model and `" $corr "` correlation model. \n\nSee [`GaussianProcess`](egobox_gp::GaussianProcess)"]
            #[derive(Clone, Debug)]
            #[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
            pub struct [<Gp $regr $corr Surrogate>](
                pub GaussianProcess<f64, [<$regr Mean>], [<$corr Corr>]>,
            );

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl GpSurrogate for [<Gp $regr $corr Surrogate>] {
                fn dims(&self) -> (usize, usize) {
                    self.0.dims()
                }
                fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
                    Ok(self.0.predict(x)?)
                }
                fn predict_var(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
                    Ok(self.0.predict_var(x)?)
                }
                fn predict_valvar(&self, x: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
                    Ok(self.0.predict_valvar(x)?)
                }

                #[cfg(feature = "persistent")]
                fn save(&self, path: &str, format: GpFileFormat) -> Result<()> {
                    let mut file = fs::File::create(path).unwrap();
                    let bytes = match format {
                        GpFileFormat::Json => serde_json::to_vec(self as &dyn GpSurrogate)
                            .map_err(MoeError::SaveJsonError)?,
                        GpFileFormat::Binary => {
                            bincode::serde::encode_to_vec(self as &dyn GpSurrogate, bincode::config::standard()).map_err(MoeError::SaveBinaryError)?
                        }
                    };
                    file.write_all(&bytes)?;

                    Ok(())
                }

            }

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl GpSurrogateExt for [<Gp $regr $corr Surrogate>] {
                fn predict_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_gradients(x))
                }
                fn predict_var_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_var_gradients(x))
                }
                fn predict_valvar_gradients(&self, x: &ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
                    Ok(self.0.predict_valvar_gradients(x))
                }
                fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> Result<Array2<f64>> {
                    Ok(self.0.sample(x, n_traj))
                }
            }

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl GpParameterized for [<Gp $regr $corr Surrogate>] {
                fn theta(&self) -> &Array1<f64> {
                    self.0.theta()
                }

                fn variance(&self) -> f64 {
                    self.0.variance()
                }

                fn noise_variance(&self) -> f64 {
                    0.0
                }

                fn likelihood(&self) -> f64 {
                    self.0.likelihood()
                }
            }

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl FullGpSurrogate for [<Gp $regr $corr Surrogate>] {
                fn update(
                    &self,
                    x_new: &ArrayView2<f64>,
                    y_new: &ArrayView2<f64>,
                ) -> Result<Box<dyn FullGpSurrogate>> {
                    // Use the efficient GP update method
                    let updated_gp = self.0.clone().update(x_new, &y_new.remove_axis(Axis(1)))?;
                    Ok(Box::new([<Gp $regr $corr Surrogate>](updated_gp)))
                }
            }

            impl std::fmt::Display for [<Gp $regr $corr Surrogate>] {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}_{}{}{}", stringify!($regr), stringify!($corr),
                        match self.0.kpls_dim() {
                            None => String::from(""),
                            Some(dim) => format!("_PLS({})", dim),
                        },
                        self.0.to_string()
                    )
                }
            }
        }
    };
}

declare_surrogate!(Constant, SquaredExponential);
declare_surrogate!(Constant, AbsoluteExponential);
declare_surrogate!(Constant, Matern32);
declare_surrogate!(Constant, Matern52);
declare_surrogate!(Linear, SquaredExponential);
declare_surrogate!(Linear, AbsoluteExponential);
declare_surrogate!(Linear, Matern32);
declare_surrogate!(Linear, Matern52);
declare_surrogate!(Quadratic, SquaredExponential);
declare_surrogate!(Quadratic, AbsoluteExponential);
declare_surrogate!(Quadratic, Matern32);
declare_surrogate!(Quadratic, Matern52);

/// A macro to declare SGP surrogate using correlation model names.
///
/// Correlation model is either `SquaredExponential`, `AbsoluteExponential`, `Matern32` or `Matern52`.
macro_rules! declare_sgp_surrogate {
    ($corr:ident) => {
        paste! {

            #[doc(hidden)]
            #[doc = "SGP surrogate parameters with `" $corr "` correlation model. \n\nSee [SgpParams](egobox_gp::SgpParams)"]
            #[derive(Clone, Debug)]
            pub struct [<Sgp $corr SurrogateParams>](
                SgpParams<f64, [<$corr Corr>]>,
            );

            impl [<Sgp $corr SurrogateParams>] {
                /// Constructor
                pub fn new(gp_params: SgpParams<f64, [<$corr Corr>]>) -> [<Sgp $corr SurrogateParams>] {
                    [<Sgp $corr SurrogateParams>](gp_params)
                }
            }

            impl GpSurrogateParams for [<Sgp $corr SurrogateParams>] {
                fn theta_tuning(&mut self, theta_tuning: ThetaTuning<f64>) {
                    self.0 = self.0.clone().theta_tuning(theta_tuning);
                }

                fn kpls_dim(&mut self, kpls_dim: Option<usize>) {
                    self.0 = self.0.clone().kpls_dim(kpls_dim);
                }

                fn n_start(&mut self, n_start: usize) {
                    self.0 = self.0.clone().n_start(n_start);
                }

                fn max_eval(&mut self, max_eval: usize) {
                    self.0 = self.0.clone().max_eval(max_eval);
                }

                fn nugget(&mut self, nugget: f64) {
                    self.0 = self.0.clone().nugget(nugget);
                }

                fn train(
                    &self,
                    x: &ArrayView2<f64>,
                    y: &ArrayView2<f64>,
                ) -> Result<Box<dyn FullGpSurrogate>> {
                    Ok(Box::new([<Sgp $corr Surrogate>](
                        self.0.clone().fit(&Dataset::new(x.to_owned(), y.to_owned().remove_axis(Axis(1))))?,
                    )))
                }
            }

            impl SgpSurrogateParams for [<Sgp $corr SurrogateParams>] {
                fn sparse_method(&mut self, method: SparseMethod) {
                    self.0 = self.0.clone().sparse_method(method);
                }

                fn seed(&mut self, seed: Option<u64>) {
                    self.0 = self.0.clone().seed(seed);
                }
            }

            #[doc = "SGP surrogate with `" $corr "` correlation model. \n\nSee [`SparseGaussianProcess`](egobox_gp::SparseGaussianProcess)"]
            #[derive(Clone, Debug)]
            #[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
            pub struct [<Sgp $corr Surrogate>](
                pub SparseGaussianProcess<f64, [<$corr Corr>]>,
            );

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl GpSurrogate for [<Sgp $corr Surrogate>] {
                fn dims(&self) -> (usize, usize) {
                    self.0.dims()
                }
                fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
                    Ok(self.0.predict(x)?)
                }
                fn predict_var(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
                    Ok(self.0.predict_var(x)?)
                }
                fn predict_valvar(&self, x: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
                    Ok((self.0.predict(x)?, self.0.predict_var(x)?))
                }

                #[cfg(feature = "persistent")]
                fn save(&self, path: &str, format: GpFileFormat) -> Result<()> {
                    let mut file = fs::File::create(path).unwrap();
                    let bytes = match format {
                        GpFileFormat::Json => serde_json::to_vec(self as &dyn SgpSurrogate)
                            .map_err(MoeError::SaveJsonError)?,
                        GpFileFormat::Binary => {
                            bincode::serde::encode_to_vec(self as &dyn SgpSurrogate, bincode::config::standard()).map_err(MoeError::SaveBinaryError)?
                        }
                    };
                    file.write_all(&bytes)?;
                    Ok(())
                }
            }

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl GpSurrogateExt for [<Sgp $corr Surrogate>] {
                fn predict_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_gradients(x))
                }
                fn predict_var_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
                    Ok(self.0.predict_var_gradients(x))
                }
                fn predict_valvar_gradients(&self, x: &ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
                    Ok((self.0.predict_gradients(x), self.0.predict_var_gradients(x)))
                }
                fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> Result<Array2<f64>> {
                    Ok(self.0.sample(x, n_traj))
                }
            }

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl GpParameterized for [<Sgp $corr Surrogate>] {
                fn theta(&self) -> &Array1<f64> {
                    self.0.theta()
                }

                fn variance(&self) -> f64 {
                    self.0.variance()
                }

                fn noise_variance(&self) -> f64 {
                    self.0.noise_variance()
                }

                fn likelihood(&self) -> f64 {
                    self.0.likelihood()
                }
            }

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl FullGpSurrogate for [<Sgp $corr Surrogate>] {
                fn update(
                    &self,
                    _x_new: &ArrayView2<f64>,
                    _y_new: &ArrayView2<f64>,
                ) -> Result<Box<dyn FullGpSurrogate>> {
                    // Sparse GP does not support efficient incremental updates.
                    // For now, return an error indicating this limitation.
                    Err(crate::MoeError::GpError(egobox_gp::GpError::InvalidValueError(
                        "Sparse GP update not supported - requires full retraining".to_string()
                    )))
                }
            }

            #[cfg_attr(feature = "serializable", typetag::serde)]
            impl SgpSurrogate for [<Sgp $corr Surrogate>] {}

            impl std::fmt::Display for [<Sgp $corr Surrogate>] {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}{}{}", stringify!($corr),
                        match self.0.kpls_dim() {
                            None => String::from(""),
                            Some(dim) => format!("_PLS({})", dim),
                        },
                        self.0.to_string()
                    )
                }
            }
        }
    };
}

declare_sgp_surrogate!(SquaredExponential);
declare_sgp_surrogate!(AbsoluteExponential);
declare_sgp_surrogate!(Matern32);
declare_sgp_surrogate!(Matern52);

#[cfg(feature = "persistent")]
/// Load GP surrogate from given json file.
pub fn load(path: &str, format: GpFileFormat) -> Result<Box<dyn GpSurrogate>> {
    let data = fs::read(path)?;
    match format {
        GpFileFormat::Json => {
            serde_json::from_slice::<Box<dyn GpSurrogate>>(&data).map_err(|err| {
                MoeError::LoadError(format!("Error while loading from {path}: ({err})"))
            })
        }
        GpFileFormat::Binary => bincode::serde::decode_from_slice::<Box<dyn GpSurrogate>, _>(
            &data,
            bincode::config::standard(),
        )
        .map(|(surrogate, _)| surrogate)
        .map_err(|err| MoeError::LoadError(format!("Error while loading from {path} ({err})"))),
    }
}

#[doc(hidden)]
// Create GP surrogate parameters with given regression and correlation models.
macro_rules! make_surrogate_params {
    ($regr:ident, $corr:ident) => {
        paste! {
            #[allow(unused_allocation)]
            Box::new([<Gp $regr $corr SurrogateParams>]::new(
                GaussianProcess::<f64, [<$regr Mean>], [<$corr Corr>] >::params(
                    [<$regr Mean>]::default(),
                    [<$corr Corr>]::default(),
                )
            ))
        }
    };
}

#[doc(hidden)]
// Create GP surrogate parameters with given regression and correlation models.
macro_rules! make_sgp_surrogate_params {
    ($corr:ident, $inducings:ident) => {
        paste! {
            #[allow(unused_allocation)]
            Box::new([<Sgp $corr SurrogateParams>]::new(
                SparseGaussianProcess::<f64, [<$corr Corr>] >::params(
                    [<$corr Corr>]::default(),
                    $inducings
                )
            ))
        }
    };
}

pub(crate) use make_sgp_surrogate_params;
pub(crate) use make_surrogate_params;

#[cfg(feature = "persistent")]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_doe::{Lhs, SamplingMethod};
    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    use ndarray::array;
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;
    use ndarray_stats::DeviationExt;

    fn xsinx(x: &Array2<f64>) -> Array1<f64> {
        ((x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())).remove_axis(Axis(1))
    }

    #[test]
    fn test_save_load() {
        let xlimits = array![[0., 25.]];
        let xt = Lhs::new(&xlimits).sample(10);
        let yt = xsinx(&xt);
        let gp = make_surrogate_params!(Constant, SquaredExponential)
            .train(&xt.view(), &yt.insert_axis(Axis(1)).view())
            .expect("GP fit error");
        gp.save("target/tests/save_gp.json", GpFileFormat::Json)
            .expect("GP not saved");
        let gp = load("target/tests/save_gp.json", GpFileFormat::Json).expect("GP not loaded");
        let xv = Lhs::new(&xlimits).sample(20);
        let yv = xsinx(&xv);
        let ytest = gp.predict(&xv.view()).unwrap();
        let err = ytest.l2_dist(&yv).unwrap() / yv.norm_l2();
        assert_abs_diff_eq!(err, 0., epsilon = 2e-1);
    }

    #[test]
    fn test_load_fail() {
        let gp = load("notfound.json", GpFileFormat::Json);
        assert!(gp.is_err());
    }
}

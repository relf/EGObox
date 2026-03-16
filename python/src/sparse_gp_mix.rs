//! `egobox`, Rust toolbox for efficient global optimization
//!
//! Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions,
//! the mixture of gaussian process surrogates is binded in Python. You can install the Python package using:
//!
//! ```bash
//! pip install egobox
//! ```
//!
//! See the [tutorial notebook](https://github.com/relf/egobox/doc/Sgp_Tutorial.ipynb) for usage.
//!
use std::path::Path;

use crate::{logging::init_logger, types::*};
use egobox_moe::{
    Clustered, GpMixture, GpSurrogate, GpType, Inducings, MixtureGpSurrogate, ThetaTuning,
};
use linfa::{Dataset, traits::Fit};
use log::error;
use ndarray::{Array1, Array2, Axis, Ix1, Ix2, Zip, array};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rand_xoshiro::Xoshiro256Plus;

/// Sparse Gaussian processes mixture builder
///
///     n_clusters (int >= 0):
///         Number of clusters used by the mixture of surrogate experts.
///         When set to 0, the number of cluster is determined automatically and refreshed every
///         10-points addition (should say 'tentative addition' because addition may fail for some points
///         but failures are counted anyway).
///
///     corr_spec (CorrelationSpec flags, an int in [1, 15]):
///         Specification of correlation models used in mixture.
///         Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
///         CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
///         any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
///
///     recombination (Recombination.Smooth or Recombination.Hard):
///         Specify how the various experts predictions are recombined
///         * Smooth: prediction is a combination of experts prediction wrt their responsabilities,
///         the heaviside factor which controls steepness of the change between experts regions is optimized
///         to get best mixture quality.
///         * Hard: prediction is taken from the expert with highest responsability
///         resulting in a model with discontinuities.
///
///     kpls_dim (0 < int < nx where nx is the dimension of inputs x):
///         Number of components to be used when PLS projection is used (a.k.a KPLS method).
///         This is used to address high-dimensional problems typically when nx > 9.
///
///     n_start (int >= 0):
///         Number of internal GP hyperpameters optimization restart (multistart)
///
///     method (SparseMethod.FITC or SparseMethod.VFE):
///         Sparse method to be used (default is FITC)
///
///     seed (int >= 0):
///         Random generator seed to allow computation reproducibility.
///
///     verbose (Verbose or int in [0, 4]):
///         Optional verbose level to control logging output (default is 0)
///         Used mainly for debugging and development purposes
///         
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct SparseGpMix {
    pub correlation_spec: CorrelationSpec,
    pub theta_init: Option<Vec<f64>>,
    pub theta_bounds: Option<Vec<Vec<f64>>>,
    pub kpls_dim: Option<usize>,
    pub n_start: usize,
    pub nz: Option<usize>,
    pub z: Option<Array2<f64>>,
    pub method: SparseMethod,
    pub seed: Option<u64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl SparseGpMix {
    #[new]
    #[pyo3(signature = (
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        theta_init = None,
        theta_bounds = None,
        kpls_dim = None,
        n_start = 10,
        nz = None,
        z = None,
        method = SparseMethod::Fitc,
        seed = None,
        verbose = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        corr_spec: u8,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        kpls_dim: Option<usize>,
        n_start: usize,
        nz: Option<usize>,
        z: Option<PyReadonlyArray2<f64>>,
        method: SparseMethod,
        seed: Option<u64>,
        verbose: Option<Py<PyAny>>,
    ) -> Self {
        init_logger(py, verbose);
        SparseGpMix {
            correlation_spec: CorrelationSpec(corr_spec),
            theta_init,
            theta_bounds,
            kpls_dim,
            n_start,
            nz,
            z: z.map(|z| z.as_array().to_owned()),
            method,
            seed,
        }
    }

    /// Fit the parameters of the model using the training dataset to build a trained model
    ///
    /// # Parameters
    ///     xt (array[nsamples, nx]): input samples
    ///     yt (array[nsamples, 1]): output samples
    ///
    /// # Returns Sgp object
    ///     the fitted Gaussian process mixture  
    ///
    fn fit(
        &mut self,
        py: Python,
        xt: PyReadonlyArray2<f64>,
        yt: PyReadonlyArray2<f64>,
    ) -> SparseGpx {
        let xt = xt.as_array();
        let xt = match xt.to_owned().into_dimensionality::<Ix2>() {
            Ok(xt) => xt,
            Err(_) => match xt.into_dimensionality::<Ix1>() {
                Ok(xt) => xt.insert_axis(Axis(1)).to_owned(),
                _ => {
                    error!("Training input has to be an [nsamples, nx] array");
                    panic!("Bad training input data");
                }
            },
        };

        let yt = yt.as_array();
        let yt = match yt.to_owned().into_dimensionality::<Ix1>() {
            Ok(yt) => yt,
            Err(_) => match yt.into_dimensionality::<Ix2>() {
                Ok(yt) => {
                    if yt.dim().1 == 1 {
                        yt.to_owned().remove_axis(Axis(1))
                    } else {
                        error!("Training output has to be one dimensional");
                        panic!("Bad training output data");
                    }
                }
                Err(_) => {
                    error!("Training output has to be one dimensional");
                    panic!("Bad training output data");
                }
            },
        };

        let dataset = Dataset::new(xt, yt);

        let rng = if let Some(seed) = self.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };

        let inducings = if let Some(z) = self.z.as_ref() {
            Inducings::Located(z.clone())
        } else if let Some(nz) = self.nz {
            Inducings::Randomized(nz)
        } else {
            panic!("You must specify inducing points")
        };

        let method = match self.method {
            SparseMethod::Fitc => egobox_gp::SparseMethod::Fitc,
            SparseMethod::Vfe => egobox_gp::SparseMethod::Vfe,
        };

        let mut theta_tuning = ThetaTuning::default();
        if let Some(init) = self.theta_init.as_ref() {
            theta_tuning = ThetaTuning::Full {
                init: Array1::from_vec(init.to_vec()),
                bounds: array![ThetaTuning::<f64>::DEFAULT_BOUNDS],
            }
        }
        if let Some(bounds) = self.theta_bounds.as_ref() {
            theta_tuning = ThetaTuning::Full {
                init: theta_tuning.init().to_owned(),
                bounds: bounds.iter().map(|v| (v[0], v[1])).collect(),
            }
        }
        let theta_tunings = vec![theta_tuning];

        if let Err(ctrlc::Error::MultipleHandlers) = ctrlc::set_handler(|| std::process::exit(2)) {
            // ignore multiple handlers error
        };
        let sgp = py.detach(|| {
            GpMixture::params()
                .gp_type(GpType::SparseGp {
                    sparse_method: method,
                    inducings,
                })
                .correlation_spec(
                    egobox_moe::CorrelationSpec::from_bits(self.correlation_spec.0).unwrap(),
                )
                .theta_tunings(&theta_tunings)
                .kpls_dim(self.kpls_dim)
                .n_start(self.n_start)
                .with_rng(rng)
                .fit(&dataset)
                .expect("Sgp model training")
        });
        SparseGpx(Box::new(sgp))
    }
}

/// A trained Gaussian processes mixture
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct SparseGpx(Box<GpMixture>);

#[gen_stub_pymethods]
#[pymethods]
impl SparseGpx {
    /// Get Gaussian processes mixture builder aka `GpSparse`
    ///
    /// See `GpSparse` constructor
    #[staticmethod]
    #[pyo3(signature = (
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        theta_init = None,
        theta_bounds = None,
        kpls_dim = None,
        n_start = 10,
        nz = None,
        z = None,
        method = SparseMethod::Fitc,
        seed = None,
        verbose = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn builder(
        py: Python,
        corr_spec: u8,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        kpls_dim: Option<usize>,
        n_start: usize,
        nz: Option<usize>,
        z: Option<PyReadonlyArray2<f64>>,
        method: SparseMethod,
        seed: Option<u64>,
        verbose: Option<Py<PyAny>>,
    ) -> SparseGpMix {
        SparseGpMix::new(
            py,
            corr_spec,
            theta_init,
            theta_bounds,
            kpls_dim,
            n_start,
            nz,
            z,
            method,
            seed,
            verbose,
        )
    }

    /// Returns the String representation from serde json serializer
    fn __repr__(&self) -> String {
        serde_json::to_string(&self.0).unwrap()
    }

    /// Returns a String informal representation
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    /// Save Gaussian processes mixture in a file.
    /// If the filename has .json JSON human readable format is used
    /// otherwise an optimized binary format is used.
    ///
    /// # Parameters
    ///     filename with .json or .bin extension (string)
    ///         file generated in the current directory
    ///
    /// # Returns True if save succeeds otherwise False
    ///
    fn save(&self, filename: String) -> bool {
        let format = match Path::new(&filename).extension().unwrap().to_str().unwrap() {
            "json" => egobox_moe::GpFileFormat::Json,
            _ => egobox_moe::GpFileFormat::Binary,
        };
        self.0.save(&filename, format).is_ok()
    }

    /// Load Gaussian processes mixture from a json file.
    ///
    /// # Parameters
    ///     filename (string)
    ///         json filepath generated by saving a trained Gaussian processes mixture
    ///
    #[staticmethod]
    fn load(filename: String) -> SparseGpx {
        let format = match Path::new(&filename).extension().unwrap().to_str().unwrap() {
            "json" => egobox_moe::GpFileFormat::Json,
            _ => egobox_moe::GpFileFormat::Binary,
        };
        SparseGpx(GpMixture::load(&filename, format).unwrap())
    }

    /// Predict output values at nsamples points.
    ///
    /// # Parameters
    ///     x (array[nsamples, nx])
    ///         input values
    ///
    /// Returns
    ///     the output values at nsamples x points (array[nsamples,])
    ///
    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> Bound<'py, PyArray1<f64>> {
        self.0.predict(&x.as_array()).unwrap().into_pyarray(py)
    }

    /// Predict variances at nsample points.
    ///
    /// # Parameters
    ///     x (array[nsamples, nx])
    ///         input values
    ///
    /// # Returns
    ///     the variances of the output values at nsamples input points (array[nsamples,])
    ///
    fn predict_var<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        self.0
            .predict_var(&x.as_array().to_owned())
            .unwrap()
            .into_pyarray(py)
    }

    /// Predict surrogate output derivatives at nsamples points.
    ///
    /// Implementation note: central finite difference technique
    /// on `predict()` function is used which may be subject to numerical issues
    ///
    /// # Parameters
    ///     x (array[nsamples, nx])
    ///         input values
    ///
    /// # Returns
    ///     the output derivatives at nsamples x points (array[nsamples, nx]) wrt inputs
    ///     The ith column is the partial derivative value wrt to the ith component of x at the given samples.
    ///
    fn predict_gradients<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        self.0
            .predict_gradients(&x.as_array())
            .unwrap()
            .into_pyarray(py)
    }

    /// Predict variance derivatives at nsamples points.
    ///
    /// Implementation note: central finite difference technique
    /// on `predict_var()` function is used which may be subject to numerical issues
    ///
    /// # Parameters
    ///     x (array[nsamples, nx])
    ///         input values
    ///
    /// # Returns
    ///     the variance derivatives at nsamples x points (array[nsamples, nx]) wrt inputs
    ///     The ith column is the partial derivative value wrt to the ith component of x at the given samples.
    ///
    fn predict_var_gradients<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        self.0
            .predict_var_gradients(&x.as_array())
            .unwrap()
            .into_pyarray(py)
    }

    /// Sample gaussian process trajectories.
    ///
    /// # Parameters
    ///     x (array[nsamples, nx])
    ///         locations of the sampled trajectories
    ///     n_traj number of trajectories to generate
    ///
    /// # Returns
    ///     the trajectories as an array[nsamples, n_traj]
    ///
    fn sample<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        n_traj: usize,
    ) -> Bound<'py, PyArray2<f64>> {
        self.0
            .sample(&x.as_array(), n_traj)
            .unwrap()
            .into_pyarray(py)
    }

    /// Get optimized thetas hyperparameters (ie once GP experts are fitted)
    ///
    /// # Returns
    ///     thetas as an array[n_clusters, nx or kpls_dim]
    ///
    fn thetas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let experts = self.0.experts();
        let proto = experts.first().expect("Mixture should contain an expert");
        let mut thetas = Array2::zeros((self.0.n_clusters(), proto.theta().len()));
        Zip::from(thetas.rows_mut())
            .and(experts)
            .for_each(|mut theta, expert| theta.assign(expert.theta()));
        thetas.into_pyarray(py)
    }

    /// Get GP expert variance (ie posterior GP variance)
    ///
    /// # Returns
    ///     variances as an array[n_clusters]
    ///
    fn variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let experts = self.0.experts();
        let mut variances = Array1::zeros(self.0.n_clusters());
        Zip::from(&mut variances)
            .and(experts)
            .for_each(|var, expert| *var = expert.variance());
        variances.into_pyarray(py)
    }

    /// Get reduced likelihood values gotten when fitting the GP experts
    ///
    /// Maybe used to compare various parameterization
    ///
    /// # Returns
    ///     likelihood as an array[n_clusters]
    ///
    fn likelihoods<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let experts = self.0.experts();
        let mut likelihoods = Array1::zeros(self.0.n_clusters());
        Zip::from(&mut likelihoods)
            .and(experts)
            .for_each(|lkh, expert| *lkh = expert.likelihood());
        likelihoods.into_pyarray(py)
    }
}

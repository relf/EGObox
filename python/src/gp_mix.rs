//! `egobox`, Rust toolbox for efficient global optimization
//!
//! Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions,
//! the mixture of gaussian process surrogates is binded in Python. You can install the Python package using:
//!
//! ```bash
//! pip install egobox
//! ```
//!
//! See the [tutorial notebook](https://github.com/relf/egobox/doc/Gpx_Tutorial.ipynb) for usage.
//!
use std::{cmp::Ordering, path::Path};

use crate::types::*;
use crate::{domain::parse, gp_config::GpConfig};
use egobox_ego::{EGO_GP_OPTIM_MAX_EVAL, EGO_GP_OPTIM_N_START};
#[allow(unused_imports)] // Avoid linting problem
use egobox_moe::{GpMixture, GpSurrogate, GpSurrogateExt};
use egobox_moe::{MixintGpMixture, MixtureGpSurrogate, NbClusters, ThetaTuning};
use linfa::{Dataset, traits::Fit};
use log::error;
use ndarray::{Array1, Array2, Axis, Ix1, Ix2, Zip, array};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rand_xoshiro::Xoshiro256Plus;

#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct GpMix {
    gp_config: GpConfig,
    xtypes: Option<Vec<egobox_moe::XType>>,
    seed: Option<u64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl GpMix {
    /// Gaussian processes mixture builder
    ///
    /// # Parameters
    ///
    ///     xspecs (list(XSpec)) where XSpec(xtype=FLOAT|INT|ORD|ENUM, xlimits=[<f(xtype)>] or tags=[strings]):
    ///         Specifications of the nx components of the input x (eg. len(xspecs) == nx)
    ///         Depending on the x type we get the following for xlimits:
    ///         * when FLOAT: xlimits is [float lower_bound, float upper_bound],
    ///         * when INT: xlimits is [int lower_bound, int upper_bound],
    ///         * when ORD: xlimits is [float_1, float_2, ..., float_n],
    ///         * when ENUM: xlimits is just the int size of the enumeration otherwise a list of tags is specified
    ///           (eg xlimits=[3] or tags=["red", "green", "blue"], tags are there for documention purpose but
    ///            tags specific values themselves are not used only indices in the enum are used hence
    ///            we can just specify the size of the enum, xlimits=[3]),
    ///
    ///         When None, inputs are expected to be floats and no input space restriction is applied
    ///         (ie. xlimits is [-inf, inf] for all components).
    ///
    ///     regr_spec (RegressionSpec flags, an int in [1, 7]):
    ///         Specification of regression models used in mixture.
    ///         Can be RegressionSpec.CONSTANT (1), RegressionSpec.LINEAR (2), RegressionSpec.QUADRATIC (4) or
    ///         any bit-wise union of these values (e.g. RegressionSpec.CONSTANT | RegressionSpec.LINEAR)
    ///
    ///     corr_spec (CorrelationSpec flags, an int in [1, 15]):
    ///         Specification of correlation models used in mixture.
    ///         Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
    ///         CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
    ///         any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
    ///
    ///     n_clusters (int):
    ///         Number of clusters used by the mixture of surrogate experts (default is 1).
    ///         When set to 0, the number of cluster is determined automatically and refreshed every
    ///         10-points addition (should say 'tentative addition' because addition may fail for some points
    ///         but it is counted anyway).
    ///         When set to negative number -n, the number of clusters is determined automatically in [1, n]
    ///         this is used to limit the number of trials hence the execution time.
    ///
    ///     recombination (Recombination.Smooth or Recombination.Hard (default)):
    ///         Specify how the various experts predictions are recombined
    ///         * Smooth: prediction is a combination of experts prediction wrt their responsabilities,
    ///         the heaviside factor which controls steepness of the change between experts regions is optimized
    ///         to get best mixture quality.
    ///         * Hard: prediction is taken from the expert with highest responsability
    ///         resulting in a model with discontinuities.
    ///
    ///     theta_init ([nx] where nx is the dimension of inputs x):
    ///         Initial guess for GP theta hyperparameters.
    ///         When None the default is 1e-2 for all components
    ///
    ///     theta_bounds ([[lower_1, upper_1], ..., [lower_nx, upper_nx]] where nx is the dimension of inputs x):
    ///         Space search when optimizing theta GP hyperparameters
    ///         When None the default is [1e-6, 1e2] for all components
    ///
    ///     kpls_dim (0 < int < nx where nx is the dimension of inputs x):
    ///         Number of components to be used when PLS projection is used (a.k.a KPLS method).
    ///         This is used to address high-dimensional problems typically when nx > 9.
    ///
    ///     n_start (int >= 0):
    ///         Number of internal GP hyperpameters optimization restart (multistart)
    ///
    ///     max_eval (int >= 0):
    ///         Max number of likelihood evaluations during GP hyperparameters optimization
    ///
    ///     seed (int >= 0):
    ///         Random generator seed to allow computation reproducibility.
    ///
    /// # Returns
    ///
    ///     GpMix object which can be fitted to data to get a Gpx object (a trained Gaussian processes mixture)
    ///         
    #[new]
    #[pyo3(signature = (
        xspecs=None,
        regr_spec=RegressionSpec::CONSTANT,
        corr_spec=CorrelationSpec::SQUARED_EXPONENTIAL,
        kpls_dim=None,
        n_clusters=1,
        recombination=Recombination::Hard,
        theta_init=None,
        theta_bounds=None,
        n_start=EGO_GP_OPTIM_N_START as isize,
        max_eval=EGO_GP_OPTIM_MAX_EVAL,
        seed=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        xspecs: Option<Py<PyAny>>,
        regr_spec: u8,
        corr_spec: u8,
        kpls_dim: Option<usize>,
        n_clusters: isize,
        recombination: Recombination,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        n_start: isize,
        max_eval: usize,
        seed: Option<u64>,
    ) -> Self {
        let xtypes = xspecs
            .as_ref()
            .map(|xspecs| parse(py, xspecs.clone_ref(py)));
        GpMix {
            gp_config: GpConfig::new(
                regr_spec,
                corr_spec,
                kpls_dim,
                n_clusters,
                recombination,
                theta_init,
                theta_bounds,
                n_start,
                max_eval,
            ),
            xtypes,
            seed,
        }
    }

    /// Fit the parameters of the model using the training dataset to build a trained model
    ///
    /// # Parameters
    ///     xt (array[nsamples, nx]): input samples
    ///     yt (array[nsamples, 1]): output samples
    ///
    /// # Returns Gpx object
    ///     the fitted Gaussian process mixture  
    ///
    fn fit(&mut self, py: Python, xt: PyReadonlyArrayDyn<f64>, yt: PyReadonlyArrayDyn<f64>) -> Gpx {
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

        let recomb = match self.gp_config.recombination {
            Recombination::Hard => egobox_moe::Recombination::Hard,
            Recombination::Smooth => egobox_moe::Recombination::Smooth(None),
        };
        let rng = if let Some(seed) = self.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };

        let mut theta_tuning = ThetaTuning::default();
        if let Some(init) = self.gp_config.theta_init.as_ref() {
            theta_tuning = ThetaTuning::Full {
                init: Array1::from_vec(init.to_vec()),
                bounds: array![ThetaTuning::<f64>::DEFAULT_BOUNDS],
            }
        }
        if let Some(bounds) = self.gp_config.theta_bounds.as_ref() {
            theta_tuning = ThetaTuning::Full {
                init: theta_tuning.init().to_owned(),
                bounds: bounds.iter().map(|v| (v[0], v[1])).collect(),
            }
        }

        let n_clusters = match self.gp_config.n_clusters.cmp(&0) {
            Ordering::Greater => NbClusters::fixed(self.gp_config.n_clusters as usize),
            Ordering::Equal => NbClusters::auto(),
            Ordering::Less => NbClusters::automax(-self.gp_config.n_clusters as usize),
        };
        let n_start: usize = if self.gp_config.n_start < 0 {
            // no multistart, use theta_init as is
            theta_tuning = ThetaTuning::Fixed(theta_tuning.init().to_owned());
            0 // no multistart, value not used
        } else {
            self.gp_config.n_start.try_into().unwrap()
        };

        let theta_tunings = if let NbClusters::Fixed { nb } = n_clusters {
            vec![theta_tuning; nb]
        } else {
            vec![theta_tuning; 1] // used as default theta tuning for all experts
        };

        if let Err(ctrlc::Error::MultipleHandlers) = ctrlc::set_handler(|| std::process::exit(2)) {
            // ignore multiple handlers error
        };
        let moe = py.detach(|| {
            let regr = RegressionSpec(self.gp_config.regr_spec);
            let corr = CorrelationSpec(self.gp_config.corr_spec);
            if let Some(xtypes) = self.xtypes.as_ref() {
                Box::new(
                    MixintGpMixture::params(xtypes)
                        .n_clusters(n_clusters)
                        .recombination(recomb)
                        .regression_spec(egobox_moe::RegressionSpec::from_bits(regr.0).unwrap())
                        .correlation_spec(egobox_moe::CorrelationSpec::from_bits(corr.0).unwrap())
                        .theta_tunings(&theta_tunings)
                        .kpls_dim(self.gp_config.kpls_dim)
                        .n_start(n_start)
                        .with_rng(rng)
                        .fit(&dataset)
                        .expect("MoE model training"),
                ) as Box<dyn MixtureGpSurrogate>
            } else {
                Box::new(
                    GpMixture::params()
                        .n_clusters(n_clusters)
                        .recombination(recomb)
                        .regression_spec(egobox_moe::RegressionSpec::from_bits(regr.0).unwrap())
                        .correlation_spec(egobox_moe::CorrelationSpec::from_bits(corr.0).unwrap())
                        .theta_tunings(&theta_tunings)
                        .kpls_dim(self.gp_config.kpls_dim)
                        .n_start(n_start)
                        .with_rng(rng)
                        .fit(&dataset)
                        .expect("MoE model training"),
                ) as Box<dyn MixtureGpSurrogate>
            }
        });

        Gpx(moe)
    }
}

/// A trained Gaussian processes mixture
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct Gpx(Box<dyn MixtureGpSurrogate>);

#[gen_stub_pymethods]
#[pymethods]
impl Gpx {
    /// Get Gaussian processes mixture builder aka `GpMix`
    ///
    /// See `GpMix` constructor
    #[staticmethod]
    #[pyo3(signature = (
        xspecs=None,
        regr_spec=GpConfig::default().regr_spec,
        corr_spec=GpConfig::default().corr_spec,
        kpls_dim=GpConfig::default().kpls_dim,
        n_clusters=GpConfig::default().n_clusters,
        recombination=GpConfig::default().recombination,
        theta_init=GpConfig::default().theta_init,
        theta_bounds=GpConfig::default().theta_bounds,
        n_start=GpConfig::default().n_start,
        max_eval=GpConfig::default().max_eval,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn builder(
        py: Python,
        xspecs: Option<Py<PyAny>>,
        regr_spec: u8,
        corr_spec: u8,
        kpls_dim: Option<usize>,
        n_clusters: isize,
        recombination: Recombination,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        n_start: isize,
        max_eval: usize,
        seed: Option<u64>,
    ) -> GpMix {
        GpMix::new(
            py,
            xspecs,
            regr_spec,
            corr_spec,
            kpls_dim,
            n_clusters,
            recombination,
            theta_init,
            theta_bounds,
            n_start,
            max_eval,
            seed,
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

    /// Load Gaussian processes mixture from file.
    ///
    /// # Parameters
    ///     filename (string)
    ///         json filepath generated by saving a trained Gaussian processes mixture
    ///
    #[staticmethod]
    fn load(filename: String) -> Gpx {
        let format = match Path::new(&filename).extension().unwrap().to_str().unwrap() {
            "json" => egobox_moe::GpFileFormat::Json,
            _ => egobox_moe::GpFileFormat::Binary,
        };
        Gpx(GpMixture::load(&filename, format).unwrap())
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
        self.0.predict_var(&x.as_array()).unwrap().into_pyarray(py)
    }

    /// Predict surrogate output derivatives at nsamples points.
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

    /// Get the input and output dimensions of the surrogate
    ///
    /// # Returns
    ///     the couple (nx, ny)
    ///
    fn dims(&self) -> (usize, usize) {
        self.0.dims()
    }

    /// Get the nt training data points used to fit the surrogate
    ///
    /// # Returns
    ///     the couple (ndarray[nt, nx], ndarray[nt,])
    ///
    fn training_data<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
        let (xdata, ydata) = self.0.training_data();
        (
            xdata.to_owned().into_pyarray(py),
            ydata.to_owned().into_pyarray(py),
        )
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

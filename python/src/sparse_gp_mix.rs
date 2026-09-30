//! `egobox`, Rust toolbox for efficient global optimization
//!
//! Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions,
//! the mixture of gaussian process surrogates is binded in Python. You can install the Python package using:
//!
//! ```bash
//! pip install egobox
//! ```
//!
//! See the [tutorial notebook](https://github.com/relf/egobox/notebooks/Sgp_Tutorial.ipynb) for usage.
//!
use crate::deprecation::resolve_renamed;
use crate::errors::{gp_file_format, input_x, moe_err, moe_file_err, training_data};
use crate::gp_config::{validate_corr_spec, validate_theta};
use crate::{logging::init_logger, types::*};
use egobox_moe::{
    Clustered, GpMetrics, GpMixture, GpSurrogate, GpType, Inducings, MixtureGpSurrogate,
    ThetaTuning,
};
use linfa::{Dataset, traits::Fit};
use ndarray::{Array1, Array2, Zip, array};
use ndarray_rand::rand::SeedableRng;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rand_xoshiro::Xoshiro256Plus;

/// Default number of GP hyperparameters optimization restarts of sparse GP
const SPARSE_GP_OPTIM_N_START: usize = 10;

/// Sparse Gaussian process builder
///
/// Inducing points are required: give either their number `nz` or their locations `z`.
///
/// Parameters
/// ----------
/// corr_spec : int
///     CorrelationSpec flags, an int in [1, 15]. Specification of correlation models.
///     Can be CorrelationSpec.SQUARED_EXPONENTIAL (1), CorrelationSpec.ABSOLUTE_EXPONENTIAL (2),
///     CorrelationSpec.MATERN32 (4), CorrelationSpec.MATERN52 (8) or
///     any bit-wise union of these values (e.g. CorrelationSpec.MATERN32 | CorrelationSpec.MATERN52)
/// theta_init : list of float, optional
///     Initial guess for GP theta hyperparameters, one value per input component.
///     When None the default is 1e-1 for all components
/// theta_bounds : list of [float, float], optional
///     Search space [[lower_1, upper_1], ..., [lower_nx, upper_nx]] when optimizing theta GP hyperparameters.
///     When None the default is [1e-2, 1e1] for all components.
/// kpls_dim : int, optional
///     Number of components to be used when PLS projection is used (a.k.a KPLS method), 0 < kpls_dim < nx.
///     This is used to address high-dimensional problems typically when nx > 9.
/// theta_n_start : int >= 0, optional
///     Number of internal GP hyperparameters optimization restarts (multistart, default is 10)
/// nz : int, optional
///     Number of inducing points, randomly picked among the training inputs.
///     Used when `z` is not given.
/// z : array[nz, nx], optional
///     Locations of the inducing points. Takes precedence over `nz`.
/// method : SparseMethod
///     Sparse method to be used (default is SparseMethod.FITC)
/// seed : int >= 0, optional
///     Random generator seed to allow computation reproducibility.
/// verbose : Verbose or int in [0, 4], optional
///     Optional verbose level to control logging output (default is 0)
///     Used mainly for debugging and development purposes
///
/// Deprecated
/// ----------
/// n_start : int >= 0, optional
///     Deprecated since 0.38.0, use `theta_n_start` instead.
///
/// Returns
/// -------
/// SparseGpMix
///     A builder which can be fitted to data to get a SparseGpx object (a trained sparse Gaussian process)
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct SparseGpMix {
    pub correlation_spec: CorrelationSpec,
    pub theta_init: Option<Vec<f64>>,
    pub theta_bounds: Option<Vec<Vec<f64>>>,
    pub kpls_dim: Option<usize>,
    pub theta_n_start: usize,
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
        theta_n_start = None,
        nz = None,
        z = None,
        method = SparseMethod::Fitc,
        seed = None,
        verbose = None,
        *,
        n_start = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        corr_spec: u8,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        kpls_dim: Option<usize>,
        theta_n_start: Option<usize>,
        nz: Option<usize>,
        z: Option<PyReadonlyArray2<f64>>,
        method: SparseMethod,
        seed: Option<u64>,
        #[gen_stub(override_type(type_repr = "Verbose | builtins.int | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        verbose: Option<Py<PyAny>>,
        n_start: Option<usize>,
    ) -> PyResult<Self> {
        let theta_n_start = resolve_renamed(
            py,
            "n_start",
            n_start,
            "theta_n_start",
            theta_n_start,
            SPARSE_GP_OPTIM_N_START,
        )?;
        init_logger(py, verbose);
        Ok(SparseGpMix {
            correlation_spec: CorrelationSpec(corr_spec),
            theta_init,
            theta_bounds,
            kpls_dim,
            theta_n_start,
            nz,
            z: z.map(|z| z.as_array().to_owned()),
            method,
            seed,
        })
    }

    /// Fit the parameters of the model using the training dataset to build a trained model
    ///
    /// Parameters
    /// ----------
    /// xt : array[nsamples, nx] or array[nsamples] when nx == 1
    ///     input samples
    /// yt : array[nsamples] or array[nsamples, 1]
    ///     output samples
    ///
    /// Returns
    /// -------
    /// SparseGpx
    ///     the fitted sparse Gaussian process
    ///
    fn fit(
        &mut self,
        py: Python,
        xt: PyReadonlyArrayDyn<f64>,
        yt: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<SparseGpx> {
        validate_corr_spec(self.correlation_spec.0)?;
        validate_theta(self.theta_init.as_ref(), self.theta_bounds.as_ref())?;
        let (xt, yt) = training_data(xt.as_array(), yt.as_array())?;

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
            return Err(PyValueError::new_err(
                "inducing points should be specified either with nz or z",
            ));
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
                .n_start(self.theta_n_start)
                .with_rng(rng)
                .fit(&dataset)
        });
        Ok(SparseGpx(Box::new(sgp.map_err(moe_err)?)))
    }
}

/// A trained sparse Gaussian process
///
/// Unlike `Gpx`, it has no `update` method: sparse GPs have to be refitted with the new data.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct SparseGpx(Box<GpMixture>);

#[gen_stub_pymethods]
#[pymethods]
impl SparseGpx {
    /// Get sparse Gaussian process builder aka `SparseGpMix`
    ///
    /// See `SparseGpMix` constructor for parameters description
    #[staticmethod]
    #[pyo3(signature = (
        corr_spec = CorrelationSpec::SQUARED_EXPONENTIAL,
        theta_init = None,
        theta_bounds = None,
        kpls_dim = None,
        theta_n_start = None,
        nz = None,
        z = None,
        method = SparseMethod::Fitc,
        seed = None,
        verbose = None,
        *,
        n_start = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn builder(
        py: Python,
        corr_spec: u8,
        theta_init: Option<Vec<f64>>,
        theta_bounds: Option<Vec<Vec<f64>>>,
        kpls_dim: Option<usize>,
        theta_n_start: Option<usize>,
        nz: Option<usize>,
        z: Option<PyReadonlyArray2<f64>>,
        method: SparseMethod,
        seed: Option<u64>,
        #[gen_stub(override_type(type_repr = "Verbose | builtins.int | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        verbose: Option<Py<PyAny>>,
        n_start: Option<usize>,
    ) -> PyResult<SparseGpMix> {
        SparseGpMix::new(
            py,
            corr_spec,
            theta_init,
            theta_bounds,
            kpls_dim,
            theta_n_start,
            nz,
            z,
            method,
            seed,
            verbose,
            n_start,
        )
    }

    /// Returns the String representation from serde json serializer
    fn __repr__(&self) -> PyResult<String> {
        serde_json::to_string(&self.0).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Returns a String informal representation
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    /// Save Gaussian processes mixture in a file.
    /// If the filename has .json JSON human readable format is used
    /// otherwise an optimized binary format is used.
    ///
    /// Parameters
    /// ----------
    /// filename : str
    ///     file path with .json or .bin extension
    ///
    /// Returns
    /// -------
    /// bool
    ///     True when save succeeds
    ///
    /// Raises
    /// ------
    /// OSError or ValueError
    ///     when the model can not be saved
    ///
    fn save(&self, filename: String) -> PyResult<bool> {
        self.0
            .save(&filename, gp_file_format(&filename))
            .map_err(|e| moe_file_err(e, &filename))?;
        Ok(true)
    }

    /// Load Gaussian processes mixture from file.
    ///
    /// Parameters
    /// ----------
    /// filename : str
    ///     .json or .bin file path generated by saving a trained model
    ///
    /// Returns
    /// -------
    /// SparseGpx
    ///     the loaded model
    ///
    /// Raises
    /// ------
    /// OSError or ValueError
    ///     when the model can not be loaded
    ///
    #[staticmethod]
    fn load(filename: String) -> PyResult<SparseGpx> {
        let sgp = GpMixture::load(&filename, gp_file_format(&filename))
            .map_err(|e| moe_file_err(e, &filename))?;
        Ok(SparseGpx(sgp))
    }

    /// Predict output values at nsamples points.
    ///
    /// Parameters
    /// ----------
    /// x : array[nsamples, nx] or array[nsamples] when nx == 1
    ///     input values
    ///
    /// Returns
    /// -------
    /// array[nsamples]
    ///     the output values at the nsamples x points
    ///
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = input_x(x.as_array(), self.0.dims().0)?;
        Ok(self.0.predict(&x).map_err(moe_err)?.into_pyarray(py))
    }

    /// Predict variances at nsamples points.
    ///
    /// Parameters
    /// ----------
    /// x : array[nsamples, nx] or array[nsamples] when nx == 1
    ///     input values
    ///
    /// Returns
    /// -------
    /// array[nsamples]
    ///     the variances of the output values at the nsamples x points
    ///
    fn predict_var<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = input_x(x.as_array(), self.0.dims().0)?;
        Ok(self.0.predict_var(&x).map_err(moe_err)?.into_pyarray(py))
    }

    /// Predict surrogate output derivatives at nsamples points.
    ///
    /// Implementation note: central finite difference technique
    /// on `predict()` function is used which may be subject to numerical issues
    ///
    /// Parameters
    /// ----------
    /// x : array[nsamples, nx] or array[nsamples] when nx == 1
    ///     input values
    ///
    /// Returns
    /// -------
    /// array[nsamples, nx]
    ///     the output derivatives wrt inputs at the nsamples x points.
    ///     The ith column is the partial derivative wrt the ith component of x.
    ///
    fn predict_gradients<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = input_x(x.as_array(), self.0.dims().0)?;
        Ok(self
            .0
            .predict_gradients(&x)
            .map_err(moe_err)?
            .into_pyarray(py))
    }

    /// Predict variance derivatives at nsamples points.
    ///
    /// Implementation note: central finite difference technique
    /// on `predict_var()` function is used which may be subject to numerical issues
    ///
    /// Parameters
    /// ----------
    /// x : array[nsamples, nx] or array[nsamples] when nx == 1
    ///     input values
    ///
    /// Returns
    /// -------
    /// array[nsamples, nx]
    ///     the variance derivatives wrt inputs at the nsamples x points.
    ///     The ith column is the partial derivative wrt the ith component of x.
    ///
    fn predict_var_gradients<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = input_x(x.as_array(), self.0.dims().0)?;
        Ok(self
            .0
            .predict_var_gradients(&x)
            .map_err(moe_err)?
            .into_pyarray(py))
    }

    /// Sample gaussian process trajectories.
    ///
    /// Parameters
    /// ----------
    /// x : array[nsamples, nx] or array[nsamples] when nx == 1
    ///     locations of the sampled trajectories
    /// n_traj : int
    ///     number of trajectories to generate
    ///
    /// Returns
    /// -------
    /// array[nsamples, n_traj]
    ///     the trajectories
    ///
    fn sample<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        n_traj: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = input_x(x.as_array(), self.0.dims().0)?;
        Ok(self.0.sample(&x, n_traj).map_err(moe_err)?.into_pyarray(py))
    }

    /// Get the input and output dimensions of the surrogate
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    ///     the couple (nx, ny)
    ///
    fn dims(&self) -> (usize, usize) {
        self.0.dims()
    }

    /// Get the nt training data points used to fit the surrogate
    ///
    /// Returns
    /// -------
    /// tuple[array[nt, nx], array[nt]]
    ///     the couple (xt, yt)
    ///
    fn training_data<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
        let (xdata, ydata) = <GpMixture as GpMetrics<_, _, _>>::training_data(&self.0);
        (
            xdata.to_owned().into_pyarray(py),
            ydata.to_owned().into_pyarray(py),
        )
    }

    /// Get optimized thetas hyperparameters (ie once GP experts are fitted)
    ///
    /// Returns
    /// -------
    /// array[n_clusters, nx or kpls_dim]
    ///     thetas of each expert
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
    /// Returns
    /// -------
    /// array[n_clusters]
    ///     variance of each expert
    ///
    fn variances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let experts = self.0.experts();
        let mut variances = Array1::zeros(self.0.n_clusters());
        Zip::from(&mut variances)
            .and(experts)
            .for_each(|var, expert| *var = expert.variance());
        variances.into_pyarray(py)
    }

    /// Get reduced likelihood values obtained when fitting the GP experts
    ///
    /// May be used to compare various parameterizations
    ///
    /// Returns
    /// -------
    /// array[n_clusters]
    ///     likelihood of each expert
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

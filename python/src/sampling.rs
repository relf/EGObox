use crate::deprecation::warn_deprecated;
use crate::domain;
use egobox_doe::{LhsKind, SamplingMethod};
use egobox_moe::MixintContext;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass_enum, gen_stub_pyfunction};

/// Sampling specifies the method used to generate samples, see `sampling()`.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Optimized Latin Hypercube Sampling: sample locations are optimized using the
    /// Enhanced Stochastic Evolutionary algorithm (ESE), see Jin et al. (2005)
    /// "An efficient algorithm for constructing optimal design of computer experiments"
    Lhs = 1,
    /// Full factorial sampling: points of a regular grid
    FullFactorial = 2,
    /// Uniform random sampling
    Random = 3,
    /// Classic LHS: each sample is chosen randomly within its latin hypercube interval
    LhsClassic = 4,
    /// Centered LHS: each sample is the middle of its latin hypercube interval
    LhsCentered = 5,
    /// Maximin LHS: the minimal distance between samples is maximized
    LhsMaximin = 6,
    /// Centered maximin LHS: centered samples with the minimal distance between samples maximized
    LhsCenteredMaximin = 7,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Sampling {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Lhs),
            Ok(2) => Ok(Self::FullFactorial),
            Ok(3) => Ok(Self::Random),
            Ok(4) => Ok(Self::LhsClassic),
            Ok(5) => Ok(Self::LhsCentered),
            Ok(6) => Ok(Self::LhsMaximin),
            Ok(7) => Ok(Self::LhsCenteredMaximin),
            Ok(v) => Err(PyValueError::new_err(format!(
                "sampling method integer value must be in [1, 7], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "method must be a Sampling enum or an integer in [1, 7]",
            )),
        }
    }
}

/// Samples generation using given method
///
/// Parameters
/// ----------
/// xspecs : list of XSpec, list of [lower, upper] or array[nx, 2]
///     Specifications of the nx input variables
/// n_samples : int
///     number of samples
/// method : Sampling, optional
///     Sampling.LHS (default when None), FULL_FACTORIAL, RANDOM, LHS_CLASSIC, LHS_CENTERED,
///     LHS_MAXIMIN or LHS_CENTERED_MAXIMIN. Plain LHS is the optimized (ESE) LHS.
/// seed : int >= 0, optional
///     random seed
///
/// Returns
/// -------
/// array[n_samples, nx]
///     the samples
///
/// Deprecated
/// ----------
/// The former argument order `sampling(method, xspecs, n_samples, seed=None)` is deprecated
/// since 0.38.0, use `sampling(xspecs, n_samples, method=..., seed=None)` instead.
///
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (xspecs, n_samples, method=None, seed=None))]
pub fn sampling<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Sequence[XSpec] | typing.Sequence[typing.Sequence[builtins.float]] | numpy.typing.NDArray[numpy.float64]", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
    xspecs: Py<PyAny>,
    #[gen_stub(override_type(type_repr = "builtins.int", imports = ("builtins")))] n_samples: Py<
        PyAny,
    >,
    #[gen_stub(override_type(type_repr = "Sampling | None", imports = ()))] method: Option<
        Py<PyAny>,
    >,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (method, xspecs, n_samples) = if let Ok(old_method) = xspecs.extract::<Sampling>(py) {
        // Former positional order sampling(M, X, N) detected by a Sampling first argument,
        // received as (xspecs=M, n_samples=X, method=N)
        warn_deprecated(
            py,
            "sampling(method, xspecs, n_samples)",
            "sampling(xspecs, n_samples, method=...)",
        )?;
        let old_n_samples = method.ok_or_else(|| {
            PyTypeError::new_err("sampling() missing required argument: 'n_samples'")
        })?;
        (old_method, n_samples, old_n_samples.extract::<usize>(py)?)
    } else {
        let method = match method {
            Some(method) => method.extract::<Sampling>(py)?,
            None => Sampling::Lhs,
        };
        (method, xspecs, n_samples.extract::<usize>(py)?)
    };
    sample(py, method, xspecs, n_samples, seed)
}

/// Generate `n_samples` samples of the `xspecs` domain with the given `method`
fn sample(
    py: Python<'_>,
    method: Sampling,
    xspecs: Py<PyAny>,
    n_samples: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let xtypes: Vec<egobox_moe::XType> = domain::parse(py, xspecs)?;
    let mixin = MixintContext::new(&xtypes);
    let doe = match method {
        Sampling::Lhs => Box::new(mixin.create_lhs_sampling(LhsKind::default(), seed))
            as Box<dyn SamplingMethod<_>>,
        Sampling::LhsClassic => Box::new(mixin.create_lhs_sampling(LhsKind::Classic, seed))
            as Box<dyn SamplingMethod<_>>,
        Sampling::LhsMaximin => Box::new(mixin.create_lhs_sampling(LhsKind::Maximin, seed))
            as Box<dyn SamplingMethod<_>>,
        Sampling::LhsCentered => Box::new(mixin.create_lhs_sampling(LhsKind::Centered, seed))
            as Box<dyn SamplingMethod<_>>,
        Sampling::LhsCenteredMaximin => {
            Box::new(mixin.create_lhs_sampling(LhsKind::CenteredMaximin, seed))
                as Box<dyn SamplingMethod<_>>
        }
        Sampling::FullFactorial => Box::new(mixin.create_ffact_sampling()),
        Sampling::Random => {
            Box::new(mixin.create_rand_sampling(seed)) as Box<dyn SamplingMethod<_>>
        }
    }
    .sample(n_samples);
    Ok(doe.into_pyarray(py))
}

/// Samples generation using optimized Latin Hypercube Sampling,
/// same as `sampling(xspecs, n_samples, method=Sampling.LHS, seed=seed)`
///
/// Parameters
/// ----------
/// xspecs : list of XSpec, list of [lower, upper] or array[nx, 2]
///     Specifications of the nx input variables
/// n_samples : int
///     number of samples
/// seed : int >= 0, optional
///     random seed
///
/// Returns
/// -------
/// array[n_samples, nx]
///     the samples
///
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (xspecs, n_samples, seed=None))]
pub(crate) fn lhs(
    py: Python,
    #[gen_stub(override_type(type_repr = "typing.Sequence[XSpec] | typing.Sequence[typing.Sequence[builtins.float]] | numpy.typing.NDArray[numpy.float64]", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
    xspecs: Py<PyAny>,
    n_samples: usize,
    seed: Option<u64>,
) -> PyResult<Bound<PyArray2<f64>>> {
    sample(py, Sampling::Lhs, xspecs, n_samples, seed)
}

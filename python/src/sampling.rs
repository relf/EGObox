use crate::domain;
use egobox_doe::{LhsKind, SamplingMethod};
use egobox_moe::MixintContext;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass_enum, gen_stub_pyfunction};

#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    Lhs = 1,
    FullFactorial = 2,
    Random = 3,
    LhsClassic = 4,
    LhsCentered = 5,
    LhsMaximin = 6,
    LhsCenteredMaximin = 7,
}

/// Samples generation using given method
///
/// # Parameters
///     method: LHS, FULL_FACTORIAL, RANDOM,
///             LHS_CLASSIC, LHS_CENTERED,
///             LHS_MAXIMIN, LHS_CENTERED_MAXIMIN
///     xspecs: list of XSpec
///     n_samples: number of samples
///     seed: random seed
///
/// # Returns
///    ndarray of shape (n_samples, n_variables)
///
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (method, xspecs, n_samples, seed=None))]
pub fn sampling(
    py: Python<'_>,
    method: Sampling,
    xspecs: Py<PyAny>,
    n_samples: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let xtypes: Vec<egobox_moe::XType> = domain::parse(py, xspecs);
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
    doe.into_pyarray(py)
}

/// Samples generation using optimized Latin Hypercube Sampling
///
/// # Parameters
///     xspecs: list of XSpec
///     n_samples: number of samples
///     seed: random seed
///
/// # Returns
///    ndarray of shape (n_samples, n_variables)
///
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (xspecs, n_samples, seed=None))]
pub(crate) fn lhs(
    py: Python,
    xspecs: Py<PyAny>,
    n_samples: usize,
    seed: Option<u64>,
) -> Bound<PyArray2<f64>> {
    sampling(py, Sampling::Lhs, xspecs, n_samples, seed)
}

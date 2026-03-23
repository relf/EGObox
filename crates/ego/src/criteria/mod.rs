//! Available infill criteria to be used by Egor solver
mod ei;
mod wb2;

pub use ei::{EI, ExpectedImprovement, LOG_EI, LogExpectedImprovement};
pub use wb2::{WB2, WB2Criterion, WB2S};

use dyn_clonable::*;
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView2};

/// A trait for infill criterion which maximmum location will
/// determine the next most promising point expected to be the
/// optimum location of the objective function
#[clonable]
#[typetag::serde(tag = "type_infill")]
pub trait InfillCriterion: Clone + Sync {
    /// Name of the infill criterion
    fn name(&self) -> &'static str;

    /// Criterion value at given point x with regards to given
    /// surrogate of the objective function, the current found min,
    /// and an optional scaling factor
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        scale: Option<f64>,
    ) -> f64;

    /// Derivatives wrt x components of the criterion value
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        scale: Option<f64>,
    ) -> Array1<f64>;

    /// Scaling factor computation
    fn scaling(
        &self,
        _x: &ArrayView2<f64>,
        _obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
    ) -> f64 {
        1.0
    }
}

impl std::fmt::Debug for dyn InfillCriterion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

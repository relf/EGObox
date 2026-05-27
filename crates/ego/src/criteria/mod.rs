//! Available infill criteria to be used by Egor solver
mod ei;
mod wb2;

pub use ei::{EI, ExpectedImprovement, LOG_EI, LogExpectedImprovement};
pub use wb2::{WB2, WB2Criterion, WB2S};

use dyn_clonable::*;
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView2};

/// How an infill criterion combines with feasibility terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfillComposition {
    /// Criterion values combine multiplicatively with feasibility terms.
    Linear,
    /// Criterion values combine additively in log-space with feasibility terms.
    Log,
}

/// A trait for infill criterion which maximmum location will
/// determine the next most promising point expected to be the
/// optimum location of the objective function
#[clonable]
#[typetag::serde(tag = "type_infill")]
pub trait InfillCriterion: Clone + Sync {
    /// Name of the infill criterion
    fn name(&self) -> &'static str;

    /// How the criterion combines with feasibility terms.
    fn composition(&self) -> InfillComposition {
        InfillComposition::Linear
    }

    /// Whether the criterion uses a dynamic scaling factor computed from samples.
    fn uses_dynamic_scaling(&self) -> bool {
        false
    }

    /// Criterion value at given point x with regards to given
    /// surrogate of the objective function, the current found min,
    /// an optional scaling factor and an optional weight for the
    /// standard deviation
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        sigma_weight: Option<f64>,
        scale: Option<f64>,
    ) -> f64;

    /// Derivatives wrt x components of the criterion value
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        fmin: f64,
        sigma_weight: Option<f64>,
        scale: Option<f64>,
    ) -> Array1<f64>;

    /// Scaling factor computation
    fn scaling(
        &self,
        _x: &ArrayView2<f64>,
        _obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
        _sigma_weight: Option<f64>,
    ) -> f64 {
        1.0
    }
}

impl std::fmt::Debug for dyn InfillCriterion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criteria::{EI, LOG_EI, WB2, WB2S};

    #[test]
    fn test_default_composition_is_linear() {
        assert_eq!(EI.composition(), InfillComposition::Linear);
        assert_eq!(WB2.composition(), InfillComposition::Linear);
        assert_eq!(WB2S.composition(), InfillComposition::Linear);
    }

    #[test]
    fn test_logei_composition_is_log() {
        assert_eq!(LOG_EI.composition(), InfillComposition::Log);
    }
}

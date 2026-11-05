//! Factory of consistent mixed-integer samplings and surrogates

use crate::errors::Result;
use crate::xtypes::XType;
use egobox_doe::{FullFactorial, Lhs, LhsKind, Random};
use linfa::traits::Fit;
use linfa::{DatasetBase, Float};
use ndarray::{Array1, Array2};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::marker::PhantomData;

use super::sampling::{LevelMapping, MixintSampling};
use super::space::{as_continuous_limits, compute_continuous_dim};
use super::surrogate::MixintGpMixture;
use super::{MixintGpMixtureParams, MoeBuilder};

/// A factory to build consistent sampling method and surrogate regarding
/// XType specifications
pub struct MixintContext {
    /// The input specifications (None means all continuous variables)
    xtypes: Option<Vec<XType>>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    /// For sampling data refers to DOE data. For surrogate data refers to training input data
    work_in_folded_space: bool,
}

impl MixintContext {
    /// Constructor with given `xtypes` specification
    /// where working in folded space is the default
    pub fn new(xtypes: &[XType]) -> Self {
        MixintContext {
            xtypes: Some(xtypes.to_vec()),
            work_in_folded_space: true,
        }
    }

    /// Creates a new context for continuous-only variables (no mixed-integer handling)
    pub fn new_continuous() -> Self {
        MixintContext {
            xtypes: None,
            work_in_folded_space: false,
        }
    }

    /// Compute input dim once unfolded due to continupous relaxation
    pub fn get_unfolded_dim(&self) -> usize {
        match &self.xtypes {
            Some(xtypes) => compute_continuous_dim(xtypes),
            None => {
                // All continuous, dim equals training data dimension
                0 // Will be determined from actual data when needed
            }
        }
    }

    /// Create a mixed integer LHS
    pub fn create_lhs_sampling<F: Float>(
        &self,
        kind: LhsKind,
        seed: Option<u64>,
    ) -> MixintSampling<F, Lhs<F, Xoshiro256Plus>> {
        let xtypes = self.xtypes.as_ref().expect("xtypes required for sampling");
        let lhs = seed
            .map_or(Lhs::new(&as_continuous_limits(xtypes)), |seed| {
                let rng = Xoshiro256Plus::seed_from_u64(seed);
                Lhs::new(&as_continuous_limits(xtypes)).with_rng(rng)
            })
            .kind(kind);
        MixintSampling {
            method: lhs,
            xtypes: Some(xtypes.clone()),
            output_in_folded_space: self.work_in_folded_space,
            level_mapping: LevelMapping::EqualSlices,
            phantom: PhantomData,
        }
    }

    /// Create a mixed integer full factorial
    pub fn create_ffact_sampling<F: Float>(&self) -> MixintSampling<F, FullFactorial<F>> {
        let xtypes = self.xtypes.as_ref().expect("xtypes required for sampling");
        MixintSampling {
            method: FullFactorial::new(&as_continuous_limits(xtypes)),
            xtypes: Some(xtypes.clone()),
            output_in_folded_space: self.work_in_folded_space,
            level_mapping: LevelMapping::Nearest,
            phantom: PhantomData,
        }
    }

    /// Create a mixed integer random sampling
    pub fn create_rand_sampling<F: Float>(
        &self,
        seed: Option<u64>,
    ) -> MixintSampling<F, Random<F, Xoshiro256Plus>> {
        let xtypes = self.xtypes.as_ref().expect("xtypes required for sampling");
        let rand = seed.map_or(Random::new(&as_continuous_limits(xtypes)), |seed| {
            let rng = Xoshiro256Plus::seed_from_u64(seed);
            Random::new(&as_continuous_limits(xtypes)).with_rng(rng)
        });
        MixintSampling {
            method: rand,
            xtypes: Some(xtypes.clone()),
            output_in_folded_space: self.work_in_folded_space,
            level_mapping: LevelMapping::EqualSlices,
            phantom: PhantomData,
        }
    }

    /// Create a mixed integer mixture of experts surrogate
    pub fn create_surrogate(
        &self,
        surrogate_builder: &MoeBuilder,
        dataset: &DatasetBase<Array2<f64>, Array1<f64>>,
    ) -> Result<MixintGpMixture> {
        let params = match &self.xtypes {
            Some(xtypes) => {
                let mut params =
                    MixintGpMixtureParams::new(Some(xtypes.as_slice()), surrogate_builder);
                params.work_in_folded_space(self.work_in_folded_space);
                params
            }
            None => {
                let mut params = MixintGpMixtureParams::new_continuous(surrogate_builder);
                params.work_in_folded_space(self.work_in_folded_space);
                params
            }
        };
        params.fit(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::xtypes::XType;

    use egobox_doe::SamplingMethod;

    #[test]
    fn test_mixint_context_continuous() {
        // Test MixintContext with continuous-only variables
        let xtypes = vec![XType::Float(0.0, 1.0), XType::Float(0.0, 1.0)];
        let mixi = MixintContext::new(&xtypes);

        // Verify xtypes is Some
        assert!(mixi.xtypes.is_some());

        // Create continuous sampling
        let sampler = mixi.create_rand_sampling::<f64>(Some(42));
        let samples = sampler.sample(5);

        // Should have 2 columns (same as the xtypes)
        assert_eq!(samples.ncols(), 2);
        assert_eq!(samples.nrows(), 5);

        // All values should be in [0, 1]
        for i in 0..samples.nrows() {
            for j in 0..samples.ncols() {
                let val = samples[[i, j]];
                assert!((0.0..=1.0).contains(&val), "Value {} out of range", val);
            }
        }
    }
}

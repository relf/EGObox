//! Mixed-integer Gaussian process mixture models

pub mod mixint;
pub mod types;

pub use types::{XType, discrete};
pub use mixint::{
    MixintGpMixture, MixintGpMixtureParams, MixintGpMixtureValidParams,
    MixintSampling, as_continuous_limits, to_continuous_space, to_discrete_space,
};

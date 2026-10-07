//! Multi-objective optimization building blocks: Pareto dominance, hypervolume and
//! scalarization utilities used by the [`crate::Egor`] multi-objective strategies.
//!
//! All objectives are minimized.

mod config;
pub(crate) mod eim;
// Used by hypervolume-based strategies and termination (not available yet)
#[allow(dead_code)]
pub(crate) mod hypervolume;
pub(crate) mod parego;
pub(crate) mod pareto;
mod result;
pub(crate) mod scalarization;

pub use config::{EimAggregation, MooConfig, MooStrategy};
pub use result::ParetoResult;

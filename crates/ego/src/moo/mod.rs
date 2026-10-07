//! Multi-objective optimization building blocks: Pareto dominance, hypervolume and
//! scalarization utilities used by the [`crate::Egor`] multi-objective strategies.
//!
//! All objectives are minimized.

// Used by multi-objective solver strategies (not wired in the solver yet)
#![allow(dead_code)]

pub(crate) mod hypervolume;
pub(crate) mod pareto;
pub(crate) mod scalarization;

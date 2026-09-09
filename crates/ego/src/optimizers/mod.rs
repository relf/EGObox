//! # Optimizers Module
//!
//! This module provides optimization backends for the infill criterion optimization.
//!
//! ## Architecture
//!
//! The module follows SOLID principles:
//!
//! - **Single Responsibility**: Each backend handles one optimization algorithm
//! - **Open/Closed**: New optimizers can be added by implementing [`InfillOptimizerTrait`]
//! - **Dependency Inversion**: Code depends on the trait abstraction
//!
//! ## Available Optimizers
//!
//! - **SLSQP** - Sequential Least Squares Programming (gradient-based, faster for differentiable criteria)
//! - **COBYLA** - Constrained Optimization BY Linear Approximations (derivative-free, more robust)
//! - **IPOPT** - Interior Point OPTimizer (gradient-based), via the pure-Rust `pounce` crate
//!   (<https://github.com/jkitchin/pounce>). Requires the `pounce` feature.

mod optimizer;

pub(crate) use optimizer::*;

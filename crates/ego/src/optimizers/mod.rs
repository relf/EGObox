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
//!
//! ## Feature Flags
//!
//! - `nlopt`: Use NLopt library (default). Falls back to pure-Rust implementations otherwise.

mod optimizer;

pub(crate) use optimizer::*;

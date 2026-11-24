//! # Optimizers Module
//!
//! This module provides optimization backends for the infill criterion optimization.
//!
//! ## Architecture
//!
//! The `Optimizer` facade runs the algorithm selected by the `Algorithm` enum
//! (mapped from the `InfillOptimizer` configuration) with the [Basin](https://basin.rs)
//! pure Rust implementations.
//!
//! ## Available Optimizers
//!
//! - **SLSQP** - Sequential Least Squares Programming (gradient-based, faster for differentiable criteria)
//! - **COBYLA** - Constrained Optimization BY Linear Approximations (derivative-free, more robust)
//! - **L-BFGS-B** - Limited-memory BFGS with bounds (gradient-based, bound constraints only,
//!   used with the constrained infill criterion)

mod optimizer;

pub(crate) use optimizer::*;

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
//! Basin pure Rust COBYLA and SLSQP implementations are used by default.
//!
//! - `c-cobyla`: Use the `cobyla` crate (C-ported NLopt COBYLA) instead of Basin COBYLA
//!   (also for GP hyperparameters training).
//! - `c-slsqp`: Use the `slsqp` crate (C-ported NLopt SLSQP) instead of Basin SLSQP.

mod optimizer;

pub(crate) use optimizer::*;

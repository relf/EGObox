//! # Solver Module - EGO Optimizer Implementation
//!
//! This module contains the core implementation of the Efficient Global Optimization (EGO)
//! algorithm using the argmin optimization framework.
//!
//! ## Architecture Overview
//!
//! The solver is organized following SOLID principles with clear separation of concerns:
//!
//! ### Public API
//!
//! - [`EgorSolver`] - Main optimizer implementing `argmin::Solver` trait
//! - [`EgorState`] - Optimizer state implementing `argmin::State` trait  
//! - [`EgorConfig`] / [`ValidEgorConfig`] - Configuration builders and validated config
//! - [`EgorService`] - Ask-and-tell interface for external control
//!
//! ### State Organization ([`egor_state`])
//!
//! The optimizer state is decomposed into logical sub-states for maintainability:
//!
//! - [`DoeState`] - Design of Experiments tracking (points added, DOE size, retries)
//! - [`SurrogateState`] - GP surrogate model state (clusterings, hyperparameters, training data)
//! - [`TregoState`] - Trust Region EGO algorithm state (sigma, iteration counters)
//! - [`CoegoState`] - Cooperative EGO state (component activity for high-dim problems)
//!
//! ### Algorithm Variants
//!
//! - **Standard EGO** - Default algorithm using global infill optimization
//! - **TREGO** ([`trego`]) - Trust Region EGO for improved local convergence
//! - **CoEGO** ([`coego`]) - Cooperative EGO for high-dimensional problems (dim > 100)
//!
//! ### Internal Implementation
//!
//! - [`solver_impl`] - Core EGO iteration logic, surrogate training, point addition
//! - [`solver_computations`] - Infill criterion computation, scaling, multistart strategies
//! - [`solver_infill_optim`] - Infill criterion optimization (SLSQP, Cobyla backends)
//!
//! ## Configuration
//!
//! Runtime behavior can be controlled via [`RuntimeFlags`] instead of environment variables:
//!
//! ```ignore
//! EgorBuilder::optimize(f)
//!     .configure(|config| config
//!         .max_iters(100)
//!         .configure_runtime_flags(|flags| flags
//!             .enable_logging(true)
//!             .use_gp_var_portfolio(true)))
//!     .min_within(&xlimits)
//!     .run()
//! ```
//!
//! ## Usage with argmin
//!
//! The solver integrates with argmin's executor for checkpointing and observation:
//!
//! ```ignore
//! use argmin::core::{Executor, observers::ObserverMode};
//!
//! let solver = EgorSolver::new(config);
//! let result = Executor::new(problem, solver)
//!     .configure(|state| state.max_iters(50))
//!     .add_observer(MyObserver, ObserverMode::Always)
//!     .run()?;
//! ```

mod coego;
mod egor_config;
mod egor_service;
mod egor_solver;
mod egor_state;
mod solver_computations;
mod solver_impl;
mod solver_infill_optim;
mod trego;

pub use egor_config::*;
pub use egor_service::*;
pub use egor_solver::*;
pub use egor_state::*;

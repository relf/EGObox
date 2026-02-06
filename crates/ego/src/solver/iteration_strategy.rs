//! # Iteration Strategy - Strategy Pattern for EGO Algorithm Variants
//!
//! This module defines the [`IterationStrategy`] trait that abstracts the
//! iteration control flow of the EGO optimizer. Different strategy implementations
//! control how iterations alternate between global and local search:
//!
//! - [`StandardEgoStrategy`] - Default strategy: always performs global search
//! - [`TregoStrategy`] - Trust Region EGO: alternates global/local phases with
//!   adaptive trust region sizing (Diouane et al., 2023)
//!
//! ## Design
//!
//! Strategies are **stateless configuration objects** stored as `Box<dyn IterationStrategy>`
//! in [`ValidEgorConfig`](super::ValidEgorConfig). Mutable per-iteration state lives in
//! [`TregoState`](super::TregoState) within [`EgorState`](super::EgorState).
//!
//! The strategy's [`prepare`](IterationStrategy::prepare) method reads/updates state and
//! returns an [`IterationMode`] that tells the solver whether to perform a global or
//! local search step.
//!
//! ## Example
//!
//! ```ignore
//! use egobox_ego::{EgorConfig, TregoStrategy};
//!
//! let config = EgorConfig::default()
//!     .iteration_strategy(Box::new(
//!         TregoStrategy::default()
//!             .n_gl_steps((2, 4))
//!             .beta(0.8)
//!     ))
//!     .xtypes(&xtypes)
//!     .check()?;
//! ```
use dyn_clonable::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use argmin::core::State;
use super::egor_state::EgorState;
use super::trego::{Phase, next_phase};

/// Describes the mode of the current iteration as determined by the
/// [`IterationStrategy`].
#[derive(Debug)]
pub enum IterationMode {
    /// Standard global EGO search over the full design space.
    Global,
    /// Local search within a trust region around the current best point.
    Local {
        /// Maximum distance from the current best point defining the trust
        /// region bounds (sigma * d_max from TREGO parameterization).
        max_dist: f64,
        /// Minimum distance for a candidate point to be accepted
        /// (sigma * d_min from TREGO parameterization).
        min_acceptance_distance: f64,
    },
}

/// A trait abstracting the iteration control flow of the EGO optimizer.
///
/// Implementations decide whether each iteration performs a global or local
/// search and manage any associated state transitions (e.g., trust region
/// size adjustments).
///
/// # Serialization
///
/// Uses `typetag` for polymorphic serialization of `Box<dyn IterationStrategy>`,
/// following the same pattern as [`InfillCriterion`](crate::criteria::InfillCriterion).
#[clonable]
#[typetag::serde(tag = "type_iteration_strategy")]
pub trait IterationStrategy: Clone + Sync + Debug {
    /// Returns the name of this strategy.
    fn name(&self) -> &str;

    /// Initialize strategy-specific state fields at the start of optimization.
    ///
    /// Called once during `Solver::init()`. Implementations should set any
    /// initial values in `state` (e.g., TREGO sigma).
    ///
    /// # Arguments
    /// * `state` - Mutable reference to the optimizer state
    /// * `xlimits` - Design space bounds (nrows = number of variables)
    fn init_state(&self, _state: &mut EgorState<f64>, _xlimits: &Array2<f64>) {
        // Default: no-op
    }

    /// Evaluate the previous iteration and determine the mode for the current one.
    ///
    /// This method is called at the start of each `next_iter()`. It may update
    /// mutable state (e.g., trust region sigma, phase counters) and returns an
    /// [`IterationMode`] telling the solver what kind of step to execute.
    ///
    /// # Arguments
    /// * `state` - Mutable reference to the optimizer state
    /// * `xlimits` - Design space bounds
    fn prepare(
        &self,
        state: &mut EgorState<f64>,
        xlimits: &Array2<f64>,
    ) -> IterationMode;

    /// Post-iteration hook called after the iteration completes.
    ///
    /// Default implementation is a no-op. Override to perform any
    /// end-of-iteration bookkeeping.
    fn finalize(&self, _state: &mut EgorState<f64>) {
        // Default: no-op
    }
}

// =============================================================================
// StandardEgoStrategy - always global search
// =============================================================================

/// The default EGO iteration strategy: every iteration uses global search
/// over the full design space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StandardEgoStrategy;

#[typetag::serde]
impl IterationStrategy for StandardEgoStrategy {
    fn name(&self) -> &str {
        "Standard EGO"
    }

    fn prepare(
        &self,
        _state: &mut EgorState<f64>,
        _xlimits: &Array2<f64>,
    ) -> IterationMode {
        IterationMode::Global
    }
}

// =============================================================================
// TregoStrategy - Trust Region EGO
// =============================================================================

/// Trust Region EGO (TREGO) iteration strategy.
///
/// Alternates between global EGO steps and local trust-region steps.
/// The trust region size adapts based on sufficient decrease criteria:
/// - Expands when progress is sufficient (sigma / beta)
/// - Contracts when local progress is insufficient (sigma * beta)
///
/// See: Diouane et al. (2023) - "TREGO: a Trust-Region framework for
/// Efficient Global Optimization"
///
/// # Parameters
///
/// - `n_gl_steps`: Number of global and local steps per phase `(n_global, n_local)`
/// - `d`: Trust region size bounds `(d_min, d_max)`
/// - `alpha`: Threshold ratio for sufficient decrease: `rho(sigma) = alpha * sigma^2`
/// - `beta`: Trust region contraction factor in `(0, 1)`
/// - `sigma0`: Initial trust region radius (overridden by computed value in `init_state`)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TregoStrategy {
    /// Number of global and local optimization steps.
    pub n_gl_steps: (usize, usize),
    /// Trust region size bounds (min, max).
    pub d: (f64, f64),
    /// Threshold ratio for iteration acceptance.
    pub alpha: f64,
    /// Trust region contraction factor in (0, 1).
    pub beta: f64,
    /// Initial trust region radius.
    pub sigma0: f64,
}

impl Default for TregoStrategy {
    fn default() -> Self {
        TregoStrategy {
            n_gl_steps: (1, 4),
            d: (1e-6, 1.),
            alpha: 1.0,
            beta: 0.9,
            sigma0: 1e-1,
        }
    }
}

impl TregoStrategy {
    /// Sets the number of global and local steps per phase.
    pub fn n_gl_steps(mut self, n_gl_steps: (usize, usize)) -> Self {
        self.n_gl_steps = n_gl_steps;
        self
    }

    /// Sets the trust region size bounds (min, max).
    pub fn d(mut self, d: (f64, f64)) -> Self {
        self.d = d;
        self
    }

    /// Sets the threshold ratio for sufficient decrease.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the trust region contraction factor.
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Sets the initial trust region radius.
    pub fn sigma0(mut self, sigma0: f64) -> Self {
        self.sigma0 = sigma0;
        self
    }
}

#[typetag::serde]
impl IterationStrategy for TregoStrategy {
    fn name(&self) -> &str {
        "TREGO"
    }

    fn init_state(&self, state: &mut EgorState<f64>, xlimits: &Array2<f64>) {
        // TREGO initial sigma = 0.5 * (0.2)^(1/nx)
        state.trego.sigma = 0.5 * (0.2f64).powf(1.0 / xlimits.nrows() as f64);
    }

    fn prepare(
        &self,
        state: &mut EgorState<f64>,
        _xlimits: &Array2<f64>,
    ) -> IterationMode {
        let rho = |sigma: f64| self.alpha * sigma * sigma;
        let (_, y_data, _) = state.surrogate.data.as_ref().unwrap();
        let best = state.surrogate.best_index.unwrap();
        let prev_best = state.surrogate.prev_best_index.unwrap();

        // Update cumulative decrease over the trego phase
        let decrease = y_data[[prev_best, 0]] - y_data[[best, 0]];
        state.trego.best_decrease += decrease.max(0.0);
        let sufficient_decrease = state.trego.best_decrease >= rho(state.trego.sigma);

        if state.trego.global_trego_iter == self.n_gl_steps.0
            || state.trego.local_trego_iter == self.n_gl_steps.1
        {
            log::info!(
                "Cumulative decrease: {}, required {}",
                state.trego.best_decrease,
                rho(state.trego.sigma)
            );
        }

        log::debug!(
            "TREGO update: iter={}, global_ego_iter = {}, local_trego_iter = {}",
            state.get_iter(),
            state.trego.global_trego_iter,
            state.trego.local_trego_iter
        );

        // Check step success and update trust region
        if state.get_iter() > 0 {
            if state.trego.global_trego_iter == self.n_gl_steps.0 {
                // End of global phase: adjust trust region
                if sufficient_decrease {
                    let old = state.trego.sigma;
                    state.trego.sigma *= 1. / self.beta;
                    log::info!(
                        "Previous EGO global step successful: sigma {} -> {}",
                        old, state.trego.sigma
                    );
                } else {
                    log::info!("Previous EGO global step progress fail");
                }
                state.trego.best_decrease = 0.0;
            } else if state.trego.local_trego_iter == self.n_gl_steps.1 {
                // End of local phase: adjust trust region
                if sufficient_decrease {
                    let old = state.trego.sigma;
                    state.trego.sigma *= 1. / self.beta;
                    log::info!(
                        "Previous TREGO local step successful: sigma {} -> {}",
                        old, state.trego.sigma
                    );
                } else {
                    let old = state.trego.sigma;
                    state.trego.sigma *= self.beta;
                    log::info!(
                        "Previous TREGO local step progress fail: sigma {} -> {}",
                        old, state.trego.sigma
                    );
                }
                state.trego.best_decrease = 0.0;
            }
        }

        // Determine phase and update counters
        let (phase, global_iter, local_iter) = next_phase(
            sufficient_decrease,
            state.trego.global_trego_iter,
            state.trego.local_trego_iter,
            self.n_gl_steps.0,
            self.n_gl_steps.1,
        );
        state.trego.global_trego_iter = global_iter;
        state.trego.local_trego_iter = local_iter;

        match phase {
            Phase::Global => {
                log::info!(
                    ">>> EGO global step {}/{}",
                    state.trego.global_trego_iter, self.n_gl_steps.0
                );
                IterationMode::Global
            }
            Phase::Local => {
                log::info!(
                    ">>> TREGO local step {}/{}",
                    state.trego.local_trego_iter, self.n_gl_steps.1
                );
                IterationMode::Local {
                    max_dist: self.d.1 * state.trego.sigma,
                    min_acceptance_distance: self.d.0 * state.trego.sigma,
                }
            }
        }
    }
}

impl std::fmt::Display for dyn IterationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

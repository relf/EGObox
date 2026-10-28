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
    fn prepare(&self, state: &mut EgorState<f64>, xlimits: &Array2<f64>) -> IterationMode;

    /// Post-iteration hook called after the iteration completes.
    ///
    /// Default implementation is a no-op. Override to perform any
    /// end-of-iteration bookkeeping.
    fn finalize(&self, _state: &mut EgorState<f64>) {
        // Default: no-op
    }

    /// Check the strategy parameters consistency, returning an error message otherwise.
    ///
    /// Called when the optimizer configuration is checked. Default implementation accepts any parameters.
    fn check(&self) -> std::result::Result<(), String> {
        Ok(())
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

    fn prepare(&self, _state: &mut EgorState<f64>, _xlimits: &Array2<f64>) -> IterationMode {
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
/// When GP models a noise on function values (see [`crate::GpConfig::noise_ratio`]),
/// the sufficient decrease test uses posterior means of the surrogate models
/// (estimates of the noise-free values) re-estimated at each iteration, and a failed
/// local phase is either (see Diouane et al. (2026) - "STREGO: A Scalable Implementation
/// of a Trust-Region Framework for Noisy Bayesian Optimization"):
/// - certain, when the best estimate of the phase points is worse than the phase start
///   best estimate by more than `rho(sigma)`: contraction by `beta`,
/// - uncertain otherwise: milder contraction by `beta_uncertain`,
/// - empty, when no point was added during the phase: contraction by `beta_empty`.
///
/// # Parameters
///
/// - `n_gl_steps`: Number of global and local steps per phase `(n_global, n_local)`
/// - `d`: Trust region size bounds `(d_min, d_max)`
/// - `alpha`: Threshold ratio for sufficient decrease: `rho(sigma) = alpha * sigma^2`
/// - `beta`: Trust region contraction factor in `(0, 1)`
/// - `sigma0`: Initial trust region radius (overridden by computed value in `init_state`)
/// - `beta_uncertain`: Contraction factor after an uncertain failure (noisy case only),
///   in `[beta, 1)`, default `sqrt(beta)`
/// - `beta_empty`: Contraction factor after an empty local phase (noisy case only),
///   in `[beta_uncertain, 1)`, default `sqrt(beta)`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// Trust region contraction factor after an uncertain failure (default `sqrt(beta)`).
    #[serde(default)]
    pub beta_uncertain: Option<f64>,
    /// Trust region contraction factor after an empty local phase (default `sqrt(beta)`).
    #[serde(default)]
    pub beta_empty: Option<f64>,
}

impl Default for TregoStrategy {
    fn default() -> Self {
        TregoStrategy {
            n_gl_steps: (1, 4),
            d: (1e-6, 1.),
            alpha: 1.0,
            beta: 0.9,
            sigma0: 1e-1,
            beta_uncertain: None,
            beta_empty: None,
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

    /// Sets the trust region contraction factor after an uncertain failure of a local phase,
    /// used only when GP models a noise on function values.
    /// Should be in `[beta, beta_empty]` (default `sqrt(beta)`).
    pub fn beta_uncertain(mut self, beta_uncertain: f64) -> Self {
        self.beta_uncertain = Some(beta_uncertain);
        self
    }

    /// Sets the trust region contraction factor after a local phase where no point was added,
    /// used only when GP models a noise on function values.
    /// Should be in `[beta_uncertain, 1)` (default `sqrt(beta)`).
    pub fn beta_empty(mut self, beta_empty: f64) -> Self {
        self.beta_empty = Some(beta_empty);
        self
    }

    /// Contraction factor after an uncertain failure
    fn uncertain_factor(&self) -> f64 {
        self.beta_uncertain.unwrap_or(self.beta.sqrt())
    }

    /// Contraction factor after an empty local phase
    fn empty_factor(&self) -> f64 {
        self.beta_empty.unwrap_or(self.beta.sqrt())
    }

    /// Multiply trust region radius by `factor` and log the reason
    fn scale_sigma(state: &mut EgorState<f64>, factor: f64, reason: &str) {
        let old = state.trego.sigma;
        state.trego.sigma *= factor;
        log::info!("{reason}: sigma {} -> {}", old, state.trego.sigma);
    }

    /// Sufficient decrease test and trust region update at the end of a phase
    /// when function values are estimated by surrogate models posterior means.
    /// Returns whether the decrease is sufficient.
    fn update_with_estimates(&self, state: &mut EgorState<f64>, y_est: &Array2<f64>) -> bool {
        let rho = self.alpha * state.trego.sigma * state.trego.sigma;
        let best = state.surrogate.best_index.unwrap();
        let start_best = state.trego.phase_start_best.unwrap_or(best);
        let start_rows = state.trego.phase_start_rows.min(y_est.nrows());

        // Phase start best re-estimated with current models
        let f_start = y_est[[start_best, 0]];
        // Best estimate among points added during the phase
        let f_trial = y_est
            .slice(ndarray::s![start_rows.., 0])
            .iter()
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let n_added = y_est.nrows() - start_rows;
        // Success only if the best point was found during the phase
        let sufficient_decrease = best >= start_rows && f_start - y_est[[best, 0]] >= rho;
        log::info!(
            "Estimated decrease: {} (best of {} added point(s): {}), required {}",
            f_start - y_est[[best, 0]],
            n_added,
            f_trial,
            rho
        );

        let local_phase_end = state.trego.local_trego_iter == self.n_gl_steps.1;
        if sufficient_decrease {
            let phase = if local_phase_end { "local" } else { "global" };
            Self::scale_sigma(
                state,
                1. / self.beta,
                &format!("Previous TREGO {phase} phase successful"),
            );
        } else if !local_phase_end {
            log::info!("Previous EGO global step progress fail");
        } else if n_added == 0 {
            Self::scale_sigma(
                state,
                self.empty_factor(),
                "Previous TREGO local phase empty",
            );
        } else if f_trial >= f_start + rho {
            Self::scale_sigma(state, self.beta, "Previous TREGO local phase certain fail");
        } else {
            Self::scale_sigma(
                state,
                self.uncertain_factor(),
                "Previous TREGO local phase uncertain fail",
            );
        }
        sufficient_decrease
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

    fn prepare(&self, state: &mut EgorState<f64>, _xlimits: &Array2<f64>) -> IterationMode {
        let y_estimate = state.surrogate.y_estimate.take();
        let sufficient_decrease = match y_estimate.as_ref() {
            Some(y_est) => {
                let phase_end = state.get_iter() > 0
                    && (state.trego.global_trego_iter == self.n_gl_steps.0
                        || state.trego.local_trego_iter == self.n_gl_steps.1);
                phase_end && self.update_with_estimates(state, y_est)
            }
            None => self.update_with_observations(state),
        };
        state.surrogate.y_estimate = y_estimate;

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

        // Record the phase start (used to test the decrease when values are estimated)
        if (phase == Phase::Global && global_iter == 1)
            || (phase == Phase::Local && local_iter == 1)
        {
            state.trego.phase_start_best = state.surrogate.best_index;
            state.trego.phase_start_rows = state
                .surrogate
                .data
                .as_ref()
                .map_or(0, |(x_data, _, _)| x_data.nrows());
        }

        match phase {
            Phase::Global => {
                log::info!(
                    ">>> EGO global step {}/{}",
                    state.trego.global_trego_iter,
                    self.n_gl_steps.0
                );
                IterationMode::Global
            }
            Phase::Local => {
                log::info!(
                    ">>> TREGO local step {}/{}",
                    state.trego.local_trego_iter,
                    self.n_gl_steps.1
                );
                IterationMode::Local {
                    max_dist: self.d.1 * state.trego.sigma,
                    min_acceptance_distance: self.d.0 * state.trego.sigma,
                }
            }
        }
    }

    fn check(&self) -> std::result::Result<(), String> {
        let (beta, beta_uncertain, beta_empty) =
            (self.beta, self.uncertain_factor(), self.empty_factor());
        if 0. < beta && beta <= beta_uncertain && beta_uncertain <= beta_empty && beta_empty < 1. {
            Ok(())
        } else {
            Err(format!(
                "TREGO contraction factors should verify 0 < beta <= beta_uncertain <= beta_empty < 1, \
                 got beta={beta}, beta_uncertain={beta_uncertain}, beta_empty={beta_empty}"
            ))
        }
    }
}

impl TregoStrategy {
    /// Sufficient decrease test and trust region update from observed values,
    /// the decrease being accumulated over the phase iterations.
    /// Returns whether the decrease is sufficient.
    fn update_with_observations(&self, state: &mut EgorState<f64>) -> bool {
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
                        old,
                        state.trego.sigma
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
                        old,
                        state.trego.sigma
                    );
                } else {
                    let old = state.trego.sigma;
                    state.trego.sigma *= self.beta;
                    log::info!(
                        "Previous TREGO local step progress fail: sigma {} -> {}",
                        old,
                        state.trego.sigma
                    );
                }
                state.trego.best_decrease = 0.0;
            }
        }
        sufficient_decrease
    }
}

impl std::fmt::Display for dyn IterationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const SIGMA: f64 = 0.5; // rho = alpha * sigma^2 = 0.25 with default alpha

    /// State at the end of a TREGO phase (`local` or global) started with 3 data points
    /// (best at index 0), `y` giving the values of all data points (estimated if `estimated`)
    fn phase_end_state(y: &[f64], best: usize, local: bool, estimated: bool) -> EgorState<f64> {
        let strategy = TregoStrategy::default();
        let n = y.len();
        let y = Array2::from_shape_vec((n, 1), y.to_vec()).unwrap();
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let mut state = EgorState::new().data((x, y.clone(), Array2::zeros((n, 0))));
        state.iter = 5;
        state.surrogate.prev_best_index = Some(0);
        state.surrogate.best_index = Some(best);
        state.surrogate.y_estimate = estimated.then_some(y);
        state.trego.sigma = SIGMA;
        state.trego.phase_start_best = Some(0);
        state.trego.phase_start_rows = 3;
        if local {
            state.trego.local_trego_iter = strategy.n_gl_steps.1;
        } else {
            state.trego.global_trego_iter = strategy.n_gl_steps.0;
        }
        state
    }

    #[test]
    fn test_estimated_local_success() {
        let strategy = TregoStrategy::default();
        let mut state = phase_end_state(&[1., 2., 3., 0.5, 0.9], 3, true, true);
        let mode = strategy.prepare(&mut state, &Array2::zeros((1, 2)));
        assert!(matches!(mode, IterationMode::Global));
        assert_abs_diff_eq!(state.trego.sigma, SIGMA / strategy.beta);
        // A new phase starts from the current best and data
        assert_eq!(state.trego.phase_start_best, Some(3));
        assert_eq!(state.trego.phase_start_rows, 5);
        assert!(state.surrogate.y_estimate.is_some());
    }

    #[test]
    fn test_estimated_local_failures() {
        let strategy = TregoStrategy::default()
            .beta_uncertain(0.95)
            .beta_empty(0.99);
        let cases = [
            // all added points worse than phase start best estimate by more than rho
            (vec![1., 2., 3., 1.3, 1.4], 0, strategy.beta),
            // added points not clearly worse
            (vec![1., 2., 3., 1.1, 1.2], 0, 0.95),
            // a slightly better added point is not a sufficient decrease
            (vec![1., 2., 3., 0.9, 1.2], 3, 0.95),
            // no point added
            (vec![1., 2., 3.], 0, 0.99),
        ];
        for (y, best, factor) in cases {
            let mut state = phase_end_state(&y, best, true, true);
            let mode = strategy.prepare(&mut state, &Array2::zeros((1, 2)));
            assert!(matches!(mode, IterationMode::Global), "{y:?}");
            assert_abs_diff_eq!(state.trego.sigma, factor * SIGMA);
        }
    }

    #[test]
    fn test_estimated_global_phase() {
        let strategy = TregoStrategy::default();
        // Sufficient decrease: trust region expanded, global phase goes on
        let mut state = phase_end_state(&[1., 2., 3., 0.5], 3, false, true);
        let mode = strategy.prepare(&mut state, &Array2::zeros((1, 2)));
        assert!(matches!(mode, IterationMode::Global));
        assert_abs_diff_eq!(state.trego.sigma, SIGMA / strategy.beta);

        // Failure: trust region unchanged, local phase starts
        let mut state = phase_end_state(&[1., 2., 3., 1.5], 0, false, true);
        let mode = strategy.prepare(&mut state, &Array2::zeros((1, 2)));
        assert!(matches!(mode, IterationMode::Local { .. }));
        assert_abs_diff_eq!(state.trego.sigma, SIGMA);
        assert_eq!(state.trego.phase_start_rows, 4);
    }

    #[test]
    fn test_observed_local_failure() {
        // Without estimates, any failed local phase contracts the trust region by beta
        let strategy = TregoStrategy::default()
            .beta_uncertain(0.95)
            .beta_empty(0.99);
        for y in [vec![1., 2., 3., 1.1], vec![1., 2., 3.]] {
            let mut state = phase_end_state(&y, 0, true, false);
            strategy.prepare(&mut state, &Array2::zeros((1, 2)));
            assert_abs_diff_eq!(state.trego.sigma, strategy.beta * SIGMA);
        }
    }

    #[test]
    fn test_trego_check() {
        assert!(TregoStrategy::default().check().is_ok());
        assert!(
            TregoStrategy::default()
                .beta_uncertain(0.95)
                .beta_empty(0.95)
                .check()
                .is_ok()
        );
        assert!(
            TregoStrategy::default()
                .beta_uncertain(0.8)
                .check()
                .is_err()
        );
        assert!(TregoStrategy::default().beta_empty(0.9).check().is_err());
        assert!(TregoStrategy::default().beta_empty(1.).check().is_err());
        assert!(TregoStrategy::default().beta(1.5).check().is_err());
    }

    #[test]
    fn test_trego_strategy_without_new_fields_deserialization() {
        let mut value = serde_json::to_value(TregoStrategy::default()).unwrap();
        let fields = value.as_object_mut().unwrap();
        fields.remove("beta_uncertain");
        fields.remove("beta_empty");
        let strategy: TregoStrategy = serde_json::from_value(value).unwrap();
        assert_eq!(strategy.beta_uncertain, None);
        assert_eq!(strategy.beta_empty, None);
    }
}

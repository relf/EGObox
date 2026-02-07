//! # Activity Strategy - Strategy Pattern for Variable Grouping
//!
//! This module defines the [`ActivityStrategy`] trait that abstracts how the
//! optimizer manages variable activity (which variables are optimized at each step):
//!
//! - [`FullActivity`] - Default: all variables are active in every iteration
//! - [`CooperativeActivity`] - CoEGO: randomly groups variables into subsets
//!   for cooperative optimization of high-dimensional problems (Zhan et al., 2024)
//!
//! ## Design
//!
//! Strategies are **stateless configuration objects** stored as `Box<dyn ActivityStrategy>`
//! in [`ValidEgorConfig`](super::ValidEgorConfig). Mutable per-iteration state (the current
//! activity matrix) lives in [`CoegoState`](super::CoegoState) within [`EgorState`](super::EgorState).
//!
//! ## Example
//!
//! ```ignore
//! use egobox_ego::{EgorConfig, CooperativeActivity};
//!
//! let config = EgorConfig::default()
//!     .activity_strategy(Box::new(CooperativeActivity::new(5)))
//!     .xtypes(&xtypes)
//!     .check()?;
//! ```
use dyn_clonable::*;
use egobox_gp::ThetaTuning;
use ndarray::{Array, Array1, Array2, s};
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use ndarray_rand::rand::seq::SliceRandom;

/// A trait abstracting the variable activity management of the EGO optimizer.
///
/// Implementations decide which variables are active during each optimization
/// step and handle related concerns like theta tuning adjustments.
///
/// # Serialization
///
/// Uses `typetag` for polymorphic serialization of `Box<dyn ActivityStrategy>`,
/// following the same pattern as [`InfillCriterion`](crate::criteria::InfillCriterion).
#[clonable]
#[typetag::serde(tag = "type_activity_strategy")]
pub trait ActivityStrategy: Clone + Sync + Debug {
    /// Returns the name of this strategy.
    fn name(&self) -> &str;

    /// Generate an activity matrix for the current iteration.
    ///
    /// Returns `None` when all variables are always active (full activity),
    /// or `Some(activity)` with a matrix of shape `(n_groups, group_size)`
    /// containing variable indices for cooperative optimization.
    ///
    /// # Arguments
    /// * `nx` - Total number of variables (dimension of design space)
    /// * `rng` - Random number generator for shuffling
    fn generate_activity(&self, nx: usize, rng: &mut Xoshiro256Plus) -> Option<Array2<usize>>;

    /// Adjust theta tuning parameters for partial optimization.
    ///
    /// Called during surrogate training when only a subset of variables
    /// are active. The default implementation is a no-op.
    ///
    /// # Arguments
    /// * `active` - Indices of currently active variables
    /// * `theta_tunings` - Mutable slice of theta tuning parameters to adjust
    fn adjust_theta_tuning(&self, _active: &[usize], _theta_tunings: &mut [ThetaTuning<f64>]) {
        // Default: no-op
    }

    /// Whether this strategy supports automated clustering.
    ///
    /// Returns `true` for full activity (standard EGO), `false` for cooperative
    /// activity (CoEGO) where automated clustering is not available.
    fn supports_auto_clustering(&self) -> bool {
        true
    }

    /// Whether this strategy uses cooperative (partial) optimization.
    ///
    /// Returns `true` for `CooperativeActivity`, `false` for `FullActivity`.
    /// Used for validation checks (e.g., CoEGO and KPLS cannot coexist).
    fn is_cooperative(&self) -> bool {
        false
    }
}

// =============================================================================
// FullActivity - all variables active (standard EGO)
// =============================================================================

/// Default activity strategy where all variables are active in every iteration.
///
/// This corresponds to standard EGO behavior with no variable decomposition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FullActivity;

#[typetag::serde]
impl ActivityStrategy for FullActivity {
    fn name(&self) -> &str {
        "Full Activity"
    }

    fn generate_activity(&self, _nx: usize, _rng: &mut Xoshiro256Plus) -> Option<Array2<usize>> {
        None
    }
}

// =============================================================================
// CooperativeActivity - CoEGO variable grouping
// =============================================================================

/// Cooperative EGO (CoEGO) activity strategy for high-dimensional problems.
///
/// Randomly decomposes the design space into `n_coop` groups of variables.
/// At each iteration, only one group is optimized while others are held fixed.
/// This reduces the effective dimensionality of the surrogate models.
///
/// Intended for problems with dimension > 100.
///
/// # Parameters
///
/// - `n_coop`: Number of cooperative groups to split variables into
///
/// See: Zhan et al. (2024)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CooperativeActivity {
    /// Number of cooperative groups.
    pub n_coop: usize,
}

impl CooperativeActivity {
    /// Creates a new `CooperativeActivity` with the given number of groups.
    pub fn new(n_coop: usize) -> Self {
        CooperativeActivity { n_coop }
    }

    /// Sets the number of cooperative groups.
    pub fn n_coop(mut self, n_coop: usize) -> Self {
        self.n_coop = n_coop;
        self
    }

    /// Remove out-of-range indices from the activity row.
    ///
    /// When `n_coop` does not evenly divide `nx`, the last row of the activity
    /// matrix may contain `nx` as a sentinel value. This function filters those out.
    fn strip(active: &[usize], dim: usize) -> Vec<usize> {
        active.iter().filter(|&&i| i < dim).cloned().collect()
    }
}

#[typetag::serde]
impl ActivityStrategy for CooperativeActivity {
    fn name(&self) -> &str {
        "Cooperative Activity (CoEGO)"
    }

    fn generate_activity(&self, nx: usize, rng: &mut Xoshiro256Plus) -> Option<Array2<usize>> {
        let g_nb = self.n_coop.min(nx);
        let remainder = nx % g_nb;
        let activity = if remainder == 0 {
            let g_size = nx / g_nb;
            let mut idx: Vec<usize> = (0..nx).collect();
            idx.shuffle(rng);
            Array2::from_shape_vec((g_nb, g_size), idx.to_vec()).unwrap()
        } else {
            let g_size = nx / g_nb + 1;
            let mut idx: Vec<usize> = (0..nx).collect();
            idx.shuffle(rng);
            let cut = g_nb * (g_size - 1);
            let fill = Array::from_shape_vec((g_nb, g_size - 1), idx[..cut].to_vec()).unwrap();
            let last_vals = Array1::from_vec(idx[cut..].to_vec());

            let mut indices = Array::from_elem((g_nb, g_size), nx);
            indices.slice_mut(s![.., ..(g_size - 1)]).assign(&fill);
            indices
                .slice_mut(s![..remainder, g_size - 1])
                .assign(&last_vals);
            indices
        };
        Some(activity)
    }

    fn adjust_theta_tuning(&self, active: &[usize], theta_tunings: &mut [ThetaTuning<f64>]) {
        theta_tunings.iter_mut().for_each(|theta| {
            *theta = match theta {
                ThetaTuning::Fixed(init) => ThetaTuning::Partial {
                    init: init.clone(),
                    bounds: Array1::from_vec(vec![ThetaTuning::<f64>::DEFAULT_BOUNDS; init.len()]),
                    active: Self::strip(active, init.len()),
                },
                ThetaTuning::Full { init, bounds } => ThetaTuning::Partial {
                    init: init.clone(),
                    bounds: bounds.clone(),
                    active: Self::strip(active, init.len()),
                },
                ThetaTuning::Partial {
                    init,
                    bounds,
                    active: _,
                } => ThetaTuning::Partial {
                    init: init.clone(),
                    bounds: bounds.clone(),
                    active: Self::strip(active, init.len()),
                },
            };
        });
    }

    fn supports_auto_clustering(&self) -> bool {
        false
    }

    fn is_cooperative(&self) -> bool {
        true
    }
}

impl std::fmt::Display for dyn ActivityStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

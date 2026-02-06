//! # EgorState - Optimizer State Implementation
//!
//! This module implements the [`EgorState`] struct which tracks all state information
//! during EGO optimization. It implements the `argmin::State` trait for integration
//! with the argmin optimization framework.
//!
//! ## State Organization
//!
//! The state is decomposed into logical sub-states following the Single Responsibility
//! Principle (SRP):
//!
//! - [`DoeState`] - Design of Experiments management (point tracking, retries)
//! - [`SurrogateState`] - GP surrogate models (clusterings, hyperparameters, data)
//! - [`TregoState`] - TREGO algorithm variant state (trust region parameters)
//! - [`CoegoState`] - CoEGO algorithm variant state (component activity)
//!
//! ## Serialization
//!
//! All state structs support serialization via serde. Sub-states use `#[serde(flatten)]`
//! for backward-compatible JSON format where all fields appear at the top level.

use crate::{
    InfillObjData,
    utils::{find_best_result_index, is_update_ok, run_recorder::EgorRunData},
};
use egobox_moe::Clustering;

use argmin::core::{ArgminFloat, Problem, State, TerminationReason, TerminationStatus};
use linfa::Float;
use ndarray::{Array1, Array2};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DEPRECATED: to be removed in future versions
/// (was 3 before but retry strategy is not useful anymore)
/// Value is kept for backward compatibility and set to 1.
/// Kept also to validate there is no point keeping this mechanism (!)
///
/// Max number of retry when adding a new point. Point addition may fail
/// if new point is too close to a previous point in the growing doe used
/// to train surrogate models modeling objective and constraints functions.
pub(crate) const MAX_POINT_ADDITION_RETRY: i32 = 1;

// =============================================================================
// Sub-state structs for better organization (SRP - Single Responsibility Principle)
// =============================================================================

/// State related to Design of Experiments (DOE) management.
///
/// Tracks the growing DOE as points are added during optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoeState<F: Float> {
    /// Initial doe size
    pub doe_size: usize,
    /// Number of added points
    pub added: usize,
    /// Previous number of added points
    pub prev_added: usize,
    /// Current number of retry without adding point
    pub no_point_added_retries: i32,
    /// Flag to trigger LHS optimization
    pub lhs_optim: bool,
    /// Constraint tolerance cstr < cstr_tol.
    /// It used to assess the validity of the param point and hence the corresponding cost
    pub cstr_tol: Array1<F>,
}

impl<F: Float> Default for DoeState<F> {
    fn default() -> Self {
        DoeState {
            doe_size: 0,
            added: 0,
            prev_added: 0,
            no_point_added_retries: MAX_POINT_ADDITION_RETRY,
            lhs_optim: false,
            cstr_tol: Array1::zeros(0),
        }
    }
}

/// State related to surrogate model management.
///
/// Tracks clusterings, hyperparameters, and training data for GP mixture surrogates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurrogateState<F: Float> {
    /// Current clusterings for objective and constraints GP mixture surrogate models
    pub clusterings: Option<Vec<Option<Clustering>>>,
    /// ThetaTunings controlled by q_optmod configuration triggering
    /// GP surrogate models hyperparameters optimization or reusing previous ones
    pub theta_inits: Option<Vec<Option<Array2<F>>>>,
    /// Historic data (params, objective and constraints values, function constraints)
    pub data: Option<(Array2<F>, Array2<F>, Array2<F>)>,
    /// Points resulting in failure during objective/constraints evaluation
    pub x_fail: Option<Array2<F>>,
    /// Previous index of best result in data
    pub prev_best_index: Option<usize>,
    /// index of best result in data
    pub best_index: Option<usize>,
    /// Infill data used to optimized infill criterion
    pub infill_data: InfillObjData<F>,
    /// Infill criterion value
    pub infill_value: F,
}

impl<F: Float> Default for SurrogateState<F> {
    fn default() -> Self {
        SurrogateState {
            clusterings: None,
            theta_inits: None,
            data: None,
            x_fail: None,
            prev_best_index: None,
            best_index: None,
            infill_data: Default::default(),
            infill_value: F::infinity(),
        }
    }
}

/// State specific to the TREGO algorithm variant.
///
/// TREGO (Trust Region EGO) alternates between global and local search phases.
/// See Diouane et al. (2023) for algorithm details.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TregoState<F: Float> {
    /// Trust region size parameter
    pub sigma: F,
    /// Counter for global TREGO iterations in current phase
    pub global_trego_iter: usize,
    /// Counter for local TREGO iterations in current phase
    pub local_trego_iter: usize,
    /// Cumulative decrease of best value over a trego global or local phase
    pub best_decrease: f64,
}

impl<F: Float> Default for TregoState<F> {
    fn default() -> Self {
        TregoState {
            sigma: F::cast(1e-1),
            global_trego_iter: 0,
            local_trego_iter: 0,
            best_decrease: 0.0,
        }
    }
}

/// State specific to the CoEGO algorithm variant.
///
/// CoEGO (Cooperative EGO) decomposes the problem into subproblems.
/// See Zhan et al. (2024) for algorithm details.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CoegoState {
    /// Activity matrix tracking which variables are active in each subproblem
    pub activity: Option<Array2<usize>>,
}

// =============================================================================
// Main EgorState struct - composed from sub-states
// =============================================================================

/// Maintains the state from iteration to iteration of the [crate::EgorSolver].
///
/// This struct is passed from one iteration of an algorithm to the next.
/// It is organized into logical sub-states for better maintainability:
/// - Core iteration state (param, cost, iter, etc.) - required by argmin `State` trait
/// - [`DoeState`]: Design of Experiments management
/// - [`SurrogateState`]: GP surrogate model state
/// - [`TregoState`]: TREGO algorithm variant state
/// - [`CoegoState`]: CoEGO algorithm variant state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EgorState<F: Float> {
    // -------------------------------------------------------------------------
    // Core iteration state (required by argmin State trait)
    // -------------------------------------------------------------------------
    /// Current parameter vector
    pub param: Option<Array1<F>>,
    /// Previous parameter vector
    pub prev_param: Option<Array1<F>>,
    /// Current best parameter vector
    pub best_param: Option<Array1<F>>,
    /// Previous best parameter vector
    pub prev_best_param: Option<Array1<F>>,
    /// At least one point is feasible
    pub feasibility: bool,

    /// Current cost function value
    /// The first component is the actual cost value
    /// while the remaining ones are the constraints values
    pub cost: Option<Array1<F>>,
    /// Previous cost function value
    pub prev_cost: Option<Array1<F>>,
    /// Current best cost function value
    pub best_cost: Option<Array1<F>>,
    /// Previous best cost function value
    pub prev_best_cost: Option<Array1<F>>,
    /// Target cost function value
    pub target_cost: F,

    /// Current iteration
    pub iter: u64,
    /// Iteration number of last best cost
    pub last_best_iter: u64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Evaluation counts
    pub counts: HashMap<String, u64>,
    /// Time required so far
    pub time: Option<web_time::Duration>,
    /// Optimization status
    pub termination_status: TerminationStatus,

    // -------------------------------------------------------------------------
    // DOE state - Design of Experiments management
    // -------------------------------------------------------------------------
    /// DOE-related state (size, added points, retries, constraints tolerance)
    pub doe: DoeState<F>,

    // -------------------------------------------------------------------------
    // Surrogate state - GP model management
    // -------------------------------------------------------------------------
    /// Surrogate model state (clusterings, theta, data, infill)
    pub surrogate: SurrogateState<F>,

    // -------------------------------------------------------------------------
    // Algorithm variant states
    // -------------------------------------------------------------------------
    /// TREGO algorithm state (trust region optimization)
    pub trego: TregoState<F>,

    /// CoEGO algorithm state (cooperative optimization)
    pub coego: CoegoState,

    // -------------------------------------------------------------------------
    // Persistence and RNG
    // -------------------------------------------------------------------------
    /// Run data for persistent logging
    #[cfg(feature = "persistent")]
    pub run_data: Option<EgorRunData>,

    /// Random number generator for reproducibility
    pub rng: Option<Xoshiro256Plus>,
}

impl<F> EgorState<F>
where
    Self: State<Float = F>,
    F: Float,
{
    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::State;
    /// # use egobox_ego::EgorState;
    /// # use ndarray::array;
    /// # let state: EgorState<f64> = EgorState::new();
    /// # let param_old = array![1.0f64, 2.0f64];
    /// # let state = state.param(param_old);
    /// # assert!(state.prev_param.is_none());
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let param = array![0.0f64, 3.0f64];
    /// let state = state.param(param);
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn param(mut self, param: Array1<F>) -> Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set target cost.
    ///
    /// When this cost is reached, the algorithm will stop. The default is
    /// `Self::Float::NEG_INFINITY`.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// let state = state.target_cost(0.0);
    /// # assert_eq!(state.target_cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Set maximum number of iterations
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.max_iters, u64::MAX);
    /// let state = state.max_iters(1000);
    /// # assert_eq!(state.max_iters, 1000);
    /// ```
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use argmin::core::State;
    /// # use egobox_ego::EgorState;
    /// # let state: EgorState<f64> = EgorState::new();
    /// # let cost_old = 1.0f64;
    /// # let state = state.cost(array![cost_old]);
    /// # assert!(state.prev_cost.is_none());
    /// # assert_eq!(state.cost.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # let cost = 0.0f64;
    /// let state = state.cost(array![cost]);
    /// # assert_eq!(state.prev_cost.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.cost.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn cost(mut self, cost: Array1<F>) -> Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = Some(cost);
        self
    }

    /// Set best index pointing to best cost found so far.
    /// This shifts the stored best index to the previous best index.
    #[must_use]
    pub fn best_index(mut self, best_index: usize) -> Self {
        self.surrogate.prev_best_index = self.surrogate.best_index;
        self.surrogate.best_index = Some(best_index);
        self
    }

    /// Adds the given nb of added point to the total added points.
    /// This shifts the total added points value to the previous total added points value.
    #[must_use]
    pub fn count_added_points(mut self, nb: usize) -> Self {
        self.doe.prev_added = self.doe.added;
        self.doe.added += nb;
        self
    }

    /// Set the current clusterings used by surrogate models
    #[must_use]
    pub fn clusterings(mut self, clustering: Vec<Option<Clustering>>) -> Self {
        self.surrogate.clusterings = Some(clustering);
        self
    }

    /// Moves the current clusterings out and replaces it internally with `None`.
    pub fn take_clusterings(&mut self) -> Option<Vec<Option<Clustering>>> {
        self.surrogate.clusterings.take()
    }

    /// Set the current theta init value used by surrogate models
    #[must_use]
    pub fn theta_inits(mut self, theta_inits: Vec<Option<Array2<F>>>) -> Self {
        self.surrogate.theta_inits = Some(theta_inits);
        self
    }

    /// Moves the current theta inits out and replaces it internally with `None`.
    pub fn take_theta_inits(&mut self) -> Option<Vec<Option<Array2<F>>>> {
        self.surrogate.theta_inits.take()
    }

    /// Set the current data points as training points for the surrogate models
    /// These points are gradually selected by the EGO algorithm regarding an infill criterion.
    /// Data is expressed as a triple (xdata, ydata, cdata) where :
    /// * xdata is a (p, nx matrix),
    /// * ydata is a (p, 1 + nb of cstr) matrix and ydata_i = fcost(xdata_i) for i in [1, p],  
    /// * cdata is a (p, nb of fcstr) matrix and cdata_i = fcstr_j(xdata_i) for i in [1, p], j in [1, nb of fcstr]
    #[must_use]
    pub fn data(mut self, data: (Array2<F>, Array2<F>, Array2<F>)) -> Self {
        self.surrogate.data = Some(data);
        self
    }

    /// Moves the current data out and replaces it internally with `None`.
    pub fn take_data(&mut self) -> Option<(Array2<F>, Array2<F>, Array2<F>)> {
        self.surrogate.data.take()
    }

    /// Moves the current failed points out and replaces it internally with `None`.
    pub fn take_x_fail(&mut self) -> Option<Array2<F>> {
        self.surrogate.x_fail.take()
    }

    /// Set the points resulting in failure during objective/constraints evaluation
    #[must_use]
    pub fn x_fail(mut self, x_fail: Array2<F>) -> Self {
        self.surrogate.x_fail = Some(x_fail);
        self
    }

    /// Store failed points by appending them to existing ones (if any)
    pub fn store_failed_points(self, x_fail_points: Option<Array2<F>>) -> Self {
        if let Some(fail_points) = x_fail_points {
            log::info!("Failed point(s): {}", fail_points);
            if let Some(x_fail) = self.surrogate.x_fail.as_ref() {
                // Append failed points if not too close to existing failed points
                let mut x_fail = x_fail.clone();
                fail_points.outer_iter().for_each(|x_new| {
                    if is_update_ok(&x_fail, &x_new) {
                        x_fail.push_row(x_new).unwrap();
                    } else {
                        log::info!("Failed point {} too close to existing failed points", x_new);
                    }
                });
                self.x_fail(x_fail)
            } else {
                self.x_fail(fail_points)
            }
        } else {
            self
        }
    }

    /// Set the activity matrix  
    #[must_use]
    pub fn activity(mut self, activity: Array2<usize>) -> Self {
        self.coego.activity = Some(activity);
        self
    }

    /// Moves the current activity out and replaces it internally with `None`.
    pub fn take_activity(&mut self) -> Option<Array2<usize>> {
        self.coego.activity.take()
    }

    /// Set the run data
    #[cfg(feature = "persistent")]
    #[must_use]
    pub fn run_data(mut self, run_data: crate::utils::run_recorder::EgorRunData) -> Self {
        self.run_data = Some(run_data);
        self
    }

    /// Moves the current rundata out and replaces it internally with `None`.
    #[cfg(feature = "persistent")]
    pub fn take_run_data(&mut self) -> Option<EgorRunData> {
        self.run_data.take()
    }

    /// Set the random number generator used to draw random points
    #[must_use]
    pub fn rng(mut self, rng: Xoshiro256Plus) -> Self {
        self.rng = Some(rng);
        self
    }

    /// Moves the current random number generator out and replaces it internally with `None`.
    pub fn take_rng(&mut self) -> Option<Xoshiro256Plus> {
        self.rng.take()
    }

    /// Set the infill criterion value    
    #[must_use]
    pub fn infill_value(mut self, value: F) -> Self {
        self.surrogate.infill_value = value;
        self
    }

    /// Returns the infill criterion value
    pub fn get_infill_value(&self) -> F {
        self.surrogate.infill_value
    }

    /// Returns current cost (ie objective) function and constraint values.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.cost = Some(array![12.0, 0.1]);
    /// let cost = state.get_full_cost();
    /// # assert_eq!(cost.unwrap()[0].to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// # assert_eq!(cost.unwrap()[1].to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    pub fn get_full_cost(&self) -> Option<&Array1<F>> {
        self.cost.as_ref()
    }

    /// Returns current cost (ie objective) function and constraint values.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.best_cost = Some(array![12.0, 0.1]);
    /// let cost = state.get_full_best_cost();
    /// # assert_eq!(cost.unwrap()[0].to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// # assert_eq!(cost.unwrap()[1].to_ne_bytes(), 0.1f64.to_ne_bytes());
    /// ```
    pub fn get_full_best_cost(&self) -> Option<&Array1<F>> {
        self.best_cost.as_ref()
    }
}

impl<F> EgorState<F>
where
    F: Float + ArgminFloat,
{
    /// Allow hot start feature by extending current max_iters
    pub fn extend_max_iters(&mut self, ext_iters: u64) {
        self.max_iters += ext_iters;
    }
}

impl<F> State for EgorState<F>
where
    F: Float + ArgminFloat,
{
    /// Type of parameter vector
    type Param = Array1<F>;
    /// Floating point precision
    type Float = F;

    /// Create new `EgorState` instance
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time;
    /// # use std::collections::HashMap;
    /// # use argmin::core::{State, TerminationStatus};
    /// use egobox_ego::EgorState;
    /// let state: EgorState<f64> = EgorState::new();
    ///
    /// # assert!(state.param.is_none());
    /// # assert!(state.prev_param.is_none());
    /// # assert!(state.best_param.is_none());
    /// # assert!(state.prev_best_param.is_none());
    /// # assert!(state.cost.is_none());
    /// # assert!(state.prev_cost.is_none());
    /// # assert!(state.best_cost.is_none());
    /// # assert!(state.prev_best_cost.is_none());
    /// # assert_eq!(state.target_cost, f64::NEG_INFINITY);
    /// # assert_eq!(state.iter, 0);
    /// # assert_eq!(state.last_best_iter, 0);
    /// # assert_eq!(state.max_iters, u64::MAX);
    /// # assert_eq!(state.counts, HashMap::new());
    /// # assert_eq!(state.time.unwrap(), web_time::Duration::new(0, 0));
    /// # assert_eq!(state.termination_status, TerminationStatus::NotTerminated);
    /// ```
    fn new() -> Self {
        EgorState {
            param: None,
            prev_param: None,
            best_param: None,
            prev_best_param: None,
            feasibility: false,

            cost: None,
            prev_cost: None,
            best_cost: None,
            prev_best_cost: None,
            target_cost: F::neg_infinity(),

            iter: 0,
            last_best_iter: 0,
            max_iters: u64::MAX,
            counts: HashMap::new(),
            time: Some(web_time::Duration::new(0, 0)),
            termination_status: TerminationStatus::NotTerminated,

            doe: DoeState::default(),
            surrogate: SurrogateState::default(),
            trego: TregoState::default(),
            coego: CoegoState::default(),

            #[cfg(feature = "persistent")]
            run_data: None,
            rng: Some(Xoshiro256Plus::from_entropy()),
        }
    }

    /// Checks if the current parameter vector is better than the previous best parameter value. If
    /// a new best parameter vector was found, the state is updated accordingly.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{State, ArgminFloat};
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    ///
    /// let mut state: EgorState<f64> = EgorState::new();
    ///
    /// // Simulating a new, better parameter vector
    /// let mut state = state.data((array![[1.0f64], [2.0f64], [3.0]], array![[10.0], [5.0], [0.5]], array![[], [], []]));
    /// state.iter = 2;
    /// state.surrogate.prev_best_index = Some(0);
    /// state.surrogate.best_index = Some(2);
    /// state.param = Some(array![10.0f64]);
    /// state.cost = Some(array![5.0]);
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_param.as_ref().unwrap()[0], 3.0f64);
    /// assert_eq!(state.best_cost.as_ref().unwrap()[0], 0.5);
    /// assert!(state.is_best());
    /// ```
    fn update(&mut self) {
        if let Some((x_data, y_data, c_data)) = self.surrogate.data.as_ref() {
            let best_index = self
                .surrogate
                .best_index
                .unwrap_or_else(|| find_best_result_index(y_data, c_data, &self.doe.cstr_tol));

            let param = x_data.row(best_index).to_owned();
            std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
            self.best_param = Some(param);

            let cost = y_data.row(best_index).to_owned();
            std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
            self.best_cost = Some(cost);

            if best_index > self.doe.doe_size {
                if let Some(prev_best_index) = self.surrogate.prev_best_index
                    && best_index != prev_best_index
                {
                    self.last_best_iter = self.iter + 1;
                }
            } else {
                // best point in doe => self.last_best_iter remains 0
            }
        }
    }

    /// Returns a reference to the current parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert!(state.param.is_none());
    /// # state.param = Some(array![1.0, 2.0]);
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let param = state.get_param();  // Option<&P>
    /// # assert_eq!(param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_param(&self) -> Option<&Array1<F>> {
        self.param.as_ref()
    }

    /// Returns a reference to the current best parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    ///
    /// # let mut state: EgorState<f64> = EgorState::new();
    ///
    /// # assert!(state.best_param.is_none());
    /// # state.best_param = Some(array![1.0, 2.0]);
    /// # assert_eq!(state.best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_param = state.get_best_param();  // Option<&P>
    /// # assert_eq!(best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_best_param(&self) -> Option<&Array1<F>> {
        self.best_param.as_ref()
    }

    /// Sets the termination status to [`Terminated`](`TerminationStatus::Terminated`) with the given reason
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{State, ArgminFloat, TerminationReason, TerminationStatus};
    /// # use egobox_ego::EgorState;
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.termination_status, TerminationStatus::NotTerminated);
    /// let state = state.terminate_with(TerminationReason::MaxItersReached);
    /// # assert_eq!(state.termination_status, TerminationStatus::Terminated(TerminationReason::MaxItersReached));
    /// ```
    fn terminate_with(mut self, reason: TerminationReason) -> Self {
        self.termination_status = TerminationStatus::Terminated(reason);
        self
    }

    /// Sets the time required so far.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat, TerminationReason};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// let state = state.time(Some(web_time::Duration::new(0, 12)));
    /// # assert_eq!(state.time.unwrap(), web_time::Duration::new(0, 12));
    /// ```
    fn time(&mut self, time: Option<web_time::Duration>) -> &mut Self {
        self.time = time;
        self
    }

    /// Returns current cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.cost = Some(array![12.0]);
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_cost(&self) -> Self::Float {
        match self.cost.as_ref() {
            Some(c) => *(c.get(0).unwrap_or(&Self::Float::infinity())),
            None => Self::Float::infinity(),
        }
    }

    /// Returns current best cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.best_cost = Some(array![12.0]);
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_best_cost(&self) -> Self::Float {
        match self.best_cost.as_ref() {
            Some(c) => *(c.get(0).unwrap_or(&Self::Float::infinity())),
            None => Self::Float::infinity(),
        }
    }

    /// Returns target cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::array;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.target_cost = 12.0;
    /// let target_cost = state.get_target_cost();
    /// # assert_eq!(target_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_target_cost(&self) -> Self::Float {
        self.target_cost
    }

    /// Returns current number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.iter = 12;
    /// let iter = state.get_iter();
    /// # assert_eq!(iter, 12);
    /// ```
    fn get_iter(&self) -> u64 {
        self.iter
    }

    /// Returns iteration number of last best parameter vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.last_best_iter = 12;
    /// let last_best_iter = state.get_last_best_iter();
    /// # assert_eq!(last_best_iter, 12);
    /// ```
    fn get_last_best_iter(&self) -> u64 {
        self.last_best_iter
    }

    /// Returns the maximum number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.max_iters = 12;
    /// let max_iters = state.get_max_iters();
    /// # assert_eq!(max_iters, 12);
    /// ```
    fn get_max_iters(&self) -> u64 {
        self.max_iters
    }

    /// Returns the termination status.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{State, ArgminFloat, TerminationStatus};
    /// # use egobox_ego::EgorState;
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// let termination_status = state.get_termination_status();
    /// # assert_eq!(*termination_status, TerminationStatus::NotTerminated);
    /// ```
    fn get_termination_status(&self) -> &TerminationStatus {
        &self.termination_status
    }

    /// Returns the termination reason if terminated, otherwise None.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{State, ArgminFloat, TerminationReason};
    /// # use egobox_ego::EgorState;
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// let termination_reason = state.get_termination_reason();
    /// # assert_eq!(termination_reason, None);
    /// ```
    fn get_termination_reason(&self) -> Option<&TerminationReason> {
        match &self.termination_status {
            TerminationStatus::Terminated(reason) => Some(reason),
            TerminationStatus::NotTerminated => None,
        }
    }

    /// Returns the time elapsed since the start of the optimization.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate web_time;
    /// # use web_time;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// let time = state.get_time();
    /// # assert_eq!(time.unwrap(), web_time::Duration::new(0, 0));
    /// ```
    fn get_time(&self) -> Option<web_time::Duration> {
        self.time
    }

    /// Increments the number of iterations by one
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.iter, 0);
    /// state.increment_iter();
    /// # assert_eq!(state.iter, 1);
    /// ```
    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Set all function evaluation counts to the evaluation counts of another `Problem`.
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{Problem, State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// #
    /// # #[derive(Eq, PartialEq, Debug)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # let mut problem = Problem::new(UserDefinedProblem {});
    /// # problem.counts.insert("test1", 10u64);
    /// # problem.counts.insert("test2", 2);
    /// state.func_counts(&problem);
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test1".to_string(), 10u64);
    /// # hm.insert("test2".to_string(), 2u64);
    /// # assert_eq!(state.counts, hm);
    /// ```
    fn func_counts<O>(&mut self, problem: &Problem<O>) {
        for (k, &v) in problem.counts.iter() {
            let count = self.counts.entry(k.to_string()).or_insert(0);
            *count = v
        }
    }

    /// Returns function evaluation counts
    ///
    /// # Example
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// let counts = state.get_func_counts();
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test2".to_string(), 10u64);
    /// # assert_eq!(*counts, hm);
    /// ```
    fn get_func_counts(&self) -> &HashMap<String, u64> {
        &self.counts
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    ///
    /// # Example
    ///
    /// ```
    /// # use egobox_ego::EgorState;
    /// # use argmin::core::{State, ArgminFloat};
    /// # let mut state: EgorState<f64> = EgorState::new();
    /// # state.last_best_iter = 13;
    /// # state.iter = 12;
    /// let is_best = state.is_best();
    /// # assert!(is_best);
    /// # state.last_best_iter = 12;
    /// # state.iter = 21;
    /// # let is_best = state.is_best();
    /// # assert!(!is_best);
    /// ```
    fn is_best(&self) -> bool {
        // FIXME: last best iter is 1-based index while iter is 0-based
        // This is done because last iter number is displayed in
        self.last_best_iter == self.iter + 1
    }
}

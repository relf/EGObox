//! Egor implementation of the EGO algorithm as a [basin] solver.
//!
//! The [`EgorSolver`] is run by the basin executor wrapped in the [`crate::Egor`]
//! optimizer which provides hot start checkpointing, interruption and
//! observation (history saving) features. Use [`crate::EgorBuilder`] to build it:
//!
//! ```no_run
//! use ndarray::{array, Array2, ArrayView2, Zip};
//! use egobox_ego::EgorBuilder;
//! use argmin_testfunctions::rosenbrock;
//!
//! // Rosenbrock test function: minimum y_opt = 0 at x_opt = (1, 1)
//! fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
//!     y
//! }
//! let res = EgorBuilder::optimize(rosenb)
//!             .configure(|config| config.seed(42).max_iters(20))
//!             .min_within(&array![[-2., 2.], [-2., 2.]])
//!             .expect("optimizer configured")
//!             .run()
//!             .expect("Rosenbrock minimization");
//! println!("Rosenbrock min result = {:?}", res.state);
//! ```
//!
//! Alternatively, [`crate::EgorServiceBuilder`] provides an ask-and-tell interface
//! when the optimization loop has to be controlled externally.
//!
use crate::solver::iteration_strategy::IterationMode;
use crate::solver::solver_impl::DataClustering;
use crate::utils::{
    EGOR_USE_GP_VAR_PORTFOLIO, EGOR_USE_STATE_RECORDING, filter_nans, find_best_result_index,
};
use crate::{EgoError, EgorState, MAX_POINT_ADDITION_RETRY, ValidEgorConfig};

use crate::types::*;

use egobox_doe::{Lhs, SamplingMethod};
use log::{debug, info};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip, concatenate, s};
use ndarray_npy::{read_npy, write_npy};

use crate::{TerminationReason, errors::Result};
use basin::{CostFunction, Problem};

use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::marker::PhantomData;
use std::time::Instant;

/// Numpy filename for initial DOE dump
pub const DOE_INITIAL_FILE: &str = "egor_initial_doe.npy";
/// Numpy filename for current DOE dump
pub const DOE_FILE: &str = "egor_doe.npy";

/// Default tolerance value for constraints to be satisfied (ie cstr < tol)
pub const DEFAULT_CSTR_TOL: f64 = 1e-4;
/// Termination reason when the objective function returns an error.
pub const OBJECTIVE_FUNCTION_ERROR: &str = "Objective function error";

/// EGO solver implementing the `basin::Solver` trait used by [`crate::Egor`]
/// to benefit from basin executor features (checkpointing, observers, interruption).
#[derive(Clone, Serialize, Deserialize)]
pub struct EgorSolver<SB: SurrogateBuilder, C: CstrFn = Cstr> {
    pub(crate) config: ValidEgorConfig,
    /// Matrix (nx, 2) of [lower bound, upper bound] of the nx components of x
    /// Note: used for continuous variables handling, the optimizer base.
    pub(crate) xlimits: Array2<f64>,
    /// An optional surrogate builder used to model objective and constraint
    /// functions, otherwise [mixture of expert](egobox_moe) is used
    /// Note: if specified takes precedence over individual settings
    pub(crate) surrogate_builder: SB,
    /// Phantom data for constraint function type
    pub phantom: PhantomData<C>,
}

/// Build `xtypes` from simple float bounds of `x` input components when x belongs to R^n.
/// xlimits are bounds of the x components expressed a matrix (dim, 2) where dim is the dimension of x
/// the ith row is the bounds interval [lower, upper] of the ith comonent of `x`.  
pub fn to_xtypes(xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Vec<XType> {
    let mut xtypes: Vec<XType> = vec![];
    Zip::from(xlimits.rows()).for_each(|limits| xtypes.push(XType::Float(limits[0], limits[1])));
    xtypes
}

impl<SB, C> EgorSolver<SB, C>
where
    C: CstrFn,
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
{
    /// Initialize the optimization: evaluate the initial DOE and set up the state
    pub(crate) fn init_state<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = EgoError> + Constraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> Result<EgorState<f64>> {
        let mut rng = if let Some(seed) = self.config.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };

        let warm_start_doe: Option<Array2<f64>> = if self.config.warm_start
            && let Some(path) = self.config.outdir.as_ref()
        {
            let filepath = std::path::Path::new(&path).join(DOE_FILE);
            if filepath.is_file() {
                info!("Reading DOE from {filepath:?}");
                Some(read_npy(filepath)?)
            } else if std::path::Path::new(&path).join(DOE_INITIAL_FILE).is_file() {
                let filepath = std::path::Path::new(&path).join(DOE_INITIAL_FILE);
                info!("Reading DOE from {filepath:?}");
                Some(read_npy(filepath)?)
            } else {
                None
            }
        } else {
            None
        };

        let doe = warm_start_doe.as_ref().or(self.config.doe.as_ref());

        let (y_data, x_data) = if let Some(doe) = doe {
            if doe.ncols() == self.xlimits.nrows() {
                // only x are specified
                info!("Compute initial DOE on specified {} points", doe.nrows());
                (self.eval_obj(problem, doe)?, doe.to_owned())
            } else {
                // split doe in x and y
                info!("Use specified DOE {} samples", doe.nrows());
                (
                    doe.slice(s![.., self.xlimits.nrows()..]).to_owned(),
                    doe.slice(s![.., ..self.xlimits.nrows()]).to_owned(),
                )
            }
        } else {
            let n_doe = if self.config.n_doe == 0 {
                (self.xlimits.nrows() + 1).max(5)
            } else {
                self.config.n_doe
            };
            info!("Compute initial LHS with {n_doe} points");
            let sampling = Lhs::new(&self.xlimits).with_rng(rng.clone());
            let x = sampling.sample(n_doe);
            (self.eval_obj(problem, &x)?, x)
        };
        // Warm-start DOE constraint columns are already in canonical form.
        // otherwise transform constraints to canonical form (ie. cstr < 0)
        let y_data = if !warm_start_doe.is_some()
            && let Some(ref specs) = self.config.cstr_specs
        {
            crate::types::transform_constraints(&y_data, specs)
        } else {
            y_data
        };
        let doe = concatenate![Axis(1), x_data, y_data];
        if let Some(path) = self.config.outdir.as_ref() {
            std::fs::create_dir_all(path)?;
            let filepath = std::path::Path::new(path).join(DOE_INITIAL_FILE);
            info!("Save initial doe shape {:?} in {:?}", doe.shape(), filepath);
            write_npy(filepath, &doe).expect("Write initial doe");
        }

        let n_int_cstr = self.config.n_internal_cstr();
        let clusterings = vec![None; n_int_cstr + 1];
        let theta_inits = vec![None; n_int_cstr + 1];

        let c_data = self.eval_problem_fcstrs(problem, &x_data);

        let activity = self
            .config
            .activity_strategy
            .generate_activity(self.xlimits.nrows(), &mut rng);
        debug!("Component activity = {activity:?}");

        let (valid_idx, invalid_idx) = filter_nans(&y_data);
        let x_fail = x_data.select(Axis(0), &invalid_idx).clone();
        let x_data = x_data.select(Axis(0), &valid_idx);
        let y_data = y_data.select(Axis(0), &valid_idx);
        let c_data = c_data.select(Axis(0), &valid_idx);
        if !invalid_idx.is_empty() {
            log::warn!(
                "{} failed points out of {} points in initial DOE ",
                x_fail.nrows(),
                x_data.nrows() + x_fail.nrows()
            );
        }
        let mut initial_state = state
            .data((x_data.clone(), y_data.clone(), c_data.clone()))
            .x_fail(x_fail)
            .count_added_points(valid_idx.len())
            .clusterings(clusterings)
            .theta_inits(theta_inits)
            .rng(rng);

        initial_state.doe.doe_size = doe.nrows();
        initial_state.max_iters = self.config.max_iters as u64;
        initial_state.doe.no_point_added_retries = MAX_POINT_ADDITION_RETRY;
        let n_total_cstr = n_int_cstr + c_data.ncols();
        initial_state.doe.cstr_tol = if let Some(cstr_tol) = self.config.cstr_tol.clone() {
            if cstr_tol.len() > n_total_cstr {
                return Err(EgoError::InvalidConfigError(format!(
                    "cstr_tol length ({}) is larger than total internal constraint count ({})",
                    cstr_tol.len(),
                    n_total_cstr
                )));
            }
            if cstr_tol.len() < n_total_cstr {
                let mut tol = cstr_tol.to_vec();
                tol.extend(std::iter::repeat_n(
                    DEFAULT_CSTR_TOL,
                    n_total_cstr - cstr_tol.len(),
                ));
                Array1::from_vec(tol)
            } else {
                cstr_tol
            }
        } else {
            Array1::from_elem(n_total_cstr, DEFAULT_CSTR_TOL)
        };
        initial_state.target_cost = self.config.target;

        let best_index = find_best_result_index(&y_data, &c_data, &initial_state.doe.cstr_tol);
        initial_state.surrogate.best_index = Some(best_index);
        initial_state.surrogate.prev_best_index = initial_state.surrogate.best_index;
        initial_state.last_best_iter = 0;

        // Initialize iteration strategy state (e.g., TREGO sigma)
        self.config
            .iteration_strategy
            .init_state(&mut initial_state, &self.xlimits);

        initial_state.coego.activity = activity;
        debug!("Initial State = {initial_state:?}");
        info!(
            "{} setting: {}",
            EGOR_USE_GP_VAR_PORTFOLIO, self.config.runtime_flags.use_gp_var_portfolio
        );
        info!(
            "{} setting: {}",
            EGOR_USE_STATE_RECORDING, self.config.runtime_flags.use_state_recording
        );

        info!(
            "********* Initialization: Best fun(x[{}])={} at x={}",
            best_index,
            y_data.row(best_index),
            x_data.row(best_index)
        );

        Ok(initial_state)
    }

    /// Run one iteration of the EGO algorithm
    pub(crate) fn next_state<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = EgoError> + Constraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> Result<EgorState<f64>> {
        debug!(
            "********* Start iteration {}/{}",
            state.get_iter() + 1,
            state.get_max_iters()
        );
        let now = Instant::now();

        // Use iteration strategy to determine global vs local step
        let mut state = state;
        let mode = self
            .config
            .iteration_strategy
            .prepare(&mut state, &self.xlimits);
        let mut new_state = match mode {
            IterationMode::Global => self.ego_iteration(problem, state)?,
            IterationMode::Local {
                max_dist,
                min_acceptance_distance,
            } => self.local_iteration(problem, state, max_dist, min_acceptance_distance)?,
        };
        let (x_data, y_data, _c_data) = new_state.surrogate.data.clone().unwrap();

        // Post-iteration hook
        self.config.iteration_strategy.finalize(&mut new_state);

        // Update cooperative activity for next iteration
        let new_state = {
            let nx = self.xlimits.nrows();
            let mut rng = new_state.take_rng().unwrap();
            let activity = self
                .config
                .activity_strategy
                .generate_activity(nx, &mut rng);
            debug!("Component activity = {activity:?}");
            new_state.rng(rng).activity(activity)
        };

        info!(
            "********* End iteration {}/{} in {:.3}s: Best fun(x[{}])={} at x={}",
            new_state.get_iter() + 1,
            new_state.get_max_iters(),
            now.elapsed().as_secs_f64(),
            new_state.surrogate.best_index.unwrap(),
            y_data.row(new_state.surrogate.best_index.unwrap()),
            x_data.row(new_state.surrogate.best_index.unwrap())
        );
        debug!("Current Cost {:?}", new_state.get_cost());
        debug!("Best cost {:?}", new_state.get_best_cost());
        debug!("Best index {:?}", new_state.surrogate.best_index);
        debug!("Data {:?}", new_state.surrogate.data.as_ref().unwrap());

        Ok(new_state)
    }

    /// Iteration of EGO algorithm
    fn ego_iteration<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = EgoError> + Constraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> Result<EgorState<f64>> {
        match self.ego_step(problem, state.clone()) {
            Ok(new_state) => Ok(new_state),
            Err(EgoError::NoMorePointToAddError(state)) => {
                Ok(state.terminate_with(TerminationReason::SolverConverged))
            }
            Err(EgoError::ObjectiveFunctionError(_)) => Ok(state.terminate_with(
                TerminationReason::SolverExit(OBJECTIVE_FUNCTION_ERROR.to_string()),
            )),
            Err(err) => Err(err),
        }
    }

    /// Iteration of TREGO/local algorithm using trust region bounds.
    ///
    /// Performs a local search within the trust region defined by `max_dist`
    /// around the current best point. Uses `min_acceptance_distance` to
    /// decide whether a candidate point is sufficiently far from the best.
    fn local_iteration<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = EgoError> + Constraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
        max_dist: f64,
        min_acceptance_distance: f64,
    ) -> Result<EgorState<f64>> {
        // Local step
        let mut local_state = state;
        let mut clusterings = local_state
            .take_clusterings()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No clustering!".to_string()))?;
        let mut theta_inits = local_state
            .take_theta_inits()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No theta inits!".to_string()))?;
        // Persisted models are usually up to date (or lag by the point added
        // by a previous local step), hence reused as-is or incrementally updated.
        let mut models = std::mem::take(&mut local_state.surrogate.models);
        {
            let (x_data, y_data, _) = local_state
                .surrogate
                .data
                .as_ref()
                .ok_or_else(|| EgoError::InternalError("EgorSolver: No data!".to_string()))?;
            let point_index = local_state.get_iter() as usize * self.config.qei_config.batch;
            self.refresh_models(
                &mut clusterings,
                &mut theta_inits,
                &mut models,
                x_data,
                y_data,
                &local_state.coego.activity,
                DataClustering::Fixed,
                point_index,
            );
        }
        let mut local_state = local_state
            .clusterings(clusterings)
            .theta_inits(theta_inits);
        let infill_data = self.refresh_infill_data(problem, &mut local_state, &models);
        let fallback_state = local_state.clone();
        match self.trego_step(
            problem,
            local_state,
            models,
            &infill_data,
            max_dist,
            min_acceptance_distance,
        ) {
            Ok(state) => Ok(state),
            Err(EgoError::ObjectiveFunctionError(_)) => Ok(fallback_state.terminate_with(
                TerminationReason::SolverExit(OBJECTIVE_FUNCTION_ERROR.to_string()),
            )),
            Err(err) => Err(err),
        }
    }
}

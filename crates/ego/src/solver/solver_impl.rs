use std::marker::PhantomData;

use crate::errors::{EgoError, Result};
use crate::solver::solver_computations::MiddlePickerMultiStarter;
use crate::solver::solver_infill_optim::InfillOptProblem;
use crate::utils::{
    EGOBOX_LOG, find_best_result_index_from, is_feasible, select_from_portfolio, update_data,
    usable_data,
};
use crate::{ActivityStrategy, FullActivity, find_best_result_index};
use crate::{DEFAULT_CSTR_TOL, EgorSolver, MAX_POINT_ADDITION_RETRY, ValidEgorConfig};
use crate::{EgorState, types::*};
use egobox_moe::{as_continuous_limits, to_discrete_space};

use basin::{CostFunction, Problem};

use egobox_doe::{Lhs, LhsKind};
use egobox_gp::ThetaTuning;
use env_logger::{Builder, Env};

#[cfg(feature = "persistent")]
use egobox_moe::AffinedSurrogate;
use egobox_moe::{Clustering, CorrelationSpec, MixtureGpSurrogate, NbClusters, RegressionSpec};
use log::{debug, info};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip, concatenate, s};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use serde::{Serialize, de::DeserializeOwned};

impl<SB: SurrogateBuilder + Serialize + DeserializeOwned, C: CstrFn> EgorSolver<SB, C> {
    /// Constructor of the optimization of the function `f` with specified random generator
    /// to get reproducibility.
    ///
    /// The function `f` should return an objective but also constraint values if any.
    /// Design space is specified by a list of types for input variables `x` of `f` (see [`XType`]).
    pub fn new(config: ValidEgorConfig) -> Self {
        let env = Env::new().filter_or(EGOBOX_LOG, "error");
        let mut builder = Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();
        let xtypes = config.xtypes.clone();
        EgorSolver {
            config,
            xlimits: as_continuous_limits(&xtypes),
            surrogate_builder: SB::new_with_xtypes(&xtypes),
            phantom: PhantomData,
        }
    }

    /// Given an evaluated doe (x, y) data, return the next promising x point
    /// where optimum may occurs regarding the infill criterium.
    /// This function inverse the control of the optimization and can used
    /// ask-and-tell interface to the EGO optimizer.
    pub fn suggest(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        // Apply constraint transformation if cstr_specs are set
        let y_data_owned;
        let y_data: &ArrayBase<_, Ix2> = if let Some(ref specs) = self.config.cstr_specs {
            y_data_owned = crate::types::transform_constraints(&y_data.to_owned(), specs);
            &y_data_owned
        } else {
            y_data_owned = y_data.to_owned();
            &y_data_owned
        };
        let mut rng = if let Some(seed) = self.config.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };
        let n_int_cstr = self.config.n_internal_cstr();
        let mut clusterings = vec![None; 1 + n_int_cstr];
        let mut theta_tunings = vec![None; 1 + n_int_cstr];
        let cstr_tol = self
            .config
            .cstr_tol
            .clone()
            .unwrap_or(Array1::from_elem(n_int_cstr, DEFAULT_CSTR_TOL));

        // TODO: Manage fonction constraints
        let fcstrs = Vec::<Cstr>::new();
        let fcstr_specs = None;
        // TODO: c_data has to be passed as argument or better computed using fcstrs(x_data)
        let c_data = Array2::zeros((x_data.nrows(), 0));
        // TODO: Coego not implemented
        let activity = FullActivity.generate_activity(x_data.ncols(), &mut rng);

        let best_index = find_best_result_index(y_data, &c_data, &cstr_tol);
        let feasibility = is_feasible(&y_data.row(best_index), &c_data.row(best_index), &cstr_tol);

        let mut models: Vec<Box<dyn MixtureGpSurrogate>> = Vec::new();
        let (x_dat, _, _, _, _) = self.select_next_points(
            true,
            0,
            false, // done anyway
            &mut clusterings,
            &mut theta_tunings,
            &mut models,
            &activity,
            x_data,
            y_data,
            &c_data,
            None,
            &cstr_tol,
            best_index,
            &fcstrs,
            fcstr_specs,
            feasibility,
            &mut rng,
        );
        x_dat
    }
}

/// Minimum number of training points required before considering
/// the surrogate data-starved wrt to the dimension of the problem.
const MIN_POINTS_THRESOLD: usize = 50;

/// Below `MIN_POINTS_DIM_FACTOR * dim` training points, the surrogate is
/// considered data-starved: theta hyperparameters are always (re)optimized
/// from scratch instead of reused/incrementally updated.
const MIN_POINTS_DIM_FACTOR: usize = 2;

/// z-score threshold on the newly added point(s) prediction error
/// `(y_new - y_pred).abs() / var_pred.sqrt()`, above which the incremental
/// "fast path" model update is abandoned in favor of a full retraining with
/// theta optimization: a large z-score means the current hyperparameters no
/// longer explain the data well.
const ZSCORE_THETA_OPTIM_THRESHOLD: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DataClustering {
    /// Clustering is not updated given values are used as is
    Fixed,
    /// Clustering is recomputed
    Regenerate,
}

impl From<bool> for DataClustering {
    fn from(value: bool) -> Self {
        if value {
            DataClustering::Regenerate
        } else {
            DataClustering::Fixed
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ThetaOptimization {
    /// Theta is not optimized given values are used as is
    Disabled,
    /// Theta is optimized with given values as initialization
    Enabled,
}

impl From<bool> for ThetaOptimization {
    fn from(value: bool) -> Self {
        if value {
            ThetaOptimization::Enabled
        } else {
            ThetaOptimization::Disabled
        }
    }
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
    C: CstrFn,
{
    /// Whether we have to recluster the data
    pub fn have_to_recluster(&self, added: usize, prev_added: usize) -> bool {
        self.config.gp.n_clusters.is_auto()
            && (added != 0 && added.is_multiple_of(10) && added - prev_added > 0)
    }

    /// Build surrogate given training data and surrogate builder
    /// Reclustering is triggered when make_clustering boolean is true otherwise
    /// previous clustering is used. theta_init allows to reuse
    /// previous theta without fully retraining the surrogates
    /// (faster execution at the cost of surrogate quality)
    #[allow(clippy::too_many_arguments)]
    fn make_clustered_surrogate(
        &self,
        model_name: &str,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        make_clustering: DataClustering,
        optimize_theta: ThetaOptimization,
        clustering: Option<&Clustering>,
        theta_inits: Option<&Array2<f64>>,
        actives: &Array2<usize>,
    ) -> (Box<dyn MixtureGpSurrogate>, Array2<f64>) {
        let mut builder = self.surrogate_builder.clone();
        builder.set_kpls_dim(self.config.gp.kpls_dim);
        builder.set_regression_spec(self.config.gp.regression_spec);
        builder.set_correlation_spec(self.config.gp.correlation_spec);
        builder.set_n_clusters(self.config.gp.n_clusters.clone());
        builder.set_recombination(self.config.gp.recombination);
        builder.set_optim_params(self.config.gp.n_start, self.config.gp.max_eval);
        let mut model = None;

        let dim = self.config.gp.kpls_dim.unwrap_or(xt.ncols());

        let mut best_theta_inits = if let Some(inits) = theta_inits {
            inits.to_owned()
        } else {
            let nb = self.config.gp.n_clusters.n_or_else_one();
            let mut inits = Array2::zeros((nb, dim));
            let default_init = self.config.gp.theta_tuning.init();
            Zip::from(inits.rows_mut()).for_each(|mut r| r.assign(default_init));
            inits
        };

        let theta_bounds = crate::utils::theta_bounds(
            &self.config.gp.theta_tuning,
            dim,
            self.config.gp.correlation_spec,
        );

        let actives = if self.config.gp.kpls_dim.is_some() {
            // KPLS takes priority: use full theta optimization in PLS-reduced space
            FullActivity.activity(actives.len())
        } else {
            // Otherwise, use activity strategy to determine active variables for theta optimization
            actives.to_owned()
        };

        for (i, active) in actives.outer_iter().enumerate() {
            let gp = match make_clustering {
                DataClustering::Regenerate => {
                    /* init || recluster */
                    match self.config.gp.n_clusters {
                        NbClusters::Auto { max: _ } => {
                            if !self.config.activity_strategy.supports_auto_clustering() {
                                log::warn!(
                                    "Automated clustering not available with cooperative activity strategy"
                                )
                            }
                        }
                        NbClusters::Fixed { nb: _ } => {
                            let theta_tunings = match optimize_theta {
                                ThetaOptimization::Enabled => {
                                    // set hyperparameters optimization
                                    if self.config.gp.kpls_dim.is_some() {
                                        // KPLS takes priority: use full theta optimization
                                        // in PLS-reduced space
                                        best_theta_inits
                                            .outer_iter()
                                            .map(|init| ThetaTuning::Full {
                                                init: init.to_owned(),
                                                bounds: theta_bounds.to_owned(),
                                            })
                                            .collect::<Vec<_>>()
                                    } else {
                                        best_theta_inits
                                            .outer_iter()
                                            .map(|init| ThetaTuning::Partial {
                                                init: init.to_owned(),
                                                bounds: theta_bounds.to_owned(),
                                                active: Self::strip(&active.to_vec(), init.len()),
                                            })
                                            .collect::<Vec<_>>()
                                    }
                                }
                                ThetaOptimization::Disabled => {
                                    // just use previous hyperparameters
                                    best_theta_inits
                                        .outer_iter()
                                        .map(|init| ThetaTuning::Fixed(init.to_owned()))
                                        .collect::<Vec<_>>()
                                }
                            };
                            builder.set_theta_tunings(&theta_tunings);
                            if i == 0 && model_name == "Objective" {
                                info!(
                                    "Objective model hyperparameters optim init >>> {theta_tunings:?}"
                                );
                            }
                        }
                    }

                    if i == 0 {
                        info!("{model_name} clustering and training...");
                    }
                    let gp = builder
                        .train(xt.view(), yt.view())
                        .expect("GP training failure");

                    if i == 0 {
                        info!(
                            "... {} trained ({} / {})",
                            model_name,
                            gp.n_clusters(),
                            gp.recombination()
                        );
                    }
                    gp
                }
                DataClustering::Fixed => {
                    let clustering = clustering.unwrap();

                    let theta_tunings = match optimize_theta {
                        ThetaOptimization::Enabled => {
                            // set hyperparameters optimization
                            let mut inits = best_theta_inits
                                .outer_iter()
                                .map(|init| ThetaTuning::Full {
                                    init: init.to_owned(),
                                    bounds: theta_bounds.to_owned(),
                                })
                                .collect::<Vec<_>>();
                            // When KPLS is active, keep full theta optimization
                            // in PLS-reduced space instead of partial optimization
                            if self.config.gp.kpls_dim.is_none() {
                                self.config
                                    .activity_strategy
                                    .adjust_theta_tuning(&active.to_vec(), &mut inits);
                            }
                            if i == 0 && model_name == "Objective" {
                                info!("Objective model hyperparameters optim init >>> {inits:?}");
                            }
                            inits
                        }
                        ThetaOptimization::Disabled => {
                            // just use previous hyperparameters
                            let inits = best_theta_inits
                                .outer_iter()
                                .map(|init| ThetaTuning::Fixed(init.to_owned()))
                                .collect::<Vec<_>>();
                            if i == 0 && model_name == "Objective" {
                                info!("Objective model hyperparameters reused >>> {inits:?}");
                            }
                            inits
                        }
                    };

                    builder.set_theta_tunings(&theta_tunings);

                    builder
                        .train_on_clusters(xt.view(), yt.view(), clustering)
                        .expect("GP training failure")
                }
            };

            best_theta_inits = Array2::from_shape_vec(
                (gp.experts().len(), gp.experts()[0].theta().len()),
                gp.experts()
                    .iter()
                    .flat_map(|expert| expert.theta().to_vec())
                    .collect(),
            )
            .expect("Theta initialization failure");

            // Cooperative activity: update theta in mono cluster setting
            if self.config.activity_strategy.is_cooperative() {
                if self.config.gp.n_clusters.is_mono() {
                    best_theta_inits = Array2::from_shape_vec(
                        (gp.experts().len(), gp.experts()[0].theta().len()),
                        gp.experts()
                            .iter()
                            .flat_map(|expert| expert.theta().to_vec())
                            .collect(),
                    )
                    .expect("Theta initialization failure");
                } else {
                    log::warn!(
                        "Cooperative activity theta update wrt likelihood not implemented in multi-cluster setting"
                    );
                }
            };
            model = Some(gp)
        }
        (model.expect("Surrogate model is trained"), best_theta_inits)
    }

    /// Train GP model for viability surrogate (binary classification)
    /// given training data and surrogate builder
    pub fn make_viability_surrogate(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        x_fail: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Box<dyn MixtureGpSurrogate> {
        let mut builder = self.surrogate_builder.clone();

        // Use kpls reduction if nx>10
        if x_data.ncols() > 10 {
            builder.set_kpls_dim(Some(10));
        }

        builder.set_regression_spec(RegressionSpec::CONSTANT);
        builder.set_correlation_spec(CorrelationSpec::ABSOLUTEEXPONENTIAL);
        builder.set_n_clusters(NbClusters::Fixed { nb: 1 });
        builder.set_recombination(egobox_moe::Recombination::Hard);
        // builder.set_optim_params(self.config.gp.n_start, self.config.gp.max_eval);

        let xt = concatenate(Axis(0), &[x_data.view(), x_fail.view()]).unwrap();

        let mut yt = Array1::ones(x_data.nrows() + x_fail.nrows());
        yt.slice_mut(s![x_data.nrows()..]).fill(0.0); // failed points labeled as 0.0

        info!("Viability GP training...");
        let gp = builder
            .train(xt.view(), yt.view())
            .expect("Viability GP training failure");
        info!(
            "... Viability GP trained ({} / {})",
            gp.n_clusters(),
            gp.recombination()
        );
        gp
    }

    /// Refresh infill data used to optimize infill criterion
    pub fn refresh_infill_data<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = crate::EgoError>
            + Constraints<C>,
    >(
        &self,
        problem: &mut Problem<O>,
        state: &mut EgorState<f64>,
        models: &[Box<dyn egobox_moe::MixtureGpSurrogate>],
    ) -> InfillObjData<f64> {
        let y_data = state.surrogate.data.as_ref().unwrap().1.clone();
        let x_data = state.surrogate.data.as_ref().unwrap().0.clone();
        let (obj_model, cstr_models) = models.split_first().unwrap();

        let fmin = y_data[[state.surrogate.best_index.unwrap(), 0]];
        let xbest = x_data.row(state.surrogate.best_index.unwrap()).to_vec();

        let pb = problem.inner();
        let fcstrs = pb.constraints();
        let fcstr_specs = pb.constraint_specs();

        let fcstr_mapping = crate::types::function_cstr_affine_mapping(fcstrs.len(), fcstr_specs)
            .expect("validated function-constraint specs");
        let transformed_fcstrs = fcstr_mapping
            .iter()
            .map(|(raw_idx, affine_scale, affine_offset)| {
                let cstr = fcstrs[*raw_idx].clone();
                let affine_scale = *affine_scale;
                let affine_offset = *affine_offset;
                move |x: &[f64],
                      gradient: Option<&mut [f64]>,
                      params: &mut InfillObjData<f64>|
                      -> f64 {
                    if let Some(g) = gradient {
                        let raw = cstr(x, Some(g), params);
                        g.iter_mut().for_each(|v| *v *= affine_scale);
                        affine_scale * raw + affine_offset
                    } else {
                        affine_scale * cstr(x, None, params) + affine_offset
                    }
                }
            })
            .collect::<Vec<_>>();

        let mut rng = state.take_rng().unwrap();
        let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
        *state = state.clone().rng(rng.clone());
        let sampling = Lhs::new(&self.xlimits)
            .with_rng(sub_rng)
            .kind(LhsKind::Maximin);
        let cstr_tol = self.config.cstr_tol.clone().unwrap_or(Array1::from_elem(
            self.config.n_internal_cstr(),
            DEFAULT_CSTR_TOL,
        ));
        let (scale_infill_obj, scale_cstr, scale_fcstr, scale_wb2) = self.compute_scaling(
            &sampling,
            obj_model.as_ref(),
            cstr_models,
            &cstr_tol,
            &transformed_fcstrs,
            fmin,
            1., // FIXME: TREGO does not use sigma weighting portfolio
        );

        let all_scale_cstr = concatenate![Axis(0), scale_cstr, scale_fcstr];

        InfillObjData {
            fmin,
            xbest,
            scale_infill_obj,
            scale_cstr: Some(all_scale_cstr.to_owned()),
            scale_wb2,
            feasibility: state.feasibility,
            sigma_weight: 1., // FIXME: TREGO does not use sigma weighting portfolio
        }
    }

    /// Train all constraint columns in parallel (no deduplication).
    #[allow(clippy::too_many_arguments)]
    fn train_all_columns(
        &self,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        do_clustering: DataClustering,
        optimize_theta: ThetaOptimization,
        clusterings: &[Option<Clustering>],
        theta_inits: &[Option<Array2<f64>>],
        actives: &Array2<usize>,
    ) -> (Vec<Box<dyn MixtureGpSurrogate>>, Vec<Array2<f64>>) {
        let models_and_inits = (0..=self.config.n_internal_cstr())
            .into_par_iter()
            .map(|k| {
                let name = if k == 0 {
                    "Objective".to_string()
                } else {
                    format!("Constraint[{k}]")
                };
                self.make_clustered_surrogate(
                    &name,
                    xt,
                    &yt.slice(s![.., k]).to_owned(),
                    do_clustering,
                    optimize_theta,
                    clusterings[k].as_ref(),
                    theta_inits[k].as_ref(),
                    actives,
                )
            });
        models_and_inits.unzip()
    }

    /// Train only primary constraint columns and derive transformed ones
    /// via [`AffinedSurrogate`] wrappers.
    #[cfg(feature = "persistent")]
    #[allow(clippy::too_many_arguments)]
    fn train_with_mapping(
        &self,
        mapping: &[InternalCstrKind],
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        do_clustering: DataClustering,
        optimize_theta: ThetaOptimization,
        clusterings: &[Option<Clustering>],
        theta_inits: &[Option<Array2<f64>>],
        actives: &Array2<usize>,
    ) -> (Vec<Box<dyn MixtureGpSurrogate>>, Vec<Array2<f64>>) {
        let primary_indices: Vec<usize> = mapping
            .iter()
            .enumerate()
            .filter_map(|(i, kind)| matches!(kind, InternalCstrKind::Primary).then_some(i))
            .collect();

        let primary_results: Vec<(usize, Box<dyn MixtureGpSurrogate>, Array2<f64>)> =
            primary_indices
                .into_par_iter()
                .map(|k| {
                    let name = if k == 0 {
                        "Objective".to_string()
                    } else {
                        format!("Constraint[{k}]")
                    };
                    let (model, inits) = self.make_clustered_surrogate(
                        &name,
                        xt,
                        &yt.slice(s![.., k]).to_owned(),
                        do_clustering,
                        optimize_theta,
                        clusterings[k].as_ref(),
                        theta_inits[k].as_ref(),
                        actives,
                    );
                    (k, model, inits)
                })
                .collect();

        let mut models: Vec<Option<Box<dyn MixtureGpSurrogate>>> =
            (0..mapping.len()).map(|_| None).collect();
        let mut inits: Vec<Option<Array2<f64>>> = (0..mapping.len()).map(|_| None).collect();

        for (k, model, init) in primary_results {
            models[k] = Some(model);
            inits[k] = Some(init);
        }

        for (k, kind) in mapping.iter().enumerate() {
            if let InternalCstrKind::Derived {
                source,
                scale,
                offset,
            } = kind
            {
                let source_model = models[*source].as_ref().unwrap();
                let cloned = source_model.clone();
                models[k] = Some(Box::new(AffinedSurrogate::new(cloned, *scale, *offset)));
                inits[k] = inits[*source].clone();
            }
        }

        (
            models.into_iter().map(|o| o.unwrap()).collect(),
            inits.into_iter().map(|o| o.unwrap()).collect(),
        )
    }

    /// This function is the main EGO algorithm iteration:
    /// * Train surrogates
    /// * Find next promising location(s) of optimum
    /// * Update state: Evaluate true function, update doe and optimum
    #[allow(clippy::type_complexity)]
    pub fn ego_step<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = crate::EgoError>
            + Constraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> Result<EgorState<f64>> {
        let mut new_state = state.clone();
        let mut clusterings = new_state
            .take_clusterings()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No clustering!".to_string()))?;
        let mut theta_inits = new_state
            .take_theta_inits()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No theta inits!".to_string()))?;
        #[cfg(feature = "persistent")]
        let mut models = std::mem::take(&mut new_state.surrogate.models);
        #[cfg(not(feature = "persistent"))]
        let mut models: Vec<Box<dyn MixtureGpSurrogate>> = Vec::new();
        // Under `--no-default-features` (no "persistent"), `models` is never
        // consulted (see the `#[cfg(feature = "persistent")]`-gated blocks
        // below): keep it referenced so it isn't flagged as unused.
        #[cfg(not(feature = "persistent"))]
        let _ = &models;

        let mut rng = new_state
            .take_rng()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No rng!".to_string()))?;
        let (mut x_data, mut y_data, mut c_data) = new_state
            .take_data()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No data!".to_string()))?;

        // Computed once and reused both for the batch (virtual points) search below
        // and, further down, to incorporate the actually evaluated point(s) into
        // the persisted `models` (neither `new_state.doe.{added,prev_added}` nor
        // `new_state.get_iter()` change until after this whole retry loop).
        let recluster = self.have_to_recluster(new_state.doe.added, new_state.doe.prev_added);
        if recluster {
            info!("Reclustering surrogates...");
        }
        let init = new_state.get_iter() == 0;

        let (x_dat, c_dat, y_penalized) = loop {
            let pb = problem.inner();
            let fcstrs = pb.constraints();
            let fcstr_specs = pb.constraint_specs();

            // Batch (q-points > 1) selection picks points i=1..batch-1 using
            // Kriging-believer virtual y-values (predicted, not evaluated), which
            // must never pollute the persisted `models` (only the actually
            // evaluated point(s) get incorporated into them, further below).
            // With batch == 1 there is only ever i == 0 (no virtual point is ever
            // constructed), so it's safe -- and avoids a needless clone -- to hand
            // `models` over directly rather than searching on a clone of it.
            #[cfg(feature = "persistent")]
            let mut search_models: Vec<Box<dyn MixtureGpSurrogate>> =
                if self.config.qei_config.batch > 1 {
                    models.clone()
                } else {
                    std::mem::take(&mut models)
                };
            #[cfg(not(feature = "persistent"))]
            let mut search_models: Vec<Box<dyn MixtureGpSurrogate>> = Vec::new();

            let (x_dat, y_dat, c_dat, y_penalized, infill_value) = self.select_next_points(
                init,
                state.get_iter(),
                recluster,
                &mut clusterings,
                &mut theta_inits,
                &mut search_models,
                &state.coego.activity,
                &x_data,
                &y_data,
                &c_data,
                state.surrogate.x_fail.as_ref(),
                &state.doe.cstr_tol,
                state.surrogate.best_index.unwrap(),
                fcstrs,
                fcstr_specs,
                state.feasibility,
                &mut rng,
            );

            // batch == 1: `search_models` was `models` itself (moved out above,
            // untouched by any virtual point), so move it right back.
            #[cfg(feature = "persistent")]
            if self.config.qei_config.batch <= 1 {
                models = search_models;
            }

            debug!("Try adding {x_dat}");
            let usable_indices = usable_data(&x_data, &x_dat);

            new_state = new_state
                .clusterings(clusterings.clone())
                .theta_inits(theta_inits.clone())
                .data((x_data.clone(), y_data.clone(), c_data.clone()))
                .infill_value(infill_value)
                .rng(rng.clone())
                .param(x_dat.row(0).to_owned()) // Note: take only first point.
                .cost(y_dat.row(0).to_owned()); // Argmin framework requires param and cost to be set.

            info!(
                "{} criterion {} max found = {}",
                if self.config.cstr_infill {
                    "Constrained infill"
                } else {
                    "Infill"
                },
                self.config.infill_criterion.name(),
                new_state.get_infill_value()
            );

            let rejected_count = x_dat.nrows() - usable_indices.len();
            for i in 0..x_dat.nrows() {
                let msg = format!(
                    "  {} {}",
                    if usable_indices.contains(&i) {
                        "A"
                    } else {
                        "R"
                    },
                    x_dat.row(i)
                );
                if usable_indices.contains(&i) {
                    debug!("{msg}");
                } else {
                    info!("{msg}")
                }
            }
            if rejected_count > 0 {
                info!(
                    "Reject {}/{} point{} too close to previous ones",
                    rejected_count,
                    x_dat.nrows(),
                    if rejected_count > 1 { "s" } else { "" }
                );
            }
            if rejected_count == x_dat.nrows() {
                new_state.doe.no_point_added_retries -= 1;
                if new_state.doe.no_point_added_retries == 0 {
                    info!(
                        "Max number of retries ({}) without adding point",
                        MAX_POINT_ADDITION_RETRY
                    );
                    info!("Consider solver has converged");
                    return Err(EgoError::NoMorePointToAddError(Box::new(new_state)));
                }
            } else {
                // ok point added we can go on, just output number of rejected points
                let x_dat = x_dat.select(Axis(0), &usable_indices);
                let c_dat = c_dat.select(Axis(0), &usable_indices);
                break (x_dat, c_dat, y_penalized);
            }
        };

        let y_actual = self.eval_obj(problem, &x_dat)?;
        // Apply constraint transformation if cstr_specs are set
        let y_actual = if let Some(ref specs) = self.config.cstr_specs {
            crate::types::transform_constraints(&y_actual, specs)
        } else {
            y_actual
        };
        let y_penalized = match self.config.failsafe_strategy {
            FailsafeStrategy::Imputation => Some(y_penalized),
            _ => None,
        };
        let (add_count, x_fail_points) = update_data(
            &mut x_data,
            &mut y_data,
            &mut c_data,
            &x_dat,
            &y_actual,
            &c_dat,               // fcstr evaluation already done in select_next_points
            y_penalized.as_ref(), // penalized values in case of crash
        );

        new_state = if state.get_iter() == 0
            && self.config.failsafe_strategy == FailsafeStrategy::Imputation
            && let Some(ref xfail) = x_fail_points
            && let Some(ref xfail_doe) = new_state.surrogate.x_fail
        {
            // In first iteration, we had doe failed points stored
            // Store only new failed points (not in doe)
            info!("Initial DOE had {} failed point(s)", xfail_doe.nrows());
            info!("Total failed points after eval: {}", xfail.nrows());
            info!("{} new failed point(s)", xfail.nrows() - xfail_doe.nrows());
            let new_fails = xfail.slice(s![xfail_doe.nrows().., ..]).to_owned();
            new_state.store_failed_points(Some(new_fails))
        } else {
            new_state.store_failed_points(x_fail_points)
        }
        .count_added_points(add_count);

        // new_state = new_state
        //     .store_failed_points(x_fail_points)
        //     .count_added_points(add_count);
        info!(
            "+{} point(s), total: {} points",
            add_count, new_state.doe.added
        );
        new_state.doe.no_point_added_retries = MAX_POINT_ADDITION_RETRY;

        let best_index = find_best_result_index_from(
            state.surrogate.best_index.unwrap(),
            y_data.nrows() - add_count,
            &y_data,
            &c_data,
            &new_state.doe.cstr_tol,
        );
        new_state =
            new_state
                .best_index(best_index)
                .data((x_data.clone(), y_data.clone(), c_data.clone()));
        new_state.feasibility = state.feasibility
            || is_feasible(
                &y_data.row(best_index),
                &c_data.row(best_index),
                &new_state.doe.cstr_tol,
            );
        #[cfg(feature = "persistent")]
        {
            // Incorporate the actually evaluated point(s) into the persisted
            // models (never the virtual/Kriging-believer values used only for
            // the batch search above, see `search_models`): see
            // `must_optimize_theta` for the full-retrain-vs-fast-path decision.
            let do_clustering: DataClustering = (init || recluster).into();
            let point_index = state.get_iter() as usize * self.config.qei_config.batch;
            self.refresh_models(
                &mut clusterings,
                &mut theta_inits,
                &mut models,
                &x_data,
                &y_data,
                &state.coego.activity,
                do_clustering,
                point_index,
            );
            new_state = new_state
                .clusterings(clusterings.clone())
                .theta_inits(theta_inits.clone());
            new_state.surrogate.models = models;
        }
        Ok(new_state)
    }

    /// Returns next promising x points together with virtual (predicted) y values
    /// from surrogate models (taking into account qei strategy if q_parallel)
    /// infill criterion value is also returned
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn select_next_points(
        &self,
        init: bool,
        iter: u64,
        recluster: bool,
        clusterings: &mut [Option<Clustering>],
        theta_inits: &mut [Option<Array2<f64>>],
        models: &mut Vec<Box<dyn MixtureGpSurrogate>>,
        activity: &Array2<usize>,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        c_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        x_fail_points: Option<&Array2<f64>>,
        cstr_tol: &Array1<f64>,
        best_index: usize,
        cstr_funcs: &[impl CstrFn],
        fcstr_specs: Option<&[CstrSpec]>,
        feasibility: bool,
        rng: &mut Xoshiro256Plus,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, f64) {
        let mut portfolio = vec![];

        let sigma_weights = if self.config.runtime_flags.use_gp_var_portfolio
            && self.config.qei_config.batch == 1
        {
            // Do not believe GP variance, weight it to generate possibly several clusters
            // hence several points to add
            // logspace(0.1, 100., 13) with 1. moved in front
            vec![
                1.,
                0.1,
                0.1778279410038923,
                0.31622776601683794,
                0.5623413251903491,
                1.7782794100389228,
                3.1622776601683795,
                5.623413251903491,
                10.,
                17.78279410038923,
                31.622776601683793,
                56.23413251903491,
                100.,
            ]
        } else {
            // Fallback to default GP usage
            vec![1.]
        };

        for (j, sigma_weight) in sigma_weights.iter().enumerate() {
            debug!("Make surrogate with {x_data}");
            let mut x_dat = Array2::zeros((0, x_data.ncols()));
            let mut y_dat = Array2::zeros((0, y_data.ncols()));
            let mut c_dat = Array2::zeros((0, c_data.ncols()));
            let mut y_penalized = Array2::zeros((0, y_data.ncols()));
            let mut infill_val = f64::INFINITY;

            for i in 0..self.config.qei_config.batch {
                let (xt, yt) = if i == 0 {
                    (x_data.to_owned(), y_data.to_owned())
                } else {
                    (
                        concatenate![Axis(0), x_data.to_owned(), x_dat.to_owned()],
                        concatenate![Axis(0), y_data.to_owned(), y_dat.to_owned()],
                    )
                };

                log::debug!("activity: {activity:?}");
                let actives = activity;

                let do_clustering: DataClustering = ((init && i == 0) || recluster).into();

                // The fast (incremental) path is the default strategy; see
                // `must_optimize_theta` for when a full retrain is triggered instead.
                // Otherwise the fast path is attempted, itself able to escalate to a
                // full theta-optimized retraining should the incoming point(s) turn
                // out to be a poor surprise for the current surrogate, i.e. its
                // z-score exceeds `ZSCORE_THETA_OPTIM_THRESHOLD` (see
                // `update_models`'s z-score check).
                let dim = self.config.gp.kpls_dim.unwrap_or(xt.ncols());
                let point_index = iter as usize * self.config.qei_config.batch + i;
                let optimize_theta: ThetaOptimization = (j == 0
                    && self.must_optimize_theta(do_clustering, xt.nrows(), dim, point_index))
                .into();

                // Persisted models are already trained on the evaluated points at
                // the end of the previous iteration: do not retrain them on the
                // very same data.
                let inits = if Self::models_up_to_date(models, do_clustering, xt.nrows()) {
                    debug!("Surrogates already up to date with {} points", xt.nrows());
                    vec![None; models.len()]
                } else {
                    info!(
                        "Update surrogates with {} points... clustering={:?} optimize_theta={:?}",
                        xt.nrows(),
                        do_clustering,
                        optimize_theta
                    );
                    self.update_models(
                        clusterings,
                        theta_inits,
                        models,
                        &xt,
                        &yt,
                        actives,
                        do_clustering,
                        optimize_theta,
                    )
                };

                // Handle failsafe imputation on the first iteration
                if iter == 0
                    && i == 0
                    && let Some(xfail_points) = x_fail_points
                {
                    match self.config.failsafe_strategy {
                        FailsafeStrategy::Imputation => {
                            // we have to predict penalization on these points
                            // and add them to data

                            info!(
                                "Impute failed initial points ({} points)...",
                                xfail_points.nrows()
                            );
                            let mut y_pen_imputed = Array2::zeros((
                                xfail_points.nrows(),
                                1 + self.config.n_internal_cstr(),
                            ));
                            Zip::from(y_pen_imputed.rows_mut())
                                .and(xfail_points.rows())
                                .for_each(|mut y_row, xfail| {
                                    let y_pred = self.compute_penalized_point(
                                        &xfail,
                                        &*models[0],
                                        &models[1..],
                                    );
                                    y_row.assign(&y_pred);
                                });
                            (x_dat, y_dat, c_dat, y_penalized) = (
                                xfail_points.to_owned(),
                                Array2::from_elem((xfail_points.nrows(), y_data.ncols()), f64::NAN),
                                self.eval_fcstrs(cstr_funcs, xfail_points),
                                y_pen_imputed,
                            );
                        }
                        FailsafeStrategy::Rejection | FailsafeStrategy::Viability => (),
                    }
                };

                #[cfg(feature = "persistent")]
                if self.config.outdir.is_some() && i == 0 && j == 0 {
                    use crate::utils::{EGOR_GP_FILENAME, EGOR_INITIAL_GP_FILENAME, gp_recorder};

                    let default_dir = String::from("./");
                    let outdir = self.config.outdir.as_ref().unwrap_or(&default_dir);
                    let filename = if iter == 0 {
                        EGOR_INITIAL_GP_FILENAME
                    } else {
                        EGOR_GP_FILENAME
                    };
                    let filepath = std::path::Path::new(outdir).join(filename);
                    match gp_recorder::save_gp_models(&filepath, models) {
                        Ok(_) => log::info!("GP models saved to {:?}", filepath),
                        Err(err) => log::info!("Cannot save GP models: {:?}", err),
                    };
                }

                self.sync_clustering_and_theta_inits(clusterings, theta_inits, models, &inits);

                let (obj_model, cstr_models) = models.split_first().unwrap();
                debug!("... surrogates trained");

                let fmin = y_data[[best_index, 0]];
                let ybest = y_data.row(best_index).to_owned();
                let xbest = x_data.row(best_index).to_owned();
                let cbest = c_data.row(best_index).to_owned();

                let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
                let sampling = Lhs::new(&self.xlimits)
                    .kind(LhsKind::Maximin)
                    .with_rng(sub_rng);

                let fcstr_mapping =
                    crate::types::function_cstr_affine_mapping(cstr_funcs.len(), fcstr_specs)
                        .expect("validated function-constraint specs");

                let transformed_fcstrs = fcstr_mapping
                    .iter()
                    .map(|(raw_idx, affine_scale, affine_offset)| {
                        let cstr = cstr_funcs[*raw_idx].clone();
                        let affine_scale = *affine_scale;
                        let affine_offset = *affine_offset;
                        move |x: &[f64],
                              gradient: Option<&mut [f64]>,
                              params: &mut InfillObjData<f64>|
                              -> f64 {
                            if let Some(g) = gradient {
                                let raw = cstr(x, Some(g), params);
                                g.iter_mut().for_each(|v| *v *= affine_scale);
                                affine_scale * raw + affine_offset
                            } else {
                                affine_scale * cstr(x, None, params) + affine_offset
                            }
                        }
                    })
                    .collect::<Vec<_>>();

                let (scale_infill_obj, scale_cstr, scale_fcstr, scale_wb2) = self.compute_scaling(
                    &sampling,
                    obj_model.as_ref(),
                    cstr_models,
                    cstr_tol,
                    &transformed_fcstrs,
                    fmin,
                    *sigma_weight,
                );
                let scale_pov_cstr = Array1::ones((1,)); // PoV cstr is normalized 
                let all_scale_cstr = concatenate![Axis(0), scale_cstr, scale_fcstr, scale_pov_cstr];

                // fmin and xbest are kept the same for all q points
                // Would it be best to update them with regard to predicted virtual points?
                // Keep it simple: For the moment we keep them fixed to the current best observed point
                let mut infill_data = InfillObjData {
                    fmin,
                    xbest: xbest.to_vec(),
                    scale_infill_obj,
                    scale_cstr: Some(all_scale_cstr.to_owned()),
                    scale_wb2,
                    feasibility,
                    sigma_weight: *sigma_weight,
                };

                let cstr_funcs = transformed_fcstrs
                    .iter()
                    .enumerate()
                    .map(|(i, cstr)| {
                        let scale_fc = scale_fcstr[i];
                        move |x: &[f64],
                              gradient: Option<&mut [f64]>,
                              params: &mut InfillObjData<f64>|
                              -> f64 {
                            let x = if self.config.discrete() {
                                let xary =
                                    Array2::from_shape_vec((1, x.len()), x.to_vec()).unwrap();
                                // We have to cast x to folded space as EgorSolver
                                // works internally in the continuous space while
                                // the constraint function expects discrete variable in folded space
                                to_discrete_space(&self.config.xtypes, &xary)
                                    .row(0)
                                    .into_owned();
                                &xary.into_iter().collect::<Vec<_>>()
                            } else {
                                x
                            };
                            if let Some(g) = gradient {
                                let v = cstr(x, Some(g), params) / scale_fc;
                                g.iter_mut().for_each(|gi| *gi /= scale_fc);
                                v
                            } else {
                                cstr(x, None, params) / scale_fc
                            }
                        }
                    })
                    .collect::<Vec<_>>();

                // Make viability surrogate
                let viability_model = if (self.config.failsafe_strategy
                    == FailsafeStrategy::Viability
                    || self.config.feasibility_infill.is_enabled())
                    && let Some(points) = x_fail_points
                    && points.nrows() > 0
                {
                    info!(
                        "Build viability surrogate with {} safe and {} failed points...",
                        x_data.nrows(),
                        points.nrows()
                    );
                    x_fail_points
                        .map(|xfail_points| self.make_viability_surrogate(&xt, xfail_points))
                } else {
                    None
                };

                let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
                // let multistarter = GlobalMultiStarter::new(&self.xlimits, sub_rng);
                let xsamples = x_data.to_owned();
                let multistarter = MiddlePickerMultiStarter::new(&self.xlimits, &xsamples, sub_rng);

                let infill_optpb = InfillOptProblem::new(
                    obj_model.as_ref(),
                    cstr_models,
                    &cstr_funcs,
                    cstr_tol,
                    viability_model,
                    self.config.feasibility_infill.alpha(),
                    &infill_data,
                    actives,
                );

                let (infill_obj, xk) = self.optimize_infill_criterion(
                    infill_optpb,
                    multistarter,
                    (xbest, ybest, cbest),
                );
                debug!("+++++++  xk = {xk}");

                match self.compute_virtual_point(&xk, y_data, obj_model.as_ref(), cstr_models) {
                    Ok(yk) => {
                        let yk = Array2::from_shape_vec((1, 1 + self.config.n_internal_cstr()), yk)
                            .unwrap();
                        y_dat = concatenate![Axis(0), y_dat, yk];

                        let yk_pen =
                            self.compute_penalized_point(&xk, obj_model.as_ref(), cstr_models);
                        let yk_pen = yk_pen.insert_axis(Axis(0));
                        y_penalized = concatenate![Axis(0), y_penalized, yk_pen];

                        let ck = cstr_funcs
                            .iter()
                            .map(|cstr| cstr(&xk.to_vec(), None, &mut infill_data))
                            .collect::<Vec<_>>();
                        c_dat = concatenate![
                            Axis(0),
                            c_dat,
                            Array2::from_shape_vec((1, cstr_funcs.len()), ck).unwrap()
                        ];

                        x_dat = concatenate![Axis(0), x_dat, xk.clone().insert_axis(Axis(0))];

                        // infill objective was minimized while infill criterion itself
                        // is expected to be maximized hence the negative sign here
                        infill_val = -infill_obj;
                    }
                    Err(err) => {
                        // Error while predict at best point: ignore
                        info!("Error while getting virtual point: {err}");
                        break;
                    }
                }
            }
            portfolio.push((x_dat.to_owned(), y_dat, c_dat, y_penalized, infill_val));
        }
        let (x_dat, y_dat, c_dat, y_penalized, infill_value) = if portfolio.len() > 1 {
            info!(
                "Portfolio : {:?}",
                portfolio.iter().map(|v| v.0[[0, 0]]).collect::<Vec<_>>()
            );
            // Use portfolio strategy: Pick one point from portfolio
            select_from_portfolio(portfolio)
        } else {
            // Fallback to default returning one or several points (in case of qEI strategy)
            portfolio.remove(0)
        };

        (x_dat, y_dat, c_dat, y_penalized, infill_value)
    }

    /// Retrain surrogates from scratch (the "slow path"), whatever clustering/theta
    /// optimization settings are given, and store the resulting models/inits.
    #[allow(clippy::too_many_arguments)]
    fn retrain_from_scratch(
        &self,
        clusterings: &[Option<Clustering>],
        theta_inits: &[Option<Array2<f64>>],
        models: &mut Vec<Box<dyn MixtureGpSurrogate>>,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        actives: &Array2<usize>,
        do_clustering: DataClustering,
        optimize_theta: ThetaOptimization,
    ) -> Vec<Option<Array2<f64>>> {
        let mapping = self
            .config
            .cstr_specs
            .as_ref()
            .map(|s| internal_cstr_mapping(s));

        #[cfg(feature = "persistent")]
        let (models_new, inits) = if let Some(ref mapping) = mapping {
            self.train_with_mapping(
                mapping,
                xt,
                yt,
                do_clustering,
                optimize_theta,
                clusterings,
                theta_inits,
                actives,
            )
        } else {
            self.train_all_columns(
                xt,
                yt,
                do_clustering,
                optimize_theta,
                clusterings,
                theta_inits,
                actives,
            )
        };

        #[cfg(not(feature = "persistent"))]
        let (models_new, inits) = self.train_all_columns(
            xt,
            yt,
            do_clustering,
            optimize_theta,
            clusterings,
            theta_inits,
            actives,
        );

        *models = models_new;
        inits.into_iter().map(Some).collect()
    }

    /// Check whether the point(s) about to be added trigger the z-score guard:
    /// for each output `k`, compare the new `y` value(s) to the current model
    /// prediction `y_pred +- sqrt(var_pred)`. A z-score
    /// `(y_new - y_pred).abs() / var_pred.sqrt()` greater than
    /// `ZSCORE_THETA_OPTIM_THRESHOLD` means the new point is a poor surprise
    /// for the current surrogate, so its hyperparameters can no longer be
    /// trusted as-is.
    fn zscore_exceeds_threshold(
        &self,
        models: &[Box<dyn MixtureGpSurrogate>],
        xt: &Array2<f64>,
        yt: &Array2<f64>,
    ) -> bool {
        for (k, model) in models.iter().enumerate() {
            let (xt_model, _yt_model) = model.training_data();
            let n_old = xt_model.nrows();
            let n_new = xt.nrows();
            if n_new <= n_old {
                continue;
            }
            let x_new = xt.slice(s![n_old..n_new, ..]);
            let y_new = yt.slice(s![n_old..n_new, k]);

            let (y_pred, var_pred) = match model.predict_valvar(&x_new.view()) {
                Ok(pv) => pv,
                Err(err) => {
                    info!("z-score check: prediction failed for model {k}: {err}, skipping");
                    continue;
                }
            };

            for i in 0..y_new.len() {
                let std_pred = var_pred[i].max(0.0).sqrt();
                let err = (y_new[i] - y_pred[i]).abs();
                let zscore = if std_pred > f64::EPSILON {
                    err / std_pred
                } else if err > f64::EPSILON {
                    f64::INFINITY
                } else {
                    0.0
                };
                if zscore > ZSCORE_THETA_OPTIM_THRESHOLD {
                    info!(
                        "New point z-score={zscore:.3} > {ZSCORE_THETA_OPTIM_THRESHOLD} for model {k}: triggering theta optimization"
                    );
                    return true;
                }
            }
        }
        false
    }

    /// Whether theta hyperparameters must be fully (re)optimized (as opposed to
    /// reused as-is / via the fast incremental-update path), given the current
    /// training set size and clustering state.
    ///
    /// True while the surrogate is still data-starved (`nb_points < 10 * dim`),
    /// whenever the clustering itself is (re)generated, or -- only relevant for
    /// q-points batches (`batch > 1`), where the incoming points i>0 within a
    /// batch are Kriging-believer virtual (predicted, not evaluated) values, for
    /// which the z-score check is not meaningful -- on the legacy periodic
    /// `optmod` override, kept as a fallback in that case. `point_index` is the
    /// global point index (`iter * batch [+ i]`) used for that periodic check.
    fn must_optimize_theta(
        &self,
        do_clustering: DataClustering,
        nb_points: usize,
        dim: usize,
        point_index: usize,
    ) -> bool {
        if self.config.gp.theta_tuning.is_fixed() {
            return false;
        }
        let periodic_optim = self.config.qei_config.batch > 1
            && self.config.qei_config.optmod > 1
            && point_index.is_multiple_of(self.config.qei_config.optmod);
        do_clustering == DataClustering::Regenerate
            || nb_points < (MIN_POINTS_DIM_FACTOR * dim).max(MIN_POINTS_THRESOLD)
            || periodic_optim
    }

    /// Whether `models` do not need any training given `nb_points` training
    /// points: clustering is kept and models are already trained on all of them.
    fn models_up_to_date(
        models: &[Box<dyn MixtureGpSurrogate>],
        do_clustering: DataClustering,
        nb_points: usize,
    ) -> bool {
        do_clustering == DataClustering::Fixed
            && !models.is_empty()
            && models
                .iter()
                .all(|m| m.training_data().0.nrows() == nb_points)
    }

    /// Bring `models` in line with the whole training data `(x_data, y_data)`
    /// (see `must_optimize_theta` for the full-retrain-vs-fast-path decision)
    /// then sync `clusterings`/`theta_inits` with the resulting models.
    ///
    /// With a fixed clustering, models already trained on all `x_data` rows are
    /// left untouched: retraining on the very same data is pointless.
    /// `point_index` is the global point index used by `must_optimize_theta`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn refresh_models(
        &self,
        clusterings: &mut [Option<Clustering>],
        theta_inits: &mut [Option<Array2<f64>>],
        models: &mut Vec<Box<dyn MixtureGpSurrogate>>,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        actives: &Array2<usize>,
        do_clustering: DataClustering,
        point_index: usize,
    ) {
        if Self::models_up_to_date(models, do_clustering, x_data.nrows()) {
            debug!(
                "Surrogates already up to date with {} points",
                x_data.nrows()
            );
            return;
        }
        let dim = self.config.gp.kpls_dim.unwrap_or(x_data.ncols());
        let optimize_theta: ThetaOptimization = self
            .must_optimize_theta(do_clustering, x_data.nrows(), dim, point_index)
            .into();
        info!(
            "Update surrogates with {} points... clustering={:?} optimize_theta={:?}",
            x_data.nrows(),
            do_clustering,
            optimize_theta
        );
        let inits = self.update_models(
            clusterings,
            theta_inits,
            models,
            x_data,
            y_data,
            actives,
            do_clustering,
            optimize_theta,
        );
        self.sync_clustering_and_theta_inits(clusterings, theta_inits, models, &inits);
    }

    /// Sync `clusterings`/`theta_inits` with the `models`/`inits` that just came
    /// out of `update_models`, so a future full retrain reuses the fitted
    /// clustering/theta as a warm start.
    fn sync_clustering_and_theta_inits(
        &self,
        clusterings: &mut [Option<Clustering>],
        theta_inits: &mut [Option<Array2<f64>>],
        models: &[Box<dyn MixtureGpSurrogate>],
        inits: &[Option<Array2<f64>>],
    ) {
        (0..=self.config.n_internal_cstr()).for_each(|k| {
            clusterings[k] = Some(models[k].to_clustering());
            if let Some(init) = inits.get(k).and_then(|i| i.as_ref()) {
                theta_inits[k] = Some(init.to_owned());
            }
        });
    }

    /// Update the surrogate models with new data.
    /// When clustering is fixed and theta optimization is disabled, attempts
    /// an incremental update of existing models before falling back to full retraining.
    /// Before performing this incremental "fast path" update, the new point(s) z-score
    /// is checked against the current models: a large z-score (see
    /// `ZSCORE_THETA_OPTIM_THRESHOLD`) means the surrogate hyperparameters are no
    /// longer trustworthy, so a full theta-optimized retraining is triggered instead.
    #[allow(clippy::too_many_arguments)]
    fn update_models(
        &self,
        clusterings: &mut [Option<Clustering>],
        theta_inits: &mut [Option<Array2<f64>>],
        models: &mut Vec<Box<dyn MixtureGpSurrogate>>,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        actives: &Array2<usize>,
        do_clustering: DataClustering,
        optimize_theta: ThetaOptimization,
    ) -> Vec<Option<Array2<f64>>> {
        if do_clustering == DataClustering::Regenerate
            || optimize_theta == ThetaOptimization::Enabled
        {
            // Slow path: retrain from scratch
            return self.retrain_from_scratch(
                clusterings,
                theta_inits,
                models,
                xt,
                yt,
                actives,
                do_clustering,
                optimize_theta,
            );
        }

        // Fast path candidate: incremental update.
        if !models.is_empty() {
            // Check the incoming point(s) against the current surrogate before
            // committing to the cheap incremental update: a large z-score means
            // the fast path is no longer trustworthy, so escalate to a full
            // theta-optimized retraining instead of adding the point as-is.
            if self.zscore_exceeds_threshold(models, xt, yt) {
                return self.retrain_from_scratch(
                    clusterings,
                    theta_inits,
                    models,
                    xt,
                    yt,
                    actives,
                    do_clustering,
                    ThetaOptimization::Enabled,
                );
            }

            let mut update_failed = false;
            let old_models: Vec<Box<dyn MixtureGpSurrogate>> = std::mem::take(models);
            let mut results = Vec::with_capacity(old_models.len());
            for (k, model) in old_models.into_iter().enumerate() {
                let (xt_model, _yt_model) = model.training_data();
                let n_old = xt_model.nrows();
                let n_new = xt.nrows();
                if n_new > n_old {
                    let x_new = xt.slice(s![n_old..n_new, ..]);
                    let y_new = yt.slice(s![n_old..n_new, k]);
                    match model.update(&x_new.view(), &y_new.view()) {
                        Ok(updated) => {
                            results.push(updated);
                        }
                        Err(err) => {
                            info!("Incremental update failed for model {k}: {err}, retraining...");
                            update_failed = true;
                            results.push(model);
                            break;
                        }
                    }
                } else {
                    results.push(model);
                }
            }
            if !update_failed {
                *models = results;
                return vec![None; models.len()];
            }
        }

        // Fall through to slow path: retrain from scratch
        self.retrain_from_scratch(
            clusterings,
            theta_inits,
            models,
            xt,
            yt,
            actives,
            do_clustering,
            optimize_theta,
        )
    }
}

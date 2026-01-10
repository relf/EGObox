use std::marker::PhantomData;

use crate::errors::{EgoError, Result};
use crate::find_best_result_index;
use crate::gpmix::mixint::{as_continuous_limits, to_discrete_space};
use crate::solver::solver_computations::MiddlePickerMultiStarter;
use crate::solver::solver_infill_optim::InfillOptProblem;
use crate::utils::{
    EGOBOX_LOG, EGOR_USE_GP_VAR_PORTFOLIO, find_best_result_index_from, is_feasible,
    select_from_portfolio, update_data,
};
use crate::{DEFAULT_CSTR_TOL, EgorSolver, MAX_POINT_ADDITION_RETRY, ValidEgorConfig};
use crate::{EgorState, types::*};

use argmin::argmin_error_closure;
use argmin::core::{CostFunction, Problem, State};

use egobox_doe::{Lhs, LhsKind};
use egobox_gp::ThetaTuning;
use env_logger::{Builder, Env};

use egobox_moe::{Clustering, MixtureGpSurrogate, NbClusters};
use log::{debug, info};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip, concatenate, s};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use serde::de::DeserializeOwned;

use super::coego::COEGO_IMPROVEMENT_CHECK;

impl<SB: SurrogateBuilder + DeserializeOwned, C: CstrFn> EgorSolver<SB, C> {
    /// Constructor of the optimization of the function `f` with specified random generator
    /// to get reproducibility.
    ///
    /// The function `f` should return an objective but also constraint values if any.
    /// Design space is specified by a list of types for input variables `x` of `f` (see [`XType`]).
    pub fn new(config: ValidEgorConfig) -> Self {
        let env = Env::new().filter_or(EGOBOX_LOG, "info");
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
        let mut rng = if let Some(seed) = self.config.seed {
            Xoshiro256Plus::seed_from_u64(seed)
        } else {
            Xoshiro256Plus::from_entropy()
        };
        let mut clusterings = vec![None; 1 + self.config.n_cstr];
        let mut theta_tunings = vec![None; 1 + self.config.n_cstr];
        let cstr_tol = self
            .config
            .cstr_tol
            .clone()
            .unwrap_or(Array1::from_elem(self.config.n_cstr, DEFAULT_CSTR_TOL));

        // TODO: Manage fonction constraints
        let fcstrs = Vec::<Cstr>::new();
        // TODO: c_data has to be passed as argument or better computed using fcstrs(x_data)
        let c_data = Array2::zeros((x_data.nrows(), 0));
        // TODO: Coego not implemented
        let activity = None;

        let best_index = find_best_result_index(y_data, &c_data, &cstr_tol);
        let feasibility = is_feasible(&y_data.row(best_index), &c_data.row(best_index), &cstr_tol);

        let (x_dat, _, _, _, _) = self.select_next_points(
            true,
            0,
            false, // done anyway
            &mut clusterings,
            &mut theta_tunings,
            activity,
            x_data,
            y_data,
            &c_data,
            &cstr_tol,
            best_index,
            &fcstrs,
            feasibility,
            &mut rng,
        );
        x_dat
    }
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
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
        make_clustering: bool,
        optimize_theta: bool,
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
        let mut best_likelihood = -f64::INFINITY;

        let dim = self.config.gp.kpls_dim.unwrap_or(xt.ncols());

        let mut best_theta_inits = if let Some(inits) = theta_inits {
            inits.to_owned()
        } else {
            // otherwise suppose one cluster but will not be used
            // as number of cluster determination is automated
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

        for (i, active) in actives.outer_iter().enumerate() {
            let gp = if make_clustering {
                /* init || recluster */
                match self.config.gp.n_clusters {
                    NbClusters::Auto { max: _ } => {
                        if self.config.coego.activated {
                            log::warn!("Automated clustering not available with CoEGO")
                        }
                    }
                    NbClusters::Fixed { nb: _ } => {
                        let theta_tunings = best_theta_inits
                            .outer_iter()
                            .map(|init| ThetaTuning::Partial {
                                init: init.to_owned(),
                                bounds: theta_bounds.to_owned(),
                                active: Self::strip(&active.to_vec(), init.len()),
                            })
                            .collect::<Vec<_>>();
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
                best_theta_inits = Array2::from_shape_vec(
                    (gp.experts().len(), gp.experts()[0].theta().len()),
                    gp.experts()
                        .iter()
                        .flat_map(|expert| expert.theta().to_vec())
                        .collect(),
                )
                .expect("Theta initialization failure");

                if i == 0 {
                    info!(
                        "... {} trained ({} / {})",
                        model_name,
                        gp.n_clusters(),
                        gp.recombination()
                    );
                }
                gp
            } else {
                let clustering = clustering.unwrap();

                let theta_tunings = if optimize_theta {
                    // set hyperparameters optimization
                    let mut inits = best_theta_inits
                        .outer_iter()
                        .map(|init| ThetaTuning::Full {
                            init: init.to_owned(),
                            bounds: theta_bounds.to_owned(),
                        })
                        .collect::<Vec<_>>();
                    if self.config.coego.activated {
                        self.set_partial_theta_tuning(&active.to_vec(), &mut inits);
                    }
                    if i == 0 && model_name == "Objective" {
                        info!("Objective model hyperparameters optim init >>> {inits:?}");
                    }
                    inits
                } else {
                    // just use previous hyperparameters
                    let inits = best_theta_inits
                        .outer_iter()
                        .map(|init| ThetaTuning::Fixed(init.to_owned()))
                        .collect::<Vec<_>>();
                    if i == 0 && model_name == "Objective" {
                        info!("Objective model hyperparameters reused >>> {inits:?}");
                    }
                    inits
                };

                builder.set_theta_tunings(&theta_tunings);

                builder
                    .train_on_clusters(xt.view(), yt.view(), clustering)
                    .expect("GP training failure")
            };

            // CoEGO only in mono cluster, update theta if better likelihood
            if self.config.coego.activated {
                if self.config.gp.n_clusters.is_mono() {
                    if COEGO_IMPROVEMENT_CHECK {
                        let likelihood = gp.experts()[0].likelihood();
                        // We update only if better likelihood
                        if likelihood > best_likelihood && model_name == "Objective" {
                            if i > 0 {
                                log::info!(
                                    "Partial likelihood optim c={i} has improved value={likelihood}"
                                );
                            };
                            best_likelihood = likelihood;
                            best_theta_inits = Array2::from_shape_vec(
                                (gp.experts().len(), gp.experts()[0].theta().len()),
                                gp.experts()
                                    .iter()
                                    .flat_map(|expert| expert.theta().to_vec())
                                    .collect(),
                            )
                            .expect("Theta initialization failure");
                        } else if model_name == "Objective" {
                            log::debug!(
                                "Partial likelihood optim c={i} has not improved value={likelihood}"
                            );
                        };
                    } else {
                        best_theta_inits = Array2::from_shape_vec(
                            (gp.experts().len(), gp.experts()[0].theta().len()),
                            gp.experts()
                                .iter()
                                .flat_map(|expert| expert.theta().to_vec())
                                .collect(),
                        )
                        .expect("Theta initialization failure");
                    }
                } else {
                    log::warn!(
                        "CoEGO theta update wrt likelihood not implemented in multi-cluster setting"
                    );
                }
            };
            model = Some(gp)
        }
        (model.expect("Surrogate model is trained"), best_theta_inits)
    }

    /// Refresh infill data used to optimize infill criterion
    pub fn refresh_infill_data<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    >(
        &self,
        problem: &mut Problem<O>,
        state: &mut EgorState<f64>,
        models: &[Box<dyn egobox_moe::MixtureGpSurrogate>],
    ) -> InfillObjData<f64> {
        let y_data = state.data.as_ref().unwrap().1.clone();
        let x_data = state.data.as_ref().unwrap().0.clone();
        let (obj_model, cstr_models) = models.split_first().unwrap();

        let fmin = y_data[[state.best_index.unwrap(), 0]];
        let xbest = x_data.row(state.best_index.unwrap()).to_vec();

        let pb = problem.take_problem().unwrap();
        let fcstrs = pb.fn_constraints();

        let mut rng = state.take_rng().unwrap();
        let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
        *state = state.clone().rng(rng.clone());
        let sampling = Lhs::new(&self.xlimits)
            .with_rng(sub_rng)
            .kind(LhsKind::Maximin);
        let cstr_tol = self
            .config
            .cstr_tol
            .clone()
            .unwrap_or(Array1::from_elem(self.config.n_cstr, DEFAULT_CSTR_TOL));
        let (scale_infill_obj, scale_cstr, scale_fcstr, scale_wb2) = self.compute_scaling(
            &sampling,
            obj_model.as_ref(),
            cstr_models,
            &cstr_tol,
            fcstrs,
            fmin,
            1., // FIXME: TREGO does not use sigma weighting portfolio
        );

        let all_scale_cstr = concatenate![Axis(0), scale_cstr, scale_fcstr];

        problem.problem = Some(pb);

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

    /// Regenerate surrogate models from current state
    /// This method supposes that clustering is done and thetas has to be optimized
    pub fn refresh_surrogates(
        &self,
        state: &EgorState<f64>,
    ) -> Vec<Box<dyn egobox_moe::MixtureGpSurrogate>> {
        info!(
            "Train surrogates with {} points...",
            &state.data.as_ref().unwrap().0.nrows()
        );

        let actives = state
            .activity
            .as_ref()
            .unwrap_or(&self.full_activity())
            .to_owned();

        (0..=self.config.n_cstr)
            .into_par_iter()
            .map(|k| {
                let name = if k == 0 {
                    "Objective".to_string()
                } else {
                    format!("Constraint[{k}]")
                };
                self.make_clustered_surrogate(
                    &name,
                    &state.data.as_ref().unwrap().0,
                    &state.data.as_ref().unwrap().1.slice(s![.., k]).to_owned(),
                    false,
                    true,
                    state.clusterings.as_ref().unwrap()[k].as_ref(),
                    state.theta_inits.as_ref().unwrap()[k].as_ref(),
                    &actives,
                )
                .0
            })
            .collect()
    }

    /// This function is the main EGO algorithm iteration:
    /// * Train surrogates
    /// * Find next promising location(s) of optimum
    /// * Update state: Evaluate true function, update doe and optimum
    #[allow(clippy::type_complexity)]
    pub fn ego_step<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
    ) -> Result<EgorState<f64>> {
        let mut new_state = state.clone();
        let mut clusterings = new_state
            .take_clusterings()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "EgorSolver: No clustering!"
            ))?;
        let mut theta_inits = new_state
            .take_theta_inits()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "EgorSolver: No theta inits!"
            ))?;
        let activity = new_state.take_activity();
        let mut rng = new_state
            .take_rng()
            .ok_or_else(argmin_error_closure!(PotentialBug, "EgorSolver: No rng!"))?;
        let (mut x_data, mut y_data, mut c_data) = new_state
            .take_data()
            .ok_or_else(argmin_error_closure!(PotentialBug, "EgorSolver: No data!"))?;

        let (try_add_count, rejected_count, _) = loop {
            let recluster = self.have_to_recluster(new_state.added, new_state.prev_added);
            if recluster {
                info!("Reclustering surrogates...");
            }

            let init = new_state.get_iter() == 0;
            let pb = problem.take_problem().unwrap();
            let fcstrs = pb.fn_constraints();

            let (x_dat, y_dat, c_dat, infill_value, infill_data) = self.select_next_points(
                init,
                state.get_iter(),
                recluster,
                &mut clusterings,
                &mut theta_inits,
                activity.as_ref(),
                &x_data,
                &y_data,
                &c_data,
                &state.cstr_tol,
                state.best_index.unwrap(),
                fcstrs,
                state.feasibility,
                &mut rng,
            );

            problem.problem = Some(pb);

            debug!("Try adding {x_dat}");
            let added_indices = update_data(
                &mut x_data,
                &mut y_data,
                &mut c_data,
                &x_dat,
                &y_dat,
                &c_dat,
            );

            new_state = new_state
                .clusterings(clusterings.clone())
                .theta_inits(theta_inits.clone())
                .data((x_data.clone(), y_data.clone(), c_data.clone()))
                .infill_value(infill_value)
                .rng(rng.clone())
                .param(x_dat.row(0).to_owned())
                .cost(y_dat.row(0).to_owned());
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

            let rejected_count = x_dat.nrows() - added_indices.len();
            for i in 0..x_dat.nrows() {
                let msg = format!(
                    "  {} {}",
                    if added_indices.contains(&i) { "A" } else { "R" },
                    x_dat.row(i)
                );
                if added_indices.contains(&i) {
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
                new_state.no_point_added_retries -= 1;
                if new_state.no_point_added_retries == 0 {
                    info!("Max number of retries ({}) without adding point", 3);
                    info!("Consider solver has converged");
                    return Err(EgoError::NoMorePointToAddError(Box::new(new_state)));
                }
            } else {
                // ok point added we can go on, just output number of rejected point
                break (x_dat.nrows(), rejected_count, infill_data);
            }
        };
        let add_count = (try_add_count - rejected_count) as i32;
        let x_to_eval = x_data.slice(s![-add_count.., ..]).to_owned();
        debug!(
            "Eval {} point{} {}",
            add_count,
            if add_count > 1 { "s" } else { "" },
            if new_state.no_point_added_retries == 0 {
                " from sampling"
            } else {
                ""
            }
        );

        new_state.prev_added = new_state.added;
        new_state.added += add_count as usize;
        info!("+{} point(s), total: {} points", add_count, new_state.added);
        new_state.no_point_added_retries = MAX_POINT_ADDITION_RETRY;
        let y_actual = self.eval_obj(problem, &x_to_eval);
        Zip::from(y_data.slice_mut(s![-add_count.., ..]).rows_mut())
            .and(y_actual.rows())
            .for_each(|mut y, val| y.assign(&val));
        let best_index = find_best_result_index_from(
            state.best_index.unwrap(),
            y_data.nrows() - add_count as usize,
            &y_data,
            &c_data,
            &new_state.cstr_tol,
        );
        new_state.prev_best_index = state.best_index;
        new_state.best_index = Some(best_index);
        new_state = new_state.data((x_data.clone(), y_data.clone(), c_data.clone()));
        new_state.feasibility = state.feasibility
            || is_feasible(
                &y_data.row(best_index),
                &c_data.row(best_index),
                &new_state.cstr_tol,
            );
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
        activity: Option<&Array2<usize>>,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        c_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        cstr_tol: &Array1<f64>,
        best_index: usize,
        cstr_funcs: &[impl CstrFn],
        feasibility: bool,
        rng: &mut Xoshiro256Plus,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        f64,
        InfillObjData<f64>,
    ) {
        let mut portfolio = vec![];

        let sigma_weights = if std::env::var(EGOR_USE_GP_VAR_PORTFOLIO).is_ok()
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
            let mut infill_val = f64::INFINITY;
            let mut infill_data = InfillObjData {
                feasibility,
                ..Default::default()
            };
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
                let actives = activity.unwrap_or(&self.full_activity()).to_owned();

                info!("Train surrogates with {} points...", xt.nrows());
                let models_and_inits = (0..=self.config.n_cstr).into_par_iter().map(|k| {
                    let name = if k == 0 {
                        "Objective".to_string()
                    } else {
                        format!("Constraint[{k}]")
                    };
                    let make_clustering = (init && i == 0) || recluster;
                    let optimize_theta = (iter as usize * self.config.qei_config.batch + i)
                        .is_multiple_of(self.config.qei_config.optmod)
                        && j == 0;
                    self.make_clustered_surrogate(
                        &name,
                        &xt,
                        &yt.slice(s![.., k]).to_owned(),
                        make_clustering,
                        optimize_theta,
                        clusterings[k].as_ref(),
                        theta_inits[k].as_ref(),
                        &actives,
                    )
                });
                let (models, inits): (Vec<_>, Vec<_>) = models_and_inits.unzip();
                #[cfg(feature = "persistent")]
                if std::env::var(crate::EGOR_USE_GP_RECORDER).is_ok() {
                    use crate::utils::{EGOR_GP_FILENAME, EGOR_INITIAL_GP_FILENAME, gp_recorder};

                    let default_dir = String::from("./");
                    let outdir = self.config.outdir.as_ref().unwrap_or(&default_dir);
                    let filename = if iter == 0 {
                        EGOR_INITIAL_GP_FILENAME
                    } else {
                        EGOR_GP_FILENAME
                    };
                    let filepath = std::path::Path::new(outdir).join(filename);
                    match gp_recorder::save_gp_models(&filepath, &models) {
                        Ok(_) => log::info!("GP models saved to {:?}", filepath),
                        Err(err) => log::info!("Cannot save GP models: {:?}", err),
                    };
                }

                (0..=self.config.n_cstr).for_each(|k| {
                    clusterings[k] = Some(models[k].to_clustering());
                    theta_inits[k] = Some(inits[k].to_owned());
                });

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

                let (scale_infill_obj, scale_cstr, scale_fcstr, scale_wb2) = self.compute_scaling(
                    &sampling,
                    obj_model.as_ref(),
                    cstr_models,
                    cstr_tol,
                    cstr_funcs,
                    fmin,
                    *sigma_weight,
                );

                let all_scale_cstr = concatenate![Axis(0), scale_cstr, scale_fcstr];

                infill_data = InfillObjData {
                    fmin,
                    xbest: xbest.to_vec(),
                    scale_infill_obj,
                    scale_cstr: Some(all_scale_cstr.to_owned()),
                    scale_wb2,
                    feasibility,
                    sigma_weight: *sigma_weight,
                };

                let cstr_funcs = cstr_funcs
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
                            cstr(x, gradient, params) / scale_fc
                        }
                    })
                    .collect::<Vec<_>>();

                let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
                // let multistarter = GlobalMultiStarter::new(&self.xlimits, sub_rng);
                let xsamples = x_data.to_owned();
                let multistarter = MiddlePickerMultiStarter::new(&self.xlimits, &xsamples, sub_rng);

                let infill_optpb = InfillOptProblem {
                    obj_model: obj_model.as_ref(),
                    cstr_models,
                    cstr_funcs: &cstr_funcs,
                    cstr_tols: cstr_tol,
                    infill_data: &infill_data,
                    actives: &actives,
                };

                let (infill_obj, xk) = self.optimize_infill_criterion(
                    infill_optpb,
                    multistarter,
                    (xbest, ybest, cbest),
                );
                debug!("+++++++  xk = {xk}");

                match self.compute_virtual_point(&xk, y_data, obj_model.as_ref(), cstr_models) {
                    Ok(yk) => {
                        let yk = Array2::from_shape_vec((1, 1 + self.config.n_cstr), yk).unwrap();
                        y_dat = concatenate![Axis(0), y_dat, yk];

                        let ck = cstr_funcs
                            .iter()
                            .map(|cstr| cstr(&xk.to_vec(), None, &mut infill_data))
                            .collect::<Vec<_>>();
                        c_dat = concatenate![
                            Axis(0),
                            c_dat,
                            Array2::from_shape_vec((1, cstr_funcs.len()), ck).unwrap()
                        ];

                        x_dat = concatenate![Axis(0), x_dat, xk.insert_axis(Axis(0))];

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
            portfolio.push((x_dat.to_owned(), y_dat, c_dat, infill_val, infill_data));
        }
        let (x_dat, y_dat, c_dat, infill_value, infill_data) = if portfolio.len() > 1 {
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

        (x_dat, y_dat, c_dat, infill_value, infill_data)
    }
}

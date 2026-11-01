use std::marker::PhantomData;

use crate::errors::{EgoError, Result};
use crate::moo::scalarization::compromise_index;
use crate::solver::coego;
use crate::solver::point_selection::SelectionInputs;
use crate::solver::surrogate_set::{DataClustering, SurrogateSet, ThetaOptimization, TrainSpec};
use crate::utils::{find_best_result_index_from, is_feasible_at, usable_data};
use crate::{ActivityStrategy, FullActivity, find_best_result_index};
use crate::{EgorSolver, ValidEgorConfig};
use crate::{EgorState, types::*};
use egobox_moe::as_continuous_limits;

use basin::{CostFunction, Problem};

use egobox_doe::{Lhs, LhsKind};
use egobox_gp::{ParamTuning, ThetaTuning};

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
        crate::utils::logging::init_logger(None);
        let xtypes = config.xtypes.clone();
        let mut surrogate_builder = SB::new_with_xtypes(&xtypes);
        // Surrogate training randomness (e.g. clustering) follows the optimizer seed
        if let Some(seed) = config.seed {
            surrogate_builder.set_rng(Xoshiro256Plus::seed_from_u64(seed));
        }
        EgorSolver {
            config,
            xlimits: as_continuous_limits(&xtypes),
            surrogate_builder,
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
        let (y_data_owned, cstr_tol) = self.service_data(y_data);
        let y_data = &y_data_owned;
        let mut rng = self.config.rng();
        let mut surrogates = SurrogateSet::new(self.config.n_surrogates());

        // TODO: Manage fonction constraints
        let fcstrs = Vec::<Cstr>::new();
        let fcstr_specs = None;
        // TODO: c_data has to be passed as argument or better computed using fcstrs(x_data)
        let c_data = Array2::zeros((x_data.nrows(), 0));
        // TODO: Coego not implemented
        let activity = FullActivity.generate_activity(x_data.ncols(), &mut rng);

        // With ParEGO, surrogates are trained on the scalarized view [s | cstrs] of the data
        // (see `ego_step`). Weights change from one call to the next as the number of data
        // points grows.
        let n_obj = self.config.n_obj();
        let y_view;
        let y_data: &ArrayBase<_, Ix2> = if self.config.is_scalarized() {
            let weights =
                crate::moo::parego::shuffled_weights(n_obj, self.config.moo.n_divisions, &mut rng);
            let weights = &weights[x_data.nrows() % weights.len()];
            y_view =
                crate::moo::parego::scalarized_view(y_data, n_obj, weights, self.config.moo.rho);
            &y_view
        } else {
            y_data
        };

        // With noisy values, models are trained first to select the best point from their
        // estimates, then reused as is to select next points
        let estimated_best = if self.has_noisy_values() {
            self.train_and_estimate_best(
                &mut surrogates,
                &activity,
                &x_data.to_owned(),
                y_data,
                &c_data,
                &cstr_tol,
            )
        } else {
            None
        };
        let (best_index, feasibility) = estimated_best.unwrap_or_else(|| {
            let best_index = if self.config.is_per_objective() {
                compromise_index(y_data, &c_data, n_obj, &cstr_tol, &[]).unwrap_or(0)
            } else {
                find_best_result_index(y_data, &c_data, &cstr_tol)
            };
            let n_train_obj = if self.config.is_per_objective() {
                n_obj
            } else {
                1
            };
            let feasibility = is_feasible_at(
                &y_data.row(best_index),
                &c_data.row(best_index),
                n_train_obj,
                &cstr_tol,
            );
            (best_index, feasibility)
        });

        let inputs = SelectionInputs {
            init: estimated_best.is_none(), // models already trained otherwise
            iter: 0,
            recluster: false, // done anyway
            activity: &activity,
            x_data: x_data.view(),
            y_data: y_data.view(),
            c_data: c_data.view(),
            x_fail: None,
            cstr_tol: &cstr_tol,
            best_index,
            feasibility,
            fcstrs: &fcstrs,
            fcstr_specs,
        };
        self.select_candidates(&inputs, &mut surrogates, &mut rng).x
    }

    /// Given an evaluated doe (x, y) data, return the index of the best evaluated point,
    /// as reported by the optimizer: the minimum wrt constraints, or the compromise point of
    /// the Pareto front with several objectives.
    /// When GP models a noise on function values (mono-objective), the best point is selected
    /// from the surrogate models posterior means, which requires training the models.
    pub fn best_index(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> usize {
        let (y_data, cstr_tol) = self.service_data(y_data);
        let c_data = Array2::zeros((x_data.nrows(), 0));
        if self.config.n_obj() == 1 && self.has_noisy_values() {
            let mut rng = self.config.rng();
            let activity = FullActivity.generate_activity(x_data.ncols(), &mut rng);
            if let Some((best_index, _)) = self.train_and_estimate_best(
                &mut SurrogateSet::new(self.config.n_surrogates()),
                &activity,
                &x_data.to_owned(),
                &y_data,
                &c_data,
                &cstr_tol,
            ) {
                return best_index;
            }
        }
        self.observed_best_index(&y_data, &c_data, &cstr_tol)
    }

    /// Index of the best point from observed values: the minimum wrt constraints, or the
    /// compromise point of the Pareto front with several objectives
    pub(crate) fn observed_best_index(
        &self,
        y_data: &Array2<f64>,
        c_data: &Array2<f64>,
        cstr_tol: &Array1<f64>,
    ) -> usize {
        let n_obj = self.config.n_obj();
        if n_obj > 1 {
            compromise_index(y_data, c_data, n_obj, cstr_tol, &[]).unwrap_or(0)
        } else {
            find_best_result_index(y_data, c_data, cstr_tol)
        }
    }

    /// Data given to the ask-and-tell interface: output values with constraints transformed
    /// with `cstr_specs` if set, and internal constraints tolerances
    fn service_data(
        &self,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> (Array2<f64>, Array1<f64>) {
        let y_data = self.config.to_internal_outputs(y_data.to_owned());
        let cstr_tol = self.config.cstr_tol_padded(self.config.n_internal_cstr());
        (y_data, cstr_tol)
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

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
    C: CstrFn,
{
    /// Name of the k-th surrogate model used in logs
    fn model_name(&self, k: usize) -> String {
        let n_obj_models = self.config.n_obj_models();
        if k >= n_obj_models {
            format!("Constraint[{}]", k - n_obj_models + 1)
        } else if n_obj_models == 1 {
            "Objective".to_string()
        } else {
            format!("Objective[{}]", k + 1)
        }
    }

    /// Whether we have to recluster the data
    pub fn have_to_recluster(&self, added: usize, prev_added: usize) -> bool {
        self.config.gp.n_clusters.is_auto()
            && (added != 0 && added.is_multiple_of(10) && added - prev_added > 0)
    }

    /// Copy of the surrogate builder configured with the GP settings,
    /// the given number of clusters and recombination
    fn configured_builder(
        &self,
        n_clusters: NbClusters,
        recombination: egobox_moe::Recombination<f64>,
    ) -> SB {
        let mut builder = self.surrogate_builder.clone();
        builder.set_kpls_dim(self.config.gp.kpls_dim);
        builder.set_regression_spec(self.config.gp.regression_spec);
        builder.set_correlation_spec(self.config.gp.correlation_spec);
        builder.set_n_clusters(n_clusters);
        builder.set_recombination(recombination);
        builder.set_optim_params(self.config.gp.n_start, self.config.gp.max_eval);
        builder.set_noise_ratio(self.config.gp.noise_ratio.clone());
        builder
    }

    /// Train a single-cluster objective surrogate on the given (local) training data
    /// with the configured GP settings and hyperparameters optimization.
    /// Returns `None` when the training fails.
    pub(crate) fn train_local_obj_model(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    ) -> Option<Box<dyn MixtureGpSurrogate>> {
        let mut builder =
            self.configured_builder(NbClusters::Fixed { nb: 1 }, egobox_moe::Recombination::Hard);
        let dim = self.config.gp.kpls_dim.unwrap_or(xt.ncols());
        let theta_tuning = if self.config.gp.theta_tuning.is_fixed() {
            self.config.gp.theta_tuning.clone()
        } else {
            ThetaTuning::Full {
                init: self.config.gp.theta_tuning.init().to_owned(),
                bounds: crate::utils::theta_bounds(
                    &self.config.gp.theta_tuning,
                    dim,
                    self.config.gp.correlation_spec,
                ),
            }
        };
        builder.set_theta_tunings(&[theta_tuning]);
        match builder.train(xt.view(), yt.view()) {
            Ok(model) => Some(model),
            Err(err) => {
                log::warn!("Local GP training failure, global model used instead: {err:?}");
                None
            }
        }
    }

    /// Copy of this solver optimizing the objective posterior mean instead of the
    /// configured infill criterion. Probability of feasibility and feasibility enhanced
    /// infill criteria are not used with a mean: surrogated constraints are handled as
    /// optimizer constraints.
    pub(crate) fn with_mean_criterion(&self) -> Self {
        let mut solver = self.clone();
        solver.config.infill_criterion = Box::new(crate::criteria::PosteriorMean);
        solver.config.cstr_infill = false;
        solver.config.feasibility_infill = FeasibleInfillStrategy::None;
        solver
    }

    /// Build surrogate given training data and surrogate builder
    /// Reclustering is triggered when make_clustering boolean is true otherwise
    /// previous clustering is used. theta_init allows to reuse
    /// previous theta without fully retraining the surrogates
    /// (faster execution at the cost of surrogate quality)
    fn make_clustered_surrogate(
        &self,
        model_name: &str,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        spec: TrainSpec,
        clustering: Option<&Clustering>,
        theta_inits: Option<&Array2<f64>>,
    ) -> (Box<dyn MixtureGpSurrogate>, Array2<f64>) {
        let mut builder = self.configured_builder(
            self.config.gp.n_clusters.clone(),
            self.config.gp.recombination,
        );
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
            FullActivity.activity(spec.actives.len())
        } else {
            // Otherwise, use activity strategy to determine active variables for theta optimization
            spec.actives.to_owned()
        };

        for (i, active) in actives.outer_iter().enumerate() {
            let gp = match spec.clustering {
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
                            let theta_tunings = match spec.theta {
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
                                                active: coego::strip(&active.to_vec(), init.len()),
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

                    let theta_tunings = match spec.theta {
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

            if self.config.activity_strategy.is_cooperative()
                && !self.config.gp.n_clusters.is_mono()
            {
                log::warn!(
                    "Cooperative activity theta update wrt likelihood not implemented in multi-cluster setting"
                );
            }
            model = Some(gp)
        }
        (model.expect("Surrogate model is trained"), best_theta_inits)
    }

    /// Viability surrogate trained on `x_train` (safe points) and `x_fail` (failed points),
    /// when the failsafe strategy or the feasibility infill strategy requires it
    /// and there are failed points
    pub(crate) fn viability_model(
        &self,
        x_train: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        x_fail: Option<&Array2<f64>>,
    ) -> Option<Box<dyn MixtureGpSurrogate>> {
        let required = self.config.failsafe_strategy == FailsafeStrategy::Viability
            || self.config.feasibility_infill.is_enabled();
        match x_fail {
            Some(x_fail) if required && x_fail.nrows() > 0 => {
                info!(
                    "Build viability surrogate with {} safe and {} failed points...",
                    x_train.nrows(),
                    x_fail.nrows()
                );
                Some(self.make_viability_surrogate(x_train, x_fail))
            }
            _ => None,
        }
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

    /// Whether GP models a noise on function values, in which case values are estimated
    /// by surrogate models posterior means (mono-objective optimization only)
    pub(crate) fn has_noisy_values(&self) -> bool {
        self.config.n_obj() == 1 && self.config.gp.noise_ratio != ParamTuning::Fixed(0.)
    }

    /// Value of the incumbent (best point) used by infill criteria: the observed objective value
    /// or its estimate by the objective model posterior mean when values are noisy
    pub(crate) fn incumbent_value(
        &self,
        obj_model: &dyn MixtureGpSurrogate,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        best_index: usize,
    ) -> f64 {
        let observed = y_data[[best_index, 0]];
        if self.has_noisy_values() {
            obj_model
                .predict(&x_data.slice(s![best_index..best_index + 1, ..]))
                .map_or(observed, |mu| mu[0])
        } else {
            observed
        }
    }

    /// When values are noisy, estimate objective and constraints values at data points
    /// with the surrogate models posterior means, and select the best point from these
    /// estimates (see [`crate::SurrogateState::y_estimate`]).
    /// Expects persisted models to be trained on the whole data, otherwise estimates are dropped.
    pub(crate) fn refresh_estimates(&self, state: &mut EgorState<f64>) {
        state.surrogate.y_estimate = None;
        if !self.has_noisy_values() {
            return;
        }
        let Some((x_data, y_data, c_data)) = state.surrogate.data.as_ref() else {
            return;
        };
        // Failed points with imputed values are never candidates for the best
        let excluded = self
            .config
            .imputed_rows(x_data, state.surrogate.x_fail.as_ref());
        let Some(y_est) = Self::estimate_values(&state.surrogate.models, x_data, y_data, &excluded)
        else {
            return;
        };

        let best_index = find_best_result_index(&y_est, c_data, &state.doe.cstr_tol);
        debug!(
            "Estimated best fun(x[{best_index}])={} (observed {})",
            y_est.row(best_index),
            y_data.row(best_index)
        );
        state.surrogate.best_index = Some(best_index);
        state.surrogate.y_estimate = Some(y_est);
    }

    /// Estimate objective and constraints values at data points with the posterior means
    /// of the surrogate `models`. Rows which are not candidates for the best (`excluded` rows,
    /// rows with non finite values) are NaN.
    /// Returns `None` when models are not trained on the whole data or a prediction fails.
    fn estimate_values(
        models: &[Box<dyn MixtureGpSurrogate>],
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        excluded: &[usize],
    ) -> Option<Array2<f64>> {
        if models.len() != y_data.ncols()
            || models
                .iter()
                .any(|m| m.training_data().0.nrows() != x_data.nrows())
        {
            return None;
        }

        let mut y_est = Array2::zeros(y_data.dim());
        for (mut col, model) in y_est.columns_mut().into_iter().zip(models) {
            match model.predict(&x_data.view()) {
                Ok(mu) => col.assign(&mu),
                Err(err) => {
                    log::warn!("Cannot estimate function values: {err:?}");
                    return None;
                }
            }
        }
        for (i, (mut row, y)) in y_est.rows_mut().into_iter().zip(y_data.rows()).enumerate() {
            if excluded.contains(&i) || y.iter().any(|v| !v.is_finite()) {
                row.fill(f64::NAN);
            }
        }
        Some(y_est)
    }

    /// Train surrogate `models` on the whole data as done for a first iteration, then return
    /// the index of the best point and its feasibility from the models estimated values.
    /// Returns `None` when values cannot be estimated.
    fn train_and_estimate_best(
        &self,
        surrogates: &mut SurrogateSet,
        activity: &Array2<usize>,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        c_data: &Array2<f64>,
        cstr_tol: &Array1<f64>,
    ) -> Option<(usize, bool)> {
        self.refresh_models(
            surrogates,
            x_data,
            y_data,
            activity,
            DataClustering::Regenerate,
            0,
        );
        let y_est = Self::estimate_values(&surrogates.models, x_data, y_data, &[])?;
        let best_index = find_best_result_index(&y_est, c_data, cstr_tol);
        let feasibility =
            is_feasible_at(&y_est.row(best_index), &c_data.row(best_index), 1, cstr_tol);
        debug!(
            "Estimated best fun(x[{best_index}])={} (observed {})",
            y_est.row(best_index),
            y_data.row(best_index)
        );
        Some((best_index, feasibility))
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

        let fmin = self.incumbent_value(
            obj_model.as_ref(),
            &x_data,
            &y_data,
            state.surrogate.best_index.unwrap(),
        );
        let xbest = x_data.row(state.surrogate.best_index.unwrap()).to_vec();

        let pb = problem.inner();
        let fcstrs = pb.constraints();
        let fcstr_specs = pb.constraint_specs();

        let transformed_fcstrs = crate::types::affine_fcstrs(fcstrs, fcstr_specs);

        let mut rng = state.take_rng().unwrap();
        let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
        *state = state.clone().rng(rng.clone());
        let sampling = Lhs::new(&self.xlimits)
            .with_rng(sub_rng)
            .kind(LhsKind::Maximin);
        let cstr_tol = self.config.cstr_tol_padded(self.config.n_internal_cstr());
        let (scale_infill_obj, scale_cstr, scale_fcstr, scale_wb2) = self.compute_scaling(
            &sampling,
            obj_model.as_ref(),
            cstr_models,
            &cstr_tol,
            transformed_fcstrs.len(),
            fmin,
            1., // FIXME: TREGO does not use sigma weighting portfolio
        );

        // Same layout as in global steps: surrogate constraints, function constraints
        // then probability of viability constraint (normalized)
        let scale_pov_cstr = Array1::ones((1,));
        let all_scale_cstr = concatenate![Axis(0), scale_cstr, scale_fcstr, scale_pov_cstr];

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
    fn train_all_columns(
        &self,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        spec: TrainSpec,
        surrogates: &SurrogateSet,
    ) -> (Vec<Box<dyn MixtureGpSurrogate>>, Vec<Array2<f64>>) {
        let models_and_inits = (0..self.config.n_surrogates()).into_par_iter().map(|k| {
            self.make_clustered_surrogate(
                &self.model_name(k),
                xt,
                &yt.slice(s![.., k]).to_owned(),
                spec,
                surrogates.clusterings[k].as_ref(),
                surrogates.theta_inits[k].as_ref(),
            )
        });
        models_and_inits.unzip()
    }

    /// Train only primary constraint columns and derive transformed ones
    /// via [`AffinedSurrogate`] wrappers.
    fn train_with_mapping(
        &self,
        mapping: &[InternalCstrKind],
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        spec: TrainSpec,
        surrogates: &SurrogateSet,
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
                    let (model, inits) = self.make_clustered_surrogate(
                        &self.model_name(k),
                        xt,
                        &yt.slice(s![.., k]).to_owned(),
                        spec,
                        surrogates.clusterings[k].as_ref(),
                        surrogates.theta_inits[k].as_ref(),
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
        let mut surrogates = SurrogateSet::take(&mut new_state)?;

        let mut rng = new_state
            .take_rng()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No rng!".to_string()))?;
        let (mut x_data, mut y_data, mut c_data) = new_state
            .take_data()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No data!".to_string()))?;

        // Computed once and reused both for the batch (virtual points) search below
        // and, further down, to incorporate the actually evaluated point(s) into
        // the persisted models (neither `new_state.doe.{added,prev_added}` nor
        // `new_state.get_iter()` change until after this whole retry loop).
        let recluster = self.have_to_recluster(new_state.doe.added, new_state.doe.prev_added);
        if recluster {
            info!("Reclustering surrogates...");
        }
        let init = new_state.get_iter() == 0;

        let n_obj = self.config.n_obj();
        // Multi-objective (ParEGO): weight vectors tried in a random order, a distinct one at each try
        let parego_weights = if self.config.is_scalarized() {
            crate::moo::parego::shuffled_weights(n_obj, self.config.moo.n_divisions, &mut rng)
        } else {
            vec![]
        };
        let mut n_tries = 0;

        let accepted = loop {
            // Multi-objective (ParEGO): surrogates are trained on a view of the data where the
            // objectives are scalarized with the weights of the try. As their training targets
            // change, the surrogates are retrained (never incrementally updated).
            let (scalarized_y, train_best_index) = if self.config.is_scalarized() {
                let weights = &parego_weights[n_tries % parego_weights.len()];
                n_tries += 1;
                info!("ParEGO weights = {weights}");
                let view = crate::moo::parego::scalarized_view(
                    &y_data,
                    n_obj,
                    weights,
                    self.config.moo.rho,
                );
                let best = find_best_result_index(&view, &c_data, &state.doe.cstr_tol);
                surrogates.models.clear();
                (Some(view), best)
            } else {
                (None, state.surrogate.best_index.unwrap())
            };
            let y_train = scalarized_y.as_ref().unwrap_or(&y_data);

            let pb = problem.inner();
            let fcstrs = pb.constraints();
            let fcstr_specs = pb.constraint_specs();

            // Batch (q-points > 1) selection picks points i=1..batch-1 using
            // Kriging-believer virtual y-values (predicted, not evaluated), which
            // must never pollute the persisted models (only the actually
            // evaluated point(s) get incorporated into them, further below):
            // they are restored after the search.
            // With batch == 1 there is only ever i == 0 (no virtual point is ever
            // constructed), so it's safe -- and avoids a needless clone -- to
            // search with the persisted models directly.
            let persisted_models =
                (self.config.qei_config.batch > 1).then(|| surrogates.models.clone());

            let inputs = SelectionInputs {
                init,
                iter: state.get_iter(),
                recluster,
                activity: &state.coego.activity,
                x_data: x_data.view(),
                y_data: y_train.view(),
                c_data: c_data.view(),
                x_fail: state.surrogate.x_fail.as_ref(),
                cstr_tol: &state.doe.cstr_tol,
                best_index: train_best_index,
                feasibility: state.feasibility,
                fcstrs,
                fcstr_specs,
            };
            let candidates = self.select_candidates(&inputs, &mut surrogates, &mut rng);

            if let Some(models) = persisted_models {
                surrogates.models = models;
            }
            debug!("Try adding {}", candidates.x);
            let usable_indices = usable_data(&x_data, &candidates.x);

            new_state = new_state
                .clusterings(surrogates.clusterings.clone())
                .theta_inits(surrogates.theta_inits.clone())
                .data((x_data.clone(), y_data.clone(), c_data.clone()))
                .infill_value(candidates.infill_value)
                .rng(rng.clone());

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

            Self::log_rejections(&candidates.x, &usable_indices);
            if usable_indices.is_empty() {
                new_state.doe.no_point_added_retries -= 1;
                if new_state.doe.no_point_added_retries == 0 {
                    info!(
                        "Max number of retries ({}) without adding point",
                        self.config.max_point_addition_retries()
                    );
                    info!("Consider solver has converged");
                    return Err(EgoError::NoMorePointToAddError(Box::new(new_state)));
                }
            } else {
                // ok point added we can go on
                break candidates.select(&usable_indices);
            }
        };

        // Penalized values are used in case of failed evaluation
        let evaluation = self.evaluate_and_append(
            problem,
            &accepted.x,
            Some(&accepted.y_penalized),
            (&mut x_data, &mut y_data, &mut c_data),
        )?;
        // Current param and cost are those of the first evaluated point (internal layout)
        new_state = new_state
            .param(accepted.x.row(0).to_owned())
            .cost(evaluation.y.row(0).to_owned());
        let (add_count, valid_count, x_fail_points) = (
            evaluation.add_count,
            evaluation.valid_count,
            evaluation.x_fail,
        );

        let new_fails = self.new_failed_points(
            state.get_iter(),
            x_fail_points,
            new_state.surrogate.x_fail.as_ref(),
        );
        new_state = new_state
            .store_failed_points(new_fails)
            .count_added_points(add_count);
        info!(
            "+{} point(s), total: {} points",
            add_count, new_state.doe.added
        );
        new_state.doe.no_point_added_retries = self.config.max_point_addition_retries();

        let best_index = self.updated_best_index(
            &new_state,
            state.surrogate.best_index.unwrap(),
            (&x_data, &y_data, &c_data),
            valid_count,
        );
        new_state =
            new_state
                .best_index(best_index)
                .data((x_data.clone(), y_data.clone(), c_data.clone()));
        new_state.feasibility = state.feasibility
            || is_feasible_at(
                &y_data.row(best_index),
                &c_data.row(best_index),
                n_obj,
                &new_state.doe.cstr_tol,
            );
        if self.hypervolume_converged(state.get_iter(), &new_state, (&x_data, &y_data, &c_data)) {
            new_state = new_state.terminate_with(crate::TerminationReason::SolverConverged);
        }
        // With ParEGO, surrogates are retrained on the next scalarized view at next iteration
        if !self.config.is_scalarized() {
            // Incorporate the actually evaluated point(s) into the persisted
            // models (never the virtual/Kriging-believer values used only for
            // the batch search above, see `persisted_models`): see
            // `must_optimize_theta` for the full-retrain-vs-fast-path decision.
            let do_clustering: DataClustering = (init || recluster).into();
            let point_index = state.get_iter() as usize * self.config.qei_config.batch;
            self.refresh_models(
                &mut surrogates,
                &x_data,
                &y_data,
                &state.coego.activity,
                do_clustering,
                point_index,
            );
        }
        Ok(surrogates.put(new_state))
    }

    /// Log accepted (`usable_indices`) and rejected points among selected ones `x_dat`
    fn log_rejections(x_dat: &Array2<f64>, usable_indices: &[usize]) {
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
    }

    /// Failed points to store given failed points among evaluated ones `x_fail_points`.
    /// In first iteration with imputation, failed points of the initial DOE `x_fail_doe`,
    /// evaluated again with the selected points, are already stored: only new ones are kept.
    fn new_failed_points(
        &self,
        iter: u64,
        x_fail_points: Option<Array2<f64>>,
        x_fail_doe: Option<&Array2<f64>>,
    ) -> Option<Array2<f64>> {
        match (x_fail_points, x_fail_doe) {
            (Some(xfail), Some(xfail_doe))
                if iter == 0 && self.config.failsafe_strategy == FailsafeStrategy::Imputation =>
            {
                info!("Initial DOE had {} failed point(s)", xfail_doe.nrows());
                info!("Total failed points after eval: {}", xfail.nrows());
                info!("{} new failed point(s)", xfail.nrows() - xfail_doe.nrows());
                Some(xfail.slice(s![xfail_doe.nrows().., ..]).to_owned())
            }
            (x_fail_points, _) => x_fail_points,
        }
    }

    /// Index of the best point once the last `n_valid` evaluated points are appended to the
    /// data, `prev_best` being the index of the best point before.
    fn updated_best_index(
        &self,
        new_state: &EgorState<f64>,
        prev_best: usize,
        (x_data, y_data, c_data): (&Array2<f64>, &Array2<f64>, &Array2<f64>),
        n_valid: usize,
    ) -> usize {
        let n_obj = self.config.n_obj();
        if n_obj > 1 {
            // Compromise point of the Pareto front (recomputed as normalization changes)
            // failed points with imputed values are never the compromise point
            let excluded = self
                .config
                .imputed_rows(x_data, new_state.surrogate.x_fail.as_ref());
            compromise_index(y_data, c_data, n_obj, &new_state.doe.cstr_tol, &excluded)
                .unwrap_or(prev_best)
        } else {
            // Only actually evaluated points (appended last) are candidates for the best,
            // failed points with imputed values are not.
            find_best_result_index_from(
                prev_best,
                y_data.nrows() - n_valid,
                y_data,
                c_data,
                &new_state.doe.cstr_tol,
            )
        }
    }

    /// Multi-objective optional stop on hypervolume progress over the last iterations
    fn hypervolume_converged(
        &self,
        iter: u64,
        new_state: &EgorState<f64>,
        (x_data, y_data, c_data): (&Array2<f64>, &Array2<f64>, &Array2<f64>),
    ) -> bool {
        let n_obj = self.config.n_obj();
        if n_obj > 1
            && let Some((tol, n_iters)) = self.config.moo.hv_stop
            && iter as usize + 1 >= n_iters
        {
            // Each iteration adds at most `batch` points: when points are rejected or fail,
            // this window covers more iterations, which only delays the stop
            // (iteration boundaries are not kept in the state)
            let n_prev_rows = y_data
                .nrows()
                .saturating_sub(n_iters * self.config.qei_config.batch)
                .max(1);
            let (hv_prev, hv_now) = crate::moo::hypervolume::hypervolume_progress(
                y_data,
                c_data,
                n_obj,
                &new_state.doe.cstr_tol,
                n_prev_rows,
                &self
                    .config
                    .imputed_rows(x_data, new_state.surrogate.x_fail.as_ref()),
            );
            info!("Hypervolume over the last {n_iters} iteration(s): {hv_prev} -> {hv_now}");
            if hv_now > 0. && hv_now - hv_prev <= tol * hv_now {
                info!("Hypervolume improvement below {tol}: consider solver has converged");
                return true;
            }
        }
        false
    }

    /// Retrain surrogates from scratch (the "slow path"), whatever clustering/theta
    /// optimization settings are given, and store the resulting models/inits.
    fn retrain_from_scratch(
        &self,
        surrogates: &mut SurrogateSet,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        spec: TrainSpec,
    ) -> Vec<Option<Array2<f64>>> {
        let mapping = self
            .config
            .cstr_specs
            .as_ref()
            .map(|s| output_mapping(self.config.n_obj_models(), s));

        let (models_new, inits) = if let Some(ref mapping) = mapping {
            self.train_with_mapping(mapping, xt, yt, spec, surrogates)
        } else {
            self.train_all_columns(xt, yt, spec, surrogates)
        };

        surrogates.models = models_new;
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
    pub(crate) fn must_optimize_theta(
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

    /// Bring `surrogates` models in line with the whole training data `(x_data, y_data)`
    /// (see `must_optimize_theta` for the full-retrain-vs-fast-path decision)
    /// then sync their clusterings/theta inits with the resulting models.
    /// `point_index` is the global point index used by `must_optimize_theta`.
    pub(crate) fn refresh_models(
        &self,
        surrogates: &mut SurrogateSet,
        x_data: &Array2<f64>,
        y_data: &Array2<f64>,
        actives: &Array2<usize>,
        do_clustering: DataClustering,
        point_index: usize,
    ) {
        let dim = self.config.gp.kpls_dim.unwrap_or(x_data.ncols());
        let optimize_theta: ThetaOptimization = self
            .must_optimize_theta(do_clustering, x_data.nrows(), dim, point_index)
            .into();
        let spec = TrainSpec {
            clustering: do_clustering,
            theta: optimize_theta,
            actives,
        };
        self.update_surrogates(surrogates, x_data, y_data, spec);
    }

    /// Train `surrogates` models on `(xt, yt)` as specified by `spec`, then sync their
    /// clusterings/theta inits with the resulting models.
    ///
    /// With a fixed clustering, models already trained on all `xt` rows are
    /// left untouched: retraining on the very same data is pointless.
    pub(crate) fn update_surrogates(
        &self,
        surrogates: &mut SurrogateSet,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        spec: TrainSpec,
    ) {
        let inits = if surrogates.is_up_to_date(spec.clustering, xt.nrows()) {
            debug!("Surrogates already up to date with {} points", xt.nrows());
            vec![None; surrogates.models.len()]
        } else {
            info!(
                "Update surrogates with {} points... clustering={:?} optimize_theta={:?}",
                xt.nrows(),
                spec.clustering,
                spec.theta
            );
            self.update_models(surrogates, xt, yt, spec)
        };
        surrogates.sync(&inits);
    }

    /// Update the surrogate models with new data.
    /// When clustering is fixed and theta optimization is disabled, attempts
    /// an incremental update of existing models before falling back to full retraining.
    /// Before performing this incremental "fast path" update, the new point(s) z-score
    /// is checked against the current models: a large z-score (see
    /// `ZSCORE_THETA_OPTIM_THRESHOLD`) means the surrogate hyperparameters are no
    /// longer trustworthy, so a full theta-optimized retraining is triggered instead.
    fn update_models(
        &self,
        surrogates: &mut SurrogateSet,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        spec: TrainSpec,
    ) -> Vec<Option<Array2<f64>>> {
        if spec.clustering == DataClustering::Regenerate || spec.theta == ThetaOptimization::Enabled
        {
            // Slow path: retrain from scratch
            return self.retrain_from_scratch(surrogates, xt, yt, spec);
        }

        // Fast path candidate: incremental update.
        let models = &mut surrogates.models;
        if !models.is_empty() {
            // Check the incoming point(s) against the current surrogate before
            // committing to the cheap incremental update: a large z-score means
            // the fast path is no longer trustworthy, so escalate to a full
            // theta-optimized retraining instead of adding the point as-is.
            if self.zscore_exceeds_threshold(models, xt, yt) {
                let spec = TrainSpec {
                    theta: ThetaOptimization::Enabled,
                    ..spec
                };
                return self.retrain_from_scratch(surrogates, xt, yt, spec);
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
        self.retrain_from_scratch(surrogates, xt, yt, spec)
    }
}

//! Selection of the next points to evaluate in a global EGO step: surrogates training,
//! infill criterion optimization for each point of a batch, and portfolio of GP variance weights.

use crate::moo::criterion::MooCriterion;
use crate::moo::ehvi::EhviCriterion;
use crate::moo::eim::EimCriterion;
use crate::moo::pareto::pareto_front_indices;
use crate::moo::qehvi::{QEhviCriterion, fd_steps};
use crate::solver::multistart::MiddlePickerMultiStarter;
use crate::solver::solver_infill_optim::InfillOptProblem;
use crate::solver::surrogate_set::{DataClustering, SurrogateSet, TrainSpec};
use crate::utils::select_from_portfolio;
use crate::{EgorSolver, types::*};

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_moe::{Clustering, MixtureGpSurrogate};
use log::{debug, info};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, Zip, concatenate, s};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use serde::{Serialize, de::DeserializeOwned};

/// Weights of the GP standard deviation used by the GP variance portfolio: the GP variance
/// is not believed and weighted to generate possibly several clusters hence several points
/// to add (logspace(0.1, 100., 13) with 1. moved in front)
const GP_VAR_PORTFOLIO_SIGMA_WEIGHTS: [f64; 13] = [
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
];

/// Data and status of the optimization the next points are selected from
pub(crate) struct SelectionInputs<'a, CF: CstrFn> {
    /// Whether surrogates are trained for the first time (clustering is generated)
    pub(crate) init: bool,
    /// Current iteration
    pub(crate) iter: u64,
    /// Whether surrogates clustering is regenerated
    pub(crate) recluster: bool,
    /// Components activity (cooperative strategy)
    pub(crate) activity: &'a Array2<usize>,
    /// Evaluated points
    pub(crate) x_data: ArrayView2<'a, f64>,
    /// Output values (internal layout) surrogates are trained on
    pub(crate) y_data: ArrayView2<'a, f64>,
    /// Function constraints values
    pub(crate) c_data: ArrayView2<'a, f64>,
    /// Points where the evaluation failed
    pub(crate) x_fail: Option<&'a Array2<f64>>,
    /// Internal constraints tolerances
    pub(crate) cstr_tol: &'a Array1<f64>,
    /// Index of the current best point
    pub(crate) best_index: usize,
    /// Whether a feasible point is found
    pub(crate) feasibility: bool,
    /// Function constraints
    pub(crate) fcstrs: &'a [CF],
    /// Function constraints specifications
    pub(crate) fcstr_specs: Option<&'a [CstrSpec]>,
}

/// Points selected so far in a batch, the GP standard deviation being weighted by `sigma_weight`
struct Batch {
    /// Imputed failed points of the initial DOE, if any, then selected points
    points: Candidates,
    /// Number of imputed failed points
    n_imputed: usize,
    sigma_weight: f64,
}

impl Batch {
    /// Points selected by the infill criterion optimization
    fn selected_x(&self) -> ArrayView2<'_, f64> {
        self.points.x.slice(s![self.n_imputed.., ..])
    }
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
    C: CstrFn,
{
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
        let mut surrogates = SurrogateSet {
            clusterings: clusterings.iter_mut().map(Option::take).collect(),
            theta_inits: theta_inits.iter_mut().map(Option::take).collect(),
            models: std::mem::take(models),
        };
        let inputs = SelectionInputs {
            init,
            iter,
            recluster,
            activity,
            x_data: x_data.view(),
            y_data: y_data.view(),
            c_data: c_data.view(),
            x_fail: x_fail_points,
            cstr_tol,
            best_index,
            feasibility,
            fcstrs: cstr_funcs,
            fcstr_specs,
        };
        let candidates = self.select_candidates(&inputs, &mut surrogates, rng);
        std::iter::zip(clusterings, surrogates.clusterings).for_each(|(dst, src)| *dst = src);
        std::iter::zip(theta_inits, surrogates.theta_inits).for_each(|(dst, src)| *dst = src);
        *models = surrogates.models;
        (
            candidates.x,
            candidates.y,
            candidates.c,
            candidates.y_penalized,
            candidates.infill_value,
        )
    }

    /// Returns next promising x points together with virtual (predicted) y values
    /// from `surrogates` (taking into account qei strategy if q_parallel)
    /// and the infill criterion value. `surrogates` are trained beforehand if needed.
    pub(crate) fn select_candidates(
        &self,
        inputs: &SelectionInputs<impl CstrFn>,
        surrogates: &mut SurrogateSet,
        rng: &mut Xoshiro256Plus,
    ) -> Candidates {
        let sigma_weights: &[f64] = if self.config.runtime_flags.use_gp_var_portfolio
            && self.config.qei_config.batch == 1
        {
            &GP_VAR_PORTFOLIO_SIGMA_WEIGHTS
        } else {
            // Fallback to default GP usage
            &[1.]
        };
        // qEHVI: surrogates of the actual data only (no virtual points), the points
        // already selected in the batch (after imputed failed points) entering the criterion
        let qehvi = self.config.is_per_objective()
            && self.config.moo_strategy() == crate::MooStrategy::QEhvi;

        let (x_data, y_data) = (&inputs.x_data, &inputs.y_data);
        let mut portfolio = vec![];
        for (j, &sigma_weight) in sigma_weights.iter().enumerate() {
            debug!("Make surrogate with {x_data}");
            let mut batch = Batch {
                points: Candidates::empty(x_data.ncols(), y_data.ncols(), inputs.c_data.ncols()),
                n_imputed: 0,
                sigma_weight,
            };

            for i in 0..self.config.qei_config.batch {
                let (xt, yt) = if i == 0 || qehvi {
                    (x_data.to_owned(), y_data.to_owned())
                } else {
                    (
                        concatenate![Axis(0), x_data.to_owned(), batch.points.x.to_owned()],
                        concatenate![Axis(0), y_data.to_owned(), batch.points.y.to_owned()],
                    )
                };
                self.train_for_point(inputs, surrogates, &xt, &yt, i, j == 0);

                if inputs.iter == 0
                    && i == 0
                    && self.config.failsafe_strategy == FailsafeStrategy::Imputation
                    && let Some(x_fail) = inputs.x_fail
                {
                    // we have to predict penalization on these points and add them to data
                    batch.points = self.impute_failed_points(inputs, &surrogates.models, x_fail);
                    batch.n_imputed = x_fail.nrows();
                }

                if i == 0 && j == 0 {
                    self.save_models(&surrogates.models, inputs.iter);
                }

                match self.select_point(inputs, &surrogates.models, &batch, (&xt, &yt), rng) {
                    Some(point) => batch.points.append(point),
                    None => break,
                }
            }
            portfolio.push(batch.points);
        }
        if portfolio.len() > 1 {
            info!(
                "Portfolio : {:?}",
                portfolio.iter().map(|v| v.x[[0, 0]]).collect::<Vec<_>>()
            );
            // Use portfolio strategy: Pick one point from portfolio
            select_from_portfolio(portfolio)
        } else {
            // Fallback to default returning one or several points (in case of qEI strategy)
            portfolio.remove(0)
        }
    }

    /// Train `surrogates` on `(xt, yt)` before selecting the `i`-th point of a batch.
    /// Theta hyperparameters are only optimized with the `first_weight` of the portfolio.
    fn train_for_point(
        &self,
        inputs: &SelectionInputs<impl CstrFn>,
        surrogates: &mut SurrogateSet,
        xt: &Array2<f64>,
        yt: &Array2<f64>,
        i: usize,
        first_weight: bool,
    ) {
        log::debug!("activity: {:?}", inputs.activity);
        let do_clustering: DataClustering = ((inputs.init && i == 0) || inputs.recluster).into();

        // The fast (incremental) path is the default strategy; see
        // `must_optimize_theta` for when a full retrain is triggered instead.
        // Otherwise the fast path is attempted, itself able to escalate to a
        // full theta-optimized retraining should the incoming point(s) turn
        // out to be a poor surprise for the current surrogate, i.e. its
        // z-score exceeds `ZSCORE_THETA_OPTIM_THRESHOLD` (see
        // `update_models`'s z-score check).
        let dim = self.config.gp.kpls_dim.unwrap_or(xt.ncols());
        let point_index = inputs.iter as usize * self.config.qei_config.batch + i;
        let optimize_theta =
            first_weight && self.must_optimize_theta(do_clustering, xt.nrows(), dim, point_index);

        // Persisted models are already trained on the evaluated points at
        // the end of the previous iteration: they are not retrained on the
        // very same data.
        let spec = TrainSpec {
            clustering: do_clustering,
            theta: optimize_theta.into(),
            actives: inputs.activity,
        };
        self.update_surrogates(surrogates, xt, yt, spec);
    }

    /// Failed points of the initial DOE with output values penalized by `models`
    /// (failsafe imputation strategy)
    fn impute_failed_points(
        &self,
        inputs: &SelectionInputs<impl CstrFn>,
        models: &[Box<dyn MixtureGpSurrogate>],
        x_fail: &Array2<f64>,
    ) -> Candidates {
        info!(
            "Impute failed initial points ({} points)...",
            x_fail.nrows()
        );
        let mut y_pen_imputed = Array2::zeros((x_fail.nrows(), self.config.n_surrogates()));
        let (obj_models, cstr_models) = models.split_at(self.config.n_obj_models());
        Zip::from(y_pen_imputed.rows_mut())
            .and(x_fail.rows())
            .for_each(|mut y_row, xfail| {
                let y_pred =
                    self.compute_penalized_point(&xfail, obj_models, cstr_models, &inputs.y_data);
                y_row.assign(&y_pred);
            });
        Candidates {
            x: x_fail.to_owned(),
            y: Array2::from_elem((x_fail.nrows(), inputs.y_data.ncols()), f64::NAN),
            c: self.eval_fcstrs(
                &crate::types::affine_fcstrs(inputs.fcstrs, inputs.fcstr_specs),
                x_fail,
            ),
            y_penalized: y_pen_imputed,
            infill_value: f64::INFINITY,
        }
    }

    /// Save `models` in the output directory, if any
    fn save_models(&self, models: &[Box<dyn MixtureGpSurrogate>], iter: u64) {
        use crate::{EGOR_GP_FILENAME, EGOR_INITIAL_GP_FILENAME, utils::gp_recorder};

        if let Some(outdir) = self.config.outdir.as_ref() {
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
    }

    /// Select the next point of the `batch` by optimizing the infill criterion built from
    /// `models` trained on `(xt, yt)`. Returns the point with its virtual (predicted) values,
    /// or `None` when no point can be selected.
    fn select_point(
        &self,
        inputs: &SelectionInputs<impl CstrFn>,
        models: &[Box<dyn MixtureGpSurrogate>],
        batch: &Batch,
        (xt, yt): (&Array2<f64>, &Array2<f64>),
        rng: &mut Xoshiro256Plus,
    ) -> Option<Candidates> {
        let (x_data, y_data, c_data) = (&inputs.x_data, &inputs.y_data, &inputs.c_data);
        let cstr_tol = inputs.cstr_tol;
        let best_index = inputs.best_index;

        let (obj_models, cstr_models) = models.split_at(self.config.n_obj_models());
        let obj_model = &obj_models[0];
        debug!("... surrogates trained");

        let fmin = self.incumbent_value(obj_model.as_ref(), x_data, y_data, best_index);
        let ybest = y_data.row(best_index).to_owned();
        let xbest = x_data.row(best_index).to_owned();
        let cbest = c_data.row(best_index).to_owned();

        let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
        let sampling = Lhs::new(&self.xlimits)
            .kind(LhsKind::Maximin)
            .with_rng(sub_rng);

        let transformed_fcstrs = crate::types::affine_fcstrs(inputs.fcstrs, inputs.fcstr_specs);

        let (scale_infill_obj, scale_cstr, scale_fcstr, scale_wb2) = self.compute_scaling(
            &sampling,
            obj_model.as_ref(),
            cstr_models,
            cstr_tol,
            transformed_fcstrs.len(),
            fmin,
            batch.sigma_weight,
        );
        let moo_criterion = if self.config.is_per_objective() {
            Some(self.moo_criterion(inputs, obj_models, batch, yt, rng)?)
        } else {
            None
        };
        let scale_infill_obj = if let Some(criterion) = moo_criterion.as_ref() {
            let npts = (100 * self.xlimits.nrows()).min(1000);
            let scale = criterion.scaling(&sampling.sample(npts));
            info!(
                "{} criterion ({} front points) scaling is updated to {scale}",
                criterion.name(),
                criterion.front_size()
            );
            scale
        } else {
            scale_infill_obj
        };
        let scale_pov_cstr = Array1::ones((1,)); // PoV cstr is normalized
        let all_scale_cstr = concatenate![Axis(0), scale_cstr, scale_fcstr, scale_pov_cstr];

        // fmin and xbest are kept the same for all q points
        // Would it be best to update them with regard to predicted virtual points?
        // Keep it simple: For the moment we keep them fixed to the current best observed point
        let infill_data = InfillObjData {
            fmin,
            xbest: xbest.to_vec(),
            scale_infill_obj,
            scale_cstr: Some(all_scale_cstr.to_owned()),
            scale_wb2,
            feasibility: inputs.feasibility,
            sigma_weight: batch.sigma_weight,
        };

        let cstr_funcs = self.infill_fcstrs(&transformed_fcstrs);

        let viability_model = self.viability_model(xt, inputs.x_fail);

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
            inputs.activity,
            moo_criterion.as_ref(),
        );

        let (infill_obj, xk) =
            self.optimize_infill_criterion(infill_optpb, multistarter, (xbest, ybest, cbest));
        debug!("+++++++  xk = {xk}");

        match self.compute_virtual_point(&xk, y_data, obj_models, cstr_models) {
            Ok(yk) => {
                let yk = Array2::from_shape_vec((1, self.config.n_surrogates()), yk).unwrap();
                let yk_pen = self.compute_penalized_point(&xk, obj_models, cstr_models, y_data);
                let xk = xk.insert_axis(Axis(0));
                // Function constraints values as stored in data (unscaled)
                let ck = self.eval_fcstrs(&transformed_fcstrs, &xk);
                Some(Candidates {
                    x: xk,
                    y: yk,
                    c: ck,
                    y_penalized: yk_pen.insert_axis(Axis(0)),
                    // infill objective was minimized while infill criterion itself
                    // is expected to be maximized hence the negative sign here
                    infill_value: -infill_obj,
                })
            }
            Err(err) => {
                // Error while predict at best point: ignore
                info!("Error while getting virtual point: {err}");
                None
            }
        }
    }

    /// Multi-objective criterion (EIM, EHVI, qEHVI) wrt the Pareto front of the data `yt`,
    /// including virtual points of the batch so far (actual data only with qEHVI).
    /// Returns `None` when the qEHVI criterion is not available.
    fn moo_criterion<'a>(
        &self,
        inputs: &SelectionInputs<impl CstrFn>,
        obj_models: &'a [Box<dyn MixtureGpSurrogate>],
        batch: &Batch,
        yt: &Array2<f64>,
        rng: &mut Xoshiro256Plus,
    ) -> Option<MooCriterion<'a>> {
        let n_obj = self.config.n_obj();
        let ct = if self.config.moo_strategy() == crate::MooStrategy::QEhvi {
            inputs.c_data.to_owned()
        } else {
            concatenate![Axis(0), inputs.c_data.to_owned(), batch.points.c.to_owned()]
        };
        // failed points with imputed values are never part of the front
        let excluded = self.config.imputed_rows(&inputs.x_data, inputs.x_fail);
        let front = pareto_front_indices(yt, &ct, n_obj, inputs.cstr_tol, &excluded);
        let objs = yt.slice(s![.., ..n_obj]);
        let criterion = match self.config.moo_strategy() {
            crate::MooStrategy::QEhvi if batch.selected_x().nrows() > 0 => {
                match QEhviCriterion::new(
                    obj_models,
                    &objs,
                    &front,
                    &batch.selected_x(),
                    fd_steps(&self.config.xtypes),
                    rng.r#gen(),
                ) {
                    Ok(criterion) => MooCriterion::QEhvi(criterion),
                    Err(err) => {
                        log::error!(
                            "qEHVI criterion not available, batch stopped at {} points: {err}",
                            batch.selected_x().nrows()
                        );
                        return None;
                    }
                }
            }
            crate::MooStrategy::Ehvi | crate::MooStrategy::QEhvi => {
                MooCriterion::Ehvi(EhviCriterion::new(obj_models, &objs, &front))
            }
            _ => MooCriterion::Eim(EimCriterion::new(
                obj_models,
                &objs,
                &front,
                self.config.moo.eim_aggregation,
            )),
        };
        Some(criterion)
    }
}

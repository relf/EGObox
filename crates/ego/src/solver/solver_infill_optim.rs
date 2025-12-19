use crate::optimizers::*;
use crate::types::*;

use crate::EgorSolver;

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

use egobox_moe::MixtureGpSurrogate;
use log::info;
use ndarray::{Array, Array1, Array2};

use rayon::prelude::*;
use serde::de::DeserializeOwned;

use super::coego;

/// A trait for multi start initial points computation
pub(crate) trait MultiStarter {
    /// Return initial points for optimization multistart
    /// taking into account active components given as a set of indices
    fn multistart(&mut self, n_start: usize, active: &[usize]) -> Array2<f64>;

    /// Return the bounds of the design space
    /// Compared to xlimits which are given by the user for all x components,
    /// xbounds are the actual bounds used during optimization taking into account
    /// active components and possibly trust region restrictions
    fn xbounds(&self, active: &[usize]) -> Array2<f64>;
}

pub(crate) struct InfillOptProblem<'a, CstrFn> {
    pub obj_model: &'a dyn MixtureGpSurrogate,
    pub cstr_models: &'a [Box<dyn MixtureGpSurrogate>],
    pub cstr_funcs: &'a [CstrFn],
    pub cstr_tols: &'a Array1<f64>,
    pub infill_data: &'a InfillObjData<f64>,
    pub actives: &'a Array2<usize>,
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
    C: CstrFn,
{
    /// Find best x promising points by optimizing the chosen infill criterion
    /// The optimized value of the criterion is returned together with the
    /// optimum location
    /// Returns (infill_obj, x_opt)
    pub(crate) fn optimize_infill_criterion<MS>(
        &self,
        infill_optpb: InfillOptProblem<impl CstrFn>,
        mut multistarter: MS,
        current_best: (Array1<f64>, Array1<f64>, Array1<f64>),
    ) -> (f64, Array1<f64>)
    where
        MS: MultiStarter,
    {
        let InfillOptProblem {
            obj_model,
            cstr_models,
            cstr_funcs,
            cstr_tols,
            infill_data,
            actives,
        } = infill_optpb;
        let mut infill_data = infill_data.clone();

        let mut best_point = (current_best.1[0], current_best.0.to_owned());
        let mut current_best_point = current_best.to_owned();

        for (i, active) in actives.outer_iter().enumerate() {
            let mut success = false;
            let mut n_optim = 1;
            let n_max_optim = 3;

            let active = active.to_vec();
            let obj =
                |x: &[f64], gradient: Option<&mut [f64]>, params: &mut InfillObjData<f64>| -> f64 {
                    let InfillObjData {
                        scale_infill_obj,
                        scale_wb2,
                        xbest: xcoop,
                        fmin,
                        feasibility,
                        sigma_weight,
                        ..
                    } = params;
                    let mut xcoop = xcoop.clone();
                    coego::set_active_x(&mut xcoop, &active, x);

                    // Defensive programming NlOpt::Cobyla may pass NaNs
                    if xcoop.iter().any(|x| x.is_nan()) {
                        return f64::INFINITY;
                    }

                    if let Some(grad) = gradient {
                        let g_infill_obj = if self.config.cstr_infill {
                            // Use constrained infill criterion
                            self.eval_grad_infill_obj_with_cstrs(
                                &xcoop,
                                obj_model,
                                cstr_models,
                                cstr_tols,
                                *fmin,
                                *scale_infill_obj,
                                *scale_wb2,
                                *feasibility,
                                *sigma_weight,
                            )
                        } else {
                            self.eval_grad_infill_obj(
                                &xcoop,
                                obj_model,
                                *fmin,
                                *scale_infill_obj,
                                *scale_wb2,
                            )
                        };
                        let g_infill_obj = g_infill_obj
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| active.contains(i))
                            .map(|(_, &g)| g)
                            .collect::<Vec<_>>();
                        grad[..].copy_from_slice(&g_infill_obj);
                    }
                    if self.config.cstr_infill {
                        // Use constrained infill criterion
                        self.eval_infill_obj_with_cstrs(
                            &xcoop,
                            obj_model,
                            cstr_models,
                            cstr_tols,
                            *fmin,
                            *scale_infill_obj,
                            *scale_wb2,
                            *feasibility,
                            *sigma_weight,
                        )
                    } else {
                        self.eval_infill_obj(
                            &xcoop,
                            obj_model,
                            *fmin,
                            *scale_infill_obj,
                            *scale_wb2,
                            *sigma_weight,
                        )
                    }
                };

            let cstrs: Vec<_> = if self.config.cstr_infill {
                // When constrained infill criterion is used
                // internal infill criterion optimizer does not
                // handle constraints metamodelized constraints
                vec![]
            } else {
                (0..self.config.n_cstr)
                    .map(|i| {
                        let active = active.to_vec();
                        let cstr = move |x: &[f64],
                                         gradient: Option<&mut [f64]>,
                                         params: &mut InfillObjData<f64>|
                              -> f64 {
                            let InfillObjData { xbest: xcoop, .. } = params;
                            let mut xcoop = xcoop.clone();
                            coego::set_active_x(&mut xcoop, &active, x);

                            let scale_cstr =
                                params.scale_cstr.as_ref().expect("constraint scaling")[i];
                            if self.config.cstr_strategy == ConstraintStrategy::MeanConstraint {
                                Self::mean_cstr(
                                    &*cstr_models[i],
                                    &xcoop,
                                    gradient,
                                    scale_cstr,
                                    &active,
                                )
                            } else {
                                Self::upper_trust_bound_cstr(
                                    &*cstr_models[i],
                                    &xcoop,
                                    gradient,
                                    scale_cstr,
                                    &active,
                                )
                            }
                        };
                        #[cfg(feature = "nlopt")]
                        {
                            Box::new(cstr) as Box<dyn nlopt::ObjFn<InfillObjData<f64>> + Sync>
                        }
                        #[cfg(not(feature = "nlopt"))]
                        {
                            Box::new(cstr)
                                as Box<dyn crate::types::ObjFn<InfillObjData<f64>> + Sync>
                        }
                    })
                    .collect()
            };

            // We merge metamodelized constraints and function constraints
            let mut cstr_refs: Vec<_> = cstrs.iter().map(|c| c.as_ref()).collect();
            let cstr_funcs = cstr_funcs
                .iter()
                .map(|cstr| cstr as &(dyn ObjFn<InfillObjData<f64>> + Sync))
                .collect::<Vec<_>>();
            cstr_refs.extend(cstr_funcs.clone());

            // Limits of activated components
            let xbounds = multistarter.xbounds(&active).to_owned();

            let algorithm = match self.config.infill_optimizer {
                InfillOptimizer::Slsqp => crate::optimizers::Algorithm::Slsqp,
                InfillOptimizer::Cobyla => crate::optimizers::Algorithm::Cobyla,
            };

            if i == 0 {
                info!("Optimize infill criterion...");
            }
            while !success && n_optim <= n_max_optim {
                let x_start = multistarter.multistart(self.config.n_start, &active);
                let res = (0..x_start.nrows())
                    .into_par_iter()
                    .map(|i| {
                        Optimizer::new(algorithm, &obj, &cstr_refs, &infill_data, &xbounds)
                            .xinit(&x_start.row(i))
                            .max_eval((10 * x_start.len()).min(INFILL_MAX_EVAL_DEFAULT))
                            .ftol_rel(1e-4)
                            .ftol_abs(1e-4)
                            .minimize()
                    })
                    .reduce(
                        || (f64::INFINITY, Array::ones((xbounds.nrows(),))),
                        |a, b| if b.0 < a.0 { b } else { a },
                    );

                if res.0.is_nan() || res.0.is_infinite() {
                    success = false;
                } else {
                    let mut xopt_coop = current_best_point.0.to_vec();
                    coego::set_active_x(&mut xopt_coop, &active, &res.1.to_vec());
                    infill_data.xbest = xopt_coop.clone();
                    let xopt_coop = Array1::from(xopt_coop);

                    if crate::solver::coego::COEGO_IMPROVEMENT_CHECK {
                        let (is_better, best) = self.is_objective_improved(
                            &current_best_point,
                            &xopt_coop,
                            obj_model,
                            cstr_models,
                            cstr_tols,
                            &cstr_funcs,
                        );
                        if is_better || i == 0 {
                            if i > 0 {
                                info!(
                                    "Partial infill criterion optim c={} has better result={} at x={}",
                                    i, best.0, xopt_coop
                                );
                            }
                            best_point = (res.0, xopt_coop);
                            current_best_point = best;
                        }
                    } else {
                        best_point = (res.0, xopt_coop.to_owned());
                        current_best_point =
                            (xopt_coop, current_best_point.1, current_best_point.2);
                    }
                    success = true;
                }
                n_optim += 1;
            }
        }
        best_point
    }
}

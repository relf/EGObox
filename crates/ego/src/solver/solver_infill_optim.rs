use crate::optimizers::*;
use crate::types::*;

use crate::EgorSolver;
use crate::moo::eim::EimCriterion;
use crate::utils::{pofs, pofs_grad};

use egobox_moe::MixtureGpSurrogate;
use log::info;
use ndarray::{Array, Array1, Array2};

use rayon::prelude::*;
use serde::{Serialize, de::DeserializeOwned};

use super::coego;

/// Minimal probability of viability required for an infill point
/// when viability failsafe strategy is used
const MIN_PROBA_OF_VIABILITY: f64 = 0.25;

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
    pub viability_model: Option<Box<dyn MixtureGpSurrogate>>,
    pub alpha: Option<f64>,
    pub infill_data: &'a InfillObjData<f64>,
    pub actives: &'a Array2<usize>,
    /// Multi-objective EIM criterion replacing the mono-objective infill criterion (if any)
    pub eim: Option<&'a EimCriterion<'a>>,
}

impl<'a, CstrFn> InfillOptProblem<'a, CstrFn> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        obj_model: &'a dyn MixtureGpSurrogate,
        cstr_models: &'a [Box<dyn MixtureGpSurrogate>],
        cstr_funcs: &'a [CstrFn],
        cstr_tols: &'a Array1<f64>,
        viability_model: Option<Box<dyn MixtureGpSurrogate>>,
        alpha: Option<f64>,
        infill_data: &'a InfillObjData<f64>,
        actives: &'a Array2<usize>,
        eim: Option<&'a EimCriterion<'a>>,
    ) -> Self {
        Self {
            obj_model,
            cstr_models,
            cstr_funcs,
            cstr_tols,
            viability_model,
            alpha,
            infill_data,
            actives,
            eim,
        }
    }
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
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
            viability_model,
            alpha,
            infill_data,
            actives,
            eim,
        } = infill_optpb;
        let mut infill_data = infill_data.clone();

        let mut best_point = (current_best.1[0], current_best.0.to_owned());
        let mut current_best_point = current_best.to_owned();

        for (i, active) in actives.outer_iter().enumerate() {
            let active = active.to_vec();
            let viab_model = viability_model
                .as_deref()
                .filter(|_| self.config.feasibility_infill.is_enabled());

            // Infill objective: when `cstr_infill` is true, the infill criterion
            // is weighted by the probability of feasibility (PoF) of the
            // metamodelized constraints, otherwise the criterion is used as is
            // and the metamodelized constraints are handled by the optimizer.
            let make_obj = |cstr_infill: bool| {
                let active = active.clone();
                move |x: &[f64],
                      gradient: Option<&mut [f64]>,
                      params: &mut InfillObjData<f64>|
                      -> f64 {
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

                    // Defensive programming COBYLA may pass NaNs
                    if xcoop.iter().any(|x| x.is_nan()) {
                        return f64::INFINITY;
                    }

                    if let Some(eim) = eim {
                        // Multi-objective EIM criterion (linear composition with PoF)
                        let with_grad = gradient.is_some();
                        let (mut obj, mut g_obj) = if cstr_infill && !*feasibility {
                            // neutral factor: only use the probability of feasibility
                            (-1., Array1::zeros(xcoop.len()))
                        } else if with_grad {
                            let (v, g) = eim.value_grad(&xcoop);
                            (-v / *scale_infill_obj, -g / *scale_infill_obj)
                        } else {
                            (
                                -eim.value(&xcoop) / *scale_infill_obj,
                                Array1::zeros(xcoop.len()),
                            )
                        };
                        if cstr_infill {
                            let tols = cstr_tols.to_vec();
                            let pof = pofs(&xcoop, cstr_models, &tols);
                            if with_grad {
                                g_obj = g_obj * pof + pofs_grad(&xcoop, cstr_models, &tols) * obj;
                            }
                            obj *= pof;
                        }
                        if let Some(grad) = gradient {
                            let g_obj = g_obj
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| active.contains(i))
                                .map(|(_, &g)| g)
                                .collect::<Vec<_>>();
                            grad[..].copy_from_slice(&g_obj);
                        }
                        return obj;
                    }

                    if let Some(grad) = gradient {
                        let g_infill_obj = if cstr_infill {
                            // Use constrained infill criterion
                            self.eval_grad_infill_obj_with_cstrs(
                                &xcoop,
                                obj_model,
                                cstr_models,
                                cstr_tols,
                                *fmin,
                                viab_model,
                                alpha,
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
                                viab_model,
                                alpha,
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
                    if cstr_infill {
                        // Use constrained infill criterion
                        self.eval_infill_obj_with_cstrs(
                            &xcoop,
                            obj_model,
                            cstr_models,
                            cstr_tols,
                            *fmin,
                            viab_model,
                            alpha,
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
                            viab_model,
                            alpha,
                            *scale_infill_obj,
                            *scale_wb2,
                            *sigma_weight,
                        )
                    }
                }
            };
            let obj = make_obj(self.config.cstr_infill);

            // Metamodelized constraints, only used when the infill criterion
            // is not the constrained one (i.e. not already weighted by PoF)
            let cstrs: Vec<_> = (0..self.config.n_internal_cstr())
                .map(|i| {
                    let active = active.to_vec();
                    let cstr = move |x: &[f64],
                                     gradient: Option<&mut [f64]>,
                                     params: &mut InfillObjData<f64>|
                          -> f64 {
                        let InfillObjData { xbest: xcoop, .. } = params;
                        let mut xcoop = xcoop.clone();
                        coego::set_active_x(&mut xcoop, &active, x);

                        let scale_cstr = params.scale_cstr.as_ref().expect("constraint scaling")[i];
                        if self.config.cstr_strategy == ConstraintStrategy::MeanConstraint {
                            Self::mean_cstr(&*cstr_models[i], &xcoop, gradient, scale_cstr, &active)
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

                    Box::new(cstr) as Box<dyn OptFn<InfillObjData<f64>> + Sync>
                })
                .collect();

            // Constraints always handled by the optimizer: function constraints
            // and viability constraint (if viability strategy)
            let mut other_cstr_refs: Vec<_> = cstr_funcs
                .iter()
                .map(|cstr| cstr as &(dyn OptFn<InfillObjData<f64>> + Sync))
                .collect::<Vec<_>>();

            // If viability strategy, we add the corresponding constraint
            let viability_cstr =
                |x: &[f64], gradient: Option<&mut [f64]>, params: &mut InfillObjData<f64>| -> f64 {
                    let mut gradient = gradient;
                    if let Some(viab_model) = &viability_model {
                        let active = active.to_vec();
                        let InfillObjData { xbest: xcoop, .. } = params;
                        let mut xcoop = xcoop.clone();
                        coego::set_active_x(&mut xcoop, &active, x);
                        let pov = Self::mean_cstr(
                            &**viab_model,
                            &xcoop,
                            gradient.as_deref_mut(),
                            1.0,
                            &active,
                        );
                        // Constraint is MIN_PROBA_OF_VIABILITY - pov: negate pov gradient,
                        // and zero it where clamping makes the constraint flat
                        if let Some(grad) = gradient {
                            if (0.0..=1.0).contains(&pov) {
                                grad.iter_mut().for_each(|g| *g = -*g);
                            } else {
                                grad.fill(0.0);
                            }
                        }
                        MIN_PROBA_OF_VIABILITY - pov.clamp(0.0, 1.0)
                    } else {
                        // If no viability model is provided, consider the point as feasible by default
                        if let Some(grad) = gradient {
                            grad.fill(0.0);
                        }
                        -1.0
                    }
                };
            if self.config.failsafe_strategy == FailsafeStrategy::Viability {
                other_cstr_refs.push(&viability_cstr as &(dyn OptFn<InfillObjData<f64>> + Sync));
            }

            // We merge metamodelized constraints and function constraints
            let cstr_refs: Vec<_> = if self.config.cstr_infill {
                // When constrained infill criterion is used
                // internal infill criterion optimizer does not
                // handle metamodelized constraints
                other_cstr_refs.clone()
            } else {
                cstrs
                    .iter()
                    .map(|c| c.as_ref())
                    .chain(other_cstr_refs.iter().copied())
                    .collect()
            };

            // Limits of activated components
            let xbounds = multistarter.xbounds(&active).to_owned();

            if i == 0 {
                info!("Optimize infill criterion...");
            }
            let mut res = self.multistart_minimize(
                &obj,
                &cstr_refs,
                &infill_data,
                &xbounds,
                &active,
                &mut multistarter,
            );

            if res.is_none() && !self.config.cstr_infill && !cstrs.is_empty() {
                // No point satisfying the metamodelized constraints was found
                // (typically when no feasible point is known yet): fall back to
                // the infill criterion weighted by the probability of feasibility
                // which reduces to maximizing the PoF when no feasible point is known
                info!(
                    "Infill optimization failed to satisfy constraint models, retry with constrained infill criterion"
                );
                let pof_obj = make_obj(true);
                res = self.multistart_minimize(
                    &pof_obj,
                    &other_cstr_refs,
                    &infill_data,
                    &xbounds,
                    &active,
                    &mut multistarter,
                );
            }

            let res = res.or_else(|| {
                if infill_data.feasibility {
                    // A feasible point is known: keep current best point which will be
                    // rejected as too close to existing data, eventually leading
                    // to solver convergence
                    return None;
                }
                // Last chance when no feasible point is known yet: rather than returning
                // the current best point (hence rejected and stopping the solver),
                // pick the best random starting point regarding the infill objective
                info!("Infill optimization failed, pick best random point");
                let x_start = multistarter.multistart(self.config.n_start, &active);
                let mut data = infill_data.clone();
                let res = x_start
                    .outer_iter()
                    .map(|x| (obj(&x.to_vec(), None, &mut data), x.to_owned()))
                    .fold((f64::INFINITY, x_start.row(0).to_owned()), |a, b| {
                        if b.0 < a.0 { b } else { a }
                    });
                Some(res)
            });

            if let Some(res) = res {
                let mut xopt_coop = current_best_point.0.to_vec();
                coego::set_active_x(&mut xopt_coop, &active, &res.1.to_vec());
                infill_data.xbest = xopt_coop.clone();
                let xopt_coop = Array1::from(xopt_coop);

                best_point = (res.0, xopt_coop.to_owned());
                current_best_point = (xopt_coop, current_best_point.1, current_best_point.2);
            }
        }
        best_point
    }

    /// Minimize `obj` subject to `cstrs` from several starting points,
    /// retrying with new starting points on failure.
    /// Returns the best (value, x) found or None if all attempts failed
    fn multistart_minimize<MS: MultiStarter>(
        &self,
        obj: &(dyn OptFn<InfillObjData<f64>> + Sync),
        cstrs: &[&(dyn OptFn<InfillObjData<f64>> + Sync)],
        infill_data: &InfillObjData<f64>,
        xbounds: &Array2<f64>,
        active: &[usize],
        multistarter: &mut MS,
    ) -> Option<(f64, Array1<f64>)> {
        let algorithm = match self.config.infill_optimizer {
            InfillOptimizer::Slsqp => crate::optimizers::Algorithm::Slsqp,
            InfillOptimizer::Cobyla => crate::optimizers::Algorithm::Cobyla,
        };
        let n_max_optim = 3;
        for _ in 0..n_max_optim {
            let x_start = multistarter.multistart(self.config.n_start, active);
            let res = (0..x_start.nrows())
                .into_par_iter()
                .map(|i| {
                    Optimizer::new(algorithm, obj, cstrs, infill_data, xbounds)
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
            if res.0.is_finite() {
                return Some(res);
            }
        }
        None
    }
}

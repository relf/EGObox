#![allow(dead_code)]
use crate::CstrFn;
use crate::EgorSolver;
use crate::EgorState;
use crate::InfillObjData;
use crate::SurrogateBuilder;
use crate::solver::solver_infill_optim::InfillOptProblem;
use crate::types::DomainConstraints;
use crate::utils::check_update_ok;
use crate::utils::find_best_result_index_from;
use crate::utils::is_feasible;
use crate::utils::update_data;

use argmin::core::{CostFunction, Problem, State};

use egobox_doe::Lhs;
use egobox_doe::SamplingMethod;
use egobox_moe::MixtureGpSurrogate;

use log::debug;
use log::info;
use ndarray::Zip;
use ndarray::aview1;
use ndarray::{Array1, Array2, Axis};

use ndarray_rand::rand::Rng;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use serde::de::DeserializeOwned;

use super::coego;
use super::solver_infill_optim::MultiStarter;

/// LocalMultiStarter is a multistart strategy that samples points in the local area
/// defined by the trust region and the xlimits.
struct LocalLhsMultiStarter<R: Rng + Clone> {
    xlimits: Array2<f64>,
    origin: Array1<f64>,
    local_bounds: (f64, f64),
    rng: R,
}

impl<R: Rng + Clone> LocalLhsMultiStarter<R> {
    fn new(xlimits: Array2<f64>, origin: Array1<f64>, local_bounds: (f64, f64), rng: R) -> Self {
        Self {
            xlimits,
            origin,
            local_bounds,
            rng,
        }
    }
}

impl<R: Rng + Clone> MultiStarter for LocalLhsMultiStarter<R> {
    fn multistart(&mut self, n_start: usize, active: &[usize]) -> Array2<f64> {
        // Draw n_start initial points (multistart optim) in the local_area
        // xbounds = intersection(trust_region, xlimits)
        let xbounds = self.xbounds(active);
        let lhs = Lhs::new(&xbounds)
            .kind(egobox_doe::LhsKind::Maximin)
            .with_rng(&mut self.rng);
        lhs.sample(n_start)
    }

    fn xbounds(&self, active: &[usize]) -> Array2<f64> {
        // local_area = intersection(trust_region, xlimits[active])
        let xlimits = coego::get_active_x(Axis(0), &self.xlimits, active);
        let origin = coego::get_active_x(Axis(0), &self.origin, active);
        let mut local_area = Array2::zeros(xlimits.dim());
        Zip::from(local_area.rows_mut())
            .and(&origin)
            .and(xlimits.rows())
            .for_each(|mut row, xb, xlims| {
                let (lo, up) = (
                    xlims[0].max(xb - self.local_bounds.0),
                    xlims[1].min(xb + self.local_bounds.1),
                );
                row.assign(&aview1(&[lo, up]))
            });
        local_area
    }
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + DeserializeOwned,
    C: CstrFn,
{
    pub(crate) fn update_trego_state(&mut self, state: &EgorState<f64>) -> (Phase, EgorState<f64>) {
        let rho = |sigma| sigma * sigma;
        let (_, y_data, _) = state.data.as_ref().unwrap();
        // initialized in init
        let best = state.best_index.unwrap();
        // initialized in init
        let prev_best = state.prev_best_index.unwrap();
        // initialized in init

        let mut new_state = state.clone();

        // Update cumulative decrease over the trego phase
        let decrease = y_data[[prev_best, 0]] - y_data[[best, 0]];
        new_state.best_decrease += decrease.max(0.0);
        let sufficient_decrease = new_state.best_decrease >= rho(state.sigma);

        if state.global_trego_iter == self.config.trego_config.n_gl_steps.0
            || state.local_trego_iter == self.config.trego_config.n_gl_steps.1
        {
            log::info!(
                "Cumulative decrease: {}, required {}",
                new_state.best_decrease,
                rho(state.sigma)
            );
        }

        log::debug!(
            "TREGO update: iter={}, global_ego_iter = {}, local_trego_iter = {}",
            state.get_iter(),
            state.global_trego_iter,
            state.local_trego_iter
        );

        // Check step success and update trust region
        if state.get_iter() > 0 {
            // Check end global or local phase
            if state.global_trego_iter == self.config.trego_config.n_gl_steps.0 {
                // Adjust trust region wrt global step success
                if sufficient_decrease {
                    let old = state.sigma;
                    new_state.sigma *= 1. / self.config.trego_config.beta;
                    info!(
                        "Previous EGO global step successful: sigma {} -> {}",
                        old, new_state.sigma
                    );
                } else {
                    info!("Previous EGO global step progress fail");
                }
                // Reset cumulative decrease
                new_state.best_decrease = 0.0;
            } else if state.local_trego_iter == self.config.trego_config.n_gl_steps.1 {
                // Adjust trust region wrt local step success
                if sufficient_decrease {
                    let old = state.sigma;
                    new_state.sigma *= 1. / self.config.trego_config.beta;
                    info!(
                        "Previous TREGO local step successful: sigma {} -> {}",
                        old, new_state.sigma
                    );
                } else {
                    let old = state.sigma;
                    new_state.sigma *= self.config.trego_config.beta;
                    info!(
                        "Previous TREGO local step progress fail: sigma {} -> {}",
                        old, new_state.sigma
                    );
                }
                // Reset cumulative decrease
                new_state.best_decrease = 0.0;
            }
        }
        let phase = self.next_phase(sufficient_decrease, &mut new_state);
        (phase, new_state)
    }

    /// Local step where infill criterion is optimized within trust region
    pub fn trego_step<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>> + DomainConstraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
        models: Vec<Box<dyn MixtureGpSurrogate>>,
        infill_data: &InfillObjData<f64>,
    ) -> EgorState<f64> {
        let mut new_state = state.clone();
        let (mut x_data, mut y_data, mut c_data) = new_state.take_data().expect("DOE data");

        let best_index = new_state.best_index.unwrap();
        let y_old = y_data[[best_index, 0]];
        let rho = |sigma| self.config.trego_config.alpha * sigma * sigma;
        let (obj_model, cstr_models) = models.split_first().unwrap();
        let cstr_tols = new_state.cstr_tol.clone();

        let ybest = y_data.row(best_index).to_owned();
        let xbest = x_data.row(best_index).to_owned();
        let cbest = c_data.row(best_index).to_owned();

        let pb = problem.take_problem().unwrap();
        let fcstrs = pb.fn_constraints();
        // Optimize infill criterion
        let activity = new_state.activity.clone();
        let actives = activity.unwrap_or(self.full_activity()).to_owned();

        let mut rng = new_state.take_rng().unwrap();
        let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
        let multistarter = LocalLhsMultiStarter::new(
            self.xlimits.clone(),
            xbest.to_owned(),
            (
                self.config.trego_config.d.0 * new_state.sigma,
                self.config.trego_config.d.1 * new_state.sigma,
            ),
            sub_rng,
        );

        let infill_optpb = InfillOptProblem {
            obj_model: obj_model.as_ref(),
            cstr_models,
            cstr_funcs: fcstrs,
            cstr_tols: &cstr_tols,
            infill_data,
            actives: &actives,
        };

        let (infill_obj, x_opt) = self.optimize_infill_criterion(
            infill_optpb,
            multistarter,
            (xbest.to_owned(), ybest, cbest),
        );

        problem.problem = Some(pb);

        let mut new_state = new_state.infill_value(-infill_obj);
        info!(
            "{} criterion {} max found = {}",
            if self.config.cstr_infill {
                "Constrained infill"
            } else {
                "Infill"
            },
            self.config.infill_criterion.name(),
            state.get_infill_value()
        );

        let x_new = x_opt.insert_axis(Axis(0));
        debug!("x_old={} x_new={}", x_data.row(best_index), x_new.row(0));

        let added = if check_update_ok(&x_data, &x_new) {
            let y_new = self.eval_obj(problem, &x_new);
            debug!(
                "y_old-y_new={}, rho={}",
                y_old - y_new[[0, 0]],
                rho(new_state.sigma)
            );
            let c_new = self.eval_problem_fcstrs(problem, &x_new);

            // Update DOE and best point
            update_data(
                &mut x_data,
                &mut y_data,
                &mut c_data,
                &x_new,
                &y_new,
                &c_new,
            )
        } else {
            Vec::new()
        };

        new_state.prev_added = new_state.added;
        new_state.added += added.len();
        info!("+{} point, total: {} points", added.len(), new_state.added);

        let new_best_index = if added.is_empty() {
            best_index
        } else {
            find_best_result_index_from(
                best_index,
                y_data.nrows() - 1,
                &y_data,
                &c_data,
                &new_state.cstr_tol,
            )
        };
        new_state.feasibility = state.feasibility
            || is_feasible(
                &y_data.row(new_best_index),
                &c_data.row(new_best_index),
                &new_state.cstr_tol,
            );

        new_state = new_state.data((x_data, y_data, c_data)).rng(rng);
        new_state.prev_best_index = new_state.best_index;
        new_state.best_index = Some(new_best_index);
        new_state
    }

    fn next_phase(&self, sufficient_decrease: bool, state: &mut EgorState<f64>) -> Phase {
        let (phase, global_iter, local_iter) = next_phase(
            sufficient_decrease,
            state.global_trego_iter,
            state.local_trego_iter,
            self.config.trego_config.n_gl_steps.0,
            self.config.trego_config.n_gl_steps.1,
        );
        state.global_trego_iter = global_iter;
        state.local_trego_iter = local_iter;
        phase
    }
}

#[derive(PartialEq, Debug, Default)]
pub enum Phase {
    #[default]
    Global,
    Local,
}

#[allow(unused_variables)]
pub fn next_phase(
    sufficient_decrease: bool,
    global_iter: usize,
    local_iter: usize,
    n_global: usize,
    n_local: usize,
) -> (Phase, usize, usize) {
    if global_iter > 0 && global_iter < n_global {
        // Normal global step
        (Phase::Global, global_iter + 1, local_iter)
    } else if global_iter == n_global {
        // End of global phase unless sufficient decrease
        if sufficient_decrease {
            (Phase::Global, 1, 0)
        } else {
            (Phase::Local, 0, 1)
        }
    } else if local_iter > 0 && local_iter < n_local {
        // Normal local step
        (Phase::Local, 0, local_iter + 1)
    } else if local_iter == n_local {
        // End of local phase
        (Phase::Global, 1, 0)
    } else {
        // First iteration
        (Phase::Global, global_iter + 1, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_G: usize = 2;
    const N_L: usize = 2;

    #[test]
    fn test_it1() {
        assert_eq!(next_phase(false, 0, 0, N_G, N_L), (Phase::Global, 1, 0));
    }

    #[test]
    fn test_it2() {
        assert_eq!(next_phase(false, 1, 0, N_G, N_L), (Phase::Global, 2, 0));
    }

    #[test]
    fn test_it3_no_decrease() {
        assert_eq!(next_phase(false, 2, 0, N_G, N_L), (Phase::Local, 0, 1));
    }

    #[test]
    fn test_it4_no_decrease() {
        assert_eq!(next_phase(false, 0, 1, N_G, N_L), (Phase::Local, 0, 2));
    }

    #[test]
    fn test_it5_no_decrease() {
        assert_eq!(next_phase(false, 0, 2, N_G, N_L), (Phase::Global, 1, 0));
    }
    #[test]
    fn test_it6_no_decrease() {
        assert_eq!(next_phase(false, 1, 0, N_G, N_L), (Phase::Global, 2, 0));
    }

    #[test]
    fn test_it7_decrease() {
        assert_eq!(next_phase(true, 2, 0, N_G, N_L), (Phase::Global, 1, 0));
    }

    #[test]
    fn test_it8_no_decrease() {
        assert_eq!(next_phase(false, 1, 0, N_G, N_L), (Phase::Global, 2, 0));
    }

    #[test]
    fn test_it9_no_decrease() {
        assert_eq!(next_phase(false, 2, 0, N_G, N_L), (Phase::Local, 0, 1));
    }
}

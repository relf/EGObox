use crate::CstrFn;
use crate::EgorSolver;
use crate::EgorState;
use crate::InfillObjData;
use crate::SurrogateBuilder;
use crate::solver::solver_infill_optim::InfillOptProblem;
use crate::types::DomainConstraints;
use crate::utils::find_best_result_index_from;
use crate::utils::is_feasible;
use crate::utils::update_data;

use argmin::core::CostFunction;
use argmin::core::Problem;

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
        let rho = |sigma| sigma * sigma;
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
                self.config.trego.d.0 * new_state.sigma,
                self.config.trego.d.1 * new_state.sigma,
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
        debug!(
            "x_old={} x_new={}",
            x_data.row(new_state.best_index.unwrap()),
            x_data.row(best_index)
        );
        let y_new = self.eval_obj(problem, &x_new);
        debug!(
            "y_old-y_new={}, rho={}",
            y_old - y_new[[0, 0]],
            rho(new_state.sigma)
        );

        let c_new = self.eval_problem_fcstrs(problem, &x_new);

        // Update DOE and best point
        let added = update_data(
            &mut x_data,
            &mut y_data,
            &mut c_data,
            &x_new,
            &y_new,
            &c_new,
        );
        new_state.prev_added = new_state.added;
        new_state.added += added.len();
        info!(
            "+{} point(s), total: {} points",
            added.len(),
            new_state.added
        );

        let new_best_index = find_best_result_index_from(
            best_index,
            y_data.nrows() - 1,
            &y_data,
            &c_data,
            &new_state.cstr_tol,
        );
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
}

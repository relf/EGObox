use crate::CstrFn;
use crate::EgorSolver;
use crate::EgorState;
use crate::FailsafeStrategy;
use crate::InfillObjData;
use crate::SurrogateBuilder;
use crate::solver::solver_infill_optim::InfillOptProblem;
use crate::types::Constraints;
use crate::utils::{find_best_result_index_from, is_feasible, is_update_ok};

use basin::{CostFunction, Problem};

use egobox_moe::MixtureGpSurrogate;

use log::{debug, info};
use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_stats::DeviationExt;

use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use serde::{Serialize, de::DeserializeOwned};

use super::multistart::LocalLhsMultiStarter;
use super::surrogate_set::{DataClustering, SurrogateSet};

/// Overrides used to select the candidate point of a local step: the objective model
/// (e.g. a local GP) and the solver, hence the infill criterion (e.g. posterior mean)
pub(crate) struct LocalSearch<'a, SB: SurrogateBuilder, C: CstrFn> {
    /// Objective model used instead of the global one
    pub obj_model: Option<&'a dyn MixtureGpSurrogate>,
    /// Solver used to optimize the infill criterion instead of the current one
    pub solver: Option<&'a EgorSolver<SB, C>>,
}

impl<SB: SurrogateBuilder, C: CstrFn> LocalSearch<'_, SB, C> {
    /// No override: global objective model and configured infill criterion
    pub fn none() -> Self {
        LocalSearch {
            obj_model: None,
            solver: None,
        }
    }
}

/// Indices of the data points used to train a local GP around `xbest`: the points within
/// the trust region of radius `max_dist` (infinity norm) enlarged by the smallest factor
/// `rho >= 1` such that at least `n_min` points are selected (all points when there are fewer)
pub(crate) fn local_training_rows(
    x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    xbest: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    max_dist: f64,
    n_min: usize,
) -> Vec<usize> {
    let ratios = x_data
        .rows()
        .into_iter()
        .map(|x| {
            Zip::from(&x)
                .and(xbest)
                .fold(0., |m: f64, a, b| m.max((a - b).abs()))
                / max_dist
        })
        .collect::<Vec<_>>();
    let mut sorted = ratios.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rho = sorted
        .get(n_min.saturating_sub(1))
        .map_or(f64::INFINITY, |r| r.max(1.));
    ratios
        .iter()
        .enumerate()
        .filter(|(_, r)| **r <= rho)
        .map(|(i, _)| i)
        .collect()
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
    C: CstrFn,
{
    /// Local step where infill criterion is optimized within trust region.
    /// `models` are expected to be trained on the current state data, they
    /// are updated with the evaluated point and persisted in the returned state.
    pub fn trego_step<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = crate::EgoError>
            + Constraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
        models: Vec<Box<dyn MixtureGpSurrogate>>,
        infill_data: &InfillObjData<f64>,
        max_dist: f64,
        min_acceptance_distance: f64,
    ) -> crate::errors::Result<EgorState<f64>> {
        self.trego_step_with(
            problem,
            state,
            models,
            infill_data,
            max_dist,
            min_acceptance_distance,
            LocalSearch::none(),
        )
    }

    /// Same as [`trego_step`](Self::trego_step) where `search` gives the objective model
    /// and the solver used to select the candidate point when they differ from the global ones.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn trego_step_with<
        O: CostFunction<Param = Array2<f64>, Output = Array2<f64>, Error = crate::EgoError>
            + Constraints<C>,
    >(
        &mut self,
        problem: &mut Problem<O>,
        state: EgorState<f64>,
        models: Vec<Box<dyn MixtureGpSurrogate>>,
        infill_data: &InfillObjData<f64>,
        max_dist: f64,
        min_acceptance_distance: f64,
        search: LocalSearch<'_, SB, C>,
    ) -> crate::errors::Result<EgorState<f64>> {
        let mut new_state = state.clone();
        let (mut x_data, mut y_data, mut c_data) = new_state.take_data().expect("DOE data");

        let best_index = new_state.surrogate.best_index.unwrap();
        let y_old = y_data[[best_index, 0]];
        let (obj_models, cstr_models) = models.split_at(self.config.n_obj_models());
        let obj_model = search.obj_model.unwrap_or(obj_models[0].as_ref());
        let cstr_tols = new_state.doe.cstr_tol.clone();

        let ybest = y_data.row(best_index).to_owned();
        let xbest = x_data.row(best_index).to_owned();
        let cbest = c_data.row(best_index).to_owned();

        let pb = problem.inner();
        let fcstrs = pb.constraints();
        let fcstr_specs = pb.constraint_specs();
        let transformed_fcstrs = crate::types::affine_fcstrs(fcstrs, fcstr_specs);
        // Function constraints as given to the infill optimizer in global steps
        let infill_fcstrs = self.infill_fcstrs(&transformed_fcstrs);
        // Optimize infill criterion
        let mut rng = new_state.take_rng().unwrap();
        let sub_rng = Xoshiro256Plus::seed_from_u64(rng.r#gen());
        let multistarter =
            LocalLhsMultiStarter::new(self.xlimits.clone(), xbest.to_owned(), max_dist, sub_rng);

        let viability_model = self.viability_model(&x_data, state.surrogate.x_fail.as_ref());

        let search_solver: &Self = search.solver.unwrap_or(self);
        let infill_optpb = InfillOptProblem::new(
            obj_model,
            cstr_models,
            &infill_fcstrs,
            &cstr_tols,
            viability_model,
            search_solver.config.feasibility_infill.alpha(),
            infill_data,
            &state.coego.activity,
            None,
        );

        let (infill_obj, x_opt) = search_solver.optimize_infill_criterion(
            infill_optpb,
            multistarter,
            (xbest.to_owned(), ybest, cbest),
        );

        let mut new_state = new_state.infill_value(-infill_obj);
        info!(
            "{} criterion {} max found = {}",
            if search_solver.config.cstr_infill {
                "Constrained infill"
            } else {
                "Infill"
            },
            search_solver.config.infill_criterion.name(),
            state.get_infill_value()
        );

        let x_new = x_opt.insert_axis(Axis(0));
        debug!("x_old={} x_new={}", x_data.row(best_index), x_new.row(0));

        let (add_count, valid_count, x_fail_points) = if xbest.l1_dist(&x_new.row(0)).unwrap()
            > min_acceptance_distance
            && is_update_ok(&x_data, &x_new.row(0))
        {
            // Penalized values are used in case of failed evaluation
            let y_penalized =
                (self.config.failsafe_strategy == FailsafeStrategy::Imputation).then(|| {
                    self.compute_penalized_point(&x_new.row(0), obj_models, cstr_models, &y_data)
                        .insert_axis(Axis(0))
                });

            // Update DOE and best point
            let evaluation = self.evaluate_and_append(
                problem,
                &x_new,
                y_penalized.as_ref(),
                (&mut x_data, &mut y_data, &mut c_data),
            )?;
            debug!("y_old-y_new={}", y_old - evaluation.y[[0, 0]]);

            new_state = new_state
                .param(x_new.row(0).to_owned())
                .cost(evaluation.y.row(0).to_owned());
            (
                evaluation.add_count,
                evaluation.valid_count,
                evaluation.x_fail,
            )
        } else {
            (0, 0, None)
        };

        new_state = new_state
            .store_failed_points(x_fail_points.clone())
            .count_added_points(add_count);
        info!(
            "+{} point, total: {} points",
            add_count, new_state.doe.added
        );

        // A failed point with imputed values is never a candidate for the best
        let new_best_index = if valid_count == 0 {
            best_index
        } else {
            find_best_result_index_from(
                best_index,
                y_data.nrows() - valid_count,
                &y_data,
                &c_data,
                &new_state.doe.cstr_tol,
            )
        };

        new_state.feasibility = state.feasibility
            || is_feasible(
                &y_data.row(new_best_index),
                &c_data.row(new_best_index),
                &new_state.doe.cstr_tol,
            );

        new_state.surrogate.models = models;
        if add_count > 0 {
            // Incorporate the evaluated point into the persisted models
            // (fast incremental update unless full retraining is required)
            let mut surrogates = SurrogateSet::take(&mut new_state)?;
            let point_index = state.get_iter() as usize * self.config.qei_config.batch;
            self.refresh_models(
                &mut surrogates,
                &x_data,
                &y_data,
                &state.coego.activity,
                DataClustering::Fixed,
                point_index,
            );
            new_state = surrogates.put(new_state);
        }

        new_state = new_state.data((x_data, y_data, c_data)).rng(rng);
        new_state.surrogate.prev_best_index = new_state.surrogate.best_index;
        new_state.surrogate.best_index = Some(new_best_index);
        Ok(new_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_training_rows() {
        use ndarray::array;
        // infinity norm distances to xbest = [0, 0]: 0, 1, 2, 3, 4, 1.5
        let x_data = array![
            [0., 0.],
            [1., 0.5],
            [-2., 1.],
            [0., 3.],
            [4., -4.],
            [1.5, 1.5]
        ];
        let xbest = array![0., 0.];

        // Trust region already holds enough points (distances <= 2)
        assert_eq!(
            local_training_rows(&x_data, &xbest, 2., 3),
            vec![0, 1, 2, 5]
        );
        // Trust region enlarged until it holds n_min points (distances <= 3)
        assert_eq!(
            local_training_rows(&x_data, &xbest, 1., 5),
            vec![0, 1, 2, 3, 5]
        );
        // Fewer data points than n_min: all points
        assert_eq!(
            local_training_rows(&x_data, &xbest, 1., 10),
            vec![0, 1, 2, 3, 4, 5]
        );
    }
}

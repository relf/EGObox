use crate::EgorSolver;
use crate::errors::Result;
use crate::types::*;
use crate::utils::find_best_result_index_from;

use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, RemoveAxis, s};
use serde::{Serialize, de::DeserializeOwned};

#[cfg(not(feature = "nlopt"))]
use crate::types::ObjFn;
#[cfg(feature = "nlopt")]
use nlopt::ObjFn;

/// Whether GP objective improves when setting a component set
/// TODO: at the moment not sure improvement check is required, to be validated
pub const COEGO_IMPROVEMENT_CHECK: bool = false;
const CSTR_DOUBT: f64 = 3.;

/// Set active components to xcoop using xopt values
/// active may be longer than values
pub(crate) fn set_active_x(xcoop: &mut [f64], active: &[usize], values: &[f64]) {
    std::iter::zip(&active[..values.len()], values).for_each(|(&i, &xi)| xcoop[i] = xi)
}

/// Get active components from given ndarray following given axis
/// active may contain out of range indices meaning it should be ignore
pub(crate) fn get_active_x<A, D>(axis: Axis, xcoop: &Array<A, D>, active: &[usize]) -> Array<A, D>
where
    A: Clone,
    D: RemoveAxis,
{
    let size = xcoop.len_of(axis);
    let selection = active
        .iter()
        .filter(|&&i| i < size)
        .cloned()
        .collect::<Vec<usize>>();
    xcoop.select(axis, &selection)
}

impl<SB, C> EgorSolver<SB, C>
where
    SB: SurrogateBuilder + Serialize + DeserializeOwned,
    C: CstrFn,
{
    /// Used to remove out of range indices from activity last row
    /// Indeed the last row of activity matrix may be incomplete
    /// as n_coop might not be a divider of nx. so this last row
    /// may contain indices with an 'nx value' used as a marker
    /// meaning to discard the indice information while keeping
    /// rows of the same length.
    pub fn strip(active: &[usize], dim: usize) -> Vec<usize> {
        active.iter().filter(|&&i| i < dim).cloned().collect()
    }

    /// Check whether the objective is improved using surrogates at xcoop location
    /// Given a current best information, surrogates are used
    /// to predict objective and constraints at xcoop location
    /// to find which one is best.
    #[allow(clippy::type_complexity)]
    pub(crate) fn is_objective_improved(
        &self,
        current_best: &(Array1<f64>, Array1<f64>, Array1<f64>),
        xcoop: &Array1<f64>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
        cstr_tols: &Array1<f64>,
        cstr_funcs: &[&(dyn ObjFn<InfillObjData<f64>> + Sync)],
    ) -> (bool, (Array1<f64>, Array1<f64>, Array1<f64>)) {
        // Assign metamodelized objective (1) and constraints (n_cstr) values
        let mut y_data = Array2::zeros((2, 1 + self.config.n_cstr));

        //  Assign first row with current
        y_data.slice_mut(s![0, ..]).assign(&current_best.1);
        //  Assign second row with challenger
        y_data
            .slice_mut(s![1, ..])
            .assign(&self.predict_point(xcoop, obj_model, cstr_models).unwrap());

        // Assign function constraints values
        let mut c_data = Array2::zeros((2, cstr_funcs.len()));

        //  Assign first row with current
        c_data.slice_mut(s![0, ..]).assign(&current_best.2);
        //  Assign second row with challenger
        let evals = self.eval_fcstrs(cstr_funcs, &xcoop.to_owned().insert_axis(Axis(0)));
        if !cstr_funcs.is_empty() {
            c_data.slice_mut(s![1, ..]).assign(&evals.row(0));
        }

        // Find the best of two (row0 vs row1)
        let best_index = find_best_result_index_from(0, 1, &y_data, &c_data, cstr_tols);

        let best = if best_index == 0 {
            // current is still the best
            current_best.clone()
        } else {
            // new best
            (
                xcoop.to_owned(),
                y_data.row(1).to_owned(),
                c_data.row(1).to_owned(),
            )
        };
        (best_index != 0, best)
    }

    /// Compute predicted objective and constraints values at the given x location.
    /// An optimistic approach is used as lower bound is taken for the objective
    /// and upper bound for constraints
    pub(crate) fn predict_point(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        obj_model: &dyn MixtureGpSurrogate,
        cstr_models: &[Box<dyn MixtureGpSurrogate>],
    ) -> Result<Array1<f64>> {
        let mut res: Vec<f64> = vec![];
        let x = &x.view().insert_axis(Axis(0));

        let (pred, var) = obj_model.predict_valvar(x)?;
        let sigma = var[0].sqrt();
        // Use lower trust bound for a minimization
        let pred = pred[0] - CSTR_DOUBT * sigma;
        res.push(pred);
        for cstr_model in cstr_models {
            let (pred, var) = cstr_model.predict_valvar(x)?;
            let sigma = var[0].sqrt();
            // Use upper trust bound
            res.push(pred[0] + CSTR_DOUBT * sigma);
        }
        Ok(Array1::from_vec(res))
    }
}
#[cfg(test)]
mod tests {
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    use crate::solver::activity_strategy::ActivityStrategy;
    use crate::solver::activity_strategy::CooperativeActivity;

    #[test]
    fn test_coego_activity_balanced() {
        let dim = 125;
        let ng = 5;
        let strategy = CooperativeActivity::new(ng);
        let mut rng = Xoshiro256Plus::from_entropy();
        let activity = strategy.generate_activity(dim, &mut rng);
        assert_eq!(activity.nrows(), ng);
        let expected_ncols = 25;
        assert_eq!(activity.ncols(), expected_ncols);
        assert!(activity.iter().all(|&v| v < dim))
    }

    #[test]
    fn test_coego_activity() {
        let dim = 123;
        let ng = 5;
        let strategy = CooperativeActivity::new(ng);
        let mut rng = Xoshiro256Plus::from_entropy();
        let activity = strategy.generate_activity(dim, &mut rng);
        assert_eq!(activity.nrows(), ng);
        let expected_ncols = 25;
        assert_eq!(activity.ncols(), expected_ncols);
        assert_eq!(activity[[3, expected_ncols - 1]], dim);
        assert_eq!(activity[[4, expected_ncols - 1]], dim);
        assert!(activity.iter().enumerate().all(|(i, &v)| v < dim
            || i == ng * expected_ncols - 1 // 124
            || i == (ng - 1) * expected_ncols - 1)) // 99
    }
}

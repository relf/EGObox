use crate::EgorSolver;
use crate::types::*;

use ndarray::{Array, Axis, RemoveAxis};
use serde::{Serialize, de::DeserializeOwned};

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

//! ParEGO scalarization of the output data (Knowles 2006)

use super::scalarization::{
    Normalization, augmented_tchebycheff, default_divisions, simplex_lattice,
};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2, s};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::seq::SliceRandom;

/// Simplex lattice vectors of dimension `n_obj` with `n_divisions` divisions
/// (default divisions when `None`) in a random order
pub(crate) fn shuffled_weights<R: Rng>(
    n_obj: usize,
    n_divisions: Option<usize>,
    rng: &mut R,
) -> Vec<Array1<f64>> {
    let mut lattice = simplex_lattice(n_obj, n_divisions.unwrap_or(default_divisions(n_obj)));
    lattice.shuffle(rng);
    lattice
}

/// Number of simplex lattice vectors used as ParEGO weights
pub(crate) fn n_weights(n_obj: usize, n_divisions: Option<usize>) -> usize {
    simplex_lattice(n_obj, n_divisions.unwrap_or(default_divisions(n_obj))).len()
}

/// Training view `[s | cstrs]` of `y_data = [obj_1, ..., obj_n_obj, cstrs]` where `s` is the
/// augmented Tchebycheff aggregation of the objectives normalized with their observed bounds.
pub(crate) fn scalarized_view(
    y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    n_obj: usize,
    weights: &Array1<f64>,
    rho: f64,
) -> Array2<f64> {
    let objs = y_data.slice(s![.., ..n_obj]);
    let finite: Vec<usize> = (0..objs.nrows())
        .filter(|&i| objs.row(i).iter().all(|v| v.is_finite()))
        .collect();
    let normalization = Normalization::from_rows(&objs, &finite);
    let mut view = Array2::zeros((y_data.nrows(), 1 + y_data.ncols() - n_obj));
    for (i, mut row) in view.rows_mut().into_iter().enumerate() {
        row[0] = augmented_tchebycheff(&normalization.apply(&objs.row(i)), weights, rho);
        row.slice_mut(s![1..]).assign(&y_data.slice(s![i, n_obj..]));
    }
    view
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_shuffled_weights() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let weights = shuffled_weights(3, None, &mut rng);
        assert_eq!(weights.len(), n_weights(3, None));
        for (i, w) in weights.iter().enumerate() {
            assert_eq!(w.len(), 3);
            assert_abs_diff_eq!(w.sum(), 1., epsilon = 1e-12);
            // all weight vectors are distinct
            assert!(weights[i + 1..].iter().all(|v| v != w));
        }
    }

    #[test]
    fn test_scalarized_view() {
        // [f1, f2, c]
        let y = array![[0., 2., -1.], [1., 0., 1.]];
        let view = scalarized_view(&y, 2, &array![0.5, 0.5], 0.);
        // normalized f: [0, 1] and [1, 0]
        assert_abs_diff_eq!(view, array![[0.5, -1.], [0.5, 1.]], epsilon = 1e-12);
    }
}

//! Scalarization of multiple objectives (ParEGO, Knowles 2006)

use super::pareto::{ideal_and_nadir, pareto_front_indices};
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, s};

/// Default `rho` coefficient of the augmented Tchebycheff function
pub(crate) const PAREGO_RHO: f64 = 0.05;

/// Affine normalization of objectives to `[0, 1]` given lower and upper bounds
#[derive(Clone, Debug)]
pub(crate) struct Normalization {
    lower: Array1<f64>,
    range: Array1<f64>,
}

impl Normalization {
    /// Normalization with given bounds, degenerated ranges being taken as 1
    pub(crate) fn new(lower: Array1<f64>, upper: Array1<f64>) -> Self {
        let range = (&upper - &lower).mapv(|r| if r > f64::EPSILON { r } else { 1. });
        Normalization { lower, range }
    }

    /// Normalization from the observed bounds of the given rows of `objs`
    pub(crate) fn from_rows(objs: &ArrayBase<impl Data<Elem = f64>, Ix2>, rows: &[usize]) -> Self {
        let (ideal, nadir) = ideal_and_nadir(objs, rows);
        Self::new(ideal, nadir)
    }

    /// Ranges used to normalize the objectives
    pub(crate) fn range(&self) -> &Array1<f64> {
        &self.range
    }

    /// Normalized objectives
    pub(crate) fn apply(&self, f: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> Array1<f64> {
        (f - &self.lower) / &self.range
    }
}

/// Augmented Tchebycheff function: `max_j w_j f_j + rho * sum_j w_j f_j`
pub(crate) fn augmented_tchebycheff(
    f: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    weights: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    rho: f64,
) -> f64 {
    let weighted = f * weights;
    let max = weighted.fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    max + rho * weighted.sum()
}

/// All weight vectors of dimension `n_obj` with components in `{0, 1/s, ..., 1}` summing to 1
pub(crate) fn simplex_lattice(n_obj: usize, s: usize) -> Vec<Array1<f64>> {
    fn rec(remaining: usize, dim: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if dim == 1 {
            current.push(remaining);
            out.push(current.clone());
            current.pop();
        } else {
            for k in 0..=remaining {
                current.push(k);
                rec(remaining - k, dim - 1, current, out);
                current.pop();
            }
        }
    }
    let mut out = vec![];
    rec(s, n_obj, &mut vec![], &mut out);
    out.into_iter()
        .map(|v| Array1::from_iter(v.into_iter().map(|k| k as f64 / s as f64)))
        .collect()
}

/// Default number of divisions of the simplex lattice used by ParEGO
pub(crate) fn default_divisions(n_obj: usize) -> usize {
    match n_obj {
        0..=2 => 10,
        3 => 4,
        _ => 3,
    }
}

/// Index of the compromise point of the data: the point of the constrained Pareto front
/// minimizing the uniform-weight augmented Tchebycheff function of the objectives normalized
/// with the front bounds (ties go to the lowest index).
/// When no point is feasible, it is the point with the smallest constraint violation.
/// `excluded` rows (e.g. failed points with imputed values) are never chosen.
pub(crate) fn compromise_index(
    y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    c_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    n_obj: usize,
    cstr_tol: &Array1<f64>,
    excluded: &[usize],
) -> Option<usize> {
    let front = pareto_front_indices(y_data, c_data, n_obj, cstr_tol, excluded);
    let objs = y_data.slice(s![.., ..n_obj]);
    let normalization = Normalization::from_rows(&objs, &front);
    let weights = Array1::from_elem(n_obj, 1. / n_obj as f64);
    front
        .iter()
        .map(|&i| {
            let f = normalization.apply(&objs.row(i));
            (i, augmented_tchebycheff(&f, &weights, PAREGO_RHO))
        })
        .fold(None, |best: Option<(usize, f64)>, (i, v)| match best {
            Some((_, bv)) if bv <= v => best,
            _ => Some((i, v)),
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    #[test]
    fn test_normalization() {
        let norm = Normalization::new(array![0., 1.], array![2., 1.]);
        assert_abs_diff_eq!(
            norm.apply(&array![1., 3.]),
            array![0.5, 2.],
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_augmented_tchebycheff() {
        let v = augmented_tchebycheff(&array![0.2, 0.6], &array![0.5, 0.5], 0.05);
        assert_abs_diff_eq!(v, 0.3 + 0.05 * 0.4, epsilon = 1e-12);
    }

    #[test]
    fn test_simplex_lattice() {
        let w = simplex_lattice(2, 10);
        assert_eq!(w.len(), 11);
        let w = simplex_lattice(3, 4);
        assert_eq!(w.len(), 15);
        for wi in w {
            assert_abs_diff_eq!(wi.sum(), 1., epsilon = 1e-12);
            assert!(wi.iter().all(|&v| (0. ..=1.).contains(&v)));
        }
    }

    #[test]
    fn test_compromise_index() {
        let y = array![[0., 1.], [0.4, 0.4], [1., 0.], [0.9, 0.9]];
        let c = Array2::zeros((4, 0));
        let tol = Array1::zeros(0);
        assert_eq!(compromise_index(&y, &c, 2, &tol, &[]), Some(1));
    }

    #[test]
    fn test_compromise_index_without_feasible_point() {
        let y = array![[0., 1., 2.], [0.4, 0.4, 0.5], [1., 0., 1.]];
        let c = Array2::zeros((3, 0));
        let tol = array![1e-4];
        assert_eq!(compromise_index(&y, &c, 2, &tol, &[]), Some(1));
    }
}

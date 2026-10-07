//! Pareto dominance utilities (minimization of all objectives)

use crate::utils::{cstr_sum_at, is_feasible_at};
use ndarray::{Array1, ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2, concatenate};

/// Whether `a` dominates `b`: `a` is not worse on any objective and better on at least one.
pub(crate) fn dominates(
    a: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    b: &ArrayBase<impl Data<Elem = f64>, Ix1>,
) -> bool {
    let mut strictly_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Indices of the non-dominated rows of `objs` among the given `candidates` (ascending order).
fn non_dominated_among(
    objs: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    candidates: &[usize],
) -> Vec<usize> {
    candidates
        .iter()
        .copied()
        .filter(|&i| {
            !candidates
                .iter()
                .any(|&j| j != i && dominates(&objs.row(j), &objs.row(i)))
        })
        .collect()
}

/// Indices of the non-dominated rows of `objs` (ascending order).
/// Rows with non finite values are never part of the front.
#[allow(dead_code)] // used by upcoming per-objective strategies
pub(crate) fn non_dominated_indices(objs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Vec<usize> {
    let candidates: Vec<usize> = (0..objs.nrows())
        .filter(|&i| objs.row(i).iter().all(|v| v.is_finite()))
        .collect();
    non_dominated_among(objs, &candidates)
}

/// Indices of the constrained Pareto front of the data (ascending order).
///
/// * `y_data` holds `[obj_1, ..., obj_n_obj, cstr_1, ...]` rows (internal `<= 0` constraints),
/// * `c_data` holds the function constraint values,
/// * `cstr_tol` gives tolerances of the constraints of `y_data` followed by those of `c_data`.
///
/// Feasible points dominate infeasible ones: the front is the set of non-dominated feasible points.
/// When no point is feasible, the front reduces to the point with the smallest constraint violation.
/// Rows with non finite values are never part of the front.
pub(crate) fn pareto_front_indices(
    y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    c_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    n_obj: usize,
    cstr_tol: &Array1<f64>,
) -> Vec<usize> {
    let finite: Vec<usize> = (0..y_data.nrows())
        .filter(|&i| {
            y_data.row(i).iter().all(|v| v.is_finite())
                && c_data.row(i).iter().all(|v| v.is_finite())
        })
        .collect();
    let feasible: Vec<usize> = finite
        .iter()
        .copied()
        .filter(|&i| is_feasible_at(&y_data.row(i), &c_data.row(i), n_obj, cstr_tol))
        .collect();
    if feasible.is_empty() {
        finite
            .iter()
            .copied()
            .map(|i| {
                let yc = concatenate![Axis(0), y_data.row(i), c_data.row(i)];
                (i, cstr_sum_at(&yc, n_obj, cstr_tol))
            })
            .fold(None, |best: Option<(usize, f64)>, (i, v)| match best {
                Some((_, bv)) if bv <= v => best,
                _ => Some((i, v)),
            })
            .map(|(i, _)| vec![i])
            .unwrap_or_default()
    } else {
        let objs = y_data.slice(ndarray::s![.., ..n_obj]);
        non_dominated_among(&objs, &feasible)
    }
}

/// Componentwise minimum (ideal point) and maximum (nadir point) of the given rows of `objs`
pub(crate) fn ideal_and_nadir(
    objs: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    rows: &[usize],
) -> (Array1<f64>, Array1<f64>) {
    let mut ideal = Array1::from_elem(objs.ncols(), f64::INFINITY);
    let mut nadir = Array1::from_elem(objs.ncols(), f64::NEG_INFINITY);
    for &i in rows {
        let row: ArrayView1<f64> = objs.row(i);
        ideal.zip_mut_with(&row, |m, &v| *m = m.min(v));
        nadir.zip_mut_with(&row, |m, &v| *m = m.max(v));
    }
    (ideal, nadir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_dominates() {
        assert!(dominates(&array![1., 2.], &array![2., 2.]));
        assert!(!dominates(&array![1., 2.], &array![1., 2.]));
        assert!(!dominates(&array![1., 3.], &array![2., 2.]));
    }

    #[test]
    fn test_non_dominated_indices() {
        let objs = array![
            [1., 3.],
            [2., 2.],
            [3., 1.],
            [2.5, 2.5],
            [f64::NAN, 0.],
            [3., 3.]
        ];
        assert_eq!(non_dominated_indices(&objs), vec![0, 1, 2]);
    }

    #[test]
    fn test_pareto_front_with_constraints() {
        // [f1, f2, c] with c <= 0 feasible
        let y = array![[1., 3., 0.5], [2., 2., -1.], [3., 1., -1.], [2.5, 2.5, -1.]];
        let c = Array2::zeros((4, 0));
        let tol = array![1e-4];
        assert_eq!(pareto_front_indices(&y, &c, 2, &tol), vec![1, 2]);
    }

    #[test]
    fn test_pareto_front_with_function_constraints() {
        let y = array![[1., 3.], [2., 2.], [3., 1.]];
        let c = array![[-1.], [1.], [-1.]];
        let tol = array![1e-4];
        assert_eq!(pareto_front_indices(&y, &c, 2, &tol), vec![0, 2]);
    }

    #[test]
    fn test_pareto_front_without_feasible_point() {
        let y = array![[1., 3., 0.5], [2., 2., 0.2], [3., 1., 0.3]];
        let c = Array2::zeros((3, 0));
        let tol = array![1e-4];
        assert_eq!(pareto_front_indices(&y, &c, 2, &tol), vec![1]);
    }

    #[test]
    fn test_ideal_and_nadir() {
        let objs = array![[1., 3.], [2., 2.], [3., 1.], [9., 9.]];
        let (ideal, nadir) = ideal_and_nadir(&objs, &[0, 1, 2]);
        assert_eq!(ideal, array![1., 1.]);
        assert_eq!(nadir, array![3., 3.]);
    }
}

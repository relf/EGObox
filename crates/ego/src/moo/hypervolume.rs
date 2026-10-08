//! Hypervolume indicator (minimization): volume of the objective space dominated by a set
//! of points and bounded by a reference point.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

/// Hypervolume of the set of points `front` (one point per row) wrt `ref_point`.
///
/// Points not strictly dominating the reference point do not contribute.
/// The computation is exact: a sweep for 2 objectives, a recursive slicing along the last
/// objective otherwise (fine for the small fronts handled in EGO).
pub(crate) fn hypervolume(
    front: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ref_point: &ArrayBase<impl Data<Elem = f64>, Ix1>,
) -> f64 {
    let points: Vec<Vec<f64>> = front
        .rows()
        .into_iter()
        .filter(|p| p.iter().zip(ref_point.iter()).all(|(v, r)| v < r))
        .map(|p| p.to_vec())
        .collect();
    hv_rec(points, &ref_point.to_vec())
}

fn hv_rec(mut points: Vec<Vec<f64>>, ref_point: &[f64]) -> f64 {
    if points.is_empty() {
        return 0.;
    }
    let m = ref_point.len();
    match m {
        1 => {
            let min = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
            ref_point[0] - min
        }
        2 => {
            points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
            let mut hv = 0.;
            let mut current_f2 = ref_point[1];
            for p in points.iter() {
                if p[1] < current_f2 {
                    hv += (ref_point[0] - p[0]) * (current_f2 - p[1]);
                    current_f2 = p[1];
                }
            }
            hv
        }
        _ => {
            // Slice along the last objective: between two consecutive levels the dominated
            // region is the (m-1)-dimensional hypervolume of the points below the slab
            points.sort_by(|a, b| a[m - 1].total_cmp(&b[m - 1]));
            let mut hv = 0.;
            for i in 0..points.len() {
                let upper = if i + 1 < points.len() {
                    points[i + 1][m - 1]
                } else {
                    ref_point[m - 1]
                };
                let depth = upper - points[i][m - 1];
                if depth > 0. {
                    let projected: Vec<Vec<f64>> =
                        points[..=i].iter().map(|p| p[..m - 1].to_vec()).collect();
                    hv += depth * hv_rec(projected, &ref_point[..m - 1]);
                }
            }
            hv
        }
    }
}

/// Hypervolumes of the constrained Pareto fronts of the first `n_prev_rows` rows of the data
/// and of the whole data, in the objective space normalized with the data bounds and wrt a common
/// reference point (nadir of both fronts + 10 % of their range).
/// Only feasible points count: without feasible point, the hypervolume is zero.
/// See [`crate::moo::pareto::pareto_front_indices`] for the data layout.
pub(crate) fn hypervolume_progress(
    y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    c_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    n_obj: usize,
    cstr_tol: &Array1<f64>,
    n_prev_rows: usize,
) -> (f64, f64) {
    use super::pareto::{ideal_and_nadir, pareto_front_indices};
    use super::scalarization::Normalization;
    use crate::utils::is_feasible_at;
    use ndarray::{Array2, s};

    let objs = y_data.slice(s![.., ..n_obj]);
    let finite: Vec<usize> = (0..objs.nrows())
        .filter(|&i| objs.row(i).iter().all(|v| v.is_finite()))
        .collect();
    let normalization = Normalization::from_rows(&objs, &finite);
    // Feasible Pareto front of the first rows (pareto_front_indices falls back to the least
    // infeasible point when no point is feasible)
    let feasible_front = |n_rows: usize| -> Vec<usize> {
        pareto_front_indices(
            &y_data.slice(s![..n_rows, ..]),
            &c_data.slice(s![..n_rows, ..]),
            n_obj,
            cstr_tol,
        )
        .into_iter()
        .filter(|&i| is_feasible_at(&y_data.row(i), &c_data.row(i), n_obj, cstr_tol))
        .collect()
    };
    let front_now = feasible_front(y_data.nrows());
    if front_now.is_empty() {
        return (0., 0.);
    }
    let front_prev = feasible_front(n_prev_rows);
    let normalized = |rows: &[usize]| {
        let mut front = Array2::zeros((rows.len(), n_obj));
        for (k, &i) in rows.iter().enumerate() {
            front.row_mut(k).assign(&normalization.apply(&objs.row(i)));
        }
        front
    };
    let (front_now, front_prev) = (normalized(&front_now), normalized(&front_prev));
    let both = ndarray::concatenate![ndarray::Axis(0), front_now, front_prev];
    let all: Vec<usize> = (0..both.nrows()).collect();
    let (ideal, nadir) = ideal_and_nadir(&both, &all);
    let ref_point = reference_point(&ideal, &nadir, 0.1);
    if exact_hypervolume_cost(both.nrows(), n_obj) <= MAX_EXACT_HYPERVOLUME_COST {
        (
            hypervolume(&front_prev, &ref_point),
            hypervolume(&front_now, &ref_point),
        )
    } else {
        let hvs = hypervolumes_monte_carlo(&[&front_prev, &front_now], &ideal, &ref_point);
        (hvs[0], hvs[1])
    }
}

/// Max cost of the exact hypervolume computation (see [`exact_hypervolume_cost`])
const MAX_EXACT_HYPERVOLUME_COST: usize = 1 << 22;

/// Number of Monte Carlo samples used to estimate hypervolumes otherwise
const N_HYPERVOLUME_SAMPLES: usize = 1 << 16;

/// Rough cost of the exact recursive hypervolume computation of `size` points with `n_obj`
/// objectives (`size^(n_obj - 1)`)
fn exact_hypervolume_cost(size: usize, n_obj: usize) -> usize {
    size.checked_pow(n_obj.saturating_sub(1) as u32)
        .unwrap_or(usize::MAX)
}

/// Monte Carlo estimates of the hypervolumes of several fronts wrt `ref_point`, using the same
/// uniform samples (fixed seed) of the box `[lower, ref_point]` for all fronts (hence the
/// estimated hypervolume differences have a low variance)
fn hypervolumes_monte_carlo(
    fronts: &[&Array2<f64>],
    lower: &Array1<f64>,
    ref_point: &Array1<f64>,
) -> Vec<f64> {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256Plus;

    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let u = Array2::random_using(
        (N_HYPERVOLUME_SAMPLES, ref_point.len()),
        Uniform::new(0., 1.),
        &mut rng,
    );
    let range = ref_point - lower;
    let samples = &u * &range + lower;
    let volume = range.product();
    fronts
        .iter()
        .map(|front| {
            let dominated = samples
                .rows()
                .into_iter()
                .filter(|x| {
                    front
                        .rows()
                        .into_iter()
                        .any(|p| p.iter().zip(x.iter()).all(|(pj, xj)| pj <= xj))
                })
                .count();
            volume * dominated as f64 / N_HYPERVOLUME_SAMPLES as f64
        })
        .collect()
}

/// Reference point built from the nadir point of the front with a relative `margin`
/// of the front range (the range being taken as 1 when degenerated).
pub(crate) fn reference_point(
    ideal: &Array1<f64>,
    nadir: &Array1<f64>,
    margin: f64,
) -> Array1<f64> {
    let mut ref_point = nadir.to_owned();
    ref_point.zip_mut_with(ideal, |r, &lo| {
        let range = *r - lo;
        let range = if range > f64::EPSILON { range } else { 1. };
        *r += margin * range;
    });
    ref_point
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, Array2, array};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_hv_2d() {
        let front = array![[1., 3.], [2., 2.], [3., 1.]];
        assert_abs_diff_eq!(hypervolume(&front, &array![4., 4.]), 6., epsilon = 1e-12);
        // dominated and outside points do not contribute
        let front = array![[1., 3.], [2., 2.], [3., 1.], [3., 3.], [5., 0.]];
        assert_abs_diff_eq!(hypervolume(&front, &array![4., 4.]), 6., epsilon = 1e-12);
    }

    #[test]
    fn test_hv_3d() {
        let front = array![[0., 0., 0.]];
        assert_abs_diff_eq!(
            hypervolume(&front, &array![1., 1., 1.]),
            1.,
            epsilon = 1e-12
        );
        let front = array![[0., 0., 0.5], [0.5, 0.5, 0.]];
        assert_abs_diff_eq!(
            hypervolume(&front, &array![1., 1., 1.]),
            0.625,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_hv_3d_with_constant_last_objective_matches_2d() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let pts = Array::random_using((20, 2), Uniform::new(0., 1.), &mut rng);
        let hv2 = hypervolume(&pts, &array![1., 1.]);
        let mut pts3 = Array2::zeros((20, 3));
        pts3.slice_mut(ndarray::s![.., ..2]).assign(&pts);
        let hv3 = hypervolume(&pts3, &array![1., 1., 2.]);
        assert_abs_diff_eq!(hv3, 2. * hv2, epsilon = 1e-12);
    }

    #[test]
    fn test_hv_zdt1_front() {
        // ZDT1 front f2 = 1 - sqrt(f1): HV wrt (1, 1) is 2/3
        let n = 1001;
        let front = Array2::from_shape_fn((n, 2), |(i, j)| {
            let f1 = i as f64 / (n - 1) as f64;
            if j == 0 { f1 } else { 1. - f1.sqrt() }
        });
        assert_abs_diff_eq!(
            hypervolume(&front, &array![1., 1.]),
            2. / 3.,
            epsilon = 2e-3
        );
    }

    #[test]
    fn test_hypervolume_progress_counts_feasible_points_only() {
        // [f1, f2, c] with c <= 0 feasible
        let y = array![[0., 0., 1.], [0.5, 0.5, -1.], [0.2, 0.8, -1.]];
        let c = Array2::zeros((3, 0));
        let tol = array![1e-4];
        // no feasible point among the first row
        let (hv_prev, hv_now) = hypervolume_progress(&y, &c, 2, &tol, 1);
        assert_eq!(hv_prev, 0.);
        assert!(hv_now > 0.);
        // no feasible point at all
        let (hv_prev, hv_now) = hypervolume_progress(
            &y.slice(ndarray::s![..1, ..]),
            &c.slice(ndarray::s![..1, ..]),
            2,
            &tol,
            1,
        );
        assert_eq!((hv_prev, hv_now), (0., 0.));
        // no progress
        let (hv_prev, hv_now) = hypervolume_progress(&y, &c, 2, &tol, 3);
        assert_eq!(hv_prev, hv_now);
    }

    #[test]
    fn test_hypervolumes_monte_carlo() {
        let front = array![[1., 3.], [2., 2.], [3., 1.]];
        let hvs = hypervolumes_monte_carlo(&[&front], &array![0., 0.], &array![4., 4.]);
        assert_abs_diff_eq!(hvs[0], 6., epsilon = 0.1);
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let front = Array::random_using((30, 4), Uniform::new(0., 1.), &mut rng);
        let front = front.select(
            ndarray::Axis(0),
            &crate::moo::pareto::non_dominated_indices(&front),
        );
        let ref_point = array![1., 1., 1., 1.];
        let mc = hypervolumes_monte_carlo(&[&front], &array![0., 0., 0., 0.], &ref_point);
        assert_abs_diff_eq!(mc[0], hypervolume(&front, &ref_point), epsilon = 1e-2);
    }

    #[test]
    fn test_reference_point() {
        let r = reference_point(&array![0., 1.], &array![2., 1.], 0.1);
        assert_abs_diff_eq!(r, array![2.2, 1.1], epsilon = 1e-12);
    }
}

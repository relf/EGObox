//! Hypervolume indicator (minimization): volume of the objective space dominated by a set
//! of points and bounded by a reference point.

use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};

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
    fn test_reference_point() {
        let r = reference_point(&array![0., 1.], &array![2., 1.], 0.1);
        assert_abs_diff_eq!(r, array![2.2, 1.1], epsilon = 1e-12);
    }
}

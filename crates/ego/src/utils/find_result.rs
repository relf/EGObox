use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip, concatenate};
use ndarray_stats::QuantileExt;

use crate::utils::sort_axis::*;
use linfa::Float;
use ndarray::{array, s};
use std::iter::zip;

/// Compute sum of constraints above tolerance where y is [obj, cstr1, cstr2, ..., cstrn]
fn cstr_sum<F: Float>(y: &ArrayBase<impl Data<Elem = F>, Ix1>, cstr_tol: &Array1<F>) -> F {
    cstr_sum_at(y, 1, cstr_tol)
}

/// Compute sum of constraints above tolerance where y is [obj_1, ..., obj_n_obj, cstr1, ..., cstrn]
pub(crate) fn cstr_sum_at<F: Float>(
    y: &ArrayBase<impl Data<Elem = F>, Ix1>,
    n_obj: usize,
    cstr_tol: &Array1<F>,
) -> F {
    y.slice(s![n_obj..])
        .iter()
        .enumerate()
        .filter(|&(ref i, &c)| c > cstr_tol[*i])
        .fold(F::zero(), |acc, (i, &c)| acc + (c - cstr_tol[i]).abs())
}

/// This method compare y1 and y2 coming as the second component of a couple (index, y)
/// where y consits in [obj, cstr1, cstr2, ..., cstrn]
pub fn cstr_min<F: Float>(
    (_, y1): (usize, &ArrayBase<impl Data<Elem = F>, Ix1>),
    (_, y2): (usize, &ArrayBase<impl Data<Elem = F>, Ix1>),
    cstr_tol: &Array1<F>,
) -> std::cmp::Ordering {
    if y1.len() > 1 {
        let sum_c1 = cstr_sum(y1, cstr_tol);
        let sum_c2 = cstr_sum(y2, cstr_tol);
        if sum_c1 > F::zero() && sum_c2 > F::zero() {
            sum_c1.partial_cmp(&sum_c2).unwrap()
        } else if sum_c1 == F::zero() && sum_c2 == F::zero() {
            y1[0].partial_cmp(&y2[0]).unwrap()
        } else if sum_c1 == F::zero() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    } else {
        // unconstrained optimization
        y1[0].partial_cmp(&y2[0]).unwrap()
    }
}

/// This method find the best result in ydata wrt cstr_min partial order
/// given the current best index of the current best and the offset index
/// the starting index of the new data to compare  
///
/// This method supposes indexes consistency: current_index < offset_index < ydata.nrows()
pub fn find_best_result_index_from<F: Float>(
    current_index: usize, /* current best index */
    offset_index: usize,
    ydata: &ArrayBase<impl Data<Elem = F>, Ix2>, /* the whole data so far */
    cdata: &ArrayBase<impl Data<Elem = F>, Ix2>, /* the whole function cstrs data so far */
    cstr_tol: &Array1<F>,
) -> usize {
    let alldata = concatenate![Axis(1), ydata.to_owned(), cdata.to_owned()];
    let new_ydata = alldata.slice(s![offset_index.., ..]);

    let best = alldata.row(current_index);
    let min = new_ydata
        .outer_iter()
        .enumerate()
        // Points with non finite values (e.g. failed evaluations) are never the best
        .filter(|(_, y)| y.iter().all(|v| v.is_finite()))
        .fold((usize::MAX, best), |a, b| {
            std::cmp::min_by(a, b, |(i, u), (j, v)| cstr_min((*i, u), (*j, v), cstr_tol))
        });
    match min {
        (usize::MAX, _) => current_index,
        (index, _) => offset_index + index,
    }
}

/// Find best (eg minimal) cost value (y_data\[0\]) with valid constraints, meaning
/// * y_data\[1..\] < cstr_tol
/// * c_data[..] < 0
///
/// y_data containing ns samples [objective, cstr_1, ... cstr_nc] is given as a matrix (ns, nc + 1)
/// c_data containing [fcstr_1, ... fcstr1_nfc] where fcstr_i is the value of function constraints at x_i
pub fn find_best_result_index<F: Float>(
    y_data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    c_data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    cstr_tol: &Array1<F>,
) -> usize {
    if y_data.ncols() > 1 || c_data.ncols() > 0 {
        // Merge metamodelised constraints and function constraints
        let alldata = concatenate![Axis(1), y_data.to_owned(), c_data.to_owned()];
        let alltols = cstr_tol.clone();

        // Compute sum of violated constraints
        let cstrs = &alldata.slice(s![.., 1..]);
        let mut c_obj = Array2::zeros((y_data.nrows(), 2));

        Zip::from(c_obj.rows_mut())
            .and(cstrs.rows())
            .and(alldata.slice(s![.., 0]))
            .for_each(|mut c_obj_row, c_row, obj| {
                if obj.is_finite() && c_row.iter().all(|c| c.is_finite()) {
                    let c_sum = zip(c_row, &alltols)
                        .filter(|(c, ctol)| *c > ctol)
                        .fold(F::zero(), |acc, (c, ctol)| acc + (*c - *ctol).abs());
                    c_obj_row.assign(&array![c_sum, *obj]);
                } else {
                    // Points with non finite values (e.g. failed evaluations) are never the best
                    c_obj_row.fill(F::infinity());
                }
            });
        let min_csum_index = c_obj.slice(s![.., 0]).argmin().ok();

        if let Some(min_index) = min_csum_index {
            if c_obj[[min_index, 0]] > F::zero() {
                // There is no feasible point take minimal cstr sum as the best one
                min_index
            } else {
                // There is at least one or several point feasible, take the minimal objective among them
                let mut index = 0;
                let mut y_best = F::infinity();
                Zip::indexed(c_obj.rows()).for_each(|i, c_o| {
                    if c_o[0] == F::zero() && c_o[1] < y_best {
                        y_best = c_o[1];
                        index = i;
                    }
                });
                index
            }
        } else {
            // Take min obj without looking at constraints
            let mut index = 0;

            // sort regarding minimal objective
            let perm = alldata.sort_axis_by(Axis(0), |i, j| alldata[[i, 0]] < alldata[[j, 0]]);
            let y_sort = alldata.to_owned().permute_axis(Axis(0), &perm);

            // Take the first one which do not violate constraints
            for (i, row) in y_sort.axis_iter(Axis(0)).enumerate() {
                let success =
                    zip(row.slice(s![1..]), &alltols).fold(true, |acc, (c, tol)| acc && c < tol);

                if success {
                    index = i;
                    break;
                }
            }
            perm.indices[index]
        }
    } else {
        // unconstrained optimization, points with non finite values are never the best
        y_data
            .column(0)
            .indexed_iter()
            .filter(|(_, y)| y.is_finite())
            .fold((0, F::infinity()), |(i_min, y_min), (i, &y)| {
                if y < y_min { (i, y) } else { (i_min, y_min) }
            })
            .0
    }
}

/// Indices of the points of the (constrained) Pareto front of the data, all objectives being
/// minimized:
/// * `y_data` containing ns samples `[obj_1, ..., obj_n_obj, cstr_1, ... cstr_nc]` is given as
///   a matrix (ns, n_obj + nc), constraints being expected `<= 0`,
/// * `c_data` contains the function constraint values (ns, nfc),
/// * `cstr_tol` gives the tolerances of the constraints of `y_data` followed by those of `c_data`.
///
/// Feasible points dominate infeasible ones: the front is the set of non-dominated feasible points
/// (in data order). When no point is feasible, the front reduces to the point with the smallest
/// constraint violation. Rows with non finite values are never part of the front.
pub fn find_pareto_front_indices(
    y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    c_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    n_obj: usize,
    cstr_tol: &Array1<f64>,
) -> Vec<usize> {
    crate::moo::pareto::pareto_front_indices(y_data, c_data, n_obj, cstr_tol, &[])
}

/// Index of the compromise point of the data (see [`find_pareto_front_indices`] for the data
/// layout): the point of the (constrained) Pareto front minimizing the uniform-weight augmented
/// Tchebycheff function of the objectives normalized with the front bounds, as returned by
/// [`crate::Egor::run`] with several objectives. When no point is feasible, it is the point with
/// the smallest constraint violation. `None` when there is no point with finite values.
pub fn find_compromise_index(
    y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    c_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    n_obj: usize,
    cstr_tol: &Array1<f64>,
) -> Option<usize> {
    crate::moo::scalarization::compromise_index(y_data, c_data, n_obj, cstr_tol, &[])
}

/// Check if the sum of constraints above tolerance is zero
/// meaning the given point do not violate any constraint
pub fn is_feasible<F: Float>(
    y: &ArrayBase<impl Data<Elem = F>, Ix1>,
    c: &ArrayBase<impl Data<Elem = F>, Ix1>,
    cstr_tol: &Array1<F>,
) -> bool {
    is_feasible_at(y, c, 1, cstr_tol)
}

/// Same as [`is_feasible`] where y is [obj_1, ..., obj_n_obj, cstr1, ..., cstrn]
pub(crate) fn is_feasible_at<F: Float>(
    y: &ArrayBase<impl Data<Elem = F>, Ix1>,
    c: &ArrayBase<impl Data<Elem = F>, Ix1>,
    n_obj: usize,
    cstr_tol: &Array1<F>,
) -> bool {
    let y_c = concatenate![Axis(0), y.to_owned(), c.to_owned()];
    if y_c.len() > n_obj {
        let sum_c = cstr_sum_at(&y_c, n_obj, cstr_tol);
        sum_c == F::zero()
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_cstr_min() {
        // cstr respected, y1 > y2
        let (y1, y2) = (array![1.0, -0.15], array![-1.0, -0.01]);
        let cstr_tol = Array1::from_elem(1, 0.1);
        assert_eq!(
            std::cmp::Ordering::Greater,
            cstr_min((0, &y1), (1, &y2), &cstr_tol)
        );
        // cstr respected, y1 > y2
        let (y1, y2) = (array![-1.0, -0.01], array![-2.0, -0.2]);
        let cstr_tol = Array1::from_elem(1, 0.1);
        assert_eq!(
            std::cmp::Ordering::Greater,
            cstr_min((0, &y1), (1, &y2), &cstr_tol)
        );
        // cstr out of tolerance, y1 < y2
        let (y1, y2) = (array![-2.0, 1.], array![-1.0, 2.]);
        let cstr_tol = Array1::from_elem(1, 0.1);
        assert_eq!(
            std::cmp::Ordering::Less,
            cstr_min((0, &y1), (1, &y2), &cstr_tol)
        );
        // cstr1 out of tolerance, y1 > y2
        let (y1, y2) = (array![-2.0, 1.], array![-1.0, 0.01]);
        let cstr_tol = Array1::from_elem(1, 0.1);
        assert_eq!(
            std::cmp::Ordering::Greater,
            cstr_min((0, &y1), (1, &y2), &cstr_tol)
        );
    }

    #[test]
    fn test_find_best_obj() {
        // respect constraint (0, 1, 2) and minimize obj (1)
        let ydata = array![[1.0, -0.15], [-1.0, -0.01], [2.0, -0.2], [-3.0, 2.0]];
        let cdata = array![[], [], [], []];
        let cstr_tol = Array1::from_elem(4, 0.1);
        assert_abs_diff_eq!(1, find_best_result_index(&ydata, &cdata, &cstr_tol));

        // respect constraint (0, 1, 2) and minimize obj (2)
        let ydata = array![[1.0, -0.15], [-1.0, -0.01], [-2.0, -0.2], [-3.0, 2.0]];
        let cstr_tol = Array1::from_elem(4, 0.1);
        assert_abs_diff_eq!(2, find_best_result_index(&ydata, &cdata, &cstr_tol));

        // all out of tolerance => minimize constraint overshoot sum (0)
        let ydata = array![[1.0, 0.15], [-1.0, 0.3], [2.0, 0.2], [-3.0, 2.0]];
        let cstr_tol = Array1::from_elem(4, 0.1);
        assert_abs_diff_eq!(0, find_best_result_index(&ydata, &cdata, &cstr_tol));

        // all in tolerance => min obj
        let ydata = array![[1.0, 0.15], [-1.0, 0.3], [2.0, 0.2], [-3.0, 2.0]];
        let cstr_tol = Array1::from_elem(4, 3.0);
        assert_abs_diff_eq!(3, find_best_result_index(&ydata, &cdata, &cstr_tol));

        // unconstrained => min obj
        let ydata = array![[1.0], [-1.0], [2.0], [-3.0]];
        let cstr_tol = Array1::from_elem(4, 0.1);
        assert_abs_diff_eq!(3, find_best_result_index(&ydata, &cdata, &cstr_tol));
    }

    #[test]
    fn test_find_best_ignores_non_finite() {
        let cstr_tol = Array1::from_elem(1, 0.1);
        let cdata = array![[], [], [], []];

        // constrained: NaN objective with feasible constraint is not the best
        let ydata = array![[1.0, -0.15], [f64::NAN, -0.2], [-1.0, 2.0], [0.5, -0.1]];
        assert_eq!(3, find_best_result_index(&ydata, &cdata, &cstr_tol));
        assert_eq!(
            3,
            find_best_result_index_from(0, 1, &ydata, &cdata, &cstr_tol)
        );

        // unconstrained: NaN objective is not the best
        let ydata = array![[1.0], [f64::NAN], [-1.0], [f64::NEG_INFINITY]];
        assert_eq!(2, find_best_result_index(&ydata, &cdata, &cstr_tol));
        assert_eq!(
            2,
            find_best_result_index_from(0, 1, &ydata, &cdata, &cstr_tol)
        );
    }

    #[test]
    fn test_find_best_result() {
        let y_data = array![
            [-1.05044744051, -3.854157649246, 0.03068950747],
            [-2.74965213562, -1.955703115787, -0.70939921583],
            [-2.35364705246, 0.322760821911, -31.26001920874],
            [-4.30684045535, -0.188609601161, 0.12375208631],
            [-2.66585377971, -1.665992782883, -3.31489212502],
            [-5.76598597442, 1.767753631322, -0.23219495778],
            [-3.84677718652, -0.164470342807, -0.43935857142],
            [-4.23672675117, -2.343687786724, -0.86266607911],
            [-1.23999705899, -1.653209288978, -12.42363834689],
            [-5.81590801801, -11.725502513342, 2.72175031293],
            [-5.57379997815, -0.075893786744, 0.12260068082],
            [-5.26821022904, -0.093334332384, -0.29931405911],
            [-5.50558228637, -0.008847697249, 0.00015874647],
            [-5.50802373110, -2.479726473358e-5, 2.46930218281e-5],
            [-5.50802210236, -2.586721399788e-5, 2.28386911871e-5],
            [-5.50801726964, 2.607167473023e-7, 5.50684865174e-6],
            [-5.50801509642, 1.951629235996e-7, 2.48275059533e-6],
            [-5.50801399313, -6.707576982734e-8, 1.03991762046e-6]
        ];
        let c_data = Array2::zeros((y_data.nrows(), 0));
        let cstr_tol = Array1::from_vec(vec![1e-6; 2]);
        let index = find_best_result_index(&y_data, &c_data, &cstr_tol);
        assert_eq!(11, index);
        let cstr_tol = Array1::from_vec(vec![2e-6; 2]);
        let index = find_best_result_index(&y_data, &c_data, &cstr_tol);
        assert_eq!(17, index);
    }

    #[test]
    fn test_find_best_result2() {
        let y_data = array![
            [-1.05044744051, -3.854157649246, 0.03068950747],
            [-2.74965213562, -1.955703115787, -0.70939921583],
            [-2.35364705246, 0.322760821911, -31.26001920874],
            [-4.30684045535, -0.188609601161, 0.12375208631],
            [-2.66585377971, -1.665992782883, -3.31489212502],
            [-5.76598597442, 1.767753631322, -0.23219495778],
            [-3.84677718652, -0.164470342807, -0.43935857142],
            [-4.23672675117, -2.343687786724, -0.86266607911],
            [-1.23999705899, -1.653209288978, -12.42363834689],
            [-5.81590801801, -11.725502513342, 2.72175031293],
            [-5.57379997815, -0.075893786744, 0.12260068082],
            [-5.26821022904, -0.093334332384, -0.29931405911],
            [-5.50558228637, -0.008847697249, 0.00015874647],
            [-5.50802373110, -2.479726473358e-5, 2.46930218281e-5],
            [-5.50802210236, -2.586721399788e-5, 2.28386911871e-5],
            [-5.50801726964, 2.607167473023e-7, 5.50684865174e-6],
            [-5.50801509642, 1.951629235996e-7, 2.48275059533e-6],
            [-5.50801399313, -6.707576982734e-8, 1.03991762046e-6]
        ];
        let c_data = Array2::zeros((y_data.nrows(), 0));
        let cstr_tol = Array1::from_vec(vec![2e-6; 2]);
        let index = find_best_result_index_from(11, 12, &y_data, &c_data, &cstr_tol);
        assert_eq!(17, index);
        let cstr_tol = Array1::from_vec(vec![2e-6; 2]);
        let index = find_best_result_index(&y_data, &c_data, &cstr_tol);
        assert_eq!(17, index);
    }

    #[test]
    fn test_find_best_result_with_fcstrs() {
        let y_data = array![
            [-4.381165144133752,],
            [-1.4191462071077778,],
            [-2.471184878434778,],
            [-4.249256421169557,],
            [-5.005839957150093,],
            [-7.,],
            [-4.451820502132815,],
            [-5.297643230694506,],
            [-5.6868527967782185,],
            [-5.527722722951809,],
            [-5.508576806226495,],
            [-5.568458204371678,],
            [-5.508170150760519,]
        ];
        let c_data = array![
            [-0.5623959756729359, 1.970487875],
            [-0.7205065401055234, -27.06126982],
            [-1.624697052074522, -3.56012739],
            [-0.811916938047864, 2.795696422],
            [0.9024970290025135, -1.025385217],
            [-16.000002, 3.999999],
            [-18.271463847657092, 1.45698309],
            [0.8404390363556499, -0.63892895],
            [0.34716763270070256, 0.110383746],
            [0.021788469703452776, 0.0188682703],
            [-0.0007835473211816304, -0.00110163694],
            [-0.15955913666433047, 0.148321297],
            [0.0001603547177302953, 0.000153276785]
        ];
        let cstr_tol = Array1::from_vec(vec![1e-4, 1e-4]);
        let index = find_best_result_index(&y_data, &c_data, &cstr_tol);
        assert_eq!(10, index);
    }

    #[test]
    fn test_find_best_result3() {
        let y_data = array![
            [-4.3811651452, -0.5623939359, 1.97048885964],
            [-1.41914620778, -0.7205045234, -27.061269998],
            [-2.4711848788, -1.624695052, -3.5601263922],
            [-4.24925637625, -0.8119149629, 2.7956976406],
            [-5.0098081271, 0.8985787091, -1.0211260386],
            [-6.9999999994, -15.999999977, 4.0],
            [-4.4538736874, -16.233686686, 1.4661610],
            [-6.22835919555, -16.771640707, 3.22835973],
            [-5.4343175144, 0.9360301001, -0.48674605],
            [-5.5911039556, 0.28048135904, 0.00272429826],
            [-5.5111019931, -0.00015742342304, 0.0046364857],
            [-5.5684582048, -0.15955713047, 0.148322297],
            [-5.5081425079, -0.0002708475389, 0.000249357005],
            [-5.5084961418, 0.000486278266, 0.00048131223],
            [-5.5083185083, 0.00027296342827, 0.00034037966],
            [-5.5083775871, 0.0002591858623, 0.000405809253],
            [-5.5087501669, 0.0007208375406, 0.00074673883],
            [-5.50837384415, 8.713473299e-5, 0.00047661671],
            [-5.5083009626, 2.207449073e-6, 0.000402265087],
            [-5.50841782515, 0.00040617187083, 0.0000902609],
            [-5.5083840184, 0.00037015993495, 0.000360132015],
            [-5.5082258411, -0.0012073930784, 0.00073968081],
            [-5.5289355022, 0.017804389555, 0.022180971]
        ];
        let c_data = Array2::zeros((y_data.nrows(), 0));
        let cstr_tol = Array1::from_vec(vec![2e-2, 1e-2]);
        let index = find_best_result_index(&y_data, &c_data, &cstr_tol);
        assert_eq!(10, index);
    }
}

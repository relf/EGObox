use crate::types::XType;
use egobox_moe::MixtureGpSurrogate;
use libm::erfc;
use log::info;
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip};
use ndarray_stats::{DeviationExt, QuantileExt};
use rayon::prelude::*;
const SQRT_2PI: f64 = 2.5066282746310007;

/// Computes scaling factors used to scale constraint functions values.
pub fn compute_cstr_scales(
    x: &ArrayView2<f64>,
    cstr_models: &[Box<dyn MixtureGpSurrogate>],
) -> Array1<f64> {
    let scales: Vec<f64> = cstr_models
        .par_iter()
        .map(|cstr_model| {
            let preds: Array1<f64> = cstr_model
                .predict(x)
                .unwrap()
                .into_iter()
                .filter(|v| !v.is_infinite()) // filter out infinite values
                .map(|v| v.abs())
                .collect();
            *preds.max().unwrap_or(&1.0)
        })
        .collect();
    Array1::from_shape_vec(cstr_models.len(), scales).unwrap()
}

/// Cumulative distribution function of Standard Normal at x
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Probability density function of Standard Normal at x
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

// DOE handling functions
///////////////////////////////////////////////////////////////////////////////

const MIN_DISTANCE: f64 = 1e-10;

/// Check if new point is not too close to previous ones `x_data`
pub fn is_update_ok(
    x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    x_new: &ArrayBase<impl Data<Elem = f64>, Ix1>,
) -> bool {
    for row in x_data.rows() {
        if row.l1_dist(x_new).unwrap() < MIN_DISTANCE {
            log::info!("Point {} too close to existing data point {}", x_new, row);
            return false;
        }
    }
    true
}

/// Returns the ind of usable points in `x_new` not too close to `x_data` points
pub fn usable_data(
    x_data: &Array2<f64>,
    x_new: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Vec<usize> {
    let mut appended = vec![];
    Zip::indexed(x_new.rows()).for_each(|idx, x| {
        if is_update_ok(x_data, &x) {
            appended.push(idx);
        }
    });
    appended
}

/// Returns the indices of valid (not containing NaN) and invalid rows in `ydata`
pub fn filter_nans(ydata: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> (Vec<usize>, Vec<usize>) {
    let mut valid_idxs = vec![];
    let mut invalid_idxs = vec![];
    for (i, yrow) in ydata.rows().into_iter().enumerate() {
        if yrow.iter().all(|v| !v.is_nan()) {
            valid_idxs.push(i);
        } else {
            invalid_idxs.push(i);
        }
    }
    (valid_idxs, invalid_idxs)
}

/// Append `x_new` (resp. `y_new`, `c_new`) to `x_data` (resp. y_data, resp. c_data)
/// if `y_new` and `c_new` do not contain NaN values
/// Returns the number of added points and the failed points (if any)
pub fn update_data(
    x_data: &mut Array2<f64>,
    y_data: &mut Array2<f64>,
    c_data: &mut Array2<f64>,
    x_new: &Array2<f64>,
    y_new: &Array2<f64>,
    c_new: &Array2<f64>,
    y_penalized: Option<&Array2<f64>>,
) -> (usize, Option<Array2<f64>>) {
    let (valid_idx, invalid_idx) = filter_nans(y_new);

    let mut add_count = 0;
    let x_fail_points = if !invalid_idx.is_empty() {
        info!(
            "{} point(s) resulted in NaN during evaluation",
            invalid_idx.len()
        );
        if let Some(y_pen) = y_penalized {
            // Use penalized values for failed points
            invalid_idx.iter().for_each(|&i| {
                add_count += 1;
                x_data.push_row(x_new.row(i)).unwrap();
                y_data.push_row(y_pen.row(i)).unwrap();
                c_data.push_row(c_new.row(i)).unwrap();
            });
        }
        Some(x_new.select(Axis(0), &invalid_idx).to_owned())
    } else {
        None
    };

    let x_eval_valid = x_new.select(Axis(0), &valid_idx);
    let y_eval_valid = y_new.select(Axis(0), &valid_idx);
    let c_eval_valid = c_new.select(Axis(0), &valid_idx);
    Zip::from(x_eval_valid.rows()).for_each(|x_row| {
        x_data.push_row(x_row.view()).unwrap();
    });
    Zip::from(y_eval_valid.rows()).for_each(|y_row| {
        y_data.push_row(y_row.view()).unwrap();
    });
    Zip::from(c_eval_valid.rows()).for_each(|c_row| {
        c_data.push_row(c_row.view()).unwrap();
    });

    add_count += valid_idx.len();

    (add_count, x_fail_points)
}

pub fn discrete(xtypes: &[XType]) -> bool {
    xtypes
        .iter()
        .any(|t| matches!(t, &XType::Int(_, _) | &XType::Ord(_) | &XType::Enum(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_is_update_ok() {
        let data = array![[0., 1.], [2., 3.]];
        assert!(is_update_ok(&data, &array![3., 4.]));
        assert!(!is_update_ok(&data, &array![1e-15, 1.]));
    }

    #[test]
    fn test_usable_data() {
        let xdata = array![[0., 1.], [2., 3.]];
        assert_eq!(usable_data(&xdata, &array![[3., 4.], [1e-15, 1.]],), &[0]);
    }

    #[test]
    fn test_update_data() {
        let mut xdata = array![[0., 1.], [2., 3.]];
        let mut ydata = array![[3.], [4.]];
        let mut cdata = array![[5.], [6.]];
        assert_eq!(
            update_data(
                &mut xdata,
                &mut ydata,
                &mut cdata,
                &array![[3., 4.], [1e-15, 1.]],
                &array![[6.], [f64::NAN]],
                &array![[8.], [9.]],
                None
            ),
            (1, Some(array![[1e-15, 1.]]))
        );
        assert_eq!(&array![[0., 1.], [2., 3.], [3., 4.]], xdata);
        assert_eq!(&array![[3.], [4.], [6.]], ydata);
        assert_eq!(&array![[5.], [6.], [8.]], cdata);
    }
}

//! Conversions between the mixed-integer space and its continuous relaxation

use crate::xtypes::XType;
use linfa::Float;
use ndarray::{Array, Array2, ArrayBase, Axis, CowArray, Data, DataMut, Ix2, Zip, s};
use ndarray_stats::QuantileExt;

/// Expand xlimits to add continuous dimensions for enumeration x features.
///
/// Each level of an enumerate gives a new continuous dimension in [0, 1].
/// Each integer dimensions are relaxed continuously.
pub fn as_continuous_limits<F: Float>(xtypes: &[XType]) -> Array2<F> {
    let mut xlimits: Vec<F> = vec![];
    let mut dim = 0;
    xtypes.iter().for_each(|xtype| match xtype {
        XType::Float(lower, upper) => {
            dim += 1;
            xlimits.extend([F::cast(*lower), F::cast(*upper)]);
        }
        XType::Int(lower, upper) => {
            dim += 1;
            xlimits.extend([F::cast(*lower), F::cast(*upper)]);
        }
        XType::Ord(v) => {
            dim += 1;
            xlimits.extend([
                (v.iter()
                    .fold(F::infinity(), |a, &b| F::cast(a).min(F::cast(b)))),
                (v.iter()
                    .fold(-F::infinity(), |a, &b| F::cast(a).max(F::cast(b)))),
            ]);
        }
        XType::Enum(v) => {
            dim += v;
            (1..=*v).for_each(|_| {
                xlimits.extend([F::zero(), F::one()]);
            })
        }
    });
    Array::from_shape_vec((dim, 2), xlimits).unwrap()
}

/// Reduce categorical inputs from discrete unfolded space to
/// initial x dimension space where categorical x dimensions are valued by the index
/// in the corresponding enumerate list.
///
/// For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
/// the input x may contain the mask [..., 0, 0, 1, ...] which will be contracted in [..., 2, ...]
/// meaning the "green" value.
/// This function is the opposite of unfold_with_enum_mask().
pub(crate) fn fold_with_enum_index<F: Float>(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array2<F> {
    let mut xfold = Array::zeros((x.nrows(), xtypes.len()));
    let mut unfold_index = 0;
    Zip::indexed(xfold.columns_mut()).for_each(|j, mut col| match &xtypes[j] {
        XType::Float(_, _) | XType::Int(_, _) | XType::Ord(_) => {
            col.assign(&x.column(unfold_index));
            unfold_index += 1;
        }
        XType::Enum(v) => {
            let xenum = x.slice(s![.., unfold_index..unfold_index + v]);
            let argmaxx = xenum.map_axis(Axis(1), |row| F::cast(row.argmax().unwrap()));
            col.assign(&argmaxx);
            unfold_index += v;
        }
    });
    xfold
}

/// Compute dimension when all variables are continuously relaxed
pub(super) fn compute_continuous_dim(xtypes: &[XType]) -> usize {
    xtypes
        .iter()
        .map(|s| match s {
            XType::Enum(v) => *v,
            _ => 1,
        })
        .reduce(|acc, l| -> usize { acc + l })
        .unwrap()
}

/// Expand categorical inputs from initial x dimension space where categorical x dimensions
/// are valued by the index in the corresponding enumerate list to the discrete unfolded space.
///
/// For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
/// the input x may contain [..., 2, ...] which will be expanded in [..., 0, 0, 1, ...].
/// This function is the opposite of fold_with_enum_index().
pub(crate) fn unfold_with_enum_mask(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let mut xunfold = Array::zeros((x.nrows(), compute_continuous_dim(xtypes)));
    let mut unfold_index = 0;
    xtypes.iter().enumerate().for_each(|(i, s)| match s {
        XType::Float(_, _) | XType::Int(_, _) | XType::Ord(_) => {
            xunfold.column_mut(unfold_index).assign(&x.column(i));
            unfold_index += 1;
        }
        XType::Enum(v) => {
            let mut unfold = Array::zeros((x.nrows(), *v));
            Zip::from(unfold.rows_mut())
                .and(x.rows())
                .for_each(|mut row, xrow| {
                    let index = xrow[i] as usize;
                    row[index] = 1.;
                });
            xunfold
                .slice_mut(s![.., unfold_index..unfold_index + v])
                .assign(&unfold);
            unfold_index += v;
        }
    });
    xunfold
}

/// Continuous relaxation of x given possibly discrete types
/// Alias of `unfold_with_enum_mask`
pub fn to_continuous_space(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    unfold_with_enum_mask(xtypes, x)
}

/// Find closest value to `val` in given slice `v`.
fn take_closest<F: Float>(v: &[F], val: F) -> F {
    let idx = Array::from_vec(v.to_vec())
        .map(|refval| (val - *refval).abs())
        .argmin()
        .unwrap();
    v[idx]
}

/// Project continuously relaxed values to their closer assessable values.
///
/// See cast_to_discrete_values
pub(super) fn cast_to_discrete_values_mut<F: Float>(
    xtypes: &[XType],
    x: &mut ArrayBase<impl DataMut<Elem = F>, Ix2>,
) {
    let mut xcol = 0;
    xtypes.iter().for_each(|s| match s {
        XType::Float(_, _) => xcol += 1,
        XType::Int(_, _) => {
            let xround = x.column(xcol).mapv(|v| v.round()).to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        XType::Ord(v) => {
            let vals: Vec<F> = v.iter().map(|&v| F::cast(v)).collect();
            let xround = x
                .column(xcol)
                .mapv(|val| take_closest(&vals, val))
                .to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        XType::Enum(v) => {
            let mut xenum = x.slice_mut(s![.., xcol..xcol + *v]);
            let argmaxx = xenum.map_axis(Axis(1), |row| row.argmax().unwrap());
            Zip::from(xenum.rows_mut())
                .and(&argmaxx)
                .for_each(|mut row, &m| {
                    let mut xcast = Array::zeros(*v);
                    xcast[m] = F::one();
                    row.assign(&xcast);
                });
            xcol += *v;
        }
    });
}

/// Cast/unfold points from the mixed-integer ("public") space into the
/// continuous, discrete-cast space the inner `moe` surrogate model is actually
/// trained/updated on (unfolding enum dimensions when working in folded
/// space, then rounding/snapping to the nearest assessable discrete value).
pub(super) fn cast_to_model_space<'a>(
    xtypes: Option<&[XType]>,
    work_in_folded_space: bool,
    x: CowArray<'a, f64, Ix2>,
) -> CowArray<'a, f64, Ix2> {
    if let Some(xtypes) = xtypes {
        let mut xcast = if work_in_folded_space {
            CowArray::from(unfold_with_enum_mask(xtypes, &x.view()))
        } else {
            x
        };
        cast_to_discrete_values_mut(xtypes, &mut xcast);
        CowArray::from(xcast)
    } else {
        x
    }
}

/// Project continuously relaxed values to their closer assessable values.
///
/// Note: categorical (or enum) x dimensions are still expanded that is
/// there are still as many columns as categorical possible values for the given x dimension.
/// For instance, if an input dimension is typed ["blue", "red", "green"] in xlimits a sample/row of
/// the input x may contain the values (or mask) [..., 0, 0, 1, ...] to specify "green" for
/// this original dimension.
pub(crate) fn cast_to_discrete_values(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let mut xcast = x.to_owned();
    cast_to_discrete_values_mut(xtypes, &mut xcast);
    xcast
}

/// Convenient method to pass from continuous unfolded space to discrete folded space
pub fn to_discrete_space(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let x = cast_to_discrete_values(xtypes, x);
    fold_with_enum_index(xtypes, &x)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::xtypes::XType;
    use approx::assert_abs_diff_eq;

    use ndarray::array;

    #[test]
    fn test_fold_unfold_with_successive_enums() {
        let xtypes = vec![
            XType::Float(-5., 5.),
            XType::Enum(3),
            XType::Enum(2),
            XType::Ord(vec![0., 2., 3.]),
        ];
        let xfold = array![[1.5, 2., 0., 3.], [-2., 0., 1., 0.]];
        let xunfold = array![[1.5, 0., 0., 1., 1., 0., 3.], [-2., 1., 0., 0., 0., 1., 0.]];
        assert_abs_diff_eq!(xunfold, to_continuous_space(&xtypes, &xfold));
        assert_abs_diff_eq!(xfold, to_discrete_space(&xtypes, &xunfold));
    }
}

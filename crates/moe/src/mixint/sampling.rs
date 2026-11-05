//! Sampling of mixed-integer spaces

use crate::xtypes::XType;
use linfa::Float;
use ndarray::{Array2, ArrayBase, Data, Ix2, Zip, s};
use std::marker::PhantomData;

use super::space::{cast_to_discrete_values_mut, fold_with_enum_index};
use serde::{Deserialize, Serialize};

/// Index of the level in `0..n` given a normalized value `u` in [0, 1]
/// such that each level gets an equal share of [0, 1].
fn level_index<F: Float>(u: F, n: usize) -> usize {
    let k = (u * F::cast(n)).floor().to_usize().unwrap_or(0);
    k.min(n - 1)
}

/// Map normalized samples in [0, 1] (in continuous unfolded space) to values
/// of the discrete unfolded space such that each discrete level (integer, ordered
/// value or enum level) has the same probability to be sampled.
///
/// Contrary to rounding samples scaled to the variable bounds which favors
/// interior values (bounds only get half an interval), each level is given
/// a stratum of equal width in [0, 1].
/// For enum variables, only the first column of the unfolded block is used to pick the level.
fn normalized_to_discrete<F: Float>(
    xtypes: &[XType],
    u: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array2<F> {
    let mut x = Array2::zeros(u.raw_dim());
    let mut xcol = 0;
    xtypes.iter().for_each(|s| match s {
        XType::Float(lb, ub) => {
            let (lb, ub) = (F::cast(*lb), F::cast(*ub));
            let col = u.column(xcol).mapv(|v| lb + v * (ub - lb));
            x.column_mut(xcol).assign(&col);
            xcol += 1;
        }
        XType::Int(lb, ub) => {
            let n = (ub - lb + 1).max(1) as usize;
            let lb = F::cast(*lb);
            let col = u.column(xcol).mapv(|v| lb + F::cast(level_index(v, n)));
            x.column_mut(xcol).assign(&col);
            xcol += 1;
        }
        XType::Ord(v) => {
            let mut vals: Vec<F> = v.iter().map(|&v| F::cast(v)).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let col = u
                .column(xcol)
                .mapv(|val| vals[level_index(val, vals.len())]);
            x.column_mut(xcol).assign(&col);
            xcol += 1;
        }
        XType::Enum(v) => {
            Zip::from(x.slice_mut(s![.., xcol..xcol + *v]).rows_mut())
                .and(u.column(xcol))
                .for_each(|mut row, &val| row[level_index(val, *v)] = F::one());
            xcol += *v;
        }
    });
    x
}

/// A decorator of LHS sampling that takes into account XType specifications
/// casting continuous LHS result from floats to discrete types.
#[derive(Serialize, Deserialize)]
pub struct MixintSampling<F: Float, S: egobox_doe::SamplingMethod<F>> {
    /// The continuous sampling method
    pub(super) method: S,
    /// The input specifications (None means all continuous variables)
    pub(super) xtypes: Option<Vec<XType>>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    pub(super) output_in_folded_space: bool,
    /// How continuous samples are mapped to discrete levels
    #[serde(default)]
    pub(super) level_mapping: LevelMapping,
    pub(super) phantom: PhantomData<F>,
}

/// Mapping of continuous samples to discrete levels (Int, Ord, Enum)
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(super) enum LevelMapping {
    /// [0, 1] cut into equal slices, one per level, so that each level has
    /// the same probability to be sampled (random, LHS)
    #[default]
    EqualSlices,
    /// Value scaled to bounds then snapped to the nearest level (full factorial grids)
    Nearest,
}

impl<F: Float, S: egobox_doe::SamplingMethod<F>> MixintSampling<F, S> {
    /// Constructor using `xtypes` specifications
    pub fn new(method: S, xtypes: Vec<XType>) -> Self {
        MixintSampling {
            method,
            xtypes: Some(xtypes),
            output_in_folded_space: false,
            level_mapping: LevelMapping::EqualSlices,
            phantom: PhantomData,
        }
    }

    /// Creates a new sampling method for continuous-only variables
    pub fn new_continuous(method: S) -> Self {
        MixintSampling {
            method,
            xtypes: None,
            output_in_folded_space: false,
            level_mapping: LevelMapping::EqualSlices,
            phantom: PhantomData,
        }
    }

    /// Sets whether we want to work in folded space
    /// If set, sampling data will be provided in folded space
    pub fn work_in_folded_space(&mut self, output_in_folded_space: bool) -> &mut Self {
        self.output_in_folded_space = output_in_folded_space;
        self
    }
}

impl<F: Float, S: egobox_doe::SamplingMethod<F>> egobox_doe::SamplingMethod<F>
    for MixintSampling<F, S>
{
    fn sampling_space(&self) -> &Array2<F> {
        self.method.sampling_space()
    }

    fn normalized_sample(&self, ns: usize) -> Array2<F> {
        self.method.normalized_sample(ns)
    }

    fn sample(&self, ns: usize) -> Array2<F> {
        // Only apply discrete mapping if xtypes is specified
        if let Some(xtypes) = &self.xtypes {
            let doe = match self.level_mapping {
                LevelMapping::EqualSlices => {
                    normalized_to_discrete(xtypes, &self.method.normalized_sample(ns))
                }
                LevelMapping::Nearest => {
                    let mut doe = self.method.sample(ns);
                    cast_to_discrete_values_mut(xtypes, &mut doe);
                    doe
                }
            };
            if self.output_in_folded_space {
                return fold_with_enum_index(xtypes, &doe.view());
            }
            doe
        } else {
            self.method.sample(ns)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixint::*;

    use crate::xtypes::XType;
    use approx::assert_abs_diff_eq;
    use egobox_doe::{LhsKind, SamplingMethod};

    use ndarray::{Array1, ArrayView1, Axis, array};

    #[test]
    fn test_mixint_lhs() {
        let xtypes = vec![
            XType::Float(-10.0, 10.0),
            XType::Enum(3),
            XType::Int(-10, 10),
            XType::Ord(vec![1., 3., 5., 8.]),
        ];

        let mixi = MixintContext::new(&xtypes);
        let mixi_lhs = mixi.create_lhs_sampling(LhsKind::default(), Some(0));

        let actual = mixi_lhs.sample(10);
        let expected = array![
            [-4.049003815966328, 2.0, -1.0, 1.0],
            [-3.3764166379738008, 0.0, 10.0, 8.0],
            [4.132857767184872, 0.0, 1.0, 1.0],
            [7.302048772024065, 0.0, 4.0, 8.0],
            [-7.614543694046457, 0.0, -8.0, 5.0],
            [0.028865479407640393, 1.0, 8.0, 3.0],
            [-1.4943993567665679, 2.0, -5.0, 8.0],
            [-8.291614427265058, 2.0, 5.0, 3.0],
            [9.712890742138065, 1.0, -4.0, 5.0],
            [3.392359215362074, 2.0, -10.0, 1.0]
        ];
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    fn level_counts(col: ArrayView1<f64>, levels: &[f64]) -> Vec<usize> {
        levels
            .iter()
            .map(|l| col.iter().filter(|&&v| v == *l).count())
            .collect()
    }

    fn fairness_xtypes() -> Vec<XType> {
        vec![
            XType::Int(-1, 1),
            XType::Ord(vec![10., 1., 2.]),
            XType::Enum(3),
            XType::Float(0., 1.),
        ]
    }

    #[test]
    fn test_mixint_random_fairness() {
        let xtypes = fairness_xtypes();
        let ns = 30000;
        let doe = MixintContext::new(&xtypes)
            .create_rand_sampling(Some(42))
            .sample(ns);
        for (j, levels) in [
            (0, vec![-1., 0., 1.]),
            (1, vec![1., 2., 10.]),
            (2, vec![0., 1., 2.]),
        ] {
            let counts = level_counts(doe.column(j), &levels);
            assert_eq!(counts.iter().sum::<usize>(), ns);
            for c in counts {
                assert_abs_diff_eq!(c as f64 / ns as f64, 1. / 3., epsilon = 0.02);
            }
        }
    }

    #[test]
    fn test_mixint_lhs_fairness() {
        let xtypes = fairness_xtypes();
        for kind in [
            LhsKind::Classic,
            LhsKind::Centered,
            LhsKind::Maximin,
            LhsKind::CenteredMaximin,
            LhsKind::Optimized,
        ] {
            let doe = MixintContext::new(&xtypes)
                .create_lhs_sampling(kind, Some(0))
                .sample(6);
            for (j, levels) in [
                (0, vec![-1., 0., 1.]),
                (1, vec![1., 2., 10.]),
                (2, vec![0., 1., 2.]),
            ] {
                assert_eq!(level_counts(doe.column(j), &levels), vec![2, 2, 2]);
            }
        }
    }

    #[test]
    fn test_mixint_lhs_unfolded() {
        let xtypes = fairness_xtypes();
        let mixi = MixintContext::new(&xtypes);
        let folded = mixi
            .create_lhs_sampling::<f64>(LhsKind::Classic, Some(0))
            .sample(9);
        let unfolded = mixi
            .create_lhs_sampling::<f64>(LhsKind::Classic, Some(0))
            .work_in_folded_space(false)
            .sample(9);
        assert_eq!(unfolded.ncols(), 6);
        assert_abs_diff_eq!(
            unfolded.slice(s![.., 2..5]).sum_axis(Axis(1)),
            Array1::ones(9)
        );
        assert_abs_diff_eq!(folded, fold_with_enum_index(&xtypes, &unfolded));
    }

    #[test]
    fn test_mixint_ffact() {
        let xtypes = vec![XType::Float(-10.0, 10.0), XType::Int(-10, 10)];

        let mixi = MixintContext::new(&xtypes);
        let mixi_ff = mixi.create_ffact_sampling();

        let actual = mixi_ff.sample(16);
        let expected = array![
            [-10.0, -10.0],
            [-10.0, -3.0],
            [-10.0, 3.0],
            [-10.0, 10.0],
            [-3.333333333333334, -10.0],
            [-3.333333333333334, -3.0],
            [-3.333333333333334, 3.0],
            [-3.333333333333334, 10.0],
            [3.333333333333332, -10.0],
            [3.333333333333332, -3.0],
            [3.333333333333332, 3.0],
            [3.333333333333332, 10.0],
            [10.0, -10.0],
            [10.0, -3.0],
            [10.0, 3.0],
            [10.0, 10.0]
        ];
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_mixint_random() {
        let xtypes = vec![
            XType::Float(-10.0, 10.0),
            XType::Enum(3),
            XType::Int(-10, 10),
            XType::Ord(vec![1., 3., 5., 8.]),
        ];

        let mixi = MixintContext::new(&xtypes);
        let mixi_rand = mixi.create_rand_sampling(Some(0));

        let actual = mixi_rand.sample(10);
        let expected = array![
            [7.08385572734942, 0.0, -5.0, 1.0],
            [3.923592153620703, 0.0, -3.0, 3.0],
            [1.6925857875746217, 0.0, 10.0, 3.0],
            [-0.12628232178356846, 0.0, 5.0, 3.0],
            [-4.333973977708889, 1.0, -8.0, 3.0],
            [-0.31189887669548, 1.0, 2.0, 8.0],
            [5.274476356096036, 1.0, -5.0, 3.0],
            [0.21749742902273717, 0.0, 6.0, 3.0],
            [6.267468405479235, 2.0, -4.0, 8.0],
            [-5.444093848698666, 2.0, 7.0, 8.0]
        ];
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }
}

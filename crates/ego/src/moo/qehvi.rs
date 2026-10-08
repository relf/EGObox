//! Batch Expected Hypervolume Improvement criterion (qEHVI) selecting the points of a batch
//! sequentially (greedy), see \[[Daulton2020](crate#Daulton2020)\].
//!
//! The point `x` added to the `k` points already selected in the batch maximizes the expected
//! hypervolume improvement of the batch over the Pareto front, i.e. the expected improvement
//! brought by `x` over the front augmented with the selected points, under the joint posterior
//! of the objective surrogates at the selected points and `x` (the selected points objective
//! values being unknown). The expectation is estimated by Monte Carlo with fixed base samples
//! (common random numbers), which makes the criterion a deterministic function of `x`.
//!
//! With `y_x` and `y_t` (`t` in `S`, the selected points) a joint sample of the objectives,
//! the hypervolume improvement of `y_x` over the region not dominated by the front nor by the
//! `y_t` points is computed by inclusion–exclusion on a box decomposition `B` of the region not
//! dominated by the front:
//! `sum_{T ⊆ S} (-1)^|T| sum_{[l, u] ∈ B} prod_j (u_j - max(l_j, y_xj, max_{t ∈ T} y_tj))^+`.

use super::criterion::normalized_front;
use super::ehvi::{BoxDecomposition, bounded_front};
use super::scalarization::Normalization;
use egobox_moe::{MixtureGpSurrogate, MoeError, XType};
use ndarray::{Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, Data, Ix2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256Plus;

/// Number of Monte Carlo samples of the joint posterior
pub(crate) const QEHVI_N_SAMPLES: usize = 128;

/// Max batch size handled by qEHVI (the inclusion–exclusion has `2^(batch - 1)` terms)
pub(crate) const MAX_QEHVI_BATCH: usize = 4;

/// Max number of points used to compute the criterion scaling (Monte Carlo evaluations)
pub(crate) const QEHVI_SCALING_POINTS: usize = 50;

/// Max work of a criterion evaluation: samples times subsets times boxes times objectives
const MAX_QEHVI_EVALUATION_WORK: usize = 1 << 22;

/// Relative step of the central finite differences used for the gradient (continuous variables)
const FD_STEP: f64 = 1e-6;

/// Finite difference scheme of a dimension of the continuous relaxed space, discrete values
/// being snapped by the mixed-integer surrogates before prediction (a small step would give
/// null derivatives)
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FdStep {
    /// Continuous variable: central differences with a relative step
    Continuous,
    /// Integer variable in `[lower, upper]` (values rounded)
    Int(f64, f64),
    /// Ordinal variable given its sorted levels (values snapped to the closest level)
    Ord(Vec<f64>),
    /// One-hot dimension of an enum variable in `[0, 1]` (the largest dimension being the level)
    OneHot,
}

impl FdStep {
    /// Points `(backward, forward)` of the finite difference along the dimension at `x`
    /// (`None` if the variable can not vary): for discrete variables, the adjacent valid levels
    /// of the level of `x` (the level itself at a domain bound)
    fn points(&self, x: f64) -> Option<(f64, f64)> {
        let (backward, forward) = match self {
            FdStep::Continuous => {
                let h = FD_STEP * (1. + x.abs());
                (x - h, x + h)
            }
            FdStep::Int(lower, upper) => {
                let level = x.round().clamp(*lower, *upper);
                ((level - 1.).max(*lower), (level + 1.).min(*upper))
            }
            FdStep::Ord(levels) => {
                let k = (0..levels.len())
                    .min_by(|&a, &b| (levels[a] - x).abs().total_cmp(&(levels[b] - x).abs()))?;
                (
                    levels[k.saturating_sub(1)],
                    levels[(k + 1).min(levels.len() - 1)],
                )
            }
            FdStep::OneHot => (0., 1.),
        };
        (forward > backward).then_some((backward, forward))
    }
}

/// Finite difference schemes of the dimensions of the continuous relaxed space of `xtypes`
pub(crate) fn fd_steps(xtypes: &[XType]) -> Vec<FdStep> {
    let mut steps = vec![];
    for xtype in xtypes {
        match xtype {
            XType::Float(_, _) => steps.push(FdStep::Continuous),
            XType::Int(lower, upper) => steps.push(FdStep::Int(*lower as f64, *upper as f64)),
            XType::Ord(values) => {
                let mut levels = values.clone();
                levels.sort_by(f64::total_cmp);
                levels.dedup();
                steps.push(FdStep::Ord(levels));
            }
            XType::Enum(n) => steps.extend(std::iter::repeat_n(FdStep::OneHot, *n)),
        }
    }
    steps
}

/// Batch Expected Hypervolume Improvement criterion of the objective surrogates wrt a Pareto
/// front, given the points already selected in the batch.
///
/// Objectives are normalized with their observed bounds, the criterion is to be maximized.
pub(crate) struct QEhviCriterion<'a> {
    obj_models: &'a [Box<dyn MixtureGpSurrogate>],
    normalization: Normalization,
    /// Lower bounds of the boxes of the region not dominated by the front (one row per box)
    lower: Array2<f64>,
    /// Upper bounds of the boxes of the region not dominated by the front (one row per box)
    upper: Array2<f64>,
    /// Points already selected in the batch (one row per point)
    selected: Array2<f64>,
    /// Finite difference schemes per dimension (see [`fd_steps`], continuous if missing)
    fd_steps: Vec<FdStep>,
    /// Standard normal base samples: (sample, objective, point) with the selected points first
    base_samples: Array3<f64>,
    front_size: usize,
}

impl<'a> QEhviCriterion<'a> {
    /// qEHVI criterion given the objective surrogates, the objective values `objs` of the data
    /// (used to normalize the objectives), the rows of `objs` forming the Pareto front, the points
    /// already `selected` in the batch, the finite difference schemes (see [`fd_steps`]) and the
    /// `seed` of the base samples.
    ///
    /// Fails when the surrogates can not predict the posterior covariance at the selected points.
    pub(crate) fn new(
        obj_models: &'a [Box<dyn MixtureGpSurrogate>],
        objs: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        front_rows: &[usize],
        selected: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        fd_steps: Vec<FdStep>,
        seed: u64,
    ) -> Result<Self, MoeError> {
        Self::with_samples(
            obj_models,
            objs,
            front_rows,
            selected,
            fd_steps,
            seed,
            QEHVI_N_SAMPLES,
        )
    }

    /// qEHVI criterion estimated with `n_samples` Monte Carlo samples
    pub(crate) fn with_samples(
        obj_models: &'a [Box<dyn MixtureGpSurrogate>],
        objs: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        front_rows: &[usize],
        selected: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        fd_steps: Vec<FdStep>,
        seed: u64,
        n_samples: usize,
    ) -> Result<Self, MoeError> {
        // the criterion vanishes where predictions fail: check the joint posterior is available
        if selected.nrows() > 0 {
            for model in obj_models {
                model.predict_covariance(&selected.view())?;
            }
        }
        let n_obj = objs.ncols();
        let n_selected = selected.nrows();
        let (normalization, front, ref_point) = normalized_front(objs, front_rows);
        let front_size = front.nrows();
        let front = bounded_front(
            front,
            n_samples << n_selected,
            MAX_QEHVI_EVALUATION_WORK,
            "qEHVI",
        );
        let (lower, upper) = BoxDecomposition::new(&front, &ref_point).box_bounds();
        let mut rng = Xoshiro256Plus::seed_from_u64(seed);
        let base_samples =
            Array3::random_using((n_samples, n_obj, n_selected + 1), StandardNormal, &mut rng);
        Ok(QEhviCriterion {
            obj_models,
            normalization,
            lower,
            upper,
            selected: selected.to_owned(),
            fd_steps,
            base_samples,
            front_size,
        })
    }

    /// Criterion value at `x`
    pub(crate) fn value(&self, x: &[f64]) -> f64 {
        let Some(samples) = self.joint_samples(x) else {
            return 0.;
        };
        let sum: f64 = samples
            .outer_iter()
            .map(|ys| marginal_hvi(&self.lower, &self.upper, &ys.t()))
            .sum();
        sum / samples.len_of(ndarray::Axis(0)) as f64
    }

    /// Criterion value and gradient (central finite differences) at `x`: for discrete variables,
    /// slope between the adjacent valid levels (one-sided at a domain bound)
    pub(crate) fn value_grad(&self, x: &[f64]) -> (f64, Array1<f64>) {
        let value = self.value(x);
        let mut grad = Array1::zeros(x.len());
        let mut xh = x.to_vec();
        for i in 0..x.len() {
            let step = self.fd_steps.get(i).unwrap_or(&FdStep::Continuous);
            if let Some((backward, forward)) = step.points(x[i]) {
                xh[i] = forward;
                let f_forward = self.value(&xh);
                xh[i] = backward;
                let f_backward = self.value(&xh);
                xh[i] = x[i];
                grad[i] = (f_forward - f_backward) / (forward - backward);
            }
        }
        (value, grad)
    }

    /// Number of points of the Pareto front
    pub(crate) fn front_size(&self) -> usize {
        self.front_size
    }

    /// Samples of the joint posterior of the normalized objectives at the selected points and `x`:
    /// (sample, objective, point) with `x` last, `None` if the surrogates can not predict
    fn joint_samples(&self, x: &[f64]) -> Option<Array3<f64>> {
        let n_selected = self.selected.nrows();
        let n_obj = self.obj_models.len();
        let mut points = Array2::zeros((n_selected + 1, x.len()));
        points
            .slice_mut(s![..n_selected, ..])
            .assign(&self.selected);
        points.row_mut(n_selected).assign(&ArrayView1::from(x));

        let mut means = Array2::zeros((n_selected + 1, n_obj));
        let mut factors = vec![];
        for (j, model) in self.obj_models.iter().enumerate() {
            let range = self.normalization.range()[j];
            means
                .column_mut(j)
                .assign(&model.predict(&points.view()).ok()?);
            let cov = model.predict_covariance(&points.view()).ok()? / (range * range);
            factors.push(cholesky_psd(&cov));
        }
        for mut row in means.rows_mut() {
            let normalized = self.normalization.apply(&row);
            row.assign(&normalized);
        }

        let mut samples = Array3::zeros(self.base_samples.dim());
        for (z, mut y) in self.base_samples.outer_iter().zip(samples.outer_iter_mut()) {
            for (j, factor) in factors.iter().enumerate() {
                y.row_mut(j)
                    .assign(&(&means.column(j) + &factor.dot(&z.row(j))));
            }
        }
        Some(samples)
    }
}

/// Lower Cholesky factor `L` of a symmetric positive semi-definite matrix `a` (`L L^T = a`):
/// degenerated directions (null pivots, e.g. perfectly correlated points) get null columns
fn cholesky_psd(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let max_diag = a.diag().iter().fold(0., |m: f64, v| m.max(*v));
    let tol = 1e-10 * max_diag;
    let mut l = Array2::zeros((n, n));
    for j in 0..n {
        let pivot = a[[j, j]] - (0..j).map(|k| l[[j, k]] * l[[j, k]]).sum::<f64>();
        if pivot <= tol {
            continue;
        }
        let ljj = pivot.sqrt();
        l[[j, j]] = ljj;
        for i in j + 1..n {
            l[[i, j]] = (a[[i, j]] - (0..j).map(|k| l[[i, k]] * l[[j, k]]).sum::<f64>()) / ljj;
        }
    }
    l
}

/// Hypervolume improvement of the last row of `points` over the region given by the boxes
/// `[lower, upper]` not dominated by the other rows of `points`, by inclusion–exclusion
fn marginal_hvi(lower: &Array2<f64>, upper: &Array2<f64>, points: &ArrayView2<f64>) -> f64 {
    let n_others = points.nrows() - 1;
    let x = points.row(n_others);
    let mut corner = x.to_vec();
    let mut hvi = 0.;
    for subset in 0..1usize << n_others {
        // corner of the region dominated by x and by the points of the subset
        corner.iter_mut().zip(x.iter()).for_each(|(c, v)| *c = *v);
        for t in (0..n_others).filter(|t| subset & (1 << t) != 0) {
            corner
                .iter_mut()
                .zip(points.row(t).iter())
                .for_each(|(c, v)| *c = c.max(*v));
        }
        let volume = dominated_volume(lower, upper, &corner);
        if subset == 0 && volume <= 0. {
            // x improves nothing: neither do the intersections
            return 0.;
        }
        if subset.count_ones() % 2 == 0 {
            hvi += volume;
        } else {
            hvi -= volume;
        }
    }
    hvi.max(0.)
}

/// Volume of the part of the boxes `[lower, upper]` dominated by `corner`
fn dominated_volume(lower: &Array2<f64>, upper: &Array2<f64>, corner: &[f64]) -> f64 {
    let mut volume = 0.;
    for (lo, up) in lower.rows().into_iter().zip(upper.rows()) {
        let mut v = 1.;
        for j in 0..corner.len() {
            let side = up[j] - lo[j].max(corner[j]);
            if side <= 0. {
                v = 0.;
                break;
            }
            v *= side;
        }
        volume += v;
    }
    volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moo::ehvi::EhviCriterion;
    use crate::moo::ehvi::tests::models;
    use crate::moo::hypervolume::hypervolume;
    use crate::moo::pareto::non_dominated_indices;
    use approx::assert_abs_diff_eq;
    use ndarray::{Axis, array, concatenate};
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_marginal_hvi_vs_hypervolume() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        for n_obj in [2, 3] {
            for n_others in 0..=3 {
                for _ in 0..20 {
                    let front = Array2::random_using((6, n_obj), Uniform::new(0., 1.), &mut rng);
                    let ref_point = Array1::from_elem(n_obj, 1.1);
                    let points = Array2::random_using(
                        (n_others + 1, n_obj),
                        Uniform::new(-0.1, 1.2),
                        &mut rng,
                    );
                    let (lower, upper) = BoxDecomposition::new(&front, &ref_point).box_bounds();
                    let hvi = marginal_hvi(&lower, &upper, &points.view());
                    let others =
                        concatenate![Axis(0), front.view(), points.slice(s![..n_others, ..])];
                    let expected = hypervolume(&concatenate![Axis(0), others, points], &ref_point)
                        - hypervolume(&others, &ref_point);
                    assert_abs_diff_eq!(hvi, expected, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cholesky_psd() {
        let a = array![[4., 2., 2.], [2., 2., 1.], [2., 1., 1.]];
        let l = cholesky_psd(&a);
        assert_abs_diff_eq!(l.dot(&l.t()), a, epsilon = 1e-12);
        // rank deficient: duplicated point
        let a = array![[2., 1., 2.], [1., 3., 1.], [2., 1., 2.]];
        let l = cholesky_psd(&a);
        assert_abs_diff_eq!(l.dot(&l.t()), a, epsilon = 1e-12);
    }

    #[test]
    fn test_qehvi_without_selected_is_ehvi() {
        for n_obj in [2, 3] {
            let (models, objs) = models(n_obj);
            let front_rows = non_dominated_indices(&objs);
            let ehvi = EhviCriterion::new(&models, &objs, &front_rows);
            let none = Array2::<f64>::zeros((0, 1));
            let qehvi =
                QEhviCriterion::with_samples(&models, &objs, &front_rows, &none, vec![], 0, 20_000)
                    .unwrap();
            for x in [0.1, 0.4, 0.62] {
                let expected = ehvi.value(&[x]);
                let v = qehvi.value(&[x]);
                assert_abs_diff_eq!(v, expected, epsilon = 3e-2 * expected + 1e-5);
            }
        }
    }

    #[test]
    fn test_qehvi_selected_points() {
        let (models, objs) = models(2);
        let front_rows = non_dominated_indices(&objs);
        let ehvi = EhviCriterion::new(&models, &objs, &front_rows);
        let selected = array![[0.4]];
        let qehvi =
            QEhviCriterion::with_samples(&models, &objs, &front_rows, &selected, vec![], 0, 4096)
                .unwrap();
        // no improvement over a selected point (perfectly correlated)
        assert!(ehvi.value(&[0.4]) > 1e-4);
        assert_abs_diff_eq!(qehvi.value(&[0.4]), 0., epsilon = 1e-12);
        // reduced improvement in the vicinity of a selected point
        assert!(qehvi.value(&[0.42]) < 0.5 * ehvi.value(&[0.42]));
        // improvement nearly unchanged far from the selected point
        let none = Array2::<f64>::zeros((0, 1));
        let alone =
            QEhviCriterion::with_samples(&models, &objs, &front_rows, &none, vec![], 0, 4096)
                .unwrap();
        for far in [0.1, 0.7] {
            let expected = alone.value(&[far]);
            assert_abs_diff_eq!(qehvi.value(&[far]), expected, epsilon = 5e-2 * expected);
        }
    }

    #[test]
    fn test_qehvi_gradients() {
        let (models, objs) = models(2);
        let front_rows = non_dominated_indices(&objs);
        let selected = array![[0.3], [0.85]];
        let qehvi =
            QEhviCriterion::new(&models, &objs, &front_rows, &selected, vec![], 42).unwrap();
        for x in [0.1, 0.55, 0.62] {
            let (v, g) = qehvi.value_grad(&[x]);
            assert_eq!(v, qehvi.value(&[x]));
            let h = 1e-4;
            let fd = (qehvi.value(&[x + h]) - qehvi.value(&[x - h])) / (2. * h);
            assert_abs_diff_eq!(g[0], fd, epsilon = 1e-2 * (1. + fd.abs()));
        }
    }

    #[test]
    fn test_fd_steps() {
        let xtypes = [
            XType::Float(0., 1.),
            XType::Int(0, 5),
            XType::Ord(vec![10., 1., 5.]),
            XType::Enum(3),
        ];
        assert_eq!(
            fd_steps(&xtypes),
            vec![
                FdStep::Continuous,
                FdStep::Int(0., 5.),
                FdStep::Ord(vec![1., 5., 10.]),
                FdStep::OneHot,
                FdStep::OneHot,
                FdStep::OneHot
            ]
        );
    }

    #[test]
    fn test_fd_points_within_domain() {
        let int = FdStep::Int(0., 9.);
        assert_eq!(int.points(4.3), Some((3., 5.)));
        assert_eq!(int.points(0.), Some((0., 1.)));
        assert_eq!(int.points(-0.4), Some((0., 1.)));
        assert_eq!(int.points(9.), Some((8., 9.)));
        assert_eq!(FdStep::Int(2., 2.).points(2.), None);
        let ord = FdStep::Ord(vec![1., 5., 10.]);
        assert_eq!(ord.points(4.), Some((1., 10.)));
        assert_eq!(ord.points(1.2), Some((1., 5.)));
        assert_eq!(ord.points(10.), Some((5., 10.)));
        assert_eq!(FdStep::Ord(vec![3.]).points(3.), None);
        assert_eq!(FdStep::OneHot.points(0.3), Some((0., 1.)));
        let (b, f) = FdStep::Continuous.points(2.).unwrap();
        assert!(b < 2. && f > 2. && f - b < 1e-4);
    }

    #[test]
    fn test_qehvi_discrete_gradients() {
        use egobox_moe::{MixintContext, MoeBuilder};
        use linfa::Dataset;
        // integer variable in [0, 10]
        let xtypes = [XType::Int(0, 10)];
        let mixi = MixintContext::new(&xtypes);
        let xt: Array2<f64> = array![[0.], [3.], [5.], [8.], [10.]];
        let fs = [
            xt.column(0).mapv(|v| v / 10.),
            xt.column(0).mapv(|v| 1. - (v / 10.).sqrt() + 0.1 * v.sin()),
        ];
        let mut models: Vec<Box<dyn MixtureGpSurrogate>> = vec![];
        let mut objs = Array2::zeros((xt.nrows(), 0));
        for f in fs.iter() {
            let ds = Dataset::new(xt.clone(), f.clone());
            let model = mixi
                .create_surrogate(&MoeBuilder::new(), &ds)
                .expect("Mixint surrogate");
            models.push(Box::new(model));
            objs = concatenate![Axis(1), objs, f.clone().insert_axis(Axis(0)).t()];
        }
        let front_rows = non_dominated_indices(&objs);
        let selected = array![[2.]];
        let qehvi =
            QEhviCriterion::new(&models, &objs, &front_rows, &selected, fd_steps(&xtypes), 0)
                .unwrap();
        let continuous =
            QEhviCriterion::new(&models, &objs, &front_rows, &selected, vec![], 0).unwrap();
        let x = [6.2];
        // small steps do not cross levels: null derivative
        assert_eq!(continuous.value_grad(&x).1[0], 0.);
        // slope between adjacent levels
        let expected = (qehvi.value(&[7.]) - qehvi.value(&[5.])) / 2.;
        assert!(expected != 0.);
        assert_abs_diff_eq!(qehvi.value_grad(&x).1[0], expected, epsilon = 1e-12);
        // one-sided slope at the domain bounds (no out-of-domain level)
        let expected = qehvi.value(&[1.]) - qehvi.value(&[0.]);
        assert_abs_diff_eq!(qehvi.value_grad(&[0.]).1[0], expected, epsilon = 1e-12);
        let expected = qehvi.value(&[10.]) - qehvi.value(&[9.]);
        assert_abs_diff_eq!(qehvi.value_grad(&[10.]).1[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn test_qehvi_requires_covariance() {
        use egobox_moe::{GpMixture, NbClusters};
        use linfa::{Dataset, traits::Fit};
        let (_, objs) = models(2);
        let xt = Array2::random_using(
            (20, 1),
            Uniform::new(0., 1.),
            &mut Xoshiro256Plus::seed_from_u64(0),
        );
        let mut models: Vec<Box<dyn MixtureGpSurrogate>> = vec![];
        for j in 0..2 {
            let yt = xt.column(0).mapv(|v| (v + j as f64 * 0.5).sin() * v);
            let gp = GpMixture::params()
                .n_clusters(NbClusters::fixed(2))
                .with_rng(Xoshiro256Plus::seed_from_u64(0))
                .fit(&Dataset::new(xt.clone(), yt))
                .unwrap();
            models.push(Box::new(gp));
        }
        let front_rows = non_dominated_indices(&objs);
        let selected = array![[0.3]];
        assert!(QEhviCriterion::new(&models, &objs, &front_rows, &selected, vec![], 0).is_err());
    }

    #[test]
    fn test_qehvi_determinism() {
        let (models, objs) = models(3);
        let front_rows = non_dominated_indices(&objs);
        let selected = array![[0.3]];
        let a = QEhviCriterion::new(&models, &objs, &front_rows, &selected, vec![], 7).unwrap();
        let b = QEhviCriterion::new(&models, &objs, &front_rows, &selected, vec![], 7).unwrap();
        let c = QEhviCriterion::new(&models, &objs, &front_rows, &selected, vec![], 8).unwrap();
        let x = [0.62];
        assert_eq!(a.value(&x), b.value(&x));
        assert_ne!(a.value(&x), c.value(&x));
    }
}

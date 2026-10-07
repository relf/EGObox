//! Expected Improvement Matrix infill criterion (Zhan et al. 2017)
//!
//! Zhan, D., Cheng, Y., & Liu, J. (2017). Expected improvement matrix-based infill criteria
//! for expensive multiobjective optimization. IEEE Transactions on Evolutionary Computation,
//! 21(6), 956–975.

use super::EimAggregation;
use super::hypervolume::reference_point;
use super::pareto::ideal_and_nadir;
use super::scalarization::Normalization;
use crate::utils::{norm_cdf, norm_pdf};
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, Array2, ArrayBase, ArrayView, Data, Ix2};

/// Expected improvement of a normal variable `N(mu, sigma^2)` below `fmin`
/// and its derivatives wrt `mu` and `sigma`
fn ei_and_derivatives(fmin: f64, mu: f64, sigma: f64) -> (f64, f64, f64) {
    if sigma < f64::EPSILON {
        let ei = (fmin - mu).max(0.);
        let dmu = if fmin > mu { -1. } else { 0. };
        (ei, dmu, 0.)
    } else {
        let u = (fmin - mu) / sigma;
        let (cdf, pdf) = (norm_cdf(u), norm_pdf(u));
        ((fmin - mu) * cdf + sigma * pdf, -cdf, pdf)
    }
}

/// Expected Improvement Matrix criterion of the objective surrogates wrt a Pareto front.
///
/// Objectives are normalized with their observed bounds, the criterion is to be maximized.
pub(crate) struct EimCriterion<'a> {
    obj_models: &'a [Box<dyn MixtureGpSurrogate>],
    /// Normalized Pareto front (one point per row)
    front: Array2<f64>,
    /// Normalized reference point (hypervolume aggregation)
    ref_point: Array1<f64>,
    normalization: Normalization,
    aggregation: EimAggregation,
}

impl<'a> EimCriterion<'a> {
    /// EIM criterion given the objective surrogates, the objective values `objs` of the data
    /// (used to normalize the objectives) and the rows of `objs` forming the Pareto front
    pub(crate) fn new(
        obj_models: &'a [Box<dyn MixtureGpSurrogate>],
        objs: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        front_rows: &[usize],
        aggregation: EimAggregation,
    ) -> Self {
        let finite: Vec<usize> = (0..objs.nrows())
            .filter(|&i| objs.row(i).iter().all(|v| v.is_finite()))
            .collect();
        let normalization = Normalization::from_rows(objs, &finite);
        let mut front = Array2::zeros((front_rows.len(), objs.ncols()));
        for (k, &i) in front_rows.iter().enumerate() {
            front.row_mut(k).assign(&normalization.apply(&objs.row(i)));
        }
        let all: Vec<usize> = (0..front.nrows()).collect();
        let (ideal, nadir) = ideal_and_nadir(&front, &all);
        let ref_point = reference_point(&ideal, &nadir, 0.1);
        EimCriterion {
            obj_models,
            front,
            ref_point,
            normalization,
            aggregation,
        }
    }

    /// Criterion value at `x`
    pub(crate) fn value(&self, x: &[f64]) -> f64 {
        self.eval(x, false).0
    }

    /// Criterion value and gradient at `x`
    pub(crate) fn value_grad(&self, x: &[f64]) -> (f64, Array1<f64>) {
        self.eval(x, true)
    }

    fn eval(&self, x: &[f64], with_grad: bool) -> (f64, Array1<f64>) {
        let nx = x.len();
        let n_obj = self.obj_models.len();
        let pt = ArrayView::from_shape((1, nx), x).unwrap();
        // Normalized predictions (mean, std) and their gradients
        let mut mu = Array1::zeros(n_obj);
        let mut sigma = Array1::zeros(n_obj);
        let mut dmu = Array2::zeros((n_obj, nx));
        let mut dsigma = Array2::zeros((n_obj, nx));
        for (j, model) in self.obj_models.iter().enumerate() {
            let range = self.normalization.range()[j];
            let Ok((p, v)) = model.predict_valvar(&pt) else {
                return (0., Array1::zeros(nx));
            };
            let std = v[0].max(0.).sqrt();
            mu[j] = p[0];
            sigma[j] = std / range;
            if with_grad {
                let Ok((dp, dv)) = model.predict_valvar_gradients(&pt) else {
                    return (0., Array1::zeros(nx));
                };
                dmu.row_mut(j).assign(&(&dp.row(0) / range));
                if std > f64::EPSILON {
                    dsigma.row_mut(j).assign(&(&dv.row(0) / (2. * std * range)));
                }
            }
        }
        let mu = self.normalization.apply(&mu);

        let mut best = (f64::INFINITY, Array1::zeros(nx));
        for f in self.front.rows() {
            // Expected improvements over the front point and their gradients
            let mut ei = Array1::zeros(n_obj);
            let mut dei = Array2::zeros((n_obj, nx));
            for j in 0..n_obj {
                let (e, de_dmu, de_dsigma) = ei_and_derivatives(f[j], mu[j], sigma[j]);
                ei[j] = e;
                if with_grad {
                    dei.row_mut(j)
                        .assign(&(&dmu.row(j) * de_dmu + &dsigma.row(j) * de_dsigma));
                }
            }
            let (val, grad) = match self.aggregation {
                EimAggregation::Euclidean => {
                    let norm = ei.dot(&ei).sqrt();
                    let grad = if norm > f64::EPSILON {
                        ei.dot(&dei) / norm
                    } else {
                        Array1::zeros(nx)
                    };
                    (norm, grad)
                }
                EimAggregation::Maximin => {
                    let jmax = (0..n_obj).fold(0, |jm, j| if ei[j] > ei[jm] { j } else { jm });
                    (ei[jmax], dei.row(jmax).to_owned())
                }
                EimAggregation::Hypervolume => {
                    let a = &self.ref_point - &f;
                    let improved = &a + &ei;
                    let val = improved.product() - a.product();
                    let mut grad = Array1::zeros(nx);
                    for j in 0..n_obj {
                        let others: f64 = (0..n_obj)
                            .filter(|&k| k != j)
                            .map(|k| improved[k])
                            .product();
                        grad = grad + &dei.row(j) * others;
                    }
                    (val, grad)
                }
            };
            if val < best.0 {
                best = (val, grad);
            }
        }
        if best.0.is_finite() {
            best
        } else {
            (0., Array1::zeros(nx))
        }
    }

    /// Number of points of the Pareto front
    pub(crate) fn front_size(&self) -> usize {
        self.front.nrows()
    }

    /// Scaling factor of the criterion: max of the criterion over the given points
    /// (1 if the criterion vanishes)
    pub(crate) fn scaling(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> f64 {
        let max = x
            .rows()
            .into_iter()
            .map(|xi| self.value(&xi.to_vec()))
            .filter(|v| v.is_finite())
            .fold(0., f64::max);
        if max < 100. * f64::EPSILON { 1. } else { max }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_moe::{GpMixture, NbClusters};
    use linfa::{Dataset, traits::Fit};
    use ndarray::{Axis, array, concatenate};

    fn models() -> (Vec<Box<dyn MixtureGpSurrogate>>, Array2<f64>) {
        let xt: Array2<f64> = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let f1 = xt.column(0).mapv(|v| v + 0.2 * (5. * v).sin());
        let f2 = xt.column(0).mapv(|v| 1. - v.sqrt() + 0.3 * (6. * v).sin());
        let mut models: Vec<Box<dyn MixtureGpSurrogate>> = vec![];
        for y in [&f1, &f2] {
            let gp = GpMixture::params()
                .n_clusters(NbClusters::fixed(1))
                .fit(&Dataset::new(xt.clone(), y.clone()))
                .unwrap();
            models.push(Box::new(gp));
        }
        let objs = concatenate![Axis(1), f1.insert_axis(Axis(1)), f2.insert_axis(Axis(1))];
        (models, objs)
    }

    #[test]
    fn test_ei_and_derivatives() {
        let (ei, dmu, dsigma) = ei_and_derivatives(0., 0., 1.);
        assert_abs_diff_eq!(ei, norm_pdf(0.), epsilon = 1e-12);
        assert_abs_diff_eq!(dmu, -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(dsigma, norm_pdf(0.), epsilon = 1e-12);
        let (ei, _, _) = ei_and_derivatives(1., 0., 0.);
        assert_abs_diff_eq!(ei, 1., epsilon = 1e-12);
    }

    #[test]
    fn test_eim_gradients() {
        let (models, objs) = models();
        let front: Vec<usize> = (0..objs.nrows()).collect();
        for aggregation in [
            EimAggregation::Euclidean,
            EimAggregation::Maximin,
            EimAggregation::Hypervolume,
        ] {
            let eim = EimCriterion::new(&models, &objs, &front, aggregation);
            for x in [0.1, 0.4, 0.62, 0.9] {
                let (v, g) = eim.value_grad(&[x]);
                assert!(v >= 0., "{aggregation:?} EIM({x}) = {v}");
                let h = 1e-6;
                let fd = (eim.value(&[x + h]) - eim.value(&[x - h])) / (2. * h);
                assert_abs_diff_eq!(g[0], fd, epsilon = 1e-4 * (1. + fd.abs()));
            }
        }
    }

    #[test]
    fn test_eim_vanishes_at_front_points() {
        let (models, objs) = models();
        let front: Vec<usize> = (0..objs.nrows()).collect();
        let eim = EimCriterion::new(&models, &objs, &front, EimAggregation::Euclidean);
        assert_eq!(eim.front_size(), objs.nrows());
        // at a training point, the predictions are exact: no improvement over itself
        assert_abs_diff_eq!(eim.value(&[0.5]), 0., epsilon = 1e-4);
        // between training points there is some expected improvement
        assert!(eim.value(&[0.62]) > 1e-6);
    }
}

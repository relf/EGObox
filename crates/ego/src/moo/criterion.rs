//! Multi-objective infill criteria shared building blocks

use super::ehvi::EhviCriterion;
use super::eim::EimCriterion;
use super::hypervolume::reference_point;
use super::pareto::ideal_and_nadir;
use super::scalarization::Normalization;
use crate::utils::{norm_cdf, norm_pdf};
use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, Array2, ArrayBase, ArrayView, Data, Ix2};

/// Expected improvement of a normal variable `N(mu, sigma^2)` below `fmin`
/// and its derivatives wrt `mu` and `sigma`
pub(crate) fn ei_and_derivatives(fmin: f64, mu: f64, sigma: f64) -> (f64, f64, f64) {
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

/// Normalized predictions of the objective surrogates at a point and their gradients
pub(crate) struct Predictions {
    pub mu: Array1<f64>,
    pub sigma: Array1<f64>,
    /// Gradients of `mu` (one row per objective), zeros when not requested
    pub dmu: Array2<f64>,
    /// Gradients of `sigma` (one row per objective), zeros when not requested
    pub dsigma: Array2<f64>,
}

/// Predictions of the objective surrogates at `x` in the normalized objective space
pub(crate) fn predict_normalized(
    obj_models: &[Box<dyn MixtureGpSurrogate>],
    normalization: &Normalization,
    x: &[f64],
    with_grad: bool,
) -> Option<Predictions> {
    let nx = x.len();
    let n_obj = obj_models.len();
    let pt = ArrayView::from_shape((1, nx), x).unwrap();
    let mut mu = Array1::zeros(n_obj);
    let mut sigma = Array1::zeros(n_obj);
    let mut dmu = Array2::zeros((n_obj, nx));
    let mut dsigma = Array2::zeros((n_obj, nx));
    for (j, model) in obj_models.iter().enumerate() {
        let range = normalization.range()[j];
        let (p, v) = model.predict_valvar(&pt).ok()?;
        let std = v[0].max(0.).sqrt();
        mu[j] = p[0];
        sigma[j] = std / range;
        if with_grad {
            let (dp, dv) = model.predict_valvar_gradients(&pt).ok()?;
            dmu.row_mut(j).assign(&(&dp.row(0) / range));
            if std > f64::EPSILON {
                dsigma.row_mut(j).assign(&(&dv.row(0) / (2. * std * range)));
            }
        }
    }
    Some(Predictions {
        mu: normalization.apply(&mu),
        sigma,
        dmu,
        dsigma,
    })
}

/// Normalization of the objectives with the bounds of the data `objs`, normalized Pareto front
/// (given rows of `objs`) and normalized reference point (front nadir + 10 % of its range)
pub(crate) fn normalized_front(
    objs: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    front_rows: &[usize],
) -> (Normalization, Array2<f64>, Array1<f64>) {
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
    (normalization, front, ref_point)
}

/// Multi-objective infill criterion (to be maximized) replacing the mono-objective
/// infill criterion when there is one surrogate per objective
pub(crate) enum MooCriterion<'a> {
    Eim(EimCriterion<'a>),
    Ehvi(EhviCriterion<'a>),
}

impl MooCriterion<'_> {
    /// Name used in logs
    pub(crate) fn name(&self) -> &'static str {
        match self {
            MooCriterion::Eim(_) => "EIM",
            MooCriterion::Ehvi(_) => "EHVI",
        }
    }

    /// Criterion value at `x`
    pub(crate) fn value(&self, x: &[f64]) -> f64 {
        match self {
            MooCriterion::Eim(c) => c.value(x),
            MooCriterion::Ehvi(c) => c.value(x),
        }
    }

    /// Criterion value and gradient at `x`
    pub(crate) fn value_grad(&self, x: &[f64]) -> (f64, Array1<f64>) {
        match self {
            MooCriterion::Eim(c) => c.value_grad(x),
            MooCriterion::Ehvi(c) => c.value_grad(x),
        }
    }

    /// Number of points of the Pareto front
    pub(crate) fn front_size(&self) -> usize {
        match self {
            MooCriterion::Eim(c) => c.front_size(),
            MooCriterion::Ehvi(c) => c.front_size(),
        }
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

    #[test]
    fn test_ei_and_derivatives() {
        let (ei, dmu, dsigma) = ei_and_derivatives(0., 0., 1.);
        assert_abs_diff_eq!(ei, norm_pdf(0.), epsilon = 1e-12);
        assert_abs_diff_eq!(dmu, -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(dsigma, norm_pdf(0.), epsilon = 1e-12);
        let (ei, _, _) = ei_and_derivatives(1., 0., 0.);
        assert_abs_diff_eq!(ei, 1., epsilon = 1e-12);
    }
}

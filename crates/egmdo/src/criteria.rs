//! Enrichment criteria and their optimization.
//!
//! * [ExpectedImprovement]: the expected improvement (16) over both uncertainty sources,
//!   estimated by Monte Carlo with fixed samples (specification, section 6, decision D9),
//! * [probability_of_minimum]: the probability `P_min` (18) that each `DoE_UQ` point is the
//!   minimizer of the discretized objective (section 7),
//! * [coefficient_of_variation]: the coefficient of variation of a PCE (section 7, D11, D12),
//! * [multistart_minimize]: multistart COBYLA over the design space.

use crate::errors::Result;
use crate::kl::KlGpField;
use crate::uq::PointUq;
use egobox_gp::basin_optimizer::{Algorithm, Settings, minimize};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// Maximum number of evaluations of a criterion in one local optimization
pub const OPTIM_MAX_EVALS: usize = 500;

/// The minimum over the `DoE_UQ` points of the objective PCEs, for each sample of `xi`
/// `(n, dim)`: `Ŷ_min(Ξ)` (15)
pub fn pce_minimum(uq: &[PointUq], xi: &ArrayView2<f64>) -> Array1<f64> {
    objective_samples(uq, xi).fold_axis(Axis(0), f64::INFINITY, |m, &v| m.min(v))
}

/// The objective PCEs at the `DoE_UQ` points for the samples `xi` `(n, dim)`,
/// as a `(n_UQ, n)` matrix
pub fn objective_samples(uq: &[PointUq], xi: &ArrayView2<f64>) -> Array2<f64> {
    let mut values = Array2::zeros((uq.len(), xi.nrows()));
    for (i, u) in uq.iter().enumerate() {
        values.row_mut(i).assign(&u.pce.eval(xi).column(0));
    }
    values
}

/// The probability `P_min` (18) that each `DoE_UQ` point is the minimizer of the objective
/// PCEs, estimated with the samples `xi` `(n, dim)`. Ties go to the lowest index.
pub fn probability_of_minimum(uq: &[PointUq], xi: &ArrayView2<f64>) -> Array1<f64> {
    let values = objective_samples(uq, xi);
    let mut counts = Array1::<f64>::zeros(uq.len());
    for sample in values.columns() {
        let mut best = 0;
        for (i, &v) in sample.iter().enumerate() {
            if v < sample[best] {
                best = i;
            }
        }
        counts[best] += 1.;
    }
    counts / xi.nrows().max(1) as f64
}

/// The coefficient of variation `sqrt(Var) / |E|` of a quantity of given mean and variance,
/// or its standard deviation when the mean is zero (decisions D11, D12)
pub fn coefficient_of_variation(mean: f64, variance: f64) -> f64 {
    let std = variance.max(0.).sqrt();
    if mean.abs() < 1e-12 {
        std
    } else {
        std / mean.abs()
    }
}

/// The expected improvement (16) over the disciplinary (`Ξ`) and KL-GP interpolation (`η`)
/// uncertainties, estimated with fixed samples (common random numbers, decision D9) so that
/// it is a deterministic function of the design point.
pub struct ExpectedImprovement<'a> {
    field: &'a KlGpField,
    zeta: Array2<f64>,
    eta: Array2<f64>,
    y_min: Array1<f64>,
}

impl<'a> ExpectedImprovement<'a> {
    /// The expected improvement of the random field `field` over the current minimum `y_min`
    /// `(n)`, estimated with the samples `xi` `(n, dim)` and `eta` `(n, M + 1)`.
    pub fn new(
        field: &'a KlGpField,
        y_min: Array1<f64>,
        xi: &ArrayView2<f64>,
        eta: Array2<f64>,
    ) -> Self {
        ExpectedImprovement {
            field,
            zeta: field.zeta(xi),
            eta,
            y_min,
        }
    }

    /// The expected improvement at design points `z` `(n_z, nx)`
    pub fn value(&self, z: &Array2<f64>) -> Result<Array1<f64>> {
        let pred = self.field.predict(z)?;
        let field = self
            .field
            .sample_from(&pred, &self.zeta.view(), &self.eta.view());
        let improvement = (&self.y_min.view().insert_axis(Axis(0)) - &field).mapv(|v| v.max(0.));
        Ok(improvement.mean_axis(Axis(1)).expect("non-empty samples"))
    }
}

/// The result of a local optimization
#[derive(Clone, Debug)]
pub struct LocalOptimum {
    /// The optimum
    pub x: Array1<f64>,
    /// The function value at the optimum
    pub value: f64,
}

/// Minimizes `f` over the box `xlimits` `(nx, 2)` with COBYLA started from each row of
/// `starts` `(n_starts, nx)`, in parallel. The optimization works in coordinates normalized
/// to `[0, 1]^nx` with an initial radius of `0.5`.
///
/// Returns the local optima sorted by increasing value; non-finite values are dropped.
pub fn multistart_minimize(
    f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
    xlimits: &Array2<f64>,
    starts: &Array2<f64>,
) -> Vec<LocalOptimum> {
    let lower = xlimits.column(0).to_owned();
    let width = &xlimits.column(1) - &xlimits.column(0);
    let bounds = vec![(0., 1.); xlimits.nrows()];
    let mut optima: Vec<LocalOptimum> = starts
        .outer_iter()
        .into_par_iter()
        .map(|start| {
            let xinit = ((&start - &lower) / &width).mapv(|v| v.clamp(0., 1.));
            let objective = |x: &[f64], _: Option<&mut [f64]>| {
                f(&(&lower + &(Array1::from_vec(x.to_vec()) * &width)))
            };
            let (value, x) = minimize(
                Algorithm::Cobyla,
                &objective,
                &|_, _, _| unreachable!("no constraint"),
                Settings {
                    xinit: xinit.as_slice().expect("contiguous start"),
                    bounds: &bounds,
                    n_constraints: 0,
                    max_evals: OPTIM_MAX_EVALS,
                    ftol_abs: 0.,
                    ftol_rel: 1e-8,
                    initial_radius: 0.5,
                    bounded_evaluations: true,
                },
            );
            LocalOptimum {
                x: &lower + &(Array1::from_vec(x) * &width),
                value,
            }
        })
        .filter(|o| o.value.is_finite())
        .collect();
    optima.sort_by(|a, b| a.value.total_cmp(&b.value));
    optima
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kl::KlDecomposition;
    use crate::pce::normal_samples;
    use crate::surrogates::GpSettings;
    use crate::uq::{UqSettings, coefficients, doe_uq};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, array};

    #[test]
    fn test_coefficient_of_variation() {
        assert_abs_diff_eq!(coefficient_of_variation(2., 0.04), 0.1);
        assert_abs_diff_eq!(coefficient_of_variation(-2., 0.04), 0.1);
        assert_abs_diff_eq!(coefficient_of_variation(0., 0.04), 0.2);
        assert_abs_diff_eq!(coefficient_of_variation(1., -1e-20), 0.);
    }

    #[test]
    fn test_multistart_minimize() {
        let f = |x: &Array1<f64>| (x[0] - 1.).powi(2) + (x[1] + 2.).powi(2);
        let xlimits = array![[-5., 5.], [-5., 5.]];
        let starts = array![[4., 4.], [-4., 0.], [0., 0.]];
        let optima = multistart_minimize(&f, &xlimits, &starts);
        assert_eq!(optima.len(), 3);
        assert_abs_diff_eq!(optima[0].x, array![1., -2.], epsilon = 1e-3);
        assert!(optima.windows(2).all(|w| w[0].value <= w[1].value));
        // The minimum on a bound
        let optima = multistart_minimize(&f, &array![[2., 3.], [-5., 5.]], &array![[2.5, 0.]]);
        assert_abs_diff_eq!(optima[0].x, array![2., -2.], epsilon = 1e-3);
    }

    #[test]
    fn test_criteria_on_toy_problem() {
        let (tp, models) = crate::uq::tests::toy_models();
        let settings = UqSettings::default();
        let basis = settings.basis(2);
        let xi = settings.regression_samples(&basis, 0);
        let z_uq = array![[-4.3], [-2.5], [0.5], [3.7]];
        let uq = doe_uq(
            &tp.problem,
            &models,
            &z_uq.view(),
            &xi.view(),
            &basis,
            &tp.mda,
        )
        .unwrap();

        // P_min is a probability distribution over DoE_UQ
        let xi_p = normal_samples(10_000, 2, 1);
        let pmin = probability_of_minimum(&uq, &xi_p.view());
        assert_abs_diff_eq!(pmin.sum(), 1., epsilon = 1e-12);
        assert!(pmin.iter().all(|&p| (0. ..=1.).contains(&p)));
        let y_min = pce_minimum(&uq, &xi_p.view());
        let samples = objective_samples(&uq, &xi_p.view());
        assert!(
            samples
                .columns()
                .into_iter()
                .zip(y_min.iter())
                .all(|(s, m)| s.iter().all(|v| v >= m))
        );

        let kl = KlDecomposition::new(&coefficients(&uq, 0), settings.kl_threshold).unwrap();
        let field = crate::kl::KlGpField::fit(&z_uq, kl, basis, &GpSettings::default()).unwrap();
        let xi_ei = normal_samples(2000, 2, 2);
        let eta = normal_samples(2000, field.kl().n_modes() + 1, 3);
        let ei =
            ExpectedImprovement::new(&field, pce_minimum(&uq, &xi_ei.view()), &xi_ei.view(), eta);

        // EI is non-negative, deterministic, and almost zero at DoE_UQ points
        let z = Array::linspace(-5., 5., 101).insert_axis(Axis(1));
        let values = ei.value(&z).unwrap();
        assert!(values.iter().all(|&v| v >= 0.));
        assert_eq!(values, ei.value(&z).unwrap());
        let at_doe = ei.value(&z_uq).unwrap();
        let max = values.fold(0., |m: f64, &v| m.max(v));
        assert!(max > 0.);
        assert!(
            at_doe.iter().all(|&v| v < 1e-2 * max),
            "EI at DoE_UQ {at_doe}, max {max}"
        );

        // Its maximization finds at least the best grid value
        let f = |x: &Array1<f64>| -ei.value(&x.view().insert_axis(Axis(0)).to_owned()).unwrap()[0];
        let starts = array![[-4.], [-1.], [2.], [4.5]];
        let optima = multistart_minimize(&f, tp.problem.xlimits(), &starts);
        assert!(-optima[0].value >= 0.99 * max);
    }
}

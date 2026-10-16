//! Enrichment criteria and their optimization.
//!
//! * [SampleMinima]: the (feasible) minimum of the quantities of interest over the `DoE_UQ`
//!   points for samples of `Ξ`, giving the current minimum `f'_min(Ξ)` of the expected
//!   improvement (15, C-20, C-22) and the probabilities `P_min` / `P_sol` (18, C-25),
//! * [ExpectedImprovement]: the (constrained) expected improvement (16, C-19) over both
//!   uncertainty sources, estimated by Monte Carlo with fixed samples (D9), and the WB2
//!   criterion (C-24),
//! * [Feasibility]: feasibility, constraint violation and coefficient of variation of the
//!   quantities of interest (D11, D12, D21, D27),
//! * [multistart_minimize], [multistart_minimize_constrained]: multistart COBYLA over the
//!   design space.
//!
//! Without constraints, these reduce to the EGMDO criteria (specification, sections 6, 7).

use crate::errors::Result;
use crate::kl::KlGpField;
use crate::pce::Pce;
use crate::uq::{JointSamples, PointUq};
use egobox_gp::basin_optimizer::{Algorithm, Settings, minimize};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// Maximum number of evaluations of a criterion in one local optimization
pub const OPTIM_MAX_EVALS: usize = 500;

/// Constraint value under which a local optimum is considered as feasible: COBYLA satisfies
/// active constraints up to its final trust region radius (1e-6)
pub const FEASIBILITY_EPSILON: f64 = 1e-6;

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

/// The constraints of the quantities of interest `[f_obj, g..., h...]`: their numbers and
/// tolerances (D20, D21)
#[derive(Clone, Debug)]
pub struct Feasibility {
    n_ineq: usize,
    tolerances: Array1<f64>,
    /// Floors of the denominators of the constraint coefficients of variation (D27)
    cv_floors: Array1<f64>,
}

impl Feasibility {
    /// `n_ineq` inequality constraints `g <= τ` followed by equality constraints `|h| <= τ`,
    /// with one tolerance `τ` per constraint, for the enrichment threshold `ε_CV` on the
    /// coefficients of variation.
    pub fn new(n_ineq: usize, tolerances: Array1<f64>, cv_threshold: f64) -> Self {
        let cv_floors = &tolerances / cv_threshold;
        Feasibility {
            n_ineq,
            tolerances,
            cv_floors,
        }
    }

    /// No constraint
    pub fn unconstrained() -> Self {
        Feasibility::new(0, Array1::zeros(0), 1.)
    }

    /// The number of constraints
    pub fn n_constraints(&self) -> usize {
        self.tolerances.len()
    }

    /// The number of inequality constraints
    pub fn n_ineq(&self) -> usize {
        self.n_ineq
    }

    /// The tolerances of the constraints
    pub fn tolerances(&self) -> &Array1<f64> {
        &self.tolerances
    }

    /// Whether the constraint values `c` `[g..., h...]` are feasible
    pub fn is_feasible(&self, c: &ArrayView1<f64>) -> bool {
        c.iter().enumerate().all(|(k, &v)| {
            if k < self.n_ineq {
                v <= self.tolerances[k]
            } else {
                v.abs() <= self.tolerances[k]
            }
        })
    }

    /// The constraint violation `Σ max(g, 0) + Σ |h|` (C-22)
    pub fn violation(&self, c: &ArrayView1<f64>) -> f64 {
        c.iter()
            .enumerate()
            .map(|(k, &v)| if k < self.n_ineq { v.max(0.) } else { v.abs() })
            .sum()
    }

    /// The largest coefficient of variation over the quantities of interest of a PCE
    /// `[f_obj, g..., h...]` (C-26). The denominator of a constraint is floored at
    /// `τ / ε_CV`, so that an active constraint (zero mean) is considered certain when its
    /// standard deviation is below its tolerance `τ` (D27).
    pub fn max_cv(&self, pce: &Pce) -> f64 {
        let (mean, var) = (pce.mean(), pce.variance());
        (0..mean.len())
            .map(|j| self.cv(j, mean[j], var[j]))
            .fold(0., f64::max)
    }

    /// The quantity of interest with the largest coefficient of variation (C-EGMDO, 3.3)
    pub fn most_uncertain(&self, pce: &Pce) -> usize {
        let (mean, var) = (pce.mean(), pce.variance());
        (0..mean.len())
            .max_by(|&i, &j| {
                self.cv(i, mean[i], var[i])
                    .total_cmp(&self.cv(j, mean[j], var[j]))
                    .then(j.cmp(&i))
            })
            .unwrap_or(0)
    }

    fn cv(&self, j: usize, mean: f64, variance: f64) -> f64 {
        if j == 0 {
            coefficient_of_variation(mean, variance)
        } else {
            variance.max(0.).sqrt() / mean.abs().max(self.cv_floors[j - 1])
        }
    }
}

/// The quantities of interest `[f_obj, g..., h...]` of the PCEs at the `DoE_UQ` points for
/// the samples `xi` `(n, dim)`: one `(n, n_qoi)` matrix per point
pub fn qoi_samples(uq: &[PointUq], xi: &ArrayView2<f64>) -> Vec<Array2<f64>> {
    uq.iter().map(|u| u.pce.eval(xi)).collect()
}

/// The quantities of interest of the PCEs at the `DoE_UQ` points for the joint samples
/// `samples` (decision D38): one `(n, n_qoi)` matrix per point
pub fn qoi_joint_samples(uq: &[PointUq], samples: &JointSamples) -> Vec<Array2<f64>> {
    uq.iter()
        .enumerate()
        .map(|(i, u)| u.pce.eval(&samples.point(i)))
        .collect()
}

/// The minima of the quantities of interest over the `DoE_UQ` points, for samples of `Ξ`
#[derive(Clone, Debug)]
pub struct SampleMinima {
    /// The current minimum `f'_min(Ξ)` of each sample: the minimum objective over the feasible
    /// points, or the objective at the least-violating point if none is feasible (C-20, C-22).
    /// Without constraints, it is `Ŷ_min(Ξ)` (15).
    pub f_min: Array1<f64>,
    /// For each sample, the feasible point of minimum objective, if any (ties to the lowest
    /// index)
    pub feasible_argmin: Vec<Option<usize>>,
    /// For each sample, the point of minimum constraint violation (ties to the lowest objective)
    pub least_violation: Vec<usize>,
    n_points: usize,
}

impl SampleMinima {
    /// The minima over the `DoE_UQ` points of the PCEs `uq` for the samples `xi` `(n, dim)`
    pub fn new(uq: &[PointUq], xi: &ArrayView2<f64>, feasibility: &Feasibility) -> Self {
        Self::from_values(&qoi_samples(uq, xi), feasibility)
    }

    /// The minima over the `DoE_UQ` points of the samples `values` of their quantities of
    /// interest, one `(n, n_qoi)` matrix per point ([qoi_samples], [qoi_joint_samples])
    pub fn from_values(values: &[Array2<f64>], feasibility: &Feasibility) -> Self {
        let n = values.first().map_or(0, |v| v.nrows());
        let mut f_min = Array1::zeros(n);
        let mut feasible_argmin = Vec::with_capacity(n);
        let mut least_violation = Vec::with_capacity(n);
        for s in 0..n {
            let mut best: Option<usize> = None;
            let mut least = 0;
            let mut least_theta = f64::INFINITY;
            for (i, v) in values.iter().enumerate() {
                let row = v.row(s);
                let (f, c) = (row[0], row.slice(ndarray::s![1..]));
                if feasibility.is_feasible(&c) && best.is_none_or(|b| f < values[b][[s, 0]]) {
                    best = Some(i);
                }
                let theta = feasibility.violation(&c);
                if theta < least_theta || (theta == least_theta && f < values[least][[s, 0]]) {
                    least = i;
                    least_theta = theta;
                }
            }
            f_min[s] = values[best.unwrap_or(least)][[s, 0]];
            feasible_argmin.push(best);
            least_violation.push(least);
        }
        SampleMinima {
            f_min,
            feasible_argmin,
            least_violation,
            n_points: values.len(),
        }
    }

    /// The probability `P_sol` (C-25) of each `DoE_UQ` point to be the feasible minimizer.
    /// Samples without a feasible point count for no point (D26). Without constraints, it is
    /// `P_min` (18).
    pub fn probability_of_solution(&self) -> Array1<f64> {
        let mut counts = Array1::<f64>::zeros(self.n_points);
        for i in self.feasible_argmin.iter().flatten() {
            counts[*i] += 1.;
        }
        counts / self.feasible_argmin.len().max(1) as f64
    }

    /// The probability of each `DoE_UQ` point to be the least-violating one (D26 fallback)
    pub fn probability_of_least_violation(&self) -> Array1<f64> {
        let mut counts = Array1::<f64>::zeros(self.n_points);
        for i in self.least_violation.iter() {
            counts[*i] += 1.;
        }
        counts / self.least_violation.len().max(1) as f64
    }
}

/// The expected improvement (16, C-19) over the disciplinary (`Ξ`) and KL-GP interpolation
/// (`η`) uncertainties, estimated with fixed samples (common random numbers, decision D9) so
/// that it is a deterministic function of the design point.
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
        Self::with_zeta(field, y_min, field.zeta(xi), eta)
    }

    /// The expected improvement with the samples `zeta` `(n, M)` of the KL random variables
    /// ([KlGpField::zeta_of]) and `eta` `(n, M + 1)`
    pub fn with_zeta(
        field: &'a KlGpField,
        y_min: Array1<f64>,
        zeta: Array2<f64>,
        eta: Array2<f64>,
    ) -> Self {
        ExpectedImprovement {
            field,
            zeta,
            eta,
            y_min,
        }
    }

    /// The expected improvement at design points `z` `(n_z, nx)`
    pub fn value(&self, z: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(self.values(z)?.0)
    }

    /// The WB2 criterion `scale · EI(z) - m_0(z)` at design points `z` `(n_z, nx)` (C-24)
    pub fn wb2(&self, z: &Array2<f64>, scale: f64) -> Result<Array1<f64>> {
        let (ei, m0) = self.values(z)?;
        Ok(ei * scale - m0)
    }

    /// The logarithm of the smoothed expected improvement at design points `z` `(n_z, nx)`,
    /// as the Monte Carlo qLogEI of Ament et al. (2023) (D36):
    ///
    /// ```text
    /// LogEI(z) = log( (1/N) Σ_s τ softplus(Δ_s(z) / τ) ),   Δ_s(z) = f'_min(Ξ_s) - Ỹ(z, Ξ_s, η_s)
    /// ```
    ///
    /// computed with a log-sum-exp. It is close to `log EI` where some samples improve, and
    /// stays finite and informative where none does (about `max_s Δ_s / τ`), whereas the EI
    /// estimate is then exactly 0.
    pub fn log_value(&self, z: &Array2<f64>, tau: f64) -> Result<Array1<f64>> {
        let improvement = self.improvement(z)?;
        let n = improvement.ncols() as f64;
        Ok(improvement
            .rows()
            .into_iter()
            .map(|delta| {
                let logs: Vec<f64> = delta.iter().map(|d| log_softplus(d / tau)).collect();
                let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = logs.iter().map(|l| (l - max).exp()).sum();
                tau.ln() + max + sum.ln() - n.ln()
            })
            .collect())
    }

    /// The expected improvement and the expectation `m_0` of the objective model
    fn values(&self, z: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let pred = self.field.predict(z)?;
        let improvement = self.improvement_from(&pred).mapv(|v| v.max(0.));
        Ok((
            improvement.mean_axis(Axis(1)).expect("non-empty samples"),
            pred.mean.column(0).to_owned(),
        ))
    }

    /// The improvements `Δ_s(z) = f'_min(Ξ_s) - Ỹ(z, Ξ_s, η_s)`, as a `(n_z, N)` matrix
    fn improvement(&self, z: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(self.improvement_from(&self.field.predict(z)?))
    }

    fn improvement_from(&self, pred: &crate::kl::KlGpPrediction) -> Array2<f64> {
        let field = self
            .field
            .sample_from(pred, &self.zeta.view(), &self.eta.view());
        &self.y_min.view().insert_axis(Axis(0)) - &field
    }
}

/// `log(softplus(x)) = log(log(1 + exp(x)))`, computed without overflow nor underflow
fn log_softplus(x: f64) -> f64 {
    if x < -30. {
        // softplus(x) = exp(x) (1 - exp(x)/2 + ...)
        x
    } else if x > 30. {
        // softplus(x) = x + log(1 + exp(-x)), with exp(-x) negligible
        x.ln()
    } else {
        x.exp().ln_1p().ln()
    }
}

/// The result of a local optimization
#[derive(Clone, Debug)]
pub struct LocalOptimum {
    /// The optimum
    pub x: Array1<f64>,
    /// The function value at the optimum
    pub value: f64,
    /// The largest constraint violation `max(c, 0)` at the optimum (0 when feasible)
    pub violation: f64,
}

impl LocalOptimum {
    /// Whether the optimum satisfies the constraints
    pub fn is_feasible(&self) -> bool {
        self.violation <= FEASIBILITY_EPSILON
    }
}

/// Minimizes `f` over the box `xlimits` `(nx, 2)` with COBYLA started from each row of
/// `starts` `(n_starts, nx)`, in parallel. See [multistart_minimize_constrained].
pub fn multistart_minimize(
    f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
    xlimits: &Array2<f64>,
    starts: &Array2<f64>,
) -> Vec<LocalOptimum> {
    multistart_minimize_constrained(f, &|_, _| unreachable!("no constraint"), 0, xlimits, starts)
}

/// Minimizes `f` subject to `c_k(x) <= 0`, `k = 0..n_constraints`, over the box `xlimits`
/// `(nx, 2)`, with COBYLA started from each row of `starts` `(n_starts, nx)`, in parallel.
/// `constraint(k, x)` returns `c_k(x)`. The optimization works in coordinates normalized to
/// `[0, 1]^nx` with an initial radius of `0.5`.
///
/// Returns the local optima with a finite value: the feasible ones by increasing value,
/// then the infeasible ones by increasing violation.
pub fn multistart_minimize_constrained(
    f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
    constraint: &(dyn Fn(usize, &Array1<f64>) -> f64 + Sync),
    n_constraints: usize,
    xlimits: &Array2<f64>,
    starts: &Array2<f64>,
) -> Vec<LocalOptimum> {
    let lower = xlimits.column(0).to_owned();
    let width = &xlimits.column(1) - &xlimits.column(0);
    let bounds = vec![(0., 1.); xlimits.nrows()];
    let unscale = |x: &[f64]| &lower + &(Array1::from_vec(x.to_vec()) * &width);
    let mut optima: Vec<LocalOptimum> = starts
        .outer_iter()
        .into_par_iter()
        .map(|start| {
            let xinit = ((&start - &lower) / &width).mapv(|v| v.clamp(0., 1.));
            let (_, x) = minimize(
                Algorithm::Cobyla,
                &|x: &[f64], _: Option<&mut [f64]>| f(&unscale(x)),
                &|k: usize, x: &[f64], _: Option<&mut [f64]>| constraint(k, &unscale(x)),
                Settings {
                    xinit: xinit.as_slice().expect("contiguous start"),
                    bounds: &bounds,
                    n_constraints,
                    max_evals: OPTIM_MAX_EVALS,
                    ftol_abs: 0.,
                    ftol_rel: 1e-8,
                    initial_radius: 0.5,
                    bounded_evaluations: true,
                },
            );
            let x = unscale(&x);
            let violation = (0..n_constraints)
                .map(|k| constraint(k, &x).max(0.))
                .fold(0., f64::max);
            LocalOptimum {
                value: f(&x),
                x,
                violation,
            }
        })
        .filter(|o| o.value.is_finite() && o.violation.is_finite())
        .collect();
    optima.sort_by(|a, b| match (a.is_feasible(), b.is_feasible()) {
        (true, true) => a.value.total_cmp(&b.value),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.violation.total_cmp(&b.violation),
    });
    optima
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kl::KlDecomposition;
    use crate::pce::{HermiteBasis, normal_samples};
    use crate::surrogates::GpSettings;
    use crate::uq::{RandomMda, UqSettings, coefficients, doe_uq};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, array};

    #[test]
    fn test_log_softplus() {
        for x in [-50f64, -31., -10., -1., 0., 1., 10., 29., 31., 50.] {
            let exact = x.exp().ln_1p().ln();
            assert_abs_diff_eq!(log_softplus(x), exact, epsilon = 1e-10);
        }
        // No overflow nor underflow far in the tails
        assert_eq!(log_softplus(-1e4), -1e4);
        assert_abs_diff_eq!(log_softplus(1e4), 1e4f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_coefficient_of_variation() {
        assert_abs_diff_eq!(coefficient_of_variation(2., 0.04), 0.1);
        assert_abs_diff_eq!(coefficient_of_variation(-2., 0.04), 0.1);
        assert_abs_diff_eq!(coefficient_of_variation(0., 0.04), 0.2);
        assert_abs_diff_eq!(coefficient_of_variation(1., -1e-20), 0.);
    }

    /// A PCE of 3 quantities in 1 variable with the given (mean, std) of degree 1
    fn pce(moments: &[(f64, f64)]) -> Pce {
        let basis = HermiteBasis::new(1, 1);
        let mut coefs = Array2::zeros((2, moments.len()));
        for (j, (m, s)) in moments.iter().enumerate() {
            coefs[[0, j]] = *m;
            coefs[[1, j]] = *s;
        }
        Pce::new(basis, coefs).unwrap()
    }

    #[test]
    fn test_feasibility() {
        let feas = Feasibility::new(1, array![1e-4, 1e-3], 0.01);
        assert!(feas.is_feasible(&array![0., 0.].view()));
        assert!(feas.is_feasible(&array![1e-4, -1e-3].view()));
        assert!(!feas.is_feasible(&array![1e-3, 0.].view()));
        assert!(!feas.is_feasible(&array![-1., 0.1].view()));
        assert_abs_diff_eq!(feas.violation(&array![2., -0.5].view()), 2.5);
        assert_abs_diff_eq!(feas.violation(&array![-2., 0.5].view()), 0.5);

        // Objective CV 0.1; inequality with zero mean: std / (τ / ε_CV) = 1e-3 (D27);
        // equality CV 0.5
        let p = pce(&[(2., 0.2), (0., 1e-5), (1., 0.5)]);
        assert_abs_diff_eq!(feas.max_cv(&p), 0.5);
        assert_eq!(feas.most_uncertain(&p), 2);
        // An active constraint is uncertain when its std exceeds its tolerance
        let p = pce(&[(2., 0.2), (0., 1e-3), (1., 0.05)]);
        assert_abs_diff_eq!(feas.max_cv(&p), 0.1);
        let p = pce(&[(2., 0.02), (0., 1e-3), (1., 0.05)]);
        assert_abs_diff_eq!(feas.max_cv(&p), 0.1);
        assert_eq!(feas.most_uncertain(&p), 1);
        // Without constraints, only the objective counts
        let p = pce(&[(-2., 0.2)]);
        assert_abs_diff_eq!(Feasibility::unconstrained().max_cv(&p), 0.1);
    }

    fn point(moments: &[(f64, f64)]) -> PointUq {
        PointUq {
            pce: pce(moments),
            n_failed: 0,
            reliable: true,
            linearized: false,
            y_mean: Array1::zeros(1),
        }
    }

    #[test]
    fn test_sample_minima() {
        let feas = Feasibility::new(1, array![0.], 0.01);
        // Point 0: low objective, always infeasible. Point 1: feasible iff ξ < 0.
        // Point 2: high objective, always feasible.
        let uq = vec![
            point(&[(-1., 0.), (1., 0.)]),
            point(&[(0., 0.), (0., 1.)]),
            point(&[(1., 0.), (-1., 0.)]),
        ];
        let xi = array![[-0.5], [0.5]];
        let minima = SampleMinima::new(&uq, &xi.view(), &feas);
        assert_eq!(minima.feasible_argmin, vec![Some(1), Some(2)]);
        assert_eq!(minima.f_min, array![0., 1.]);
        assert_abs_diff_eq!(minima.probability_of_solution(), array![0., 0.5, 0.5]);

        // No feasible point: f'_min at the least-violating point, which counts for no P_sol
        let uq = vec![point(&[(-1., 0.), (2., 0.)]), point(&[(0., 0.), (1., 0.)])];
        let minima = SampleMinima::new(&uq, &xi.view(), &feas);
        assert_eq!(minima.feasible_argmin, vec![None, None]);
        assert_eq!(minima.least_violation, vec![1, 1]);
        assert_eq!(minima.f_min, array![0., 0.]);
        assert_eq!(minima.probability_of_solution().sum(), 0.);
        assert_abs_diff_eq!(minima.probability_of_least_violation(), array![0., 1.]);

        // Without constraints: the minimum objective, and P_sol = P_min sums to 1
        let uq = vec![point(&[(0., 1.)]), point(&[(0., -1.)])];
        let minima = SampleMinima::new(&uq, &xi.view(), &Feasibility::unconstrained());
        assert_eq!(minima.f_min, array![-0.5, -0.5]);
        assert_eq!(minima.feasible_argmin, vec![Some(0), Some(1)]);
        assert_abs_diff_eq!(minima.probability_of_solution().sum(), 1.);
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
    fn test_multistart_minimize_constrained() {
        // min x0 + x1 s.t. x0² + x1² <= 1: optimum at -(1, 1)/sqrt(2)
        let f = |x: &Array1<f64>| x[0] + x[1];
        let c = |_: usize, x: &Array1<f64>| x[0] * x[0] + x[1] * x[1] - 1.;
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let starts = array![[1.5, 1.5], [0., 0.], [-1.5, 0.5]];
        let optima = multistart_minimize_constrained(&f, &c, 1, &xlimits, &starts);
        assert!(optima[0].is_feasible(), "{optima:?}");
        let r = 1. / 2f64.sqrt();
        assert_abs_diff_eq!(optima[0].x, array![-r, -r], epsilon = 1e-3);
        assert_abs_diff_eq!(optima[0].value, -2. * r, epsilon = 1e-3);
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
            RandomMda::Solved,
        )
        .unwrap();

        // P_min is a probability distribution over DoE_UQ
        let xi_p = normal_samples(10_000, 2, 1);
        let minima = SampleMinima::new(&uq, &xi_p.view(), &Feasibility::unconstrained());
        let pmin = minima.probability_of_solution();
        assert_abs_diff_eq!(pmin.sum(), 1., epsilon = 1e-12);
        assert!(pmin.iter().all(|&p| (0. ..=1.).contains(&p)));
        let samples = qoi_samples(&uq, &xi_p.view());
        for (s, m) in minima.f_min.iter().enumerate() {
            assert!(samples.iter().all(|v| v[[s, 0]] >= *m));
        }

        let kl = KlDecomposition::new(&coefficients(&uq, 0), settings.kl_threshold).unwrap();
        let field = crate::kl::KlGpField::fit(&z_uq, kl, basis, &GpSettings::default()).unwrap();
        let xi_ei = normal_samples(2000, 2, 2);
        let eta = normal_samples(2000, field.kl().n_modes() + 1, 3);
        let y_min = SampleMinima::new(&uq, &xi_ei.view(), &Feasibility::unconstrained()).f_min;
        let ei = ExpectedImprovement::new(&field, y_min, &xi_ei.view(), eta);

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

        // WB2 = scale · EI - m_0
        let wb2 = ei.wb2(&z, 2.).unwrap();
        let m0 = field.expectation(&z).unwrap();
        assert_abs_diff_eq!(wb2, &values * 2. - &m0, epsilon = 1e-12);

        // LogEI ≈ log EI where the EI is not negligible, finite where the EI estimate is 0
        let log_ei = ei.log_value(&z, 1e-9).unwrap();
        assert!(log_ei.iter().all(|v| v.is_finite()));
        for (l, e) in log_ei.iter().zip(values.iter()) {
            if *e > 1e-3 * max {
                assert_abs_diff_eq!(*l, e.ln(), epsilon = 1e-4);
            }
        }
        // DoE_UQ points, where (almost) no sample improves, keep a finite value
        let z_far = array![[-4.3], [0.5]];
        let log_far = ei.log_value(&z_far, 1e-9).unwrap();
        assert!(log_far.iter().all(|v| v.is_finite()));

        // Its maximization finds at least the best grid value
        let f = |x: &Array1<f64>| -ei.value(&x.view().insert_axis(Axis(0)).to_owned()).unwrap()[0];
        let starts = array![[-4.], [-1.], [2.], [4.5]];
        let optima = multistart_minimize(&f, tp.problem.xlimits(), &starts);
        assert!(-optima[0].value >= 0.99 * max);
    }
}

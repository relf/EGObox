//! Test problems of the EGMDO and C-EGMDO papers (specification, sections 11 and 22).
//!
//! Each [TestProblem] gives the problem definition, the true disciplinary solvers,
//! the MDA settings suited to the problem and a reference solution computed on the true MDA.

use crate::errors::Result;
use crate::mda::{MdaSettings, MdaStatus, TrueCouplings, solve_one};
use crate::problem::{Discipline, DisciplineFn, MdoProblem};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array, concatenate};
use std::f64::consts::PI;

/// A test problem with its true disciplinary solvers and its reference solution
pub struct TestProblem {
    /// Problem name
    pub name: &'static str,
    /// The MDO problem
    pub problem: MdoProblem,
    /// The true disciplinary solvers, in the order of the problem disciplines
    pub disciplines: Vec<Box<dyn DisciplineFn>>,
    /// MDA settings suited to the problem
    pub mda: MdaSettings,
    /// Reference optimal design point
    pub z_ref: Array1<f64>,
    /// Reference optimal objective value
    pub f_ref: f64,
}

impl TestProblem {
    /// Solves the true MDA at design point `z` and returns the coupling variables,
    /// the quantities of interest `[f_obj, g..., h...]` and the MDA status.
    pub fn evaluate(&self, z: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>, MdaStatus)> {
        let couplings = TrueCouplings::new(&self.problem, &self.disciplines)?;
        let (y, status) = solve_one(&couplings, z, &self.mda)?;
        let qoi = self
            .problem
            .eval_qoi(&z.insert_axis(Axis(0)), &y.view().insert_axis(Axis(0)))?;
        Ok((y, qoi.row(0).to_owned(), status))
    }

    /// Evaluates discipline `i` at inputs `x` (one `[z_i, y_c(i)]` point per row)
    pub fn eval_discipline(&self, i: usize, x: &Array2<f64>) -> Array2<f64> {
        (self.disciplines[i])(&x.view())
    }
}

fn column(values: Array1<f64>) -> Array2<f64> {
    values.insert_axis(Axis(1))
}

/// 1-D toy problem (EGMDO paper, section 2)
///
/// ```text
/// min  f_obj = cos((y1 + exp(-y2)) / π) + z / 20,   z ∈ [-5, 5]
/// y1 = z² - cos(y2 / 2),   y2 = z + y1
/// ```
pub fn toy() -> TestProblem {
    toy_problem(false)
}

/// Constrained 1-D toy problem (C-EGMDO paper, section 2.1): [toy] with `g = 0.15 y1 - 0.9 <= 0`.
pub fn toy_constrained() -> TestProblem {
    toy_problem(true)
}

fn toy_problem(constrained: bool) -> TestProblem {
    let disciplines = vec![
        Discipline::new("y1")
            .coupling_inputs(&[1], &array![[0., 25.]])
            .coupling_outputs(&[0]),
        Discipline::new("y2")
            .coupling_inputs(&[0], &array![[0., 25.]])
            .coupling_outputs(&[1]),
    ];
    let qoi = move |z: &ArrayView2<f64>, y: &ArrayView2<f64>| {
        let (y1, y2) = (y.column(0), y.column(1));
        let f = column(((&y1 + &y2.mapv(|v| (-v).exp())) / PI).mapv(f64::cos) + &z.column(0) / 20.);
        if constrained {
            concatenate![Axis(1), f, column(y1.mapv(|v| 0.15 * v - 0.9))]
        } else {
            f
        }
    };
    let problem = MdoProblem::new(&array![[-5., 5.]], disciplines, qoi)
        .expect("valid toy problem")
        .constraints(constrained as usize, 0);
    let f1 = |x: &ArrayView2<f64>| {
        column(&x.column(0).mapv(|z| z * z) - &x.column(1).mapv(|y2| (y2 / 2.).cos()))
    };
    let f2 = |x: &ArrayView2<f64>| column(&x.column(0) + &x.column(1));
    let (z_ref, f_ref) = if constrained {
        (-2.402_965_252_924_225_6, -0.460_965_676_050_061_7)
    } else {
        (-3.003_077_542_664_500_7, -1.149_712_984_326_785_7)
    };
    TestProblem {
        name: if constrained {
            "toy_constrained"
        } else {
            "toy"
        },
        problem,
        disciplines: vec![Box::new(f1), Box::new(f2)],
        mda: MdaSettings::default(),
        z_ref: array![z_ref],
        f_ref,
    }
}

/// Unconstrained modified Sellar problem (EGMDO paper, section 4.1)
///
/// ```text
/// min  f_obj = z1 + z3² + y1 + exp(-y2) + 10 cos(z2),   z ∈ [0, 10] x [-10, 10] x [0, 10]
/// y1 = z1 + z2² + z3 - 0.2 y2,   y2 = sqrt(y1) + z1 + z2
/// ```
///
/// `sqrt(max(y1, 0))` is used for negative `y1` (decision D18).
pub fn sellar_modified() -> TestProblem {
    let disciplines = vec![
        Discipline::new("y1")
            .coupling_inputs(&[1], &array![[-5., 24.]])
            .coupling_outputs(&[0]),
        Discipline::new("y2")
            .coupling_inputs(&[0], &array![[1., 50.]])
            .coupling_outputs(&[1]),
    ];
    let qoi = |z: &ArrayView2<f64>, y: &ArrayView2<f64>| {
        column(
            &z.column(0)
                + &z.column(2).mapv(|v| v * v)
                + y.column(0)
                + y.column(1).mapv(|v| (-v).exp())
                + z.column(1).mapv(|v| 10. * v.cos()),
        )
    };
    let problem = MdoProblem::new(&array![[0., 10.], [-10., 10.], [0., 10.]], disciplines, qoi)
        .expect("valid modified Sellar problem");
    let f1 = |x: &ArrayView2<f64>| {
        column(&x.column(0) + &x.column(1).mapv(|v| v * v) + x.column(2) - &x.column(3) * 0.2)
    };
    TestProblem {
        name: "sellar_modified",
        problem,
        disciplines: vec![Box::new(f1), Box::new(sellar_y2)],
        mda: MdaSettings::default(),
        z_ref: array![0., 2.634_495_86, 0.],
        f_ref: -2.808_522_219_463_167_6,
    }
}

/// Classic constrained Sellar problem (C-EGMDO paper, section 4.1)
///
/// ```text
/// min  f_obj = z3² + z2 + y1 + exp(-y2),   z ∈ [-10, 10] x [0, 10] x [0, 10]
/// s.t. g1 = 3.16 - y1 <= 0,   g2 = y2 - 24 <= 0
/// y1 = z1² + z2 + z3 - 0.2 y2,   y2 = sqrt(y1) + z1 + z2
/// ```
///
/// The C-EGMDO paper prints `z3³`, a typo for `z3²` (decision D30).
/// `sqrt(max(y1, 0))` is used for negative `y1` (decision D18).
pub fn sellar_constrained() -> TestProblem {
    let disciplines = vec![
        Discipline::new("y1")
            .coupling_inputs(&[1], &array![[0., 25.]])
            .coupling_outputs(&[0]),
        Discipline::new("y2")
            .coupling_inputs(&[0], &array![[0., 25.]])
            .coupling_outputs(&[1]),
    ];
    let qoi = |z: &ArrayView2<f64>, y: &ArrayView2<f64>| {
        let f = column(
            z.column(2).mapv(|v| v * v)
                + z.column(1)
                + y.column(0)
                + y.column(1).mapv(|v| (-v).exp()),
        );
        let g1 = column(y.column(0).mapv(|v| 3.16 - v));
        let g2 = column(y.column(1).mapv(|v| v - 24.));
        concatenate![Axis(1), f, g1, g2]
    };
    let problem = MdoProblem::new(&array![[-10., 10.], [0., 10.], [0., 10.]], disciplines, qoi)
        .expect("valid constrained Sellar problem")
        .constraints(2, 0);
    let f1 = |x: &ArrayView2<f64>| {
        column(x.column(0).mapv(|v| v * v) + x.column(1) + x.column(2) - &x.column(3) * 0.2)
    };
    TestProblem {
        name: "sellar_constrained",
        problem,
        disciplines: vec![Box::new(f1), Box::new(sellar_y2)],
        mda: MdaSettings::default(),
        z_ref: array![1.977_638_88, 0., 0.],
        f_ref: 3.183_393_951_640_613,
    }
}

/// Second Sellar discipline, common to both variants: inputs `[z1, z2, z3, y1]`
fn sellar_y2(x: &ArrayView2<f64>) -> Array2<f64> {
    column(x.column(3).mapv(|v| v.max(0.).sqrt()) + x.column(0) + x.column(1))
}

const CYL_E: f64 = 3e7;
const CYL_NU: f64 = 0.3;
const CYL_P0: f64 = 40000.;

/// Compound cylinder problem (EGMDO paper, section 4.2)
///
/// Design variables `z = (a, b, c, δ) ∈ [5, 7] x [9, 11] x [12, 15] x [6e-3, 1e-2]`,
/// coupling variables `y = (δ1, p)`:
///
/// ```text
/// δ1 = f1(a, b, p)     = (p b / E) ((a² + b²)/(b² - a²) - ρ)
/// p  = f2(b, c, δ, δ1) = (δ - δ1) E / b / ((b² + c²)/(c² - b²) + ρ)
/// ```
///
/// The objective is the coefficient of variation of the tangential stress over the radius,
/// with the corrected stress formulas of the specification (section 11.3). Its radial mean
/// and variance are computed exactly (decision D19). Plain Jacobi iterations diverge in part
/// of the design space, so the MDA is relaxed with `ω = 0.5` (decision D31).
pub fn compound_cylinder() -> TestProblem {
    let disciplines = vec![
        Discipline::new("inner")
            .design_inputs(&[0, 1])
            .coupling_inputs(&[1], &array![[1000., 9000.]])
            .coupling_outputs(&[0]),
        Discipline::new("outer")
            .design_inputs(&[1, 2, 3])
            .coupling_inputs(&[0], &array![[0., 6e-3]])
            .coupling_outputs(&[1]),
    ];
    let qoi = |z: &ArrayView2<f64>, y: &ArrayView2<f64>| {
        column(Array1::from_iter(
            z.rows()
                .into_iter()
                .zip(y.column(1))
                .map(|(z, &p)| cylinder_stress_cv(z[0], z[1], z[2], p)),
        ))
    };
    let problem = MdoProblem::new(
        &array![[5., 7.], [9., 11.], [12., 15.], [6e-3, 1e-2]],
        disciplines,
        qoi,
    )
    .expect("valid compound cylinder problem");
    let f1 = |x: &ArrayView2<f64>| {
        column(Array1::from_iter(x.rows().into_iter().map(|r| {
            let (a, b, p) = (r[0], r[1], r[2]);
            p * b / CYL_E * ((a * a + b * b) / (b * b - a * a) - CYL_NU)
        })))
    };
    let f2 = |x: &ArrayView2<f64>| {
        column(Array1::from_iter(x.rows().into_iter().map(|r| {
            let (b, c, delta, delta1) = (r[0], r[1], r[2], r[3]);
            (delta - delta1) * CYL_E / b / ((b * b + c * c) / (c * c - b * b) + CYL_NU)
        })))
    };
    TestProblem {
        name: "compound_cylinder",
        problem,
        disciplines: vec![Box::new(f1), Box::new(f2)],
        mda: MdaSettings {
            relaxation: 0.5,
            ..MdaSettings::default()
        },
        z_ref: array![7., 9.105_390_21, 12., 6.534_533_08e-3],
        f_ref: 2.317_700_23e-2,
    }
}

/// Coefficient of variation over `r ∈ [a, c]` of the tangential stress
/// `σ(r) = A1 + B1/r²` on `[a, b]` and `A2 + B2/r²` on `]b, c]`
fn cylinder_stress_cv(a: f64, b: f64, c: f64, p: f64) -> f64 {
    let (a2, b2, c2) = (a * a, b * b, c * c);
    let sa = -2. * p * b2 / (b2 - a2) + (a2 + c2) * CYL_P0 / (c2 - a2);
    let sbm = -p * (b2 + a2) / (b2 - a2) + a2 * CYL_P0 * (b2 + c2) / (b2 * (c2 - a2));
    let sbp = p * (b2 + c2) / (c2 - b2) + a2 * CYL_P0 * (b2 + c2) / (b2 * (c2 - a2));
    let sc = 2. * b2 * p / (c2 - b2) + 2. * a2 * CYL_P0 / (c2 - a2);
    let coef_b1 = (sa - sbm) * a2 * b2 / (a2 + b2);
    let coef_a1 = sa - coef_b1 / a2;
    let coef_b2 = (sbp - sc) * b2 * c2 / (b2 + c2);
    let coef_a2 = sbp - coef_b2 / b2;
    // Integrals of σ and σ² over [r0, r1]
    let integrals = |ca: f64, cb: f64, r0: f64, r1: f64| {
        let inv1 = 1. / r0 - 1. / r1;
        let inv3 = (1. / r0.powi(3) - 1. / r1.powi(3)) / 3.;
        (
            ca * (r1 - r0) + cb * inv1,
            ca * ca * (r1 - r0) + 2. * ca * cb * inv1 + cb * cb * inv3,
        )
    };
    let (m1, s1) = integrals(coef_a1, coef_b1, a, b);
    let (m2, s2) = integrals(coef_a2, coef_b2, b, c);
    let mean = (m1 + m2) / (c - a);
    let var = (s1 + s2) / (c - a) - mean * mean;
    var.max(0.).sqrt() / mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn check_reference(tp: &TestProblem, tol: f64) {
        let (y, qoi, status) = tp.evaluate(&tp.z_ref.view()).unwrap();
        assert!(status.is_converged(), "{}: MDA status {status:?}", tp.name);
        assert_abs_diff_eq!(qoi[0], tp.f_ref, epsilon = tol);
        // The MDA solution is a fixed point of the true disciplines
        for (i, d) in tp.problem.disciplines().iter().enumerate() {
            let x = d.inputs(&tp.z_ref.view(), &y.view().insert_axis(Axis(0)));
            let out = tp.eval_discipline(i, &x);
            for (k, &q) in d.coupling_output_indices().iter().enumerate() {
                assert_abs_diff_eq!(out[[0, k]], y[q], epsilon = 1e-4 * y[q].abs().max(1.));
            }
        }
    }

    #[test]
    fn test_toy_reference() {
        check_reference(&toy(), 1e-5);
        // Paper: z* ≈ -3.0, f_obj(z*) ≈ -1.15
        let (_, qoi, _) = toy().evaluate(&array![-3.].view()).unwrap();
        assert_abs_diff_eq!(qoi[0], -1.15, epsilon = 1e-2);
    }

    #[test]
    fn test_toy_constrained_reference() {
        let tp = toy_constrained();
        check_reference(&tp, 1e-5);
        let (_, qoi, _) = tp.evaluate(&tp.z_ref.view()).unwrap();
        assert_abs_diff_eq!(qoi[1], 0., epsilon = 1e-6);
        // The unconstrained minimum is infeasible
        let (_, qoi, _) = tp.evaluate(&array![-3.].view()).unwrap();
        assert!(qoi[1] > 0.);
    }

    #[test]
    fn test_sellar_modified_reference() {
        let tp = sellar_modified();
        check_reference(&tp, 1e-5);
        // Paper: z* ≈ (0, 2.634, 0), f* ≈ -2.808; local minimum (0, -2.595, 0), f ≈ -0.809
        let (_, qoi, _) = tp.evaluate(&array![0., -2.595, 0.].view()).unwrap();
        assert_abs_diff_eq!(qoi[0], -0.809, epsilon = 1e-3);
    }

    #[test]
    fn test_sellar_constrained_reference() {
        let tp = sellar_constrained();
        check_reference(&tp, 1e-5);
        let (_, qoi, _) = tp.evaluate(&tp.z_ref.view()).unwrap();
        assert_abs_diff_eq!(qoi[1], 0., epsilon = 1e-6);
        assert!(qoi[2] < 0.);
        // Local optimum of the C-EGMDO paper, Table 1
        let (_, qoi, _) = tp
            .evaluate(&array![-1.7172, 0.1384, 0.1128].view())
            .unwrap();
        assert_abs_diff_eq!(qoi[0], 4.1308, epsilon = 1e-3);
    }

    #[test]
    fn test_compound_cylinder_reference() {
        let tp = compound_cylinder();
        check_reference(&tp, 1e-7);
        // Paper's reference point and Fig. 25 configuration
        let (_, qoi, _) = tp.evaluate(&array![7., 9., 12., 6.37e-3].view()).unwrap();
        assert_abs_diff_eq!(qoi[0], 2.336_362e-2, epsilon = 1e-7);
        let (_, qoi, _) = tp.evaluate(&array![6., 10., 13., 8e-3].view()).unwrap();
        assert_abs_diff_eq!(qoi[0], 7.396_061e-2, epsilon = 1e-7);
    }

    #[test]
    fn test_compound_cylinder_needs_relaxation() {
        // Plain Jacobi diverges where the iteration factor exceeds 1 in magnitude
        let tp = compound_cylinder();
        let couplings = TrueCouplings::new(&tp.problem, &tp.disciplines).unwrap();
        let z = array![7., 9., 15., 8e-3];
        let (_, status) = solve_one(&couplings, &z.view(), &MdaSettings::default()).unwrap();
        assert!(!status.is_converged());
        let (_, status) = solve_one(&couplings, &z.view(), &tp.mda).unwrap();
        assert!(status.is_converged());
    }
}

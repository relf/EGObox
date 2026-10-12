//! Definition of a multidisciplinary design optimization (MDO) problem.
//!
//! An MDO problem is made of:
//! * a design space `Z`, a box of `R^n`,
//! * disciplines `y_i = f_i(z_i, y_c(i))` computing coupling variables from some design
//!   variables `z_i` and from coupling variables `y_c(i)` computed by other disciplines,
//! * a cheap function computing the objective and the constraints from the design
//!   variables `z` and the coupling variables `y`.
//!
//! The coupling variables are scalars indexed from `0` to `n_couplings - 1`. Each one is
//! computed by exactly one discipline.

use crate::errors::{EgmdoError, Result};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2, concatenate};
use std::sync::Arc;

/// A disciplinary solver, evaluated on a batch of points.
///
/// Each row of the input is `[z_i, y_c(i)]`: the design variables used by the discipline
/// followed by the coupling variables it receives, in declaration order.
/// Each row of the output holds the coupling variables computed by the discipline,
/// in declaration order.
pub trait DisciplineFn: Fn(&ArrayView2<f64>) -> Array2<f64> + Send + Sync {}
impl<T> DisciplineFn for T where T: Fn(&ArrayView2<f64>) -> Array2<f64> + Send + Sync {}

/// The cheap quantities of interest of an MDO problem, evaluated on a batch of points.
///
/// Given design variables `z` (one point per row) and the corresponding coupling
/// variables `y` (one point per row), it returns one row per point:
/// `[f_obj, g_1, ..., g_ng, h_1, ..., h_nh]`, with inequality constraints `g <= 0`
/// and equality constraints `h = 0`.
pub trait QoiFn: Fn(&ArrayView2<f64>, &ArrayView2<f64>) -> Array2<f64> + Send + Sync {}
impl<T> QoiFn for T where T: Fn(&ArrayView2<f64>, &ArrayView2<f64>) -> Array2<f64> + Send + Sync {}

/// A discipline of an MDO problem.
///
/// It only describes the discipline's inputs and outputs: the disciplinary solver itself
/// is evaluated by the caller.
#[derive(Clone, Debug)]
pub struct Discipline {
    name: String,
    design_inputs: Option<Vec<usize>>,
    coupling_inputs: Vec<usize>,
    coupling_bounds: Array2<f64>,
    coupling_outputs: Vec<usize>,
}

impl Discipline {
    /// A discipline named `name`, using all design variables, without coupling inputs nor outputs.
    pub fn new(name: &str) -> Self {
        Discipline {
            name: name.to_string(),
            design_inputs: None,
            coupling_inputs: vec![],
            coupling_bounds: Array2::zeros((0, 2)),
            coupling_outputs: vec![],
        }
    }

    /// Sets the indices of the design variables used by the discipline (all of them by default).
    pub fn design_inputs(mut self, indices: &[usize]) -> Self {
        self.design_inputs = Some(indices.to_vec());
        self
    }

    /// Sets the indices of the coupling variables received by the discipline, and the
    /// initial guess of their range, as a `(indices.len(), 2)` matrix of `[lower, upper]` rows.
    ///
    /// The range is only used to sample the initial disciplinary design of experiments.
    pub fn coupling_inputs(
        mut self,
        indices: &[usize],
        bounds: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Self {
        self.coupling_inputs = indices.to_vec();
        self.coupling_bounds = bounds.to_owned();
        self
    }

    /// Sets the indices of the coupling variables computed by the discipline.
    pub fn coupling_outputs(mut self, indices: &[usize]) -> Self {
        self.coupling_outputs = indices.to_vec();
        self
    }

    /// The discipline name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The indices of the design variables used by the discipline.
    ///
    /// Empty until the discipline is part of an [MdoProblem] if not set explicitly.
    pub fn design_input_indices(&self) -> &[usize] {
        self.design_inputs.as_deref().unwrap_or(&[])
    }

    /// The indices of the coupling variables received by the discipline
    pub fn coupling_input_indices(&self) -> &[usize] {
        &self.coupling_inputs
    }

    /// The initial range of the coupling variables received by the discipline
    pub fn coupling_bounds(&self) -> &Array2<f64> {
        &self.coupling_bounds
    }

    /// The indices of the coupling variables computed by the discipline
    pub fn coupling_output_indices(&self) -> &[usize] {
        &self.coupling_outputs
    }

    /// The dimension of a disciplinary input `[z_i, y_c(i)]`
    pub fn input_dim(&self) -> usize {
        self.design_input_indices().len() + self.coupling_inputs.len()
    }

    /// The number of coupling variables computed by the discipline
    pub fn output_dim(&self) -> usize {
        self.coupling_outputs.len()
    }

    /// Builds the disciplinary inputs `[z_i, y_c(i)]` for the design point `z` and
    /// the coupling states `y` (one state per row).
    pub(crate) fn inputs(&self, z: &ArrayView1<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
        let zi = z.select(Axis(0), self.design_input_indices());
        let zi = zi
            .broadcast((y.nrows(), zi.len()))
            .expect("broadcast design variables")
            .to_owned();
        let yi = y.select(Axis(1), &self.coupling_inputs);
        concatenate![Axis(1), zi, yi]
    }
}

/// A multidisciplinary design optimization problem
#[derive(Clone)]
pub struct MdoProblem {
    xlimits: Array2<f64>,
    disciplines: Vec<Discipline>,
    n_couplings: usize,
    qoi: Arc<dyn QoiFn>,
    n_ineq: usize,
    n_eq: usize,
}

impl std::fmt::Debug for MdoProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MdoProblem")
            .field("xlimits", &self.xlimits)
            .field("disciplines", &self.disciplines)
            .field("n_couplings", &self.n_couplings)
            .field("n_ineq", &self.n_ineq)
            .field("n_eq", &self.n_eq)
            .finish()
    }
}

impl MdoProblem {
    /// An unconstrained MDO problem.
    ///
    /// * `xlimits`: `(n, 2)` matrix of `[lower, upper]` bounds of the design variables,
    /// * `disciplines`: the disciplines; their coupling outputs must cover the coupling
    ///   variables `0..n_couplings` exactly once,
    /// * `qoi`: the cheap function computing the objective (see [QoiFn]).
    ///
    /// Use [MdoProblem::constraints] to declare constraints computed by `qoi`.
    pub fn new(
        xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        disciplines: Vec<Discipline>,
        qoi: impl QoiFn + 'static,
    ) -> Result<Self> {
        let xlimits = xlimits.to_owned();
        check_limits(&xlimits, "design space")?;
        if disciplines.is_empty() {
            return Err(EgmdoError::InvalidProblem(
                "at least one discipline is required".to_string(),
            ));
        }
        let nx = xlimits.nrows();
        let n_couplings = disciplines.iter().map(|d| d.coupling_outputs.len()).sum();

        let mut producer = vec![None; n_couplings];
        let mut disciplines = disciplines;
        for (i, d) in disciplines.iter_mut().enumerate() {
            if d.name.is_empty() {
                return Err(EgmdoError::InvalidProblem(
                    "discipline names must not be empty".to_string(),
                ));
            }
            if d.coupling_outputs.is_empty() {
                return Err(EgmdoError::InvalidProblem(format!(
                    "discipline '{}' computes no coupling variable",
                    d.name
                )));
            }
            let design_inputs = d.design_inputs.get_or_insert_with(|| (0..nx).collect());
            if let Some(k) = design_inputs.iter().find(|&&k| k >= nx) {
                return Err(EgmdoError::InvalidProblem(format!(
                    "discipline '{}' uses design variable {k}, but there are only {nx}",
                    d.name
                )));
            }
            for &q in d.coupling_outputs.iter() {
                if q >= n_couplings {
                    return Err(EgmdoError::InvalidProblem(format!(
                        "discipline '{}' computes coupling variable {q}, but coupling variables \
                         are indexed from 0 to {}",
                        d.name,
                        n_couplings - 1
                    )));
                }
                if let Some(other) = producer[q].replace(i) {
                    return Err(EgmdoError::InvalidProblem(format!(
                        "coupling variable {q} is computed by several disciplines ({} and {i})",
                        other
                    )));
                }
            }
        }
        for d in disciplines.iter() {
            if d.coupling_bounds.nrows() != d.coupling_inputs.len() {
                return Err(EgmdoError::InvalidProblem(format!(
                    "discipline '{}' declares {} coupling inputs but {} coupling bounds",
                    d.name,
                    d.coupling_inputs.len(),
                    d.coupling_bounds.nrows()
                )));
            }
            if !d.coupling_inputs.is_empty() {
                check_limits(
                    &d.coupling_bounds,
                    &format!("coupling bounds of '{}'", d.name),
                )?;
            }
            for &q in d.coupling_inputs.iter() {
                if q >= n_couplings {
                    return Err(EgmdoError::InvalidProblem(format!(
                        "discipline '{}' receives unknown coupling variable {q}",
                        d.name
                    )));
                }
                if d.coupling_outputs.contains(&q) {
                    return Err(EgmdoError::InvalidProblem(format!(
                        "discipline '{}' receives coupling variable {q} which it computes",
                        d.name
                    )));
                }
            }
        }
        let mut names: Vec<&str> = disciplines.iter().map(|d| d.name.as_str()).collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            return Err(EgmdoError::InvalidProblem(
                "discipline names must be unique".to_string(),
            ));
        }

        Ok(MdoProblem {
            xlimits,
            disciplines,
            n_couplings,
            qoi: Arc::new(qoi),
            n_ineq: 0,
            n_eq: 0,
        })
    }

    /// Declares that the quantities of interest function computes `n_ineq` inequality
    /// constraints `g <= 0` followed by `n_eq` equality constraints `h = 0`, after the objective.
    pub fn constraints(mut self, n_ineq: usize, n_eq: usize) -> Self {
        self.n_ineq = n_ineq;
        self.n_eq = n_eq;
        self
    }

    /// The `(n, 2)` bounds of the design space
    pub fn xlimits(&self) -> &Array2<f64> {
        &self.xlimits
    }

    /// The number of design variables
    pub fn nx(&self) -> usize {
        self.xlimits.nrows()
    }

    /// The disciplines
    pub fn disciplines(&self) -> &[Discipline] {
        &self.disciplines
    }

    /// The number of scalar coupling variables
    pub fn n_couplings(&self) -> usize {
        self.n_couplings
    }

    /// The number of inequality constraints
    pub fn n_ineq(&self) -> usize {
        self.n_ineq
    }

    /// The number of equality constraints
    pub fn n_eq(&self) -> usize {
        self.n_eq
    }

    /// The number of quantities of interest: objective and constraints
    pub fn n_qoi(&self) -> usize {
        1 + self.n_ineq + self.n_eq
    }

    /// The `(input_dim, 2)` bounds `Z_i x C(i)` of the inputs of discipline `i`,
    /// over which its initial design of experiments is sampled.
    pub fn discipline_limits(&self, i: usize) -> Array2<f64> {
        let d = &self.disciplines[i];
        let zlimits = self.xlimits.select(Axis(0), d.design_input_indices());
        concatenate![Axis(0), zlimits, d.coupling_bounds]
    }

    /// The starting point of the MDA fixed-point iterations: for each coupling variable,
    /// the mean of the centres of the initial ranges declared by the disciplines receiving it,
    /// or 0 when no discipline receives it.
    pub fn coupling_start(&self) -> Array1<f64> {
        let mut sum = Array1::zeros(self.n_couplings);
        let mut count = Array1::<f64>::zeros(self.n_couplings);
        for d in self.disciplines.iter() {
            for (k, &q) in d.coupling_inputs.iter().enumerate() {
                sum[q] += 0.5 * (d.coupling_bounds[[k, 0]] + d.coupling_bounds[[k, 1]]);
                count[q] += 1.;
            }
        }
        sum / count.mapv(|c| c.max(1.))
    }

    /// Evaluates the quantities of interest `[f_obj, g..., h...]` at design points `z`
    /// with coupling variables `y` (one point per row).
    pub fn eval_qoi(&self, z: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if z.nrows() != y.nrows() || z.ncols() != self.nx() || y.ncols() != self.n_couplings {
            return Err(EgmdoError::InvalidData(format!(
                "quantities of interest expect ({n}, {}) design and ({n}, {}) coupling points, \
                 got {:?} and {:?}",
                self.nx(),
                self.n_couplings,
                z.shape(),
                y.shape(),
                n = z.nrows(),
            )));
        }
        let res = (self.qoi)(z, y);
        if res.dim() != (z.nrows(), self.n_qoi()) {
            return Err(EgmdoError::InvalidData(format!(
                "quantities of interest function returned shape {:?}, expected ({}, {})",
                res.shape(),
                z.nrows(),
                self.n_qoi()
            )));
        }
        Ok(res)
    }
}

fn check_limits(limits: &Array2<f64>, what: &str) -> Result<()> {
    if limits.ncols() != 2 || limits.nrows() == 0 {
        return Err(EgmdoError::InvalidProblem(format!(
            "{what} must be a non-empty (n, 2) matrix of [lower, upper] rows, got {:?}",
            limits.shape()
        )));
    }
    if limits
        .rows()
        .into_iter()
        .any(|r| !(r[0].is_finite() && r[1].is_finite() && r[0] < r[1]))
    {
        return Err(EgmdoError::InvalidProblem(format!(
            "{what} must have finite lower < upper bounds"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn qoi(z: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
        (&z.column(0) + &y.column(0)).insert_axis(Axis(1))
    }

    fn two_disciplines() -> Vec<Discipline> {
        vec![
            Discipline::new("d1")
                .coupling_inputs(&[1], &array![[0., 10.]])
                .coupling_outputs(&[0]),
            Discipline::new("d2")
                .design_inputs(&[1])
                .coupling_inputs(&[0], &array![[-2., 2.]])
                .coupling_outputs(&[1]),
        ]
    }

    #[test]
    fn test_problem_definition() {
        let pb = MdoProblem::new(&array![[0., 1.], [-1., 1.]], two_disciplines(), qoi).unwrap();
        assert_eq!(pb.nx(), 2);
        assert_eq!(pb.n_couplings(), 2);
        assert_eq!(pb.n_qoi(), 1);
        assert_eq!(pb.disciplines()[0].design_input_indices(), &[0, 1]);
        assert_eq!(pb.disciplines()[0].input_dim(), 3);
        assert_eq!(pb.disciplines()[1].input_dim(), 2);
        assert_eq!(
            pb.discipline_limits(1),
            array![[-1., 1.], [-2., 2.]],
            "Z_2 x C(2)"
        );
        assert_eq!(pb.coupling_start(), array![0., 5.]);

        let x = pb.disciplines()[1]
            .inputs(&array![0.5, 0.7].view(), &array![[1., 2.], [3., 4.]].view());
        assert_eq!(x, array![[0.7, 1.], [0.7, 3.]]);

        let pb = pb.constraints(2, 1);
        assert_eq!(pb.n_qoi(), 4);
        assert!(
            pb.eval_qoi(&array![[0., 0.]].view(), &array![[0., 0.]].view())
                .is_err(),
            "qoi returns 1 column instead of 4"
        );
    }

    #[test]
    fn test_invalid_problems() {
        let xlimits = array![[0., 1.], [-1., 1.]];

        let mut d = two_disciplines();
        d[1] = d[1].clone().coupling_outputs(&[0]);
        assert!(
            MdoProblem::new(&xlimits, d, qoi).is_err(),
            "coupling 0 computed twice"
        );

        let mut d = two_disciplines();
        d[0] = d[0].clone().coupling_inputs(&[0], &array![[0., 1.]]);
        assert!(MdoProblem::new(&xlimits, d, qoi).is_err(), "self-coupling");

        let mut d = two_disciplines();
        d[0] = d[0].clone().coupling_inputs(&[1], &array![[1., 0.]]);
        assert!(MdoProblem::new(&xlimits, d, qoi).is_err(), "lower > upper");

        let mut d = two_disciplines();
        d[1] = d[1].clone().design_inputs(&[2]);
        assert!(
            MdoProblem::new(&xlimits, d, qoi).is_err(),
            "unknown design variable"
        );

        let mut d = two_disciplines();
        d[1] = Discipline::new("d1").coupling_outputs(&[1]);
        assert!(
            MdoProblem::new(&xlimits, d, qoi).is_err(),
            "duplicate names"
        );

        assert!(
            MdoProblem::new(&xlimits, vec![], qoi).is_err(),
            "no discipline"
        );
    }
}

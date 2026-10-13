//! Multidisciplinary analysis (MDA).
//!
//! The MDA `y_q = f_i(z_i, y_c(i))` is solved at a fixed design point `z` by non-linear
//! Jacobi iterations: all disciplines are evaluated from the previous iterate
//! (specification, section 3). The solver works on a batch of independent states,
//! so that the random MDAs of all samples of `Ξ` are solved together.
//!
//! The disciplines are given by a [Couplings] implementation:
//! * [TrueCouplings]: the true disciplinary solvers (validation and benchmarks),
//! * [SurrogateCouplings]: the disciplinary GPs, either their means (mean MDA) or the
//!   simplified random fields `μ_q + σ_q ξ_q` for given samples of `Ξ` (random MDA).

use crate::errors::{EgmdoError, Result};
use crate::problem::{DisciplineFn, MdoProblem};
use crate::surrogates::DisciplinaryModels;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Settings of the MDA fixed-point iterations
#[derive(Clone, Debug)]
pub struct MdaSettings {
    /// Convergence threshold on the mean relative change of the coupling variables
    pub tol: f64,
    /// Floor of the denominator of the relative change (decision D3)
    pub eps_abs: f64,
    /// Maximum number of iterations (decision D4)
    pub max_iters: usize,
    /// Relaxation factor `ω` in `]0, 1]`: `y ← (1 - ω) y + ω F(y)` (decision D31).
    /// `1` is the plain non-linear Jacobi method of the EGMDO paper.
    pub relaxation: f64,
    /// Relaxation factors, in `]0, 1]`, used in turn to retry the resolutions which did not
    /// converge, from their starting point. Only factors smaller than `relaxation` are used
    /// (decision D33).
    pub retry_relaxations: Vec<f64>,
}

impl Default for MdaSettings {
    fn default() -> Self {
        MdaSettings {
            tol: 1e-6,
            eps_abs: 1e-12,
            max_iters: 500,
            relaxation: 1.,
            retry_relaxations: vec![0.5, 0.25],
        }
    }
}

/// The final status of one MDA resolution
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MdaStatus {
    /// Converged after the given number of iterations
    Converged(usize),
    /// Not converged within the maximum number of iterations
    MaxIterations,
    /// Stopped because a coupling variable became NaN or infinite
    NonFinite,
}

impl MdaStatus {
    /// Whether the MDA converged
    pub fn is_converged(&self) -> bool {
        matches!(self, MdaStatus::Converged(_))
    }
}

/// The solutions of a batch of MDAs
#[derive(Clone, Debug)]
pub struct MdaSolution {
    /// Coupling variables, one state per row. Rows of failed resolutions hold the last iterate.
    pub y: Array2<f64>,
    /// Status of each resolution
    pub status: Vec<MdaStatus>,
}

impl MdaSolution {
    /// The indices of the converged resolutions
    pub fn converged(&self) -> Vec<usize> {
        (0..self.status.len())
            .filter(|&k| self.status[k].is_converged())
            .collect()
    }

    /// The number of resolutions which did not converge
    pub fn n_failed(&self) -> usize {
        self.status.iter().filter(|s| !s.is_converged()).count()
    }
}

/// The disciplines of an MDA, as a map from coupling states to updated coupling states
pub trait Couplings {
    /// The MDO problem
    fn problem(&self) -> &MdoProblem;

    /// Evaluates all disciplines at design point `z` from the coupling states `y`
    /// (one state per row) and returns the updated coupling states.
    ///
    /// `rows` gives the index, in the whole batch being solved, of each row of `y`.
    fn update(
        &self,
        z: &ArrayView1<f64>,
        y: &ArrayView2<f64>,
        rows: &[usize],
    ) -> Result<Array2<f64>>;
}

/// The true disciplinary solvers
pub struct TrueCouplings<'a> {
    problem: &'a MdoProblem,
    disciplines: &'a [Box<dyn DisciplineFn>],
}

impl<'a> TrueCouplings<'a> {
    /// The true disciplines of `problem`; `disciplines[i]` is the solver of discipline `i`.
    pub fn new(problem: &'a MdoProblem, disciplines: &'a [Box<dyn DisciplineFn>]) -> Result<Self> {
        if disciplines.len() != problem.disciplines().len() {
            return Err(EgmdoError::InvalidData(format!(
                "{} disciplinary solvers given, expected {}",
                disciplines.len(),
                problem.disciplines().len()
            )));
        }
        Ok(TrueCouplings {
            problem,
            disciplines,
        })
    }
}

impl Couplings for TrueCouplings<'_> {
    fn problem(&self) -> &MdoProblem {
        self.problem
    }

    fn update(
        &self,
        z: &ArrayView1<f64>,
        y: &ArrayView2<f64>,
        _rows: &[usize],
    ) -> Result<Array2<f64>> {
        let mut res = Array2::zeros(y.raw_dim());
        for (d, f) in self.problem.disciplines().iter().zip(self.disciplines) {
            let out = f(&d.inputs(z, y).view());
            if out.dim() != (y.nrows(), d.output_dim()) {
                return Err(EgmdoError::InvalidData(format!(
                    "discipline '{}' returned shape {:?}, expected ({}, {})",
                    d.name(),
                    out.shape(),
                    y.nrows(),
                    d.output_dim()
                )));
            }
            for (k, &q) in d.coupling_output_indices().iter().enumerate() {
                res.column_mut(q).assign(&out.column(k));
            }
        }
        Ok(res)
    }
}

/// The disciplinary GPs, as means or as simplified random fields `μ_q + σ_q ξ_q`
pub struct SurrogateCouplings<'a> {
    problem: &'a MdoProblem,
    models: &'a DisciplinaryModels,
    xi: Option<ArrayView2<'a, f64>>,
}

impl<'a> SurrogateCouplings<'a> {
    /// The GP means (mean MDA, `Ξ = 0`)
    pub fn mean(problem: &'a MdoProblem, models: &'a DisciplinaryModels) -> Self {
        SurrogateCouplings {
            problem,
            models,
            xi: None,
        }
    }

    /// The simplified random fields for the samples `xi`, of shape `(n_samples, n_couplings)`:
    /// row `k` is the sample `Ξ^(k)`, with one standard normal variable per coupling variable.
    /// The batch solved with these couplings has one state per sample.
    pub fn random(
        problem: &'a MdoProblem,
        models: &'a DisciplinaryModels,
        xi: ArrayView2<'a, f64>,
    ) -> Result<Self> {
        if xi.ncols() != problem.n_couplings() {
            return Err(EgmdoError::InvalidData(format!(
                "random samples must have {} columns (one per coupling variable), got {}",
                problem.n_couplings(),
                xi.ncols()
            )));
        }
        Ok(SurrogateCouplings {
            problem,
            models,
            xi: Some(xi),
        })
    }
}

impl Couplings for SurrogateCouplings<'_> {
    fn problem(&self) -> &MdoProblem {
        self.problem
    }

    fn update(
        &self,
        z: &ArrayView1<f64>,
        y: &ArrayView2<f64>,
        rows: &[usize],
    ) -> Result<Array2<f64>> {
        let mut res = Array2::zeros(y.raw_dim());
        for (i, d) in self.problem.disciplines().iter().enumerate() {
            let (mean, std) = self.models.predict(i, &d.inputs(z, y))?;
            for (k, &q) in d.coupling_output_indices().iter().enumerate() {
                let mut col = res.column_mut(q);
                col.assign(&mean.column(k));
                if let Some(xi) = &self.xi {
                    col += &(&std.column(k) * &xi.column(q).select(Axis(0), rows));
                }
            }
        }
        Ok(res)
    }
}

/// Solves `n` MDAs at design point `z` from the starting coupling states `y0` of shape
/// `(n, n_couplings)`.
pub fn solve(
    couplings: &impl Couplings,
    z: &ArrayView1<f64>,
    y0: Array2<f64>,
    settings: &MdaSettings,
) -> Result<MdaSolution> {
    let problem = couplings.problem();
    if z.len() != problem.nx() || y0.ncols() != problem.n_couplings() {
        return Err(EgmdoError::InvalidData(format!(
            "MDA expects a design point of size {} and coupling states with {} columns, \
             got {} and {}",
            problem.nx(),
            problem.n_couplings(),
            z.len(),
            y0.ncols()
        )));
    }
    let valid = |w: &f64| *w > 0. && *w <= 1.;
    if !valid(&settings.relaxation) || !settings.retry_relaxations.iter().all(valid) {
        return Err(EgmdoError::InvalidData(format!(
            "MDA relaxation factors must be in ]0, 1], got {} and retries {:?}",
            settings.relaxation, settings.retry_relaxations
        )));
    }
    let mut y = y0.clone();
    let mut status = vec![MdaStatus::MaxIterations; y.nrows()];
    let all: Vec<usize> = (0..y.nrows()).collect();
    iterate(
        couplings,
        z,
        &mut y,
        &mut status,
        all,
        settings.relaxation,
        settings,
    )?;

    // Retry the failed resolutions with stronger relaxation (D33)
    for &omega in settings
        .retry_relaxations
        .iter()
        .filter(|&&w| w < settings.relaxation)
    {
        let failed: Vec<usize> = (0..y.nrows())
            .filter(|&r| !status[r].is_converged())
            .collect();
        if failed.is_empty() {
            break;
        }
        for &r in failed.iter() {
            y.row_mut(r).assign(&y0.row(r));
        }
        iterate(couplings, z, &mut y, &mut status, failed, omega, settings)?;
    }
    Ok(MdaSolution { y, status })
}

/// Relaxed Jacobi iterations of the states `rows` of `y`, updating their status
fn iterate(
    couplings: &impl Couplings,
    z: &ArrayView1<f64>,
    y: &mut Array2<f64>,
    status: &mut [MdaStatus],
    rows: Vec<usize>,
    omega: f64,
    settings: &MdaSettings,
) -> Result<()> {
    for &r in rows.iter() {
        status[r] = MdaStatus::MaxIterations;
    }
    let mut active = rows;
    for iter in 1..=settings.max_iters {
        if active.is_empty() {
            break;
        }
        let ya = y.select(Axis(0), &active);
        let fa = couplings.update(z, &ya.view(), &active)?;
        let new = &ya * (1. - omega) + &fa * omega;

        let mut still_active = Vec::with_capacity(active.len());
        for (k, &r) in active.iter().enumerate() {
            let (old, next) = (ya.row(k), new.row(k));
            y.row_mut(r).assign(&next);
            if next.iter().any(|v| !v.is_finite()) {
                status[r] = MdaStatus::NonFinite;
            } else if relative_change(&old, &next, settings.eps_abs) < settings.tol {
                status[r] = MdaStatus::Converged(iter);
            } else {
                still_active.push(r);
            }
        }
        active = still_active;
    }
    Ok(())
}

/// Solves one MDA at design point `z`, starting from [MdoProblem::coupling_start].
pub fn solve_one(
    couplings: &impl Couplings,
    z: &ArrayView1<f64>,
    settings: &MdaSettings,
) -> Result<(Array1<f64>, MdaStatus)> {
    let y0 = couplings.problem().coupling_start().insert_axis(Axis(0));
    let sol = solve(couplings, z, y0, settings)?;
    Ok((sol.y.row(0).to_owned(), sol.status[0]))
}

/// Solves the random MDAs at design point `z` for the samples of `Ξ` of
/// [SurrogateCouplings::random], all starting from [MdoProblem::coupling_start].
pub fn solve_random(
    couplings: &SurrogateCouplings,
    z: &ArrayView1<f64>,
    settings: &MdaSettings,
) -> Result<MdaSolution> {
    let n = couplings.xi.as_ref().map_or(1, |xi| xi.nrows());
    let y0 = couplings
        .problem
        .coupling_start()
        .broadcast((n, couplings.problem.n_couplings()))
        .expect("broadcast starting point")
        .to_owned();
    solve(couplings, z, y0, settings)
}

/// Mean relative change between two iterates (specification, section 3)
fn relative_change(old: &ArrayView1<f64>, new: &ArrayView1<f64>, eps_abs: f64) -> f64 {
    let n = old.len().max(1) as f64;
    old.iter()
        .zip(new.iter())
        .map(|(o, v)| (v - o).abs() / o.abs().max(eps_abs))
        .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems;
    use crate::surrogates::{GpSettings, initial_doe};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, array};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::StandardNormal;
    use rand_xoshiro::Xoshiro256Plus;

    fn models(tp: &problems::TestProblem, n: usize) -> DisciplinaryModels {
        let data = (0..tp.problem.disciplines().len())
            .map(|i| {
                let x = initial_doe(&tp.problem, i, n, 42 + i as u64);
                let y = tp.eval_discipline(i, &x);
                (x, y)
            })
            .collect();
        DisciplinaryModels::fit(&tp.problem, data, GpSettings::default()).unwrap()
    }

    fn normal_samples(n: usize, m: usize, seed: u64) -> Array2<f64> {
        let mut rng = Xoshiro256Plus::seed_from_u64(seed);
        Array::random_using((n, m), StandardNormal, &mut rng)
    }

    #[test]
    fn test_true_mda_from_several_starts() {
        let tp = problems::toy();
        let couplings = TrueCouplings::new(&tp.problem, &tp.disciplines).unwrap();
        let z = array![-3.];
        let y0 = array![[0., 0.], [10., 10.], [20., 1.]];
        let sol = solve(&couplings, &z.view(), y0, &tp.mda).unwrap();
        assert_eq!(sol.n_failed(), 0);
        assert_eq!(sol.converged(), vec![0, 1, 2]);
        for r in 1..3 {
            assert_abs_diff_eq!(sol.y.row(r), sol.y.row(0), epsilon = 1e-4);
        }
        // y2 = z + y1 at the solution
        assert_abs_diff_eq!(sol.y[[0, 1]], -3. + sol.y[[0, 0]], epsilon = 1e-4);
    }

    #[test]
    fn test_mean_mda_converges_to_true_mda() {
        let tp = problems::toy();
        let z = array![-3.];
        let truth = TrueCouplings::new(&tp.problem, &tp.disciplines).unwrap();
        let (y_true, _) = solve_one(&truth, &z.view(), &tp.mda).unwrap();
        let error = |n: usize| {
            let models = models(&tp, n);
            let mean = SurrogateCouplings::mean(&tp.problem, &models);
            let (y_mean, status) = solve_one(&mean, &z.view(), &tp.mda).unwrap();
            assert!(status.is_converged());
            (&y_mean - &y_true).mapv(f64::abs).sum()
        };
        let (coarse, fine) = (error(10), error(40));
        assert!(fine < 1e-2, "error with 40 training points: {fine}");
        assert!(fine < coarse);
    }

    #[test]
    fn test_random_mda() {
        let tp = problems::sellar_modified();
        let models = models(&tp, 5);
        let z = array![1., 2., 3.];
        let mean = SurrogateCouplings::mean(&tp.problem, &models);
        let (y_mean, _) = solve_one(&mean, &z.view(), &tp.mda).unwrap();

        // Ξ = 0 gives the mean MDA
        let zeros = Array2::zeros((3, 2));
        let random = SurrogateCouplings::random(&tp.problem, &models, zeros.view()).unwrap();
        let sol = solve_random(&random, &z.view(), &tp.mda).unwrap();
        for r in 0..3 {
            assert_abs_diff_eq!(sol.y.row(r), y_mean, epsilon = 1e-9);
        }

        // Random samples give distinct solutions, one per sample
        let xi = normal_samples(50, 2, 0);
        let random = SurrogateCouplings::random(&tp.problem, &models, xi.view()).unwrap();
        let sol = solve_random(&random, &z.view(), &tp.mda).unwrap();
        assert_eq!(sol.y.nrows(), 50);
        assert!(sol.converged().len() > 40, "status: {:?}", sol.status);
        let spread = sol.y.std_axis(Axis(0), 0.);
        assert!(spread.iter().all(|&s| s > 1e-3), "spread {spread}");

        // Reproducible
        let sol2 = solve_random(&random, &z.view(), &tp.mda).unwrap();
        assert_eq!(sol.y, sol2.y);
    }

    #[test]
    fn test_random_mda_collapses_after_enrichment() {
        // Once every discipline is trained at the true MDA solution of z, this solution is a
        // fixed point of every random MDA, since the GPs interpolate with zero variance there.
        let tp = problems::toy();
        let mut models = models(&tp, 5);
        let z = array![-3.];
        let truth = TrueCouplings::new(&tp.problem, &tp.disciplines).unwrap();
        let (y_true, _) = solve_one(&truth, &z.view(), &tp.mda).unwrap();
        for (i, d) in tp.problem.disciplines().iter().enumerate() {
            let x = d.inputs(&z.view(), &y_true.view().insert_axis(Axis(0)));
            let y = tp.eval_discipline(i, &x);
            models.add_points(&tp.problem, i, &x, &y).unwrap();
        }
        let xi = normal_samples(20, 2, 1);
        let random = SurrogateCouplings::random(&tp.problem, &models, xi.view()).unwrap();
        let y0 = y_true.broadcast((20, 2)).unwrap().to_owned();
        let sol = solve(&random, &z.view(), y0, &tp.mda).unwrap();
        assert_eq!(sol.n_failed(), 0);
        for r in 0..20 {
            assert_abs_diff_eq!(sol.y.row(r), y_true, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_non_finite_and_max_iterations() {
        let tp = problems::toy();
        let couplings = TrueCouplings::new(&tp.problem, &tp.disciplines).unwrap();
        let z = array![-3.];
        let sol = solve(&couplings, &z.view(), array![[f64::NAN, 0.]], &tp.mda).unwrap();
        assert_eq!(sol.status, vec![MdaStatus::NonFinite]);
        let settings = MdaSettings {
            max_iters: 2,
            ..MdaSettings::default()
        };
        let sol = solve(&couplings, &z.view(), array![[0., 0.]], &settings).unwrap();
        assert_eq!(sol.status, vec![MdaStatus::MaxIterations]);
        assert_eq!(sol.n_failed(), 1);
    }

    #[test]
    fn test_invalid_inputs() {
        let tp = problems::toy();
        let couplings = TrueCouplings::new(&tp.problem, &tp.disciplines).unwrap();
        let settings = MdaSettings {
            relaxation: 0.,
            ..MdaSettings::default()
        };
        assert!(solve_one(&couplings, &array![-3.].view(), &settings).is_err());
        assert!(solve_one(&couplings, &array![-3., 1.].view(), &tp.mda).is_err());
        assert!(TrueCouplings::new(&tp.problem, &tp.disciplines[..1]).is_err());
        let models = models(&tp, 5);
        let xi = Array2::zeros((3, 3));
        assert!(SurrogateCouplings::random(&tp.problem, &models, xi.view()).is_err());
    }
}

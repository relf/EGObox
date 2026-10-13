//! The EGMDO algorithm (specification, sections 6 to 10), as an ask/tell solver.
//!
//! The solver never calls the disciplinary solvers itself. [EgmdoSolver::ask] runs the
//! algorithm until it needs true disciplinary evaluations and returns them as
//! [EvaluationRequest]s; the caller evaluates them and gives the outputs back with
//! [EgmdoSolver::tell]. [minimize] runs this loop with given disciplinary solvers.

use crate::constraints::ConstraintModels;
use crate::criteria::{
    ExpectedImprovement, Feasibility, SampleMinima, multistart_minimize_constrained, qoi_samples,
};
use crate::errors::{EgmdoError, Result};
use crate::kl::{KlDecomposition, KlGpField};
use crate::mda::{MdaSettings, SurrogateCouplings, solve_one};
use crate::pce::{HermiteBasis, normal_samples};
use crate::problem::{DisciplineFn, MdoProblem};
use crate::surrogates::{DisciplinaryModels, GpSettings, initial_doe};
use crate::uq::{PointUq, UqSettings, coefficients, point_uq};
use egobox_doe::{Lhs, SamplingMethod};
use ndarray::{Array1, Array2, Axis, concatenate, s};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;

/// Distance, in design variables normalized to `[0, 1]`, under which a proposed point is a
/// duplicate of a `DoE_UQ` point (decision D25)
pub const DOE_UQ_DUPLICATE_TOLERANCE: f64 = 1e-6;

/// The acquisition criterion maximized to choose the point added to `DoE_UQ`
/// (specification, sections 6 and 18)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Acquisition {
    /// The (constrained) expected improvement (16, C-19), as in the EGMDO paper
    Ei,
    /// The WB2 criterion `scale · EI - m_0` (C-24), less multimodal than the EI
    Wb2 {
        /// Scale of the expected improvement (1 in the C-EGMDO paper, D23)
        scale: f64,
    },
}

impl Default for Acquisition {
    fn default() -> Self {
        Acquisition::Wb2 { scale: 1. }
    }
}

/// The disciplines enriched at an enrichment point (specification, section 20)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Enrichment {
    /// Every discipline, as in the EGMDO paper
    All,
    /// The discipline with the largest Sobol index of the most uncertain quantity of
    /// interest, as in the C-EGMDO paper
    #[default]
    Sobol,
}

/// Configuration of the EGMDO algorithm
///
/// The defaults follow the C-EGMDO paper (specification, section 23); [EgmdoConfig::egmdo]
/// gives the settings of the EGMDO paper.
#[derive(Clone, Debug)]
pub struct EgmdoConfig {
    /// The acquisition criterion
    pub acquisition: Acquisition,
    /// The disciplines enriched at an enrichment point
    pub enrichment: Enrichment,
    /// The tolerance of each constraint `[g..., h...]` (D21); `None` gives `1e-4`
    pub constraint_tolerances: Option<Vec<f64>>,
    /// Size of the initial design of experiments of each discipline;
    /// `None` gives `2 d_i` for a discipline with `d_i` inputs
    pub n_doe_disciplines: Option<Vec<usize>>,
    /// Initial designs of experiments of the disciplines, as inputs `(n_i, d_i)` to be
    /// evaluated; overrides `n_doe_disciplines`
    pub doe_disciplines: Option<Vec<Array2<f64>>>,
    /// Size of the initial `DoE_UQ`
    pub n_doe_uq: usize,
    /// Initial `DoE_UQ` `(n, nx)`; overrides `n_doe_uq`
    pub doe_uq: Option<Array2<f64>>,
    /// Number of iterations `n_max`
    pub max_iters: usize,
    /// Optional budget of true disciplinary evaluations, all disciplines included (D13)
    pub max_evaluations: Option<usize>,
    /// Enrichment threshold `ε_CV` on the coefficient of variation
    pub cv_threshold: f64,
    /// Enrichment threshold `ε_P` on the probability of being the minimizer;
    /// `None` gives `1 / n_UQ`, as in the EGMDO paper
    pub p_threshold: Option<f64>,
    /// Number of Monte Carlo samples of the EI estimation `N_EI` (D9)
    pub n_ei_samples: usize,
    /// Number of Monte Carlo samples of the `P_min` estimation `n_P` (D10)
    pub n_pmin_samples: usize,
    /// Number of starting points of the EI maximization
    pub n_starts: usize,
    /// Number of samples `n_sim` of the final estimation of the minimum
    pub n_sim: usize,
    /// Uncertainty quantification settings
    pub uq: UqSettings,
    /// GP settings of the disciplinary GPs and KL-GPs
    pub gp: GpSettings,
    /// MDA settings
    pub mda: MdaSettings,
    /// Seed of all random draws; `None` for a non-reproducible run
    pub seed: Option<u64>,
}

impl EgmdoConfig {
    /// The settings of the EGMDO paper: expected improvement, enrichment of all disciplines,
    /// `ε_P = 1 / n_UQ`, `n_MC = 100` and squared exponential GPs
    pub fn egmdo() -> Self {
        EgmdoConfig {
            acquisition: Acquisition::Ei,
            enrichment: Enrichment::All,
            p_threshold: None,
            uq: UqSettings {
                n_mc: Some(100),
                ..UqSettings::default()
            },
            gp: GpSettings::paper(),
            ..EgmdoConfig::default()
        }
    }
}

impl Default for EgmdoConfig {
    fn default() -> Self {
        EgmdoConfig {
            acquisition: Acquisition::default(),
            enrichment: Enrichment::default(),
            constraint_tolerances: None,
            n_doe_disciplines: None,
            doe_disciplines: None,
            n_doe_uq: 20,
            doe_uq: None,
            max_iters: 10,
            max_evaluations: None,
            cv_threshold: 0.01,
            p_threshold: Some(1e-3),
            n_ei_samples: 2000,
            n_pmin_samples: 10_000,
            n_starts: 20,
            n_sim: 100,
            uq: UqSettings::default(),
            gp: GpSettings::default(),
            mda: MdaSettings::default(),
            seed: None,
        }
    }
}

/// True evaluations of a discipline requested by the solver
#[derive(Clone, Debug)]
pub struct EvaluationRequest {
    /// Index of the discipline in the problem
    pub discipline: usize,
    /// Inputs `[z_i, y_c(i)]`, one point per row
    pub x: Array2<f64>,
}

/// What happened at one iteration of the main loop
#[derive(Clone, Debug)]
pub struct IterationRecord {
    /// The point added to `DoE_UQ`, if any (D25)
    pub z_new: Option<Array1<f64>>,
    /// The maximum value of the acquisition criterion
    pub max_criterion: f64,
    /// The number of KL modes
    pub n_modes: usize,
    /// The number of enrichments of the disciplinary GPs
    pub n_enrichments: usize,
    /// The number of non-converged random MDAs over `DoE_UQ` after the iteration
    pub n_failed_mdas: usize,
    /// The number of `DoE_UQ` points dropped because too few random MDAs converged (D32)
    pub n_dropped_points: usize,
}

/// The result of an EGMDO run
#[derive(Clone)]
pub struct EgmdoResult {
    /// The `DoE_UQ` point with the lowest objective PCE mean among the points whose constraint
    /// PCE means are feasible, or with the lowest mean violation if none is (D29)
    pub z_best: Array1<f64>,
    /// The objective PCE mean at `z_best`
    pub f_best: f64,
    /// Samples of the minimum position `Ẑ*` (21), one per row
    pub z_star: Array2<f64>,
    /// Samples of the minimum value `Ŷ_min` (20)
    pub y_min: Array1<f64>,
    /// Number of true evaluations of each discipline
    pub n_evaluations: Vec<usize>,
    /// Per-iteration records
    pub history: Vec<IterationRecord>,
    /// The final disciplinary GPs
    pub models: DisciplinaryModels,
    /// The final `DoE_UQ`
    pub doe_uq: Array2<f64>,
    /// The final PCEs at the `DoE_UQ` points
    pub uq: Vec<PointUq>,
    /// The final KL-GP random field, giving the objective model (19)
    pub field: KlGpField,
    /// The final constraint models
    pub constraints: ConstraintModels,
}

impl EgmdoResult {
    /// Mean of the minimum position `Ẑ*`
    pub fn z_star_mean(&self) -> Array1<f64> {
        self.z_star.mean_axis(Axis(0)).expect("samples")
    }

    /// Standard deviation of the minimum position `Ẑ*`
    pub fn z_star_std(&self) -> Array1<f64> {
        self.z_star.std_axis(Axis(0), 0.)
    }

    /// Mean of the minimum value `Ŷ_min`
    pub fn y_min_mean(&self) -> f64 {
        self.y_min.mean().expect("samples")
    }

    /// Standard deviation of the minimum value `Ŷ_min`
    pub fn y_min_std(&self) -> f64 {
        self.y_min.std(0.)
    }
}

#[derive(Clone, Debug)]
enum State {
    Start,
    AwaitInitialDoe,
    Iterate,
    Enrich { skipped: Vec<usize> },
    AwaitEnrichment { index: usize, skipped: Vec<usize> },
    Done,
}

/// The EGMDO algorithm as an ask/tell solver
pub struct EgmdoSolver {
    problem: MdoProblem,
    config: EgmdoConfig,
    rng: Xoshiro256Plus,
    basis: HermiteBasis,
    xi_regression: Array2<f64>,
    feasibility: Feasibility,
    state: State,
    pending: Vec<EvaluationRequest>,
    models: Option<DisciplinaryModels>,
    doe_uq: Array2<f64>,
    uq: Vec<PointUq>,
    iter: usize,
    history: Vec<IterationRecord>,
    budget_exhausted: bool,
    result: Option<EgmdoResult>,
}

impl EgmdoSolver {
    /// A solver of `problem` with the given configuration
    pub fn new(problem: MdoProblem, config: EgmdoConfig) -> Result<Self> {
        let n_disciplines = problem.disciplines().len();
        if let Some(n) = &config.n_doe_disciplines
            && n.len() != n_disciplines
        {
            return Err(EgmdoError::InvalidData(format!(
                "initial DoE sizes given for {} disciplines, expected {n_disciplines}",
                n.len()
            )));
        }
        if let Some(doe) = &config.doe_uq
            && doe.ncols() != problem.nx()
        {
            return Err(EgmdoError::InvalidData(format!(
                "DoE_UQ must have {} columns, got {}",
                problem.nx(),
                doe.ncols()
            )));
        }
        let n_constraints = problem.n_ineq() + problem.n_eq();
        let tolerances = match &config.constraint_tolerances {
            Some(tol) if tol.len() != n_constraints => {
                return Err(EgmdoError::InvalidData(format!(
                    "{} constraint tolerances given for {n_constraints} constraints",
                    tol.len()
                )));
            }
            Some(tol) => Array1::from_vec(tol.clone()),
            None => Array1::from_elem(n_constraints, 1e-4),
        };
        if let Some(&k) = problem
            .deterministic_constraint_indices()
            .iter()
            .find(|&&k| k >= n_constraints)
        {
            return Err(EgmdoError::InvalidProblem(format!(
                "deterministic constraint {k} does not exist ({n_constraints} constraints)"
            )));
        }
        let feasibility = Feasibility::new(problem.n_ineq(), tolerances, config.cv_threshold);
        let seed = config
            .seed
            .unwrap_or_else(|| Xoshiro256Plus::from_entropy().r#gen());
        let mut rng = Xoshiro256Plus::seed_from_u64(seed);
        let basis = config.uq.basis(problem.n_couplings());
        // Regression samples drawn once per run, shared by all design points (D6)
        let xi_regression = config.uq.regression_samples(&basis, rng.r#gen());
        Ok(EgmdoSolver {
            doe_uq: Array2::zeros((0, problem.nx())),
            problem,
            config,
            rng,
            basis,
            xi_regression,
            feasibility,
            state: State::Start,
            pending: vec![],
            models: None,
            uq: vec![],
            iter: 0,
            history: vec![],
            budget_exhausted: false,
            result: None,
        })
    }

    /// The problem
    pub fn problem(&self) -> &MdoProblem {
        &self.problem
    }

    /// Runs the algorithm until it needs true disciplinary evaluations, and returns them.
    /// Returns `None` when the algorithm is finished; the result is then given by
    /// [EgmdoSolver::result].
    pub fn ask(&mut self) -> Result<Option<Vec<EvaluationRequest>>> {
        loop {
            match std::mem::replace(&mut self.state, State::Done) {
                State::Start => {
                    self.pending = self.initial_requests();
                    self.state = State::AwaitInitialDoe;
                    return Ok(Some(self.pending.clone()));
                }
                State::Iterate => {
                    if self.iter >= self.config.max_iters || self.budget_exhausted {
                        self.finalize()?;
                        self.state = State::Done;
                    } else {
                        self.iter += 1;
                        self.acquisition()?;
                        self.state = State::Enrich { skipped: vec![] };
                    }
                }
                State::Enrich { mut skipped } => match self.next_enrichment(&skipped)? {
                    None => self.state = State::Iterate,
                    Some(index) => {
                        let n_d = self.problem.disciplines().len();
                        if let Some(budget) = self.config.max_evaluations
                            && self.n_evaluations().iter().sum::<usize>() + n_d > budget
                        {
                            log::info!("Budget of {budget} disciplinary evaluations reached");
                            self.budget_exhausted = true;
                            self.state = State::Iterate;
                            continue;
                        }
                        match self.enrichment_requests(index)? {
                            Some(requests) => {
                                self.pending = requests;
                                self.state = State::AwaitEnrichment { index, skipped };
                                return Ok(Some(self.pending.clone()));
                            }
                            None => {
                                skipped.push(index);
                                self.state = State::Enrich { skipped };
                            }
                        }
                    }
                },
                State::Done => return Ok(None),
                state @ (State::AwaitInitialDoe | State::AwaitEnrichment { .. }) => {
                    self.state = state;
                    return Err(EgmdoError::InvalidData(
                        "evaluations were requested: call tell() before ask()".to_string(),
                    ));
                }
            }
        }
    }

    /// Gives the outputs of the evaluations requested by the last [EgmdoSolver::ask]:
    /// `outputs[k]` holds the outputs of request `k`, one row per input point.
    pub fn tell(&mut self, outputs: Vec<Array2<f64>>) -> Result<()> {
        if outputs.len() != self.pending.len() {
            return Err(EgmdoError::InvalidData(format!(
                "{} outputs given for {} requests",
                outputs.len(),
                self.pending.len()
            )));
        }
        let requests = std::mem::take(&mut self.pending);
        match std::mem::replace(&mut self.state, State::Done) {
            State::AwaitInitialDoe => {
                let data = requests.into_iter().map(|r| r.x).zip(outputs).collect();
                self.models = Some(DisciplinaryModels::fit(
                    &self.problem,
                    data,
                    self.config.gp.clone(),
                )?);
                self.init_doe_uq()?;
                self.state = State::Iterate;
            }
            State::AwaitEnrichment { index, mut skipped } => {
                let models = self.models.as_mut().expect("models");
                let mut added = 0;
                for (r, y) in requests.iter().zip(outputs.iter()) {
                    added += models.add_points(&self.problem, r.discipline, &r.x, y)?;
                }
                if added == 0 {
                    // Duplicate disciplinary points: skip this point for this call (D14)
                    skipped.push(index);
                } else {
                    let dropped = self.set_doe_uq(self.doe_uq.clone())?;
                    if dropped > 0 {
                        // DoE_UQ indices changed
                        skipped.clear();
                    }
                    if let Some(record) = self.history.last_mut() {
                        record.n_enrichments += 1;
                        record.n_failed_mdas = self.uq.iter().map(|u| u.n_failed).sum();
                        record.n_dropped_points += dropped;
                    }
                }
                self.state = State::Enrich { skipped };
            }
            state => {
                self.state = state;
                self.pending = requests;
                return Err(EgmdoError::InvalidData(
                    "no evaluation was requested: call ask() first".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// The result, once [EgmdoSolver::ask] has returned `None`
    pub fn result(&self) -> Result<EgmdoResult> {
        self.result
            .clone()
            .ok_or_else(|| EgmdoError::InvalidData("the algorithm is not finished".to_string()))
    }

    /// The number of true evaluations of each discipline so far
    pub fn n_evaluations(&self) -> Vec<usize> {
        (0..self.problem.disciplines().len())
            .map(|i| self.models.as_ref().map_or(0, |m| m.n_evaluations(i)))
            .collect()
    }

    fn initial_requests(&mut self) -> Vec<EvaluationRequest> {
        (0..self.problem.disciplines().len())
            .map(|i| {
                let x = match &self.config.doe_disciplines {
                    Some(doe) => doe[i].clone(),
                    None => {
                        let n = self.config.n_doe_disciplines.as_ref().map_or_else(
                            || 2 * self.problem.disciplines()[i].input_dim(),
                            |n| n[i],
                        );
                        initial_doe(&self.problem, i, n, self.rng.r#gen())
                    }
                };
                EvaluationRequest { discipline: i, x }
            })
            .collect()
    }

    fn init_doe_uq(&mut self) -> Result<()> {
        let doe = match &self.config.doe_uq {
            Some(doe) => doe.clone(),
            None => Lhs::new(self.problem.xlimits())
                .with_rng(Xoshiro256Plus::seed_from_u64(self.rng.r#gen()))
                .sample(self.config.n_doe_uq),
        };
        self.set_doe_uq(doe)?;
        Ok(())
    }

    /// Computes the PCEs at the points of `doe` and makes `DoE_UQ` the points where it
    /// succeeds. The points where no more than `P` random MDAs converge are dropped with a
    /// warning (D32). Returns the number of dropped points.
    fn set_doe_uq(&mut self, doe: Array2<f64>) -> Result<usize> {
        let models = self.models.as_ref().expect("models");
        let results: Vec<Result<PointUq>> = (0..doe.nrows())
            .into_par_iter()
            .map(|i| {
                point_uq(
                    &self.problem,
                    models,
                    &doe.row(i),
                    &self.xi_regression.view(),
                    &self.basis,
                    &self.config.mda,
                )
            })
            .collect();
        let mut kept = vec![];
        let mut uq = vec![];
        for (i, res) in results.into_iter().enumerate() {
            match res {
                Ok(u) => {
                    kept.push(i);
                    uq.push(u);
                }
                Err(EgmdoError::MdaFailure(msg)) => {
                    log::warn!("DoE_UQ point {} dropped: {msg}", doe.row(i));
                }
                Err(e) => return Err(e),
            }
        }
        if kept.len() < 2 {
            return Err(EgmdoError::MdaFailure(format!(
                "random MDAs converged at {} DoE_UQ points only, at least 2 are needed",
                kept.len()
            )));
        }
        self.doe_uq = doe.select(Axis(0), &kept);
        self.uq = uq;
        Ok(doe.nrows() - kept.len())
    }

    fn field(&self) -> Result<KlGpField> {
        let kl = KlDecomposition::new(&coefficients(&self.uq, 0), self.config.uq.kl_threshold)?;
        KlGpField::fit(&self.doe_uq, kl, self.basis.clone(), &self.config.gp)
    }

    fn constraint_models(&self) -> Result<ConstraintModels> {
        ConstraintModels::fit(&self.problem, &self.doe_uq, &self.uq, &self.config.gp)
    }

    /// Step 1: adds the point maximizing the acquisition criterion under the constraint models
    /// to `DoE_UQ` (sections 6 and 18)
    fn acquisition(&mut self) -> Result<()> {
        let field = self.field()?;
        let constraints = self.constraint_models()?;
        let n_modes = field.kl().n_modes();
        let xi = normal_samples(self.config.n_ei_samples, self.basis.dim(), self.rng.r#gen());
        let eta = normal_samples(self.config.n_ei_samples, n_modes + 1, self.rng.r#gen());
        let f_min = SampleMinima::new(&self.uq, &xi.view(), &self.feasibility).f_min;
        let ei = ExpectedImprovement::new(&field, f_min, &xi.view(), eta);
        let acquisition = self.config.acquisition;
        let objective = |z: &Array1<f64>| {
            let z = z.view().insert_axis(Axis(0)).to_owned();
            let value = match acquisition {
                Acquisition::Ei => ei.value(&z),
                Acquisition::Wb2 { scale } => ei.wb2(&z, scale),
            };
            value.map_or(f64::INFINITY, |v| -v[0])
        };
        let starts = self.uniform_points(self.config.n_starts);
        let (n_constraints, constraint) = constraints.as_inequalities(&self.feasibility);
        let optima = multistart_minimize_constrained(
            &objective,
            &constraint,
            n_constraints,
            self.problem.xlimits(),
            &starts,
        );
        let max_criterion = optima.first().map_or(0., |o| -o.value);

        // The best proposed point which is not already in DoE_UQ (D25)
        let width = &self.problem.xlimits().column(1) - &self.problem.xlimits().column(0);
        let candidate = optima.into_iter().map(|o| o.x).find(|z| {
            self.doe_uq.rows().into_iter().all(|zi| {
                ((&zi - z) / &width).mapv(|v| v * v).sum().sqrt() >= DOE_UQ_DUPLICATE_TOLERANCE
            })
        });
        let z_new = match candidate {
            Some(z) => match point_uq(
                &self.problem,
                self.models.as_ref().expect("models"),
                &z.view(),
                &self.xi_regression.view(),
                &self.basis,
                &self.config.mda,
            ) {
                Ok(uq) => {
                    self.doe_uq = concatenate![Axis(0), self.doe_uq, z.view().insert_axis(Axis(0))];
                    self.uq.push(uq);
                    Some(z)
                }
                Err(e) => {
                    log::warn!("Point {z} not added to DoE_UQ: {e}");
                    None
                }
            },
            None => {
                log::warn!("All proposed points already belong to DoE_UQ: no point added");
                None
            }
        };
        log::info!(
            "Iteration {}: max criterion = {max_criterion:e}, M = {n_modes}, z_new = {z_new:?}",
            self.iter
        );
        self.history.push(IterationRecord {
            z_new,
            max_criterion,
            n_modes,
            n_enrichments: 0,
            n_failed_mdas: self.uq.iter().map(|u| u.n_failed).sum(),
            n_dropped_points: 0,
        });
        Ok(())
    }

    /// Step 2 (Algorithm 1, C-EGMDO Algorithm 2): the `DoE_UQ` point where to enrich the
    /// disciplinary GPs, if any (sections 7 and 19)
    fn next_enrichment(&mut self, skipped: &[usize]) -> Result<Option<usize>> {
        let xi = normal_samples(
            self.config.n_pmin_samples,
            self.basis.dim(),
            self.rng.r#gen(),
        );
        let minima = SampleMinima::new(&self.uq, &xi.view(), &self.feasibility);
        let threshold = self.config.p_threshold.unwrap_or(1. / self.uq.len() as f64);
        let mut p = minima.probability_of_solution();
        if p.iter().all(|&v| v < threshold) {
            // No point is likely to be the feasible solution: least violation (D26)
            p = minima.probability_of_least_violation();
        }
        let mut candidates: Vec<usize> = (0..self.uq.len())
            .filter(|&i| p[i] >= threshold && !skipped.contains(&i))
            .collect();
        candidates.sort_by(|&i, &j| p[j].total_cmp(&p[i]));
        Ok(candidates
            .into_iter()
            .find(|&i| self.feasibility.max_cv(&self.uq[i].pce) > self.config.cv_threshold))
    }

    /// The disciplines to enrich at `DoE_UQ` point `index` (section 20)
    fn disciplines_to_enrich(&self, index: usize) -> Vec<usize> {
        let disciplines = self.problem.disciplines();
        match self.config.enrichment {
            Enrichment::All => (0..disciplines.len()).collect(),
            Enrichment::Sobol => {
                let pce = &self.uq[index].pce;
                let quantity = self.feasibility.most_uncertain(pce);
                // One random variable per coupling variable: a discipline's group is the
                // variables of its outputs (D28)
                let indices: Vec<f64> = disciplines
                    .iter()
                    .map(|d| pce.closed_sobol_index(quantity, d.coupling_output_indices()))
                    .collect();
                let best = (0..indices.len())
                    .max_by(|&i, &j| indices[i].total_cmp(&indices[j]).then(j.cmp(&i)))
                    .expect("at least one discipline");
                log::debug!("Sobol indices of quantity {quantity}: {indices:?}");
                vec![best]
            }
        }
    }

    /// The evaluations enriching the selected disciplines at `DoE_UQ` point `index`
    /// (sections 8 and 20), or `None` if the mean MDA does not converge there
    fn enrichment_requests(&self, index: usize) -> Result<Option<Vec<EvaluationRequest>>> {
        let z = self.doe_uq.row(index);
        let mean = SurrogateCouplings::mean(&self.problem, self.models.as_ref().expect("models"));
        let (y, status) = solve_one(&mean, &z, &self.config.mda)?;
        if !status.is_converged() {
            log::warn!("Mean MDA not converged at z = {z} ({status:?}): no enrichment there");
            return Ok(None);
        }
        let y = y.insert_axis(Axis(0));
        Ok(Some(
            self.disciplines_to_enrich(index)
                .into_iter()
                .map(|i| EvaluationRequest {
                    discipline: i,
                    x: self.problem.disciplines()[i].inputs(&z, &y.view()),
                })
                .collect(),
        ))
    }

    /// Final model (19) and estimation of the minimum under the constraint models
    /// (sections 10 and 21)
    fn finalize(&mut self) -> Result<()> {
        let field = self.field()?;
        let constraints = self.constraint_models()?;
        let xi = normal_samples(self.config.n_sim, self.basis.dim(), self.rng.r#gen());
        let zeta = field.zeta(&xi.view());
        let doe_values = qoi_samples(&self.uq, &xi.view());
        let random_starts = self.uniform_points(5 * self.config.n_sim);
        let xlimits = self.problem.xlimits().clone();
        let (n_constraints, constraint) = constraints.as_inequalities(&self.feasibility);

        let minima: Vec<(Array1<f64>, f64)> = (0..self.config.n_sim)
            .into_par_iter()
            .map(|k| {
                let model = |z: &Array1<f64>| {
                    field
                        .predict(&z.view().insert_axis(Axis(0)).to_owned())
                        .map_or(f64::INFINITY, |p| {
                            p.mean[[0, 0]] + p.mean.row(0).slice(s![1..]).dot(&zeta.row(k))
                        })
                };
                // Starts: the 5 best DoE_UQ points for this sample and 5 random points (D16)
                let mut order: Vec<usize> = (0..self.doe_uq.nrows()).collect();
                order.sort_by(|&i, &j| doe_values[i][[k, 0]].total_cmp(&doe_values[j][[k, 0]]));
                order.truncate(5);
                let starts = concatenate![
                    Axis(0),
                    self.doe_uq.select(Axis(0), &order),
                    random_starts.slice(s![5 * k..5 * k + 5, ..])
                ];
                let best = multistart_minimize_constrained(
                    &model,
                    &constraint,
                    n_constraints,
                    &xlimits,
                    &starts,
                )
                .into_iter()
                .next()
                .expect("at least one finite local minimum");
                (best.x, best.value)
            })
            .collect();
        drop(constraint);
        let mut z_star = Array2::zeros((self.config.n_sim, self.problem.nx()));
        let mut y_min = Array1::zeros(self.config.n_sim);
        for (s, (z, y)) in minima.into_iter().enumerate() {
            z_star.row_mut(s).assign(&z);
            y_min[s] = y;
        }

        // Best DoE_UQ point by PCE means, among the feasible ones if any (D29)
        let means: Vec<Array1<f64>> = self.uq.iter().map(|u| u.pce.mean()).collect();
        let feasible: Vec<usize> = (0..means.len())
            .filter(|&i| self.feasibility.is_feasible(&means[i].slice(s![1..])))
            .collect();
        let best = if feasible.is_empty() {
            (0..means.len()).min_by(|&i, &j| {
                self.feasibility
                    .violation(&means[i].slice(s![1..]))
                    .total_cmp(&self.feasibility.violation(&means[j].slice(s![1..])))
            })
        } else {
            feasible
                .into_iter()
                .min_by(|&i, &j| means[i][0].total_cmp(&means[j][0]))
        }
        .expect("non-empty DoE_UQ");
        self.result = Some(EgmdoResult {
            z_best: self.doe_uq.row(best).to_owned(),
            f_best: means[best][0],
            z_star,
            y_min,
            n_evaluations: self.n_evaluations(),
            history: self.history.clone(),
            models: self.models.clone().expect("models"),
            doe_uq: self.doe_uq.clone(),
            uq: self.uq.clone(),
            field,
            constraints,
        });
        Ok(())
    }

    /// `n` points drawn uniformly in the design space
    fn uniform_points(&mut self, n: usize) -> Array2<f64> {
        let xlimits = self.problem.xlimits();
        let mut points = Array2::zeros((n, self.problem.nx()));
        for mut p in points.rows_mut() {
            for (v, l) in p.iter_mut().zip(xlimits.rows()) {
                *v = self.rng.gen_range(l[0]..l[1]);
            }
        }
        points
    }
}

/// Runs the EGMDO algorithm on `problem` with the true disciplinary solvers `disciplines`
/// (`disciplines[i]` is the solver of discipline `i`).
pub fn minimize(
    problem: &MdoProblem,
    disciplines: &[Box<dyn DisciplineFn>],
    config: EgmdoConfig,
) -> Result<EgmdoResult> {
    if disciplines.len() != problem.disciplines().len() {
        return Err(EgmdoError::InvalidData(format!(
            "{} disciplinary solvers given, expected {}",
            disciplines.len(),
            problem.disciplines().len()
        )));
    }
    let mut solver = EgmdoSolver::new(problem.clone(), config)?;
    while let Some(requests) = solver.ask()? {
        let outputs = requests
            .iter()
            .map(|r| (disciplines[r.discipline])(&r.x.view()))
            .collect();
        solver.tell(outputs)?;
    }
    solver.result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems;
    use ndarray::array;

    /// The setting of the 1-D toy problem in the EGMDO paper (section 2)
    fn toy_config(seed: u64) -> EgmdoConfig {
        EgmdoConfig {
            acquisition: Acquisition::Ei,
            enrichment: Enrichment::All,
            n_doe_disciplines: Some(vec![5, 4]),
            doe_uq: Some(array![[-4.3], [-2.5], [0.5], [3.7]]),
            max_iters: 5,
            p_threshold: None,
            seed: Some(seed),
            ..EgmdoConfig::default()
        }
    }

    #[test]
    fn test_ask_tell_protocol() {
        let tp = problems::toy();
        let config = EgmdoConfig {
            max_iters: 2,
            ..toy_config(0)
        };
        let mut solver = EgmdoSolver::new(tp.problem.clone(), config).unwrap();
        assert!(solver.tell(vec![]).is_err(), "tell before ask");

        let requests = solver.ask().unwrap().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].x.dim(), (5, 2));
        assert_eq!(requests[1].x.dim(), (4, 2));
        assert!(solver.ask().is_err(), "ask before tell");
        assert!(solver.tell(vec![]).is_err(), "wrong number of outputs");

        let mut requests = Some(requests);
        let mut n_enrichments = 0;
        let mut first = true;
        while let Some(reqs) = requests {
            if !first {
                // Enrichment: one point per discipline (section 8)
                assert_eq!(reqs.len(), 2);
                assert!(reqs.iter().all(|r| r.x.nrows() == 1));
                n_enrichments += 1;
            }
            first = false;
            let outputs = reqs
                .iter()
                .map(|r| tp.eval_discipline(r.discipline, &r.x))
                .collect();
            solver.tell(outputs).unwrap();
            requests = solver.ask().unwrap();
        }
        assert!(solver.ask().unwrap().is_none(), "finished");

        let result = solver.result().unwrap();
        assert_eq!(result.history.len(), 2);
        let recorded: usize = result.history.iter().map(|r| r.n_enrichments).sum();
        assert!(recorded <= n_enrichments);
        assert!(result.n_evaluations[0] <= 5 + n_enrichments);
        assert!(result.n_evaluations[1] <= 4 + n_enrichments);
        assert_eq!(result.z_star.dim(), (100, 1));
        assert_eq!(result.doe_uq.nrows(), result.uq.len());
    }

    #[test]
    fn test_toy_problem_robustness() {
        // Paper: z* ≈ -3.0, f_obj(z*) ≈ -1.15. The convergence depends on the initial DoEs,
        // as in the papers' robustness studies: with these small DoEs, about 85% of the runs
        // converge (17 of seeds 0..20). A run converges when E[Ẑ*] is within 5% of z*
        // (papers' criterion); its minimum value then carries a small model bias.
        // The trajectories depend on the GP training backend (c-cobyla, blas).
        let tp = problems::toy();
        let mut summary = vec![];
        for seed in 0..10 {
            let result = minimize(&tp.problem, &tp.disciplines, toy_config(seed)).unwrap();
            let converged =
                (result.z_star_mean()[0] - tp.z_ref[0]).abs() < 0.05 * tp.z_ref[0].abs();
            summary.push((
                seed,
                converged,
                result.z_star_mean()[0],
                result.y_min_mean(),
                result.n_evaluations.clone(),
            ));
        }
        let converged: Vec<_> = summary.iter().filter(|s| s.1).collect();
        assert!(
            converged.len() >= 6,
            "runs (seed, converged, E[z*], E[y_min], evaluations): {summary:?}"
        );
        for s in converged {
            assert!((s.3 - tp.f_ref).abs() < 0.1 * tp.f_ref.abs(), "{summary:?}");
        }
        assert!(
            summary.iter().all(|s| s.4.iter().all(|&n| n <= 25)),
            "{summary:?}"
        );
    }

    #[test]
    fn test_reproducibility() {
        let tp = problems::toy();
        let result = minimize(&tp.problem, &tp.disciplines, toy_config(42)).unwrap();
        let again = minimize(&tp.problem, &tp.disciplines, toy_config(42)).unwrap();
        assert_eq!(again.z_star, result.z_star);
        assert_eq!(again.doe_uq, result.doe_uq);
        assert_eq!(again.n_evaluations, result.n_evaluations);
    }

    #[test]
    fn test_random_mdas_failing_everywhere() {
        // Without enough iterations, no random MDA converges: every DoE_UQ point is
        // dropped (D32) and the run stops with an error.
        let tp = problems::toy();
        let config = EgmdoConfig {
            mda: MdaSettings {
                max_iters: 1,
                retry_relaxations: vec![],
                ..MdaSettings::default()
            },
            ..toy_config(0)
        };
        assert!(matches!(
            minimize(&tp.problem, &tp.disciplines, config),
            Err(EgmdoError::MdaFailure(_))
        ));
    }

    #[test]
    fn test_evaluation_budget() {
        let tp = problems::toy();
        // Initial DoEs (9 evaluations) and at most one enrichment (2 evaluations)
        let config = EgmdoConfig {
            max_evaluations: Some(12),
            ..toy_config(0)
        };
        let result = minimize(&tp.problem, &tp.disciplines, config).unwrap();
        assert!(result.n_evaluations.iter().sum::<usize>() <= 12);
    }

    #[test]
    fn test_invalid_configurations() {
        let tp = problems::toy();
        let config = EgmdoConfig {
            n_doe_disciplines: Some(vec![5]),
            ..EgmdoConfig::default()
        };
        assert!(EgmdoSolver::new(tp.problem.clone(), config).is_err());
        let config = EgmdoConfig {
            doe_uq: Some(array![[0., 1.]]),
            ..EgmdoConfig::default()
        };
        assert!(EgmdoSolver::new(tp.problem.clone(), config).is_err());
        assert!(minimize(&tp.problem, &tp.disciplines[..1], EgmdoConfig::default()).is_err());

        // Constraint tolerances and deterministic constraints must match the constraints
        let tp = problems::sellar_constrained();
        let config = EgmdoConfig {
            constraint_tolerances: Some(vec![1e-4]),
            ..EgmdoConfig::default()
        };
        assert!(EgmdoSolver::new(tp.problem.clone(), config).is_err());
        let problem = tp.problem.clone().deterministic_constraints(&[2]);
        assert!(EgmdoSolver::new(problem, EgmdoConfig::default()).is_err());
    }

    #[test]
    fn test_egmdo_preset() {
        let config = EgmdoConfig::egmdo();
        assert_eq!(config.acquisition, Acquisition::Ei);
        assert_eq!(config.enrichment, Enrichment::All);
        assert_eq!(config.p_threshold, None);
        assert_eq!(config.uq.n_mc, Some(100));
        assert_eq!(config.gp.kernel, crate::Kernel::SquaredExponential);
        let default = EgmdoConfig::default();
        assert_eq!(default.acquisition, Acquisition::Wb2 { scale: 1. });
        assert_eq!(default.enrichment, Enrichment::Sobol);
        assert_eq!(default.gp.kernel, crate::Kernel::Matern52);
    }

    /// The setting of the constrained 1-D toy problem in the C-EGMDO paper (section 2.1),
    /// with the default C-EGMDO options
    fn toy_constrained_config(seed: u64) -> EgmdoConfig {
        EgmdoConfig {
            n_doe_disciplines: Some(vec![5, 4]),
            doe_uq: Some(array![[-3.6], [-2.], [1.2], [3.8]]),
            max_iters: 5,
            seed: Some(seed),
            ..EgmdoConfig::default()
        }
    }

    #[test]
    fn test_sobol_enrichment_requests_one_discipline() {
        let tp = problems::toy_constrained();
        let mut solver = EgmdoSolver::new(tp.problem.clone(), toy_constrained_config(0)).unwrap();
        let mut requests = solver.ask().unwrap();
        let mut first = true;
        let mut n_enrichments = 0;
        while let Some(reqs) = requests {
            if !first {
                assert_eq!(reqs.len(), 1, "Sobol enrichment of a single discipline");
                assert_eq!(reqs[0].x.nrows(), 1);
                n_enrichments += 1;
            }
            first = false;
            let outputs = reqs
                .iter()
                .map(|r| tp.eval_discipline(r.discipline, &r.x))
                .collect();
            solver.tell(outputs).unwrap();
            requests = solver.ask().unwrap();
        }
        let result = solver.result().unwrap();
        assert!(n_enrichments > 0);
        assert_eq!(
            result.n_evaluations.iter().sum::<usize>(),
            9 + n_enrichments,
            "one evaluation per enrichment"
        );
    }

    /// Whether the estimated minimum value is within `rel` of the reference
    /// (the papers' convergence criterion)
    fn converged(tp: &problems::TestProblem, result: &EgmdoResult, rel: f64) -> bool {
        (result.y_min_mean() - tp.f_ref).abs() < rel * tp.f_ref.abs()
    }

    #[test]
    fn test_constrained_toy_problem_robustness() {
        // Constrained optimum z* = -2.403 (C-EGMDO paper: -2.42), with the constraint active.
        // About 85% of the runs converge within 1% (17 of seeds 0..20).
        let tp = problems::toy_constrained();
        let mut summary = vec![];
        for seed in 0..10 {
            let result =
                minimize(&tp.problem, &tp.disciplines, toy_constrained_config(seed)).unwrap();
            summary.push((
                seed,
                converged(&tp, &result, 0.01),
                result.z_star_mean()[0],
                result.y_min_mean(),
                result.n_evaluations.clone(),
            ));
        }
        let n_converged = summary.iter().filter(|s| s.1).count();
        assert!(
            n_converged >= 6,
            "runs (seed, converged, E[z*], E[y_min], evaluations): {summary:?}"
        );
        // The converged runs find the constrained optimum, not the unconstrained one (-3.0)
        for s in summary.iter().filter(|s| s.1) {
            assert!((s.2 - tp.z_ref[0]).abs() < 0.2, "{summary:?}");
        }
    }

    #[test]
    fn test_constrained_sellar_problem() {
        // Setting of the C-EGMDO paper (section 4.1): 96% of the runs converge within 1% with
        // 22.8 disciplinary evaluations on average. Observed: 19 of seeds 0..20, 24.4 evaluations.
        let tp = problems::sellar_constrained();
        let mut summary = vec![];
        for seed in 0..4 {
            let config = EgmdoConfig {
                n_doe_disciplines: Some(vec![5, 5]),
                n_doe_uq: 20,
                max_iters: 10,
                uq: UqSettings {
                    n_mc: Some(200),
                    ..UqSettings::default()
                },
                seed: Some(seed),
                ..EgmdoConfig::default()
            };
            let result = minimize(&tp.problem, &tp.disciplines, config).unwrap();
            let (_, qoi, _) = tp.evaluate(&result.z_best.view()).unwrap();
            summary.push((
                seed,
                converged(&tp, &result, 0.01),
                result.z_best.clone(),
                qoi,
                result.n_evaluations.iter().sum::<usize>(),
            ));
        }
        assert!(
            summary.iter().filter(|s| s.1).count() >= 3,
            "runs (seed, converged, z_best, true qoi at z_best, evaluations): {summary:?}"
        );
        assert!(summary.iter().all(|s| s.4 <= 40), "{summary:?}");
    }
}

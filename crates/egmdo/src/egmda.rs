//! EGMDA: the disciplinary GPs and their enrichment as the solver of the multidisciplinary
//! analysis at a design point (specification, part III, issue #518).
//!
//! At a design point `z`, the MDA is solved on the means of the disciplinary GPs, giving
//! `y_0(z)`, and linearized around it: the coupling variables are approximately
//! `N(y_0, Σ)` with `Σ = L Lᵀ` and `L = (I - J)^-1 diag(σ)` (decision D37). While the
//! dispersion of this distribution is above a threshold, the disciplinary GPs are enriched at
//! `(z, y_0)` and the analysis is repeated. Then `qoi(z, y_0)` is the result of the analysis.
//!
//! * [EgmdaSolver] analyzes design points one by one, as an ask/tell solver; the disciplinary
//!   GPs persist from one point to the next.
//! * [MdaFunction] wraps it with the true disciplinary solvers as a function of the design
//!   variables, to be minimized by any optimizer.
//! * [EgmdaOptimizer] minimizes the problem with `Egor` in the design variables and EGMDA at
//!   each point, as an ask/tell solver; [minimize] runs it with the true disciplinary solvers.

use crate::criteria::Feasibility;
use crate::errors::{EgmdoError, Result};
use crate::mda::{MdaSettings, SurrogateCouplings, solve_one};
use crate::problem::{DisciplineFn, MdoProblem};
use crate::solver::{DOE_UQ_DUPLICATE_TOLERANCE, Enrichment, EvaluationRequest};
use crate::surrogates::{DisciplinaryModels, GpSettings, initial_doe};
use crate::uq::linearization_factor;
use egobox_doe::{Lhs, SamplingMethod};
use egobox_ego::EgorServiceBuilder;
use linfa_linalg::svd::SVD;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate, s};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// The dispersion criterion of the linearized coupling variables `N(y_0, Σ)`, with its
/// threshold (decision D41). All three are in the units of the coupling variables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dispersion {
    /// `sqrt(Tr Σ) / ‖y_0‖ <= ε_rel`
    Relative(f64),
    /// `sqrt(Tr Σ) <= ε_abs`
    Absolute(f64),
    /// `sqrt(λ_max(Σ)) <= ε_abs`
    Spectral(f64),
}

impl Default for Dispersion {
    fn default() -> Self {
        Dispersion::Relative(1e-4)
    }
}

impl Dispersion {
    /// The threshold of the criterion
    pub fn threshold(&self) -> f64 {
        match *self {
            Dispersion::Relative(eps) | Dispersion::Absolute(eps) | Dispersion::Spectral(eps) => {
                eps
            }
        }
    }

    /// The value of the criterion for the factor `L` of `Σ = L Lᵀ` and the mean MDA solution
    /// `y0`: `Tr Σ = ‖L‖_F²` and `λ_max(Σ) = s_max(L)²`
    pub fn value(&self, l: &Array2<f64>, y0: &Array1<f64>) -> f64 {
        let trace = l.mapv(|v| v * v).sum();
        match self {
            Dispersion::Relative(_) => trace.sqrt() / y0.dot(y0).sqrt().max(f64::MIN_POSITIVE),
            Dispersion::Absolute(_) => trace.sqrt(),
            Dispersion::Spectral(_) => match l.svd(false, false) {
                Ok((_, s, _)) => s.fold(0f64, |m, &v| m.max(v)),
                Err(_) => f64::INFINITY,
            },
        }
    }
}

/// The share of each discipline in the variance of the linearized coupling variables
/// `Tr Σ = ‖L‖_F²`: `Σ_{q ∈ u_i} ‖L[:, q]‖² / ‖L‖_F²`, where `u_i` are the outputs of
/// discipline `i` (decision D42). The variables `ξ_q` being independent and the model linear,
/// it is the closed Sobol index of the outputs of the discipline.
pub fn discipline_shares(problem: &MdoProblem, l: &Array2<f64>) -> Vec<f64> {
    let column = l.mapv(|v| v * v).sum_axis(Axis(0));
    let total = column.sum();
    problem
        .disciplines()
        .iter()
        .map(|d| {
            let part: f64 = d.coupling_output_indices().iter().map(|&q| column[q]).sum();
            if total > 0. { part / total } else { 0. }
        })
        .collect()
}

/// Configuration of EGMDA
#[derive(Clone, Debug)]
pub struct EgmdaConfig {
    /// The dispersion criterion of the coupling variables, with its threshold
    pub dispersion: Dispersion,
    /// The maximum number of enrichments at a design point (decision D43)
    pub max_enrichments: usize,
    /// The disciplines enriched at each enrichment: the discipline with the largest share of
    /// the coupling variance (`Sobol`), or every discipline (`All`)
    pub enrichment: Enrichment,
    /// The tolerance of each constraint `[g..., h...]`; `None` gives `1e-4`
    pub constraint_tolerances: Option<Vec<f64>>,
    /// Size of the initial design of experiments of each discipline;
    /// `None` gives `2 d_i` for a discipline with `d_i` inputs
    pub n_doe_disciplines: Option<Vec<usize>>,
    /// Initial designs of experiments of the disciplines, as inputs `(n_i, d_i)` to be
    /// evaluated; overrides `n_doe_disciplines`
    pub doe_disciplines: Option<Vec<Array2<f64>>>,
    /// Size of the initial design of experiments in the design variables of the optimization;
    /// `None` gives `Egor`'s `max(nx + 1, 5)`
    pub n_doe: Option<usize>,
    /// Initial design of experiments `(n, nx)` of the optimization; overrides `n_doe`
    pub doe: Option<Array2<f64>>,
    /// Number of iterations of the optimization, after its initial design of experiments
    pub max_iters: usize,
    /// Optional budget of true disciplinary evaluations, all disciplines included
    pub max_evaluations: Option<usize>,
    /// GP settings of the disciplinary GPs
    pub gp: GpSettings,
    /// MDA settings
    pub mda: MdaSettings,
    /// Seed of all random draws; `None` for a non-reproducible run
    pub seed: Option<u64>,
}

impl Default for EgmdaConfig {
    fn default() -> Self {
        EgmdaConfig {
            dispersion: Dispersion::default(),
            max_enrichments: 20,
            enrichment: Enrichment::default(),
            constraint_tolerances: None,
            n_doe_disciplines: None,
            doe_disciplines: None,
            n_doe: None,
            doe: None,
            max_iters: 20,
            max_evaluations: None,
            gp: GpSettings::default(),
            mda: MdaSettings::default(),
            seed: None,
        }
    }
}

/// How the analysis of a design point ended
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EgmdaStatus {
    /// The dispersion is below its threshold
    Converged,
    /// The maximum number of enrichments at the point is reached
    MaxEnrichments,
    /// The budget of disciplinary evaluations is reached
    Budget,
    /// Every enrichment point duplicates a training point of its discipline
    Stalled,
    /// The mean MDA does not converge, or `I - J` is ill-conditioned: the dispersion cannot
    /// be estimated
    MdaFailure,
}

/// The analysis of a design point
#[derive(Clone, Debug)]
pub struct MdaAnalysis {
    /// The design point
    pub z: Array1<f64>,
    /// The coupling variables of the mean MDA `y_0`
    pub y: Array1<f64>,
    /// The quantities of interest `[f_obj, g..., h...]` at `(z, y_0)`; `None` if the mean MDA
    /// does not converge
    pub qoi: Option<Array1<f64>>,
    /// The value of the dispersion criterion; NaN if it cannot be estimated
    pub dispersion: f64,
    /// The number of enrichments made at the point
    pub n_enrichments: usize,
    /// How the analysis ended
    pub status: EgmdaStatus,
}

/// An enrichment of the disciplinary GPs
#[derive(Clone, Debug)]
pub struct EnrichmentRecord {
    /// The design point
    pub z: Array1<f64>,
    /// The enriched disciplines
    pub disciplines: Vec<usize>,
    /// The dispersion before the enrichment
    pub dispersion: f64,
    /// The share of each discipline in the coupling variance before the enrichment
    pub shares: Vec<f64>,
}

#[derive(Clone, Debug)]
enum SolverState {
    Start,
    AwaitInitialDoe,
    Ready,
    AwaitEnrichment,
}

/// A design point being analyzed
#[derive(Clone, Debug)]
struct Point {
    z: Array1<f64>,
    n_enrichments: usize,
    stalled: bool,
}

/// EGMDA as an ask/tell solver of the MDA at design points
///
/// The first [EgmdaSolver::ask] returns the initial designs of experiments of the
/// disciplines. [EgmdaSolver::start] then starts the analysis of a design point: the next
/// calls to [EgmdaSolver::ask] return the enrichments, and `None` when the analysis is
/// finished, its result being given by [EgmdaSolver::analysis].
pub struct EgmdaSolver {
    problem: MdoProblem,
    config: EgmdaConfig,
    rng: Xoshiro256Plus,
    state: SolverState,
    pending: Vec<EvaluationRequest>,
    models: Option<DisciplinaryModels>,
    point: Option<Point>,
    analysis: Option<MdaAnalysis>,
    history: Vec<EnrichmentRecord>,
}

impl EgmdaSolver {
    /// A solver of the MDA of `problem` with the given configuration; the settings of the
    /// optimization (`n_doe`, `doe`, `max_iters`) are not used
    pub fn new(problem: MdoProblem, config: EgmdaConfig) -> Result<Self> {
        let n_disciplines = problem.disciplines().len();
        if let Some(n) = &config.n_doe_disciplines
            && n.len() != n_disciplines
        {
            return Err(EgmdoError::InvalidData(format!(
                "initial DoE sizes given for {} disciplines, expected {n_disciplines}",
                n.len()
            )));
        }
        if let Some(doe) = &config.doe_disciplines
            && doe.len() != n_disciplines
        {
            return Err(EgmdoError::InvalidData(format!(
                "initial DoEs given for {} disciplines, expected {n_disciplines}",
                doe.len()
            )));
        }
        let threshold = config.dispersion.threshold();
        if threshold.is_nan() || threshold <= 0. {
            return Err(EgmdoError::InvalidData(format!(
                "the dispersion threshold must be positive, got {}",
                threshold
            )));
        }
        let seed = config
            .seed
            .unwrap_or_else(|| Xoshiro256Plus::from_entropy().r#gen());
        Ok(EgmdaSolver {
            problem,
            config,
            rng: Xoshiro256Plus::seed_from_u64(seed),
            state: SolverState::Start,
            pending: vec![],
            models: None,
            point: None,
            analysis: None,
            history: vec![],
        })
    }

    /// The problem
    pub fn problem(&self) -> &MdoProblem {
        &self.problem
    }

    /// Starts the analysis of design point `z`; the analysis of the previous point, if not
    /// finished, is abandoned
    pub fn start(&mut self, z: &ArrayView1<f64>) -> Result<()> {
        if matches!(
            self.state,
            SolverState::AwaitInitialDoe | SolverState::AwaitEnrichment
        ) {
            return Err(EgmdoError::InvalidData(
                "evaluations were requested: call tell() before start()".to_string(),
            ));
        }
        if z.len() != self.problem.nx() {
            return Err(EgmdoError::InvalidData(format!(
                "design point of dimension {}, expected {}",
                z.len(),
                self.problem.nx()
            )));
        }
        self.point = Some(Point {
            z: z.to_owned(),
            n_enrichments: 0,
            stalled: false,
        });
        self.analysis = None;
        Ok(())
    }

    /// Returns the true disciplinary evaluations needed: first the initial designs of
    /// experiments of the disciplines, then the enrichments of the started analysis.
    /// Returns `None` when there is nothing to evaluate: the analysis of the started point,
    /// if any, is then given by [EgmdaSolver::analysis].
    pub fn ask(&mut self) -> Result<Option<Vec<EvaluationRequest>>> {
        match self.state {
            SolverState::Start => {
                self.pending = self.initial_requests();
                self.state = SolverState::AwaitInitialDoe;
                Ok(Some(self.pending.clone()))
            }
            SolverState::Ready => {
                if self.point.is_none() {
                    return Ok(None);
                }
                match self.step()? {
                    Some(requests) => {
                        self.pending = requests;
                        self.state = SolverState::AwaitEnrichment;
                        Ok(Some(self.pending.clone()))
                    }
                    None => Ok(None),
                }
            }
            SolverState::AwaitInitialDoe | SolverState::AwaitEnrichment => {
                Err(EgmdoError::InvalidData(
                    "evaluations were requested: call tell() before ask()".to_string(),
                ))
            }
        }
    }

    /// Gives the outputs of the evaluations requested by the last [EgmdaSolver::ask]:
    /// `outputs[k]` holds the outputs of request `k`, one row per input point.
    pub fn tell(&mut self, outputs: Vec<Array2<f64>>) -> Result<()> {
        if outputs.len() != self.pending.len() {
            return Err(EgmdoError::InvalidData(format!(
                "{} outputs given for {} requests",
                outputs.len(),
                self.pending.len()
            )));
        }
        match self.state {
            SolverState::AwaitInitialDoe => {
                let requests = std::mem::take(&mut self.pending);
                let data = requests.into_iter().map(|r| r.x).zip(outputs).collect();
                self.models = Some(DisciplinaryModels::fit(
                    &self.problem,
                    data,
                    self.config.gp.clone(),
                )?);
            }
            SolverState::AwaitEnrichment => {
                let requests = std::mem::take(&mut self.pending);
                let models = self.models.as_mut().expect("models");
                let mut added = 0;
                for (r, y) in requests.iter().zip(outputs.iter()) {
                    added += models.add_points(&self.problem, r.discipline, &r.x, y)?;
                }
                let point = self.point.as_mut().expect("point under analysis");
                point.n_enrichments += 1;
                point.stalled = added == 0;
            }
            SolverState::Start | SolverState::Ready => {
                return Err(EgmdoError::InvalidData(
                    "no evaluation was requested: call ask() first".to_string(),
                ));
            }
        }
        self.state = SolverState::Ready;
        Ok(())
    }

    /// The analysis of the last started point, once [EgmdaSolver::ask] has returned `None`
    pub fn analysis(&self) -> Option<&MdaAnalysis> {
        self.analysis.as_ref()
    }

    /// The disciplinary GPs, once the initial designs of experiments are evaluated
    pub fn models(&self) -> Option<&DisciplinaryModels> {
        self.models.as_ref()
    }

    /// The enrichments so far
    pub fn history(&self) -> &[EnrichmentRecord] {
        &self.history
    }

    /// The number of true evaluations of each discipline so far
    pub fn n_evaluations(&self) -> Vec<usize> {
        (0..self.problem.disciplines().len())
            .map(|i| self.models.as_ref().map_or(0, |m| m.n_evaluations(i)))
            .collect()
    }

    /// Analyzes design point `z` with the true disciplinary solvers `disciplines`, evaluating
    /// the initial designs of experiments of the disciplines first if needed
    pub fn analyze(
        &mut self,
        z: &ArrayView1<f64>,
        disciplines: &[Box<dyn DisciplineFn>],
    ) -> Result<MdaAnalysis> {
        if disciplines.len() != self.problem.disciplines().len() {
            return Err(EgmdoError::InvalidData(format!(
                "{} disciplinary solvers given, expected {}",
                disciplines.len(),
                self.problem.disciplines().len()
            )));
        }
        self.start(z)?;
        while let Some(requests) = self.ask()? {
            let outputs = requests
                .iter()
                .map(|r| (disciplines[r.discipline])(&r.x.view()))
                .collect();
            self.tell(outputs)?;
        }
        Ok(self.analysis.clone().expect("finished analysis"))
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

    /// One step of the analysis of the started point: the enrichment requests, or `None`
    /// when the analysis is finished
    fn step(&mut self) -> Result<Option<Vec<EvaluationRequest>>> {
        let point = self.point.clone().expect("point under analysis");
        let models = self.models.as_ref().expect("models");
        let z = point.z.view();
        let (y0, mda_status) = solve_one(
            &SurrogateCouplings::mean(&self.problem, models),
            &z,
            &self.config.mda,
        )?;
        if !mda_status.is_converged() {
            log::warn!("Mean MDA not converged at z = {z} ({mda_status:?})");
            // Enrichment of every discipline at the last iterate, where the GPs fail (D43)
            if y0.iter().all(|v| v.is_finite())
                && !point.stalled
                && point.n_enrichments < self.config.max_enrichments
            {
                let order: Vec<usize> = (0..self.problem.disciplines().len()).collect();
                let requests = self.requests_at(&z, &y0, &order);
                if !requests.is_empty() && self.within_budget(requests.len()) {
                    self.history.push(EnrichmentRecord {
                        z: point.z,
                        disciplines: requests.iter().map(|r| r.discipline).collect(),
                        dispersion: f64::NAN,
                        shares: vec![],
                    });
                    return Ok(Some(requests));
                }
            }
            self.finish(y0, None, f64::NAN, EgmdaStatus::MdaFailure);
            return Ok(None);
        }
        let qoi = self
            .problem
            .eval_qoi(&z.insert_axis(Axis(0)), &y0.view().insert_axis(Axis(0)))?
            .row(0)
            .to_owned();
        let Some(l) = linearization_factor(&self.problem, models, &z, &y0)? else {
            self.finish(y0, Some(qoi), f64::NAN, EgmdaStatus::MdaFailure);
            return Ok(None);
        };
        let dispersion = self.config.dispersion.value(&l, &y0);
        let status = if dispersion <= self.config.dispersion.threshold() {
            Some(EgmdaStatus::Converged)
        } else if point.stalled {
            Some(EgmdaStatus::Stalled)
        } else if point.n_enrichments >= self.config.max_enrichments {
            Some(EgmdaStatus::MaxEnrichments)
        } else {
            None
        };
        if let Some(status) = status {
            self.finish(y0, Some(qoi), dispersion, status);
            return Ok(None);
        }

        // Enrichment at (z, y_0) of the disciplines contributing to the coupling variance,
        // the largest share first; points duplicating a training point are not requested
        let shares = discipline_shares(&self.problem, &l);
        let mut order: Vec<usize> = (0..shares.len()).filter(|&i| shares[i] > 0.).collect();
        order.sort_by(|&i, &j| shares[j].total_cmp(&shares[i]));
        let mut requests = self.requests_at(&z, &y0, &order);
        if self.config.enrichment == Enrichment::Sobol {
            requests.truncate(1);
        }
        if requests.is_empty() {
            self.finish(y0, Some(qoi), dispersion, EgmdaStatus::Stalled);
            return Ok(None);
        }
        if !self.within_budget(requests.len()) {
            self.finish(y0, Some(qoi), dispersion, EgmdaStatus::Budget);
            return Ok(None);
        }
        self.history.push(EnrichmentRecord {
            z: point.z,
            disciplines: requests.iter().map(|r| r.discipline).collect(),
            dispersion,
            shares,
        });
        Ok(Some(requests))
    }

    /// The enrichment requests of the disciplines `order` at `(z, y)`, but for the points
    /// duplicating a training point of their discipline (D14)
    fn requests_at(
        &self,
        z: &ArrayView1<f64>,
        y: &Array1<f64>,
        order: &[usize],
    ) -> Vec<EvaluationRequest> {
        let models = self.models.as_ref().expect("models");
        let y = y.view().insert_axis(Axis(0));
        order
            .iter()
            .filter_map(|&i| {
                let x = self.problem.disciplines()[i].inputs(z, &y);
                (!models.is_duplicate(i, &x.row(0)))
                    .then_some(EvaluationRequest { discipline: i, x })
            })
            .collect()
    }

    /// Whether `n` more evaluations fit in the budget of disciplinary evaluations
    fn within_budget(&self, n: usize) -> bool {
        match self.config.max_evaluations {
            Some(budget) if self.n_evaluations().iter().sum::<usize>() + n > budget => {
                log::info!("Budget of {budget} disciplinary evaluations reached");
                false
            }
            _ => true,
        }
    }

    fn finish(
        &mut self,
        y: Array1<f64>,
        qoi: Option<Array1<f64>>,
        dispersion: f64,
        status: EgmdaStatus,
    ) {
        let point = self.point.take().expect("point under analysis");
        self.analysis = Some(MdaAnalysis {
            z: point.z,
            y,
            qoi,
            dispersion,
            n_enrichments: point.n_enrichments,
            status,
        });
    }
}

/// EGMDA with the true disciplinary solvers, as a function of the design variables, to be
/// minimized by any optimizer. Clones share the same disciplinary GPs.
///
/// With `Egor`, which expects constraints `c <= 0`, an equality constraint `h` is given as the
/// two inequality constraints `h` and `-h`.
#[derive(Clone)]
pub struct MdaFunction {
    solver: Arc<Mutex<EgmdaSolver>>,
    disciplines: Arc<Vec<Box<dyn DisciplineFn>>>,
}

impl MdaFunction {
    /// EGMDA on `problem` with the true disciplinary solvers `disciplines`; the settings of
    /// the optimization (`n_doe`, `doe`, `max_iters`) are not used
    pub fn new(
        problem: MdoProblem,
        disciplines: Arc<Vec<Box<dyn DisciplineFn>>>,
        config: EgmdaConfig,
    ) -> Result<Self> {
        if disciplines.len() != problem.disciplines().len() {
            return Err(EgmdoError::InvalidData(format!(
                "{} disciplinary solvers given, expected {}",
                disciplines.len(),
                problem.disciplines().len()
            )));
        }
        Ok(MdaFunction {
            solver: Arc::new(Mutex::new(EgmdaSolver::new(problem, config)?)),
            disciplines,
        })
    }

    /// The quantities of interest `[f_obj, g..., h...]` at the design points `z` `(n, nx)`,
    /// one row per point. Fails if the analysis of a point fails ([EgmdaStatus::MdaFailure]).
    pub fn evaluate(&self, z: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut solver = self.solver.lock().expect("EGMDA solver lock");
        let mut qoi = Array2::zeros((z.nrows(), solver.problem().n_qoi()));
        for (zk, mut row) in z.rows().into_iter().zip(qoi.rows_mut()) {
            let analysis = solver.analyze(&zk, &self.disciplines)?;
            match analysis.qoi {
                Some(q) if analysis.status != EgmdaStatus::MdaFailure => row.assign(&q),
                _ => {
                    return Err(EgmdoError::MdaFailure(format!(
                        "the dispersion of the MDA cannot be estimated at z = {zk}"
                    )));
                }
            }
        }
        Ok(qoi)
    }

    /// The number of true evaluations of each discipline so far
    pub fn n_evaluations(&self) -> Vec<usize> {
        self.solver
            .lock()
            .expect("EGMDA solver lock")
            .n_evaluations()
    }

    /// The enrichments so far
    pub fn history(&self) -> Vec<EnrichmentRecord> {
        self.solver
            .lock()
            .expect("EGMDA solver lock")
            .history()
            .to_vec()
    }
}

/// The analysis of a design point in the optimization
#[derive(Clone, Debug)]
pub struct PointRecord {
    /// The iteration; 0 for the initial design of experiments
    pub iteration: usize,
    /// The analysis of the point
    pub analysis: MdaAnalysis,
    /// Whether the point was added to the data of `Egor`: not if its analysis failed or if it
    /// duplicates a point of the data (decision D44)
    pub accepted: bool,
    /// The number of true evaluations of each discipline after the analysis
    pub n_evaluations: Vec<usize>,
}

/// The result of an optimization with EGMDA
#[derive(Clone)]
pub struct EgmdaResult {
    /// The point with the lowest objective among the feasible points of the data, or with the
    /// lowest violation if none is
    pub z_best: Array1<f64>,
    /// The objective at `z_best`
    pub f_best: f64,
    /// The quantities of interest `[f_obj, g..., h...]` at `z_best`
    pub qoi_best: Array1<f64>,
    /// Whether the constraints are satisfied at `z_best`
    pub best_feasible: bool,
    /// The design points of the data of `Egor`, one per row
    pub x_doe: Array2<f64>,
    /// The quantities of interest `[f_obj, g..., h...]` at the points of `x_doe`
    pub y_doe: Array2<f64>,
    /// Number of true evaluations of each discipline
    pub n_evaluations: Vec<usize>,
    /// Every analyzed design point, in order
    pub history: Vec<PointRecord>,
    /// Every enrichment of the disciplinary GPs, in order
    pub enrichments: Vec<EnrichmentRecord>,
    /// The final disciplinary GPs
    pub models: DisciplinaryModels,
}

/// The minimization of an MDO problem with `Egor` in the design variables and EGMDA at each
/// design point, as an ask/tell solver of the true disciplinary evaluations (decision D44)
///
/// The initial design of experiments in the design variables is analyzed first, then each
/// iteration analyzes the point suggested by `Egor` from the analyzed points.
pub struct EgmdaOptimizer {
    problem: MdoProblem,
    config: EgmdaConfig,
    solver: EgmdaSolver,
    feasibility: Feasibility,
    egor_seed: u64,
    queue: VecDeque<Array1<f64>>,
    analyzing: Option<usize>,
    x: Array2<f64>,
    y: Array2<f64>,
    iter: usize,
    budget_exhausted: bool,
    history: Vec<PointRecord>,
    result: Option<EgmdaResult>,
}

impl EgmdaOptimizer {
    /// An optimizer of `problem` with the given configuration
    pub fn new(problem: MdoProblem, config: EgmdaConfig) -> Result<Self> {
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
        if let Some(doe) = &config.doe
            && doe.ncols() != problem.nx()
        {
            return Err(EgmdoError::InvalidData(format!(
                "the initial DoE must have {} columns, got {}",
                problem.nx(),
                doe.ncols()
            )));
        }
        let seed = config
            .seed
            .unwrap_or_else(|| Xoshiro256Plus::from_entropy().r#gen());
        let mut rng = Xoshiro256Plus::seed_from_u64(seed);
        let solver = EgmdaSolver::new(
            problem.clone(),
            EgmdaConfig {
                seed: Some(rng.r#gen()),
                ..config.clone()
            },
        )?;
        let doe = match &config.doe {
            Some(doe) => doe.clone(),
            None => Lhs::new(problem.xlimits())
                .with_rng(Xoshiro256Plus::seed_from_u64(rng.r#gen()))
                .sample(config.n_doe.unwrap_or((problem.nx() + 1).max(5))),
        };
        Ok(EgmdaOptimizer {
            feasibility: Feasibility::new(problem.n_ineq(), tolerances, 1.),
            egor_seed: rng.r#gen(),
            queue: doe.rows().into_iter().map(|r| r.to_owned()).collect(),
            analyzing: None,
            x: Array2::zeros((0, problem.nx())),
            y: Array2::zeros((0, problem.n_qoi())),
            iter: 0,
            budget_exhausted: false,
            history: vec![],
            result: None,
            problem,
            config,
            solver,
        })
    }

    /// The problem
    pub fn problem(&self) -> &MdoProblem {
        &self.problem
    }

    /// Runs the optimization until it needs true disciplinary evaluations, and returns them.
    /// Returns `None` when the optimization is finished; the result is then given by
    /// [EgmdaOptimizer::result].
    pub fn ask(&mut self) -> Result<Option<Vec<EvaluationRequest>>> {
        loop {
            if self.result.is_some() {
                return Ok(None);
            }
            if let Some(iteration) = self.analyzing {
                if let Some(requests) = self.solver.ask()? {
                    return Ok(Some(requests));
                }
                self.analyzing = None;
                self.record(iteration);
                continue;
            }
            match self.next_point()? {
                Some((iteration, z)) => {
                    self.solver.start(&z.view())?;
                    self.analyzing = Some(iteration);
                }
                None => self.finalize()?,
            }
        }
    }

    /// Gives the outputs of the evaluations requested by the last [EgmdaOptimizer::ask]:
    /// `outputs[k]` holds the outputs of request `k`, one row per input point.
    pub fn tell(&mut self, outputs: Vec<Array2<f64>>) -> Result<()> {
        self.solver.tell(outputs)
    }

    /// The result, once [EgmdaOptimizer::ask] has returned `None`
    pub fn result(&self) -> Result<EgmdaResult> {
        self.result
            .clone()
            .ok_or_else(|| EgmdoError::InvalidData("the optimization is not finished".to_string()))
    }

    /// The number of true evaluations of each discipline so far
    pub fn n_evaluations(&self) -> Vec<usize> {
        self.solver.n_evaluations()
    }

    /// The next design point to analyze with its iteration, or `None` when the optimization
    /// is finished
    fn next_point(&mut self) -> Result<Option<(usize, Array1<f64>)>> {
        if self.budget_exhausted {
            return Ok(None);
        }
        if let Some(z) = self.queue.pop_front() {
            return Ok(Some((0, z)));
        }
        if self.iter >= self.config.max_iters {
            return Ok(None);
        }
        if self.x.nrows() < 2 {
            return Err(EgmdoError::MdaFailure(format!(
                "the MDA is solved at {} design points of the initial DoE, at least 2 are needed",
                self.x.nrows()
            )));
        }
        self.iter += 1;
        Ok(Some((self.iter, self.suggest()?)))
    }

    /// The point suggested by `Egor` from the analyzed points (decision D44)
    fn suggest(&self) -> Result<Array1<f64>> {
        let n_ineq = self.problem.n_ineq();
        // Equality constraints |h| <= τ as h <= τ and -h <= τ
        let y = concatenate![Axis(1), self.y, -&self.y.slice(s![.., 1 + n_ineq..])];
        let tolerances = self.feasibility.tolerances();
        let tolerances = concatenate![Axis(0), *tolerances, tolerances.slice(s![n_ineq..])];
        let n_cstr = tolerances.len();
        let seed = self.egor_seed.wrapping_add(self.iter as u64);
        let egor = EgorServiceBuilder::optimize()
            .configure(|config| {
                let config = config.n_cstr(n_cstr).seed(seed);
                if n_cstr > 0 {
                    config.cstr_tol(tolerances)
                } else {
                    config
                }
            })
            .min_within(self.problem.xlimits())
            .map_err(|e| EgmdoError::InvalidData(format!("Egor configuration: {e}")))?;
        Ok(egor.suggest(&self.x, &y).row(0).to_owned())
    }

    /// Records the finished analysis, and adds its point to the data of `Egor` if it
    /// succeeded and is not a duplicate
    fn record(&mut self, iteration: usize) {
        let analysis = self.solver.analysis().cloned().expect("finished analysis");
        if analysis.status == EgmdaStatus::Budget {
            self.budget_exhausted = true;
        }
        let accepted = match &analysis.qoi {
            Some(qoi) if analysis.status != EgmdaStatus::MdaFailure => {
                if self.is_duplicate(&analysis.z.view()) {
                    log::warn!("Design point {} already analyzed: not added", analysis.z);
                    false
                } else {
                    self.x
                        .push_row(analysis.z.view())
                        .expect("design point dimension");
                    self.y.push_row(qoi.view()).expect("qoi dimension");
                    true
                }
            }
            _ => {
                log::warn!("Design point {} rejected: MDA failure", analysis.z);
                false
            }
        };
        self.history.push(PointRecord {
            iteration,
            analysis,
            accepted,
            n_evaluations: self.solver.n_evaluations(),
        });
    }

    /// Whether `z` is closer than [DOE_UQ_DUPLICATE_TOLERANCE] to a point of the data, in
    /// design variables normalized to `[0, 1]`
    fn is_duplicate(&self, z: &ArrayView1<f64>) -> bool {
        let xlimits = self.problem.xlimits();
        let scale = &xlimits.column(1) - &xlimits.column(0);
        self.x
            .rows()
            .into_iter()
            .any(|x| ((&x - z) / &scale).mapv(|v| v * v).sum().sqrt() < DOE_UQ_DUPLICATE_TOLERANCE)
    }

    fn finalize(&mut self) -> Result<()> {
        if self.x.nrows() == 0 {
            return Err(EgmdoError::MdaFailure(
                "the MDA is solved at no design point".to_string(),
            ));
        }
        // The best point, among the feasible ones if any
        let cstr = self.y.slice(s![.., 1..]);
        let feasible: Vec<usize> = (0..self.y.nrows())
            .filter(|&i| self.feasibility.is_feasible(&cstr.row(i)))
            .collect();
        let best_feasible = !feasible.is_empty();
        let best = if best_feasible {
            feasible
                .into_iter()
                .min_by(|&i, &j| self.y[[i, 0]].total_cmp(&self.y[[j, 0]]))
        } else {
            (0..self.y.nrows()).min_by(|&i, &j| {
                self.feasibility
                    .violation(&cstr.row(i))
                    .total_cmp(&self.feasibility.violation(&cstr.row(j)))
            })
        }
        .expect("non-empty data");
        self.result = Some(EgmdaResult {
            z_best: self.x.row(best).to_owned(),
            f_best: self.y[[best, 0]],
            qoi_best: self.y.row(best).to_owned(),
            best_feasible,
            x_doe: self.x.clone(),
            y_doe: self.y.clone(),
            n_evaluations: self.solver.n_evaluations(),
            history: self.history.clone(),
            enrichments: self.solver.history().to_vec(),
            models: self.solver.models().cloned().expect("models"),
        });
        Ok(())
    }
}

/// Minimizes `problem` with `Egor` in the design variables and EGMDA at each design point,
/// with the true disciplinary solvers `disciplines` (`disciplines[i]` is the solver of
/// discipline `i`).
pub fn minimize(
    problem: &MdoProblem,
    disciplines: &[Box<dyn DisciplineFn>],
    config: EgmdaConfig,
) -> Result<EgmdaResult> {
    if disciplines.len() != problem.disciplines().len() {
        return Err(EgmdoError::InvalidData(format!(
            "{} disciplinary solvers given, expected {}",
            disciplines.len(),
            problem.disciplines().len()
        )));
    }
    let mut optimizer = EgmdaOptimizer::new(problem.clone(), config)?;
    while let Some(requests) = optimizer.ask()? {
        let outputs = requests
            .iter()
            .map(|r| (disciplines[r.discipline])(&r.x.view()))
            .collect();
        optimizer.tell(outputs)?;
    }
    optimizer.result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::{self, TestProblem};
    use approx::assert_abs_diff_eq;
    use egobox_ego::EgorBuilder;
    use linfa_linalg::eigh::EighInto;
    use ndarray::array;

    fn toy_config(seed: u64) -> EgmdaConfig {
        EgmdaConfig {
            n_doe_disciplines: Some(vec![5, 4]),
            seed: Some(seed),
            ..EgmdaConfig::default()
        }
    }

    /// Whether the true MDA at `z` gives an objective within 1% of the reference, with the
    /// constraints satisfied within `1e-3`
    fn is_optimal(tp: &TestProblem, z: &Array1<f64>) -> bool {
        let (_, qoi, _) = tp.evaluate(&z.view()).unwrap();
        let n_ineq = tp.problem.n_ineq();
        (qoi[0] - tp.f_ref).abs() <= 1e-2 * tp.f_ref.abs()
            && qoi.slice(s![1..1 + n_ineq]).iter().all(|&g| g <= 1e-3)
    }

    #[test]
    fn test_dispersion_from_factor() {
        let tp = problems::toy();
        let l = array![[0.3, -1.2], [0.7, 0.4]];
        let y0 = array![2., -1.];
        let sigma = l.dot(&l.t());
        let trace = sigma[[0, 0]] + sigma[[1, 1]];
        let (eigenvalues, _) = sigma.eigh_into().unwrap();
        let lambda_max = eigenvalues.fold(0f64, |m, &v| m.max(v));
        assert_abs_diff_eq!(
            Dispersion::Relative(1.).value(&l, &y0),
            trace.sqrt() / 5f64.sqrt(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            Dispersion::Absolute(1.).value(&l, &y0),
            trace.sqrt(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            Dispersion::Spectral(1.).value(&l, &y0),
            lambda_max.sqrt(),
            epsilon = 1e-12
        );
        // Discipline i owns coupling i: its share is the variance carried by column i
        let shares = discipline_shares(&tp.problem, &l);
        assert_abs_diff_eq!(shares[0], (0.09 + 0.49) / trace, epsilon = 1e-12);
        assert_abs_diff_eq!(shares.iter().sum::<f64>(), 1., epsilon = 1e-12);
    }

    #[test]
    fn test_analysis_converges() {
        let tp = problems::toy();
        let mut solver = EgmdaSolver::new(tp.problem.clone(), toy_config(0)).unwrap();
        let z = array![-3.];
        let analysis = solver.analyze(&z.view(), &tp.disciplines).unwrap();
        assert_eq!(analysis.status, EgmdaStatus::Converged);
        assert!(analysis.dispersion <= 1e-4);
        assert!(analysis.n_enrichments > 0);
        let (y_true, qoi_true, _) = tp.evaluate(&z.view()).unwrap();
        for (y, t) in analysis.y.iter().zip(y_true.iter()) {
            assert_abs_diff_eq!(*y, *t, epsilon = 1e-2 * (1. + t.abs()));
        }
        assert_abs_diff_eq!(analysis.qoi.unwrap()[0], qoi_true[0], epsilon = 1e-3);
        let history = solver.history();
        assert_eq!(history.len(), analysis.n_enrichments);
        assert!(history[0].dispersion > 1e-4);
        // Sobol enrichment: one discipline at a time
        assert!(history.iter().all(|r| r.disciplines.len() == 1));

        // The same point again: no evaluation
        let n_evaluations = solver.n_evaluations();
        let again = solver.analyze(&z.view(), &tp.disciplines).unwrap();
        assert_eq!(again.status, EgmdaStatus::Converged);
        assert_eq!(again.n_enrichments, 0);
        assert_eq!(solver.n_evaluations(), n_evaluations);
    }

    #[test]
    fn test_enrichment_of_all_disciplines() {
        let tp = problems::toy();
        let config = EgmdaConfig {
            enrichment: Enrichment::All,
            ..toy_config(0)
        };
        let mut solver = EgmdaSolver::new(tp.problem.clone(), config).unwrap();
        let analysis = solver
            .analyze(&array![0.5].view(), &tp.disciplines)
            .unwrap();
        assert_eq!(analysis.status, EgmdaStatus::Converged);
        let mut enriched = solver.history()[0].disciplines.clone();
        enriched.sort();
        assert_eq!(enriched, vec![0, 1]);
    }

    #[test]
    fn test_caps() {
        let tp = problems::toy();
        let z = array![-3.];
        let config = EgmdaConfig {
            max_enrichments: 0,
            ..toy_config(0)
        };
        let mut solver = EgmdaSolver::new(tp.problem.clone(), config).unwrap();
        let analysis = solver.analyze(&z.view(), &tp.disciplines).unwrap();
        assert_eq!(analysis.status, EgmdaStatus::MaxEnrichments);
        assert!(analysis.dispersion > 1e-4);
        assert!(analysis.qoi.is_some());
        assert_eq!(solver.n_evaluations(), vec![5, 4]);

        let config = EgmdaConfig {
            max_evaluations: Some(10),
            ..toy_config(0)
        };
        let mut solver = EgmdaSolver::new(tp.problem.clone(), config).unwrap();
        let analysis = solver.analyze(&z.view(), &tp.disciplines).unwrap();
        assert_eq!(analysis.status, EgmdaStatus::Budget);
        assert_eq!(solver.n_evaluations().iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_enrichment_where_the_mean_mda_fails() {
        // The mean MDA never converges in one iteration: every discipline is enriched at the
        // last iterate, up to the cap, then the point fails
        let tp = problems::toy();
        let config = EgmdaConfig {
            max_enrichments: 2,
            mda: MdaSettings {
                max_iters: 1,
                retry_relaxations: vec![],
                ..MdaSettings::default()
            },
            ..toy_config(0)
        };
        let mut solver = EgmdaSolver::new(tp.problem.clone(), config).unwrap();
        let analysis = solver
            .analyze(&array![0.5].view(), &tp.disciplines)
            .unwrap();
        assert_eq!(analysis.status, EgmdaStatus::MdaFailure);
        assert!(analysis.qoi.is_none());
        assert_eq!(analysis.n_enrichments, 2);
        for record in solver.history() {
            assert_eq!(record.disciplines, vec![0, 1]);
            assert!(record.dispersion.is_nan());
        }
        assert_eq!(solver.n_evaluations(), vec![7, 6]);
    }

    #[test]
    fn test_ask_tell_protocol() {
        let tp = problems::toy();
        let mut solver = EgmdaSolver::new(tp.problem.clone(), toy_config(1)).unwrap();
        // Nothing started: only the initial designs of experiments, then nothing
        let requests = solver.ask().unwrap().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(solver.ask().is_err());
        assert!(solver.start(&array![0.].view()).is_err());
        let outputs = requests
            .iter()
            .map(|r| tp.eval_discipline(r.discipline, &r.x))
            .collect();
        solver.tell(outputs).unwrap();
        assert!(solver.ask().unwrap().is_none());
        assert!(solver.tell(vec![]).is_err());

        let z = array![2.];
        solver.start(&z.view()).unwrap();
        while let Some(requests) = solver.ask().unwrap() {
            let outputs = requests
                .iter()
                .map(|r| tp.eval_discipline(r.discipline, &r.x))
                .collect();
            solver.tell(outputs).unwrap();
        }
        let asked = solver.analysis().unwrap().clone();
        let mut driven = EgmdaSolver::new(tp.problem.clone(), toy_config(1)).unwrap();
        let analysis = driven.analyze(&z.view(), &tp.disciplines).unwrap();
        assert_eq!(asked.y, analysis.y);
        assert_eq!(asked.n_enrichments, analysis.n_enrichments);
        assert_eq!(solver.n_evaluations(), driven.n_evaluations());
    }

    #[test]
    fn test_minimize_toy_problem() {
        let tp = problems::toy();
        let config = EgmdaConfig {
            max_iters: 10,
            ..toy_config(0)
        };
        let result = minimize(&tp.problem, &tp.disciplines, config).unwrap();
        assert!(
            is_optimal(&tp, &result.z_best),
            "z_best = {}, f_best = {}",
            result.z_best,
            result.f_best
        );
        assert_eq!(result.x_doe.nrows(), result.y_doe.nrows());
        assert_eq!(result.history.len(), 5 + 10);
        assert_eq!(
            result.n_evaluations,
            result.history.last().unwrap().n_evaluations
        );
    }

    #[test]
    fn test_minimize_constrained_sellar_problem() {
        let tp = problems::sellar_constrained();
        let config = EgmdaConfig {
            seed: Some(0),
            ..EgmdaConfig::default()
        };
        let result = minimize(&tp.problem, &tp.disciplines, config).unwrap();
        assert!(result.best_feasible);
        assert!(
            is_optimal(&tp, &result.z_best),
            "z_best = {}, f_best = {}, n_evaluations = {:?}, true qoi {}",
            result.z_best,
            result.f_best,
            result.n_evaluations,
            tp.evaluate(&result.z_best.view()).unwrap().1
        );
    }

    #[test]
    fn test_optimizer_ask_tell_reproduces_minimize() {
        let tp = problems::toy_constrained();
        let config = EgmdaConfig {
            max_iters: 3,
            ..toy_config(2)
        };
        let mut optimizer = EgmdaOptimizer::new(tp.problem.clone(), config.clone()).unwrap();
        assert!(optimizer.result().is_err());
        while let Some(requests) = optimizer.ask().unwrap() {
            let outputs = requests
                .iter()
                .map(|r| tp.eval_discipline(r.discipline, &r.x))
                .collect();
            optimizer.tell(outputs).unwrap();
        }
        let asked = optimizer.result().unwrap();
        let result = minimize(&tp.problem, &tp.disciplines, config).unwrap();
        assert_eq!(asked.x_doe, result.x_doe);
        assert_eq!(asked.y_doe, result.y_doe);
        assert_eq!(asked.n_evaluations, result.n_evaluations);
    }

    #[test]
    fn test_mda_function_with_egor() {
        let tp = problems::toy();
        let disciplines = Arc::new(problems::toy().disciplines);
        let mda = MdaFunction::new(tp.problem.clone(), disciplines, toy_config(0)).unwrap();
        let f = mda.clone();
        let result = EgorBuilder::optimize(move |z: &ArrayView2<f64>| f.evaluate(z))
            .configure(|config| config.max_iters(10).seed(0))
            .min_within(tp.problem.xlimits())
            .unwrap()
            .run()
            .unwrap();
        assert!(
            is_optimal(&tp, &result.x_opt),
            "x_opt = {}, y_opt = {}",
            result.x_opt,
            result.y_opt
        );
        assert!(!mda.history().is_empty());
        assert!(mda.n_evaluations().iter().sum::<usize>() > 9);
    }
}

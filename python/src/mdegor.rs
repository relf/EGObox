//! `Mdegor`: efficient global multidisciplinary design optimization (EGMDO and its constrained
//! extension C-EGMDO), where the disciplinary solvers are replaced by GP surrogates enriched
//! adaptively, or `Egor` with these surrogates as the solver of the multidisciplinary analysis
//! (EGMDA). See the `egobox-egmdo` crate.
//!
//! The variables are declared by name in Python and resolved here into the indices used by the
//! Rust core. A discipline receives its inputs in the order of its `inputs` names, whereas the
//! core lays them out as `[design inputs, coupling inputs]`: the columns are reordered at the
//! boundary.

use crate::errors::{CallbackError, egmdo_err, install_panic_hook};
use crate::mdegor_config::{EgmdaConfig, EgmdoConfig, SharedSettings};
use crate::types::{CorrelationSpec, repr_kwargs};

use egobox_egmdo as egmdo;
use egobox_egmdo::egmda;
use ndarray::{Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// EnrichmentStrategy specifies which disciplines are evaluated at an enrichment point.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EnrichmentStrategy {
    /// Every discipline, as in the EGMDO paper (Dubreuil et al. 2020)
    All = 1,
    /// Only the discipline with the largest Sobol index of the most uncertain quantity of
    /// interest, as in the C-EGMDO paper (Cardoso et al. 2024)
    Sobol = 2,
}

impl<'a, 'py> FromPyObject<'a, 'py> for EnrichmentStrategy {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::All),
            Ok(2) => Ok(Self::Sobol),
            Ok(v) => Err(PyValueError::new_err(format!(
                "enrichment integer value must be in [1, 2], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "enrichment must be an EnrichmentStrategy enum or an integer in [1, 2]",
            )),
        }
    }
}

/// A discipline of a multidisciplinary design optimization (MDO) problem
///
/// The discipline computes some coupling variables from some design variables and from the
/// coupling variables computed by the other disciplines. Variables are referred to by the names
/// declared in `Mdegor` (`design` and `couplings`).
///
/// Parameters
/// ----------
/// name : str
///     Unique name of the discipline.
/// fun : callable, optional
///     The disciplinary solver, evaluated on a batch of points:
///     `fun(x: array[n, len(inputs)]) -> array[n, len(outputs)]`, the columns of `x` being the
///     `inputs` variables and the returned columns the `outputs` variables, in the given orders.
///     Required by `Mdegor.minimize()`, not by the ask/tell interface.
/// inputs : list of str
///     Names of the design and coupling variables received by the discipline, in any order.
/// outputs : list of str
///     Names of the coupling variables computed by the discipline.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct Discipline {
    name: String,
    fun: Option<Py<PyAny>>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Discipline {
    #[new]
    #[pyo3(signature = (name, fun = None, inputs = vec![], outputs = vec![]))]
    fn new(
        name: String,
        #[gen_stub(override_type(type_repr = "typing.Callable[[numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]] | None", imports = ("typing", "numpy", "numpy.typing")))]
        fun: Option<Py<PyAny>>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Discipline {
            name,
            fun,
            inputs,
            outputs,
        }
    }

    /// Name of the discipline
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Names of the variables received by the discipline, in the column order of its inputs
    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.inputs.clone()
    }

    /// Names of the coupling variables computed by the discipline, in the column order of its
    /// outputs
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.outputs.clone()
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "Discipline",
            &[
                ("name", self.name.clone().into_pyobject(py)?.into_any()),
                ("inputs", self.inputs.clone().into_pyobject(py)?.into_any()),
                (
                    "outputs",
                    self.outputs.clone().into_pyobject(py)?.into_any(),
                ),
            ],
        )
    }
}

/// True evaluations of a discipline requested by `Mdegor.ask()`
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct MdegorRequest {
    /// Index of the discipline in the list given to `Mdegor`
    #[pyo3(get)]
    discipline: usize,
    /// Inputs of the discipline, one point per row, columns in the order of `Discipline.inputs`
    #[pyo3(get)]
    x: Py<PyArray2<f64>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MdegorRequest {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "MdegorRequest",
            &[
                ("discipline", self.discipline.into_pyobject(py)?.into_any()),
                ("x", self.x.bind(py).clone().into_any()),
            ],
        )
    }
}

/// Training inputs and outputs of a discipline
type TrainingData = (Py<PyArray2<f64>>, Py<PyArray2<f64>>);

/// Result of an Mdegor optimization
///
/// With EGMDO (`method="egmdo"`), the minimum is estimated by Monte Carlo on the final model of
/// the objective, which accounts for the remaining uncertainty of the disciplinary surrogates:
/// `z_star` and `y_min` hold samples of the minimum position and value, summarized by their
/// means and standard deviations. With EGMDA (`method="egmda"`), these estimates are None and
/// the uncertainty quantification DoE is the set of points analyzed for `Egor`.
/// Design points have their columns in the order of `design_names`.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct MdegorResult {
    /// The method: "egmdo" or "egmda"
    #[pyo3(get)]
    method: String,
    /// Names of the design variables, in the column order of the design points
    #[pyo3(get)]
    design_names: Vec<String>,
    /// Names of the coupling variables, in the column order of the coupling points
    #[pyo3(get)]
    coupling_names: Vec<String>,
    /// Best point of the uncertainty quantification DoE: the lowest objective mean among the
    /// points whose constraint means are feasible
    #[pyo3(get)]
    z_best: Py<PyArray1<f64>>,
    /// Objective mean at `z_best`
    #[pyo3(get)]
    f_best: f64,
    /// Whether the constraint means are feasible at `z_best`; False when no point of the
    /// uncertainty quantification DoE is feasible and `z_best` has the lowest mean violation
    #[pyo3(get)]
    best_feasible: bool,
    /// Samples of the minimum position, one per row (EGMDO only)
    #[pyo3(get)]
    z_star: Option<Py<PyArray2<f64>>>,
    /// Samples of the minimum value (EGMDO only)
    #[pyo3(get)]
    y_min: Option<Py<PyArray1<f64>>>,
    /// Mean of the minimum position (EGMDO only)
    #[pyo3(get)]
    z_star_mean: Option<Py<PyArray1<f64>>>,
    /// Standard deviation of the minimum position (EGMDO only)
    #[pyo3(get)]
    z_star_std: Option<Py<PyArray1<f64>>>,
    /// Mean of the minimum value (EGMDO only)
    #[pyo3(get)]
    y_min_mean: Option<f64>,
    /// Standard deviation of the minimum value (EGMDO only)
    #[pyo3(get)]
    y_min_std: Option<f64>,
    /// Number of true evaluations of each discipline
    #[pyo3(get)]
    n_evaluations: Vec<usize>,
    /// Number of multidisciplinary analyses solved on the disciplinary surrogates, converged or
    /// not: one per resolution, e.g. one mean and n_mc random analyses per point of the
    /// uncertainty quantification DoE with EGMDO, one per analysis step with EGMDA
    #[pyo3(get)]
    n_mdas: usize,
    /// Final design of experiments of the uncertainty quantification; with EGMDA, the points
    /// analyzed and given to `Egor`
    #[pyo3(get)]
    doe_uq: Py<PyArray2<f64>>,
    /// Means of the quantities of interest (objective, then constraints) at the points of
    /// `doe_uq`, one point per row; with EGMDA, their values with the coupling variables of the
    /// multidisciplinary analysis with the surrogate means
    #[pyo3(get)]
    qoi_mean: Py<PyArray2<f64>>,
    /// Standard deviations of the quantities of interest (objective, then constraints) at the
    /// points of `doe_uq`, one point per row (EGMDO only)
    #[pyo3(get)]
    qoi_std: Option<Py<PyArray2<f64>>>,
    /// Coupling variables of the multidisciplinary analysis with the surrogate means at the
    /// points of `doe_uq`, one point per row, columns in the order of `coupling_names`
    #[pyo3(get)]
    coupling_mean: Py<PyArray2<f64>>,
    /// Training data (x, y) of each discipline, i.e. all its true evaluations, columns in the
    /// order of its `inputs` and `outputs`
    #[pyo3(get)]
    disciplinary_doe: Vec<TrainingData>,
    /// Per-iteration records.
    ///
    /// With EGMDO, one dict per iteration, with keys `z_new` (point added to the uncertainty
    /// quantification DoE, or None), `criterion` (the criterion maximized: "ei", "wb2",
    /// "log_ei" or "feasibility"), `max_criterion` (of the logarithm for "feasibility"),
    /// `wb2_scale`, `n_modes`, `n_enrichments` (in this iteration), `n_failed_mdas`
    /// (non-converged random analyses among those of the current uncertainty quantification
    /// DoE, recomputed after each enrichment: not cumulative), `n_dropped_points`, the
    /// quantities of interest (objective, then constraints) at the proposed point: `qoi_gp`
    /// (objective model expectation and GPs of the constraint means), `qoi_mean_mda`
    /// (multidisciplinary analysis with the surrogate means, None if it does not converge) and
    /// `qoi_pce_mean` (means at `z_new`, None if no point is added), and the cumulative counts
    /// at the end of the iteration since the start of the run: `n_evaluations` (of each
    /// discipline, initial designs of experiments included) and `n_mdas` (see `n_mdas`).
    ///
    /// With EGMDA, one dict per analyzed point, with keys `iteration` (0 for the initial design
    /// of experiments), `z`, `y` (coupling variables of the analysis with the surrogate means),
    /// `qoi` (None if the analysis failed), `dispersion` (of the linearized coupling variables,
    /// NaN if it cannot be estimated), `n_enrichments`, `status` ("converged",
    /// "max_enrichments", "budget", "stalled" or "mda_failure"), `accepted` (whether the point
    /// was given to `Egor`), and the cumulative counts after the analysis: `n_evaluations` (of
    /// each discipline) and `n_mdas` (see `n_mdas`).
    #[pyo3(get)]
    history: Vec<Py<PyDict>>,
    /// With EGMDA, every enrichment of the disciplinary surrogates: dicts with keys `z`,
    /// `disciplines` (indices of the enriched disciplines), `dispersion` (before the
    /// enrichment, NaN where the analysis with the surrogate means does not converge) and
    /// `shares` (of each discipline in the variance of the coupling variables); None with EGMDO
    #[pyo3(get)]
    enrichments: Option<Vec<Py<PyDict>>>,
}

impl MdegorResult {
    fn new(py: Python, mdegor: &Mdegor, result: egmdo::EgmdoResult) -> PyResult<Self> {
        let history = result
            .history
            .iter()
            .map(|h| {
                let d = PyDict::new(py);
                d.set_item("z_new", h.z_new.as_ref().map(|z| z.to_pyarray(py)))?;
                d.set_item(
                    "criterion",
                    match h.criterion {
                        egmdo::AcquisitionCriterion::Ei => "ei",
                        egmdo::AcquisitionCriterion::Wb2 => "wb2",
                        egmdo::AcquisitionCriterion::LogEi => "log_ei",
                        egmdo::AcquisitionCriterion::Feasibility => "feasibility",
                    },
                )?;
                d.set_item("max_criterion", h.max_criterion)?;
                d.set_item("wb2_scale", h.wb2_scale)?;
                d.set_item("n_modes", h.n_modes)?;
                d.set_item("n_enrichments", h.n_enrichments)?;
                d.set_item("n_failed_mdas", h.n_failed_mdas)?;
                d.set_item("n_dropped_points", h.n_dropped_points)?;
                d.set_item("n_evaluations", h.n_evaluations.clone())?;
                d.set_item("n_mdas", h.n_mdas)?;
                for (key, qoi) in [
                    ("qoi_gp", &h.qoi_gp),
                    ("qoi_mean_mda", &h.qoi_mean_mda),
                    ("qoi_pce_mean", &h.qoi_pce_mean),
                ] {
                    d.set_item(key, qoi.as_ref().map(|q| q.to_pyarray(py)))?;
                }
                Ok(d.unbind())
            })
            .collect::<PyResult<_>>()?;
        Ok(MdegorResult {
            method: "egmdo".to_string(),
            design_names: mdegor.design_names.clone(),
            coupling_names: mdegor.coupling_names.clone(),
            z_best: result.z_best.to_pyarray(py).unbind(),
            f_best: result.f_best,
            best_feasible: result.best_feasible,
            z_star_mean: Some(result.z_star_mean().into_pyarray(py).unbind()),
            z_star_std: Some(result.z_star_std().into_pyarray(py).unbind()),
            y_min_mean: Some(result.y_min_mean()),
            y_min_std: Some(result.y_min_std()),
            z_star: Some(result.z_star.to_pyarray(py).unbind()),
            y_min: Some(result.y_min.to_pyarray(py).unbind()),
            n_evaluations: result.n_evaluations.clone(),
            n_mdas: result.n_mdas,
            doe_uq: result.doe_uq.to_pyarray(py).unbind(),
            qoi_mean: result.qoi_mean().into_pyarray(py).unbind(),
            qoi_std: Some(result.qoi_std().into_pyarray(py).unbind()),
            coupling_mean: result.coupling_mean().into_pyarray(py).unbind(),
            disciplinary_doe: disciplinary_doe(py, mdegor, &result.models),
            history,
            enrichments: None,
        })
    }

    fn from_egmda(py: Python, mdegor: &Mdegor, result: egmda::EgmdaResult) -> PyResult<Self> {
        let history = result
            .history
            .iter()
            .map(|p| {
                let a = &p.analysis;
                let d = PyDict::new(py);
                d.set_item("iteration", p.iteration)?;
                d.set_item("z", a.z.to_pyarray(py))?;
                d.set_item("y", a.y.to_pyarray(py))?;
                d.set_item("qoi", a.qoi.as_ref().map(|q| q.to_pyarray(py)))?;
                d.set_item("dispersion", a.dispersion)?;
                d.set_item("n_enrichments", a.n_enrichments)?;
                d.set_item(
                    "status",
                    match a.status {
                        egmda::EgmdaStatus::Converged => "converged",
                        egmda::EgmdaStatus::MaxEnrichments => "max_enrichments",
                        egmda::EgmdaStatus::Budget => "budget",
                        egmda::EgmdaStatus::Stalled => "stalled",
                        egmda::EgmdaStatus::MdaFailure => "mda_failure",
                    },
                )?;
                d.set_item("accepted", p.accepted)?;
                d.set_item("n_evaluations", p.n_evaluations.clone())?;
                d.set_item("n_mdas", p.n_mdas)?;
                Ok(d.unbind())
            })
            .collect::<PyResult<_>>()?;
        let enrichments = result
            .enrichments
            .iter()
            .map(|e| {
                let d = PyDict::new(py);
                d.set_item("z", e.z.to_pyarray(py))?;
                d.set_item("disciplines", e.disciplines.clone())?;
                d.set_item("dispersion", e.dispersion)?;
                d.set_item("shares", e.shares.clone())?;
                Ok(d.unbind())
            })
            .collect::<PyResult<_>>()?;
        // Coupling variables of the accepted points, in the order of x_doe
        let mut coupling_mean = Array2::zeros((0, mdegor.coupling_names.len()));
        for p in result.history.iter().filter(|p| p.accepted) {
            coupling_mean
                .push_row(p.analysis.y.view())
                .expect("coupling dimension");
        }
        Ok(MdegorResult {
            method: "egmda".to_string(),
            design_names: mdegor.design_names.clone(),
            coupling_names: mdegor.coupling_names.clone(),
            z_best: result.z_best.to_pyarray(py).unbind(),
            f_best: result.f_best,
            best_feasible: result.best_feasible,
            z_star_mean: None,
            z_star_std: None,
            y_min_mean: None,
            y_min_std: None,
            z_star: None,
            y_min: None,
            n_evaluations: result.n_evaluations.clone(),
            n_mdas: result.n_mdas,
            doe_uq: result.x_doe.to_pyarray(py).unbind(),
            qoi_mean: result.y_doe.to_pyarray(py).unbind(),
            qoi_std: None,
            coupling_mean: coupling_mean.into_pyarray(py).unbind(),
            disciplinary_doe: disciplinary_doe(py, mdegor, &result.models),
            history,
            enrichments: Some(enrichments),
        })
    }
}

/// Training data of each discipline, inputs in the users' order
fn disciplinary_doe(
    py: Python,
    mdegor: &Mdegor,
    models: &egmdo::DisciplinaryModels,
) -> Vec<TrainingData> {
    mdegor
        .perms
        .iter()
        .enumerate()
        .map(|(i, perm)| {
            let (x, y) = models.training_data(i);
            (
                to_user(&x.view(), perm).into_pyarray(py).unbind(),
                y.to_pyarray(py).unbind(),
            )
        })
        .collect()
}

/// The method of an `Mdegor`, with its settings as given in Python
#[derive(Clone)]
enum MethodSettings {
    Egmdo(EgmdoConfig),
    Egmda(EgmdaConfig),
}

/// The configuration of the method of an `Mdegor`
#[derive(Clone)]
enum MethodConfig {
    Egmdo(egmdo::EgmdoConfig),
    Egmda(egmda::EgmdaConfig),
}

/// The ask/tell solver of the method of an `Mdegor`
enum Solver {
    Egmdo(egmdo::EgmdoSolver),
    Egmda(egmda::EgmdaOptimizer),
}

impl Solver {
    /// A solver of `problem`, with `max_iters` iterations if given
    fn new(
        problem: &egmdo::MdoProblem,
        config: &MethodConfig,
        max_iters: Option<usize>,
    ) -> egmdo::Result<Self> {
        Ok(match config {
            MethodConfig::Egmdo(config) => Solver::Egmdo(egmdo::EgmdoSolver::new(
                problem.clone(),
                egmdo::EgmdoConfig {
                    max_iters: max_iters.unwrap_or(config.max_iters),
                    ..config.clone()
                },
            )?),
            MethodConfig::Egmda(config) => Solver::Egmda(egmda::EgmdaOptimizer::new(
                problem.clone(),
                egmda::EgmdaConfig {
                    max_iters: max_iters.unwrap_or(config.max_iters),
                    ..config.clone()
                },
            )?),
        })
    }

    fn ask(&mut self) -> egmdo::Result<Option<Vec<egmdo::EvaluationRequest>>> {
        match self {
            Solver::Egmdo(solver) => solver.ask(),
            Solver::Egmda(solver) => solver.ask(),
        }
    }

    fn tell(&mut self, outputs: Vec<Array2<f64>>) -> egmdo::Result<()> {
        match self {
            Solver::Egmdo(solver) => solver.tell(outputs),
            Solver::Egmda(solver) => solver.tell(outputs),
        }
    }

    fn result(&self, py: Python, mdegor: &Mdegor) -> PyResult<MdegorResult> {
        match self {
            Solver::Egmdo(solver) => {
                MdegorResult::new(py, mdegor, solver.result().map_err(egmdo_err)?)
            }
            Solver::Egmda(solver) => {
                MdegorResult::from_egmda(py, mdegor, solver.result().map_err(egmdo_err)?)
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl MdegorResult {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "MdegorResult",
            &[
                ("method", self.method.clone().into_pyobject(py)?.into_any()),
                (
                    "z_star_mean",
                    self.z_star_mean
                        .as_ref()
                        .map(|z| z.bind(py).clone())
                        .into_pyobject(py)?
                        .into_any(),
                ),
                ("y_min_mean", self.y_min_mean.into_pyobject(py)?.into_any()),
                ("z_best", self.z_best.bind(py).clone().into_any()),
                ("f_best", self.f_best.into_pyobject(py)?.into_any()),
                (
                    "n_evaluations",
                    self.n_evaluations.clone().into_pyobject(py)?.into_any(),
                ),
            ],
        )
    }
}

/// Mdegor, efficient global multidisciplinary design optimizer (EGMDO and C-EGMDO, or EGMDA)
///
/// Minimizes an objective subject to constraints, both cheap functions of the design variables
/// and of coupling variables computed by expensive disciplines which depend on each other.
/// Each disciplinary solver is replaced by Gaussian process surrogates (one per coupling
/// variable), enriched adaptively, with as few disciplinary evaluations as possible.
///
/// Two methods are available, selected by the type of `method`:
///
/// * EGMDO (`EgmdoConfig`, default): the surrogates are enriched where the optimum is likely
///   to be, from the uncertainty of the objective and constraints propagated through random
///   multidisciplinary analyses (polynomial chaos expansions and a Karhunen-Loève random field).
/// * EGMDA (`EgmdaConfig`): `Egor` minimizes the objective, and the multidisciplinary analysis
///   at each of its points is solved with the surrogate means, the surrogates being enriched
///   until the linearized coupling variables are certain enough.
///
/// The defaults follow the C-EGMDO paper (Cardoso et al. 2024): WB2 acquisition, enrichment of
/// the discipline with the largest Sobol index, with Matérn 5/2 GPs. The settings of the EGMDO
/// paper (Dubreuil et al. 2020) are `enrichment=EnrichmentStrategy.ALL`,
/// `gp_kernel=CorrelationSpec.SQUARED_EXPONENTIAL` and
/// `method=EgmdoConfig(acquisition=InfillStrategy.EI, p_threshold=None, n_mc=100)`.
///
/// Parameters
/// ----------
/// design : dict of str to [lower, upper]
///     Names and bounds of the continuous design variables, in the column order of the design
///     points (`z`).
/// couplings : dict of str to [lower, upper]
///     Names and initial ranges of the scalar coupling variables, in the column order of the
///     coupling points (`y`). The ranges are only used to sample the initial disciplinary
///     designs of experiments. Each coupling variable is computed by exactly one discipline.
/// disciplines : list of Discipline
///     The disciplines, wired to the variables by their names.
/// qoi : callable
///     The cheap quantities of interest, evaluated on a batch of points:
///     `qoi(z: array[n, len(design)], y: array[n, len(couplings)]) -> array[n, 1 + n_ineq + n_eq]`
///     returning the objective, then the inequality constraints `g <= 0`, then the equality
///     constraints `h = 0`. It is called from several threads.
/// n_ineq : int
///     Number of inequality constraints returned by `qoi`.
/// n_eq : int
///     Number of equality constraints returned by `qoi`.
/// deterministic_constraints : list of int, optional
///     Indices (in 0..n_ineq + n_eq) of the constraints which depend on the design variables
///     only: they are evaluated directly instead of being modelled.
/// cstr_tol : list of float, optional
///     Tolerance of each constraint, default 1e-4.
/// max_iters : int, optional
///     Number of iterations: points added to the uncertainty quantification DoE with EGMDO
///     (default 10), iterations of `Egor` with EGMDA (default 20).
/// enrichment : EnrichmentStrategy
///     Disciplines evaluated at an enrichment point, EnrichmentStrategy.SOBOL (default) or
///     EnrichmentStrategy.ALL.
/// gp_kernel : CorrelationSpec
///     Kernel of the GPs, CorrelationSpec.MATERN52 (default) or
///     CorrelationSpec.SQUARED_EXPONENTIAL.
/// n_doe : list of int, optional
///     Size of the initial design of experiments of each discipline (default 2 x its number of
///     inputs).
/// doe : list of array[n_i, len(inputs_i)], optional
///     Initial designs of experiments of the disciplines (inputs to evaluate, columns in the
///     order of `Discipline.inputs`), overrides `n_doe`.
/// n_doe_uq : int, optional
///     Size of the initial design of experiments of the uncertainty quantification (default
///     20); with EGMDA, of the initial design of experiments of `Egor` (default
///     max(len(design) + 1, 5)).
/// doe_uq : array[n, len(design)], optional
///     Initial design of experiments of the uncertainty quantification, or of `Egor` with
///     EGMDA, overrides `n_doe_uq`.
/// max_evaluations : int, optional
///     Budget of true disciplinary evaluations, all disciplines included.
/// mda_relaxation : float in ]0, 1]
///     Relaxation of the Jacobi iterations of the multidisciplinary analyses.
/// method : EgmdoConfig or EgmdaConfig, optional
///     The method and its specific settings: EGMDO with `EgmdoConfig` (default
///     `EgmdoConfig()`), or EGMDA with `EgmdaConfig`.
/// seed : int >= 0, optional
///     Random generator seed, for reproducible runs.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct Mdegor {
    problem: egmdo::MdoProblem,
    method: MethodSettings,
    config: MethodConfig,
    funs: Vec<Option<Py<PyAny>>>,
    design_names: Vec<String>,
    coupling_names: Vec<String>,
    /// Per discipline: `perms[i][j]` is the core input column of the user input column `j`
    perms: Vec<Vec<usize>>,
    callback_error: Arc<CallbackError>,
    solver: Option<Solver>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Mdegor {
    #[new]
    #[pyo3(signature = (
        design,
        couplings,
        disciplines,
        qoi,
        n_ineq = 0,
        n_eq = 0,
        deterministic_constraints = None,
        cstr_tol = None,
        max_iters = None,
        enrichment = EnrichmentStrategy::Sobol,
        gp_kernel = CorrelationSpec::MATERN52,
        n_doe = None,
        doe = None,
        n_doe_uq = None,
        doe_uq = None,
        max_evaluations = None,
        mda_relaxation = 1.,
        method = None,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        #[gen_stub(override_type(type_repr = "typing.Mapping[builtins.str, typing.Sequence[builtins.float]]", imports = ("typing", "builtins")))]
        design: Bound<'_, PyDict>,
        #[gen_stub(override_type(type_repr = "typing.Mapping[builtins.str, typing.Sequence[builtins.float]]", imports = ("typing", "builtins")))]
        couplings: Bound<'_, PyDict>,
        #[gen_stub(override_type(type_repr = "typing.Sequence[Discipline]", imports = ("typing")))]
        disciplines: Vec<Py<Discipline>>,
        #[gen_stub(override_type(type_repr = "typing.Callable[[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]]", imports = ("typing", "numpy", "numpy.typing")))]
        qoi: Py<PyAny>,
        n_ineq: usize,
        n_eq: usize,
        deterministic_constraints: Option<Vec<usize>>,
        cstr_tol: Option<Vec<f64>>,
        max_iters: Option<usize>,
        enrichment: EnrichmentStrategy,
        gp_kernel: u8,
        n_doe: Option<Vec<usize>>,
        doe: Option<Vec<PyReadonlyArray2<f64>>>,
        n_doe_uq: Option<usize>,
        doe_uq: Option<PyReadonlyArray2<f64>>,
        max_evaluations: Option<usize>,
        mda_relaxation: f64,
        #[gen_stub(override_type(type_repr = "EgmdoConfig | EgmdaConfig | None", imports = ()))]
        method: Option<Py<PyAny>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let method = match method {
            None => MethodSettings::Egmdo(EgmdoConfig::default()),
            Some(method) => {
                let method = method.bind(py);
                if let Ok(config) = method.cast::<EgmdoConfig>() {
                    MethodSettings::Egmdo(config.borrow().clone())
                } else if let Ok(config) = method.cast::<EgmdaConfig>() {
                    MethodSettings::Egmda(config.borrow().clone())
                } else {
                    return Err(PyTypeError::new_err(
                        "method should be an EgmdoConfig, an EgmdaConfig or None",
                    ));
                }
            }
        };
        let (design_names, xlimits) = parse_variables(&design, "design variable")?;
        let (coupling_names, coupling_limits) = parse_variables(&couplings, "coupling variable")?;
        if design_names.is_empty() {
            return Err(PyValueError::new_err(
                "design should declare at least one variable",
            ));
        }
        let wiring = Wiring::new(&design_names, &coupling_names)?;
        let disciplines: Vec<PyRef<Discipline>> =
            disciplines.iter().map(|d| d.borrow(py)).collect();
        let resolved = wiring.resolve(&disciplines, &coupling_limits)?;
        let perms: Vec<Vec<usize>> = resolved.iter().map(|(_, perm)| perm.clone()).collect();
        let funs = disciplines
            .iter()
            .map(|d| d.fun.as_ref().map(|f| f.clone_ref(py)))
            .collect();

        let callback_error = Arc::new(CallbackError::default());
        let n_qoi = 1 + n_ineq + n_eq;
        let qoi = {
            let errors = callback_error.clone();
            move |z: &ArrayView2<f64>, y: &ArrayView2<f64>| -> Array2<f64> {
                Python::attach(|py| {
                    let args = (z.to_owned().into_pyarray(py), y.to_owned().into_pyarray(py));
                    match qoi
                        .bind(py)
                        .call1(args)
                        .and_then(|res| extract_2d(&res, z.nrows(), n_qoi, "qoi"))
                    {
                        Ok(values) => values,
                        Err(e) => errors.abort(e),
                    }
                })
            }
        };
        let problem = egmdo::MdoProblem::new(
            &xlimits,
            resolved.into_iter().map(|(d, _)| d).collect(),
            qoi,
        )
        .map_err(egmdo_err)?
        .constraints(n_ineq, n_eq)
        .deterministic_constraints(&deterministic_constraints.unwrap_or_default());

        // Initial disciplinary DoEs given in the users' input order
        let doe = match doe {
            Some(doe) => {
                if doe.len() != perms.len() {
                    return Err(PyValueError::new_err(format!(
                        "doe should give one array per discipline ({}), got {}",
                        perms.len(),
                        doe.len()
                    )));
                }
                let mut core = vec![];
                for ((x, perm), d) in doe.iter().zip(perms.iter()).zip(disciplines.iter()) {
                    let x = x.as_array();
                    if x.ncols() != perm.len() {
                        return Err(PyValueError::new_err(format!(
                            "doe of discipline '{}' should have {} columns (its inputs), got {}",
                            d.name,
                            perm.len(),
                            x.ncols()
                        )));
                    }
                    core.push(to_core(&x, perm));
                }
                Some(core)
            }
            None => None,
        };

        let kernel = match gp_kernel {
            CorrelationSpec::MATERN52 => egmdo::Kernel::Matern52,
            CorrelationSpec::SQUARED_EXPONENTIAL => egmdo::Kernel::SquaredExponential,
            _ => {
                return Err(PyValueError::new_err(
                    "gp_kernel should be CorrelationSpec.MATERN52 or \
                     CorrelationSpec.SQUARED_EXPONENTIAL",
                ));
            }
        };
        let enrichment = match enrichment {
            EnrichmentStrategy::All => egmdo::Enrichment::All,
            EnrichmentStrategy::Sobol => egmdo::Enrichment::Sobol,
        };
        let shared = SharedSettings {
            enrichment,
            constraint_tolerances: cstr_tol,
            n_doe_disciplines: n_doe,
            doe_disciplines: doe,
            n_doe_uq,
            doe_uq: doe_uq.map(|d| d.as_array().to_owned()),
            max_iters,
            max_evaluations,
            gp: egmdo::GpSettings {
                kernel,
                ..egmdo::GpSettings::default()
            },
            mda: egmdo::mda::MdaSettings {
                relaxation: mda_relaxation,
                ..egmdo::mda::MdaSettings::default()
            },
            seed,
        };
        let config = match &method {
            MethodSettings::Egmdo(method) => MethodConfig::Egmdo(method.to_core(shared)),
            MethodSettings::Egmda(method) => MethodConfig::Egmda(method.to_core(shared)),
        };
        // Validate the configuration against the problem
        Solver::new(&problem, &config, None).map_err(egmdo_err)?;

        Ok(Mdegor {
            problem,
            method,
            config,
            funs,
            design_names,
            coupling_names,
            perms,
            callback_error,
            solver: None,
        })
    }

    /// Names of the design variables, in the column order of the design points
    #[getter]
    fn design_names(&self) -> Vec<String> {
        self.design_names.clone()
    }

    /// The method and its settings: a copy of the EgmdoConfig or EgmdaConfig given
    #[getter]
    #[gen_stub(override_return_type(type_repr = "EgmdoConfig | EgmdaConfig", imports = ()))]
    fn method(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(match &self.method {
            MethodSettings::Egmdo(config) => Py::new(py, config.clone())?.into_any(),
            MethodSettings::Egmda(config) => Py::new(py, config.clone())?.into_any(),
        })
    }

    /// Names of the coupling variables, in the column order of the coupling points
    #[getter]
    fn coupling_names(&self) -> Vec<String> {
        self.coupling_names.clone()
    }

    /// Minimize the objective with the disciplinary solvers given as `Discipline.fun`.
    ///
    /// Parameters
    /// ----------
    /// max_iters : int, optional
    ///     Number of iterations, overriding the one given to the constructor.
    ///
    /// Returns
    /// -------
    /// MdegorResult
    ///
    #[pyo3(signature = (max_iters = None))]
    fn minimize(&self, py: Python, max_iters: Option<usize>) -> PyResult<MdegorResult> {
        let funs = self
            .funs
            .iter()
            .zip(self.problem.disciplines())
            .map(|(f, d)| {
                f.as_ref().map(|f| f.clone_ref(py)).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "discipline '{}' has no function: give `fun` or use ask()/tell()",
                        d.name()
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut solver = Solver::new(&self.problem, &self.config, max_iters).map_err(egmdo_err)?;
        let errors = &self.callback_error;
        while let Some(requests) = run(py, errors, || solver.ask())? {
            let outputs = requests
                .iter()
                .map(|r| {
                    let d = &self.problem.disciplines()[r.discipline];
                    let x = to_user(&r.x.view(), &self.perms[r.discipline]);
                    let res = funs[r.discipline].bind(py).call1((x.into_pyarray(py),))?;
                    extract_2d(&res, r.x.nrows(), d.output_dim(), d.name())
                })
                .collect::<PyResult<Vec<_>>>()?;
            run(py, errors, || solver.tell(outputs))?;
        }
        solver.result(py, self)
    }

    /// Ask for the next true disciplinary evaluations (ask/tell interface).
    ///
    /// Runs the algorithm until it needs true evaluations of the disciplines. The outputs of the
    /// requested evaluations have to be given back with `tell()`. The first call starts a new
    /// optimization of `max_iters` iterations.
    ///
    /// Returns
    /// -------
    /// list of MdegorRequest, or None when the optimization is finished (see `result()`)
    ///
    fn ask(&mut self, py: Python) -> PyResult<Option<Vec<MdegorRequest>>> {
        if self.solver.is_none() {
            self.solver = Some(Solver::new(&self.problem, &self.config, None).map_err(egmdo_err)?);
        }
        let Mdegor {
            solver,
            callback_error,
            perms,
            ..
        } = self;
        let solver = solver.as_mut().expect("solver");
        let requests = run(py, callback_error, || solver.ask())?;
        Ok(requests.map(|requests| {
            requests
                .into_iter()
                .map(|r| MdegorRequest {
                    discipline: r.discipline,
                    x: to_user(&r.x.view(), &perms[r.discipline])
                        .into_pyarray(py)
                        .unbind(),
                })
                .collect()
        }))
    }

    /// Give the outputs of the evaluations requested by the last `ask()`.
    ///
    /// Parameters
    /// ----------
    /// outputs : list of array[n, len(outputs)]
    ///     outputs[k] holds the outputs of request k, one row per input point, columns in the
    ///     order of `Discipline.outputs`.
    ///
    fn tell(&mut self, py: Python, outputs: Vec<PyReadonlyArray2<f64>>) -> PyResult<()> {
        let outputs: Vec<Array2<f64>> = outputs.iter().map(|y| y.as_array().to_owned()).collect();
        let Mdegor {
            solver,
            callback_error,
            ..
        } = self;
        let solver = solver.as_mut().ok_or_else(|| {
            PyValueError::new_err("no evaluation was requested: call ask() first")
        })?;
        run(py, callback_error, || solver.tell(outputs))
    }

    /// The result of the ask/tell optimization, once `ask()` has returned None.
    ///
    /// Returns
    /// -------
    /// MdegorResult
    ///
    fn result(&self, py: Python) -> PyResult<MdegorResult> {
        let solver = self
            .solver
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the optimization is not started: call ask()"))?;
        solver.result(py, self)
    }

    /// Discard the current ask/tell optimization: the next `ask()` starts a new one.
    fn reset(&mut self) {
        self.solver = None;
    }
}

/// Parses an ordered dict of `name: [lower, upper]` variables
fn parse_variables(vars: &Bound<'_, PyDict>, what: &str) -> PyResult<(Vec<String>, Array2<f64>)> {
    let mut names = Vec::with_capacity(vars.len());
    let mut limits = Array2::zeros((vars.len(), 2));
    for (k, (key, value)) in vars.iter().enumerate() {
        let name: String = key.extract().map_err(|_| {
            PyTypeError::new_err(format!("{what} names should be strings, got {key:?}"))
        })?;
        if name.is_empty() {
            return Err(PyValueError::new_err(format!(
                "{what} names should not be empty"
            )));
        }
        let bounds: Vec<f64> = value.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "{what} '{name}': bounds should be [lower, upper] floats, got {value:?}"
            ))
        })?;
        match bounds.as_slice() {
            [lower, upper] if lower.is_finite() && upper.is_finite() && lower < upper => {
                limits[[k, 0]] = *lower;
                limits[[k, 1]] = *upper;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "{what} '{name}': bounds should be [lower, upper] with finite lower < upper, \
                     got {bounds:?}"
                )));
            }
        }
        names.push(name);
    }
    Ok((names, limits))
}

/// A declared variable: index of a design or coupling variable
#[derive(Clone, Copy)]
enum Variable {
    Design(usize),
    Coupling(usize),
}

/// The variables declared by name
struct Wiring<'a> {
    names: HashMap<&'a str, Variable>,
    coupling_names: &'a [String],
}

impl<'a> Wiring<'a> {
    fn new(design_names: &'a [String], coupling_names: &'a [String]) -> PyResult<Self> {
        let mut names = HashMap::new();
        let declared = design_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n, Variable::Design(i)))
            .chain(
                coupling_names
                    .iter()
                    .enumerate()
                    .map(|(q, n)| (n, Variable::Coupling(q))),
            );
        for (name, var) in declared {
            if names.insert(name.as_str(), var).is_some() {
                return Err(PyValueError::new_err(format!(
                    "variable '{name}' is declared twice in design and couplings"
                )));
            }
        }
        Ok(Wiring {
            names,
            coupling_names,
        })
    }

    /// Resolves the disciplines' names into core disciplines, with the permutation from the
    /// users' input columns to the core input columns
    fn resolve(
        &self,
        disciplines: &[PyRef<Discipline>],
        coupling_limits: &Array2<f64>,
    ) -> PyResult<Vec<(egmdo::Discipline, Vec<usize>)>> {
        let mut producers: Vec<Option<&str>> = vec![None; self.coupling_names.len()];
        let mut resolved = Vec::with_capacity(disciplines.len());
        for d in disciplines.iter() {
            let err =
                |msg: String| PyValueError::new_err(format!("discipline '{}': {msg}", d.name));
            if d.inputs.is_empty() {
                return Err(err("no input".to_string()));
            }
            let mut seen = HashSet::new();
            let mut inputs = Vec::with_capacity(d.inputs.len());
            for name in d.inputs.iter() {
                if !seen.insert(name.as_str()) {
                    return Err(err(format!("input '{name}' is given twice")));
                }
                match self.names.get(name.as_str()) {
                    Some(var) => inputs.push(*var),
                    None => return Err(err(format!("unknown input '{name}'"))),
                }
            }
            // Canonical core layout, whatever the users' order: design inputs sorted by design
            // index, then coupling inputs sorted by coupling index
            let mut design_inputs: Vec<usize> = inputs
                .iter()
                .filter_map(|v| match v {
                    Variable::Design(i) => Some(*i),
                    Variable::Coupling(_) => None,
                })
                .collect();
            let mut coupling_inputs: Vec<usize> = inputs
                .iter()
                .filter_map(|v| match v {
                    Variable::Coupling(q) => Some(*q),
                    Variable::Design(_) => None,
                })
                .collect();
            design_inputs.sort_unstable();
            coupling_inputs.sort_unstable();
            let mut outputs = Vec::with_capacity(d.outputs.len());
            for name in d.outputs.iter() {
                if !seen.insert(name.as_str()) {
                    return Err(if d.inputs.contains(name) {
                        err(format!("'{name}' is both an input and an output"))
                    } else {
                        err(format!("output '{name}' is given twice"))
                    });
                }
                match self.names.get(name.as_str()) {
                    Some(Variable::Coupling(q)) => {
                        if let Some(other) = producers[*q].replace(&d.name) {
                            return Err(PyValueError::new_err(format!(
                                "coupling variable '{name}' is computed by both '{other}' and '{}'",
                                d.name
                            )));
                        }
                        outputs.push(*q);
                    }
                    Some(Variable::Design(_)) => {
                        return Err(err(format!(
                            "output '{name}' is a design variable, not a coupling variable"
                        )));
                    }
                    None => return Err(err(format!("unknown output '{name}'"))),
                }
            }
            if outputs.is_empty() {
                return Err(err("no output".to_string()));
            }
            let position = |list: &[usize], index: usize| {
                list.iter()
                    .position(|&k| k == index)
                    .expect("input in its list")
            };
            let perm = inputs
                .iter()
                .map(|v| match v {
                    Variable::Design(i) => position(&design_inputs, *i),
                    Variable::Coupling(q) => design_inputs.len() + position(&coupling_inputs, *q),
                })
                .collect();
            let discipline = egmdo::Discipline::new(&d.name)
                .design_inputs(&design_inputs)
                .coupling_inputs(
                    &coupling_inputs,
                    &coupling_limits.select(Axis(0), &coupling_inputs),
                )
                .coupling_outputs(&outputs);
            resolved.push((discipline, perm));
        }
        if let Some(q) = producers.iter().position(|p| p.is_none()) {
            return Err(PyValueError::new_err(format!(
                "coupling variable '{}' is computed by no discipline",
                self.coupling_names[q]
            )));
        }
        Ok(resolved)
    }
}

/// The users' input columns from the core input columns: `x_user[:, j] = x_core[:, perm[j]]`
fn to_user(x: &ArrayView2<f64>, perm: &[usize]) -> Array2<f64> {
    x.select(Axis(1), perm)
}

/// The core input columns from the users' input columns
fn to_core(x: &ArrayView2<f64>, perm: &[usize]) -> Array2<f64> {
    let mut core = Array2::zeros(x.raw_dim());
    for (j, &c) in perm.iter().enumerate() {
        core.column_mut(c).assign(&x.column(j));
    }
    core
}

/// Runs a step of the algorithm without the GIL, so that the `qoi` callback can be called from
/// several threads, and raises the error of an aborting callback if any.
fn run<T: Send>(
    py: Python,
    callback_error: &CallbackError,
    step: impl FnOnce() -> egmdo::Result<T> + Send,
) -> PyResult<T> {
    install_panic_hook();
    match py.detach(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(step))) {
        Ok(res) => res.map_err(egmdo_err),
        // The algorithm was aborted: raise the recorded callback error if any
        // (the panic payload may have been rewrapped when crossing threads)
        Err(payload) => match callback_error.take() {
            Some(err) => Err(err),
            None => std::panic::resume_unwind(payload),
        },
    }
}

/// Extract the value returned by a callback as an (n, m) float array
fn extract_2d(res: &Bound<'_, PyAny>, n: usize, m: usize, what: &str) -> PyResult<Array2<f64>> {
    let arr = res.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} should return a 2D float64 numpy array of shape ({n}, {m}), got {}",
            res.get_type()
        ))
    })?;
    let arr = arr.as_array();
    if arr.dim() != (n, m) {
        return Err(PyValueError::new_err(format!(
            "{what} should return an array of shape ({n}, {m}), got {:?}",
            arr.shape()
        )));
    }
    Ok(arr.to_owned())
}

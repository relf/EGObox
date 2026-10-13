//! `Emdegor`: efficient global multidisciplinary design optimization (EGMDO and its constrained
//! extension C-EGMDO), where the disciplinary solvers are replaced by GP surrogates enriched
//! adaptively. See the `egobox-egmdo` crate.

use crate::domain;
use crate::errors::{CallbackError, egmdo_err, install_panic_hook};
use crate::types::{CorrelationSpec, InfillStrategy, repr_kwargs};

use egobox_egmdo as egmdo;
use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
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
/// coupling variables computed by the other disciplines. Coupling variables are scalars
/// identified by their index, from 0 to n_couplings - 1; each one is computed by exactly one
/// discipline.
///
/// Parameters
/// ----------
/// name : str
///     Unique name of the discipline.
/// fun : callable, optional
///     The disciplinary solver, evaluated on a batch of points:
///     `fun(x: array[n, d_in]) -> array[n, d_out]`, where each row of `x` holds the design
///     variables used by the discipline (in `design_inputs` order) followed by the coupling
///     variables it receives (in `coupling_inputs` order), and each output row holds the
///     coupling variables it computes (in `coupling_outputs` order).
///     Required by `Emdegor.minimize()`, not by the ask/tell interface.
/// coupling_inputs : list of int
///     Indices of the coupling variables received by the discipline.
/// coupling_bounds : list of [lower, upper] or array[len(coupling_inputs), 2]
///     Initial guess of the range of the received coupling variables, used to sample the
///     initial disciplinary design of experiments. Required when `coupling_inputs` is not empty.
/// coupling_outputs : list of int
///     Indices of the coupling variables computed by the discipline.
/// design_inputs : list of int, optional
///     Indices of the design variables used by the discipline (all of them by default).
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct Discipline {
    inner: egmdo::Discipline,
    fun: Option<Py<PyAny>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Discipline {
    #[new]
    #[pyo3(signature = (
        name,
        fun = None,
        coupling_inputs = vec![],
        coupling_bounds = None,
        coupling_outputs = vec![],
        design_inputs = None,
    ))]
    fn new(
        name: &str,
        #[gen_stub(override_type(type_repr = "typing.Callable[[numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]] | None", imports = ("typing", "numpy", "numpy.typing")))]
        fun: Option<Py<PyAny>>,
        coupling_inputs: Vec<usize>,
        #[gen_stub(override_type(type_repr = "typing.Sequence[typing.Sequence[builtins.float]] | numpy.typing.NDArray[numpy.float64] | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        coupling_bounds: Option<Vec<Vec<f64>>>,
        coupling_outputs: Vec<usize>,
        design_inputs: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let bounds = coupling_bounds.unwrap_or_default();
        if bounds.len() != coupling_inputs.len() {
            return Err(PyValueError::new_err(format!(
                "discipline '{name}': coupling_bounds should give [lower, upper] bounds for each \
                 of the {} coupling inputs, got {} bounds",
                coupling_inputs.len(),
                bounds.len()
            )));
        }
        let mut limits = Array2::zeros((bounds.len(), 2));
        for (k, b) in bounds.iter().enumerate() {
            match b.as_slice() {
                [lower, upper] => {
                    limits[[k, 0]] = *lower;
                    limits[[k, 1]] = *upper;
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "discipline '{name}': coupling_bounds[{k}] should be [lower, upper], \
                         got {b:?}"
                    )));
                }
            }
        }
        let mut inner = egmdo::Discipline::new(name)
            .coupling_inputs(&coupling_inputs, &limits)
            .coupling_outputs(&coupling_outputs);
        if let Some(indices) = design_inputs {
            inner = inner.design_inputs(&indices);
        }
        Ok(Discipline { inner, fun })
    }

    /// Name of the discipline
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Indices of the coupling variables received by the discipline
    #[getter]
    fn coupling_inputs(&self) -> Vec<usize> {
        self.inner.coupling_input_indices().to_vec()
    }

    /// Indices of the coupling variables computed by the discipline
    #[getter]
    fn coupling_outputs(&self) -> Vec<usize> {
        self.inner.coupling_output_indices().to_vec()
    }

    /// Initial range of the coupling variables received by the discipline
    #[getter]
    fn coupling_bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coupling_bounds().to_pyarray(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "Discipline",
            &[
                ("name", self.inner.name().into_pyobject(py)?.into_any()),
                (
                    "coupling_inputs",
                    self.coupling_inputs().into_pyobject(py)?.into_any(),
                ),
                (
                    "coupling_outputs",
                    self.coupling_outputs().into_pyobject(py)?.into_any(),
                ),
            ],
        )
    }
}

/// True evaluations of a discipline requested by `Emdegor.ask()`
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct EmdegorRequest {
    /// Index of the discipline in the problem
    #[pyo3(get)]
    discipline: usize,
    /// Inputs of the discipline, one point per row (see `Discipline`)
    #[pyo3(get)]
    x: Py<PyArray2<f64>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl EmdegorRequest {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "EmdegorRequest",
            &[
                ("discipline", self.discipline.into_pyobject(py)?.into_any()),
                ("x", self.x.bind(py).clone().into_any()),
            ],
        )
    }
}

/// Training inputs and outputs of a discipline
type TrainingData = (Py<PyArray2<f64>>, Py<PyArray2<f64>>);

/// Result of an EGMDO optimization
///
/// The minimum is estimated by Monte Carlo on the final model of the objective, which accounts
/// for the remaining uncertainty of the disciplinary surrogates: `z_star` and `y_min` hold samples
/// of the minimum position and value, summarized by their means and standard deviations.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct EmdegorResult {
    /// Best point of the uncertainty quantification DoE: the lowest objective mean among the
    /// points whose constraint means are feasible
    #[pyo3(get)]
    z_best: Py<PyArray1<f64>>,
    /// Objective mean at `z_best`
    #[pyo3(get)]
    f_best: f64,
    /// Samples of the minimum position, one per row
    #[pyo3(get)]
    z_star: Py<PyArray2<f64>>,
    /// Samples of the minimum value
    #[pyo3(get)]
    y_min: Py<PyArray1<f64>>,
    /// Mean of the minimum position
    #[pyo3(get)]
    z_star_mean: Py<PyArray1<f64>>,
    /// Standard deviation of the minimum position
    #[pyo3(get)]
    z_star_std: Py<PyArray1<f64>>,
    /// Mean of the minimum value
    #[pyo3(get)]
    y_min_mean: f64,
    /// Standard deviation of the minimum value
    #[pyo3(get)]
    y_min_std: f64,
    /// Number of true evaluations of each discipline
    #[pyo3(get)]
    n_evaluations: Vec<usize>,
    /// Final design of experiments of the uncertainty quantification
    #[pyo3(get)]
    doe_uq: Py<PyArray2<f64>>,
    /// Training data (x, y) of each discipline, i.e. all its true evaluations
    #[pyo3(get)]
    disciplinary_doe: Vec<TrainingData>,
    /// Per-iteration records: dicts with keys `z_new` (point added to the uncertainty
    /// quantification DoE, or None), `max_criterion`, `wb2_scale`, `n_modes`, `n_enrichments`,
    /// `n_failed_mdas` and `n_dropped_points`
    #[pyo3(get)]
    history: Vec<Py<PyDict>>,
}

impl EmdegorResult {
    fn new(py: Python, n_disciplines: usize, result: egmdo::EgmdoResult) -> PyResult<Self> {
        let history = result
            .history
            .iter()
            .map(|h| {
                let d = PyDict::new(py);
                d.set_item("z_new", h.z_new.as_ref().map(|z| z.to_pyarray(py)))?;
                d.set_item("max_criterion", h.max_criterion)?;
                d.set_item("wb2_scale", h.wb2_scale)?;
                d.set_item("n_modes", h.n_modes)?;
                d.set_item("n_enrichments", h.n_enrichments)?;
                d.set_item("n_failed_mdas", h.n_failed_mdas)?;
                d.set_item("n_dropped_points", h.n_dropped_points)?;
                Ok(d.unbind())
            })
            .collect::<PyResult<_>>()?;
        Ok(EmdegorResult {
            z_best: result.z_best.to_pyarray(py).unbind(),
            f_best: result.f_best,
            z_star_mean: result.z_star_mean().into_pyarray(py).unbind(),
            z_star_std: result.z_star_std().into_pyarray(py).unbind(),
            y_min_mean: result.y_min_mean(),
            y_min_std: result.y_min_std(),
            z_star: result.z_star.to_pyarray(py).unbind(),
            y_min: result.y_min.to_pyarray(py).unbind(),
            n_evaluations: result.n_evaluations.clone(),
            doe_uq: result.doe_uq.to_pyarray(py).unbind(),
            disciplinary_doe: (0..n_disciplines)
                .map(|i| {
                    let (x, y) = result.models.training_data(i);
                    (x.to_pyarray(py).unbind(), y.to_pyarray(py).unbind())
                })
                .collect(),
            history,
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl EmdegorResult {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "EmdegorResult",
            &[
                ("z_star_mean", self.z_star_mean.bind(py).clone().into_any()),
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

/// Emdegor, efficient global multidisciplinary design optimizer (EGMDO and C-EGMDO)
///
/// Minimizes an objective subject to constraints, both cheap functions of the design variables
/// and of coupling variables computed by expensive disciplines which depend on each other.
/// Each disciplinary solver is replaced by Gaussian process surrogates (one per coupling
/// variable), enriched adaptively where the optimum is likely to be, with as few disciplinary
/// evaluations as possible.
///
/// The defaults follow the C-EGMDO paper (Cardoso et al. 2024): WB2 acquisition, enrichment of
/// the discipline with the largest Sobol index, with Matérn 5/2 GPs. The settings of the EGMDO
/// paper (Dubreuil et al. 2020) are `acquisition=InfillStrategy.EI`,
/// `enrichment=EnrichmentStrategy.ALL`, `p_threshold=None`, `n_mc=100` and
/// `gp_kernel=CorrelationSpec.SQUARED_EXPONENTIAL`.
///
/// Parameters
/// ----------
/// xspecs : list of [lower, upper] or array[nx, 2]
///     Bounds of the nx continuous design variables.
/// disciplines : list of Discipline
///     The disciplines; their coupling outputs must cover the coupling variables
///     0..n_couplings - 1 exactly once.
/// qoi : callable
///     The cheap quantities of interest, evaluated on a batch of points:
///     `qoi(z: array[n, nx], y: array[n, n_couplings]) -> array[n, 1 + n_ineq + n_eq]`
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
/// max_iters : int
///     Number of iterations (points added to the uncertainty quantification DoE).
/// acquisition : InfillStrategy
///     Acquisition criterion: InfillStrategy.WB2 (default), InfillStrategy.WB2S,
///     InfillStrategy.EI or InfillStrategy.LOG_EI (Monte Carlo estimates, constrained when the
///     problem has constraints).
/// wb2_scale : float
///     Scale of the expected improvement in the WB2 criterion (InfillStrategy.WB2 only).
/// enrichment : EnrichmentStrategy
///     Disciplines evaluated at an enrichment point, EnrichmentStrategy.SOBOL (default) or
///     EnrichmentStrategy.ALL.
/// gp_kernel : CorrelationSpec
///     Kernel of the GPs, CorrelationSpec.MATERN52 (default) or
///     CorrelationSpec.SQUARED_EXPONENTIAL.
/// n_doe : list of int, optional
///     Size of the initial design of experiments of each discipline (default 2 x its number of
///     inputs).
/// doe : list of array[n_i, d_in_i], optional
///     Initial designs of experiments of the disciplines (inputs to evaluate), overrides `n_doe`.
/// n_doe_uq : int
///     Size of the initial design of experiments of the uncertainty quantification.
/// doe_uq : array[n, nx], optional
///     Initial design of experiments of the uncertainty quantification, overrides `n_doe_uq`.
/// cv_threshold : float
///     Enrichment threshold on the coefficients of variation of the quantities of interest.
/// p_threshold : float, optional
///     Enrichment threshold on the probability of being the solution; None gives 1 / n_doe_uq.
/// n_mc : int, optional
///     Number of random MDAs per point of the uncertainty quantification (default max(200, 2P),
///     P being the size of the polynomial chaos basis).
/// pce_degree : int
///     Total degree of the polynomial chaos expansions.
/// n_ei_samples : int
///     Number of Monte Carlo samples of the acquisition criterion.
/// n_pmin_samples : int
///     Number of Monte Carlo samples of the probability of being the solution.
/// n_starts : int
///     Number of starting points of the acquisition optimization.
/// n_sim : int
///     Number of samples of the final estimation of the minimum.
/// max_evaluations : int, optional
///     Budget of true disciplinary evaluations, all disciplines included.
/// mda_relaxation : float in ]0, 1]
///     Relaxation of the Jacobi iterations of the multidisciplinary analyses.
/// seed : int >= 0, optional
///     Random generator seed, for reproducible runs.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct Emdegor {
    problem: egmdo::MdoProblem,
    config: egmdo::EgmdoConfig,
    funs: Vec<Option<Py<PyAny>>>,
    callback_error: Arc<CallbackError>,
    solver: Option<egmdo::EgmdoSolver>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Emdegor {
    #[new]
    #[pyo3(signature = (
        xspecs,
        disciplines,
        qoi,
        n_ineq = 0,
        n_eq = 0,
        deterministic_constraints = None,
        cstr_tol = None,
        max_iters = 10,
        acquisition = InfillStrategy::Wb2,
        wb2_scale = 1.,
        enrichment = EnrichmentStrategy::Sobol,
        gp_kernel = CorrelationSpec::MATERN52,
        n_doe = None,
        doe = None,
        n_doe_uq = 20,
        doe_uq = None,
        cv_threshold = 0.01,
        p_threshold = Some(1e-3),
        n_mc = None,
        pce_degree = 3,
        n_ei_samples = 2000,
        n_pmin_samples = 10_000,
        n_starts = 20,
        n_sim = 100,
        max_evaluations = None,
        mda_relaxation = 1.,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        #[gen_stub(override_type(type_repr = "typing.Sequence[typing.Sequence[builtins.float]] | numpy.typing.NDArray[numpy.float64]", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        xspecs: Py<PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Sequence[Discipline]", imports = ("typing")))]
        disciplines: Vec<Py<Discipline>>,
        #[gen_stub(override_type(type_repr = "typing.Callable[[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]]", imports = ("typing", "numpy", "numpy.typing")))]
        qoi: Py<PyAny>,
        n_ineq: usize,
        n_eq: usize,
        deterministic_constraints: Option<Vec<usize>>,
        cstr_tol: Option<Vec<f64>>,
        max_iters: usize,
        acquisition: InfillStrategy,
        wb2_scale: f64,
        enrichment: EnrichmentStrategy,
        gp_kernel: u8,
        n_doe: Option<Vec<usize>>,
        doe: Option<Vec<PyReadonlyArray2<f64>>>,
        n_doe_uq: usize,
        doe_uq: Option<PyReadonlyArray2<f64>>,
        cv_threshold: f64,
        p_threshold: Option<f64>,
        n_mc: Option<usize>,
        pce_degree: usize,
        n_ei_samples: usize,
        n_pmin_samples: usize,
        n_starts: usize,
        n_sim: usize,
        max_evaluations: Option<usize>,
        mda_relaxation: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        // Design space: continuous bounds only
        let xtypes = domain::parse(py, xspecs)?;
        let mut xlimits = Array2::zeros((xtypes.len(), 2));
        for (i, xtype) in xtypes.iter().enumerate() {
            match xtype {
                egobox_moe::XType::Float(lower, upper) => {
                    xlimits[[i, 0]] = *lower;
                    xlimits[[i, 1]] = *upper;
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "xspecs[{i}]: Emdegor handles continuous design variables only"
                    )));
                }
            }
        }

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

        let funs = disciplines
            .iter()
            .map(|d| d.borrow(py).fun.as_ref().map(|f| f.clone_ref(py)))
            .collect();
        let disciplines = disciplines
            .iter()
            .map(|d| d.borrow(py).inner.clone())
            .collect();
        let problem = egmdo::MdoProblem::new(&xlimits, disciplines, qoi)
            .map_err(egmdo_err)?
            .constraints(n_ineq, n_eq)
            .deterministic_constraints(&deterministic_constraints.unwrap_or_default());

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
        let acquisition = match acquisition {
            InfillStrategy::Ei => egmdo::Acquisition::Ei,
            InfillStrategy::Wb2 => egmdo::Acquisition::Wb2 { scale: wb2_scale },
            InfillStrategy::Wb2s => egmdo::Acquisition::Wb2s,
            InfillStrategy::LogEi => egmdo::Acquisition::LogEi,
        };
        let enrichment = match enrichment {
            EnrichmentStrategy::All => egmdo::Enrichment::All,
            EnrichmentStrategy::Sobol => egmdo::Enrichment::Sobol,
        };
        let config = egmdo::EgmdoConfig {
            acquisition,
            enrichment,
            constraint_tolerances: cstr_tol,
            n_doe_disciplines: n_doe,
            doe_disciplines: doe.map(|d| d.iter().map(|x| x.as_array().to_owned()).collect()),
            n_doe_uq,
            doe_uq: doe_uq.map(|d| d.as_array().to_owned()),
            max_iters,
            max_evaluations,
            cv_threshold,
            p_threshold,
            n_ei_samples,
            n_pmin_samples,
            n_starts,
            n_sim,
            uq: egmdo::uq::UqSettings {
                degree: pce_degree,
                n_mc,
                ..egmdo::uq::UqSettings::default()
            },
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
        // Validate the configuration against the problem
        egmdo::EgmdoSolver::new(problem.clone(), config.clone()).map_err(egmdo_err)?;

        Ok(Emdegor {
            problem,
            config,
            funs,
            callback_error,
            solver: None,
        })
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
    /// EmdegorResult
    ///
    #[pyo3(signature = (max_iters = None))]
    fn minimize(&self, py: Python, max_iters: Option<usize>) -> PyResult<EmdegorResult> {
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
        let config = egmdo::EgmdoConfig {
            max_iters: max_iters.unwrap_or(self.config.max_iters),
            ..self.config.clone()
        };
        let mut solver =
            egmdo::EgmdoSolver::new(self.problem.clone(), config).map_err(egmdo_err)?;
        let errors = &self.callback_error;
        while let Some(requests) = run(py, errors, || solver.ask())? {
            let outputs = requests
                .iter()
                .map(|r| {
                    let d = &self.problem.disciplines()[r.discipline];
                    let res = funs[r.discipline].bind(py).call1((r.x.to_pyarray(py),))?;
                    extract_2d(&res, r.x.nrows(), d.output_dim(), d.name())
                })
                .collect::<PyResult<Vec<_>>>()?;
            run(py, errors, || solver.tell(outputs))?;
        }
        EmdegorResult::new(
            py,
            self.problem.disciplines().len(),
            solver.result().map_err(egmdo_err)?,
        )
    }

    /// Ask for the next true disciplinary evaluations (ask/tell interface).
    ///
    /// Runs the algorithm until it needs true evaluations of the disciplines. The outputs of the
    /// requested evaluations have to be given back with `tell()`. The first call starts a new
    /// optimization of `max_iters` iterations.
    ///
    /// Returns
    /// -------
    /// list of EmdegorRequest, or None when the optimization is finished (see `result()`)
    ///
    fn ask(&mut self, py: Python) -> PyResult<Option<Vec<EmdegorRequest>>> {
        if self.solver.is_none() {
            self.solver = Some(
                egmdo::EgmdoSolver::new(self.problem.clone(), self.config.clone())
                    .map_err(egmdo_err)?,
            );
        }
        let Emdegor {
            solver,
            callback_error,
            ..
        } = self;
        let solver = solver.as_mut().expect("solver");
        let requests = run(py, callback_error, || solver.ask())?;
        Ok(requests.map(|requests| {
            requests
                .into_iter()
                .map(|r| EmdegorRequest {
                    discipline: r.discipline,
                    x: r.x.into_pyarray(py).unbind(),
                })
                .collect()
        }))
    }

    /// Give the outputs of the evaluations requested by the last `ask()`.
    ///
    /// Parameters
    /// ----------
    /// outputs : list of array[n, d_out]
    ///     outputs[k] holds the outputs of request k, one row per input point.
    ///
    fn tell(&mut self, py: Python, outputs: Vec<PyReadonlyArray2<f64>>) -> PyResult<()> {
        let outputs: Vec<Array2<f64>> = outputs.iter().map(|y| y.as_array().to_owned()).collect();
        let Emdegor {
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
    /// EmdegorResult
    ///
    fn result(&self, py: Python) -> PyResult<EmdegorResult> {
        let solver = self
            .solver
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the optimization is not started: call ask()"))?;
        EmdegorResult::new(
            py,
            self.problem.disciplines().len(),
            solver.result().map_err(egmdo_err)?,
        )
    }

    /// Discard the current ask/tell optimization: the next `ask()` starts a new one.
    fn reset(&mut self) {
        self.solver = None;
    }
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

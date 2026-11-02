//! The method-specific settings of `Mdegor`: `EgmdoConfig` for EGMDO and C-EGMDO, `EgmdaConfig`
//! for EGMDA (`Egor` with the disciplinary surrogates as the solver of the multidisciplinary
//! analysis).

use crate::types::{InfillStrategy, repr_kwargs};

use egobox_egmdo as egmdo;
use egobox_egmdo::egmda;
use ndarray::Array2;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

/// The settings of `Mdegor` shared by both methods
pub(crate) struct SharedSettings {
    pub enrichment: egmdo::Enrichment,
    pub constraint_tolerances: Option<Vec<f64>>,
    pub n_doe_disciplines: Option<Vec<usize>>,
    pub doe_disciplines: Option<Vec<Array2<f64>>>,
    pub n_doe_uq: Option<usize>,
    pub doe_uq: Option<Array2<f64>>,
    pub max_iters: Option<usize>,
    pub max_evaluations: Option<usize>,
    pub gp: egmdo::GpSettings,
    pub mda: egmdo::mda::MdaSettings,
    pub seed: Option<u64>,
}

/// Settings of EGMDO and C-EGMDO, given as `Mdegor(..., method=EgmdoConfig(...))`
///
/// The surrogates are enriched where the optimum is likely to be, from the uncertainty of the
/// objective and constraints propagated through random multidisciplinary analyses (polynomial
/// chaos expansions and a Karhunen-Loève random field).
///
/// The defaults follow the C-EGMDO paper (Cardoso et al. 2024). The settings of the EGMDO paper
/// (Dubreuil et al. 2020) are `acquisition=InfillStrategy.EI`, `p_threshold=None` and
/// `n_mc=100`, with `Mdegor(enrichment=EnrichmentStrategy.ALL,
/// gp_kernel=CorrelationSpec.SQUARED_EXPONENTIAL)`.
///
/// Parameters
/// ----------
/// acquisition : InfillStrategy
///     Acquisition criterion: InfillStrategy.WB2 (default), InfillStrategy.WB2S,
///     InfillStrategy.EI or InfillStrategy.LOG_EI (Monte Carlo estimates, constrained when the
///     problem has constraints).
/// wb2_scale : float
///     Scale of the expected improvement in the WB2 criterion (InfillStrategy.WB2 only).
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
/// linearized_mda : bool
///     Linearize the random multidisciplinary analysis at the solution of the mean one,
///     instead of solving it for each of the n_mc samples (much faster, accurate when the
///     disciplinary GP uncertainty is small). Points where the mean analysis does not converge
///     fall back to the solved random analyses.
/// gp_correlation : bool
///     Correlate the disciplinary GP uncertainty across the design points of the uncertainty
///     quantification as the GP posteriors, instead of perfectly (one random sample shared by
///     every point, as in the EGMDO papers).
/// feasibility_first : bool
///     While no point of the uncertainty quantification DoE is feasible by its constraint
///     means, maximize the probability of feasibility of the constraint models instead of the
///     acquisition criterion.
/// mean_mda_constraints : bool
///     In the acquisition, evaluate the constraints by the multidisciplinary analysis solved on
///     the means of the disciplinary GPs at each candidate point, instead of the GPs of the
///     constraint means over the uncertainty quantification DoE.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Clone, Debug)]
pub(crate) struct EgmdoConfig {
    /// Acquisition criterion
    #[pyo3(get, set)]
    pub acquisition: InfillStrategy,
    /// Scale of the expected improvement in the WB2 criterion
    #[pyo3(get, set)]
    pub wb2_scale: f64,
    /// Enrichment threshold on the coefficients of variation
    #[pyo3(get, set)]
    pub cv_threshold: f64,
    /// Enrichment threshold on the probability of being the solution
    #[pyo3(get, set)]
    pub p_threshold: Option<f64>,
    /// Number of random MDAs per point of the uncertainty quantification
    #[pyo3(get, set)]
    pub n_mc: Option<usize>,
    /// Total degree of the polynomial chaos expansions
    #[pyo3(get, set)]
    pub pce_degree: usize,
    /// Number of Monte Carlo samples of the acquisition criterion
    #[pyo3(get, set)]
    pub n_ei_samples: usize,
    /// Number of Monte Carlo samples of the probability of being the solution
    #[pyo3(get, set)]
    pub n_pmin_samples: usize,
    /// Number of starting points of the acquisition optimization
    #[pyo3(get, set)]
    pub n_starts: usize,
    /// Number of samples of the final estimation of the minimum
    #[pyo3(get, set)]
    pub n_sim: usize,
    /// Linearize the random multidisciplinary analysis
    #[pyo3(get, set)]
    pub linearized_mda: bool,
    /// Correlate the disciplinary GP uncertainty across design points as the GP posteriors
    #[pyo3(get, set)]
    pub gp_correlation: bool,
    /// Maximize the probability of feasibility while no point is feasible
    #[pyo3(get, set)]
    pub feasibility_first: bool,
    /// Constraints of the acquisition by the multidisciplinary analysis on the surrogate means
    #[pyo3(get, set)]
    pub mean_mda_constraints: bool,
}

impl Default for EgmdoConfig {
    fn default() -> Self {
        EgmdoConfig {
            acquisition: InfillStrategy::Wb2,
            wb2_scale: 1.,
            cv_threshold: 0.01,
            p_threshold: Some(1e-3),
            n_mc: None,
            pce_degree: 3,
            n_ei_samples: 2000,
            n_pmin_samples: 10_000,
            n_starts: 20,
            n_sim: 100,
            linearized_mda: false,
            gp_correlation: false,
            feasibility_first: false,
            mean_mda_constraints: false,
        }
    }
}

impl EgmdoConfig {
    /// The configuration of the Rust core
    pub(crate) fn to_core(&self, shared: SharedSettings) -> egmdo::EgmdoConfig {
        egmdo::EgmdoConfig {
            acquisition: match self.acquisition {
                InfillStrategy::Ei => egmdo::Acquisition::Ei,
                InfillStrategy::Wb2 => egmdo::Acquisition::Wb2 {
                    scale: self.wb2_scale,
                },
                InfillStrategy::Wb2s => egmdo::Acquisition::Wb2s,
                InfillStrategy::LogEi => egmdo::Acquisition::LogEi,
            },
            feasibility_first: self.feasibility_first,
            acquisition_constraints: if self.mean_mda_constraints {
                egmdo::AcquisitionConstraints::MeanMda
            } else {
                egmdo::AcquisitionConstraints::PceMeanGp
            },
            enrichment: shared.enrichment,
            constraint_tolerances: shared.constraint_tolerances,
            n_doe_disciplines: shared.n_doe_disciplines,
            doe_disciplines: shared.doe_disciplines,
            n_doe_uq: shared.n_doe_uq.unwrap_or(20),
            doe_uq: shared.doe_uq,
            max_iters: shared.max_iters.unwrap_or(10),
            max_evaluations: shared.max_evaluations,
            cv_threshold: self.cv_threshold,
            p_threshold: self.p_threshold,
            n_ei_samples: self.n_ei_samples,
            n_pmin_samples: self.n_pmin_samples,
            n_starts: self.n_starts,
            n_sim: self.n_sim,
            uq: egmdo::uq::UqSettings {
                degree: self.pce_degree,
                n_mc: self.n_mc,
                random_mda: if self.linearized_mda {
                    egmdo::uq::RandomMda::Linearized
                } else {
                    egmdo::uq::RandomMda::Solved
                },
                correlation: if self.gp_correlation {
                    egmdo::uq::Correlation::GpPosterior
                } else {
                    egmdo::uq::Correlation::Perfect
                },
                ..egmdo::uq::UqSettings::default()
            },
            gp: shared.gp,
            mda: shared.mda,
            seed: shared.seed,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl EgmdoConfig {
    #[new]
    #[pyo3(signature = (
        acquisition = EgmdoConfig::default().acquisition,
        wb2_scale = EgmdoConfig::default().wb2_scale,
        cv_threshold = EgmdoConfig::default().cv_threshold,
        p_threshold = EgmdoConfig::default().p_threshold,
        n_mc = EgmdoConfig::default().n_mc,
        pce_degree = EgmdoConfig::default().pce_degree,
        n_ei_samples = EgmdoConfig::default().n_ei_samples,
        n_pmin_samples = EgmdoConfig::default().n_pmin_samples,
        n_starts = EgmdoConfig::default().n_starts,
        n_sim = EgmdoConfig::default().n_sim,
        linearized_mda = EgmdoConfig::default().linearized_mda,
        gp_correlation = EgmdoConfig::default().gp_correlation,
        feasibility_first = EgmdoConfig::default().feasibility_first,
        mean_mda_constraints = EgmdoConfig::default().mean_mda_constraints,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        acquisition: InfillStrategy,
        wb2_scale: f64,
        cv_threshold: f64,
        p_threshold: Option<f64>,
        n_mc: Option<usize>,
        pce_degree: usize,
        n_ei_samples: usize,
        n_pmin_samples: usize,
        n_starts: usize,
        n_sim: usize,
        linearized_mda: bool,
        gp_correlation: bool,
        feasibility_first: bool,
        mean_mda_constraints: bool,
    ) -> Self {
        EgmdoConfig {
            acquisition,
            wb2_scale,
            cv_threshold,
            p_threshold,
            n_mc,
            pce_degree,
            n_ei_samples,
            n_pmin_samples,
            n_starts,
            n_sim,
            linearized_mda,
            gp_correlation,
            feasibility_first,
            mean_mda_constraints,
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "EgmdoConfig",
            &[
                (
                    "acquisition",
                    self.acquisition.into_pyobject(py)?.into_any(),
                ),
                ("wb2_scale", self.wb2_scale.into_pyobject(py)?.into_any()),
                (
                    "cv_threshold",
                    self.cv_threshold.into_pyobject(py)?.into_any(),
                ),
                (
                    "p_threshold",
                    self.p_threshold.into_pyobject(py)?.into_any(),
                ),
                ("n_mc", self.n_mc.into_pyobject(py)?.into_any()),
                ("pce_degree", self.pce_degree.into_pyobject(py)?.into_any()),
                (
                    "n_ei_samples",
                    self.n_ei_samples.into_pyobject(py)?.into_any(),
                ),
                (
                    "n_pmin_samples",
                    self.n_pmin_samples.into_pyobject(py)?.into_any(),
                ),
                ("n_starts", self.n_starts.into_pyobject(py)?.into_any()),
                ("n_sim", self.n_sim.into_pyobject(py)?.into_any()),
                (
                    "linearized_mda",
                    self.linearized_mda.into_pyobject(py)?.to_owned().into_any(),
                ),
                (
                    "gp_correlation",
                    self.gp_correlation.into_pyobject(py)?.to_owned().into_any(),
                ),
                (
                    "feasibility_first",
                    self.feasibility_first
                        .into_pyobject(py)?
                        .to_owned()
                        .into_any(),
                ),
                (
                    "mean_mda_constraints",
                    self.mean_mda_constraints
                        .into_pyobject(py)?
                        .to_owned()
                        .into_any(),
                ),
            ],
        )
    }
}

/// DispersionCriterion specifies how EGMDA measures the dispersion of the coupling variables
/// linearized around the analysis with the surrogate means, `N(y_0, Σ)`.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DispersionCriterion {
    /// `sqrt(Tr Σ) / ‖y_0‖`
    Relative = 1,
    /// `sqrt(Tr Σ)`
    Absolute = 2,
    /// The square root of the largest eigenvalue of `Σ`
    Spectral = 3,
}

impl<'a, 'py> FromPyObject<'a, 'py> for DispersionCriterion {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Relative),
            Ok(2) => Ok(Self::Absolute),
            Ok(3) => Ok(Self::Spectral),
            Ok(v) => Err(PyValueError::new_err(format!(
                "dispersion integer value must be in [1, 3], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "dispersion must be a DispersionCriterion enum or an integer in [1, 3]",
            )),
        }
    }
}

/// Settings of EGMDA, given as `Mdegor(..., method=EgmdaConfig(...))`
///
/// `Egor` minimizes the objective, and the multidisciplinary analysis at each of its points is
/// solved with the surrogate means, the surrogates being enriched until the coupling variables,
/// linearized as `N(y_0, Σ)`, are certain enough.
///
/// Parameters
/// ----------
/// dispersion : DispersionCriterion
///     The dispersion criterion of the linearized coupling variables:
///     DispersionCriterion.RELATIVE (default, sqrt(Tr Σ) / ‖y_0‖), DispersionCriterion.ABSOLUTE
///     (sqrt(Tr Σ)) or DispersionCriterion.SPECTRAL (square root of the largest eigenvalue of Σ).
/// tolerance : float > 0
///     The threshold of the dispersion criterion, in the units of the coupling variables (or
///     relative for DispersionCriterion.RELATIVE).
/// max_enrichments : int
///     The maximum number of enrichments at a design point.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Clone, Debug)]
pub(crate) struct EgmdaConfig {
    /// The dispersion criterion of the linearized coupling variables
    #[pyo3(get, set)]
    pub dispersion: DispersionCriterion,
    /// The threshold of the dispersion criterion
    #[pyo3(get, set)]
    pub tolerance: f64,
    /// The maximum number of enrichments at a design point
    #[pyo3(get, set)]
    pub max_enrichments: usize,
}

impl Default for EgmdaConfig {
    fn default() -> Self {
        EgmdaConfig {
            dispersion: DispersionCriterion::Relative,
            tolerance: egmda::Dispersion::default().threshold(),
            max_enrichments: egmda::EgmdaConfig::default().max_enrichments,
        }
    }
}

impl EgmdaConfig {
    /// The configuration of the Rust core
    pub(crate) fn to_core(&self, shared: SharedSettings) -> egmda::EgmdaConfig {
        egmda::EgmdaConfig {
            dispersion: match self.dispersion {
                DispersionCriterion::Relative => egmda::Dispersion::Relative(self.tolerance),
                DispersionCriterion::Absolute => egmda::Dispersion::Absolute(self.tolerance),
                DispersionCriterion::Spectral => egmda::Dispersion::Spectral(self.tolerance),
            },
            max_enrichments: self.max_enrichments,
            enrichment: shared.enrichment,
            constraint_tolerances: shared.constraint_tolerances,
            n_doe_disciplines: shared.n_doe_disciplines,
            doe_disciplines: shared.doe_disciplines,
            n_doe: shared.n_doe_uq,
            doe: shared.doe_uq,
            max_iters: shared
                .max_iters
                .unwrap_or(egmda::EgmdaConfig::default().max_iters),
            max_evaluations: shared.max_evaluations,
            gp: shared.gp,
            mda: shared.mda,
            seed: shared.seed,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl EgmdaConfig {
    #[new]
    #[pyo3(signature = (
        dispersion = EgmdaConfig::default().dispersion,
        tolerance = EgmdaConfig::default().tolerance,
        max_enrichments = EgmdaConfig::default().max_enrichments,
    ))]
    fn new(dispersion: DispersionCriterion, tolerance: f64, max_enrichments: usize) -> Self {
        EgmdaConfig {
            dispersion,
            tolerance,
            max_enrichments,
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "EgmdaConfig",
            &[
                ("dispersion", self.dispersion.into_pyobject(py)?.into_any()),
                ("tolerance", self.tolerance.into_pyobject(py)?.into_any()),
                (
                    "max_enrichments",
                    self.max_enrichments.into_pyobject(py)?.into_any(),
                ),
            ],
        )
    }
}

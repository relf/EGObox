use crate::deprecation::{resolve_renamed_key, warn_deprecated};
use egobox_ego::OBJECTIVE_FUNCTION_ERROR;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, PartialEq)]
pub enum Recombination {
    /// Prediction is taken from the expert with highest responsibility,
    /// resulting in a model with discontinuities
    Hard = 0,
    /// Prediction is a combination of experts predictions wrt their responsibilities,
    /// an optional heaviside factor might be used to control steepness of the change between
    /// experts regions.
    Smooth = 1,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Recombination {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(value.clone());
        }
        match obj.extract::<u8>() {
            Ok(0) => Ok(Self::Hard),
            Ok(1) => Ok(Self::Smooth),
            Ok(v) => Err(PyValueError::new_err(format!(
                "recombination integer value must be in [0, 1], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "recombination must be a Recombination enum or an integer in [0, 1]",
            )),
        }
    }
}

/// RegressionSpec is a bitfield that specifies which regression terms to include in the model.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Clone, Default, Debug)]
pub(crate) struct RegressionSpec(pub(crate) u8);

#[gen_stub_pymethods]
#[pymethods]
impl RegressionSpec {
    #[classattr]
    pub(crate) const ALL: u8 = egobox_moe::RegressionSpec::ALL.bits();
    #[classattr]
    pub(crate) const CONSTANT: u8 = egobox_moe::RegressionSpec::CONSTANT.bits();
    #[classattr]
    pub(crate) const LINEAR: u8 = egobox_moe::RegressionSpec::LINEAR.bits();
    #[classattr]
    pub(crate) const QUADRATIC: u8 = egobox_moe::RegressionSpec::QUADRATIC.bits();
}

/// CorrelationSpec is a bitfield that specifies which correlation terms to include in the model.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Clone, Default, Debug)]
pub(crate) struct CorrelationSpec(pub(crate) u8);

#[gen_stub_pymethods]
#[pymethods]
impl CorrelationSpec {
    #[classattr]
    pub(crate) const ALL: u8 = egobox_moe::CorrelationSpec::ALL.bits();
    #[classattr]
    pub(crate) const SQUARED_EXPONENTIAL: u8 =
        egobox_moe::CorrelationSpec::SQUAREDEXPONENTIAL.bits();
    #[classattr]
    pub(crate) const ABSOLUTE_EXPONENTIAL: u8 =
        egobox_moe::CorrelationSpec::ABSOLUTEEXPONENTIAL.bits();
    #[classattr]
    pub(crate) const MATERN32: u8 = egobox_moe::CorrelationSpec::MATERN32.bits();
    #[classattr]
    pub(crate) const MATERN52: u8 = egobox_moe::CorrelationSpec::MATERN52.bits();
}

/// InfillStrategy specifies the acquisition function to use for infill optimization.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InfillStrategy {
    /// Expected Improvement
    /// see Mockus et al. (1978) "The application of Bayesian methods for seeking the extremum"
    Ei = 1,
    /// Watson and Barnes 2nd criterion (WB2): EI shifted by the GP mean,
    /// easier to optimize than EI but may not explore as much as EI
    /// see Watson and Barnes (1995) "Infill sampling criteria to locate extremes"
    Wb2 = 2,
    /// Scaled version of WB2 (WB2S) to improve exploration
    Wb2s = 3,
    /// Logarithm of Expected Improvement (LogEI)
    /// see Ament et al. (2023) "Unexpected Improvements to Expected Improvement for Bayesian Optimization"
    LogEi = 4,
}

impl<'a, 'py> FromPyObject<'a, 'py> for InfillStrategy {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Ei),
            Ok(2) => Ok(Self::Wb2),
            Ok(3) => Ok(Self::Wb2s),
            Ok(4) => Ok(Self::LogEi),
            Ok(v) => Err(PyValueError::new_err(format!(
                "infill_strategy integer value must be in [1, 4], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "infill_strategy must be an InfillStrategy enum or an integer in [1, 4]",
            )),
        }
    }
}

/// ConstraintStrategy specifies the strategy to use for handling constraints in infill optimization.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ConstraintStrategy {
    /// Mean Constraint (MC): the mean of the GP is used to evaluate the constraint,
    /// which is equivalent to ignoring the uncertainty on the constraint
    Mc = 1,
    /// Upper Trust Bound (UTB): the upper trust bound of the GP is used to evaluate the constraint,
    /// which takes into account the uncertainty on the constraint
    Utb = 2,
}

#[gen_stub_pymethods]
#[pymethods]
impl ConstraintStrategy {
    /// Long name alias of MC
    #[classattr]
    #[allow(non_snake_case)]
    fn MEAN_CONSTRAINT() -> ConstraintStrategy {
        Self::Mc
    }
    /// Long name alias of UTB
    #[classattr]
    #[allow(non_snake_case)]
    fn UPPER_TRUST_BOUND() -> ConstraintStrategy {
        Self::Utb
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ConstraintStrategy {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Mc),
            Ok(2) => Ok(Self::Utb),
            Ok(v) => Err(PyValueError::new_err(format!(
                "cstr_strategy integer value must be in [1, 2], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "cstr_strategy must be a ConstraintStrategy enum or an integer in [1, 2]",
            )),
        }
    }
}

/// QEiStrategy specifies how the points of a qEI batch are selected: after each selected point,
/// the GP is updated with a virtual value given by the strategy, then the next point is selected.
/// qEI is the multi-point extension of EI, see Chevalier and Ginsbourger (2013)
/// "Fast Computation of the Multi-Points Expected Improvement with Applications in Batch Selection"
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum QEiStrategy {
    /// Kriging Believer (KB), the next point is added to the GP with its predicted mean value,
    /// which is equivalent to assuming that the prediction is perfect
    Kb = 1,
    /// Kriging Believer Lower Bound (KBLB), the next point is added to the GP with
    /// its predicted mean value minus a multiple of the predicted standard deviation,
    /// which is equivalent to assuming that the prediction is pessimistic
    Kblb = 2,
    /// Kriging Believer Upper Bound (KBUB), the next point is added to the GP with
    /// its predicted mean value plus a multiple of the predicted standard deviation,
    /// which is equivalent to assuming that the prediction is optimistic
    Kbub = 3,
    /// Constant Liar Minimum (CLMIN), the next point is added to the GP by using the current minimum
    /// value observed in the DOE, which is equivalent to assuming that
    /// the prediction is the current best value
    Clmin = 4,
}

#[gen_stub_pymethods]
#[pymethods]
impl QEiStrategy {
    /// Long name alias of KB
    #[classattr]
    #[allow(non_snake_case)]
    fn KRIGING_BELIEVER() -> QEiStrategy {
        Self::Kb
    }
    /// Long name alias of KBLB
    #[classattr]
    #[allow(non_snake_case)]
    fn KRIGING_BELIEVER_LOWER_BOUND() -> QEiStrategy {
        Self::Kblb
    }
    /// Long name alias of KBUB
    #[classattr]
    #[allow(non_snake_case)]
    fn KRIGING_BELIEVER_UPPER_BOUND() -> QEiStrategy {
        Self::Kbub
    }
    /// Long name alias of CLMIN
    #[classattr]
    #[allow(non_snake_case)]
    fn CONSTANT_LIAR_MINIMUM() -> QEiStrategy {
        Self::Clmin
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for QEiStrategy {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Kb),
            Ok(2) => Ok(Self::Kblb),
            Ok(3) => Ok(Self::Kbub),
            Ok(4) => Ok(Self::Clmin),
            Ok(v) => Err(PyValueError::new_err(format!(
                "qei strategy integer value must be in [1, 4], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "qei strategy must be a QEiStrategy enum or an integer in [1, 4]",
            )),
        }
    }
}

/// InfillOptimizer specifies the optimization algorithm to use for infill optimization.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InfillOptimizer {
    /// Gradient free optimization algorithm that uses a simplex of n+1 points for n-dimensional optimization
    Cobyla = 1,
    /// Gradient based optimization algorithm that uses a quasi-Newton method to optimize the acquisition function
    Slsqp = 2,
}

impl<'a, 'py> FromPyObject<'a, 'py> for InfillOptimizer {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Cobyla),
            Ok(2) => Ok(Self::Slsqp),
            Ok(v) => Err(PyValueError::new_err(format!(
                "infill_optimizer integer value must be in [1, 2], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "infill_optimizer must be an InfillOptimizer enum or an integer in [1, 2]",
            )),
        }
    }
}

/// FeasibleInfillStrategy activates the Expected Feasible Improvement (EFI) to handle hidden constraints,
/// i.e. points where the objective function fails (returns NaN or raises).
/// The infill criterion is weighted by the probability of viability given by a surrogate trained
/// on successful and failed points, see Tfaily et al. (2024).
/// This is independent of `Egor(cstr_infill=True)` which weights the criterion by the probability
/// of feasibility of the explicit constraints (`n_cstr`), both can be used together.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub(crate) enum FeasibleInfillStrategy {
    /// Do not use feasibility information
    None = 1,
    /// EFI with Probability (EFI_P): the criterion is weighted by the probability of viability
    EfiP = 2,
    /// EFI Feasibility Enhanced (EFI_FE): the criterion is weighted by the probability of viability
    /// to the power 0.3, which is more exploratory than EFI_P
    EfiFe = 3,
}

#[gen_stub_pymethods]
#[pymethods]
impl FeasibleInfillStrategy {
    /// Long name alias of EFI_P
    #[classattr]
    #[allow(non_snake_case)]
    fn EFI_PROBABILITY() -> FeasibleInfillStrategy {
        Self::EfiP
    }
    /// Long name alias of EFI_FE
    #[classattr]
    #[allow(non_snake_case)]
    fn EFI_FEASIBILITY_ENHANCED() -> FeasibleInfillStrategy {
        Self::EfiFe
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FeasibleInfillStrategy {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::None),
            Ok(2) => Ok(Self::EfiP),
            Ok(3) => Ok(Self::EfiFe),
            Ok(v) => Err(PyValueError::new_err(format!(
                "feasible_infill_strategy integer value must be in [1, 3], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "feasible_infill_strategy must be a FeasibleInfillStrategy enum or an integer in [1, 3]",
            )),
        }
    }
}

/// FailsafeStrategy specifies the strategy to use for handling failures during infill optimization.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub(crate) enum FailsafeStrategy {
    /// The point is ignored, the optimization continues but may fail to explore
    /// another region of the search space
    Rejection = 1,
    /// The point is added to the DOE with a penalized value, which allows
    /// the optimization to continue exploring other regions of the search space
    Imputation = 2,
    /// The viability of the point is modeled with a surrogate, which allows the optimization
    /// to learn which regions of the search space are more likely to fail and avoid them in the future
    Viability = 3,
}

impl<'a, 'py> FromPyObject<'a, 'py> for FailsafeStrategy {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Rejection),
            Ok(2) => Ok(Self::Imputation),
            Ok(3) => Ok(Self::Viability),
            Ok(v) => Err(PyValueError::new_err(format!(
                "failsafe_strategy integer value must be in [1, 3], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "failsafe_strategy must be a FailsafeStrategy enum or an integer in [1, 3]",
            )),
        }
    }
}

/// Verbose specifies the level of verbosity for logging.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) enum Verbose {
    Error = 0,
    Warning = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Verbose {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, PyErr> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(0) => Ok(Self::Error),
            Ok(1) => Ok(Self::Warning),
            Ok(2) => Ok(Self::Info),
            Ok(3) => Ok(Self::Debug),
            Ok(4) => Ok(Self::Trace),
            Ok(v) => Err(PyValueError::new_err(format!(
                "verbose integer value must be in [0, 4], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "verbose must be a Verbose enum or an integer in [0, 4]",
            )),
        }
    }
}

impl From<Verbose> for log::LevelFilter {
    fn from(value: Verbose) -> Self {
        match value {
            Verbose::Error => log::LevelFilter::Error,
            Verbose::Warning => log::LevelFilter::Warn,
            Verbose::Info => log::LevelFilter::Info,
            Verbose::Debug => log::LevelFilter::Debug,
            Verbose::Trace => log::LevelFilter::Trace,
        }
    }
}

/// XType specifies the type of the input variables.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum XType {
    Float = 1,
    Int = 2,
    Ord = 3,
    Enum = 4,
}

impl<'a, 'py> FromPyObject<'a, 'py> for XType {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Float),
            Ok(2) => Ok(Self::Int),
            Ok(3) => Ok(Self::Ord),
            Ok(4) => Ok(Self::Enum),
            Ok(v) => Err(PyValueError::new_err(format!(
                "xtype integer value must be in [1, 4], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "xtype must be an XType enum or an integer in [1, 4]",
            )),
        }
    }
}

/// XSpec specifies the type and limits of the input variables (aka design space).
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(FromPyObject, Debug)]
pub(crate) struct XSpec {
    #[pyo3(get)]
    pub(crate) xtype: XType,
    #[pyo3(get)]
    pub(crate) xlimits: Vec<f64>,
    #[pyo3(get)]
    pub(crate) tags: Vec<String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl XSpec {
    #[new]
    #[pyo3(signature = (xtype, xlimits=vec![], tags=vec![]))]
    pub(crate) fn new(xtype: XType, xlimits: Vec<f64>, tags: Vec<String>) -> Self {
        XSpec {
            xtype,
            xlimits,
            tags,
        }
    }
}

/// SparseMethod specifies the method to use for sparse Gaussian process regression.
/// See "Sparse Gaussian Process Regression for Big Data" by V. Vanhatalo, J. Riihimäki, J. Hartikainen, and A. Vehtari (2010)
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SparseMethod {
    /// FITC (Fully Independent Training Conditional) method, which uses a subset of the training data to make predictions, resulting in a faster but less accurate model
    Fitc = 1,
    /// VFE (Variational Free Energy) method, which uses a variational approach to approximate the posterior, resulting in a more accurate but slower model
    Vfe = 2,
}

impl<'a, 'py> FromPyObject<'a, 'py> for SparseMethod {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(value) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(*value);
        }
        match obj.extract::<u8>() {
            Ok(1) => Ok(Self::Fitc),
            Ok(2) => Ok(Self::Vfe),
            Ok(v) => Err(PyValueError::new_err(format!(
                "sparse method integer value must be in [1, 2], got {v}"
            ))),
            Err(_) => Err(PyTypeError::new_err(
                "method must be a SparseMethod enum or an integer in [1, 2]",
            )),
        }
    }
}

/// CstrSpec specifies how a constraint should be interpreted by the optimizer.
///
/// Instead of requiring constraints to be formulated as c <= 0,
/// users can specify constraint bounds directly.
///
/// Each spec can have its own tolerance `tol`: the constraint is considered satisfied
/// when the violation is below `tol`. It takes precedence over `Egor(cstr_tol=...)`.
///
/// # Examples
///
/// ```python
/// import egobox as egx
///
/// # c <= 5.0
/// spec1 = egx.CstrSpec.leq(5.0)
///
/// # c >= 2.0 with a tolerance of 1e-2
/// spec2 = egx.CstrSpec.geq(2.0, tol=1e-2)
///
/// # c = 4.0 (equality constraint, expands to two internal constraints)
/// spec3 = egx.CstrSpec.eq(4.0)
///
/// # 1.0 <= c <= 3.0 (double-sided, expands to two internal constraints)
/// spec4 = egx.CstrSpec.between(1.0, 3.0)
///
/// # dict form
/// spec5 = {"between": (1.0, 3.0), "tol": 1e-2}
/// ```
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Debug, Clone)]
pub(crate) struct CstrSpec {
    pub(crate) inner: egobox_ego::CstrSpec,
    /// Tolerance of the constraint, None means the `Egor(cstr_tol=...)` value or the default is used
    #[pyo3(get)]
    pub(crate) tol: Option<f64>,
}

/// Renamed `CstrSpec` dict keys as (deprecated, new) pairs
const CSTR_SPEC_RENAMED: [(&str, &str); 1] = [("btw", "between")];

impl<'a, 'py> FromPyObject<'a, 'py> for CstrSpec {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(spec) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(spec.clone());
        }

        let dict = obj.cast::<pyo3::types::PyDict>()?;
        let tol = dict
            .get_item("tol")?
            .map(|tol| tol.extract::<f64>())
            .transpose()?;
        let n_kinds = dict.len() - tol.is_some() as usize;
        if n_kinds != 1 {
            return Err(PyValueError::new_err(
                "CstrSpec dict form must contain exactly one key among: leq, geq, eq, between (and optionally tol)",
            ));
        }

        for (key, value) in dict.iter() {
            let key = key.extract::<String>()?;
            if key == "tol" {
                continue;
            }
            return match resolve_renamed_key(&dict, &key, &CSTR_SPEC_RENAMED)? {
                "leq" => Ok(CstrSpec::leq(value.extract()?, tol)),
                "geq" => Ok(CstrSpec::geq(value.extract()?, tol)),
                "eq" => Ok(CstrSpec::eq(value.extract()?, tol)),
                "between" => {
                    let (lower, upper): (f64, f64) = value.extract()?;
                    Ok(CstrSpec::between(lower, upper, tol))
                }
                _ => Err(PyValueError::new_err(format!(
                    "Unknown CstrSpec dict key '{key}'. Expected one of: leq, geq, eq, between (and optionally tol)"
                ))),
            };
        }
        unreachable!("dict has exactly one constraint kind key")
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CstrSpec {
    /// Constraint c <= bound, transformed to c - bound <= 0
    #[staticmethod]
    #[pyo3(signature = (bound, tol=None))]
    pub fn leq(bound: f64, tol: Option<f64>) -> Self {
        CstrSpec {
            inner: egobox_ego::CstrSpec::Leq(bound),
            tol,
        }
    }

    /// Constraint c >= bound, transformed to bound - c <= 0
    #[staticmethod]
    #[pyo3(signature = (bound, tol=None))]
    pub fn geq(bound: f64, tol: Option<f64>) -> Self {
        CstrSpec {
            inner: egobox_ego::CstrSpec::Geq(bound),
            tol,
        }
    }

    /// Equality constraint c = value, expands to two internal constraints:
    /// c - value <= 0 and value - c <= 0
    #[staticmethod]
    #[pyo3(signature = (value, tol=None))]
    pub fn eq(value: f64, tol: Option<f64>) -> Self {
        CstrSpec {
            inner: egobox_ego::CstrSpec::Eq(value),
            tol,
        }
    }

    /// Double-sided constraint lower <= c <= upper, expands to two internal constraints:
    /// lower - c <= 0 and c - upper <= 0
    #[staticmethod]
    #[pyo3(signature = (lower, upper, tol=None))]
    pub fn between(lower: f64, upper: f64, tol: Option<f64>) -> Self {
        CstrSpec {
            inner: egobox_ego::CstrSpec::Btw(lower, upper),
            tol,
        }
    }

    /// Deprecated since 0.38.0, use `CstrSpec.between` instead.
    #[staticmethod]
    #[pyo3(signature = (lower, upper, tol=None))]
    pub fn btw(py: Python, lower: f64, upper: f64, tol: Option<f64>) -> PyResult<Self> {
        warn_deprecated(py, "CstrSpec.btw", "CstrSpec.between")?;
        Ok(CstrSpec::between(lower, upper, tol))
    }

    fn __repr__(&self) -> String {
        match self.tol {
            Some(tol) => format!("{:?} (tol={tol})", self.inner),
            None => format!("{:?}", self.inner),
        }
    }
}

/// Format `Name(key1=repr1, key2=repr2, ...)` using the Python repr of each value
pub(crate) fn repr_kwargs(name: &str, fields: &[(&str, Bound<'_, PyAny>)]) -> PyResult<String> {
    let args = fields
        .iter()
        .map(|(key, value)| Ok(format!("{key}={}", value.repr()?)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(format!("{name}({})", args.join(", ")))
}

/// RunInfo contains information about a single run of the optimization algorithm,
/// the name of the function being optimized and the run number (useful for logging and saving results).
/// This is given by the user when calling the optimization function and is used for logging and saving results.
/// This information is also returned in the RunStatus to allow the user to correlate the results
/// with the function and run number.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Debug, Clone)]
pub(crate) struct RunInfo {
    /// A name for the function being optimized, used for logging and saving results
    #[pyo3(get, set)]
    pub(crate) fname: String,
    /// A number for the run, used for logging and saving results
    #[pyo3(get, set)]
    pub(crate) num: usize,
}

impl RunInfo {
    /// Default function name, the same as the one used by the Rust `egobox_ego::RunInfo`
    pub(crate) const DEFAULT_FNAME: &'static str = "objective_function";
}

impl Default for RunInfo {
    fn default() -> Self {
        RunInfo {
            fname: Self::DEFAULT_FNAME.to_string(),
            num: 1,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for RunInfo {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(info) = obj.extract::<PyRef<'py, Self>>() {
            return Ok(info.clone());
        }

        let dict = obj.cast::<pyo3::types::PyDict>()?;
        let mut info = RunInfo::default();

        for key_any in dict.keys().iter() {
            let key = key_any.extract::<String>()?;
            match key.as_str() {
                "fname" => info.fname = dict.get_item("fname")?.unwrap().extract()?,
                "num" => info.num = dict.get_item("num")?.unwrap().extract()?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown run_info key '{key}'"
                    )));
                }
            }
        }

        Ok(info)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl RunInfo {
    #[new]
    #[pyo3(signature = (fname=RunInfo::DEFAULT_FNAME.to_string(), num = 1))]
    pub fn new(fname: String, num: usize) -> Self {
        RunInfo { fname, num }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "RunInfo",
            &[
                ("fname", self.fname.as_str().into_pyobject(py)?.into_any()),
                ("num", self.num.into_pyobject(py)?.into_any()),
            ],
        )
    }
}

/// ExitStatus specifies the reason for the termination of the optimization algorithm.
#[gen_stub_pyclass_enum]
#[pyclass(
    skip_from_py_object,
    module = "egobox",
    eq,
    eq_int,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExitStatus {
    /// Reached maximum number of iterations
    MaxItersReached = 1,
    /// Reached target cost function value
    TargetCostReached = 2,
    /// Algorithm manually interrupted with SIGINT (Ctrl+C), SIGTERM or SIGHUP
    Interrupt = 3,
    /// Algorithm picked the same point twice. We consider it is converged.
    SolverConverged = 4,
    /// Timeout reached
    Timeout = 5,
    /// Solver unexpected exit. See logs for details.
    UnexpectedExit = 6,
    /// Objective function returned an error. See logs for details.
    ObjectiveFunctionError = 7,
}

impl From<egobox_ego::TerminationStatus> for ExitStatus {
    fn from(value: egobox_ego::TerminationStatus) -> Self {
        use egobox_ego::{TerminationReason, TerminationStatus};
        match value {
            TerminationStatus::Terminated(reason) => match reason {
                TerminationReason::MaxItersReached => ExitStatus::MaxItersReached,
                TerminationReason::TargetCostReached => ExitStatus::TargetCostReached,
                TerminationReason::SolverConverged => ExitStatus::SolverConverged,
                TerminationReason::Timeout => ExitStatus::Timeout,
                TerminationReason::SolverExit(val) if val == OBJECTIVE_FUNCTION_ERROR => {
                    ExitStatus::ObjectiveFunctionError
                }
                TerminationReason::SolverExit(_) => unreachable!("Unexpected solver exit reason"),
                TerminationReason::Interrupt => ExitStatus::Interrupt,
            },
            TerminationStatus::NotTerminated => ExitStatus::UnexpectedExit,
        }
    }
}

/// RunStatus contains information about the status of a run of the optimization algorithm
/// It is returned by the optimizer together with the optimization results.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Debug, Clone)]
pub(crate) struct RunStatus {
    /// Information about the run, provided by the user when calling the optimization function
    #[pyo3(get)]
    pub(crate) info: RunInfo,
    /// Exit status of the optimization algorithm, which indicates the reason for termination of the algorithm
    #[pyo3(get)]
    pub(crate) exit: ExitStatus,
    /// Number of points in the initial DOE, which is useful to correlate with the results and understand the behavior of the optimization algorithm
    #[pyo3(get)]
    pub(crate) init_doe_size: usize,
    /// Best iteration of the optimization algorithm, allows to retrieve optimal values in the optimization history
    #[pyo3(get)]
    pub(crate) best_iter: usize,
    /// Total number of iterations performed by the optimization algorithm
    #[pyo3(get)]
    pub(crate) total_iters: usize,
    /// Elapsed time of the optimization algorithm in seconds
    #[pyo3(get)]
    pub(crate) elapsed_time: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl RunStatus {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "RunStatus",
            &[
                ("info", self.info.clone().into_pyobject(py)?.into_any()),
                ("exit", self.exit.clone().into_pyobject(py)?.into_any()),
                (
                    "init_doe_size",
                    self.init_doe_size.into_pyobject(py)?.into_any(),
                ),
                ("best_iter", self.best_iter.into_pyobject(py)?.into_any()),
                (
                    "total_iters",
                    self.total_iters.into_pyobject(py)?.into_any(),
                ),
                (
                    "elapsed_time",
                    self.elapsed_time.into_pyobject(py)?.into_any(),
                ),
            ],
        )
    }
}

/// OptimResult contains the results of a run of the optimization algorithm,
/// including the optimal point and value found, the DOE points and values which
/// includes initial points and the optimization history.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Debug)]
pub(crate) struct OptimResult {
    /// Optimal x point found by the optimization algorithm
    #[pyo3(get)]
    pub(crate) x_opt: Py<PyArray1<f64>>,
    /// Optimal y point found by the optimization algorithm
    #[pyo3(get)]
    pub(crate) y_opt: Py<PyArray1<f64>>,
    /// DOE x points, including initial points and optimization history
    #[pyo3(get)]
    pub(crate) x_doe: Py<PyArray2<f64>>,
    /// DOE y points, including initial points and optimization history
    #[pyo3(get)]
    pub(crate) y_doe: Py<PyArray2<f64>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl OptimResult {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let n_doe = self.x_doe.bind(py).len()?;
        repr_kwargs(
            "OptimResult",
            &[
                ("x_opt", self.x_opt.bind(py).clone().into_any()),
                ("y_opt", self.y_opt.bind(py).clone().into_any()),
                ("n_doe", n_doe.into_pyobject(py)?.into_any()),
            ],
        )
    }
}

/// Egor optimization output
///
/// The optimization result fields are also available directly (`x_opt`, `y_opt`, `x_doe`, `y_doe`)
/// and the output can be unpacked as `x_opt, y_opt = egor.minimize(...)`.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Debug)]
pub(crate) struct EgorOptim {
    /// Result of optimization run
    #[pyo3(get)]
    pub(crate) result: Py<OptimResult>,
    /// Status of optimization run
    #[pyo3(get)]
    pub(crate) status: RunStatus,
}

#[gen_stub_pymethods]
#[pymethods]
impl EgorOptim {
    /// Optimal x point found by the optimization algorithm, same as `result.x_opt`
    #[getter]
    fn x_opt(&self, py: Python) -> Py<PyArray1<f64>> {
        self.result.borrow(py).x_opt.clone_ref(py)
    }

    /// Optimal y point found by the optimization algorithm, same as `result.y_opt`
    #[getter]
    fn y_opt(&self, py: Python) -> Py<PyArray1<f64>> {
        self.result.borrow(py).y_opt.clone_ref(py)
    }

    /// DOE x points, including initial points and optimization history, same as `result.x_doe`
    #[getter]
    fn x_doe(&self, py: Python) -> Py<PyArray2<f64>> {
        self.result.borrow(py).x_doe.clone_ref(py)
    }

    /// DOE y points, including initial points and optimization history, same as `result.y_doe`
    #[getter]
    fn y_doe(&self, py: Python) -> Py<PyArray2<f64>> {
        self.result.borrow(py).y_doe.clone_ref(py)
    }

    /// Iterate over (x_opt, y_opt) to allow `x_opt, y_opt = egor.minimize(...)`
    #[gen_stub(override_return_type(type_repr = "typing.Iterator[numpy.typing.NDArray[numpy.float64]]", imports = ("typing", "numpy", "numpy.typing")))]
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let result = self.result.borrow(py);
        PyTuple::new(
            py,
            [
                result.x_opt.bind(py).as_any(),
                result.y_opt.bind(py).as_any(),
            ],
        )?
        .try_iter()
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "EgorOptim",
            &[
                ("result", self.result.bind(py).clone().into_any()),
                ("status", self.status.clone().into_pyobject(py)?.into_any()),
            ],
        )
    }
}

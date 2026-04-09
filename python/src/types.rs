use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, PartialEq)]
pub enum Recombination {
    /// prediction is taken from the expert with highest responsability
    /// resulting in a model with discontinuities
    Hard = 0,
    /// Prediction is a combination experts prediction wrt their responsabilities,
    /// an optional heaviside factor might be used control steepness of the change between
    /// experts regions.
    Smooth = 1,
}

/// RegressionSpec is a bitfield that specifies which regression terms to include in the model.
#[gen_stub_pyclass]
#[pyclass]
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
#[pyclass]
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
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InfillStrategy {
    /// Expected Improvement
    /// see Mockus et al. (1978) "The application of Bayesian methods for seeking the extremum"
    Ei = 1,
    /// Warnes and Barnes 2nd EI improvement, shift EI by the GP mean
    /// easier to optimize than EI but may not explore as much as EI
    /// see Warnes and Barnes (2020) "A new acquisition function for batch Bayesian optimization"
    Wb2 = 2,
    /// Warnes and Barnes 2nd scaling to improve exploration
    Wb2s = 3,
    /// Logarithm of Expected Improvement
    /// see Ament et al. (2020) "Logarithmic Expected Improvement for Robust and Noisy Bayesian Optimization"
    LogEi = 4,
}

/// ConstraintStrategy specifies the strategy to use for handling constraints in infill optimization.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ConstraintStrategy {
    /// Mean of the GP is used to evaluate the constraint, which is equivalent to ignoring the uncertainty on the constraint
    Mc = 1,
    /// Upper trusted bound of the GP is used to evaluate the constraint, which takes into account the uncertainty on the constraint
    Utb = 2,
}

/// QEiStrategy specifies the strategy to use for handling constraints in infill optimization.
/// see QEI is the multi-point extension of EI, see Chevalier and Ginsbourger (2013)
/// "Fast Computation of the Multi-Points Expected Improvement with Applications in Batch Selection"
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum QEiStrategy {
    /// Kriging Believer, the next point is added to the GP with its predicted mean value,
    /// which is equivalent to assuming that the prediction is perfect
    Kb = 1,
    /// Kriging Believer lower bound, the next point is added to the GP with
    /// its predicted mean value minus a multiple of the predicted standard deviation,
    /// which is equivalent to assuming that the prediction is pessimistic
    Kblb = 2,
    /// Kriging Believer upper bound, the next point is added to the GP with
    /// its predicted mean value plus a multiple of the predicted standard deviation,
    /// which is equivalent to assuming that the prediction is optimistic
    Kbub = 3,
    /// Constant Liar, the next point is added to the GP by using the current minimum
    /// value observed in the DOE, which is equivalent to assuming that
    /// the prediction is the current best value
    Clmin = 4,
}

/// InfillOptimizer specifies the optimization algorithm to use for infill optimization.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InfillOptimizer {
    /// Gradient free optimization algorithm that uses a simplex of n+1 points for n-dimensional optimization
    Cobyla = 1,
    /// Gradient based optimization algorithm that uses a quasi-Newton method to optimize the acquisition function
    Slsqp = 2,
}

/// FailsafeStrategy specifies the strategy to use for handling failures during infill optimization.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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

/// Verbose specifies the level of verbosity for logging.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) enum Verbose {
    Error = 0,
    Warning = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
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
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum XType {
    Float = 1,
    Int = 2,
    Ord = 3,
    Enum = 4,
}

/// XSpec specifies the type and limits of the input variables (aka design space).
#[gen_stub_pyclass]
#[pyclass]
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
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[gen_stub_pyclass_enum]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SparseMethod {
    /// FITC (Fully Independent Training Conditional) method, which uses a subset of the training data to make predictions, resulting in a faster but less accurate model
    Fitc = 1,
    /// VFE (Variational Free Energy) method, which uses a variational approach to approximate the posterior, resulting in a more accurate but slower model
    Vfe = 2,
}

/// CstrSpec specifies how a constraint should be interpreted by the optimizer.
///
/// Instead of requiring constraints to be formulated as c <= 0,
/// users can specify constraint bounds directly.
///
/// # Examples
///
/// ```python
/// import egobox as egx
///
/// # c <= 5.0
/// spec1 = egx.CstrSpec.leq(5.0)
///
/// # c >= 2.0
/// spec2 = egx.CstrSpec.geq(2.0)
///
/// # c = 4.0 (equality constraint, expands to two internal constraints)
/// spec3 = egx.CstrSpec.eq(4.0)
///
/// # 1.0 <= c <= 3.0 (double-sided, expands to two internal constraints)
/// spec4 = egx.CstrSpec.btw(1.0, 3.0)
/// ```
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct CstrSpec {
    pub(crate) inner: egobox_ego::CstrSpec,
}

#[gen_stub_pymethods]
#[pymethods]
impl CstrSpec {
    /// Constraint c <= bound, transformed to c - bound <= 0
    #[staticmethod]
    pub fn leq(bound: f64) -> Self {
        CstrSpec {
            inner: egobox_ego::CstrSpec::Leq(bound),
        }
    }

    /// Constraint c >= bound, transformed to bound - c <= 0
    #[staticmethod]
    pub fn geq(bound: f64) -> Self {
        CstrSpec {
            inner: egobox_ego::CstrSpec::Geq(bound),
        }
    }

    /// Equality constraint c = value, expands to two internal constraints:
    /// c - value <= 0 and value - c <= 0
    #[staticmethod]
    pub fn eq(value: f64) -> Self {
        CstrSpec {
            inner: egobox_ego::CstrSpec::Eq(value),
        }
    }

    /// Double-sided constraint lower <= c <= upper, expands to two internal constraints:
    /// lower - c <= 0 and c - upper <= 0
    #[staticmethod]
    pub fn btw(lower: f64, upper: f64) -> Self {
        CstrSpec {
            inner: egobox_ego::CstrSpec::Btw(lower, upper),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// RunInfo contains information about a single run of the optimization algorithm,
/// the name of the function being optimized and the run number (useful for logging and saving results).
/// This is given by the user when calling the optimization function and is used for logging and saving results.
/// This information is also returned in the RunStatus to allow the user to correlate the results
/// with the function and run number.
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct RunInfo {
    /// A name for the function being optimized, used for logging and saving results
    #[pyo3(get, set)]
    pub(crate) fname: String,
    /// A number for the run, used for logging and saving results
    #[pyo3(get, set)]
    pub(crate) num: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl RunInfo {
    #[new]
    #[pyo3(signature = (fname="fobj".to_string(), num = 1))]
    pub fn new(fname: String, num: usize) -> Self {
        RunInfo { fname, num }
    }
}

/// ExitStatus specifies the reason for the termination of the optimization algorithm.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExitStatus {
    /// Reached maximum number of iterations
    MaxItersReached = 1,
    /// Reached target cost function value
    TargetCostReached = 2,
    /// Algorithm manually interrupted with SIGINT (Ctrl+C), SIGTERM or SIGHUP
    Interrupt = 3,
    /// Algorithm peek at the same point twice. We consider it is converged.
    SolverConverged = 4,
    /// Timeout reached
    Timeout = 5,
    /// Solver unexpected exit. See logs for details.
    UnexpectedExit = 6,
}

impl From<argmin::core::TerminationStatus> for ExitStatus {
    fn from(value: argmin::core::TerminationStatus) -> Self {
        use argmin::core::{TerminationReason, TerminationStatus};
        match value {
            TerminationStatus::Terminated(reason) => match reason {
                TerminationReason::MaxItersReached => ExitStatus::MaxItersReached,
                TerminationReason::TargetCostReached => ExitStatus::TargetCostReached,
                TerminationReason::SolverConverged => ExitStatus::SolverConverged,
                TerminationReason::Timeout => ExitStatus::Timeout,
                TerminationReason::SolverExit(_) => ExitStatus::UnexpectedExit,
                TerminationReason::Interrupt => ExitStatus::Interrupt,
            },
            TerminationStatus::NotTerminated => ExitStatus::UnexpectedExit,
        }
    }
}

/// RunStatus contains information about the status of a run of the optimization algorithm
/// It is returned by the optimizer together with the optimization results.
#[gen_stub_pyclass]
#[pyclass]
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

/// OptimResult contains the results of a run of the optimization algorithm,
/// including the optimal point and value found, the DOE points and values which
/// includes initial points and the optimization history.
#[gen_stub_pyclass]
#[pyclass]
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

/// Egor optimization output
///
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug)]
pub(crate) struct EgorOptim {
    /// Result of optimization run
    #[pyo3(get)]
    pub(crate) result: Py<OptimResult>,
    /// Status of optimization run
    #[pyo3(get)]
    pub(crate) status: RunStatus,
}

#![allow(clippy::useless_conversion)]
//! `egobox`, Rust toolbox for efficient global optimization
//!
//! Thanks to the [PyO3 project](https://pyo3.rs), which makes Rust well suited for building Python extensions,
//! the EGO algorithm written in Rust (aka `Egor`) is binded in Python. You can install the Python package using:
//!
//! ```bash
//! pip install egobox
//! ```
//!
//! See the [tutorial notebook](https://github.com/relf/egobox/notebooks/Egor_Tutorial.ipynb) for usage.
//!

use crate::deprecation::{resolve_renamed, warn_deprecated};
use crate::domain::*;
use crate::errors::{CallbackError, ego_err, install_panic_hook};
use crate::gp_config::*;
use crate::logging::init_logger;
use crate::qei_config::*;
use crate::trego_config::{TregoConfig, TregoConfigSpec};
use crate::types::*;

use egobox_ego::{CoegoStatus, EGO_DEFAULT_N_START, InfillObjData, find_best_result_index};
use egobox_gp::ThetaTuning;
use egobox_moe::NbClusters;
use ndarray::{Array1, Array2, ArrayView2, Axis, array, concatenate};
use numpy::{
    IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::cmp::Ordering;

fn parse_trego_config(py: Python, value: Py<PyAny>) -> PyResult<TregoConfigSpec> {
    let value = value.bind(py);
    if let Ok(active) = value.cast::<PyBool>() {
        return Ok(TregoConfigSpec::Activated(active.is_true()));
    }
    if value.is_instance_of::<TregoConfig>() || value.is_instance_of::<pyo3::types::PyDict>() {
        // dict keys are checked by TregoConfig extraction
        return Ok(TregoConfigSpec::Custom(value.extract()?));
    }
    Err(PyTypeError::new_err(
        "trego should be a TregoConfig, a dict, a bool or None",
    ))
}

fn parse_run_info(py: Python, value: Py<PyAny>) -> PyResult<RunInfo> {
    let value = value.bind(py);
    if !value.is_instance_of::<RunInfo>() && !value.is_instance_of::<pyo3::types::PyDict>() {
        return Err(PyTypeError::new_err(
            "run_info should be a RunInfo or a dict",
        ));
    }
    value.extract()
}

/// Optimizer constructor
///
/// Parameters
/// ----------
/// xspecs : list of XSpec, list of [lower, upper] or array[nx, 2]
///     Specifications of the nx components of the input x (eg. len(xspecs) == nx),
///     with XSpec(xtype=FLOAT|INT|ORD|ENUM, xlimits=[<f(xtype)>] or tags=[strings]).
///     Depending on the x type we get the following for xlimits:
///
///     * when FLOAT: xlimits is [float lower_bound, float upper_bound],
///     * when INT: xlimits is [int lower_bound, int upper_bound],
///     * when ORD: xlimits is [float_1, float_2, ..., float_n],
///     * when ENUM: xlimits is just the int size of the enumeration otherwise a list of tags is specified
///       (eg xlimits=[3] or tags=["red", "green", "blue"], tags are there for documentation purpose but
///       tags specific values themselves are not used only indices in the enum are used hence
///       we can just specify the size of the enum, xlimits=[3]).
/// gp_config : GpConfig or dict, optional
///     GP configuration used by the optimizer, see GpConfig for details.
/// n_cstr : int
///     Number of constraints returned by `fun` (see `minimize`) which will be approximated by surrogates.
///     Can be omitted when `cstr_specs` is given.
/// cstr_specs : list of CstrSpec or dict, optional
///     Describe how each surrogate-modeled constraint (returned by `fun`) should be interpreted.
///     This allows users to define bounds directly instead of manually rewriting
///     constraints in `c <= 0` form:
///
///     * CstrSpec.leq(bound): c <= bound (less or equal)
///     * CstrSpec.geq(bound): c >= bound (greater or equal)
///     * CstrSpec.eq(value): c == value (expands to two internal constraints)
///     * CstrSpec.between(lower, upper): lower <= c <= upper (expands to two internal constraints)
///
///     Each spec accepts an optional `tol` argument, the tolerance of that constraint
///     (cstr < tol after the rewriting in `c <= 0` form, default is DEFAULT_CSTR_TOL=1e-4).
///     When set, `n_cstr` is inferred from `len(cstr_specs)` (`n_cstr` can be omitted,
///     otherwise it must match, ValueError is raised).
/// infill_n_start : int > 0, optional
///     Number of starts of the multistart optimization of the infill criterion (best result taken, default is 20).
///     Not to be confused with `GpConfig(theta_n_start=...)`, the GP hyperparameters optimization multistart.
/// n_doe : int >= 0
///     Number of samples of initial LHS sampling (used when DOE not provided by the user).
///     When 0 a number of points is computed automatically regarding the number of input variables
///     of the function under optimization.
/// x_doe : array[ns, nx], optional
///     Initial DOE inputs containing ns samples. When `y_doe` is not given,
///     ns evaluations are done to get the output values.
/// y_doe : array[ns, ny], optional
///     Initial DOE outputs [obj, cstr_1, ... cstr_k] (ny = 1 + n_cstr) corresponding to `x_doe`,
///     requires `x_doe`.
/// infill_strategy : InfillStrategy
///     Infill criterion to decide best next promising point.
///     Can be either InfillStrategy.LOG_EI (default), InfillStrategy.EI, InfillStrategy.WB2, InfillStrategy.WB2S
/// feasible_infill_strategy : FeasibleInfillStrategy
///     Weight the infill criterion by the probability of viability to avoid regions where
///     `fun` fails (hidden constraints). Can be either FeasibleInfillStrategy.NONE (default),
///     FeasibleInfillStrategy.EFI_P, or FeasibleInfillStrategy.EFI_FE.
///     Independent of `cstr_infill`, both can be used together.
/// cstr_infill : bool
///     Activate constrained infill criterion where the product of probabilities of feasibility
///     of the `n_cstr` surrogate constraints is used as a factor of the infill criterion.
///     Independent of `feasible_infill_strategy`, both can be used together.
/// cstr_strategy : ConstraintStrategy
///     Constraint management, either use the mean value or the upper trust bound of the constraint surrogates.
///     Can be either ConstraintStrategy.MC (mean constraint, default) or ConstraintStrategy.UTB (upper trust bound).
/// qei_config : QEiConfig or dict, optional
///     Configuration for parallel (qEI) evaluation also known as batch or multipoint evaluation.
///     q points are selected at each iteration of the EGO algorithm.
///     See QEiConfig for details.
/// infill_optimizer : InfillOptimizer
///     Internal optimizer used to optimize infill criteria.
///     Can be either InfillOptimizer.COBYLA (default) or InfillOptimizer.SLSQP
/// trego : TregoConfig, bool or dict, optional
///     TREGO configuration to activate TREGO strategy for global optimization.
///     When True activate TREGO with default configuration.
///     To activate TREGO with custom configuration see TregoConfig for details.
///     When None or False TREGO is not used.
/// coego_n_coop : int >= 0
///     Number of cooperative components groups which will be used by the CoEGO algorithm.
///     Better to have n_coop a divider of nx or if not with a remainder as large as possible.
///     The CoEGO algorithm is used to tackle high-dimensional problems turning it in a set of
///     partial optimizations using only nx / n_coop components at a time.
///     The default value is 0 meaning that the CoEGO algorithm is not used.
/// target : float, optional
///     Known optimum used as stopping criterion: the optimization stops once
///     an objective value lower than or equal to target is found.
///     When None (default) no target is used.
/// failsafe_strategy : FailsafeStrategy
///     Strategy to handle objective computation failure at a given x point.
///     A failure is detected when the objective function returns NaN value(s).
///     Can be either FailsafeStrategy.REJECTION (default), FailsafeStrategy.IMPUTATION, or FailsafeStrategy.VIABILITY.
///     Rejection simply ignores the failed point whereas Imputation
///     uses the objective surrogate prediction to fill the missing value.
///     In the third case Viability, a surrogate is used to model the failure region
///     which is used as a constraint and drive the optimization toward the viable region.
/// seed : int >= 0, optional
///     Random generator seed used by `minimize()` and `suggest()` when they are not given one.
/// verbose : Verbose or int, optional
///     Logging verbosity level used by `minimize()` and `suggest()` when `minimize()` is not given one.
///     See `minimize()` for the possible values.
///
/// Deprecated
/// ----------
/// cstr_tol : list of float, optional
///     Deprecated since 0.38.0, give a `tol` to each spec of `cstr_specs` / `fcstr_specs` instead
///     (e.g. `CstrSpec.leq(0.0, tol=1e-3)`).
///     Tolerances for constraints to be satisfied (cstr < tol), covering all internal constraints:
///     the `n_cstr` surrogate constraints (after `cstr_specs` expansion) followed by the function
///     constraints `fcstrs` given to `minimize` (after `fcstr_specs` expansion).
///     A spec `tol` takes precedence over `cstr_tol`.
/// n_start : int > 0, optional
///     Deprecated since 0.38.0, use `infill_n_start` instead.
/// doe : array[ns, nt], optional
///     Deprecated since 0.38.0, use `x_doe` and `y_doe` instead.
///     Initial DOE containing ns samples:
///     either nt = nx then only x are specified and ns evals are done to get y doe values,
///     or nt = nx + ny then x = doe[:, :nx] and y = doe[:, nx:] are specified.
///
/// Returns
/// -------
/// Egor
///     An optimizer which can be used to optimize a function using the minimize method.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct Egor {
    pub xtypes: Vec<egobox_moe::XType>,
    pub gp_config: GpConfig,
    pub n_cstr: usize,
    pub cstr_tol: Option<Vec<f64>>,
    pub cstr_specs: Option<Vec<CstrSpec>>,
    pub infill_n_start: usize,
    pub n_doe: usize,
    pub doe: Option<Array2<f64>>,
    pub infill_strategy: InfillStrategy,
    pub feasible_infill_strategy: FeasibleInfillStrategy,
    pub cstr_infill: bool,
    pub cstr_strategy: ConstraintStrategy,
    pub qei_config: QEiConfig,
    pub infill_optimizer: InfillOptimizer,
    pub trego: Option<TregoConfig>,
    pub coego_n_coop: usize,
    pub target: Option<f64>,
    pub failsafe_strategy: FailsafeStrategy,
    pub seed: Option<u64>,
    pub verbose: Option<Py<PyAny>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Egor {
    #[new]
    #[pyo3(signature = (
        xspecs,
        gp_config = None,
        n_cstr = 0,
        cstr_tol = None,
        cstr_specs = None,
        infill_n_start = None,
        n_doe = 0,
        x_doe = None,
        y_doe = None,
        infill_strategy = InfillStrategy::LogEi,
        feasible_infill_strategy = FeasibleInfillStrategy::None,
        cstr_infill = false,
        cstr_strategy = ConstraintStrategy::Mc,
        qei_config = None,
        infill_optimizer = InfillOptimizer::Cobyla,
        trego = None,
        coego_n_coop = 0,
        target = None,
        failsafe_strategy = FailsafeStrategy::Rejection,
        seed = None,
        verbose = None,
        *,
        n_start = None,
        doe = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        #[gen_stub(override_type(type_repr = "typing.Sequence[XSpec] | typing.Sequence[typing.Sequence[builtins.float]] | numpy.typing.NDArray[numpy.float64]", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        xspecs: Py<PyAny>,
        #[gen_stub(override_type(type_repr = "GpConfig | builtins.dict[builtins.str, typing.Any] | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        gp_config: Option<GpConfig>,
        n_cstr: usize,
        cstr_tol: Option<Vec<f64>>,
        #[gen_stub(override_type(type_repr = "typing.Sequence[CstrSpec | builtins.dict[builtins.str, typing.Any]] | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        cstr_specs: Option<Vec<CstrSpec>>,
        infill_n_start: Option<usize>,
        n_doe: usize,
        x_doe: Option<PyReadonlyArray2<f64>>,
        y_doe: Option<PyReadonlyArray2<f64>>,
        infill_strategy: InfillStrategy,
        feasible_infill_strategy: FeasibleInfillStrategy,
        cstr_infill: bool,
        cstr_strategy: ConstraintStrategy,
        #[gen_stub(override_type(type_repr = "QEiConfig | builtins.dict[builtins.str, typing.Any] | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        qei_config: Option<QEiConfig>,
        infill_optimizer: InfillOptimizer,
        #[gen_stub(override_type(type_repr = "TregoConfig | builtins.bool | builtins.dict[builtins.str, typing.Any] | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        trego: Option<Py<PyAny>>,
        coego_n_coop: usize,
        target: Option<f64>,
        failsafe_strategy: FailsafeStrategy,
        seed: Option<u64>,
        #[gen_stub(override_type(type_repr = "Verbose | builtins.int | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        verbose: Option<Py<PyAny>>,
        n_start: Option<usize>,
        doe: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        if cstr_tol.is_some() {
            warn_deprecated(py, "cstr_tol", "CstrSpec(..., tol=...)")?;
        }
        let infill_n_start = resolve_renamed(
            py,
            "n_start",
            n_start,
            "infill_n_start",
            infill_n_start,
            EGO_DEFAULT_N_START,
        )?;
        let xtypes = parse(py, xspecs.clone_ref(py))?;
        let doe = initial_doe(py, xtypes.len(), doe, x_doe, y_doe)?;
        let n_cstr = match cstr_specs.as_ref() {
            Some(specs) if n_cstr != 0 && n_cstr != specs.len() => {
                return Err(PyValueError::new_err(format!(
                    "n_cstr ({n_cstr}) must match cstr_specs length ({}), n_cstr can be omitted",
                    specs.len()
                )));
            }
            Some(specs) => specs.len(),
            None => n_cstr,
        };
        let gp_config = gp_config.unwrap_or_default();
        gp_config.validate()?;
        let qei_config = qei_config.unwrap_or_default();

        // Parse trego configuration: boolean or custom configuration
        let trego = match trego {
            Some(trego_py) => {
                let trego_typ = parse_trego_config(py, trego_py)?;
                match trego_typ {
                    TregoConfigSpec::Activated(active) => {
                        if active {
                            // True case
                            Some(TregoConfig::default())
                        } else {
                            // False case
                            None
                        }
                    }
                    TregoConfigSpec::Custom(cfg) => Some(cfg.into()),
                }
            }
            // None case
            None => None,
        };
        log::info!("TREGO config: {:?}", trego);

        Ok(Egor {
            xtypes,
            gp_config,
            n_cstr,
            cstr_tol,
            cstr_specs,
            infill_n_start,
            n_doe,
            doe,
            infill_strategy,
            cstr_infill,
            cstr_strategy,
            feasible_infill_strategy,
            qei_config,
            infill_optimizer,
            trego,
            coego_n_coop,
            target,
            failsafe_strategy,
            seed,
            verbose,
        })
    }

    /// This function finds the minimum of a given function "fun"
    ///
    /// Parameters
    /// ----------
    /// fun : callable (array[n, nx]) -> array[n, ny]
    ///     The function to be minimized: fun(x) = [obj(x), cstr_1(x), ... cstr_k(x)] where
    ///
    ///     * obj is the objective function [n, nx] -> [n, 1]
    ///     * cstr_i is the ith constraint function [n, nx] -> [n, 1]
    ///     * k is the number of constraints (n_cstr), hence ny = 1 (obj) + k (cstrs)
    ///
    ///     cstr functions are expected to be negative (<=0) at the optimum (unless `cstr_specs` is used).
    ///     These constraints will be approximated using surrogates, so
    ///     if constraints are cheap to evaluate better to pass them through `fcstrs`.
    /// fcstrs : list, optional
    ///     Cheap constraint functions g, evaluated directly (not approximated by surrogates),
    ///     which have to be made negative (g(x) <= 0, unless `fcstr_specs` is used) by the optimizer.
    ///     Each item is given in one of the following forms:
    ///
    ///     * (g, grad_g): a tuple of two callables, g(x) returns the constraint float value
    ///       and grad_g(x) returns its gradient (array[nx]) wrt the nx components of x,
    ///     * {"fun": g, "jac": grad_g}: the same as a dict (a scipy "type" key is rejected,
    ///       as scipy "ineq" constraints are g(x) >= 0, use `fcstr_specs` instead),
    ///     * g(x, return_grad): a single callable returning the constraint float value
    ///       when return_grad is False, its gradient (array[nx]) otherwise.
    ///
    ///     The gradient is only computed when the infill optimizer needs it (InfillOptimizer.SLSQP).
    /// fcstr_specs : list of CstrSpec or dict, optional
    ///     One CstrSpec per fcstr specifying how each function constraint should be interpreted.
    ///     Length must be zero (legacy behavior) or equal to len(fcstrs).
    ///     This allows raw constraints not written as c <= 0, for example:
    ///     CstrSpec.leq(b), CstrSpec.geq(b), CstrSpec.eq(v), CstrSpec.between(lo, hi).
    ///     Note: CstrSpec.eq and CstrSpec.between expand to two internal constraints each.
    ///     A spec `tol` (e.g. CstrSpec.leq(b, tol=1e-3)) gives the tolerance of that constraint.
    /// max_iters : int
    ///     The iteration budget, number of fun calls is "n_doe + q_batch * max_iters".
    ///     Not to be confused with `GpConfig(theta_max_eval=...)`, the likelihood evaluations budget.
    /// run_info : RunInfo or dict, optional
    ///     Information about the run to be passed to the optimizer with the following attributes:
    ///
    ///     * fname (str): name of the function under optimization, used for checkpoint file naming
    ///     * num (int): number of the run, used for checkpoint file naming
    /// outdir : str, optional
    ///     Directory to write optimization history and used as search path for warm start doe
    /// warm_start : bool
    ///     Start by loading initial doe from <outdir> directory
    /// hot_start : bool or int >= 0, optional
    ///     When hot_start>=0 saves optimizer state at each iteration and starts from a previous checkpoint
    ///     for the given hot_start number of iterations beyond the max_iters nb of iterations.
    ///     In an unstable environment where there can be crashes it allows to restart the optimization
    ///     from the last iteration till stopping criterion is reached. Just use hot_start=0 in this case.
    ///     When True, hot_start behaves like hot_start=0 with no iteration extension.
    ///     Checkpoint information is stored in .checkpoint or under outdir if outdir is specified.
    /// seed : int >= 0, optional
    ///     Random generator seed to allow computation reproducibility.
    ///     When None, the seed given to the constructor (if any) is used.
    /// timeout : float, optional
    ///     Timeout in seconds. The optimization is stopped when the elapsed time
    ///     exceeds this duration. The actual runtime may slightly exceed the specified timeout
    ///     as the check is performed after each iteration.
    /// verbose : Verbose or int, optional
    ///     Logging verbosity level for the optimizer.
    ///     Can be either an integer or a Verbose enum value:
    ///     0 or Verbose.ERROR, 1 or Verbose.WARNING, 2 or Verbose.INFO,
    ///     3 or Verbose.DEBUG, 4 (or greater) or Verbose.TRACE.
    ///     Default is None which means the verbosity given to the constructor if any,
    ///     otherwise Verbose.ERROR level and possible control by the EGOBOX_LOG environment variable.
    /// stop_on_error : bool
    ///     If true, terminate optimization when the objective function raises an error.
    ///     Otherwise, the error is handled according to failsafe_strategy.
    ///
    /// Returns
    /// -------
    /// EgorOptim
    ///     result (OptimResult) and status (RunStatus) of the optimization, where result holds:
    ///
    ///     * x_opt (array[nx]): x value where fun is at its minimum subject to constraints
    ///     * y_opt (array[ny]): fun(x_opt) where ny = 1 + n_cstr
    ///     * x_doe (array[ns, nx]): x values of the final DOE
    ///     * y_doe (array[ns, ny]): y values of the final DOE
    ///
    #[pyo3(signature = (fun, fcstrs=None, fcstr_specs=None, max_iters = 20, run_info = None, outdir = None, warm_start = false, hot_start = None, seed = None, timeout = None, verbose = None, stop_on_error = false))]
    #[allow(clippy::too_many_arguments)]
    fn minimize(
        &self,
        py: Python,
        #[gen_stub(override_type(type_repr = "typing.Callable[[numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]]", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        fun: Py<PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Sequence[typing.Callable[[numpy.typing.NDArray[numpy.float64], builtins.bool], builtins.float | numpy.typing.NDArray[numpy.float64]] | tuple[typing.Callable[[numpy.typing.NDArray[numpy.float64]], builtins.float], typing.Callable[[numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]]] | builtins.dict[builtins.str, typing.Callable[[numpy.typing.NDArray[numpy.float64]], builtins.float | numpy.typing.NDArray[numpy.float64]]]] | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        fcstrs: Option<Vec<Py<PyAny>>>,
        #[gen_stub(override_type(type_repr = "typing.Sequence[CstrSpec | builtins.dict[builtins.str, typing.Any]] | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        fcstr_specs: Option<Vec<CstrSpec>>,
        max_iters: usize,
        #[gen_stub(override_type(type_repr = "RunInfo | builtins.dict[builtins.str, typing.Any] | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        run_info: Option<Py<PyAny>>,
        outdir: Option<String>,
        warm_start: bool,
        #[gen_stub(override_type(type_repr = "builtins.bool | builtins.int | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        hot_start: Option<Py<PyAny>>,
        seed: Option<u64>,
        timeout: Option<f64>,
        #[gen_stub(override_type(type_repr = "Verbose | builtins.int | None", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        verbose: Option<Py<PyAny>>,
        stop_on_error: bool,
    ) -> PyResult<EgorOptim> {
        init_logger(
            py,
            verbose.or_else(|| self.verbose.as_ref().map(|v| v.clone_ref(py))),
        );
        let seed = seed.or(self.seed);

        let hot_start = normalize_hot_start(py, hot_start)?;

        // Errors raised within user callbacks which have to abort the optimization
        let callback_error = CallbackError::default();
        let callback_error = &callback_error;
        install_panic_hook();

        let ny = 1 + self
            .cstr_specs
            .as_ref()
            .map_or(self.n_cstr, |specs| specs.len());
        let obj = |x: &ArrayView2<f64>| -> std::result::Result<Array2<f64>, String> {
            Python::attach(|py| {
                let args = (x.to_owned().into_pyarray(py),);
                let res = fun.bind(py).call1(args);
                match res {
                    // Python exception in objective function is handled by the optimizer
                    // wrt stop_on_error and failsafe_strategy options
                    Err(e) => {
                        log::error!("Error during objective function evaluation: {:?}", e);
                        Err(e.to_string())
                    }
                    // Wrong returned value is a usage error which aborts the optimization
                    Ok(res) => match extract_obj_value(&res, x.nrows(), ny) {
                        Ok(y) => Ok(y),
                        Err(e) => callback_error.abort(e),
                    },
                }
            })
        };

        let fcstrs = fcstrs
            .unwrap_or_default()
            .iter()
            .map(|cstr| FcstrFn::parse(cstr.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        let fcstr_specs = fcstr_specs.unwrap_or_default();
        let n_fcstr = fcstrs.len();
        if !fcstr_specs.is_empty() && fcstr_specs.len() != n_fcstr {
            return Err(PyValueError::new_err(format!(
                "fcstr_specs length ({}) must match fcstrs length ({})",
                fcstr_specs.len(),
                n_fcstr
            )));
        }

        let cstr_tol = self.internal_cstr_tol(&fcstr_specs, n_fcstr);
        let fcstr_specs = fcstr_specs
            .into_iter()
            .map(|spec| spec.inner)
            .collect::<Vec<_>>();

        let fcstrs = fcstrs
            .iter()
            .map(|cstr| {
                |x: &[f64], g: Option<&mut [f64]>, _u: &mut InfillObjData<f64>| -> f64 {
                    Python::attach(|py| {
                        if let Some(g) = g
                            && let Err(e) = cstr
                                .call(py, x, true)
                                .and_then(|res| extract_cstr_gradient(&res, g))
                        {
                            callback_error.abort(e)
                        }
                        cstr.call(py, x, false)
                            .and_then(|res| extract_cstr_value(&res))
                            .unwrap_or_else(|e| callback_error.abort(e))
                    })
                }
            })
            .collect::<Vec<_>>();

        let factory = egobox_ego::EgorFactory::optimize(obj);
        let factory = if fcstr_specs.is_empty() {
            factory.subject_to(fcstrs)
        } else {
            factory.subject_to_with_specs(fcstrs, fcstr_specs)
        };

        let mixintegor = factory
            .configure(|config| {
                self.apply_config(
                    config,
                    Some(max_iters),
                    cstr_tol,
                    self.doe.as_ref(),
                    outdir,
                    warm_start,
                    hot_start,
                    seed,
                    timeout,
                    stop_on_error,
                )
            })
            .min_within_mixint_space(&self.xtypes)
            .map_err(ego_err)?;

        let py_run_info = if let Some(ri) = run_info {
            parse_run_info(py, ri)?
        } else {
            RunInfo::default()
        };

        let mixintegor = mixintegor.run_info(egobox_ego::RunInfo {
            fname: py_run_info.fname.clone(),
            num: py_run_info.num,
        });

        let res = py
            .detach(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mixintegor.run())));
        let res = match res {
            Ok(res) => res.map_err(ego_err)?,
            // The optimizer was aborted: raise the recorded callback error if any
            // (the panic payload may have been rewrapped when crossing threads)
            Err(payload) => match callback_error.take() {
                Some(err) => return Err(err),
                None => std::panic::resume_unwind(payload),
            },
        };

        let status = RunStatus {
            info: py_run_info,
            exit: (res.state.termination_status).into(),
            init_doe_size: res.state.doe.doe_size,
            best_iter: res.state.last_best_iter as usize,
            total_iters: res.state.iter as usize,
            elapsed_time: res
                .state
                .time
                .map(|d| d.as_millis() as f64 / 1000.0)
                .unwrap_or(0.0),
        };

        let x_opt = res.x_opt.into_pyarray(py).to_owned();
        let y_opt = res.y_opt.into_pyarray(py).to_owned();
        let x_doe = res.x_doe.into_pyarray(py).to_owned();
        let y_doe = res.y_doe.into_pyarray(py).to_owned();
        let result: Py<OptimResult> = Bound::new(
            py,
            OptimResult {
                x_opt: x_opt.into(),
                y_opt: y_opt.into(),
                x_doe: x_doe.into(),
                y_doe: y_doe.into(),
            },
        )?
        .into();

        Ok(EgorOptim { result, status })
    }

    /// This function gives the next best location where to evaluate the function
    /// under optimization wrt to previous evaluations.
    /// The function returns several points when multi point qEI strategy is used.
    ///
    /// Parameters
    /// ----------
    /// x_doe : array[ns, nx]
    ///     ns samples where function has been evaluated
    /// y_doe : array[ns, 1 + n_cstr]
    ///     ns values of objective and constraints
    /// seed : int >= 0, optional
    ///     Random generator seed to allow computation reproducibility.
    ///     When None, the seed given to the constructor (if any) is used.
    ///
    /// Returns
    /// -------
    /// array[batch, nx]
    ///     suggested locations where to evaluate objective and constraints
    ///     where batch is the qEI batch size (qei_config.batch, 1 by default)
    ///
    #[pyo3(signature = (x_doe, y_doe, seed = None))]
    fn suggest(
        &self,
        py: Python,
        x_doe: PyReadonlyArray2<f64>,
        y_doe: PyReadonlyArray2<f64>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        init_logger(py, self.verbose.as_ref().map(|v| v.clone_ref(py)));
        let seed = seed.or(self.seed);
        let x_doe = x_doe.as_array();
        let y_doe = y_doe.as_array();
        check_doe(Some(&x_doe), &y_doe)?;
        if x_doe.ncols() != self.xtypes.len() {
            return Err(PyValueError::new_err(format!(
                "x_doe should be of shape (ns, {}), got {:?}",
                self.xtypes.len(),
                x_doe.shape()
            )));
        }
        let doe = concatenate(Axis(1), &[x_doe.view(), y_doe.view()])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mixintegor = egobox_ego::EgorServiceBuilder::optimize()
            .configure(|config| {
                self.apply_config(
                    config,                         // config
                    Some(1),                        // max_iters
                    self.internal_cstr_tol(&[], 0), // cstr_tol
                    Some(&doe),                     // doe
                    None,                           // outdir
                    false,                          // warm_start
                    None,                           // hot_start
                    seed,                           // seed
                    None,                           // timeout
                    true,                           // stop_on_error
                )
            })
            .min_within_mixint_space(&self.xtypes)
            .map_err(ego_err)?;

        let x_suggested = py.detach(|| mixintegor.suggest(&x_doe, &y_doe));
        Ok(x_suggested.to_pyarray(py).into())
    }

    /// This function gives the best evaluation index given the outputs
    /// of the function (objective wrt constraints) under minimization.
    /// Caveat: This function does not take into account function constraints values
    ///
    /// Parameters
    /// ----------
    /// y_doe : array[ns, 1 + n_cstr]
    ///     ns values of objective and constraints as returned by `fun` (see `minimize`),
    ///     constraints are interpreted with `cstr_specs` and their tolerances if given
    ///
    /// Returns
    /// -------
    /// int
    ///     index in y_doe of the best evaluation
    ///
    #[pyo3(signature = (y_doe))]
    fn best_index(&self, y_doe: PyReadonlyArray2<f64>) -> PyResult<usize> {
        let y_doe = y_doe.as_array();
        check_doe(None, &y_doe)?;
        self.best_row_index(&y_doe)
    }

    /// This function gives the best result given inputs and outputs
    /// of the function (objective wrt constraints) under minimization.
    /// Caveat: This function does not take into account function constraints values
    ///
    /// Parameters
    /// ----------
    /// x_doe : array[ns, nx]
    ///     ns samples where function has been evaluated
    /// y_doe : array[ns, 1 + n_cstr]
    ///     ns values of objective and constraints as returned by `fun` (see `minimize`),
    ///     constraints are interpreted with `cstr_specs` and their tolerances if given
    ///
    /// Returns
    /// -------
    /// OptimResult
    ///     * x_opt (array[nx]): x value where fun is at its minimum subject to constraints
    ///     * y_opt (array[ny]): fun(x_opt) where ny = 1 + n_cstr
    ///     * x_doe (array[ns, nx]): the given x_doe
    ///     * y_doe (array[ns, ny]): the given y_doe
    ///
    #[pyo3(signature = (x_doe, y_doe))]
    fn best_result(
        &self,
        py: Python,
        x_doe: PyReadonlyArray2<f64>,
        y_doe: PyReadonlyArray2<f64>,
    ) -> PyResult<OptimResult> {
        let x_doe = x_doe.as_array();
        let y_doe = y_doe.as_array();
        check_doe(Some(&x_doe), &y_doe)?;
        let idx = self.best_row_index(&y_doe)?;
        let x_opt = x_doe.row(idx).to_pyarray(py).into();
        let y_opt = y_doe.row(idx).to_pyarray(py).into();
        let x_doe = x_doe.to_pyarray(py).into();
        let y_doe = y_doe.to_pyarray(py).into();
        Ok(OptimResult {
            x_opt,
            y_opt,
            x_doe,
            y_doe,
        })
    }

    /// Deprecated since 0.38.0, use `best_index` instead.
    #[pyo3(signature = (y_doe))]
    fn get_result_index(&self, py: Python, y_doe: PyReadonlyArray2<f64>) -> PyResult<usize> {
        warn_deprecated(py, "Egor.get_result_index", "Egor.best_index")?;
        self.best_index(y_doe)
    }

    /// Deprecated since 0.38.0, use `best_result` instead.
    #[pyo3(signature = (x_doe, y_doe))]
    fn get_result(
        &self,
        py: Python,
        x_doe: PyReadonlyArray2<f64>,
        y_doe: PyReadonlyArray2<f64>,
    ) -> PyResult<OptimResult> {
        warn_deprecated(py, "Egor.get_result", "Egor.best_result")?;
        self.best_result(py, x_doe, y_doe)
    }
}

/// Build the initial DOE given either with the deprecated `doe` or with `x_doe` and optional `y_doe`
fn initial_doe(
    py: Python,
    nx: usize,
    doe: Option<PyReadonlyArray2<f64>>,
    x_doe: Option<PyReadonlyArray2<f64>>,
    y_doe: Option<PyReadonlyArray2<f64>>,
) -> PyResult<Option<Array2<f64>>> {
    if let Some(doe) = doe {
        if x_doe.is_some() || y_doe.is_some() {
            return Err(PyTypeError::new_err(
                "`doe` and `x_doe`/`y_doe` cannot be both given, `doe` is deprecated, use `x_doe`/`y_doe` only",
            ));
        }
        warn_deprecated(py, "doe", "x_doe` and `y_doe")?;
        return Ok(Some(doe.to_owned_array()));
    }
    let Some(x_doe) = x_doe else {
        if y_doe.is_some() {
            return Err(PyValueError::new_err("y_doe requires x_doe to be given"));
        }
        return Ok(None);
    };
    let x_doe = x_doe.as_array();
    if x_doe.ncols() != nx {
        return Err(PyValueError::new_err(format!(
            "x_doe should be of shape (ns, {nx}), got {:?}",
            x_doe.shape()
        )));
    }
    match y_doe {
        Some(y_doe) => {
            let y_doe = y_doe.as_array();
            check_doe(Some(&x_doe), &y_doe)?;
            Ok(Some(
                concatenate(Axis(1), &[x_doe, y_doe])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ))
        }
        None => Ok(Some(x_doe.to_owned())),
    }
}

/// Check (x_doe, y_doe) are non empty with the same number of rows
fn check_doe(x_doe: Option<&ArrayView2<f64>>, y_doe: &ArrayView2<f64>) -> PyResult<()> {
    if y_doe.nrows() == 0 || y_doe.ncols() == 0 {
        return Err(PyValueError::new_err(format!(
            "y_doe should be a non empty array of shape (ns, 1 + n_cstr), got {:?}",
            y_doe.shape()
        )));
    }
    if let Some(x_doe) = x_doe
        && x_doe.nrows() != y_doe.nrows()
    {
        return Err(PyValueError::new_err(format!(
            "x_doe and y_doe should have the same number of rows, got {} and {}",
            x_doe.nrows(),
            y_doe.nrows()
        )));
    }
    Ok(())
}

impl Egor {
    fn n_clusters(&self) -> NbClusters {
        match self.gp_config.n_clusters.cmp(&0) {
            Ordering::Greater => NbClusters::fixed(self.gp_config.n_clusters as usize),
            Ordering::Equal => NbClusters::auto(),
            Ordering::Less => NbClusters::automax(-self.gp_config.n_clusters as usize),
        }
    }

    fn infill_strategy(&self) -> egobox_ego::InfillStrategy {
        match self.infill_strategy {
            InfillStrategy::Ei => egobox_ego::InfillStrategy::EI,
            InfillStrategy::Wb2 => egobox_ego::InfillStrategy::WB2,
            InfillStrategy::Wb2s => egobox_ego::InfillStrategy::WB2S,
            InfillStrategy::LogEi => egobox_ego::InfillStrategy::LogEI,
        }
    }

    fn feasible_infill_strategy(&self) -> egobox_ego::FeasibleInfillStrategy {
        match self.feasible_infill_strategy {
            FeasibleInfillStrategy::None => egobox_ego::FeasibleInfillStrategy::None,
            FeasibleInfillStrategy::EfiP => egobox_ego::FeasibleInfillStrategy::EfiP,
            FeasibleInfillStrategy::EfiFe => egobox_ego::FeasibleInfillStrategy::EfiFe(0.3),
        }
    }

    fn cstr_strategy(&self) -> egobox_ego::ConstraintStrategy {
        match self.cstr_strategy {
            ConstraintStrategy::Mc => egobox_ego::ConstraintStrategy::MeanConstraint,
            ConstraintStrategy::Utb => egobox_ego::ConstraintStrategy::UpperTrustBound,
        }
    }

    fn qei_strategy(&self) -> egobox_ego::QEiStrategy {
        match self.qei_config.strategy {
            QEiStrategy::Kb => egobox_ego::QEiStrategy::KrigingBeliever,
            QEiStrategy::Kblb => egobox_ego::QEiStrategy::KrigingBelieverLowerBound,
            QEiStrategy::Kbub => egobox_ego::QEiStrategy::KrigingBelieverUpperBound,
            QEiStrategy::Clmin => egobox_ego::QEiStrategy::ConstantLiarMinimum,
        }
    }

    fn infill_optimizer(&self) -> egobox_ego::InfillOptimizer {
        match self.infill_optimizer {
            InfillOptimizer::Cobyla => egobox_ego::InfillOptimizer::Cobyla,
            InfillOptimizer::Slsqp => egobox_ego::InfillOptimizer::Slsqp,
        }
    }

    fn failsafe_strategy(&self) -> egobox_ego::FailsafeStrategy {
        match self.failsafe_strategy {
            FailsafeStrategy::Rejection => egobox_ego::FailsafeStrategy::Rejection,
            FailsafeStrategy::Imputation => egobox_ego::FailsafeStrategy::Imputation,
            FailsafeStrategy::Viability => egobox_ego::FailsafeStrategy::Viability,
        }
    }

    /// Index of the best row of `y_doe` [obj, cstr_1, ... cstr_n_cstr] holding raw constraint values,
    /// interpreted with `cstr_specs` (if any) and their tolerances, as the optimizer does.
    fn best_row_index(&self, y_doe: &ArrayView2<f64>) -> PyResult<usize> {
        let ny = 1 + self.n_cstr;
        if y_doe.ncols() != ny {
            return Err(PyValueError::new_err(format!(
                "y_doe should be of shape (ns, 1 + n_cstr) = (ns, {ny}), got {:?}",
                y_doe.shape()
            )));
        }
        let y_doe = match self.cstr_specs.as_ref() {
            Some(specs) => {
                let specs = specs.iter().map(|s| s.inner.clone()).collect::<Vec<_>>();
                egobox_ego::transform_constraints(&y_doe.to_owned(), &specs)
            }
            None => y_doe.to_owned(),
        };
        let cstr_tol = self.internal_cstr_tol(&[], 0).unwrap_or_else(|| {
            Array1::from_elem(y_doe.ncols() - 1, egobox_ego::DEFAULT_CSTR_TOL)
        });
        let c_doe = Array2::zeros((y_doe.nrows(), 0));
        Ok(find_best_result_index(&y_doe, &c_doe, &cstr_tol))
    }

    /// Tolerances of all internal constraints (surrogate constraints then function constraints,
    /// after specs expansion) when given by the user either with `cstr_tol` or with specs `tol`
    /// (which take precedence), None otherwise to let the optimizer use its defaults.
    /// `fcstr_specs` is either empty or one spec per function constraint.
    fn internal_cstr_tol(&self, fcstr_specs: &[CstrSpec], n_fcstr: usize) -> Option<Array1<f64>> {
        let cstr_specs = self.cstr_specs.as_deref().unwrap_or(&[]);
        let has_spec_tol = cstr_specs
            .iter()
            .chain(fcstr_specs)
            .any(|s| s.tol.is_some());
        if self.cstr_tol.is_none() && !has_spec_tol {
            return None;
        }
        // (number of internal constraints, spec tolerance) for each user constraint
        let expansion = |specs: &[CstrSpec], n: usize| -> Vec<(usize, Option<f64>)> {
            if specs.is_empty() {
                vec![(1, None); n]
            } else {
                specs
                    .iter()
                    .map(|s| (s.inner.n_internal(), s.tol))
                    .collect()
            }
        };
        let mut tol = self.cstr_tol.clone().unwrap_or_default();
        let mut i = 0;
        for (n, spec_tol) in expansion(cstr_specs, self.n_cstr)
            .into_iter()
            .chain(expansion(fcstr_specs, n_fcstr))
        {
            for _ in 0..n {
                if i == tol.len() {
                    tol.push(egobox_ego::DEFAULT_CSTR_TOL);
                }
                if let Some(spec_tol) = spec_tol {
                    tol[i] = spec_tol;
                }
                i += 1;
            }
        }
        Some(Array1::from_vec(tol))
    }

    fn recombination(&self) -> egobox_moe::Recombination<f64> {
        match self.gp_config.recombination {
            Recombination::Hard => egobox_moe::Recombination::Hard,
            Recombination::Smooth => egobox_moe::Recombination::Smooth(Some(1.0)),
        }
    }

    fn theta_tuning(&self) -> ThetaTuning<f64> {
        let mut theta_tuning = ThetaTuning::<f64>::default();
        if let Some(init) = self.gp_config.theta_init.as_ref() {
            theta_tuning = ThetaTuning::Full {
                init: Array1::from_vec(init.to_vec()),
                bounds: array![ThetaTuning::<f64>::DEFAULT_BOUNDS],
            }
        }
        if let Some(bounds) = self.gp_config.theta_bounds.as_ref() {
            theta_tuning = ThetaTuning::Full {
                init: theta_tuning.init().to_owned(),
                bounds: bounds.iter().map(|v| (v[0], v[1])).collect(),
            }
        }
        theta_tuning
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_config(
        &self,
        config: egobox_ego::EgorConfig,
        max_iters: Option<usize>,
        cstr_tol: Option<Array1<f64>>,
        doe: Option<&Array2<f64>>,
        outdir: Option<String>,
        warm_start: bool,
        hot_start: Option<u64>,
        seed: Option<u64>,
        timeout: Option<f64>,
        stop_on_error: bool,
    ) -> egobox_ego::EgorConfig {
        let infill_strategy = self.infill_strategy();
        let feasible_infill_strategy = self.feasible_infill_strategy();
        let cstr_strategy = self.cstr_strategy();
        let qei_strategy = self.qei_strategy();
        let infill_optimizer = self.infill_optimizer();
        let failsafe_strategy = self.failsafe_strategy();
        let coego_status = if self.coego_n_coop == 0 {
            CoegoStatus::Disabled
        } else {
            CoegoStatus::Enabled(self.coego_n_coop)
        };

        let mut config = config
            .n_cstr(self.n_cstr)
            .max_iters(max_iters.unwrap_or(1))
            .n_start(self.infill_n_start)
            .n_doe(self.n_doe);

        // Only set cstr_tol explicitly when user provided it.
        // Otherwise let Rust infer the correct total length, including
        // expanded constraints and function constraints.
        if let Some(cstr_tol) = cstr_tol {
            config = config.cstr_tol(cstr_tol);
        }

        if let Some(ref cstr_specs) = self.cstr_specs {
            config = config.cstr_specs(cstr_specs.iter().map(|s| s.inner.clone()).collect());
        }

        let mut config = config
            .configure_gp(|gp| {
                let regr = RegressionSpec(self.gp_config.regr_spec);
                let corr = CorrelationSpec(self.gp_config.corr_spec);
                gp.regression_spec(egobox_moe::RegressionSpec::from_bits(regr.0).unwrap())
                    .correlation_spec(egobox_moe::CorrelationSpec::from_bits(corr.0).unwrap())
                    .kpls_dim(self.gp_config.kpls_dim)
                    .n_clusters(self.n_clusters())
                    .recombination(self.recombination())
                    .theta_tuning(self.theta_tuning())
                    .n_start(self.gp_config.theta_n_start)
                    .max_eval(self.gp_config.theta_max_eval)
            })
            .infill_strategy(infill_strategy)
            .feasible_infill_strategy(feasible_infill_strategy)
            .cstr_infill(self.cstr_infill)
            .cstr_strategy(cstr_strategy)
            .configure_qei(|qei_config| {
                qei_config
                    .batch(self.qei_config.batch)
                    .strategy(qei_strategy)
                    .optmod(self.qei_config.optim_every)
            })
            .infill_optimizer(infill_optimizer)
            .coego(coego_status)
            .stop_on_error(stop_on_error)
            .warm_start(warm_start)
            .hot_start(hot_start.into())
            .failsafe_strategy(failsafe_strategy);

        if let Some(target) = self.target {
            config = config.target(target);
        }

        if let Some(timeout) = timeout {
            config = config.timeout(timeout);
        }

        if let Some(trego) = self.trego.as_ref() {
            let strategy: egobox_ego::TregoStrategy = trego.clone().into();
            config = config.iteration_strategy(Box::new(strategy))
        }

        if let Some(doe) = doe {
            config = config.doe(doe);
        };

        if let Some(outdir) = outdir {
            config = config.outdir(outdir.to_owned());
        };
        if let Some(seed) = seed {
            config = config.seed(seed);
        };
        config
    }
}

/// A function constraint given to `minimize`, either as a single callable `g(x, return_grad)`
/// or as a value callable `g(x)` and a gradient callable `grad_g(x)`
enum FcstrFn {
    WithGradFlag(Py<PyAny>),
    FunJac { fun: Py<PyAny>, jac: Py<PyAny> },
}

impl FcstrFn {
    /// Parse `g(x, return_grad)`, `(g, grad_g)` or `{"fun": g, "jac": grad_g}`
    fn parse(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let callable = |f: Bound<'_, PyAny>, what: &str| -> PyResult<Py<PyAny>> {
            if f.is_callable() {
                Ok(f.unbind())
            } else {
                Err(PyTypeError::new_err(format!(
                    "function constraint {what} should be callable, got {}",
                    f.get_type()
                )))
            }
        };
        if let Ok(dict) = value.cast::<PyDict>() {
            if dict.contains("type")? {
                return Err(PyValueError::new_err(
                    "function constraint dict does not take a \"type\" key: egobox constraints \
                     are g(x) <= 0 (unlike scipy \"ineq\" constraints g(x) >= 0), \
                     use fcstr_specs (e.g. CstrSpec.geq(0.0)) to give other bounds",
                ));
            }
            for key in dict.keys() {
                let key: String = key.extract()?;
                if key != "fun" && key != "jac" {
                    return Err(PyValueError::new_err(format!(
                        "unknown function constraint dict key \"{key}\", expected \"fun\" and \"jac\""
                    )));
                }
            }
            let get = |key: &str| -> PyResult<Py<PyAny>> {
                let f = dict.get_item(key)?.ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "function constraint dict requires a \"{key}\" key"
                    ))
                })?;
                callable(f, &format!("\"{key}\""))
            };
            return Ok(FcstrFn::FunJac {
                fun: get("fun")?,
                jac: get("jac")?,
            });
        }
        if let Ok(tuple) = value.cast::<PyTuple>() {
            if tuple.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "function constraint tuple should be (g, grad_g), got {} items",
                    tuple.len()
                )));
            }
            return Ok(FcstrFn::FunJac {
                fun: callable(tuple.get_item(0)?, "g")?,
                jac: callable(tuple.get_item(1)?, "gradient")?,
            });
        }
        if value.is_callable() {
            return Ok(FcstrFn::WithGradFlag(value.clone().unbind()));
        }
        Err(PyTypeError::new_err(format!(
            "function constraint should be a callable g(x, return_grad), a tuple (g, grad_g) \
             or a dict {{\"fun\": g, \"jac\": grad_g}}, got {}",
            value.get_type()
        )))
    }

    /// Call the constraint value (`grad` false) or gradient (`grad` true) function at x
    fn call<'py>(&self, py: Python<'py>, x: &[f64], grad: bool) -> PyResult<Bound<'py, PyAny>> {
        let x = Array1::from(x.to_vec()).into_pyarray(py);
        match self {
            FcstrFn::WithGradFlag(g) => g.bind(py).call1((x, grad)),
            FcstrFn::FunJac { fun, jac } => {
                let f = if grad { jac } else { fun };
                f.bind(py).call1((x,))
            }
        }
    }
}

/// Extract the value returned by the objective function as an (n, ny) float array
fn extract_obj_value(res: &Bound<'_, PyAny>, n: usize, ny: usize) -> PyResult<Array2<f64>> {
    let arr = res.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "objective function should return a 2D float64 numpy array of shape ({n}, {ny}), got {}",
            describe(res)
        ))
    })?;
    let arr = arr.as_array();
    if arr.dim() != (n, ny) {
        return Err(PyValueError::new_err(format!(
            "objective function should return an array of shape ({n}, {ny}) \
             (objective + {} constraints), got {:?}",
            ny - 1,
            arr.shape()
        )));
    }
    Ok(arr.to_owned())
}

/// Describe a Python value type (with shape and dtype for numpy arrays) for error messages
fn describe(res: &Bound<'_, PyAny>) -> String {
    match (res.getattr("shape"), res.getattr("dtype")) {
        (Ok(shape), Ok(dtype)) => format!("{} of shape {shape} and dtype {dtype}", res.get_type()),
        _ => res.get_type().to_string(),
    }
}

/// Extract the value of a function constraint as a scalar.
/// Accepts a Python float or any numpy array holding a single element
/// (e.g. shape (1,) or (1, 1)) as NumPy >= 2.4 no longer converts those implicitly.
fn extract_cstr_value(res: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(v) = res.extract::<f64>() {
        return Ok(v);
    }
    let arr = res.extract::<PyReadonlyArrayDyn<f64>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "function constraint should return a float, got {}",
            res.get_type()
        ))
    })?;
    let arr = arr.as_array();
    if arr.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "function constraint should return a single value, got array of shape {:?}",
            arr.shape()
        )));
    }
    Ok(*arr.iter().next().unwrap())
}

/// Copy the gradient of a function constraint into `g`.
/// Accepts any numpy array holding `g.len()` elements (e.g. shape (nx,) or (1, nx)).
fn extract_cstr_gradient(res: &Bound<'_, PyAny>, g: &mut [f64]) -> PyResult<()> {
    let arr = res.extract::<PyReadonlyArrayDyn<f64>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "function constraint gradient should be a float numpy array, got {}",
            res.get_type()
        ))
    })?;
    let arr = arr.as_array();
    if arr.len() != g.len() {
        return Err(PyValueError::new_err(format!(
            "function constraint gradient should have {} elements, got array of shape {:?}",
            g.len(),
            arr.shape()
        )));
    }
    g.iter_mut().zip(arr.iter()).for_each(|(gi, ai)| *gi = *ai);
    Ok(())
}

fn normalize_hot_start(py: Python, hot_start: Option<Py<PyAny>>) -> PyResult<Option<u64>> {
    match hot_start {
        Some(hot_start) => {
            let hot_start = hot_start.bind(py);
            if hot_start.is_none() {
                Ok(None)
            } else if hot_start.is_instance_of::<PyBool>() {
                Ok(hot_start.extract::<bool>()?.then_some(0))
            } else if let Ok(ext_iters) = hot_start.extract::<u64>() {
                Ok(Some(ext_iters))
            } else {
                Err(PyTypeError::new_err(
                    "hot_start must be a bool, a non-negative integer, or None",
                ))
            }
        }
        None => Ok(None),
    }
}

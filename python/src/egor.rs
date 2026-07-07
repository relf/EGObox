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

use crate::domain::*;
use crate::gp_config::*;
use crate::gp_mix::Gpx;
use crate::logging::init_logger;
use crate::qei_config::*;
use crate::trego_config::{TregoConfig, TregoConfigSpec};
use crate::types::*;

use egobox_ego::{CoegoStatus, InfillObjData, Result, find_best_result_index};
use egobox_gp::ThetaTuning;
use egobox_moe::NbClusters;
use ndarray::{Array1, Array2, ArrayView2, Axis, array, concatenate};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::cmp::Ordering;

fn parse_trego_config(py: Python, value: Py<PyAny>) -> PyResult<TregoConfigSpec> {
    if let Ok(spec) = value.extract(py) {
        return Ok(spec);
    }

    let dict = value.bind(py).cast::<pyo3::types::PyDict>()?;
    let mut cfg = TregoConfig::default();

    for key_any in dict.keys().iter() {
        let key = key_any.extract::<String>()?;
        match key.as_str() {
            "n_gl_steps" => cfg.n_gl_steps = dict.get_item("n_gl_steps")?.unwrap().extract()?,
            "d" => cfg.d = dict.get_item("d")?.unwrap().extract()?,
            "alpha" => cfg.alpha = dict.get_item("alpha")?.unwrap().extract()?,
            "beta" => cfg.beta = dict.get_item("beta")?.unwrap().extract()?,
            "sigma0" => cfg.sigma0 = dict.get_item("sigma0")?.unwrap().extract()?,
            _ => return Err(PyValueError::new_err(format!("unknown trego key '{key}'"))),
        }
    }

    Ok(TregoConfigSpec::Custom(cfg))
}

fn parse_run_info(py: Python, value: Py<PyAny>) -> PyResult<RunInfo> {
    if let Ok(info) = value.extract(py) {
        return Ok(info);
    }

    let dict = value.bind(py).cast::<pyo3::types::PyDict>()?;
    let mut info = RunInfo::new("fobj".to_string(), 1);

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

/// Optimizer constructor
///
/// # Parameters
///
///     xspecs (list(XSpec)) where XSpec(xtype=FLOAT|INT|ORD|ENUM, xlimits=[<f(xtype)>] or tags=[strings]):
///         Specifications of the nx components of the input x (eg. len(xspecs) == nx)
///         Depending on the x type we get the following for xlimits:
///         * when FLOAT: xlimits is [float lower_bound, float upper_bound],
///         * when INT: xlimits is [int lower_bound, int upper_bound],
///         * when ORD: xlimits is [float_1, float_2, ..., float_n],
///         * when ENUM: xlimits is just the int size of the enumeration otherwise a list of tags is specified
///           (eg xlimits=[3] or tags=["red", "green", "blue"], tags are there for documention purpose but
///            tags specific values themselves are not used only indices in the enum are used hence
///            we can just specify the size of the enum, xlimits=[3]),
///
///     gp_config (GpConfig):
///        GP configuration used by the optimizer, see GpConfig for details.
///
///     n_cstr (int):
///         the number of constraints which will be approximated by surrogates (see `fun` argument)
///
///     cstr_tol (list(n_cstr + n_fcstr,)):
///         List of tolerances for constraints to be satisfied (cstr < tol),
///         list size should be equal to n_cstr + n_fctrs where n_cstr is the `n_cstr` argument
///         and `n_fcstr` the number of constraints passed as functions.
///         When None, tolerances default to DEFAULT_CSTR_TOL=1e-4.
///
///     cstr_specs (list(n_cstr,) or None):
///         Optional list of CstrSpec objects describing how each surrogate-modeled
///         constraint (returned by `fun`) should be interpreted.
///         This allows users to define bounds directly instead of manually rewriting
///         constraints in `c <= 0` form:
///           * CstrSpec.leq(bound): c <= bound
///           * CstrSpec.geq(bound): c >= bound
///           * CstrSpec.eq(value): c == value (expands to two internal constraints)
///           * CstrSpec.btw(lower, upper): lower <= c <= upper
///             (expands to two internal constraints)
///
///         When set, `n_cstr` is inferred from `len(cstr_specs)` (legacy `n_cstr`
///         value is ignored if set to zero, or must match otherwise).
///         If `cstr_tol` is explicitly provided, its length must match the total
///         number of internal constraints after expansion.
///
///     n_start (int > 0):
///         Number of runs of infill strategy optimizations (best result taken)
///
///     n_doe (int >= 0):
///         Number of samples of initial LHS sampling (used when DOE not provided by the user).
///         When 0 a number of points is computed automatically regarding the number of input variables
///         of the function under optimization.
///
///     doe (array[ns, nt]):
///         Initial DOE containing ns samples:
///             either nt = nx then only x are specified and ns evals are done to get y doe values,
///             or nt = nx + ny then x = doe[:, :nx] and y = doe[:, nx:] are specified
///
///     infill_strategy (InfillStrategy enum):
///         Infill criteria to decide best next promising point.
///         Can be either InfillStrategy.LOG_EI, InfillStrategy.EI, InfillStrategy.WB2, InfillStrategy.WB2S
///
///     feasible_infill_strategy (FeasibleInfillStrategy enum):
///         Strategy to handle feasibility information in the infill criterion.
///         Can be either FeasibleInfillStrategy.NONE, FeasibleInfillStrategy.EFI_P, or FeasibleInfillStrategy.EFI_FE
///
///     cstr_infill (bool):
///         Activate constrained infill criterion where the product of probability of feasibility of constraints
///         used as a factor of the infill criterion specified via infill_strategy
///         
///     cstr_strategy (ConstraintStrategy enum):
///         Constraint management either use the mean value or upper bound
///         Can be either ConstraintStrategy.MeanValue or ConstraintStrategy.UpperTrustedBound.
///
///     infill_optimizer (InfillOptimizer enum):
///         Internal optimizer used to optimize infill criteria.
///         Can be either InfillOptimizer.COBYLA or InfillOptimizer.SLSQP
///
///     qei_config (QEiConfig):
///         Configuration for parallel (qEI) evaluation also known as batch or multipoint evaluation.
///         q points are selected at each iteration of the EGO algorithm.
///         See QEiConfig for details.
///
///     trego (TregoConfig, bool or None):
///         TREGO configuration to activate TREGO strategy for global optimization.
///         When True activate TREGO with default configuration.
///         To activate TREGO with custom configuration see TregoConfig for details.
///         When None or False TREGO is not used.
///
///     coego_n_coop (int >= 0):
///         Number of cooperative components groups which will be used by the CoEGO algorithm.
///         Better to have n_coop a divider of nx or if not with a remainder as large as possible.  
///         The CoEGO algorithm is used to tackle high-dimensional problems turning it in a set of
///         partial optimizations using only nx / n_coop components at a time.
///         The default value is 0 meaning that the CoEGO algorithm is not used.
///   
///     target (float):
///         Known optimum used as stopping criterion.
///
///     failsafe_strategy (FailsafeStrategy enum):
///         Strategy to handle objective computation failure at a given x point.
///         A failure is detected when the objective function returns NaN value(s).
///         Can be either FailsafeStrategy.REJECTION, FailsafeStrategy.IMPUTATION, or FailsafeStrategy.VIABILITY
///         Rejection simply ignores the failed point whereas Imputation
///         uses the objective surrogate prediction to fill the missing value.
///         In the third case Viability, a surrogate is used to model the failure region
///         which is used as a constraint and drive the optimization toward the viable region.
///
///     seed (int >= 0 or None):
///         Deprecated: use seed argument in minimize() or suggest() instead.
///
///     outdir (String or None):
///         Deprecated: use outdir argument in minimize() instead.
///
///     warm_start (bool):
///         Deprecated: use warm_start argument in minimize() instead.
///
///     hot_start (bool, int >= 0 or None):
///         Deprecated: use hot_start argument in minimize() instead.
///
///     verbose (int, Verbose enum, or None):
///         Deprecated: use verbose argument in minimize() instead.
///
/// # Returns
///
///     Egor object which can be used to optimize a function using the minimize method.
///      
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object)]
pub(crate) struct Egor {
    pub xtypes: Vec<egobox_moe::XType>,
    pub gp_config: GpConfig,
    pub n_cstr: usize,
    pub cstr_tol: Option<Vec<f64>>,
    pub cstr_specs: Option<Vec<egobox_ego::CstrSpec>>,
    pub n_start: usize,
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
    pub target: f64,
    pub failsafe_strategy: FailsafeStrategy,
    // Deprecated fields (kept for backward compatibility)
    pub seed: Option<u64>,
    pub outdir: Option<String>,
    pub warm_start: bool,
    pub hot_start: Option<u64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Egor {
    #[new]
    #[pyo3(signature = (
        xspecs,
        gp_config = GpConfig::default(),
        n_cstr = 0,
        cstr_tol = None,
        cstr_specs = None,
        n_start = 20,
        n_doe = 0,
        doe = None,
        infill_strategy = InfillStrategy::LogEi,
        feasible_infill_strategy = FeasibleInfillStrategy::None,
        cstr_infill = false,
        cstr_strategy = ConstraintStrategy::Mc,
        qei_config = QEiConfig::default(),
        infill_optimizer = InfillOptimizer::Cobyla,
        trego = None,
        coego_n_coop = 0,
        target = f64::MIN,
        failsafe_strategy = FailsafeStrategy::Rejection,
        seed = None,
        outdir = None,
        warm_start = false,
        hot_start = None,
        verbose = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        xspecs: Py<PyAny>,
        gp_config: GpConfig,
        n_cstr: usize,
        cstr_tol: Option<Vec<f64>>,
        cstr_specs: Option<Vec<CstrSpec>>,
        n_start: usize,
        n_doe: usize,
        doe: Option<PyReadonlyArray2<f64>>,
        infill_strategy: InfillStrategy,
        feasible_infill_strategy: FeasibleInfillStrategy,
        cstr_infill: bool,
        cstr_strategy: ConstraintStrategy,
        qei_config: QEiConfig,
        infill_optimizer: InfillOptimizer,
        trego: Option<Py<PyAny>>,
        coego_n_coop: usize,
        target: f64,
        failsafe_strategy: FailsafeStrategy,
        seed: Option<u64>,
        outdir: Option<String>,
        warm_start: bool,
        hot_start: Option<Py<PyAny>>,
        verbose: Option<Py<PyAny>>,
    ) -> Self {
        // Emit deprecation warnings for parameters that moved to minimize()/suggest()
        let warn = |msg: &str| {
            let warnings = py.import("warnings").unwrap();
            let depr = py
                .import("builtins")
                .unwrap()
                .getattr("DeprecationWarning")
                .unwrap();
            warnings.call_method1("warn", (msg, depr)).ok();
        };

        if seed.is_some() {
            warn(
                "Passing 'seed' to Egor() is deprecated. Use 'seed' argument of minimize() or suggest() instead.",
            );
        }
        if outdir.is_some() {
            warn(
                "Passing 'outdir' to Egor() is deprecated. Use 'outdir' argument of minimize() instead.",
            );
        }
        if warm_start {
            warn(
                "Passing 'warm_start' to Egor() is deprecated. Use 'warm_start' argument of minimize() instead.",
            );
        }
        if hot_start.is_some() {
            warn(
                "Passing 'hot_start' to Egor() is deprecated. Use 'hot_start' argument of minimize() instead.",
            );
        }
        if verbose.is_some() {
            init_logger(py, verbose);
            warn(
                "Passing 'verbose' to Egor() is deprecated. Use 'verbose' argument of minimize() instead.",
            );
        }

        let hot_start = normalize_hot_start(py, hot_start).expect("Bad hot_start value");

        let doe = doe.map(|x| x.to_owned_array());
        let xtypes = parse(py, xspecs.clone_ref(py));

        // Parse trego configuration: boolean or custom configuration
        let trego = match trego {
            Some(trego_py) => {
                let trego_typ = parse_trego_config(py, trego_py).expect("Bad TREGO configuration");
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

        Egor {
            xtypes,
            gp_config,
            n_cstr,
            cstr_tol,
            cstr_specs: cstr_specs.map(|specs| specs.into_iter().map(|s| s.inner).collect()),
            n_start,
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
            outdir,
            warm_start,
            hot_start,
        }
    }

    /// This function finds the minimum of a given function "fun"
    ///
    /// # Parameters
    ///
    ///     fun: (array[n, nx] -> array[n, ny])
    ///         the function to be minimized
    ///         fun(x) = [obj(x), cstr_1(x), ... cstr_k(x)] where
    ///            obj is the objective function [n, nx] -> [n, 1]
    ///            cstr_i is the ith constraint function [n, nx] -> [n, 1]
    ///            an k the number of constraints (n_cstr)
    ///            hence ny = 1 (obj) + k (cstrs)
    ///         cstr functions are expected be negative (<=0) at the optimum.
    ///         This constraints will be approximated using surrogates, so
    ///         if constraints are cheap to evaluate better to pass them through run(fcstrs=[...])
    ///
    ///     fcstrs:
    ///         list of constraints functions defined as g(x, return_grad): (ndarray[nx], bool) -> float or ndarray[nx,]
    ///         If the given "return_grad" boolean is "False" the function has to return the constraint float value
    ///         to be made negative by the optimizer (which drives the input array "x").
    ///         Otherwise the function has to return the gradient (ndarray[nx,]) of the constraint function
    ///         wrt the "nx" components of "x".
    ///
    ///     fcstr_specs:
    ///         optional list of CstrSpec objects, one per fcstr, specifying how each function
    ///         constraint should be interpreted.
    ///         Length must be zero (legacy behavior) or equal to len(fcstrs).
    ///         This allows raw constraints not written as c <= 0, for example:
    ///         CstrSpec.leq(b), CstrSpec.geq(b), CstrSpec.eq(v), CstrSpec.btw(lo, hi).
    ///
    ///         Note: CstrSpec.eq and CstrSpec.btw expand to two internal constraints each.
    ///         When cstr_tol is explicitly provided, ensure its size covers all internal
    ///         constraints: surrogate constraints + expanded function constraints.
    ///
    ///     max_iters:
    ///         the iteration budget, number of fun calls is "n_doe + q_batch * max_iters".
    ///
    ///     run_info:
    ///         Optional information about the run to be passed to the optimizer
    ///         It should be an object of type RunInfo with the following attributes:
    ///           - fname (string): name of the function under optimization, used for checkpoint file naming
    ///           - num (int): number of the run, used for checkpoint file naming
    ///
    ///     outdir (String):
    ///         Directory to write optimization history and used as search path for warm start doe
    ///
    ///     warm_start (bool):
    ///         Start by loading initial doe from <outdir> directory
    ///
    ///     hot_start (bool, int >= 0 or None):
    ///         When hot_start>=0 saves optimizer state at each iteration and starts from a previous checkpoint
    ///         for the given hot_start number of iterations beyond the max_iters nb of iterations.
    ///         In an unstable environment were there can be crashes it allows to restart the optimization
    ///         from the last iteration till stopping criterion is reached. Just use hot_start=0 in this case.
    ///         When True, hot_start behaves like hot_start=0 with no iteration extension.
    ///         Checkpoint information is stored in .checkpoint or under outdir if outdir is specified.
    ///
    ///     seed (int >= 0):
    ///         Random generator seed to allow computation reproducibility.
    ///
    ///     timeout (float or None):
    ///         Optional timeout in seconds. The optimization is stopped when the elapsed time
    ///         exceeds this duration. The actual runtime may slightly exceed the specified timeout
    ///         as the check is performed after each iteration.
    ///
    ///     verbose (int, Verbose enum, or None):
    ///         Logging verbosity level for the optimizer.
    ///         Can be either an integer or a Verbose enum value:
    ///         0 or Verbose.ERROR, 1 or Verbose.WARNING, 2 or Verbose.INFO,
    ///         3 or Verbose.DEBUG, 4 (or greater) or Verbose.TRACE.
    ///         Default is None which means Verbose.ERROR level and possible control by
    ///         the EGOBOX_LOG environment variable.
    ///
    /// # Returns
    ///
    ///     optimization result
    ///         x_opt (array[1, nx]): x value where fun is at its minimum subject to constraints
    ///         y_opt (array[1, nx]): fun(x_opt)
    ///
    #[pyo3(signature = (fun, fcstrs=vec![], fcstr_specs=vec![], max_iters = 20, run_info = None, outdir = None, warm_start = false, hot_start = None, seed = None, timeout = None, verbose = None))]
    #[allow(clippy::too_many_arguments)]
    fn minimize(
        &self,
        py: Python,
        fun: Py<PyAny>,
        fcstrs: Vec<Py<PyAny>>,
        fcstr_specs: Vec<CstrSpec>,
        max_iters: usize,
        run_info: Option<Py<PyAny>>,
        outdir: Option<String>,
        warm_start: bool,
        hot_start: Option<Py<PyAny>>,
        seed: Option<u64>,
        timeout: Option<f64>,
        verbose: Option<Py<PyAny>>,
    ) -> PyResult<EgorOptim> {
        init_logger(py, verbose);

        // Merge: minimize() args take precedence over deprecated constructor args
        let seed = seed.or(self.seed);
        let outdir = outdir.or_else(|| self.outdir.clone());
        let warm_start = if warm_start { true } else { self.warm_start };
        let hot_start = match hot_start {
            Some(hot_start) => normalize_hot_start(py, Some(hot_start))?,
            None => self.hot_start,
        };

        let obj = |x: &ArrayView2<f64>| -> Result<Array2<f64>> {
            Python::attach(|py| {
                let args = (x.to_owned().into_pyarray(py),);
                let res = fun.bind(py).call1(args);
                match res {
                    Ok(res) => {
                        let pyarray = res.cast_into::<PyArray2<f64>>().unwrap();
                        Ok(pyarray.to_owned_array())
                    }
                    Err(e) => {
                        log::error!("Error during objective function evaluation: {:?}", e);
                        Err(egobox_ego::EgoError::UserFnError(e.to_string()))
                    }
                }
            })
        };

        let n_fcstr = fcstrs.len();
        if !fcstr_specs.is_empty() && fcstr_specs.len() != n_fcstr {
            return Err(PyValueError::new_err(format!(
                "fcstr_specs length ({}) must match fcstrs length ({})",
                fcstr_specs.len(),
                n_fcstr
            )));
        }

        let fcstr_specs = fcstr_specs
            .into_iter()
            .map(|spec| spec.inner)
            .collect::<Vec<_>>();

        let fcstrs = fcstrs
            .iter()
            .map(|cstr| {
                |x: &[f64], g: Option<&mut [f64]>, _u: &mut InfillObjData<f64>| -> f64 {
                    Python::attach(|py| {
                        if let Some(g) = g {
                            let args = (Array1::from(x.to_vec()).into_pyarray(py), true);
                            let grad = cstr.bind(py).call1(args).unwrap();
                            let grad = grad.cast_into::<PyArray1<f64>>().unwrap().readonly();
                            g.copy_from_slice(grad.as_slice().unwrap())
                        }
                        let args = (Array1::from(x.to_vec()).into_pyarray(py), false);
                        cstr.bind(py).call1(args).unwrap().extract().unwrap()
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
                    n_fcstr,
                    self.doe.as_ref(),
                    outdir.as_deref(),
                    warm_start,
                    hot_start,
                    seed,
                    timeout,
                )
            })
            .min_within_mixint_space(&self.xtypes)
            .expect("Egor configured");

        let py_run_info = if let Some(ri) = run_info {
            parse_run_info(py, ri)?
        } else {
            RunInfo {
                fname: "objective_function".to_string(),
                num: 1,
            }
        };

        let mixintegor = mixintegor.run_info(egobox_ego::RunInfo {
            fname: py_run_info.fname.clone(),
            num: py_run_info.num,
        });

        let res = py.detach(|| {
            mixintegor
                .run()
                .expect("Egor should optimize the objective function")
        });

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
    /// The function returns several point when multi point qEI strategy is used.
    ///
    /// # Parameters
    ///     x_doe (array[ns, nx]): ns samples where function has been evaluated
    ///     y_doe (array[ns, 1 + n_cstr]): ns values of objecctive and constraints
    ///
    ///     seed (int >= 0):
    ///         Random generator seed to allow computation reproducibility.
    ///
    /// # Returns
    ///     (array[1, nx]): suggested location where to evaluate objective and constraints
    ///
    #[pyo3(signature = (x_doe, y_doe, seed = None))]
    fn suggest(
        &self,
        py: Python,
        x_doe: PyReadonlyArray2<f64>,
        y_doe: PyReadonlyArray2<f64>,
        seed: Option<u64>,
    ) -> Py<PyArray2<f64>> {
        // Merge: suggest() seed arg takes precedence over deprecated constructor seed
        let seed = seed.or(self.seed);

        let x_doe = x_doe.as_array();
        let y_doe = y_doe.as_array();
        let doe = concatenate(Axis(1), &[x_doe.view(), y_doe.view()]).unwrap();

        let mixintegor = egobox_ego::EgorServiceBuilder::optimize()
            .configure(|config| {
                self.apply_config(
                    config,
                    Some(1),
                    0,
                    Some(&doe),
                    None,
                    false,
                    None,
                    seed,
                    None,
                )
            })
            .min_within_mixint_space(&self.xtypes)
            .expect("Egor configured");

        let x_suggested = py.detach(|| mixintegor.suggest(&x_doe, &y_doe));
        x_suggested.to_pyarray(py).into()
    }

    /// This function gives the best evaluation index given the outputs
    /// of the function (objective wrt constraints) under minimization.
    /// Caveat: This function does not take into account function constraints values
    ///
    /// # Parameters
    ///     y_doe (array[ns, 1 + n_cstr]): ns values of objective and constraints
    ///     
    /// # Returns
    ///     index in y_doe of the best evaluation
    ///
    #[pyo3(signature = (y_doe))]
    fn get_result_index(&self, y_doe: PyReadonlyArray2<f64>) -> usize {
        let y_doe = y_doe.as_array();
        // TODO: Make c_doe an optional argument ?
        let n_fcstrs = 0;
        let c_doe = Array2::zeros((y_doe.nrows(), n_fcstrs));
        find_best_result_index(&y_doe, &c_doe, &self.cstr_tol(n_fcstrs))
    }

    /// This function gives the best result given inputs and outputs
    /// of the function (objective wrt constraints) under minimization.
    /// Caveat: This function does not take into account function constraints values
    ///
    /// # Parameters
    ///     x_doe (array[ns, nx]): ns samples where function has been evaluated
    ///     y_doe (array[ns, 1 + n_cstr]): ns values of objective and constraints
    ///     
    /// # Returns
    ///     result
    ///         x_opt (array[1, nx]): x value where fun is at its minimum subject to constraints
    ///         y_opt (array[1, nx]): fun(x_opt)
    ///         x_doe (array[ns, nx]): x values of the final DOE
    ///         y_doe (array[ns, 1 + n_cstr]): y values of the final DOE
    ///
    #[pyo3(signature = (x_doe, y_doe))]
    fn get_result(
        &self,
        py: Python,
        x_doe: PyReadonlyArray2<f64>,
        y_doe: PyReadonlyArray2<f64>,
    ) -> OptimResult {
        let x_doe = x_doe.as_array();
        let y_doe = y_doe.as_array();
        // TODO: Make c_doe an optional argument ?
        let n_fcstrs = 0;
        let c_doe = Array2::zeros((y_doe.nrows(), n_fcstrs));
        let idx = find_best_result_index(&y_doe, &c_doe, &self.cstr_tol(n_fcstrs));
        let x_opt = x_doe.row(idx).to_pyarray(py).into();
        let y_opt = y_doe.row(idx).to_pyarray(py).into();
        let x_doe = x_doe.to_pyarray(py).into();
        let y_doe = y_doe.to_pyarray(py).into();
        OptimResult {
            x_opt,
            y_opt,
            x_doe,
            y_doe,
        }
    }

    /// This function loads surrogate models from a file and returns them as a list of Gpx objects.
    /// The file is expected to be a binary file containing a serialized vector of boxed
    /// surrogate models (Vec<Box<dyn MixtureGpSurrogate>>) generated during optimization execution
    #[pyo3(signature = (file))]
    fn load_gp_models(&self, file: String) -> Vec<Gpx> {
        let msg = format!(
            "Failed to load GP models from file {}. Make sure the file exists and is a valid GP models file.",
            file
        );
        let gp_models = egobox_ego::load_gp_models(file.clone()).expect(&msg);
        gp_models.into_iter().map(Gpx::from).collect()
    }
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

    /// Either use user defined cstr_tol or else use default tolerance for all constraints
    /// n_fcstr is the number of function constraints
    fn cstr_tol(&self, n_fcstr: usize) -> Array1<f64> {
        let cstr_tol = self
            .cstr_tol
            .clone()
            .unwrap_or(vec![egobox_ego::DEFAULT_CSTR_TOL; self.n_cstr + n_fcstr]);
        Array1::from_vec(cstr_tol)
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
        n_fcstr: usize,
        doe: Option<&Array2<f64>>,
        outdir: Option<&str>,
        warm_start: bool,
        hot_start: Option<u64>,
        seed: Option<u64>,
        timeout: Option<f64>,
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
            .n_start(self.n_start)
            .n_doe(self.n_doe);

        // Only set cstr_tol explicitly when user provided it.
        // Otherwise let Rust infer the correct total length, including
        // expanded constraints and function constraints.
        if self.cstr_tol.is_some() {
            let cstr_tol = self.cstr_tol(n_fcstr);
            config = config.cstr_tol(cstr_tol);
        }

        if let Some(ref cstr_specs) = self.cstr_specs {
            config = config.cstr_specs(cstr_specs.clone());
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
                    .n_start(self.gp_config.n_start)
                    .max_eval(self.gp_config.max_eval)
            })
            .infill_strategy(infill_strategy)
            .feasible_infill_strategy(feasible_infill_strategy)
            .cstr_infill(self.cstr_infill)
            .cstr_strategy(cstr_strategy)
            .configure_qei(|qei_config| {
                qei_config
                    .batch(self.qei_config.batch)
                    .strategy(qei_strategy)
                    .optmod(self.qei_config.optmod)
            })
            .infill_optimizer(infill_optimizer)
            .coego(coego_status)
            .target(self.target)
            .warm_start(warm_start)
            .hot_start(hot_start.into())
            .failsafe_strategy(failsafe_strategy);

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

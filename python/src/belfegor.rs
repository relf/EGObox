//! `Belfegor`, multi-objective optimizer: a facade of the Egor optimizer configured with several
//! objectives, returning an approximation of the Pareto front.

use crate::egor::{Egor, MooSetup, Outcome, check_doe};
use crate::gp_config::GpConfig;
use crate::moo_config::MooConfig;
use crate::qei_config::QEiConfig;
use crate::types::*;

use egobox_ego::{EGO_DEFAULT_N_START, find_compromise_index, find_pareto_front_indices};
use ndarray::{Array1, Array2, ArrayView2, Axis, concatenate, s};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Multi-objective optimizer constructor
///
/// Belfegor approximates the Pareto front of several objectives, all minimized, subject to
/// constraints. It shares the options of Egor which apply to several objectives
/// (the multi-objective optimization is experimental).
///
/// Parameters
/// ----------
/// xspecs : list of XSpec, list of [lower, upper] or array[nx, 2]
///     Specifications of the nx components of the input x (eg. len(xspecs) == nx),
///     see Egor for details.
/// n_obj : int >= 1
///     Number of objectives returned first by `fun` (see `minimize`), default is 2.
/// moo_config : MooConfig or dict, optional
///     Multi-objective configuration (strategy, batch size, EIM aggregation, hypervolume-based
///     stop, ParEGO options), see MooConfig for details.
/// gp_config : GpConfig or dict, optional
///     GP configuration of the surrogates, see GpConfig for details.
///     MooStrategy.QEHVI requires single-cluster surrogates (n_clusters=1, the default).
/// n_cstr : int
///     Number of constraints returned by `fun` after the objectives, approximated by surrogates.
///     Can be omitted when `cstr_specs` is given.
/// cstr_specs : list of CstrSpec or dict, optional
///     Describe how each surrogate-modeled constraint (returned by `fun`) should be interpreted
///     (CstrSpec.leq, CstrSpec.geq, CstrSpec.eq, CstrSpec.between, with an optional `tol`),
///     see Egor for details.
/// infill_n_start : int > 0, optional
///     Number of starts of the multistart optimization of the infill criterion (default is 20).
/// n_doe : int >= 0
///     Number of samples of initial LHS sampling (used when DOE not provided by the user).
///     When 0 a number of points is computed automatically.
/// x_doe : array[ns, nx], optional
///     Initial DOE inputs containing ns samples. When `y_doe` is not given,
///     ns evaluations are done to get the output values.
/// y_doe : array[ns, ny], optional
///     Initial DOE outputs [obj_1, ..., obj_n_obj, cstr_1, ... cstr_k] (ny = n_obj + n_cstr)
///     corresponding to `x_doe`, requires `x_doe`.
/// infill_strategy : InfillStrategy
///     Infill criterion of the scalarized objective used by MooStrategy.PAREGO
///     (default InfillStrategy.LOG_EI).
/// feasible_infill_strategy : FeasibleInfillStrategy
///     Weight the infill criterion by the probability of viability (MooStrategy.PAREGO only),
///     see Egor for details.
/// cstr_infill : bool
///     Activate constrained infill criterion (product of probabilities of feasibility of the
///     constraint surrogates).
/// cstr_strategy : ConstraintStrategy
///     Constraint management, either ConstraintStrategy.MC (mean constraint, default) or
///     ConstraintStrategy.UTB (upper trust bound).
/// infill_optimizer : InfillOptimizer
///     Internal optimizer used to optimize infill criteria, InfillOptimizer.COBYLA (default)
///     or InfillOptimizer.SLSQP.
/// failsafe_strategy : FailsafeStrategy
///     Strategy to handle objective computation failure (NaN values) at a given x point:
///     FailsafeStrategy.REJECTION (default), FailsafeStrategy.IMPUTATION (not with
///     MooStrategy.PAREGO) or FailsafeStrategy.VIABILITY.
/// seed : int >= 0, optional
///     Random generator seed used by `minimize()` and `suggest()` when they are not given one.
/// verbose : Verbose or int, optional
///     Logging verbosity level used by `minimize()` and `suggest()` when `minimize()` is not given one.
///
/// Returns
/// -------
/// Belfegor
///     A multi-objective optimizer which can be used to optimize a function using the minimize method.
///
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
pub(crate) struct Belfegor {
    egor: Egor,
    moo: MooSetup,
    moo_config: MooConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl Belfegor {
    #[new]
    #[pyo3(signature = (
        xspecs,
        n_obj = 2,
        moo_config = None,
        gp_config = None,
        n_cstr = 0,
        cstr_specs = None,
        infill_n_start = None,
        n_doe = 0,
        x_doe = None,
        y_doe = None,
        infill_strategy = InfillStrategy::LogEi,
        feasible_infill_strategy = FeasibleInfillStrategy::None,
        cstr_infill = false,
        cstr_strategy = ConstraintStrategy::Mc,
        infill_optimizer = InfillOptimizer::Cobyla,
        failsafe_strategy = FailsafeStrategy::Rejection,
        seed = None,
        verbose = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        #[gen_stub(override_type(type_repr = "typing.Sequence[XSpec] | typing.Sequence[typing.Sequence[builtins.float]] | numpy.typing.NDArray[numpy.float64]", imports = ("typing", "builtins", "numpy", "numpy.typing")))]
        xspecs: Py<PyAny>,
        n_obj: usize,
        #[gen_stub(override_type(type_repr = "MooConfig | builtins.dict[builtins.str, typing.Any] | None", imports = ("typing", "builtins")))]
        moo_config: Option<MooConfig>,
        #[gen_stub(override_type(type_repr = "GpConfig | builtins.dict[builtins.str, typing.Any] | None", imports = ("typing", "builtins")))]
        gp_config: Option<GpConfig>,
        n_cstr: usize,
        #[gen_stub(override_type(type_repr = "typing.Sequence[CstrSpec | builtins.dict[builtins.str, typing.Any]] | None", imports = ("typing", "builtins")))]
        cstr_specs: Option<Vec<CstrSpec>>,
        infill_n_start: Option<usize>,
        n_doe: usize,
        x_doe: Option<PyReadonlyArray2<f64>>,
        y_doe: Option<PyReadonlyArray2<f64>>,
        infill_strategy: InfillStrategy,
        feasible_infill_strategy: FeasibleInfillStrategy,
        cstr_infill: bool,
        cstr_strategy: ConstraintStrategy,
        infill_optimizer: InfillOptimizer,
        failsafe_strategy: FailsafeStrategy,
        seed: Option<u64>,
        #[gen_stub(override_type(type_repr = "Verbose | builtins.int | None", imports = ("typing", "builtins")))]
        verbose: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if n_obj == 0 {
            return Err(PyValueError::new_err("n_obj should be at least 1"));
        }
        let moo_config = moo_config.unwrap_or_default();
        if moo_config.batch == 0 {
            return Err(PyValueError::new_err(
                "moo_config batch should be at least 1",
            ));
        }
        // batches of points: qEHVI or Kriging believer with default settings
        let qei_config = QEiConfig {
            batch: moo_config.batch,
            ..QEiConfig::default()
        };
        let egor = Egor::build(
            py,
            xspecs,
            gp_config,
            n_cstr,
            None,
            cstr_specs,
            infill_n_start.unwrap_or(EGO_DEFAULT_N_START),
            n_doe,
            None,
            x_doe,
            y_doe,
            infill_strategy,
            feasible_infill_strategy,
            cstr_infill,
            cstr_strategy,
            Some(qei_config),
            infill_optimizer,
            None,
            0,
            None,
            failsafe_strategy,
            seed,
            verbose,
        )?;
        let moo = MooSetup {
            n_obj,
            config: (&moo_config).into(),
        };
        Ok(Belfegor {
            egor,
            moo,
            moo_config,
        })
    }

    /// Number of objectives
    #[getter]
    fn n_obj(&self) -> usize {
        self.moo.n_obj
    }

    /// Multi-objective configuration
    #[getter]
    fn moo_config(&self) -> MooConfig {
        self.moo_config.clone()
    }

    /// This function approximates the Pareto front of the objectives of a given function "fun"
    ///
    /// Parameters
    /// ----------
    /// fun : callable (array[n, nx]) -> array[n, ny]
    ///     The function to be minimized: fun(x) = [obj_1(x), ..., obj_m(x), cstr_1(x), ... cstr_k(x)]
    ///     where m is the number of objectives (n_obj) and k the number of constraints (n_cstr),
    ///     hence ny = n_obj + n_cstr. All objectives are minimized, cstr functions are expected
    ///     to be negative (<=0) at the Pareto points (unless `cstr_specs` is used).
    /// fcstrs : list, optional
    ///     Cheap constraint functions g, evaluated directly (not approximated by surrogates),
    ///     which have to be made negative (g(x) <= 0, unless `fcstr_specs` is used),
    ///     see Egor.minimize for the accepted forms.
    /// fcstr_specs : list of CstrSpec or dict, optional
    ///     One CstrSpec per fcstr specifying how each function constraint should be interpreted.
    /// max_iters : int
    ///     The iteration budget, number of fun calls is "n_doe + batch * max_iters"
    ///     (batch being moo_config.batch).
    /// run_info : RunInfo or dict, optional
    ///     Information about the run (fname, num) used for checkpoint file naming.
    /// outdir : str, optional
    ///     Directory to write optimization history and used as search path for warm start doe
    /// warm_start : bool
    ///     Start by loading initial doe from <outdir> directory
    /// hot_start : bool or int >= 0, optional
    ///     Save the optimizer state at each iteration and restart from a previous checkpoint,
    ///     see Egor.minimize for details.
    /// seed : int >= 0, optional
    ///     Random generator seed to allow computation reproducibility.
    ///     When None, the seed given to the constructor (if any) is used.
    /// timeout : float, optional
    ///     Timeout in seconds, checked after each iteration.
    /// verbose : Verbose or int, optional
    ///     Logging verbosity level, see Egor.minimize for the possible values.
    ///     When None, the verbosity given to the constructor is used.
    /// stop_on_error : bool
    ///     If true, terminate optimization when the objective function raises an error.
    ///     Otherwise, the error is handled according to failsafe_strategy.
    ///
    /// Returns
    /// -------
    /// BelfegorOptim
    ///     result (ParetoResult) and status (RunStatus) of the optimization, where result holds:
    ///
    ///     * x_pareto (array[np, nx]): x values of the np points of the (constrained) Pareto front
    ///     * y_pareto (array[np, ny]): fun(x_pareto) where ny = n_obj + n_cstr
    ///     * x_opt (array[nx]): compromise point of the front
    ///     * y_opt (array[ny]): fun(x_opt)
    ///     * x_doe (array[ns, nx]): x values of the final DOE
    ///     * y_doe (array[ns, ny]): y values of the final DOE
    ///
    ///     y values hold the raw constraint values as returned by `fun` (not transformed
    ///     by `cstr_specs`).
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
        #[gen_stub(override_type(type_repr = "typing.Sequence[CstrSpec | builtins.dict[builtins.str, typing.Any]] | None", imports = ("typing", "builtins")))]
        fcstr_specs: Option<Vec<CstrSpec>>,
        max_iters: usize,
        #[gen_stub(override_type(type_repr = "RunInfo | builtins.dict[builtins.str, typing.Any] | None", imports = ("typing", "builtins")))]
        run_info: Option<Py<PyAny>>,
        outdir: Option<String>,
        warm_start: bool,
        #[gen_stub(override_type(type_repr = "builtins.bool | builtins.int | None", imports = ("builtins",)))]
        hot_start: Option<Py<PyAny>>,
        seed: Option<u64>,
        timeout: Option<f64>,
        #[gen_stub(override_type(type_repr = "Verbose | builtins.int | None", imports = ("typing", "builtins")))]
        verbose: Option<Py<PyAny>>,
        stop_on_error: bool,
    ) -> PyResult<BelfegorOptim> {
        let (outcome, status) = self.egor.optimize(
            py,
            fun,
            fcstrs,
            fcstr_specs,
            max_iters,
            run_info,
            outdir,
            warm_start,
            hot_start,
            seed,
            timeout,
            verbose,
            stop_on_error,
            Some(&self.moo),
        )?;
        let Outcome::Pareto(res) = outcome else {
            unreachable!("multi-objective optimization gives a Pareto front")
        };
        let result = ParetoResult {
            x_pareto: res.x_pareto.into_pyarray(py).into(),
            y_pareto: res.y_pareto.into_pyarray(py).into(),
            x_opt: res.x_opt.into_pyarray(py).into(),
            y_opt: res.y_opt.into_pyarray(py).into(),
            x_doe: res.x_doe.into_pyarray(py).into(),
            y_doe: res.y_doe.into_pyarray(py).into(),
        };
        Ok(BelfegorOptim {
            result: Bound::new(py, result)?.into(),
            status,
        })
    }

    /// This function gives the next best locations where to evaluate the function
    /// under optimization wrt to previous evaluations.
    /// The function returns several points when a batch is configured (moo_config.batch).
    ///
    /// Parameters
    /// ----------
    /// x_doe : array[ns, nx]
    ///     ns samples where function has been evaluated
    /// y_doe : array[ns, n_obj + n_cstr]
    ///     ns values of objectives and constraints
    /// seed : int >= 0, optional
    ///     Random generator seed to allow computation reproducibility.
    ///     When None, the seed given to the constructor (if any) is used.
    ///
    /// Returns
    /// -------
    /// array[batch, nx]
    ///     suggested locations where to evaluate objectives and constraints
    ///     where batch is the batch size (moo_config.batch, 1 by default)
    ///
    #[pyo3(signature = (x_doe, y_doe, seed = None))]
    fn suggest(
        &self,
        py: Python,
        x_doe: PyReadonlyArray2<f64>,
        y_doe: PyReadonlyArray2<f64>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        self.egor
            .suggest_points(py, x_doe, y_doe, seed, Some(&self.moo))
    }

    /// This function gives the indices of the points of the (constrained) Pareto front
    /// given the outputs of the function (objectives and constraints) under minimization.
    /// Caveat: This function does not take into account function constraints values
    ///
    /// Parameters
    /// ----------
    /// y_doe : array[ns, n_obj + n_cstr]
    ///     ns values of objectives and constraints as returned by `fun` (see `minimize`),
    ///     constraints are interpreted with `cstr_specs` and their tolerances if given
    ///
    /// Returns
    /// -------
    /// list of int
    ///     indices in y_doe of the Pareto front points (in y_doe order). When no point is feasible,
    ///     the index of the point with the smallest constraint violation.
    ///
    #[pyo3(signature = (y_doe))]
    fn pareto_indices(&self, y_doe: PyReadonlyArray2<f64>) -> PyResult<Vec<usize>> {
        let y_doe = y_doe.as_array();
        check_doe(None, &y_doe)?;
        let (y, c, cstr_tol) = self.internal_data(&y_doe)?;
        Ok(find_pareto_front_indices(&y, &c, self.moo.n_obj, &cstr_tol))
    }

    /// This function gives the Pareto front and the compromise point given inputs and outputs
    /// of the function (objectives and constraints) under minimization.
    /// Caveat: This function does not take into account function constraints values
    ///
    /// Parameters
    /// ----------
    /// x_doe : array[ns, nx]
    ///     ns samples where function has been evaluated
    /// y_doe : array[ns, n_obj + n_cstr]
    ///     ns values of objectives and constraints as returned by `fun` (see `minimize`),
    ///     constraints are interpreted with `cstr_specs` and their tolerances if given
    ///
    /// Returns
    /// -------
    /// ParetoResult
    ///     * x_pareto (array[np, nx]), y_pareto (array[np, ny]): the Pareto front points
    ///     * x_opt (array[nx]), y_opt (array[ny]): the compromise point of the front
    ///     * x_doe (array[ns, nx]), y_doe (array[ns, ny]): the given data
    ///
    #[pyo3(signature = (x_doe, y_doe))]
    fn pareto_result(
        &self,
        py: Python,
        x_doe: PyReadonlyArray2<f64>,
        y_doe: PyReadonlyArray2<f64>,
    ) -> PyResult<ParetoResult> {
        let x_doe = x_doe.as_array();
        let y_doe = y_doe.as_array();
        check_doe(Some(&x_doe), &y_doe)?;
        let (y, c, cstr_tol) = self.internal_data(&y_doe)?;
        let front = find_pareto_front_indices(&y, &c, self.moo.n_obj, &cstr_tol);
        let best = find_compromise_index(&y, &c, self.moo.n_obj, &cstr_tol).ok_or_else(|| {
            PyValueError::new_err("y_doe should contain at least one point with finite values")
        })?;
        Ok(ParetoResult {
            x_pareto: x_doe.select(Axis(0), &front).into_pyarray(py).into(),
            y_pareto: y_doe.select(Axis(0), &front).into_pyarray(py).into(),
            x_opt: x_doe.row(best).to_pyarray(py).into(),
            y_opt: y_doe.row(best).to_pyarray(py).into(),
            x_doe: x_doe.to_pyarray(py).into(),
            y_doe: y_doe.to_pyarray(py).into(),
        })
    }
}

impl Belfegor {
    /// Internal data given raw `y_doe` [obj_1, ..., obj_n_obj, cstr_1, ... cstr_n_cstr]:
    /// objectives and constraints interpreted with `cstr_specs` (`<= 0` form), no function
    /// constraint values and the constraint tolerances, as the optimizer does
    fn internal_data(
        &self,
        y_doe: &ArrayView2<f64>,
    ) -> PyResult<(Array2<f64>, Array2<f64>, Array1<f64>)> {
        let n_obj = self.moo.n_obj;
        let ny = n_obj + self.egor.n_cstr;
        if y_doe.ncols() != ny {
            return Err(PyValueError::new_err(format!(
                "y_doe should be of shape (ns, n_obj + n_cstr) = (ns, {ny}), got {:?}",
                y_doe.shape()
            )));
        }
        let y = match self.egor.cstr_specs.as_ref() {
            Some(specs) => {
                let specs = specs.iter().map(|s| s.inner.clone()).collect::<Vec<_>>();
                // constraints transformation expects [obj, cstr_1, ...] rows
                let y1 = concatenate![
                    Axis(1),
                    y_doe.slice(s![.., ..1]),
                    y_doe.slice(s![.., n_obj..])
                ];
                let cstrs = egobox_ego::transform_constraints(&y1, &specs);
                concatenate![
                    Axis(1),
                    y_doe.slice(s![.., ..n_obj]),
                    cstrs.slice(s![.., 1..])
                ]
            }
            None => y_doe.to_owned(),
        };
        let cstr_tol = self
            .egor
            .internal_cstr_tol(&[], 0)
            .unwrap_or_else(|| Array1::from_elem(y.ncols() - n_obj, egobox_ego::DEFAULT_CSTR_TOL));
        let c = Array2::zeros((y.nrows(), 0));
        Ok((y, c, cstr_tol))
    }
}

/// ParetoResult contains the results of a multi-objective optimization run: the approximation
/// of the Pareto front, its compromise point and the DOE points and values which include
/// initial points and the optimization history.
/// y values hold the objectives and the raw constraint values as returned by the objective
/// function (ny = n_obj + n_cstr columns, even when `cstr_specs` is used).
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Debug)]
pub(crate) struct ParetoResult {
    /// Pareto set: x values of the points of the (constrained) Pareto front
    #[pyo3(get)]
    pub(crate) x_pareto: Py<PyArray2<f64>>,
    /// Pareto front: y values of the points of the Pareto set
    #[pyo3(get)]
    pub(crate) y_pareto: Py<PyArray2<f64>>,
    /// Compromise point of the front: the point minimizing the uniform-weight augmented
    /// Tchebycheff function of the objectives normalized with the front bounds
    #[pyo3(get)]
    pub(crate) x_opt: Py<PyArray1<f64>>,
    /// y value of the compromise point
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
impl ParetoResult {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let n_pareto = self.x_pareto.bind(py).len()?;
        let n_doe = self.x_doe.bind(py).len()?;
        repr_kwargs(
            "ParetoResult",
            &[
                ("n_pareto", n_pareto.into_pyobject(py)?.into_any()),
                ("x_opt", self.x_opt.bind(py).clone().into_any()),
                ("y_opt", self.y_opt.bind(py).clone().into_any()),
                ("n_doe", n_doe.into_pyobject(py)?.into_any()),
            ],
        )
    }
}

/// Belfegor optimization output
///
/// The optimization result fields are also available directly (`x_pareto`, `y_pareto`,
/// `x_opt`, `y_opt`, `x_doe`, `y_doe`) and the output can be unpacked as
/// `x_pareto, y_pareto = belfegor.minimize(...)`.
#[gen_stub_pyclass]
#[pyclass(skip_from_py_object, module = "egobox")]
#[derive(Debug)]
pub(crate) struct BelfegorOptim {
    /// Result of optimization run
    #[pyo3(get)]
    pub(crate) result: Py<ParetoResult>,
    /// Status of optimization run
    #[pyo3(get)]
    pub(crate) status: RunStatus,
}

#[gen_stub_pymethods]
#[pymethods]
impl BelfegorOptim {
    /// Pareto set, same as `result.x_pareto`
    #[getter]
    fn x_pareto(&self, py: Python) -> Py<PyArray2<f64>> {
        self.result.borrow(py).x_pareto.clone_ref(py)
    }

    /// Pareto front, same as `result.y_pareto`
    #[getter]
    fn y_pareto(&self, py: Python) -> Py<PyArray2<f64>> {
        self.result.borrow(py).y_pareto.clone_ref(py)
    }

    /// Compromise point of the front, same as `result.x_opt`
    #[getter]
    fn x_opt(&self, py: Python) -> Py<PyArray1<f64>> {
        self.result.borrow(py).x_opt.clone_ref(py)
    }

    /// y value of the compromise point, same as `result.y_opt`
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

    /// Iterate over (x_pareto, y_pareto) to allow `x_pareto, y_pareto = belfegor.minimize(...)`
    #[gen_stub(override_return_type(type_repr = "typing.Iterator[numpy.typing.NDArray[numpy.float64]]", imports = ("typing", "numpy", "numpy.typing")))]
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let result = self.result.borrow(py);
        PyTuple::new(
            py,
            [
                result.x_pareto.bind(py).as_any(),
                result.y_pareto.bind(py).as_any(),
            ],
        )?
        .try_iter()
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        repr_kwargs(
            "BelfegorOptim",
            &[
                ("result", self.result.bind(py).clone().into_any()),
                ("status", self.status.clone().into_pyobject(py)?.into_any()),
            ],
        )
    }
}

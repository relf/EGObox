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
//! See the [tutorial notebook](https://github.com/relf/egobox/doc/Egor_Tutorial.ipynb) for usage.
//!

use crate::domain::*;
use crate::gp_config::*;
use crate::trego_config::TregoConfig;
use crate::types::*;
use egobox_ego::{CoegoStatus, InfillObjData, find_best_result_index};
use egobox_gp::ThetaTuning;
use egobox_moe::NbClusters;
use ndarray::{Array1, Array2, ArrayView2, Axis, array, concatenate};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::cmp::Ordering;

/// Optimizer constructor
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
///     infill_optimizer (InfillOptimizer enum):
///         Internal optimizer used to optimize infill criteria.
///         Can be either InfillOptimizer.COBYLA or InfillOptimizer.SLSQP
///
///     cstr_infill (bool):
///         Activate constrained infill criterion where the product of probability of feasibility of constraints
///         used as a factor of the infill criterion specified via infill_strategy
///         
///     cstr_strategy (ConstraintStrategy enum):
///         Constraint management either use the mean value or upper bound
///         Can be either ConstraintStrategy.MeanValue or ConstraintStrategy.UpperTrustedBound.
///
///     q_infill_strategy (QInfillStrategy enum):
///         Parallel infill criteria (aka qEI) to get virtual next promising points in order to allow
///         q parallel evaluations of the function under optimization (only used when q_points > 1)
///         Can be either QInfillStrategy.KB (Kriging Believer),
///         QInfillStrategy.KBLB (KB Lower Bound), QInfillStrategy.KBUB (KB Upper Bound),
///         QInfillStrategy.CLMIN (Constant Liar Minimum)
///
///     q_points (int > 0):
///         Number of points to be evaluated to allow parallel evaluation of the function under optimization.
///
///     q_optmod (int >= 1):
///         Number of iterations between two surrogate models true training (hypermarameters optimization)
///         otherwise previous hyperparameters are re-used only when computing q_points to be evaluated in parallel.
///         The default value is 1 meaning surrogates are properly trained for each q points determination.
///         The value is used as a modulo of iteration number * q_points to trigger true training.
///         This is used to decrease the number of training at the expense of surrogate accuracy.    
///
///     trego (TregoConfig or None):
///         When TregoConfig is given the TREGO algorithm is used to enhance local convergence
///         of the EGO algorithm by using a trust region strategy. See TregoConfig for details.
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
///     outdir (String):
///         Directory to write optimization history and used as search path for warm start doe
///
///     warm_start (bool):
///         Start by loading initial doe from <outdir> directory
///
///     hot_start (int >= 0 or None):
///         When hot_start>=0 saves optimizer state at each iteration and starts from a previous checkpoint
///         if any for the given hot_start number of iterations beyond the max_iters nb of iterations.
///         In an unstable environment were there can be crashes it allows to restart the optimization
///         from the last iteration till stopping criterion is reached. Just use hot_start=0 in this case.
///         When specifying an extended nb of iterations (hot_start > 0) it can allow to continue till max_iters +
///         hot_start nb of iters is reached (provided the stopping criterion is max_iters)
///         Checkpoint information is stored in .checkpoint/egor.arg binary file.
///
///     seed (int >= 0):
///         Random generator seed to allow computation reproducibility.
///      
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct Egor {
    pub xspecs: Py<PyAny>,
    pub gp_config: GpConfig,
    pub n_cstr: usize,
    pub cstr_tol: Option<Vec<f64>>,
    pub n_start: usize,
    pub n_doe: usize,
    pub doe: Option<Array2<f64>>,
    pub infill_strategy: InfillStrategy,
    pub cstr_infill: bool,
    pub cstr_strategy: ConstraintStrategy,
    pub q_points: usize,
    pub q_infill_strategy: QInfillStrategy,
    pub infill_optimizer: InfillOptimizer,
    pub trego: Option<TregoConfig>,
    pub coego_n_coop: usize,
    pub q_optmod: usize,
    pub target: f64,
    pub outdir: Option<String>,
    pub warm_start: bool,
    pub hot_start: Option<u64>,
    pub seed: Option<u64>,
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
        n_start = 20,
        n_doe = 0,
        doe = None,
        infill_strategy = InfillStrategy::LogEi,
        cstr_infill = false,
        cstr_strategy = ConstraintStrategy::Mc,
        q_points = 1,
        q_infill_strategy = QInfillStrategy::Kb,
        infill_optimizer = InfillOptimizer::Cobyla,
        trego = None,
        coego_n_coop = 0,
        q_optmod = 1,
        target = f64::MIN,
        outdir = None,
        warm_start = false,
        hot_start = None,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        xspecs: Py<PyAny>,
        gp_config: GpConfig,
        n_cstr: usize,
        cstr_tol: Option<Vec<f64>>,
        n_start: usize,
        n_doe: usize,
        doe: Option<PyReadonlyArray2<f64>>,
        infill_strategy: InfillStrategy,
        cstr_infill: bool,
        cstr_strategy: ConstraintStrategy,
        q_points: usize,
        q_infill_strategy: QInfillStrategy,
        infill_optimizer: InfillOptimizer,
        trego: Option<TregoConfig>,
        coego_n_coop: usize,
        q_optmod: usize,
        target: f64,
        outdir: Option<String>,
        warm_start: bool,
        hot_start: Option<u64>,
        seed: Option<u64>,
    ) -> Self {
        let doe = doe.map(|x| x.to_owned_array());

        Egor {
            xspecs,
            gp_config,
            n_cstr,
            cstr_tol,
            n_start,
            n_doe,
            doe,
            infill_strategy,
            cstr_infill,
            cstr_strategy,
            q_points,
            q_infill_strategy,
            infill_optimizer,
            trego,
            coego_n_coop,
            q_optmod,
            target,
            outdir,
            warm_start,
            hot_start,
            seed,
        }
    }

    /// ```ignore
    /// This function finds the minimum of a given function "fun"
    ///
    /// Parameters
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
    ///     max_iters:
    ///         the iteration budget, number of fun calls is "n_doe + q_points * max_iters".
    ///
    ///     fcstrs:
    ///         list of constraints functions defined as g(x, return_grad): (ndarray[nx], bool) -> float or ndarray[nx,]
    ///         If the given "return_grad" boolean is "False" the function has to return the constraint float value
    ///         to be made negative by the optimizer (which drives the input array "x").
    ///         Otherwise the function has to return the gradient (ndarray[nx,]) of the constraint function
    ///         wrt the "nx" components of "x".
    ///
    /// Returns
    ///     optimization result
    ///         x_opt (array[1, nx]): x value where fun is at its minimum subject to constraints
    ///         y_opt (array[1, nx]): fun(x_opt)
    ///
    #[pyo3(signature = (fun, fcstrs=vec![], max_iters = 20, run_info = None))]
    fn minimize(
        &self,
        py: Python,
        fun: Py<PyAny>,
        fcstrs: Vec<Py<PyAny>>,
        max_iters: usize,
        run_info: Option<Py<PyAny>>,
    ) -> PyResult<OptimResult> {
        let obj = |x: &ArrayView2<f64>| -> Array2<f64> {
            Python::attach(|py| {
                let args = (x.to_owned().into_pyarray(py),);
                let res = fun.bind(py).call1(args).unwrap();
                let pyarray = res.downcast_into::<PyArray2<f64>>().unwrap();
                pyarray.to_owned_array()
            })
        };

        let n_fcstr = fcstrs.len();
        let fcstrs = fcstrs
            .iter()
            .map(|cstr| {
                |x: &[f64], g: Option<&mut [f64]>, _u: &mut InfillObjData<f64>| -> f64 {
                    Python::attach(|py| {
                        if let Some(g) = g {
                            let args = (Array1::from(x.to_vec()).into_pyarray(py), true);
                            let grad = cstr.bind(py).call1(args).unwrap();
                            let grad = grad.downcast_into::<PyArray1<f64>>().unwrap().readonly();
                            g.copy_from_slice(grad.as_slice().unwrap())
                        }
                        let args = (Array1::from(x.to_vec()).into_pyarray(py), false);
                        cstr.bind(py).call1(args).unwrap().extract().unwrap()
                    })
                }
            })
            .collect::<Vec<_>>();

        let xtypes: Vec<egobox_ego::XType> = parse(py, self.xspecs.clone_ref(py));

        let mixintegor = egobox_ego::EgorFactory::optimize(obj)
            .subject_to(fcstrs)
            .configure(|config| {
                self.apply_config(config, Some(max_iters), n_fcstr, self.doe.as_ref())
            })
            .min_within_mixint_space(&xtypes)
            .expect("Egor configured");

        let mixintegor = if let Some(ri) = run_info {
            let ri: RunInfo = ri.extract(py).unwrap();
            mixintegor.run_info(egobox_ego::RunInfo {
                fname: ri.fname,
                num: ri.num,
            })
        } else {
            mixintegor
        };

        let res = py.detach(|| {
            mixintegor
                .run()
                .expect("Egor should optimize the objective function")
        });
        let x_opt = res.x_opt.into_pyarray(py).to_owned();
        let y_opt = res.y_opt.into_pyarray(py).to_owned();
        let x_doe = res.x_doe.into_pyarray(py).to_owned();
        let y_doe = res.y_doe.into_pyarray(py).to_owned();
        Ok(OptimResult {
            x_opt: x_opt.into(),
            y_opt: y_opt.into(),
            x_doe: x_doe.into(),
            y_doe: y_doe.into(),
        })
    }

    /// This function gives the next best location where to evaluate the function
    /// under optimization wrt to previous evaluations.
    /// The function returns several point when multi point qEI strategy is used.
    ///
    /// Parameters
    ///     x_doe (array[ns, nx]): ns samples where function has been evaluated
    ///     y_doe (array[ns, 1 + n_cstr]): ns values of objecctive and constraints
    ///
    /// Returns
    ///     (array[1, nx]): suggested location where to evaluate objective and constraints
    ///
    #[pyo3(signature = (x_doe, y_doe))]
    fn suggest(
        &self,
        py: Python,
        x_doe: PyReadonlyArray2<f64>,
        y_doe: PyReadonlyArray2<f64>,
    ) -> Py<PyArray2<f64>> {
        let x_doe = x_doe.as_array();
        let y_doe = y_doe.as_array();
        let doe = concatenate(Axis(1), &[x_doe.view(), y_doe.view()]).unwrap();
        let xtypes: Vec<egobox_ego::XType> = parse(py, self.xspecs.clone_ref(py));

        let mixintegor = egobox_ego::EgorServiceBuilder::optimize()
            .configure(|config| self.apply_config(config, Some(1), 0, Some(&doe)))
            .min_within_mixint_space(&xtypes)
            .expect("Egor configured");

        let x_suggested = py.detach(|| mixintegor.suggest(&x_doe, &y_doe));
        x_suggested.to_pyarray(py).into()
    }

    /// This function gives the best evaluation index given the outputs
    /// of the function (objective wrt constraints) under minimization.
    /// Caveat: This function does not take into account function constraints values
    ///
    /// Parameters
    ///     y_doe (array[ns, 1 + n_cstr]): ns values of objective and constraints
    ///     
    /// Returns
    ///     index in y_doe of the best evaluation
    ///
    #[pyo3(signature = (y_doe))]
    fn get_result_index(&self, y_doe: PyReadonlyArray2<f64>) -> usize {
        let y_doe = y_doe.as_array();
        // TODO: Make c_doe an optional argument ?
        let n_fcstrs = 0;
        let c_doe = Array2::zeros((y_doe.ncols(), n_fcstrs));
        find_best_result_index(&y_doe, &c_doe, &self.cstr_tol(n_fcstrs))
    }

    /// This function gives the best result given inputs and outputs
    /// of the function (objective wrt constraints) under minimization.
    /// Caveat: This function does not take into account function constraints values
    ///
    /// Parameters
    ///     x_doe (array[ns, nx]): ns samples where function has been evaluated
    ///     y_doe (array[ns, 1 + n_cstr]): ns values of objective and constraints
    ///     
    /// Returns
    ///     result
    ///         x_opt (array[1, nx]): x value where fun is at its minimum subject to constraints
    ///         y_opt (array[1, nx]): fun(x_opt)
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
        let c_doe = Array2::zeros((y_doe.ncols(), n_fcstrs));
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

    fn cstr_strategy(&self) -> egobox_ego::ConstraintStrategy {
        match self.cstr_strategy {
            ConstraintStrategy::Mc => egobox_ego::ConstraintStrategy::MeanConstraint,
            ConstraintStrategy::Utb => egobox_ego::ConstraintStrategy::UpperTrustBound,
        }
    }

    fn qei_strategy(&self) -> egobox_ego::QEiStrategy {
        match self.q_infill_strategy {
            QInfillStrategy::Kb => egobox_ego::QEiStrategy::KrigingBeliever,
            QInfillStrategy::Kblb => egobox_ego::QEiStrategy::KrigingBelieverLowerBound,
            QInfillStrategy::Kbub => egobox_ego::QEiStrategy::KrigingBelieverUpperBound,
            QInfillStrategy::Clmin => egobox_ego::QEiStrategy::ConstantLiarMinimum,
        }
    }

    fn infill_optimizer(&self) -> egobox_ego::InfillOptimizer {
        match self.infill_optimizer {
            InfillOptimizer::Cobyla => egobox_ego::InfillOptimizer::Cobyla,
            InfillOptimizer::Slsqp => egobox_ego::InfillOptimizer::Slsqp,
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

    fn apply_config(
        &self,
        config: egobox_ego::EgorConfig,
        max_iters: Option<usize>,
        n_fcstr: usize,
        doe: Option<&Array2<f64>>,
    ) -> egobox_ego::EgorConfig {
        let infill_strategy = self.infill_strategy();
        let cstr_strategy = self.cstr_strategy();
        let qei_strategy = self.qei_strategy();
        let infill_optimizer = self.infill_optimizer();
        let coego_status = if self.coego_n_coop == 0 {
            CoegoStatus::Disabled
        } else {
            CoegoStatus::Enabled(self.coego_n_coop)
        };

        let cstr_tol = self.cstr_tol(n_fcstr);

        let mut config = config
            .n_cstr(self.n_cstr)
            .max_iters(max_iters.unwrap_or(1))
            .n_start(self.n_start)
            .n_doe(self.n_doe)
            .cstr_tol(cstr_tol)
            .configure_gp(|gp| {
                let regr = RegressionSpec(self.gp_config.regr_spec);
                let corr = CorrelationSpec(self.gp_config.corr_spec);
                let n_start = self.gp_config.n_start.max(0) as usize;
                gp.regression_spec(egobox_moe::RegressionSpec::from_bits(regr.0).unwrap())
                    .correlation_spec(egobox_moe::CorrelationSpec::from_bits(corr.0).unwrap())
                    .kpls_dim(self.gp_config.kpls_dim)
                    .n_clusters(self.n_clusters())
                    .recombination(self.recombination())
                    .theta_tuning(self.theta_tuning())
                    .n_start(n_start)
                    .max_eval(self.gp_config.max_eval)
            })
            .infill_strategy(infill_strategy)
            .cstr_infill(self.cstr_infill)
            .cstr_strategy(cstr_strategy)
            .q_points(self.q_points)
            .qei_strategy(qei_strategy)
            .infill_optimizer(infill_optimizer)
            .configure_trego(|trego| {
                if let Some(trego_cfg) = self.trego.as_ref() {
                    trego_cfg.clone().into()
                } else {
                    trego
                }
            })
            .coego(coego_status)
            .q_optmod(self.q_optmod)
            .target(self.target)
            .warm_start(self.warm_start)
            .hot_start(self.hot_start.into());
        if let Some(doe) = doe {
            config = config.doe(doe);
        };

        if let Some(outdir) = self.outdir.as_ref().cloned() {
            config = config.outdir(outdir);
        };
        if let Some(seed) = self.seed {
            config = config.seed(seed);
        };
        config
    }
}

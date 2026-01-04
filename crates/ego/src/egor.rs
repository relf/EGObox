//! Egor optimizer implements EGO algorithm with basic handling of constraints
//! with the following features:
//!
//! * Mixture of Gaussian processes
//! * Mixed-integer variables handling through continuous relaxation
//! * Trust-region EGO optional activation
//! * Infill criteria: EI, LogEI, WB2, WB2S, CEI
//! * Multi-point infill strategy (aka qEI)
//! * Warm/hot start
//!
//! See refences below.
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{EgorBuilder, InfillStrategy, InfillOptimizer};
//!
//! # use rand_xoshiro::Xoshiro256Plus;
//! # use ndarray_rand::rand::SeedableRng;
//! use argmin_testfunctions::rosenbrock;
//!
//! // Rosenbrock test function: minimum y_opt = 0 at x_opt = (1, 1)
//! fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
//!     y
//! }
//!
//! let xlimits = array![[-2., 2.], [-2., 2.]];
//! let res = EgorBuilder::optimize(rosenb).configure(|config|
//!         config
//!             .infill_strategy(InfillStrategy::EI)
//!             .n_doe(10)
//!             .target(1e-1)
//!             .max_iters(30))
//!     .min_within(&xlimits)
//!     .expect("optimizer configured")
//!     .run()
//!     .expect("Rosenbrock minimization");
//! println!("Rosenbrock min result = {:?}", res);
//! ```
//!
//! Constraints are expected to be evaluated with the objective function
//! meaning that the function passed to the optimizer has to return
//! a vector consisting of [obj, cstr_1, ..., cstr_n] and the cstr values
//! are intended to be negative at the end of the optimization.
//! Constraint number should be declared with `n_cstr` setter.
//! A tolerance can be adjust with `cstr_tol` setter for relaxing constraint violation
//! if specified cstr values should be < `cstr_tol` (instead of < 0)
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{EgorBuilder, InfillStrategy, InfillOptimizer};
//! # use rand_xoshiro::Xoshiro256Plus;
//! # use ndarray_rand::rand::SeedableRng;
//!
//! // Function G24: 1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
//! fn g24(x: &ArrayView1<f64>) -> f64 {
//!    -x[0] - x[1]
//! }
//!
//! // Constraints < 0
//! fn g24_c1(x: &ArrayView1<f64>) -> f64 {
//!     -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
//! }
//!
//! fn g24_c2(x: &ArrayView1<f64>) -> f64 {
//!     -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0)
//!     - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
//!     - 36.0
//! }
//!
//! // Gouped function : objective + constraints
//! fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
//!     let mut y = Array2::zeros((x.nrows(), 3));
//!     Zip::from(y.rows_mut())
//!         .and(x.rows())
//!         .for_each(|mut yi, xi| {
//!             yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]);
//!         });
//!     y
//! }
//!
//! let xlimits = array![[0., 3.], [0., 4.]];
//! let doe = Lhs::new(&xlimits).sample(10);
//! let res = EgorBuilder::optimize(f_g24).configure(|config|
//!             config
//!                 .n_cstr(2)
//!                 .infill_strategy(InfillStrategy::EI)
//!                 .infill_optimizer(InfillOptimizer::Cobyla)
//!                 .doe(&doe)
//!                 .max_iters(40)
//!                 .target(-5.5080))
//!            .min_within(&xlimits)
//!            .expect("optimizer configured")
//!            .run()
//!            .expect("g24 minimized");
//! println!("G24 min result = {:?}", res);
//! ```
//!
use crate::EgorConfig;
use crate::EgorState;
use crate::HotStartMode;
use crate::errors::Result;
use crate::gpmix::mixint::*;
use crate::types::*;
use crate::{CHECKPOINT_FILE, CheckpointingFrequency, HotStartCheckpoint};
use crate::{EgorSolver, to_xtypes};

use argmin::core::observers::ObserverMode;

use egobox_moe::GpMixtureParams;
use log::info;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, concatenate};

use argmin::core::{Error, Executor, KV, State, observers::Observe};
use serde::de::DeserializeOwned;

use ndarray_npy::write_npy;
use std::path::PathBuf;

/// Json filename for configuration
pub const CONFIG_FILE: &str = "egor_config.json";
/// Numpy filename for optimization history
pub const HISTORY_FILE: &str = "egor_history.npy";

/// Egor run metadata
#[derive(Clone, Default)]
pub struct RunInfo {
    /// The objective function name
    pub fname: String,
    /// A number of replication
    pub num: usize,
}

/// EGO optimizer builder allowing to specify function to be minimized
/// subject to constraints intended to be negative.
///
pub struct EgorFactory<O: GroupFunc, C: CstrFn = Cstr> {
    fobj: O,
    fcstrs: Vec<C>,
    config: EgorConfig,
    run_info: Option<RunInfo>,
}

impl<O: GroupFunc, C: CstrFn> EgorFactory<O, C> {
    /// Function to be minimized domain should be basically R^nx -> R^ny
    /// where nx is the dimension of input x and ny the output dimension
    /// equal to 1 (obj) + n (cstrs).
    /// But function has to be able to evaluate several points in one go
    /// hence take an (p, nx) matrix and return an (p, ny) matrix
    pub fn optimize(fobj: O) -> Self {
        EgorFactory {
            fobj,
            fcstrs: vec![],
            config: EgorConfig::default(),
            run_info: None,
        }
    }

    /// Set configuration of the optimizer
    pub fn configure<F: FnOnce(EgorConfig) -> EgorConfig>(mut self, init: F) -> Self {
        self.config = init(self.config);
        self
    }

    /// This function allows to define complex constraints on inputs using functions [CstrFn] trait.
    /// Bounds constraints are better specified using `min_within()` or `min_within_mixint_space()`
    /// arguments.
    pub fn subject_to(mut self, fcstrs: Vec<C>) -> Self {
        self.fcstrs = fcstrs;
        self
    }

    /// Set execution metadata used to qualify optimization run
    pub fn run_info(mut self, info: RunInfo) -> Self {
        self.run_info = Some(info);
        self
    }

    /// Build an Egor optimizer to minimize the function within
    /// the continuous `xlimits` specified as [[lower, upper], ...] array where the
    /// number of rows gives the dimension of the inputs (continuous optimization)
    /// and the ith row is the interval of the ith component of the input x.
    pub fn min_within(
        self,
        xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Egor<O, C, GpMixtureParams<f64>>> {
        let config = self.config.xtypes(&to_xtypes(xlimits));
        Ok(Egor {
            fobj: ObjFunc::new(self.fobj).subject_to(self.fcstrs),
            solver: EgorSolver::new(config.check()?),
            run_info: self.run_info,
        })
    }

    /// Build an Egor optimizer to minimize the function R^n -> R^p taking
    /// inputs specified with given xtypes where some of components may be
    /// discrete variables (mixed-integer optimization).
    pub fn min_within_mixint_space(
        self,
        xtypes: &[XType],
    ) -> Result<Egor<O, C, MixintGpMixtureParams>> {
        let config = self.config.xtypes(xtypes);
        Ok(Egor {
            fobj: ObjFunc::new(self.fobj).subject_to(self.fcstrs),
            solver: EgorSolver::new(config.check()?),
            run_info: self.run_info,
        })
    }
}

/// Egor optimizer structure used to parameterize the underlying `argmin::Solver`
/// and trigger the optimization using `argmin::Executor`.
#[derive(Clone)]
pub struct Egor<
    O: GroupFunc,
    C: CstrFn = Cstr,
    SB: SurrogateBuilder + DeserializeOwned = GpMixtureParams<f64>,
> {
    fobj: ObjFunc<O, C>,
    solver: EgorSolver<SB, C>,
    run_info: Option<RunInfo>,
}

impl<O: GroupFunc, C: CstrFn, SB: SurrogateBuilder + DeserializeOwned> Egor<O, C, SB> {
    /// Runs the (constrained) optimization of the objective function.
    pub fn run(&self) -> Result<OptimResult<f64>> {
        let xtypes = self.solver.config.xtypes.clone();
        info!("{:?}", self.solver.config);
        if let Some(outdir) = self.solver.config.outdir.as_ref() {
            std::fs::create_dir_all(outdir)?;
            let filepath = std::path::Path::new(outdir).join(CONFIG_FILE);
            let json = serde_json::to_string(&self.solver.config).unwrap();
            std::fs::write(filepath, json).expect("Unable to write file");
        }

        let exec = Executor::new(self.fobj.clone(), self.solver.clone()).timer(true);

        let exec = if self.solver.config.hot_start != HotStartMode::Disabled {
            let chkpt_dir = if let Some(outdir) = self.solver.config.outdir.as_ref() {
                outdir
            } else {
                ".checkpoints"
            };
            let checkpoint = HotStartCheckpoint::new(
                chkpt_dir,
                CHECKPOINT_FILE,
                CheckpointingFrequency::Always,
                self.solver.config.hot_start.clone(),
            );
            exec.checkpointing(checkpoint)
        } else {
            exec
        };

        let result = if let Some(outdir) = self.solver.config.outdir.as_ref() {
            let hist = OptimizationObserver::new(outdir.clone());
            exec.add_observer(hist, ObserverMode::Always).run()?
        } else {
            exec.run()?
        };

        info!("{result}");
        let (x_data, y_data, c_data) = result.state().clone().take_data().unwrap();

        let res = if !self.solver.config.discrete() {
            info!("Data: \n{}", concatenate![Axis(1), x_data, y_data, c_data]);
            OptimResult {
                x_opt: result.state.get_best_param().unwrap().to_owned(),
                y_opt: result.state.get_full_best_cost().unwrap().to_owned(),
                x_doe: x_data,
                y_doe: y_data,
                state: result.state,
            }
        } else {
            let x_data = to_discrete_space(&xtypes, &x_data.view());
            info!("Data: \n{}", concatenate![Axis(1), x_data, y_data, c_data]);

            let x_opt = result
                .state
                .get_best_param()
                .unwrap()
                .to_owned()
                .insert_axis(Axis(0));
            let x_opt = to_discrete_space(&xtypes, &x_opt.view());
            OptimResult {
                x_opt: x_opt.row(0).to_owned(),
                y_opt: result.state.get_full_best_cost().unwrap().to_owned(),
                x_doe: x_data,
                y_doe: y_data,
                state: result.state,
            }
        };

        #[cfg(feature = "persistent")]
        if std::env::var(crate::utils::EGOR_USE_RUN_RECORDER).is_ok() {
            use crate::utils::{EGOR_RUN_FILENAME, run_recorder};

            let default_dir = String::from("./");
            let outdir = self.solver.config.outdir.as_ref().unwrap_or(&default_dir);
            let filename = EGOR_RUN_FILENAME;
            let filepath = std::path::Path::new(outdir).join(filename);

            let mut run_data = res.state.run_data.as_ref().unwrap().clone();
            let default = RunInfo::default();
            let meta = self.run_info.as_ref().unwrap_or(&default);
            run_data.problem_metadata.test_function = meta.fname.clone();
            run_data.problem_metadata.replication_number = meta.num;

            match run_recorder::save_run(&filepath, &run_data) {
                Ok(_) => log::info!("Run data saved to {:?}", filepath),
                Err(err) => log::info!("Cannot save run data: {:?}", err),
            };
        }

        info!("Optim Result: min f(x)={} at x={}", res.y_opt, res.x_opt);

        Ok(res)
    }

    /// Set execution metadata used to qualify optimization run
    pub fn run_info(mut self, info: RunInfo) -> Self {
        self.run_info = Some(info);
        self
    }
}

// The optimization observer collects best costs ans params
// during the optimization execution allowing to get optimization history
// saved as a numpy array for further analysis
// Note: the observer is activated only when outdir is specified
#[derive(Default)]
struct OptimizationObserver {
    pub dir: PathBuf,
    pub best_params: Option<Array2<f64>>,
    pub best_costs: Option<Array2<f64>>,
}

impl OptimizationObserver {
    fn new(dir: String) -> Self {
        Self {
            dir: PathBuf::from(dir),
            best_params: None,
            best_costs: None,
        }
    }
}

impl Observe<EgorState<f64>> for OptimizationObserver {
    fn observe_iter(&mut self, state: &EgorState<f64>, _kv: &KV) -> std::result::Result<(), Error> {
        if let Some((xdata, ydata, cdata)) = &state.data {
            let doe = concatenate![Axis(1), xdata.view(), ydata.view(), cdata.view()];
            if !self.dir.exists() {
                std::fs::create_dir_all(&self.dir)?
            }

            let filepath = self.dir.join(crate::DOE_FILE);
            info!(">>> Save doe shape {:?} in {:?}", doe.shape(), filepath);
            write_npy(filepath, &doe).expect("Write current doe");

            if self.best_params.is_none() {
                // Have to initialize best params and full best costs
                let mut state = state.clone();
                state.update();
                let bp = state.get_best_param().unwrap().to_owned();
                self.best_params = Some(bp.insert_axis(Axis(0)));
                let bc = state.get_full_best_cost().unwrap().to_owned();
                self.best_costs = Some(bc.insert_axis(Axis(0)));
            } else {
                let bp = state.get_best_param().unwrap().to_owned();
                let bp = bp.insert_axis(Axis(0));
                self.best_params =
                    Some(concatenate![Axis(0), self.best_params.take().unwrap(), bp]);
                let bc = state.get_full_best_cost().unwrap().to_owned();
                let bc = bc.insert_axis(Axis(0));
                self.best_costs = Some(concatenate![Axis(0), self.best_costs.take().unwrap(), bc]);
            };
            let hist = concatenate![
                Axis(1),
                self.best_costs.clone().unwrap(),
                self.best_params.clone().unwrap(),
            ];

            let filepath = std::path::Path::new(&self.dir).join(HISTORY_FILE);
            info!(">>> Save history {:?} in {:?}", hist.shape(), filepath);
            ndarray_npy::write_npy(filepath, &hist).expect("Write current history");
        }
        Ok(())
    }
}

/// Type alias for Egor optimizer with default constraint function type [Cstr]
pub type EgorBuilder<O> = EgorFactory<O, Cstr>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use argmin_testfunctions::rosenbrock;
    use egobox_doe::{Lhs, SamplingMethod};
    use egobox_moe::NbClusters;
    use ndarray::{Array1, Array2, ArrayView2, Ix1, Zip, array, s};
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    use ndarray_npy::read_npy;

    use serial_test::serial;
    use std::time::Instant;

    use crate::{CoegoStatus, DOE_FILE, DOE_INITIAL_FILE, gpmix::spec::*, utils::EGOBOX_LOG};

    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;

    fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    #[serial]
    fn test_xsinx_ei_quadratic_egor_builder() {
        let outdir = "target/test_egor_builder_01";
        let outfile = format!("{outdir}/{DOE_INITIAL_FILE}");
        let _ = std::fs::remove_file(&outfile);
        let initial_doe = array![[0.], [7.], [25.]];
        let res = EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                cfg.infill_strategy(InfillStrategy::EI)
                    .configure_gp(|gp| {
                        gp.regression_spec(RegressionSpec::QUADRATIC)
                            .correlation_spec(CorrelationSpec::ALL)
                    })
                    .max_iters(30)
                    .doe(&initial_doe)
                    .target(-15.1)
                    .outdir(outdir)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor should be configured")
            .run()
            .expect("Egor should minimize xsinx");
        let expected = array![-15.1];
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 0.5);
        let saved_doe: Array2<f64> = read_npy(&outfile).unwrap();
        assert_abs_diff_eq!(initial_doe, saved_doe.slice(s![..3, ..1]), epsilon = 1e-6);
    }

    #[test]
    #[serial]
    fn test_gp_config() {
        let initial_doe = array![[0.], [7.], [25.]];
        const LOWER_BOUND: f64 = 1.5;
        let egor = EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                cfg.infill_strategy(InfillStrategy::EI)
                    .configure_gp(|gp| {
                        gp.theta_tuning(egobox_gp::ThetaTuning::Full {
                            init: array![2.0],
                            bounds: array![(LOWER_BOUND, 20.)],
                        })
                        .recombination(egobox_moe::Recombination::Hard)
                        .n_start(7)
                        .max_eval(100)
                    })
                    .max_iters(1)
                    .doe(&initial_doe)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor should be configured");
        let res = egor.run().expect("Egor should minimize xsinx");
        // Inspect internal state: theta init should be equal to
        // lower bound of theta interval as with a smaller bound
        // it would be around 1.28 after first iteration
        dbg!(res.state.clone());
        assert_eq!(
            res.state.theta_inits.unwrap()[0].as_ref().unwrap(),
            array![[LOWER_BOUND]]
        );
        assert_eq!(
            res.state.clusterings.unwrap()[0]
                .as_ref()
                .unwrap()
                .recombination(),
            egobox_moe::Recombination::Hard
        );
    }

    #[test]
    #[serial]
    fn test_xsinx_ei_egor_builder() {
        let initial_doe = array![[0.], [7.], [25.]];
        let res = EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                cfg.infill_strategy(InfillStrategy::EI)
                    .infill_optimizer(InfillOptimizer::Slsqp)
                    .max_iters(10)
                    .doe(&initial_doe)
                    .seed(42)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor should be configured")
            .run()
            .expect("Egor should minimize xsinx");
        let expected = array![-15.125];
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 2e-3);
    }

    #[test]
    #[serial]
    fn test_xsinx_logei_egor_builder() {
        let initial_doe = array![[0.], [7.], [25.]];
        let res = EgorBuilder::optimize(xsinx)
            .configure(|cfg| {
                cfg.infill_strategy(InfillStrategy::LogEI)
                    .infill_optimizer(InfillOptimizer::Slsqp)
                    .max_iters(30)
                    .doe(&initial_doe)
                    .seed(42)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor should be configured")
            .run()
            .expect("Egor should minimize xsinx");
        let expected = array![-15.125];
        assert_abs_diff_eq!(expected, res.y_opt, epsilon = 2e-3);
    }

    #[test]
    #[serial]
    fn test_xsinx_with_domain_constraint() {
        let outdir = "target/test_egor_builder_01";
        let outfile = format!("{outdir}/{DOE_INITIAL_FILE}");
        let _ = std::fs::remove_file(&outfile);
        let initial_doe = array![[0.], [7.], [10.]];
        let res = EgorBuilder::optimize(xsinx)
            .subject_to(vec![|x: &[f64], g: Option<&mut [f64]>, _u| {
                if let Some(g) = g {
                    g[0] = 1.
                }
                x[0] - 17.0
            }])
            .configure(|cfg| {
                cfg.infill_strategy(InfillStrategy::EI)
                    .infill_optimizer(InfillOptimizer::Cobyla)
                    .outdir(outdir)
                    .max_iters(100)
                    .doe(&initial_doe)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor should be configured")
            .run()
            .expect("Egor should minimize xsinx");
        let expected = array![17.0];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
        let saved_doe: Array2<f64> = read_npy(&outfile).unwrap();
        assert_abs_diff_eq!(
            initial_doe,
            saved_doe.slice(s![..initial_doe.nrows(), ..initial_doe.ncols()]),
            epsilon = 1e-6
        );
    }

    #[test]
    #[serial]
    fn test_xsinx_trego_wb2_egor_builder() {
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .max_iters(20)
                    .configure_gp(|gp| {
                        gp.regression_spec(RegressionSpec::ALL)
                            .correlation_spec(CorrelationSpec::ALL)
                    })
                    .trego(true)
                    .seed(1)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor should be configured")
            .run()
            .expect("Egor should minimize");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_optmod_egor() {
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| config.max_iters(20).q_optmod(3))
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor configured")
            .run()
            .expect("Egor should minimize");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_hot_start_egor() {
        let outdir = "checkpoint_test_dir";
        let checkpoint_file = format!("{outdir}/{CHECKPOINT_FILE}");
        let _ = std::fs::remove_file(&checkpoint_file);
        let n_iter = 1;
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .infill_strategy(InfillStrategy::WB2)
                    .max_iters(n_iter)
                    .seed(42)
                    .hot_start(HotStartMode::Enabled)
                    .outdir(outdir)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor configured")
            .run()
            .expect("Egor should minimize");
        let expected = array![19.1];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);

        // without hostart we reach the same point
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .infill_strategy(InfillStrategy::WB2)
                    .max_iters(n_iter)
                    .seed(42)
                    .hot_start(HotStartMode::Disabled)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor configured")
            .run()
            .expect("Egor should minimize");
        let expected = array![19.1];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);

        // with hot start we continue
        let ext_iters = 3;
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .infill_strategy(InfillStrategy::WB2)
                    .seed(42)
                    .hot_start(HotStartMode::ExtendedIters(ext_iters))
                    .outdir(outdir)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor configured")
            .run()
            .expect("Egor should minimize");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
        assert_eq!(n_iter as u64 + ext_iters, res.state.get_iter());

        // with hot start we continue... again
        let ext_iters = 3;
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .infill_strategy(InfillStrategy::WB2)
                    .seed(42)
                    .hot_start(HotStartMode::ExtendedIters(ext_iters))
                    .outdir(outdir)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor configured")
            .run()
            .expect("Egor should minimize");
        assert_eq!(n_iter as u64 + ext_iters + ext_iters, res.state.get_iter());
        let _ = std::fs::remove_file(&checkpoint_file);
    }

    #[test]
    #[serial]
    fn test_xsinx_auto_clustering_egor_builder() {
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .configure_gp(|gp| gp.n_clusters(NbClusters::auto()))
                    .max_iters(30)
            })
            .min_within(&array![[0.0, 25.0]])
            .expect("Egor configured")
            .run()
            .expect("Egor with auto clustering should minimize xsinx");
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_xsinx_with_warmstart_egor_builder() {
        let outdir = "target/test_warmstart_01";
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_INITIAL_FILE}"));
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_FILE}"));
        let xlimits = array![[0.0, 25.0]];
        let doe = array![[0.], [7.], [23.]];
        let _ = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .infill_strategy(InfillStrategy::WB2)
                    .max_iters(2)
                    .doe(&doe)
                    .outdir(outdir)
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Minimize failure");

        let filepath = std::path::Path::new(&outdir).join(DOE_FILE);
        assert!(filepath.exists());
        let doe: Array2<f64> = read_npy(&filepath).expect("file read");

        let n_iters = 3;
        let res = EgorBuilder::optimize(xsinx)
            .configure(|config| {
                config
                    .infill_strategy(InfillStrategy::WB2)
                    .max_iters(n_iters)
                    .outdir(outdir)
                    .warm_start(true)
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Egor should minimize xsinx");
        let doe2: Array2<f64> = read_npy(&filepath).expect("file read");

        assert!(doe2.nrows() >= doe.nrows() + n_iters);

        let _ = std::fs::remove_file(format!("{outdir}/{DOE_INITIAL_FILE}"));
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_FILE}"));
        let expected = array![18.9];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    fn rosenb(x: &ArrayView2<f64>) -> Array2<f64> {
        let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
        y
    }

    #[test]
    #[serial]
    fn test_rosenbrock_2d_egor_builder() {
        let now = Instant::now();
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(10);
        let res = EgorBuilder::optimize(rosenb)
            .configure(|config| {
                config
                    .doe(&doe)
                    .max_iters(50)
                    .configure_gp(|gp| {
                        gp.regression_spec(RegressionSpec::ALL)
                            .correlation_spec(CorrelationSpec::ALL)
                    })
                    .target(1e-2)
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Minimize failure");
        println!("Rosenbrock optim result = {res:?}");
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![1., 1.];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 5e-1);
    }

    #[test]
    #[serial]
    fn test_rosenbrock_2d_trego_egor_builder() {
        let outdir = "target/test_trego";
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_INITIAL_FILE}"));
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_FILE}"));
        let now = Instant::now();
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let init_doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(10);
        let max_iters = 30;
        let res = EgorBuilder::optimize(rosenb)
            .configure(|config| {
                config
                    .infill_strategy(InfillStrategy::WB2)
                    .doe(&init_doe)
                    .max_iters(max_iters)
                    .outdir(outdir)
                    .seed(42)
                    .configure_trego(|trego_cfg| trego_cfg.n_gl_steps((4, 1)))
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Minimize failure");
        let filepath = std::path::Path::new(&outdir).join(DOE_FILE);
        assert!(filepath.exists());
        let doe: Array2<f64> = read_npy(&filepath).expect("file read");
        assert!(doe.nrows() >= init_doe.nrows());

        println!("Rosenbrock optim result = {res:?}");
        println!("Elapsed = {:?}", now.elapsed());
        let expected = array![1., 1.];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 5e-1);
    }

    fn sphere(x: &ArrayView2<f64>) -> Array2<f64> {
        let s = (x * x).sum_axis(Axis(1));
        s.insert_axis(Axis(1))
    }

    #[test]
    #[serial]
    fn test_sphere_coego_egor_builder() {
        let outdir = "target/test_coego";
        let dim = 8;
        let xlimits = Array2::from_shape_vec((dim, 2), [-10.0, 10.0].repeat(dim)).unwrap();
        let init_doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(0))
            .sample(dim + 1);
        let max_iters = 70;
        let res = EgorBuilder::optimize(sphere)
            .configure(|config| {
                config
                    .doe(&init_doe)
                    .max_iters(max_iters)
                    .outdir(outdir)
                    .seed(42)
                    .coego(CoegoStatus::Enabled(5))
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Minimize failure");

        println!("Sphere optim result = {res:?}");
        let expected = Array1::<f64>::zeros(dim);
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 6e-1);
    }

    // Objective
    fn g24(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        // Function G24: 1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
        -x[0] - x[1]
    }

    // Constraints < 0
    fn g24_c1(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        -2.0 * x[0].powf(4.0) + 8.0 * x[0].powf(3.0) - 8.0 * x[0].powf(2.0) + x[1] - 2.0
    }

    fn g24_c2(x: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64 {
        -4.0 * x[0].powf(4.0) + 32.0 * x[0].powf(3.0) - 88.0 * x[0].powf(2.0) + 96.0 * x[0] + x[1]
            - 36.0
    }

    fn f_g24(x: &ArrayView2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros((x.nrows(), 3));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .for_each(|mut yi, xi| {
                yi.assign(&array![g24(&xi), g24_c1(&xi), g24_c2(&xi)]);
            });
        y
    }

    fn f_g24_bare(x: &ArrayView2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .for_each(|mut yi, xi| {
                yi.assign(&array![g24(&xi)]);
            });
        y
    }

    #[test]
    #[serial]
    fn test_egor_g24_basic_egor_builder_cobyla() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(0))
            .sample(3);
        let res = EgorBuilder::optimize(f_g24)
            .configure(|config| {
                config
                    .infill_strategy(InfillStrategy::WB2)
                    .n_cstr(2)
                    .doe(&doe)
                    .max_iters(20)
                    .infill_optimizer(InfillOptimizer::Cobyla)
                    .cstr_tol(array![2e-3, 1e-3])
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Minimize failure");
        println!("G24 optim result = {res:?}");
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 3e-2);
    }

    #[test]
    fn test_egor_g24_basic_egor_builder_slsqp() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(3);
        let res = EgorBuilder::optimize(f_g24)
            .configure(|config| {
                config
                    .n_cstr(2)
                    .doe(&doe)
                    .max_iters(20)
                    .cstr_tol(array![1e-5, 1e-5])
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Minimize failure");
        println!("G24 optim result = {res:?}");
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 3e-2);
    }

    #[test]
    #[serial]
    fn test_egor_g24_basic_egor_builder_logei() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(0))
            .sample(3);
        let res = EgorBuilder::optimize(f_g24)
            .configure(|config| {
                config
                    .n_cstr(2)
                    .doe(&doe)
                    .max_iters(20)
                    .infill_strategy(InfillStrategy::LogEI)
                    .infill_optimizer(InfillOptimizer::Slsqp)
                    //.cstr_infill(true)
                    .cstr_tol(array![2e-3, 1e-3])
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Minimize failure");
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 3e-2);
    }

    #[test]
    #[serial]
    fn test_egor_g24_with_domain_constraints() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(3);
        let c1 = |x: &[f64], g: Option<&mut [f64]>, _u: &mut InfillObjData<f64>| {
            if g.is_some() {
                panic!("c1: gradient not implemented") // ie panic with InfillOptimizer::Slsqp
            }
            g24_c1(&Array1::from_vec(x.to_vec()))
        };
        let c2 = |x: &[f64], g: Option<&mut [f64]>, _u: &mut InfillObjData<f64>| {
            if g.is_some() {
                panic!("c2:  gradient not implemented")
            }
            g24_c2(&Array1::from_vec(x.to_vec()))
        };
        let res = EgorBuilder::optimize(f_g24_bare)
            .subject_to(vec![c1, c2])
            .configure(|config| {
                config
                    .doe(&doe)
                    .max_iters(50)
                    .infill_optimizer(InfillOptimizer::Cobyla)
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Minimize failure");
        println!("G24 optim result = {res:?}");
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 1e-1);
    }

    #[test]
    #[serial]
    fn test_egor_g24_qei_egor_builder() {
        let xlimits = array![[0., 3.], [0., 4.]];
        let doe = Lhs::new(&xlimits)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(10);
        let q = 2;
        let res = EgorBuilder::optimize(f_g24)
            .configure(|config| {
                config
                    .configure_gp(|gp| {
                        gp.regression_spec(RegressionSpec::ALL)
                            .correlation_spec(CorrelationSpec::ALL)
                    })
                    .n_cstr(2)
                    .cstr_tol(array![2e-6, 2e-6])
                    .q_points(q)
                    .qei_strategy(QEiStrategy::KrigingBeliever)
                    .doe(&doe)
                    .target(-5.5030)
                    .max_iters(20)
                    .seed(42)
            })
            .min_within(&xlimits)
            .expect("Egor configured")
            .run()
            .expect("Egor minimization");
        assert_eq!(res.x_doe.nrows(), doe.nrows() + q * res.state.iter as usize);
        println!("G24 optim result = {res:?}");
        let expected = array![2.3295, 3.1785];
        assert_abs_diff_eq!(expected, res.x_opt, epsilon = 2e-2);
    }

    // Mixed-integer tests

    fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        if (x.mapv(|v| v.round()).norm_l2() - x.norm_l2()).abs() < 1e-6 {
            (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
        } else {
            panic!("Error: mixsinx works only on integer, got {x:?}")
        }
    }

    #[test]
    #[serial]
    fn test_mixsinx_ei_mixint_egor_builder() {
        let max_iters = 20;
        let doe = array![[0.], [7.], [25.]];
        let xtypes = vec![XType::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .configure(|config| {
                config
                    .doe(&doe)
                    .max_iters(max_iters)
                    .target(-15.1)
                    .infill_strategy(InfillStrategy::EI)
                    .seed(42)
            })
            .min_within_mixint_space(&xtypes)
            .expect("Egor configured")
            .run()
            .expect("Optimization successful");
        assert_abs_diff_eq!(array![18.], res.x_opt, epsilon = 2.);
    }

    #[test]
    #[serial]
    fn test_mixsinx_reclustering_mixint_egor_builder() {
        let max_iters = 20;
        let doe = array![[0.], [7.], [25.]];
        let xtypes = vec![XType::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .configure(|config| {
                config
                    .doe(&doe)
                    .max_iters(max_iters)
                    .target(-15.1)
                    .infill_strategy(InfillStrategy::EI)
                    .seed(42)
            })
            .min_within_mixint_space(&xtypes)
            .expect("Egor configured")
            .run()
            .unwrap();
        assert_abs_diff_eq!(array![18.], res.x_opt, epsilon = 2.);
    }

    #[test]
    #[serial]
    fn test_mixsinx_wb2_mixint_egor_builder() {
        let max_iters = 20;
        let xtypes = vec![XType::Int(0, 25)];

        let res = EgorBuilder::optimize(mixsinx)
            .configure(|config| {
                config
                    .configure_gp(|gp| {
                        gp.regression_spec(RegressionSpec::CONSTANT)
                            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
                    })
                    .max_iters(max_iters)
                    .seed(42)
            })
            .min_within_mixint_space(&xtypes)
            .expect("Egor configured")
            .run()
            .unwrap();
        assert_abs_diff_eq!(&array![18.], &res.x_opt, epsilon = 3.);
    }

    fn mixobj(x: &ArrayView2<f64>) -> Array2<f64> {
        // XType.Float
        let x1 = x.column(0);
        // XType.ENUM 1
        let c1 = x.column(1);
        let x2 = c1.mapv(|v| (v == 0.) as i32 as f64);
        let x3 = c1.mapv(|v| (v == 1.) as i32 as f64);
        let x4 = c1.mapv(|v| (v == 2.) as i32 as f64);
        // XType.ENUM 2
        let c2 = x.column(2);
        let x5 = c2.mapv(|v| (v == 0.) as i32 as f64);
        let x6 = c2.mapv(|v| (v == 1.) as i32 as f64);
        // XTypes.ORD
        let i = x.column(3);

        let y = (x2.clone() + x3.mapv(|v| 2. * v) + x4.mapv(|v| 3. * v)) * x5 * x1
            + (x2 + x3.mapv(|v| 2. * v) + x4.mapv(|v| 3. * v)) * x6 * 0.95 * x1
            + i;
        let d = y.len();
        y.into_shape_with_order((d, 1)).unwrap()
    }

    #[test]
    #[serial]
    fn test_mixobj_mixint_egor_builder() {
        let env = env_logger::Env::new().filter_or(EGOBOX_LOG, "info");
        let mut builder = env_logger::Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();
        let max_iters = 10;
        let xtypes = vec![
            XType::Float(-5., 5.),
            XType::Enum(3),
            XType::Enum(2),
            XType::Ord(vec![0., 2., 3.]),
        ];

        let res = EgorBuilder::optimize(mixobj)
            .configure(|config| {
                config
                    .configure_gp(|gp| {
                        gp.regression_spec(RegressionSpec::CONSTANT)
                            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
                    })
                    .max_iters(max_iters)
                    .seed(42)
            })
            .min_within_mixint_space(&xtypes)
            .expect("Egor configured")
            .run()
            .unwrap();
        println!("res={res:?}");
        assert_abs_diff_eq!(&array![-15.], &res.y_opt, epsilon = 1.);
    }

    #[test]
    #[serial]
    fn test_mixobj_mixint_warmstart_egor_builder() {
        let env = env_logger::Env::new().filter_or(EGOBOX_LOG, "info");
        let mut builder = env_logger::Builder::from_env(env);
        let builder = builder.target(env_logger::Target::Stdout);
        builder.try_init().ok();

        let outdir = "target/test_warmstart_02";
        let outfile = format!("{outdir}/{DOE_INITIAL_FILE}");
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_INITIAL_FILE}"));
        let _ = std::fs::remove_file(format!("{outdir}/{DOE_FILE}"));

        let xtypes = vec![
            XType::Float(-5., 5.),
            XType::Enum(3),
            XType::Enum(2),
            XType::Ord(vec![0., 2., 3.]),
        ];
        let xlimits = as_continuous_limits::<f64>(&xtypes);

        EgorBuilder::optimize(mixobj)
            .configure(|config| config.outdir(outdir).max_iters(1).seed(42))
            .min_within_mixint_space(&xtypes)
            .expect("Egor configured")
            .run()
            .unwrap();

        // Check saved DOE has continuous (unfolded) dimension + one output
        let saved_doe: Array2<f64> = read_npy(outfile).unwrap();
        assert_eq!(saved_doe.shape()[1], xlimits.nrows() + 1);

        // Check that with no iteration, obj function is never called
        // as the DOE does not need to be evaluated!
        EgorBuilder::optimize(|_x| panic!("Should not call objective function!"))
            .configure(|config| config.outdir(outdir).warm_start(true).max_iters(0).seed(42))
            .min_within_mixint_space(&xtypes)
            .expect("Egor configured")
            .run()
            .unwrap();
    }
}

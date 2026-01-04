//! This library implements Efficient Global Optimization method,
//! it started as a port of the [EGO algorithm](https://smt.readthedocs.io/en/stable/_src_docs/applications/ego.html)
//! implemented as an application example in [SMT](https://smt.readthedocs.io/en/stable).
//!
//! The optimizer is able to deal with inequality constraints.
//! Objective and contraints are expected to computed grouped at the same time
//! hence the given function should return a vector where the first component
//! is the objective value and the remaining ones constraints values intended
//! to be negative in the end.   
//! The optimizer comes with a set of options to:
//! * specify the initial doe,
//! * parameterize internal optimization,
//! * parameterize mixture of experts,
//! * save intermediate results and allow warm/hot restart,
//! * handling of mixed-integer variables
//! * activation of TREGO algorithm variation
//!
//! # Examples
//!
//! ## Continuous optimization
//!
//! ```
//! use ndarray::{array, Array2, ArrayView2};
//! use egobox_ego::EgorBuilder;
//!
//! // A one-dimensional test function, x in [0., 25.] and min xsinx(x) ~ -15.1 at x ~ 18.9
//! fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
//!     (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
//! }
//!
//! // We ask for 10 evaluations of the objective function to get the result
//! let res = EgorBuilder::optimize(xsinx)
//!             .configure(|config| config.max_iters(10))
//!             .min_within(&array![[0.0, 25.0]])
//!             .expect("optimizer configured")
//!             .run()
//!             .expect("xsinx minimized");
//! println!("Minimum found f(x) = {:?} at x = {:?}", res.x_opt, res.y_opt);
//! ```
//!
//! The implementation relies on [Mixture of Experts](egobox_moe).
//!
//!
//! ## Mixed-integer optimization
//!
//! While [Egor] optimizer works with continuous data (i.e floats), the optimizer
//! allows to make basic mixed-integer optimization. The configuration of the Optimizer
//! as a mixed_integer optimizer is done though the `EgorBuilder`  
//!
//! As a second example, we define an objective function `mixsinx` taking integer
//! input values from the previous function `xsinx` defined above.
//!  
//! ```
//! use ndarray::{array, Array2, ArrayView2};
//! use linfa::ParamGuard;
//! #[cfg(feature = "blas")]
//! use ndarray_linalg::Norm;
//! #[cfg(not(feature = "blas"))]
//! use linfa_linalg::norm::*;
//! use egobox_ego::{EgorBuilder, InfillStrategy, XType};
//!
//! fn mixsinx(x: &ArrayView2<f64>) -> Array2<f64> {
//!     if (x.mapv(|v| v.round()).norm_l2() - x.norm_l2()).abs() < 1e-6 {
//!         (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
//!     } else {
//!         panic!("Error: mixsinx works only on integer, got {:?}", x)
//!     }
//! }
//!
//! let max_iters = 10;
//! let doe = array![[0.], [7.], [25.]];   // the initial doe
//!
//! // We define input as being integer
//! let xtypes = vec![XType::Int(0, 25)];
//!
//! let res = EgorBuilder::optimize(mixsinx)
//!     .configure(|config|
//!         config.doe(&doe)  // we pass the initial doe
//!               .max_iters(max_iters)
//!               .infill_strategy(InfillStrategy::EI)
//!               .seed(42))     
//!     .min_within_mixint_space(&xtypes)  // We build a mixed-integer optimizer
//!     .expect("optimizer configured")
//!     .run()
//!     .expect("Egor minimization");
//! println!("min f(x)={} at x={}", res.y_opt, res.x_opt);
//! ```  
//!
//! # Usage
//!
//! The [`EgorBuilder`] class is used to build an initial optimizer setting
//! the objective function, an optional random seed (to get reproducible runs) and
//! a design space specifying the domain and dimensions of the inputs `x`.
//!  
//! The `min_within()` and `min_within_mixed_space()` methods return an [`Egor`] object, the optimizer,
//! which can be further configured.
//! The first one is used for continuous input space (eg floats only), the second one for mixed-integer input
//! space (some variables, components of `x`, may be integer, ordered or categorical).
//!
//! Some of the most useful options are:
//!
//! * Specification of the size of the initial DoE. The default is nx+1 where nx is the dimension of x.
//!   If your objective function is not expensive you can take `3*nx` to help the optimizer
//!   approximating your objective function.
//!
//! ```no_run
//! # use egobox_ego::{EgorConfig};
//! # let egor_config = EgorConfig::default();
//!     egor_config.n_doe(100);
//! ```
//!
//! You can also provide your initial doe though the `egor.doe(your_doe)` method.
//!
//! * Specifications of constraints (expected to be negative at the end of the optimization)
//!   In this example below we specify that 2 constraints will be computed with the objective values meaning
//!   the objective function is expected to return an array '\[nsamples, 1 obj value + 2 const values\]'.
//!
//! ```no_run
//! # let egor_config = egobox_ego::EgorConfig::default();
//!     egor_config.n_cstr(2);
//! ```
//!
//! * If the default infill strategy (WB2, Watson and Barnes 2nd criterion),
//!   you can switch for either EI (Expected Improvement) or WB2S (scaled version of WB2).
//!   See \[[Priem2019](#Priem2019)\]
//!
//! ```no_run
//! # use egobox_ego::{EgorConfig, InfillStrategy};
//! # let egor_config = EgorConfig::default();
//!     egor_config.infill_strategy(InfillStrategy::EI);
//! ```
//!
//! * Constraints modeled with a surrogate can be integrated in the infill criterion
//!   through their probability of feasibility. See \[[Sasena2002](#Sasena2002)\]
//!
//! ```no_run
//! # use egobox_ego::{EgorConfig};
//! # let egor_config = EgorConfig::default();
//!     egor_config.cstr_infill(true);
//! ```
//!
//! * Constraints modeled with a surrogate can be used with their mean value or their upper trust bound
//!   See \[[Priem2019](#Priem2019)\]
//!
//! ```no_run
//! # use egobox_ego::{EgorConfig, ConstraintStrategy};
//! # let egor_config = EgorConfig::default();
//!     egor_config.cstr_strategy(ConstraintStrategy::UpperTrustBound);
//! ```
//!
//! * The default gaussian process surrogate is parameterized with a constant trend and a squared exponential correlation kernel, also
//!   known as Kriging. The optimizer use such surrogates to approximate objective and constraint functions. The kind of surrogate
//!   can be changed using `regression_spec` and `correlation_spec()` methods to specify trend and kernels tested to get the best
//!   approximation (quality tested through cross validation).
//!
//! ```no_run
//! # use egobox_ego::EgorConfig;
//! # use egobox_ego::{GpConfig, RegressionSpec, CorrelationSpec};
//! # let egor_config = EgorConfig::default();
//!     egor_config.configure_gp(|gp_conf| {
//!         gp_conf.regression_spec(RegressionSpec::CONSTANT | RegressionSpec::LINEAR)
//!                .correlation_spec(CorrelationSpec::MATERN32 | CorrelationSpec::MATERN52)
//!     });
//! ```
//! * As the dimension increase the gaussian process surrogate building may take longer or even fail
//!   in this case you can specify a PLS dimension reduction \[[Bartoli2019](#Bartoli2019)\].
//!   Gaussian process will be built using the `ndim` (usually 3 or 4) main components in the PLS projected space.
//!
//! ```no_run
//! # use egobox_ego::EgorConfig;
//! # use egobox_ego::GpConfig;
//! # let egor_config = EgorConfig::default();
//!     egor_config.configure_gp(|gp_conf| {
//!         gp_conf.kpls(3)
//!     });
//! ```
//!
//! In the above example all GP with combinations of regression and correlation will be tested and the best combination for
//! each modeled function will be retained. You can also simply specify `RegressionSpec::ALL` and `CorrelationSpec::ALL` to
//! test all available combinations but remember that the more you test the slower it runs.
//!
//! * the TREGO algorithm described in \[[Diouane2023](#Diouane2023)\] activated with
//!   the default gl1-4 configuration from the reference paper
//!
//! ```no_run
//! # use egobox_ego::{EgorConfig, RegressionSpec, CorrelationSpec};
//! # let egor_config = EgorConfig::default();
//!     egor_config.trego(true);
//! ```
//!
//! or with a custom configuration, here gl4-1 and beta=0.8
//!
//! ```no_run
//! # use egobox_ego::{EgorConfig, RegressionSpec, CorrelationSpec};
//! # let egor_config = EgorConfig::default();
//!    egor_config.configure_trego(|trego_cfg| trego_cfg.n_gl_steps((4, 1)).beta(0.8));
//! ```
//!
//! * Intermediate results can be logged at each iteration when `outdir` directory is specified.
//!   The following files :
//!   * egor_config.json: Egor configuration,
//!   * egor_initial_doe.npy: initial DOE (x, y) as numpy array,
//!   * egor_doe.npy: DOE (x, y) as numpy array,
//!   * egor_history.npy: best (x, y) wrt to iteration number as (n_iters, nx + ny) numpy array   
//!  
//! ```no_run
//! # use egobox_ego::EgorConfig;
//! # let egor_config = EgorConfig::default();
//!     egor_config.outdir("./.output");  
//! ```
//! If warm_start is set to `true`, the algorithm starts from the saved `egor_doe.npy`
//!
//! * Hot start checkpointing can be enabled with `hot_start` option specifying a number of
//!   extra iterations beyond max iters. This mechanism allows to restart after an interruption
//!   from the last saved checkpoint. While warm_start restart from saved doe for another max_iters
//!   iterations, hot start allows to continue from the last saved optimizer state till max_iters
//!   is reached with optinal extra iterations.
//!
//! ```no_run
//! # use egobox_ego::{EgorConfig, HotStartMode};
//! # let egor_config = EgorConfig::default();
//!     egor_config.hot_start(HotStartMode::Enabled);
//! ```
//!
//! # Implementation notes
//!
//! * Mixture of experts and PLS dimension reduction is explained in \[[Bartoli2019](#Bartoli2019)\]
//! * Parallel evaluation is available through the selection of a qei strategy. See in \[[Ginsbourger2010](#Ginsbourger2010)\]
//! * Mixed integer approach is implemented using continuous relaxation. See \[[Garrido2018](#Garrido2018)\]
//! * TREGO algorithm is implemented. See \[[Diouane2023](#Diouane2023)\]
//! * CoEGO approach is implemented with CCBO setting where expensive evaluations are run after context vector update.
//!   See \[[Zhan2024](#Zhan024)\] and \[[Pretsch2024](#Pretsch2024)\]
//! * Theta bounds are implemented as in \[[Appriou2023](#Appriou2023)\]
//! * Logirithm of Expected Improvement is implemented as in \[[Ament2025](#Ament2025)\]
//!
//! # References
//!
//! \[<a id="Bartoli2019">Bartoli2019</a>\]: Bartoli, Nathalie, et al. [Adaptive modeling strategy for constrained global
//! optimization with application to aerodynamic wing design](https://www.sciencedirect.com/science/article/pii/S1270963818306011)
//!  Aerospace Science and technology 90 (2019): 85-102.
//!
//! \[<a id="Ginsbourger2010">Ginsbourger2010</a>\]: Ginsbourger, D., Le Riche, R., & Carraro, L. (2010).
//! Kriging is well-suited to parallelize optimization.
//!  
//! \[<a id="Garrido2018">Garrido2018</a>\]: E.C. Garrido-Merchan and D. Hernandez-Lobato. Dealing with categorical and
//! integer-valued variables in Bayesian Optimization with Gaussian processes.
//!
//! Bouhlel, M. A., Bartoli, N., Otsmane, A., & Morlier, J. (2016). [Improving kriging surrogates
//! of high-dimensional design models by partial least squares dimension reduction.](https://doi.org/10.1007/s00158-015-1395-9)
//! Structural and Multidisciplinary Optimization, 53(5), 935–952.
//!
//! Bouhlel, M. A., Hwang, J. T., Bartoli, N., Lafage, R., Morlier, J., & Martins, J. R. R. A.
//! (2019). [A python surrogate modeling framework with derivatives](https://doi.org/10.1016/j.advengsoft.2019.03.005).
//! Advances in Engineering Software, 102662.
//!
//! Dubreuil, S., Bartoli, N., Gogu, C., & Lefebvre, T. (2020). [Towards an efficient global multi-
//! disciplinary design optimization algorithm](https://doi.org/10.1007/s00158-020-02514-6).
//! Structural and Multidisciplinary Optimization, 62(4), 1739–1765.
//!
//! Jones, D. R., Schonlau, M., & Welch, W. J. (1998). [Efficient global optimization of expensive
//! black-box functions](https://www.researchgate.net/publication/235709802_Efficient_Global_Optimization_of_Expensive_Black-Box_Functions).
//! Journal of Global Optimization, 13(4), 455–492.
//!
//! \[<a id="Diouane2023">Diouane(2023)</a>\]: Diouane, Youssef, et al.
//! [TREGO: a trust-region framework for efficient global optimization](https://arxiv.org/pdf/2101.06808)
//! Journal of Global Optimization 86.1 (2023): 1-23.
//!
//! \[<a id="Priem2019">Priem2019</a>\]: Priem, Rémy, Nathalie Bartoli, and Youssef Diouane.
//! [On the use of upper trust bounds in constrained Bayesian optimization infill criteria](https://hal.science/hal-02182492v1/file/Priem_24049.pdf).
//! AIAA aviation 2019 forum. 2019.
//!
//! \[<a id="Sasena2002">Sasena2002</a>\]: Sasena M., Papalambros P., Goovaerts P., 2002.
//! [Global optimization of problems with disconnected feasible regions via surrogate modeling](https://deepblue.lib.umich.edu/handle/2027.42/77089). AIAA Paper.
//!
//! \[<a id="Ginsbourger2010">Ginsbourger2010</a>\]: Ginsbourger, D., Le Riche, R., & Carraro, L. (2010).
//! [Kriging is well-suited to parallelize optimization](https://www.researchgate.net/publication/226716412_Kriging_Is_Well-Suited_to_Parallelize_Optimization).
//!
//! \[<a id="Garrido2018">Garrido2018</a>\]: E.C. Garrido-Merchan and D. Hernandez-Lobato.
//! [Dealing with categorical and integer-valued variables in Bayesian Optimization with Gaussian processes](https://arxiv.org/pdf/1805.03463).
//!
//! \[<a id="Zhan2024">Zhan2024</a>\]: Zhan, Dawei, et al.
//! [A cooperative approach to efficient global optimization](https://link.springer.com/article/10.1007/s10898-023-01316-6).
//! Journal of Global Optimization 88.2 (2024): 327-357
//!
//! \[<a id="Pretsch2024">Pretsch2024</a>\]: Lisa Pretsch et al.
//! [Bayesian optimization of cooperative components for multi-stage aero-structural compressor blade design](https://www.researchgate.net/publication/391492598_Bayesian_optimization_of_cooperative_components_for_multi-stage_aero-structural_compressor_blade_design).
//! Struct Multidisc Optim 68, 84 (2025)
//!
//! \[<a id="Appriou2023">Appriou2023</a>\]: Appriou, T., Rullière, D. & Gaudrie, D,
//! [Combination of optimization-free kriging models for high-dimensional problems](https://doi.org/10.1007/s00180-023-01424-7),
//! Comput Stat 39, 3049–3071 (2024).
//!
//! \[<a id="Ament2025">Ament2025</a>\]: S Ament, S Daulton, D Eriksson, M Balandat, E Bakshy,
//! [Unexpected improvements to expected improvement for bayesian optimization](https://arxiv.org/pdf/2310.20708),
//! Advances in Neural Information Processing Systems, 2023
//!
//! smtorg. (2018). Surrogate modeling toolbox. In [GitHub repository](https://github.com/SMTOrg/smt)
//!
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

pub mod criteria;
pub mod gpmix;

mod egor;
mod errors;
mod solver;
mod types;

pub use crate::egor::*;
pub use crate::errors::*;
pub use crate::gpmix::spec::{CorrelationSpec, RegressionSpec};
pub use crate::solver::*;
pub use crate::types::*;
pub use crate::utils::{
    CHECKPOINT_FILE, Checkpoint, CheckpointingFrequency, EGOBOX_LOG, EGOR_GP_FILENAME,
    EGOR_INITIAL_GP_FILENAME, EGOR_USE_GP_RECORDER, EGOR_USE_GP_VAR_PORTFOLIO,
    EGOR_USE_MAX_PROBA_OF_FEASIBILITY, HotStartCheckpoint, HotStartMode, find_best_result_index,
};

mod optimizers;
mod utils;

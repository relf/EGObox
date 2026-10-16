//! This library implements the efficient global multidisciplinary design optimization
//! algorithm (EGMDO) and its constrained extension (C-EGMDO).
//!
//! In a multidisciplinary design optimization (MDO) problem, the objective and constraints
//! depend on coupling variables computed by disciplinary solvers which depend on each other.
//! EGMDO replaces each disciplinary solver by Gaussian process (GP) surrogates and enriches
//! them adaptively, only where the optimum is likely to be.
//!
//! The crate is under development. It currently provides:
//! * the definition of MDO problems: [MdoProblem] and [Discipline],
//! * the EGMDO and C-EGMDO algorithms, for problems with or without constraints, as an
//!   ask/tell solver [EgmdoSolver] or with given disciplinary solvers: [minimize],
//! * the disciplinary GP surrogates: [DisciplinaryModels],
//! * the multidisciplinary analysis (MDA) solver: [mda],
//! * the polynomial chaos expansions (PCE) with Sobol indices: [pce],
//! * the uncertainty quantification of the objective and constraints at design points, with
//!   solved or linearized random MDAs: [uq],
//! * the Karhunen–Loève decomposition and KL-GP random field of the objective: [kl],
//! * the enrichment criteria: [criteria], and the constraint models: [constraints],
//! * the test problems of the EGMDO and C-EGMDO papers: [problems].
//!
//! # Examples
//!
//! Minimize the 1-D toy problem of the EGMDO paper:
//!
//! ```
//! use egobox_egmdo::{EgmdoConfig, minimize, problems};
//!
//! let tp = problems::toy();
//! let config = EgmdoConfig {
//!     n_doe_uq: 6,
//!     max_iters: 3,
//!     n_sim: 20,
//!     seed: Some(42),
//!     ..EgmdoConfig::default()
//! };
//! let result = minimize(&tp.problem, &tp.disciplines, config).unwrap();
//! println!(
//!     "z* = {} ± {}, f* = {}, true evaluations {:?}",
//!     result.z_star_mean(),
//!     result.z_star_std(),
//!     result.y_min_mean(),
//!     result.n_evaluations
//! );
//! ```
//!
//! Use the disciplinary GPs and the MDA solver directly:
//!
//! ```
//! use egobox_egmdo::{DisciplinaryModels, GpSettings, initial_doe, problems};
//! use egobox_egmdo::mda::{SurrogateCouplings, TrueCouplings, solve_one};
//! use ndarray::array;
//!
//! let tp = problems::toy();
//!
//! // Train the disciplinary GPs on initial designs of experiments
//! let data = (0..2)
//!     .map(|i| {
//!         let x = initial_doe(&tp.problem, i, 10, 42);
//!         let y = tp.eval_discipline(i, &x);
//!         (x, y)
//!     })
//!     .collect();
//! let models = DisciplinaryModels::fit(&tp.problem, data, GpSettings::default()).unwrap();
//!
//! // Compare the MDA solved with the GP means and with the true disciplines
//! let z = array![-3.];
//! let mean = SurrogateCouplings::mean(&tp.problem, &models);
//! let (y_mean, _) = solve_one(&mean, &z.view(), &tp.mda).unwrap();
//! let truth = TrueCouplings::new(&tp.problem, &tp.disciplines).unwrap();
//! let (y_true, _) = solve_one(&truth, &z.view(), &tp.mda).unwrap();
//! println!("mean MDA {y_mean}, true MDA {y_true}");
//! ```
//!
//! # References
//!
//! S. Dubreuil, N. Bartoli, C. Gogu, T. Lefebvre, *Towards an efficient global
//! multidisciplinary design optimization algorithm*, Struct. Multidisc. Optim. 62:1739–1765, 2020.
//!
//! I. Cardoso, S. Dubreuil, N. Bartoli, C. Gogu, M. Salaün, *Constrained efficient global
//! multidisciplinary design optimization using adaptive disciplinary surrogate enrichment*,
//! Struct. Multidisc. Optim. 67:23, 2024.
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

pub mod constraints;
pub mod criteria;
mod errors;
pub mod kl;
pub mod mda;
pub mod pce;
mod problem;
pub mod problems;
mod solver;
mod surrogates;
pub mod uq;

pub use errors::*;
pub use problem::*;
pub use solver::*;
pub use surrogates::*;

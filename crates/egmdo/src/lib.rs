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
//! * the disciplinary GP surrogates: [DisciplinaryModels],
//! * the multidisciplinary analysis (MDA) solver: [mda],
//! * the test problems of the EGMDO and C-EGMDO papers: [problems].
//!
//! # Example
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

mod errors;
pub mod mda;
mod problem;
pub mod problems;
mod surrogates;

pub use errors::*;
pub use problem::*;
pub use surrogates::*;

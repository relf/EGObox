mod bounds;
mod cstr_pof;
mod find_result;
mod hot_start;
mod logei_helper;
mod misc;
mod portfolio;
mod sort_axis;
mod start_points;

pub(crate) mod gp_recorder;
pub(crate) mod logging;
pub(crate) mod run_recorder;

pub use bounds::*;
pub use cstr_pof::*;
pub use find_result::*;
pub use hot_start::*;
pub use logei_helper::*;
pub use misc::*;
pub use portfolio::*;
pub use start_points::*;

/// Env variable to enable logging feature
pub const EGOBOX_LOG: &str = "EGOBOX_LOG";

/// Env variable to enable the use of PoF as criterion while no feasible point is found
pub const EGOR_USE_MAX_PROBA_OF_FEASIBILITY: &str = "EGOR_USE_MAX_PROBA_OF_FEASIBILITY";

/// Env variable to disable the use of the middle-picker multistarter method for global infill criterion optimization
pub const EGOR_DO_NOT_USE_MIDDLEPICKER_MULTISTARTER: &str =
    "EGOR_DO_NOT_USE_MIDDLEPICKER_MULTISTARTER";

/// Env variable to enable the portfolio method used for global infill criterion optimization
pub const EGOR_USE_GP_VAR_PORTFOLIO: &str = "EGOR_USE_GP_VAR_PORTFOLIO";

/// Env variable to trigger GP recording
pub const EGOR_USE_GP_RECORDER: &str = "EGOR_USE_GP_RECORDER";

/// Gaussian process filename to save initial GPs built from initial_doe
pub const EGOR_INITIAL_GP_FILENAME: &str = "egor_initial_gp.bin";

/// Gaussian process filename to save GPs built at the last iteration
pub const EGOR_GP_FILENAME: &str = "egor_gp.bin";

/// Env variable to trigger run recording
pub const EGOR_USE_RUN_RECORDER: &str = "EGOR_USE_RUN_RECORDER";

/// BO run filename
pub const EGOR_RUN_FILENAME: &str = "egor_run.json";

use serde::{Deserialize, Serialize};

/// Solver-aware checkpoint file used by the basin executor for hot start.
pub const CHECKPOINT_FILE: &str = "egor_checkpoint.bin";

/// An enum to specify hot start mode
#[derive(Clone, Eq, PartialEq, Debug, Hash, Default, Serialize, Deserialize)]
pub enum HotStartMode {
    /// Hot start checkpoints are not saved
    #[default]
    Disabled,
    /// Hot start checkpoints are saved and optionally used if it already exists
    Enabled,
    /// Hot start checkpoints are saved and optionally used if it already exists
    /// and optimization is run with an extended iteration budget
    ExtendedIters(u64),
}

impl std::convert::From<Option<u64>> for HotStartMode {
    fn from(value: Option<u64>) -> Self {
        if let Some(ext_iters) = value {
            if ext_iters == 0 {
                HotStartMode::Enabled
            } else {
                HotStartMode::ExtendedIters(ext_iters)
            }
        } else {
            HotStartMode::Disabled
        }
    }
}

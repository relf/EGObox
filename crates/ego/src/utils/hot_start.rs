use basin::core::checkpoint::{ExactCheckpoint, ExactCheckpointWriter, read_exact_checkpoint};
use basin::core::observer::ObserverMode;
use log::info;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::PathBuf;

use crate::EgorState;

/// Checkpoint file using basin's solver-aware exact-checkpoint format.
pub const CHECKPOINT_FILE: &str = "egor_checkpoint.bin";

/// basin doesn't have argmin's `CheckpointingFrequency` enum; its `ObserverMode`
/// plays the same role (it gates checkpoint/observer firing after completed
/// iterations; every sink also always saves once on a clean stop). Re-exported
/// under the old name so downstream code/imports don't need to change.
pub type CheckpointingFrequency = ObserverMode;

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

/// Handles saving/loading a solver-aware checkpoint to/from disk.
///
/// Under argmin, this type implemented `argmin::core::checkpointing::Checkpoint`
/// itself, hand-writing `(solver, state)` as JSON via `serde_json`. basin ships an
/// equivalent, more robust mechanism natively
/// ([`ExactCheckpointWriter`]/[`read_exact_checkpoint`], attached to the
/// `Executor` via `.checkpoint_with(...)` and read back for
/// `Executor::resume_from_checkpoint(...)`), so this type is now a thin wrapper
/// configuring that mechanism rather than an implementation of a trait itself.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct HotStartCheckpoint {
    /// Extended iteration number
    pub mode: HotStartMode,
    /// Indicates how often a checkpoint is created
    pub frequency: CheckpointingFrequency,
    /// Directory where the checkpoints are saved to
    pub directory: PathBuf,
    /// Name of the checkpoint file
    pub filename: PathBuf,
}

impl Default for HotStartCheckpoint {
    /// Create a default `HotStartCheckpoint` instance.
    fn default() -> HotStartCheckpoint {
        HotStartCheckpoint {
            mode: HotStartMode::default(),
            frequency: ObserverMode::Always,
            directory: PathBuf::from(".checkpoints"),
            filename: PathBuf::from("egor.arg"),
        }
    }
}

impl HotStartCheckpoint {
    /// Create a new `HotStartCheckpoint` instance
    pub fn new<N: AsRef<str>>(
        directory: N,
        name: N,
        frequency: CheckpointingFrequency,
        ext_iters: HotStartMode,
    ) -> Self {
        HotStartCheckpoint {
            mode: ext_iters,
            frequency,
            directory: PathBuf::from(directory.as_ref()),
            filename: PathBuf::from(name.as_ref()),
        }
    }

    /// Full path to the checkpoint file.
    pub fn path(&self) -> PathBuf {
        self.directory.join(&self.filename)
    }

    /// Build the basin `ExactCheckpointWriter` for this configuration,
    /// creating the checkpoint directory if needed.
    pub fn writer(&self) -> std::io::Result<ExactCheckpointWriter> {
        if !self.directory.exists() {
            std::fs::create_dir_all(&self.directory)?;
        }
        Ok(ExactCheckpointWriter::new(self.path()))
    }

    /// Load a checkpoint from disk, if present, applying `HotStartMode::ExtendedIters`
    /// (bumping `max_iters`) on the loaded state, matching the old argmin-era
    /// `Checkpoint::load` behavior. The returned [`ExactCheckpoint`] is meant to be
    /// fed directly to `Executor::resume_from_checkpoint`, which is the only way
    /// to get basin's "skip init, restore eval counters, preserve best history"
    /// resume behavior (those hooks are private `Executor` fields, not
    /// independently reconstructible via `Executor::new`).
    ///
    /// Returns `Ok(None)` when no checkpoint exists on disk yet.
    pub fn load<So>(&self) -> std::io::Result<Option<ExactCheckpoint<So, EgorState<f64>>>>
    where
        So: DeserializeOwned,
    {
        let path = self.path();
        if !path.exists() {
            info!("No checkpoint found at {path:?}");
            return Ok(None);
        }
        info!("Checkpoint found at {path:?}, loading...");
        let checkpoint = read_exact_checkpoint::<So, EgorState<f64>>(&path)?;
        let (solver, mut state, counts) = checkpoint.into_parts();
        if let HotStartMode::ExtendedIters(n_iters) = self.mode {
            info!(
                "Extending max iters by {} from {}",
                n_iters, state.max_iters
            );
            state.extend_max_iters(n_iters);
        }
        Ok(Some(ExactCheckpoint::from_parts(solver, state, counts)))
    }
}

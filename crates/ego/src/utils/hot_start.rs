use argmin::core::Error;
pub use argmin::core::checkpointing::{Checkpoint, CheckpointingFrequency};
use log::info;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::Write;
use std::path::PathBuf;

use crate::EgorState;

/// Checkpoint file using argmin checkpointing
#[cfg(not(feature = "basin"))]
pub const CHECKPOINT_FILE: &str = "egor_checkpoint.json";
/// Solver-aware checkpoint used by the Basin executor.
#[cfg(feature = "basin")]
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

/// Handles saving a checkpoint to disk as a binary file.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct HotStartCheckpoint {
    /// Extended iteration number
    pub mode: HotStartMode,
    /// Indicates how often a checkpoint is created
    pub frequency: CheckpointingFrequency,
    /// Directory where the checkpoints are saved to
    pub directory: PathBuf,
    /// Name of the checkpoint files
    pub filename: PathBuf,
}

impl Default for HotStartCheckpoint {
    /// Create a default `HotStartCheckpoint` instance.
    fn default() -> HotStartCheckpoint {
        HotStartCheckpoint {
            mode: HotStartMode::default(),
            frequency: CheckpointingFrequency::default(),
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
}

impl<S> Checkpoint<S, EgorState<f64>> for HotStartCheckpoint
where
    S: Serialize + DeserializeOwned,
{
    /// Writes checkpoint to disk.
    ///
    /// If the directory does not exist already, it will be created. It uses `bincode` to serialize
    /// the data.
    /// It will return an error if creating the directory or file or serialization failed.
    fn save(&self, solver: &S, state: &EgorState<f64>) -> Result<(), Error> {
        if !self.directory.exists() {
            std::fs::create_dir_all(&self.directory)?
        }
        let fname = self.directory.join(&self.filename);
        let mut file = std::fs::File::create(fname).unwrap();

        // let bytes = bincode::serde::encode_to_vec((solver, state), bincode::config::standard())?;
        let bytes = serde_json::to_vec(&(solver, state))?;

        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load a checkpoint from disk.
    ///
    ///
    /// If there is no checkpoint on disk, it will return `Ok(None)`.
    /// Returns an error if opening the file or deserialization failed.
    fn load(&self) -> Result<Option<(S, EgorState<f64>)>, Error> {
        let path = &self.directory.join(&self.filename);
        if !path.exists() {
            info!("No checkpoint found at {:?}", path);
            return Ok(None);
        }
        info!("Checkpoint found at {:?}, loading...", path);
        let data = std::fs::read(path)?;

        // let (solver, mut state): (S, EgorState<f64>) =
        //     bincode::serde::borrow_decode_from_slice(&data, bincode::config::standard())
        //         .map(|(res, _)| res)?;

        let (solver, mut state): (S, EgorState<f64>) = serde_json::from_slice(&data)?;

        if let HotStartMode::ExtendedIters(n_iters) = self.mode {
            info!(
                "Extending max iters by {} from {}",
                n_iters, state.max_iters
            );
            state.extend_max_iters(n_iters);
        }
        Ok(Some((solver, state)))
    }

    /// Returns the how often a checkpoint is to be saved.
    ///
    /// Used internally by [`save_cond`](`argmin::core::checkpointing::Checkpoint::save_cond`).
    fn frequency(&self) -> CheckpointingFrequency {
        self.frequency
    }
}

use thiserror::Error;

use crate::EgorState;

/// A result type for EGO errors
pub type Result<T> = std::result::Result<T, EgoError>;

/// An error for efficient global optimization algorithm
#[derive(Error, Debug)]
pub enum EgoError {
    /// When configuration is invalid
    #[error("Invalid configuration: {0}")]
    InvalidConfigError(String),
    /// When LikelihoodComputation computation fails
    #[error("GP error")]
    GpError(#[from] egobox_gp::GpError),
    /// When a user function evaluation fails
    #[error("User function error: {0}")]
    UserFnError(String),
    /// When an invalid value is encountered
    #[error("Value error: {0}")]
    InvalidValue(String),
    /// When Moe error occurs
    #[error("MOE error")]
    MoeError(#[from] egobox_moe::MoeError),
    /// When IO fails
    #[error("IO error")]
    IoError(#[from] std::io::Error),
    /// When numpy array read fails
    #[error("IO error")]
    ReadNpyError(#[from] ndarray_npy::ReadNpyError),
    /// When numpy array write fails
    #[error("IO error")]
    WriteNpyError(#[from] ndarray_npy::WriteNpyError),
    /// When a `linfa` error occurs
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
    /// When an Argmin framework is raised
    #[error(transparent)]
    ArgminError(#[from] argmin::core::Error),
    /// When global EGO step cannot add any point
    #[error("EGO exit (no more point)")]
    NoMorePointToAddError(Box<EgorState<f64>>),
    /// When error during saving
    #[cfg(feature = "persistent")]
    #[error("Save error: {0}")]
    SaveBinaryError(#[from] bincode::error::EncodeError),
    /// When error during loading
    #[cfg(feature = "persistent")]
    #[error("Load error: {0}")]
    LoadBinaryError(#[from] bincode::error::DecodeError),
    /// When error during saving
    #[cfg(feature = "persistent")]
    #[error("Save error: {0}")]
    JsonError(#[from] serde_json::Error),
}

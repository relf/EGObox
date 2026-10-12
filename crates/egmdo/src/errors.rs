use thiserror::Error;

/// A result type for EGMDO computations
pub type Result<T> = std::result::Result<T, EgmdoError>;

/// An error when using the EGMDO algorithm
#[derive(Error, Debug)]
pub enum EgmdoError {
    /// When the problem definition is inconsistent
    #[error("Invalid problem: {0}")]
    InvalidProblem(String),
    /// When given data or function results have unexpected shapes or values
    #[error("Invalid data: {0}")]
    InvalidData(String),
    /// When a disciplinary GP fails
    #[error(transparent)]
    GpError(#[from] egobox_gp::GpError),
}

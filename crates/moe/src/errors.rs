use thiserror::Error;

/// A result type for Moe algorithm
pub type Result<T> = std::result::Result<T, MoeError>;

/// An error when using MOE algorithm
#[derive(Error, Debug)]
pub enum MoeError {
    /// When linear algebra computation fails
    #[cfg(feature = "blas")]
    #[error("Linalg BLAS error: {0}")]
    LinalgBlasError(#[from] ndarray_linalg::error::LinalgError),
    /// When linear algebra computation fails
    #[error(transparent)]
    LinalgError(#[from] linfa_linalg::LinalgError),
    /// When clustering fails
    #[error("Empty cluster: {0}")]
    EmptyCluster(String),
    /// When Gaussian Process fails
    #[error("GP error")]
    GpError(#[from] egobox_gp::GpError),
    /// When best expert search fails
    #[error("Expert error: {0}")]
    ExpertError(String),
    /// When error on clustering
    #[error("Clustering error: {0}")]
    ClusteringError(String),
    /// When sampling fails
    #[error("Sample error: {0}")]
    SampleError(String),
    /// When error during saving
    #[error("Save error: {0}")]
    SaveJsonError(#[from] serde_json::Error),
    /// When error during saving
    #[error("Save error: {0}")]
    SaveBinaryError(#[from] bincode::error::EncodeError),
    /// When error during loading
    #[error("Load error: {0}")]
    LoadBinaryError(#[from] bincode::error::DecodeError),
    /// When error during loading
    #[error("Load IO error")]
    LoadIoError(#[from] std::io::Error),
    /// When error during loading
    #[error("Load error: {0}")]
    LoadError(String),
    /// When error during loading
    #[error("InvalidValue error: {0}")]
    InvalidValueError(String),
    /// When a linfa error occurs
    #[error(transparent)]
    LinfaError(#[from] linfa::error::Error),
    /// When a linfa GMM clustering error occurs
    #[error(transparent)]
    LinfaClusteringrror(#[from] linfa_clustering::GmmError),
}

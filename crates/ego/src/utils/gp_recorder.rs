use crate::errors::{EgoError, Result};
use egobox_moe::{GpFileFormat, MixtureGpSurrogate, MoeError};
use std::path::Path;

/// Keep the error variants of the former implementation
fn ego_error(err: MoeError) -> EgoError {
    match err {
        MoeError::LoadIoError(err) => EgoError::IoError(err),
        MoeError::SaveBinaryError(err) => EgoError::SaveBinaryError(err),
        err => EgoError::MoeError(err),
    }
}

/// Save models in a bincode file
///
/// See [egobox_moe::save_models]
pub fn save_gp_models<P: AsRef<Path>>(
    path: P,
    models: &[Box<dyn MixtureGpSurrogate>],
) -> Result<()> {
    egobox_moe::save_models(path, models, GpFileFormat::Binary).map_err(ego_error)
}

/// Load models from a file saved by `Egor` (GP models of the iterations)
///
/// Any file accepted by [egobox_moe::load_models] is loaded.
pub fn load_gp_models<P: AsRef<Path>>(path: P) -> Result<Vec<Box<dyn MixtureGpSurrogate>>> {
    egobox_moe::load_models(path).map_err(ego_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_gp_models_from_invalid_file() {
        let path = "target/test_gp_recorder_invalid.bin";
        std::fs::write(path, b"not gp models").unwrap();
        let res = load_gp_models(path);
        let _ = std::fs::remove_file(path);
        assert!(res.is_err(), "decoding error expected");
    }

    #[test]
    fn test_save_gp_models_in_missing_dir() {
        let res = save_gp_models("target/test_gp_recorder_missing_dir/gp.bin", &[]);
        assert!(res.is_err(), "file creation error expected");
    }
}

//! Names of the files written by the optimizer in its output directory
//! (see `EgorConfig::outdir`) and of the hot start checkpoint.

/// Json filename for configuration
pub const CONFIG_FILE: &str = "egor_config.json";
/// Numpy filename for initial DOE dump
pub const DOE_INITIAL_FILE: &str = "egor_initial_doe.npy";
/// Numpy filename for current DOE dump
pub const DOE_FILE: &str = "egor_doe.npy";
/// Numpy filename for optimization history
pub const HISTORY_FILE: &str = "egor_history.npy";
/// Numpy filename for failed points (if any)
/// This file is created only if some points failed to be evaluated
pub const FAILED_POINTS_FILE: &str = "egor_failed_points.npy";
/// JSON filename for recorded EgorState files
pub const STATE_FILE_PREFIX: &str = "egor_state";
/// Gaussian process filename to save initial GPs built from initial_doe
pub const EGOR_INITIAL_GP_FILENAME: &str = "egor_initial_gp.bin";
/// Gaussian process filename to save GPs built at the last iteration
pub const EGOR_GP_FILENAME: &str = "egor_gp.bin";
/// Solver-aware checkpoint file used by the basin executor for hot start.
pub const CHECKPOINT_FILE: &str = "egor_checkpoint.bin";

/// Save `array` in a npy file, through a temporary file renamed into place so that a process
/// stopped meanwhile does not leave a truncated file (see [egobox_moe::write_atomically])
pub(crate) fn save_npy<P: AsRef<std::path::Path>>(
    path: P,
    array: &ndarray::Array2<f64>,
) -> crate::Result<()> {
    use ndarray_npy::WriteNpyExt;
    egobox_moe::write_atomically(path, |file| {
        array.write_npy(file).map_err(crate::EgoError::from)
    })
}

/// Save `text` in a file (see [save_npy])
pub(crate) fn save_text<P: AsRef<std::path::Path>>(path: P, text: &str) -> std::io::Result<()> {
    use std::io::Write;
    egobox_moe::write_atomically(path, |file| file.write_all(text.as_bytes()))
}

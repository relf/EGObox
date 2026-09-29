//! Conversion of Rust errors into Python exceptions and related helpers.
//!
//! Every user reachable failure should surface as a standard Python exception
//! (ValueError, TypeError, OSError or RuntimeError) rather than a Rust panic
//! which is raised as `pyo3_runtime.PanicException` (a `BaseException` subclass).
use std::error::Error;
use std::path::Path;
use std::sync::{Mutex, Once};

use ndarray::{Array1, Array2, ArrayViewD, Axis, Ix1, Ix2};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Format an error message including its chain of sources
fn error_chain(e: &dyn Error) -> String {
    let mut msg = e.to_string();
    let mut source = e.source();
    while let Some(err) = source {
        let s = err.to_string();
        if !msg.contains(&s) {
            msg.push_str(": ");
            msg.push_str(&s);
        }
        source = err.source();
    }
    msg
}

/// Convert an optimizer error into a Python exception
pub(crate) fn ego_err(e: egobox_ego::EgoError) -> PyErr {
    use egobox_ego::EgoError;
    let msg = error_chain(&e);
    match e {
        EgoError::InvalidConfigError(_) | EgoError::InvalidValue(_) => PyValueError::new_err(msg),
        EgoError::IoError(_) | EgoError::ReadNpyError(_) | EgoError::WriteNpyError(_) => {
            PyOSError::new_err(msg)
        }
        _ => PyRuntimeError::new_err(msg),
    }
}

/// Convert a surrogate model error into a Python exception
pub(crate) fn moe_err(e: egobox_moe::MoeError) -> PyErr {
    use egobox_moe::MoeError;
    let msg = error_chain(&e);
    match e {
        MoeError::LoadIoError(_) => PyOSError::new_err(msg),
        MoeError::LoadError(_)
        | MoeError::LoadBinaryError(_)
        | MoeError::SaveJsonError(_)
        | MoeError::SaveBinaryError(_)
        | MoeError::InvalidValueError(_) => PyValueError::new_err(msg),
        _ => PyRuntimeError::new_err(msg),
    }
}

/// Convert a surrogate model save/load error into a Python exception.
/// IO errors are raised as OSError(errno, strerror, filename), hence Python
/// selects the relevant subclass (e.g. FileNotFoundError, PermissionError).
pub(crate) fn moe_file_err(e: egobox_moe::MoeError, filename: &str) -> PyErr {
    match e {
        egobox_moe::MoeError::LoadIoError(io) => match io.raw_os_error() {
            Some(errno) => PyOSError::new_err((errno, io.to_string(), filename.to_string())),
            None => PyOSError::new_err(format!("{filename}: {io}")),
        },
        e => moe_err(e),
    }
}

/// File format deduced from filename extension: JSON if .json otherwise binary
pub(crate) fn gp_file_format(filename: &str) -> egobox_moe::GpFileFormat {
    match Path::new(filename).extension().and_then(|ext| ext.to_str()) {
        Some("json") => egobox_moe::GpFileFormat::Json,
        _ => egobox_moe::GpFileFormat::Binary,
    }
}

/// Check that `x` is a 2D array with `nx` columns
pub(crate) fn check_nx(x: &ndarray::ArrayView2<f64>, nx: usize) -> PyResult<()> {
    if x.ncols() != nx {
        return Err(PyValueError::new_err(format!(
            "input x should be of shape (n, {nx}), got {:?}",
            x.shape()
        )));
    }
    Ok(())
}

/// Convert training data into (xt[nsamples, nx], yt[nsamples]) accepting
/// xt as 1D (nx = 1) and yt as 1D or 2D with one column.
pub(crate) fn training_data(
    xt: ArrayViewD<f64>,
    yt: ArrayViewD<f64>,
) -> PyResult<(Array2<f64>, Array1<f64>)> {
    let xshape = xt.shape().to_vec();
    let xt = match xt.ndim() {
        1 => xt
            .into_dimensionality::<Ix1>()
            .unwrap()
            .insert_axis(Axis(1)),
        2 => xt.into_dimensionality::<Ix2>().unwrap(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "training input should be of shape (n, nx) or (n,), got {xshape:?}"
            )));
        }
    };

    let yshape = yt.shape().to_vec();
    let yt = match yt.ndim() {
        1 => yt.into_dimensionality::<Ix1>().unwrap(),
        2 if yshape[1] == 1 => yt
            .into_dimensionality::<Ix2>()
            .unwrap()
            .remove_axis(Axis(1)),
        _ => {
            return Err(PyValueError::new_err(format!(
                "training output should be of shape (n,) or (n, 1), got {yshape:?}"
            )));
        }
    };

    if xt.nrows() != yt.len() {
        return Err(PyValueError::new_err(format!(
            "training input and output should have the same number of samples, got {} and {}",
            xt.nrows(),
            yt.len()
        )));
    }
    if xt.nrows() == 0 {
        return Err(PyValueError::new_err("training data should not be empty"));
    }
    Ok((xt.to_owned(), yt.to_owned()))
}

/// Panic payload used to abort the optimizer from within a user callback
/// once a Python error has been recorded in a [`CallbackError`].
pub(crate) struct CallbackAbort;

/// Storage of the first Python error raised within user callbacks
/// called by the optimizer (objective or constraint functions).
///
/// As those callbacks can not return an error to the optimizer, the error is recorded
/// then the optimizer is aborted by panicking with [`CallbackAbort`] payload.
/// The panic is caught at the Python API boundary and the recorded error is raised.
#[derive(Default)]
pub(crate) struct CallbackError(Mutex<Option<PyErr>>);

impl CallbackError {
    /// Record the error (only the first one is kept) and abort the optimizer
    pub(crate) fn abort(&self, err: PyErr) -> ! {
        if let Ok(mut slot) = self.0.lock()
            && slot.is_none()
        {
            *slot = Some(err);
        }
        std::panic::panic_any(CallbackAbort)
    }

    /// Retrieve the recorded error if any
    pub(crate) fn take(&self) -> Option<PyErr> {
        self.0.lock().ok().and_then(|mut slot| slot.take())
    }
}

/// Install once a panic hook which silences [`CallbackAbort`] panics
/// and delegates to the previous hook otherwise.
pub(crate) fn install_panic_hook() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info.payload().downcast_ref::<CallbackAbort>().is_none() {
                default_hook(info)
            }
        }));
    });
}

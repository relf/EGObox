//! Conversion of Rust errors into Python exceptions and related helpers.
//!
//! Every user reachable failure should surface as a standard Python exception
//! (ValueError, TypeError, OSError or RuntimeError) rather than a Rust panic
//! which is raised as `pyo3_runtime.PanicException` (a `BaseException` subclass).
use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewD, Axis, Ix1, Ix2};
use pyo3::exceptions::{
    PyException, PyKeyboardInterrupt, PyOSError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;

use crate::interrupt_flag::InterruptFlag;

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
        _ if e.is_interruption() => PyKeyboardInterrupt::new_err(msg),
        _ => PyRuntimeError::new_err(msg),
    }
}

pyo3_stub_gen::create_exception!(
    egobox,
    Interrupted,
    PyException,
    "Raised by `fit` or `suggest` interrupted with an `InterruptFlag`"
);

/// Exception of an interrupted computation: the pending `KeyboardInterrupt` on Ctrl-C (thus
/// consumed, not raised again afterwards), `Interrupted` when the interrupt flag is set
fn interruption_err(py: Python, flag: Option<&InterruptFlag>) -> Option<PyErr> {
    if let Err(err) = py.check_signals() {
        Some(err)
    } else if flag.is_some_and(InterruptFlag::is_interrupted) {
        Some(Interrupted::new_err("interrupted by InterruptFlag"))
    } else {
        None
    }
}

/// Convert an optimizer error into a Python exception, raising the interruption exception
/// when interrupted (see [`interruption_err`])
pub(crate) fn interruptible_ego_err(
    py: Python,
    e: egobox_ego::EgoError,
    flag: Option<&InterruptFlag>,
) -> PyErr {
    e.is_interruption()
        .then(|| interruption_err(py, flag))
        .flatten()
        .unwrap_or_else(|| ego_err(e))
}

/// Convert a surrogate model training error into a Python exception, raising the
/// interruption exception when interrupted (see [`interruption_err`])
pub(crate) fn interruptible_moe_err(
    py: Python,
    e: egobox_moe::MoeError,
    flag: Option<&InterruptFlag>,
) -> PyErr {
    matches!(
        e,
        egobox_moe::MoeError::GpError(egobox_gp::GpError::Interrupted)
    )
    .then(|| interruption_err(py, flag))
    .flatten()
    .unwrap_or_else(|| moe_err(e))
}

/// Convert a surrogate model error into a Python exception
pub(crate) fn moe_err(e: egobox_moe::MoeError) -> PyErr {
    use egobox_moe::MoeError;
    let msg = error_chain(&e);
    match e {
        MoeError::LoadIoError(_) => PyOSError::new_err(msg),
        MoeError::GpError(egobox_gp::GpError::Interrupted) => PyKeyboardInterrupt::new_err(msg),
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

/// View `x` as a 2D array, 1D being read as one column: (n,) -> (n, 1)
fn to_2d(x: ArrayViewD<'_, f64>) -> Option<ArrayView2<'_, f64>> {
    match x.ndim() {
        1 => x
            .into_dimensionality::<Ix1>()
            .ok()
            .map(|x| x.insert_axis(Axis(1))),
        2 => x.into_dimensionality::<Ix2>().ok(),
        _ => None,
    }
}

/// View `y` as a 1D array, 2D being accepted with one column: (n, 1) -> (n,)
fn to_1d(y: ArrayViewD<'_, f64>) -> Option<ArrayView1<'_, f64>> {
    match y.ndim() {
        1 => y.into_dimensionality::<Ix1>().ok(),
        2 if y.shape()[1] == 1 => y
            .into_dimensionality::<Ix2>()
            .ok()
            .map(|y| y.remove_axis(Axis(1))),
        _ => None,
    }
}

/// Convert input `x` of a trained model with `nx` inputs into a (n, nx) array,
/// accepting 1D x (read as (n, 1)) when nx = 1 like `fit` does.
pub(crate) fn input_x(x: ArrayViewD<'_, f64>, nx: usize) -> PyResult<ArrayView2<'_, f64>> {
    let shape = x.shape().to_vec();
    let x = Some(x)
        .filter(|x| x.ndim() == 2 || (x.ndim() == 1 && nx == 1))
        .and_then(to_2d)
        .ok_or_else(|| {
            let or_1d = if nx == 1 { " or (n,)" } else { "" };
            PyTypeError::new_err(format!(
                "input x should be of shape (n, {nx}){or_1d}, got {shape:?}"
            ))
        })?;
    check_nx(&x, nx)?;
    Ok(x)
}

/// Convert output `y` into a (n,) array accepting (n,) or (n, 1) like `fit` does.
pub(crate) fn output_y(y: ArrayViewD<'_, f64>) -> PyResult<ArrayView1<'_, f64>> {
    let shape = y.shape().to_vec();
    to_1d(y).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "output y should be of shape (n,) or (n, 1), got {shape:?}"
        ))
    })
}

/// Convert training data into (xt[nsamples, nx], yt[nsamples]) accepting
/// xt as 1D (nx = 1) and yt as 1D or 2D with one column.
pub(crate) fn training_data(
    xt: ArrayViewD<f64>,
    yt: ArrayViewD<f64>,
) -> PyResult<(Array2<f64>, Array1<f64>)> {
    let xshape = xt.shape().to_vec();
    let xt = to_2d(xt).ok_or_else(|| {
        PyValueError::new_err(format!(
            "training input should be of shape (n, nx) or (n,), got {xshape:?}"
        ))
    })?;

    let yshape = yt.shape().to_vec();
    let yt = to_1d(yt).ok_or_else(|| {
        PyValueError::new_err(format!(
            "training output should be of shape (n,) or (n, 1), got {yshape:?}"
        ))
    })?;

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

/// Interruption of an optimization from Python: Ctrl-C (`KeyboardInterrupt`) or any other
/// exception raised by a signal handler, or an exception raised by a user callback which is
/// not an `Exception` subclass (e.g. `SystemExit`).
///
/// The optimizer stops after the current iteration, then [`Interruption::finish`] gives the
/// result obtained so far on Ctrl-C, while any other exception is raised.
#[derive(Default)]
pub(crate) struct Interruption {
    interrupted: AtomicBool,
    error: Mutex<Option<PyErr>>,
}

impl Interruption {
    /// Whether an error raised by a user callback interrupts the optimization instead of
    /// being an evaluation error: `KeyboardInterrupt`, `SystemExit`, ... are not `Exception`s
    pub(crate) fn is_interruption(py: Python, err: &PyErr) -> bool {
        !err.is_instance_of::<PyException>(py)
    }

    /// Record the exception interrupting the optimization: the first one is kept,
    /// unless it is a `KeyboardInterrupt` and `err` is not, as `err` has to be raised then
    pub(crate) fn record(&self, py: Python, err: PyErr) {
        self.interrupted.store(true, Ordering::Relaxed);
        if let Ok(mut slot) = self.error.lock()
            && slot.as_ref().is_none_or(|first| {
                first.is_instance_of::<PyKeyboardInterrupt>(py)
                    && !err.is_instance_of::<PyKeyboardInterrupt>(py)
            })
        {
            *slot = Some(err);
        }
    }

    /// Whether the optimization is interrupted, without running Python signal handlers
    pub(crate) fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Relaxed)
    }

    /// Run the Python signal handlers, recording the exception raised if any
    /// (`KeyboardInterrupt` on Ctrl-C), and tell whether the optimization is interrupted.
    /// Signals are handled even when already interrupted: a pending one (e.g. Ctrl-C pressed
    /// again) would otherwise raise once the optimization is over, its result being lost.
    pub(crate) fn check(&self, py: Python) -> bool {
        if let Err(err) = py.check_signals() {
            self.record(py, err);
        }
        self.interrupted.load(Ordering::Relaxed)
    }

    /// Outcome of an optimization run once the optimizer returned, given the signals received
    /// until then: when interrupted by Ctrl-C, the result obtained so far (`KeyboardInterrupt`
    /// is raised when there is none, e.g. while the initial DOE was evaluated), otherwise the
    /// exception which interrupted the optimization is raised.
    pub(crate) fn finish<T>(&self, py: Python, res: egobox_ego::Result<T>) -> PyResult<T> {
        self.check(py);
        match (self.error.lock().ok().and_then(|mut slot| slot.take()), res) {
            (Some(err), Ok(res)) if err.is_instance_of::<PyKeyboardInterrupt>(py) => Ok(res),
            (Some(err), _) => Err(err),
            (None, res) => res.map_err(ego_err),
        }
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

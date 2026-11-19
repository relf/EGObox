//! Interruption of an optimization, a surrogate training or a suggestion from Python code.

use egobox_gp::Cancellation;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A request to interrupt `minimize`, `fit` or `suggest`, set from another thread (e.g. a GUI
/// or a supervising thread) or from within the objective function.
///
/// It is given as `interrupt` argument, then set to interrupt the call, like Ctrl-C:
///
/// * `minimize` stops and returns the result obtained so far with
///   status.exit == ExitStatus.INTERRUPT,
/// * `fit` and `suggest` raise `Interrupted`.
///
/// Surrogate training and infill optimization are stopped at once, while an objective
/// function evaluation in progress is not (it may check `is_set()`). The flag is not cleared
/// afterwards: call `clear()` before reusing it, as for a `threading.Event`.
///
/// Examples
/// --------
/// >>> flag = egx.InterruptFlag()
/// >>> threading.Timer(60.0, flag.set).start()  # interrupt after one minute
/// >>> res = egx.Egor(xspecs).minimize(fun, max_iters=100, interrupt=flag)
#[gen_stub_pyclass]
#[pyclass(frozen, module = "egobox")]
pub(crate) struct InterruptFlag(Arc<AtomicBool>);

#[gen_stub_pymethods]
#[pymethods]
impl InterruptFlag {
    /// An interrupt flag, not set
    #[new]
    fn new() -> Self {
        InterruptFlag(Arc::new(AtomicBool::new(false)))
    }

    /// Set the flag: the call given this flag is interrupted
    fn set(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Clear the flag, to reuse it for another call
    fn clear(&self) {
        self.0.store(false, Ordering::SeqCst);
    }

    /// Whether the flag is set
    fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    fn __repr__(&self) -> String {
        format!(
            "InterruptFlag(set={})",
            if self.is_set() { "True" } else { "False" }
        )
    }
}

impl InterruptFlag {
    /// Whether the flag is set
    pub(crate) fn is_interrupted(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Cheap thread-safe check of the flag, without the GIL
    pub(crate) fn cancellation(&self) -> Cancellation {
        let flag = self.0.clone();
        Cancellation::new(move || flag.load(Ordering::Relaxed))
    }
}

/// Request to stop a computation early: Ctrl-C (see [`crate::sigint::SigintGuard`]) or the
/// interrupt flag given by the user, if any
pub(crate) fn interruption(
    ctrl_c: Option<Cancellation>,
    flag: Option<&InterruptFlag>,
) -> Cancellation {
    match (ctrl_c, flag.map(InterruptFlag::cancellation)) {
        (Some(ctrl_c), Some(flag)) => {
            Cancellation::new(move || ctrl_c.is_cancelled() || flag.is_cancelled())
        }
        (Some(check), None) | (None, Some(check)) => check,
        (None, None) => Cancellation::default(),
    }
}

//! Prompt Ctrl-C handling while Rust code runs.
//!
//! On Ctrl-C (SIGINT), Python only trips a flag: `KeyboardInterrupt` is raised once Python code
//! runs again on the main thread, i.e. after a long Rust computation (surrogate training, infill
//! optimization). While such a computation runs, a hook is put in front of Python's SIGINT
//! handler: it counts the signal, which stops the Rust computation early (see
//! [`SigintGuard::cancellation`]), then calls Python's handler, so that `KeyboardInterrupt` is
//! raised as usual afterwards.

use egobox_gp::Cancellation;
use pyo3::prelude::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// SIGINT received while the hook is installed
static SIGINTS: AtomicU64 = AtomicU64::new(0);
/// Number of (nested) calls with the hook installed: it is installed once
static CALLS: Mutex<usize> = Mutex::new(0);

/// SIGINT hook installed for the duration of a call, removed when the last guard is dropped
pub(crate) struct SigintGuard {
    baseline: u64,
}

impl SigintGuard {
    /// Install the hook, unless the call does not run on the main thread or Python's SIGINT
    /// handler is not the default one (raising `KeyboardInterrupt`): `None` then, Ctrl-C being
    /// handled by Python only.
    pub(crate) fn install(py: Python) -> Option<SigintGuard> {
        if !default_sigint_handling(py) {
            return None;
        }
        let mut calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        if *calls == 0 && !imp::install() {
            return None;
        }
        *calls += 1;
        Some(SigintGuard {
            baseline: SIGINTS.load(Ordering::SeqCst),
        })
    }

    /// Cheap thread-safe check of a SIGINT received since the hook was installed
    pub(crate) fn cancellation(&self) -> Cancellation {
        let baseline = self.baseline;
        Cancellation::new(move || SIGINTS.load(Ordering::Relaxed) != baseline)
    }
}

impl Drop for SigintGuard {
    fn drop(&mut self) {
        let mut calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        *calls -= 1;
        if *calls == 0 {
            imp::restore();
        }
    }
}

/// Whether the call runs on the main thread, the only one where Python handles signals, with
/// Python's default SIGINT handler: a custom handler may not interrupt anything
fn default_sigint_handling(py: Python) -> bool {
    let check = || -> PyResult<bool> {
        let threading = py.import("threading")?;
        let main_thread = threading
            .call_method0("current_thread")?
            .is(&threading.call_method0("main_thread")?);
        let signal = py.import("signal")?;
        let handler = signal.call_method1("getsignal", (signal.getattr("SIGINT")?,))?;
        Ok(main_thread && handler.is(&signal.getattr("default_int_handler")?))
    };
    check().unwrap_or(false)
}

#[cfg(unix)]
mod imp {
    use super::SIGINTS;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    type Handler = extern "C" fn(libc::c_int);
    type SigInfoHandler = extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void);

    /// Action replaced by the hook, restored afterwards
    static PREVIOUS: Mutex<Option<libc::sigaction>> = Mutex::new(None);
    /// Handler of the replaced action, called by the hook, and whether it takes signal info
    static PREVIOUS_HANDLER: AtomicUsize = AtomicUsize::new(0);
    static PREVIOUS_SIGINFO: AtomicBool = AtomicBool::new(false);

    extern "C" fn hook(sig: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
        SIGINTS.fetch_add(1, Ordering::SeqCst);
        let handler = PREVIOUS_HANDLER.load(Ordering::SeqCst);
        if handler == libc::SIG_DFL || handler == libc::SIG_IGN {
            return;
        }
        // SAFETY: the handler installed before the hook (Python's one), called the way the
        // system would have called it, from the signal context as well
        unsafe {
            if PREVIOUS_SIGINFO.load(Ordering::SeqCst) {
                std::mem::transmute::<usize, SigInfoHandler>(handler)(sig, info, context)
            } else {
                std::mem::transmute::<usize, Handler>(handler)(sig)
            }
        }
    }

    fn hook_address() -> libc::sighandler_t {
        hook as SigInfoHandler as libc::sighandler_t
    }

    /// Current SIGINT action
    fn current() -> Option<libc::sigaction> {
        // SAFETY: only reads the current action into a zeroed struct
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            (libc::sigaction(libc::SIGINT, std::ptr::null(), &mut action) == 0).then_some(action)
        }
    }

    /// Put the hook in front of the current SIGINT handler, keeping its flags and mask
    /// (Python's handler relies on interrupted system calls, without `SA_RESTART`)
    pub(super) fn install() -> bool {
        let Some(previous) = current() else {
            return false;
        };
        let handler = previous.sa_sigaction;
        if handler == libc::SIG_DFL || handler == libc::SIG_IGN || handler == hook_address() {
            return false;
        }
        PREVIOUS_HANDLER.store(handler, Ordering::SeqCst);
        PREVIOUS_SIGINFO.store(previous.sa_flags & libc::SA_SIGINFO != 0, Ordering::SeqCst);
        let mut action = previous;
        action.sa_sigaction = hook_address();
        action.sa_flags |= libc::SA_SIGINFO;
        // SAFETY: the hook only uses an atomic counter and calls the previous handler
        let installed =
            unsafe { libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) == 0 };
        if installed {
            *PREVIOUS.lock().unwrap_or_else(|e| e.into_inner()) = Some(previous);
        }
        installed
    }

    /// Restore the replaced action, unless the hook was replaced meanwhile
    /// (e.g. by a `signal.signal()` call)
    pub(super) fn restore() {
        let previous = PREVIOUS.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(previous) = previous
            && current().is_some_and(|action| action.sa_sigaction == hook_address())
        {
            // SAFETY: restores the action replaced by `install`
            unsafe { libc::sigaction(libc::SIGINT, &previous, std::ptr::null_mut()) };
        }
    }
}

#[cfg(windows)]
mod imp {
    use super::SIGINTS;
    use std::sync::atomic::Ordering;
    use windows_sys::Win32::System::Console::{CTRL_C_EVENT, SetConsoleCtrlHandler};
    use windows_sys::core::BOOL;

    /// Console control handler called before the previously registered ones (the C runtime
    /// one calling Python's SIGINT handler), to which Ctrl-C is passed on
    unsafe extern "system" fn hook(ctrl: u32) -> BOOL {
        if ctrl == CTRL_C_EVENT {
            SIGINTS.fetch_add(1, Ordering::SeqCst);
        }
        0
    }

    pub(super) fn install() -> bool {
        // SAFETY: registers a handler which only uses an atomic counter
        unsafe { SetConsoleCtrlHandler(Some(hook), 1) != 0 }
    }

    pub(super) fn restore() {
        // SAFETY: removes the handler registered by `install`
        unsafe { SetConsoleCtrlHandler(Some(hook), 0) };
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    pub(super) fn install() -> bool {
        false
    }
    pub(super) fn restore() {}
}

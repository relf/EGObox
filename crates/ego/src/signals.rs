//! Signal handling scoped to optimization runs.
//!
//! While a run executes (see [`install`]), Ctrl-C and termination signals are requests to stop
//! the optimization gracefully after the current iteration. A further signal terminates the
//! process at once. The previous signal dispositions are restored when the last run returns.
//!
//! * Unix: SIGINT, SIGTERM, SIGHUP, SIGUSR1 and SIGUSR2. A signal which already has a handler,
//!   or is ignored (e.g. SIGHUP under `nohup`), is left alone.
//! * Windows: Ctrl-C and Ctrl-Break console events.
//! * Other targets: no handling.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Stop requests received while handlers are installed
static INTERRUPTS: AtomicU64 = AtomicU64::new(0);
/// Index (+1) in [`imp::SIGNALS`] of the signal of the last stop request
static LAST: AtomicUsize = AtomicUsize::new(0);
/// Bitmask of the indices in [`imp::SIGNALS`] of the installed handlers
static INSTALLED: AtomicU32 = AtomicU32::new(0);
/// Number of runs with installed handlers
static RUNS: Mutex<usize> = Mutex::new(0);

/// Number of stop requests received so far, to be compared with a value taken at run start
pub(crate) fn interrupts() -> u64 {
    INTERRUPTS.load(Ordering::Relaxed)
}

/// Name of the signal of the last stop request
pub(crate) fn last_signal() -> &'static str {
    match LAST.load(Ordering::Relaxed) {
        0 => "a signal",
        i => imp::SIGNALS[i - 1].1,
    }
}

/// Record a stop request from the signal of index `i` in [`imp::SIGNALS`] (async-signal-safe)
fn record(i: usize) {
    LAST.store(i + 1, Ordering::Relaxed);
    INTERRUPTS.fetch_add(1, Ordering::Relaxed);
}

/// Installed signal handlers, restored when the last guard of concurrent runs is dropped
pub(crate) struct SignalGuard(());

/// Install the signal handlers for the duration of a run
pub(crate) fn install() -> SignalGuard {
    let mut runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
    if *runs == 0 {
        INSTALLED.store(imp::install(), Ordering::Relaxed);
    }
    *runs += 1;
    SignalGuard(())
}

impl Drop for SignalGuard {
    fn drop(&mut self) {
        let mut runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
        *runs -= 1;
        if *runs == 0 {
            imp::restore(INSTALLED.swap(0, Ordering::Relaxed));
        }
    }
}

#[cfg(unix)]
mod imp {
    use super::{INSTALLED, record};
    use std::sync::atomic::{AtomicU32, Ordering};

    pub(super) const SIGNALS: [(libc::c_int, &str); 5] = [
        (libc::SIGINT, "SIGINT (Ctrl-C)"),
        (libc::SIGTERM, "SIGTERM"),
        (libc::SIGHUP, "SIGHUP"),
        (libc::SIGUSR1, "SIGUSR1"),
        (libc::SIGUSR2, "SIGUSR2"),
    ];

    /// Bitmask of the signals already reported as left alone
    static REPORTED: AtomicU32 = AtomicU32::new(0);

    extern "C" fn handler(sig: libc::c_int) {
        if let Some(i) = SIGNALS.iter().position(|(s, _)| *s == sig) {
            record(i);
        }
        // a further signal terminates the process (default action)
        let installed = INSTALLED.load(Ordering::Relaxed);
        for (i, (s, _)) in SIGNALS.iter().enumerate() {
            if installed & (1 << i) != 0 {
                // SAFETY: signal() is async-signal-safe
                unsafe { libc::signal(*s, libc::SIG_DFL) };
            }
        }
    }

    fn handler_address() -> libc::sighandler_t {
        handler as extern "C" fn(libc::c_int) as libc::sighandler_t
    }

    /// Current disposition of `sig`
    fn current(sig: libc::c_int) -> Option<libc::sighandler_t> {
        // SAFETY: sigaction only reads the current action into a zeroed struct
        unsafe {
            let mut old: libc::sigaction = std::mem::zeroed();
            (libc::sigaction(sig, std::ptr::null(), &mut old) == 0).then_some(old.sa_sigaction)
        }
    }

    /// Install the handler for the signals with the default disposition,
    /// returning the bitmask of the installed ones
    pub(super) fn install() -> u32 {
        let mut installed = 0;
        for (i, (sig, name)) in SIGNALS.iter().enumerate() {
            match current(*sig) {
                Some(libc::SIG_DFL) => {
                    // SAFETY: the handler only uses atomics and async-signal-safe functions
                    let ok = unsafe {
                        let mut action: libc::sigaction = std::mem::zeroed();
                        action.sa_sigaction = handler_address();
                        action.sa_flags = libc::SA_RESTART;
                        libc::sigemptyset(&mut action.sa_mask);
                        libc::sigaction(*sig, &action, std::ptr::null_mut()) == 0
                    };
                    if ok {
                        installed |= 1 << i;
                    }
                }
                // left alone, reported once
                Some(disposition)
                    if REPORTED.fetch_or(1 << i, Ordering::Relaxed) & (1 << i) == 0 =>
                {
                    if disposition == libc::SIG_IGN {
                        log::info!("{name} is ignored: the optimization is not interrupted by it");
                    } else {
                        log::warn!(
                            "{name} has a handler: the optimization is not interrupted by it, \
                             see Egor::interrupt_check()"
                        );
                    }
                }
                _ => {}
            }
        }
        installed
    }

    /// Restore the default disposition of the installed signals,
    /// unless a handler was set meanwhile (or already restored after a signal)
    pub(super) fn restore(installed: u32) {
        for (i, (sig, _)) in SIGNALS.iter().enumerate() {
            if installed & (1 << i) != 0 && current(*sig) == Some(handler_address()) {
                // SAFETY: restores the default disposition
                unsafe { libc::signal(*sig, libc::SIG_DFL) };
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    use super::record;
    use std::sync::atomic::{AtomicBool, Ordering};
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler,
    };
    use windows_sys::core::BOOL;

    pub(super) const SIGNALS: [(u32, &str); 2] =
        [(CTRL_C_EVENT, "Ctrl-C"), (CTRL_BREAK_EVENT, "Ctrl-Break")];

    /// Set by the first stop request: further events are left to the next handler
    static STOPPED: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn handler(ctrl: u32) -> BOOL {
        match SIGNALS.iter().position(|(event, _)| *event == ctrl) {
            Some(i) if !STOPPED.swap(true, Ordering::Relaxed) => {
                record(i);
                1
            }
            // a further event (or console close, logoff, shutdown) terminates the process
            _ => 0,
        }
    }

    pub(super) fn install() -> u32 {
        STOPPED.store(false, Ordering::Relaxed);
        // SAFETY: registers a handler which only uses atomics
        let ok = unsafe { SetConsoleCtrlHandler(Some(handler), 1) } != 0;
        if ok { 1 } else { 0 }
    }

    pub(super) fn restore(installed: u32) {
        if installed != 0 {
            // SAFETY: removes the handler registered by `install`
            unsafe { SetConsoleCtrlHandler(Some(handler), 0) };
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    pub(super) const SIGNALS: [((), &str); 0] = [];
    pub(super) fn install() -> u32 {
        0
    }
    pub(super) fn restore(_installed: u32) {}
}

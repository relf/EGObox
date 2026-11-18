//! Cooperative cancellation of long computations (e.g. GP hyperparameters optimization).

use std::sync::Arc;

/// A request to stop a long computation early, checked between optimization starts and
/// after optimizer iterations. A training cancelled this way fails with
/// [`GpError::Interrupted`](crate::GpError::Interrupted).
///
/// The check is called often, possibly from several threads at once: it must be cheap and
/// thread-safe, e.g. a load of an atomic flag. By default, nothing is ever cancelled.
///
/// As a run-time hook rather than a parameter value, a cancellation is not serialized and
/// any two cancellations compare equal.
#[derive(Clone, Default)]
pub struct Cancellation(Option<Arc<dyn Fn() -> bool + Send + Sync>>);

impl Cancellation {
    /// A cancellation requested when `check` returns true
    pub fn new(check: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(check)))
    }

    /// Whether the cancellation is requested
    pub fn is_cancelled(&self) -> bool {
        self.0.as_ref().is_some_and(|check| check())
    }
}

impl std::fmt::Debug for Cancellation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(_) => write!(f, "Cancellation(check)"),
            None => write!(f, "Cancellation(none)"),
        }
    }
}

impl PartialEq for Cancellation {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for Cancellation {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_cancellation() {
        assert!(!Cancellation::default().is_cancelled());
        let flag = Arc::new(AtomicBool::new(false));
        let check = flag.clone();
        let cancellation = Cancellation::new(move || check.load(Ordering::Relaxed));
        let clone = cancellation.clone();
        assert!(!clone.is_cancelled());
        flag.store(true, Ordering::Relaxed);
        assert!(clone.is_cancelled() && cancellation.is_cancelled());
        assert_eq!(cancellation, Cancellation::default());
    }
}

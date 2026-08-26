//! Observer/observable support allowing external code to be notified of `Egor` progress.

use std::sync::Arc;

use argmin::core::{Error, KV, State, observers::Observe};
use ndarray::ArrayView1;

use crate::EgorState;

/// Implement this trait to be notified at each iteration of an `Egor` run.
pub trait EgorObserver: Send + Sync {
    /// Called after each iteration with the current best point and its cost found so far.
    fn on_iteration(&self, iter: u64, x_opt: ArrayView1<f64>, y_opt: ArrayView1<f64>);
}

/// Relays argmin iteration events to the `EgorObserver`s registered on `Egor`.
pub(crate) struct EgorObserverDispatcher {
    pub(crate) observers: Vec<Arc<dyn EgorObserver>>,
}

impl Observe<EgorState<f64>> for EgorObserverDispatcher {
    fn observe_iter(
        &mut self,
        state: &EgorState<f64>,
        _kv: &KV,
    ) -> std::result::Result<(), Error> {
        // Best values may not be set yet on the very first observed state.
        if let (Some(x_opt), Some(y_opt)) = (state.get_best_param(), state.get_full_best_cost()) {
            let iter = state.get_iter();
            for observer in &self.observers {
                observer.on_iteration(iter, x_opt.view(), y_opt.view());
            }
        }
        Ok(())
    }
}

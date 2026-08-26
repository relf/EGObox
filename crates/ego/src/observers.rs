//! Observer/observable support allowing external code to be notified of `Egor` progress.

use std::sync::Arc;

use argmin::core::{Error, KV, State, observers::Observe};
use ndarray::ArrayView1;

use crate::EgorState;

/// Snapshot of the optimizer state passed to `EgorObserver::on_iteration`.
pub struct EgorObservableState<'a> {
    /// Current iteration number.
    pub iter: u64,
    /// Best point found so far.
    pub x_opt: ArrayView1<'a, f64>,
    /// Cost (objective + constraints) at `x_opt`.
    pub y_opt: ArrayView1<'a, f64>,
}

/// Implement this trait to be notified at each iteration of an `Egor` run.
pub trait EgorObserver: Send + Sync {
    /// Called after each iteration with a snapshot of the optimizer's current best state.
    fn on_iteration(&self, state: &EgorObservableState);
}

/// Relays argmin iteration events to the `EgorObserver`s registered on `Egor`.
pub(crate) struct EgorObserverDispatcher {
    pub(crate) observers: Vec<Arc<dyn EgorObserver>>,
}

impl Observe<EgorState<f64>> for EgorObserverDispatcher {
    fn observe_iter(&mut self, state: &EgorState<f64>, _kv: &KV) -> std::result::Result<(), Error> {
        // Best values may not be set yet on the very first observed state.
        if let (Some(x_opt), Some(y_opt)) = (state.get_best_param(), state.get_full_best_cost()) {
            let observed = EgorObservableState {
                iter: state.get_iter(),
                x_opt: x_opt.view(),
                y_opt: y_opt.view(),
            };
            for observer in &self.observers {
                observer.on_iteration(&observed);
            }
        }
        Ok(())
    }
}

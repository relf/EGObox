//! Surrogate models of the EGO solver, together with the clusterings and theta
//! initial values reused as warm start when they are retrained.

use crate::EgorState;
use crate::errors::{EgoError, Result};

use egobox_moe::{Clustering, MixtureGpSurrogate};
use ndarray::Array2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DataClustering {
    /// Clustering is not updated given values are used as is
    Fixed,
    /// Clustering is recomputed
    Regenerate,
}

impl From<bool> for DataClustering {
    fn from(value: bool) -> Self {
        if value {
            DataClustering::Regenerate
        } else {
            DataClustering::Fixed
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ThetaOptimization {
    /// Theta is not optimized given values are used as is
    Disabled,
    /// Theta is optimized with given values as initialization
    Enabled,
}

impl From<bool> for ThetaOptimization {
    fn from(value: bool) -> Self {
        if value {
            ThetaOptimization::Enabled
        } else {
            ThetaOptimization::Disabled
        }
    }
}

/// How surrogates are (re)trained: clustering and theta hyperparameters handling,
/// and components activity (cooperative strategy) for the theta optimization
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrainSpec<'a> {
    pub(crate) clustering: DataClustering,
    pub(crate) theta: ThetaOptimization,
    pub(crate) actives: &'a Array2<usize>,
}

/// Surrogate models (objectives then constraints) with, for each of them, the clustering
/// and the theta values used to warm start a retraining
pub(crate) struct SurrogateSet {
    pub(crate) clusterings: Vec<Option<Clustering>>,
    pub(crate) theta_inits: Vec<Option<Array2<f64>>>,
    pub(crate) models: Vec<Box<dyn MixtureGpSurrogate>>,
}

impl SurrogateSet {
    /// No trained model yet for `n` surrogates
    pub(crate) fn new(n: usize) -> Self {
        SurrogateSet {
            clusterings: vec![None; n],
            theta_inits: vec![None; n],
            models: Vec::new(),
        }
    }

    /// Move the surrogates out of `state`
    pub(crate) fn take(state: &mut EgorState<f64>) -> Result<Self> {
        let clusterings = state
            .take_clusterings()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No clustering!".to_string()))?;
        let theta_inits = state
            .take_theta_inits()
            .ok_or_else(|| EgoError::InternalError("EgorSolver: No theta inits!".to_string()))?;
        let models = std::mem::take(&mut state.surrogate.models);
        Ok(SurrogateSet {
            clusterings,
            theta_inits,
            models,
        })
    }

    /// Move the surrogates into `state`
    pub(crate) fn put(self, state: EgorState<f64>) -> EgorState<f64> {
        let mut state = state
            .clusterings(self.clusterings)
            .theta_inits(self.theta_inits);
        state.surrogate.models = self.models;
        state
    }

    /// Whether models do not need any training given `nb_points` training
    /// points: clustering is kept and models are already trained on all of them.
    pub(crate) fn is_up_to_date(&self, clustering: DataClustering, nb_points: usize) -> bool {
        clustering == DataClustering::Fixed
            && !self.models.is_empty()
            && self
                .models
                .iter()
                .all(|m| m.training_data().0.nrows() == nb_points)
    }

    /// Sync clusterings and theta inits with the models and the `inits` that just came
    /// out of their update, so a future full retrain reuses the fitted clustering/theta
    /// as a warm start.
    pub(crate) fn sync(&mut self, inits: &[Option<Array2<f64>>]) {
        (0..self.models.len()).for_each(|k| {
            self.clusterings[k] = Some(self.models[k].to_clustering());
            if let Some(init) = inits.get(k).and_then(|i| i.as_ref()) {
                self.theta_inits[k] = Some(init.to_owned());
            }
        });
    }
}

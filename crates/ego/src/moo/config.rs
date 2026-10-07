//! Multi-objective optimization configuration

use serde::{Deserialize, Serialize};

/// Strategy used to optimize several objectives
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MooStrategy {
    /// ParEGO (Knowles 2006): at each iteration, objectives normalized with their observed
    /// bounds are aggregated with an augmented Tchebycheff function using a weight vector
    /// randomly drawn from a simplex lattice, then the mono-objective EGO machinery is applied
    /// to that scalarized objective.
    #[default]
    ParEgo,
}

/// Multi-objective optimization configuration, used when the number of objectives is
/// greater than 1 (see [`crate::EgorConfig::n_obj`])
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MooConfig {
    /// Strategy used to optimize the objectives
    pub(crate) strategy: MooStrategy,
    /// ParEGO augmented Tchebycheff coefficient
    pub(crate) rho: f64,
    /// ParEGO number of divisions of the weight simplex lattice
    /// (`None` for the default: 10 for 2 objectives, 4 for 3 objectives, 3 otherwise)
    pub(crate) n_divisions: Option<usize>,
}

impl Default for MooConfig {
    fn default() -> Self {
        MooConfig {
            strategy: MooStrategy::default(),
            rho: crate::moo::scalarization::PAREGO_RHO,
            n_divisions: None,
        }
    }
}

impl MooConfig {
    /// Sets the multi-objective strategy
    pub fn strategy(mut self, strategy: MooStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the ParEGO augmented Tchebycheff coefficient (default 0.05)
    pub fn rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    /// Sets the ParEGO number of divisions of the weight simplex lattice
    pub fn n_divisions(mut self, n_divisions: usize) -> Self {
        self.n_divisions = Some(n_divisions);
        self
    }
}

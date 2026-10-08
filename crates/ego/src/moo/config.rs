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
    /// Expected Improvement Matrix (Zhan et al. 2017): one surrogate per objective, the
    /// expected improvements of each objective over each point of the current Pareto front
    /// (objectives normalized with their observed bounds) are aggregated into an infill
    /// criterion (see [`MooConfig::eim_aggregation`]).
    Eim,
    /// Expected Hypervolume Improvement (Emmerich et al. 2006): one surrogate per objective,
    /// expected improvement of the hypervolume dominated by the current Pareto front
    /// (objectives normalized with their observed bounds, reference point at the front nadir
    /// plus 10 % of its range), computed in closed form.
    ///
    /// The computation decomposes the region not dominated by the front into
    /// `(front_size + 1)^n_obj` grid cells: with large fronts (more than 405 points for
    /// 2 objectives, 89 for 3, 18 for 5), the front is approximated by a spread subset of its
    /// points, which overestimates the improvement near the left out points. At most 8 objectives
    /// are supported (use [`MooStrategy::Eim`] beyond).
    Ehvi,
}

/// Aggregation of the expected improvement matrix used by [`MooStrategy::Eim`]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EimAggregation {
    /// Minimum over front points of the Euclidean norm of the expected improvements
    #[default]
    Euclidean,
    /// Minimum over front points of the maximum expected improvement
    Maximin,
    /// Minimum over front points of the expected hypervolume improvement of the
    /// hyper-rectangle dominated by the point improved by the expected improvements
    Hypervolume,
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
    /// EIM aggregation of the expected improvement matrix
    #[serde(default)]
    pub(crate) eim_aggregation: EimAggregation,
    /// Optional hypervolume-based stop: (relative tolerance, number of iterations)
    #[serde(default)]
    pub(crate) hv_stop: Option<(f64, usize)>,
}

impl Default for MooConfig {
    fn default() -> Self {
        MooConfig {
            strategy: MooStrategy::default(),
            rho: crate::moo::scalarization::PAREGO_RHO,
            n_divisions: None,
            eim_aggregation: EimAggregation::default(),
            hv_stop: None,
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

    /// Sets the aggregation of the expected improvement matrix (EIM strategy, default Euclidean)
    pub fn eim_aggregation(mut self, aggregation: EimAggregation) -> Self {
        self.eim_aggregation = aggregation;
        self
    }

    /// Stops the optimization when the hypervolume of the (feasible) Pareto front increased by less
    /// than `tol` (relative to its current value) during the last `n_iters` iterations.
    /// The termination reason is then [`crate::TerminationReason::SolverConverged`].
    ///
    /// The front of `n_iters` iterations ago is the one of the data without the last
    /// `n_iters * batch` points (`batch` being the qEI batch size): when proposed points are
    /// rejected (too close to existing ones) or fail, the window covers more iterations, which
    /// only delays the stop.
    pub fn hv_stop(mut self, tol: f64, n_iters: usize) -> Self {
        self.hv_stop = Some((tol, n_iters));
        self
    }
}

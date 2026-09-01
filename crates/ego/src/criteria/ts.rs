use crate::criteria::InfillCriterion;

use egobox_moe::MixtureGpSurrogate;
use ndarray::{Array1, ArrayView};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256Plus;

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// A structure for Thompson Sampling infill criterion implementation.
///
/// # Thompson Sampling as an infill criterion
///
/// Classical Thompson Sampling draws one function from the surrogate's
/// posterior distribution and selects the point optimizing this sampled
/// function. Drawing and optimizing an *exact* joint posterior sample over a
/// continuous domain (e.g. through a Karhunen-Loeve/random-Fourier-features
/// expansion of the underlying kernel) is expensive and is not directly
/// compatible with `Egor`'s gradient-based infill optimization, which
/// requires a single differentiable, deterministic criterion `value(x)` to
/// be repeatedly evaluated (value and gradient) at many candidate points
/// during one optimization run.
///
/// This implementation therefore uses the standard *reparameterized*
/// approximation to Thompson Sampling (see e.g. the "one-sample"
/// reparameterization used in randomized-UCB/TS variants of Bayesian
/// optimization): for each candidate at every `Egor` outer iteration, a
/// standard normal deviate `z ~ N(0, 1)` is drawn. The candidate infill
/// criterion is then the pointwise reparameterized posterior sample
///
/// ```text
/// ts(x) = mu(x) + z * sigma(x)
/// ```
///
/// where `mu(x)` and `sigma(x)` are the surrogate posterior mean and
/// standard deviation at `x`. `Egor` looks for the point minimizing this
/// sampled surrogate, exactly as it would minimize the true objective. Since
/// `z` is fixed for the duration of the iteration's infill optimization,
/// `ts(x)` is smooth in `x` and can be handled by the gradient-based
/// (SLSQP) or gradient-free (Cobyla) infill optimizers exactly like the
/// other criteria.
///
/// Each `z` is shared across its candidate's whole domain rather than being
/// spatially correlated the way an exact GP sample path would be, this is
/// an approximation: it reproduces the explore/exploit trade-off and the
/// iteration-to-iteration stochasticity that make Thompson Sampling
/// effective (some iterations favor points with a good mean prediction,
/// others favor points with high uncertainty), but it does not capture
/// fine-grained spatial correlation of a genuine posterior sample.
///
/// # Reproducibility
///
/// By default, the random generator is seeded from a combination of the
/// current time and a global counter to ensure different seeds across
/// instances. To enable reproducible runs, use [`ThompsonSampling::new_with_seed`]
/// with an explicit seed value.
#[derive(Clone, Serialize, Deserialize)]
pub struct ThompsonSampling {
    #[serde(skip, default = "ThompsonSampling::default_rng")]
    rng: Arc<Mutex<Xoshiro256Plus>>,
    #[serde(default = "ThompsonSampling::default_candidate_count")]
    candidate_count: usize,
}

impl ThompsonSampling {
    const DEFAULT_CANDIDATE_COUNT: usize = 10;

    /// Creates a new Thompson Sampling infill criterion with a
    /// freshly (entropy-)seeded random generator.
    ///
    /// This is equivalent to `new_with_seed(None)`. Runs using this constructor
    /// are not deterministic across different executions.
    pub fn new() -> Self {
        Self {
            rng: Self::default_rng(),
            candidate_count: Self::default_candidate_count(),
        }
    }

    /// Creates a new Thompson Sampling infill criterion with an optional seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional seed value for the random number generator.
    ///   - If `Some(seed)`, the RNG will be deterministically seeded with this value,
    ///     enabling reproducible runs.
    ///   - If `None`, the RNG will be seeded from a combination of time and a
    ///     global counter (non-deterministic).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Reproducible Thompson Sampling
    /// let ts = ThompsonSampling::new_with_seed(Some(42));
    ///
    /// // Non-deterministic (same as ThompsonSampling::new())
    /// let ts = ThompsonSampling::new_with_seed(None);
    /// ```
    pub fn new_with_seed(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => Arc::new(Mutex::new(Xoshiro256Plus::seed_from_u64(s))),
            None => Self::default_rng(),
        };
        Self {
            rng,
            candidate_count: Self::default_candidate_count(),
        }
    }

    /// Sets the number of independently sampled candidates considered per iteration.
    pub fn with_candidate_count(mut self, candidate_count: usize) -> Self {
        self.candidate_count = candidate_count.max(1);
        self
    }

    fn default_candidate_count() -> usize {
        Self::DEFAULT_CANDIDATE_COUNT
    }

    fn default_rng() -> Arc<Mutex<Xoshiro256Plus>> {
        // Seed from a coarse, hard-to-predict source so that successive
        // `ThompsonSampling` instances (and deserialized ones, e.g. after a
        // warm restart) do not replay the exact same sequence of `z` draws.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let seed = nanos ^ count.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Arc::new(Mutex::new(Xoshiro256Plus::seed_from_u64(seed)))
    }
}

impl Default for ThompsonSampling {
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl InfillCriterion for ThompsonSampling {
    fn name(&self) -> &'static str {
        "TS"
    }

    /// Thompson Sampling relies on a fresh random draw at the start of
    /// every candidate, so it is implemented as a (stochastic) dynamic
    /// scaling factor: [`ThompsonSampling::scaling`] draws `z ~ N(0, 1)`
    /// once per candidate and this `z` is subsequently passed
    /// back into [`ThompsonSampling::value`]/[`ThompsonSampling::grad`] as
    /// the `scale` argument for every candidate point evaluated during that
    /// iteration's infill optimization.
    fn uses_dynamic_scaling(&self) -> bool {
        true
    }

    fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Compute the reparameterized Thompson Sampling infill criterion at
    /// given `x` point using the surrogate model `obj_model`: this is the
    /// (negated, so that higher is better as for the other criteria) value
    /// of the posterior sample `mu(x) + z * sigma(x)`, where `z` is the
    /// standard normal deviate drawn for the current iteration by
    /// [`ThompsonSampling::scaling`] (via the `scale` parameter).
    fn value(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
        _viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        _sigma_weight: Option<f64>,
        scale: Option<f64>,
    ) -> f64 {
        let z = scale.unwrap_or(0.0);
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        match obj_model.predict_valvar(&pt) {
            Ok((p, s)) => {
                let sigma = if s[0] < f64::EPSILON {
                    0.0
                } else {
                    s[0].sqrt()
                };
                // infill criterion is negated so that higher is better (like the other criteria)
                -(p[0] + z * sigma)
            }
            _ => 0.0,
        }
    }

    /// Derivative wrt `x` of the reparameterized Thompson Sampling
    /// criterion, `d/dx [-(mu(x) + z * sigma(x))]`.
    fn grad(
        &self,
        x: &[f64],
        obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
        _viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        _sigma_weight: Option<f64>,
        scale: Option<f64>,
    ) -> Array1<f64> {
        let z = scale.unwrap_or(0.0);
        let pt = ArrayView::from_shape((1, x.len()), x).unwrap();
        match obj_model.predict_valvar(&pt) {
            Ok((_p, s)) => {
                let sigma = if s[0] < f64::EPSILON {
                    0.0
                } else {
                    s[0].sqrt()
                };
                let (p_prime, var_prime) = obj_model.predict_valvar_gradients(&pt).unwrap();
                let p_prime = p_prime.row(0);
                if sigma < f64::EPSILON {
                    p_prime.mapv(|v| -v)
                } else {
                    let sigma_prime = var_prime.row(0).mapv(|v| v / (2.0 * sigma));
                    p_prime.mapv(|v| -v) - sigma_prime.mapv(|v| z * v)
                }
            }
            _ => Array1::zeros(pt.len()),
        }
    }

    /// Draws the standard normal deviate `z` used as the shared
    /// reparameterization coefficient for one candidate. Called once per
    /// candidate (see
    /// [`ThompsonSampling::uses_dynamic_scaling`]), independently of the
    /// candidate points `x` it is given.
    fn scaling(
        &self,
        _x: &ndarray::ArrayView2<f64>,
        _obj_model: &dyn MixtureGpSurrogate,
        _fmin: f64,
        _viability_model: Option<&dyn MixtureGpSurrogate>,
        _alpha: Option<f64>,
        _sigma_weight: Option<f64>,
    ) -> f64 {
        let mut rng = self.rng.lock().unwrap();
        rng.sample(StandardNormal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use egobox_moe::MixintContext;
    use egobox_moe::MoeBuilder;
    use linfa::Dataset;
    use ndarray::array;

    fn build_mixi_moe() -> Box<dyn MixtureGpSurrogate> {
        let xtypes = vec![XType::Float(0., 25.)];
        let mixi = MixintContext::new(&xtypes);
        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.0], [7.0], [15.0], [25.0]];
        let yt = xt.mapv(|v: f64| (v - 3.5) * (v / std::f64::consts::PI).sin());
        let yt = yt.remove_axis(ndarray::Axis(1));
        let ds = Dataset::new(xt, yt);
        let moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");
        Box::new(moe) as Box<dyn MixtureGpSurrogate>
    }

    #[test]
    fn test_uses_dynamic_scaling() {
        let ts = ThompsonSampling::new();
        assert!(ts.uses_dynamic_scaling());
    }

    #[test]
    fn test_candidate_count() {
        assert_eq!(ThompsonSampling::new().candidate_count(), 10);
        assert_eq!(
            ThompsonSampling::new()
                .with_candidate_count(3)
                .candidate_count(),
            3
        );
        assert_eq!(
            ThompsonSampling::new()
                .with_candidate_count(0)
                .candidate_count(),
            1
        );
    }

    #[test]
    fn test_scaling_draws_vary() {
        let ts = ThompsonSampling::new();
        let x = array![[0.0], [1.0]];
        let z1 = ts.scaling(&x.view(), &*build_mixi_moe(), 0.0, None, None, None);
        let z2 = ts.scaling(&x.view(), &*build_mixi_moe(), 0.0, None, None, None);
        // Extremely unlikely two successive standard normal draws collide exactly.
        assert_ne!(z1, z2);
    }

    #[test]
    fn test_value_matches_reparameterized_sample() {
        let moe = build_mixi_moe();
        let x = [10.0];
        let pt = ArrayView::from_shape((1, 1), &x).unwrap();
        let (p, s) = moe.predict_valvar(&pt).unwrap();
        let sigma = s[0].sqrt();

        let ts = ThompsonSampling::new();
        let z = 0.7;
        let value = ts.value(&x, &*moe, 0.0, None, None, None, Some(z));
        approx::assert_abs_diff_eq!(value, -(p[0] + z * sigma), epsilon = 1e-8);
    }

    #[test]
    fn test_grad_matches_finite_difference() {
        let moe = build_mixi_moe();
        let ts = ThompsonSampling::new();
        let z = -0.4;
        let h = 1e-5;
        let x0 = 12.0;

        let grad = ts.grad(&[x0], &*moe, 0.0, None, None, None, Some(z))[0];
        let fdiff = (ts.value(&[x0 + h], &*moe, 0.0, None, None, None, Some(z))
            - ts.value(&[x0 - h], &*moe, 0.0, None, None, None, Some(z)))
            / (2.0 * h);

        approx::assert_abs_diff_eq!(grad, fdiff, epsilon = 1e-2);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let moe = build_mixi_moe();
        let x = array![[0.0], [1.0]];

        // Create two ThompsonSampling instances with the same seed
        let ts1 = ThompsonSampling::new_with_seed(Some(42));
        let ts2 = ThompsonSampling::new_with_seed(Some(42));

        // They should produce the same sequence of z values
        let z1_1 = ts1.scaling(&x.view(), &*moe, 0.0, None, None, None);
        let z2_1 = ts2.scaling(&x.view(), &*moe, 0.0, None, None, None);
        assert_eq!(
            z1_1, z2_1,
            "First z values should be identical with same seed"
        );

        let z1_2 = ts1.scaling(&x.view(), &*moe, 0.0, None, None, None);
        let z2_2 = ts2.scaling(&x.view(), &*moe, 0.0, None, None, None);
        assert_eq!(
            z1_2, z2_2,
            "Second z values should be identical with same seed"
        );

        // Verify the sequence is consistent
        assert_eq!(z1_1, z1_1, "Same instance should maintain consistency");
    }

    #[test]
    fn test_different_seeds_produce_different_results() {
        let moe = build_mixi_moe();
        let x = array![[0.0], [1.0]];

        let ts1 = ThompsonSampling::new_with_seed(Some(42));
        let ts2 = ThompsonSampling::new_with_seed(Some(123));

        let z1 = ts1.scaling(&x.view(), &*moe, 0.0, None, None, None);
        let z2 = ts2.scaling(&x.view(), &*moe, 0.0, None, None, None);

        // Very unlikely to collide by chance
        assert_ne!(z1, z2, "Different seeds should produce different z values");
    }

    #[test]
    fn test_strategy_uses_egor_seed() {
        use crate::{EgorConfig, InfillStrategy};

        let config1 = EgorConfig::default()
            .infill_strategy(InfillStrategy::TS)
            .seed(42)
            .check()
            .unwrap();
        let config2 = EgorConfig::default()
            .infill_strategy(InfillStrategy::TS)
            .seed(42)
            .check()
            .unwrap();
        let x = array![[0.0], [1.0]];
        let moe = build_mixi_moe();

        assert_eq!(
            config1
                .infill_criterion
                .scaling(&x.view(), &*moe, 0.0, None, None, None),
            config2
                .infill_criterion
                .scaling(&x.view(), &*moe, 0.0, None, None, None)
        );
    }
}

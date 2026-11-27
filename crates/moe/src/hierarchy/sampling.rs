//! Sampling of hierarchical design spaces

use super::{DesignSpace, Var, check_values, invalid, label};
use crate::errors::Result;
use crate::mixint::MixintSampling;
use crate::mixint::space::fold_with_enum_index;
use crate::xtypes::XType;
use egobox_doe::SamplingMethod;
use ndarray::{Array2, s};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::seq::{SliceRandom, index};
use rand_xoshiro::Xoshiro256Plus;
use std::sync::Mutex;

/// Default maximum number of configurations to enumerate
const MAX_CONFIGURATIONS: usize = 10_000;

/// A decorator of a sampling method that draws valid points of a
/// hierarchical [`DesignSpace`].
///
/// The *configurations* are the valid combinations of values of the discrete
/// variables that some condition or value constraint mentions, the ones that
/// shape the space (meta and partially decreed variables). The points are
/// spread over the configurations as evenly as possible: each configuration
/// gets the same number of points, give or take one, and which ones get an
/// extra point is random. When there are fewer points than configurations,
/// they go to distinct configurations drawn at random. This is the approach
/// of Sect. 4.4 of the reference (one point per configuration, then a
/// subsample).
///
/// The other variables, continuous ones and discrete ones that shape
/// nothing, are drawn by the inner sampling method as
/// [`MixintSampling`] draws them, so that a Latin hypercube keeps covering
/// them evenly. Each point is then corrected
/// ([`DesignSpace::correct_get_included`]): excluded variables take their
/// default value.
///
/// When there are more configurations than
/// [`max_configurations`](HierarchicalSampling::max_configurations), the
/// whole space is sampled as [`MixintSampling`] does and the points are
/// corrected, which favours the configurations nearest to the sampled
/// values.
///
/// # Example
///
/// ```
/// use egobox_doe::{Lhs, SamplingMethod};
/// use egobox_moe::{as_continuous_limits, DesignSpace, HierarchicalSampling, XType};
///
/// // A wing with 0, 1 or 2 motors, and the radius of each motor's propeller
/// let mut ds = DesignSpace::named(&[
///     ("m", XType::Int(0, 2)),
///     ("r_1", XType::Float(0.5, 2.)),
///     ("r_2", XType::Float(0.5, 2.)),
/// ])?;
/// ds.decreed_var("r_1", "m", &[1., 2.])?.decreed_var("r_2", "m", &[2.])?;
///
/// let lhs = Lhs::new(&as_continuous_limits::<f64>(ds.xtypes()));
/// let x = HierarchicalSampling::new(lhs, ds).sample(9);
/// // 3 points per number of motors, with excluded radii at their lower bound
/// assert_eq!(x.column(0).iter().filter(|&&m| m == 0.).count(), 3);
/// # Ok::<(), egobox_moe::MoeError>(())
/// ```
pub struct HierarchicalSampling<S: SamplingMethod<f64>> {
    /// Draws the variables that shape nothing, in the relaxed space
    mixint: MixintSampling<f64, S>,
    design_space: DesignSpace,
    /// Fixed values, one per variable (see [`HierarchicalSampling::fix`])
    pins: Vec<Option<f64>>,
    max_configurations: usize,
    output_in_folded_space: bool,
    /// Draws the configurations of the points
    rng: Mutex<Xoshiro256Plus>,
}

impl<S: SamplingMethod<f64>> HierarchicalSampling<S> {
    /// Sampler of `design_space` built on `method`, which must sample its
    /// relaxed space, whose bounds are
    /// [`as_continuous_limits(design_space.xtypes())`](crate::as_continuous_limits).
    pub fn new(method: S, design_space: DesignSpace) -> Self {
        let n = design_space.xtypes().len();
        HierarchicalSampling {
            mixint: MixintSampling::new(method, design_space.xtypes().to_vec()),
            pins: vec![None; n],
            max_configurations: MAX_CONFIGURATIONS,
            output_in_folded_space: false,
            rng: Mutex::new(Xoshiro256Plus::from_entropy()),
            design_space,
        }
    }

    /// Sets the random generator that spreads the points over the
    /// configurations. The inner sampling method has its own.
    pub fn with_rng(mut self, rng: Xoshiro256Plus) -> Self {
        self.rng = Mutex::new(rng);
        self
    }

    /// Sets the maximum number of configurations to enumerate (10 000 by
    /// default). Above it, the whole space is sampled and the points are
    /// corrected.
    pub fn max_configurations(mut self, max: usize) -> Self {
        self.max_configurations = max;
        self
    }

    /// Sets whether samples are given in folded space (one column per
    /// variable, enum variables given by their level index) rather than in
    /// the relaxed space, as [`MixintSampling::work_in_folded_space`] does
    pub fn work_in_folded_space(&mut self, output_in_folded_space: bool) -> &mut Self {
        self.output_in_folded_space = output_in_folded_space;
        self
    }

    /// Restricts sampling to the configurations where variable `var` is
    /// included and takes `value`, for instance to sample within one
    /// configuration of the meta variables.
    ///
    /// `var` is given by name or by index (see [`Var`]); it must be discrete,
    /// and `value` is given in folded space. Fixing a variable again replaces
    /// its value.
    ///
    /// Fails, leaving the sampler unchanged, when the variable is unknown or
    /// continuous, when the value is invalid, or when no valid configuration
    /// has the fixed values.
    pub fn fix<'a>(&mut self, var: impl Into<Var<'a>>, value: f64) -> Result<&mut Self> {
        let ds = &self.design_space;
        let i = ds.resolve(var.into())?;
        if let XType::Float(_, _) = ds.xtypes[i] {
            return Err(invalid(format!(
                "variable {} is continuous: only discrete variables can be fixed",
                label(&ds.names, i)
            )));
        }
        check_values(&ds.xtypes, &ds.names, i, &[value])?;
        let mut pins = self.pins.clone();
        pins[i] = Some(value);
        if !ds.has_valid_configuration(&pins) {
            return Err(invalid(format!(
                "no valid configuration has variable {} included and equal to {value}",
                label(&ds.names, i)
            )));
        }
        self.pins = pins;
        Ok(self)
    }

    /// The design space
    pub fn design_space(&self) -> &DesignSpace {
        &self.design_space
    }

    /// Whether each variable shapes the configurations: mentioned by a
    /// condition or a value constraint, or fixed
    fn shaping(&self) -> Vec<bool> {
        (self.design_space.mentioned.iter())
            .zip(&self.pins)
            .map(|(mentioned, pin)| !mentioned.is_empty() || pin.is_some())
            .collect()
    }

    /// Configuration index of each of `ns` points spread over `n`
    /// configurations
    fn assignment(&self, n: usize, ns: usize) -> Vec<usize> {
        let mut rng = self.rng.lock().unwrap();
        let mut assignment: Vec<usize> = (0..ns / n).flat_map(|_| 0..n).collect();
        assignment.extend(index::sample(&mut *rng, n, ns % n));
        assignment.shuffle(&mut *rng);
        assignment
    }
}

impl<S: SamplingMethod<f64>> SamplingMethod<f64> for HierarchicalSampling<S> {
    fn sampling_space(&self) -> &Array2<f64> {
        self.mixint.sampling_space()
    }

    /// Normalized samples of the inner sampling method, which ignore the
    /// hierarchy: use [`sample`](SamplingMethod::sample) for valid points
    fn normalized_sample(&self, ns: usize) -> Array2<f64> {
        self.mixint.normalized_sample(ns)
    }

    fn sample(&self, ns: usize) -> Array2<f64> {
        let ds = &self.design_space;
        let mut x = self.mixint.sample(ns);
        let shaping = self.shaping();
        match ds.configurations(&shaping, &self.pins, self.max_configurations) {
            Some(configurations) if !configurations.is_empty() => {
                log::info!(
                    "Hierarchical sampling: {ns} points over {} configurations (n_valid = {}, n_declared = {})",
                    configurations.len(),
                    ds.n_valid(),
                    ds.n_declared()
                );
                let starts = ds.unfolded_starts();
                let assignment = self.assignment(configurations.len(), ns);
                for (mut row, c) in x.rows_mut().into_iter().zip(assignment) {
                    let configuration = &configurations[c];
                    for (i, xtype) in ds.xtypes.iter().enumerate() {
                        if !shaping[i] {
                            continue;
                        }
                        let v = configuration[i];
                        match xtype {
                            XType::Enum(n) => {
                                row.slice_mut(s![starts[i]..starts[i] + n]).fill(0.);
                                row[starts[i] + v as usize] = 1.;
                            }
                            _ => row[starts[i]] = v,
                        }
                    }
                }
            }
            _ => log::info!(
                "Hierarchical sampling: more than {} configurations, sampling the whole space and correcting (n_valid = {}, n_declared = {})",
                self.max_configurations,
                ds.n_valid(),
                ds.n_declared()
            ),
        }
        let (x, _) = ds
            .correct_unfolded_pinned(&x, &self.pins)
            .expect("declarations and fixed values leave a valid configuration");
        if self.output_in_folded_space {
            fold_with_enum_index(&ds.xtypes, &x)
        } else {
            x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::correction::tests::{ADAM, mlp};
    use crate::{MixintSampling, as_continuous_limits};
    use egobox_doe::{Lhs, LhsKind};
    use ndarray::ArrayView1;
    use std::collections::HashMap;

    type Sampler = HierarchicalSampling<Lhs<f64, Xoshiro256Plus>>;

    fn lhs(ds: &DesignSpace, seed: u64) -> Lhs<f64, Xoshiro256Plus> {
        Lhs::new(&as_continuous_limits::<f64>(ds.xtypes()))
            .kind(LhsKind::Classic)
            .with_rng(Xoshiro256Plus::seed_from_u64(seed))
    }

    /// Sampler of `ds` in folded space
    fn sampler(ds: DesignSpace, seed: u64) -> Sampler {
        let mut sampler = HierarchicalSampling::new(lhs(&ds, seed), ds)
            .with_rng(Xoshiro256Plus::seed_from_u64(seed));
        sampler.work_in_folded_space(true);
        sampler
    }

    /// Values of the variables that shape the MLP: optimizer, layers and
    /// units per layer
    fn shape(x: ArrayView1<f64>) -> Vec<i64> {
        [2, 9, 10, 11, 12].iter().map(|&j| x[j] as i64).collect()
    }

    fn shape_counts(x: &Array2<f64>) -> HashMap<Vec<i64>, usize> {
        let mut counts = HashMap::new();
        for row in x.rows() {
            *counts.entry(shape(row)).or_default() += 1;
        }
        counts
    }

    /// Valid points are left unchanged by the correction
    fn assert_valid(ds: &DesignSpace, x: &Array2<f64>) {
        let (xcorr, _) = ds.correct_get_included(x).unwrap();
        assert_eq!(&xcorr, x);
    }

    #[test]
    fn test_mlp_configurations() {
        // 69 configurations of optimizer, layers and units, times 3
        // activation functions, give the 207 valid configurations
        let sampler = sampler(mlp(), 0);
        let ds = sampler.design_space();
        let configurations = ds
            .configurations(&sampler.shaping(), &sampler.pins, 1000)
            .unwrap();
        assert_eq!(configurations.len(), 69);
        assert_eq!(ds.n_valid(), 69 * 3);
        assert!(
            ds.configurations(&sampler.shaping(), &sampler.pins, 68)
                .is_none()
        );
    }

    #[test]
    fn test_one_point_per_configuration() {
        let sampler = sampler(mlp(), 0);
        let x = sampler.sample(69);
        assert_eq!(x.dim(), (69, 13));
        assert_valid(sampler.design_space(), &x);
        assert_eq!(shape_counts(&x).len(), 69);
    }

    #[test]
    fn test_more_points_than_configurations() {
        let sampler = sampler(mlp(), 1);
        let x = sampler.sample(150);
        assert_valid(sampler.design_space(), &x);
        let counts = shape_counts(&x);
        assert_eq!(counts.len(), 69);
        assert!(counts.values().all(|&c| c == 2 || c == 3), "{counts:?}");
    }

    #[test]
    fn test_fewer_points_than_configurations() {
        let sampler = sampler(mlp(), 2);
        let x = sampler.sample(10);
        assert_valid(sampler.design_space(), &x);
        assert_eq!(shape_counts(&x).len(), 10);
    }

    #[test]
    fn test_continuous_variables_keep_the_lhs() {
        // The neutral learning rate keeps the LHS strata: one point per
        // tenth of [0, 1]
        let x = sampler(mlp(), 3).sample(10);
        let mut strata: Vec<usize> = x.column(0).iter().map(|v| (v * 10.) as usize).collect();
        strata.sort_unstable();
        assert_eq!(strata, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_relaxed_space_output() {
        let ds = mlp();
        let x = HierarchicalSampling::new(lhs(&ds, 4), ds.clone()).sample(20);
        // 3 activation and 2 optimizer columns
        assert_eq!(x.dim(), (20, 16));
        let (xcorr, _) = ds.correct_get_included_unfolded(&x).unwrap();
        assert_eq!(xcorr, x);
    }

    #[test]
    fn test_above_max_configurations() {
        let sampler = sampler(mlp(), 5).max_configurations(10);
        let x = sampler.sample(50);
        assert_eq!(x.nrows(), 50);
        assert_valid(sampler.design_space(), &x);
    }

    #[test]
    fn test_fix() {
        let mut sampler = sampler(mlp(), 6);
        sampler.fix("optimizer", ADAM).unwrap();
        let x = sampler.sample(30);
        assert_valid(sampler.design_space(), &x);
        // Adam has 5 + 25 configurations, with 1 or 2 layers
        assert_eq!(shape_counts(&x).len(), 30);
        assert!(x.column(2).iter().all(|&o| o == ADAM));
        assert!(x.column(9).iter().all(|&l| l <= 2.));

        // Also above the maximum number of configurations
        let mut sampler = sampler.max_configurations(1);
        sampler.fix("layers", 2.).unwrap();
        let x = sampler.sample(20);
        assert_valid(sampler.design_space(), &x);
        assert!(x.column(2).iter().all(|&o| o == ADAM));
        assert!(x.column(9).iter().all(|&l| l == 2.));
    }

    #[test]
    fn test_invalid_fix() {
        let mut sampler = sampler(mlp(), 7);
        sampler.fix("layers", 3.).unwrap();
        // No 3 layers with Adam
        let Err(err) = sampler.fix("optimizer", ADAM) else {
            panic!("Adam with 3 layers should fail")
        };
        assert!(err.to_string().contains("no valid configuration"), "{err}");
        assert!(sampler.fix("decay", 0.5).is_err());
        assert!(sampler.fix("layers", 4.).is_err());
        assert!(sampler.fix("depth", 1.).is_err());
        assert_eq!(sampler.pins.iter().flatten().count(), 1);
        // 3 layers need ASGD, which forbids 40 units
        assert!(sampler.fix("units_3", 40.).is_err());
    }

    #[test]
    fn test_same_seeds_same_samples() {
        assert_eq!(sampler(mlp(), 8).sample(30), sampler(mlp(), 8).sample(30));
    }

    #[test]
    fn test_without_hierarchy_as_mixint_sampling() {
        let xtypes = [XType::Float(0., 1.), XType::Int(0, 4), XType::Enum(3)];
        let ds = DesignSpace::new(&xtypes);
        let hierarchical = HierarchicalSampling::new(lhs(&ds, 9), ds.clone()).sample(12);
        let mixint = MixintSampling::new(lhs(&ds, 9), xtypes.to_vec()).sample(12);
        assert_eq!(hierarchical, mixint);
    }
}

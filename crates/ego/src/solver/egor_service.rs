//! Egor optimizer service implements [`Egor`] optimizer with an ask-and-tell interface.
//! It allows to keep the control on the iteration loop by asking for optimum location
//! suggestions and telling objective function values at these points.
//!
//! ```no_run
//! # use ndarray::{array, Array2, ArrayView1, ArrayView2, Zip, concatenate, Axis};
//! # use egobox_doe::{Lhs, SamplingMethod};
//! # use egobox_ego::{EgorServiceBuilder, InfillStrategy, RegressionSpec, CorrelationSpec};
//!
//! # use rand_xoshiro::Xoshiro256Plus;
//! # use ndarray_rand::rand::SeedableRng;
//! use argmin_testfunctions::rosenbrock;
//!
//! fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
//!     (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
//! }
//!
//! let egor = EgorServiceBuilder::optimize()
//!             .configure(|conf| {
//!                conf.configure_gp(|gp_conf| {
//!                  gp_conf.regression_spec(RegressionSpec::ALL)
//!                         .correlation_spec(CorrelationSpec::ALL)
//!                })
//!                .infill_strategy(InfillStrategy::EI)
//!                .seed(42)
//!             })
//!             .min_within(&array![[0., 25.]])
//!             .expect("optimizer configured");
//!
//! let mut doe = array![[0.], [7.], [20.], [25.]];
//! let mut y_doe = xsinx(&doe.view());
//!
//! for _i in 0..10 {
//!     // we tell function values and ask for next suggested optimum location
//!     let x_suggested = egor.suggest(&doe, &y_doe);
//!     
//!     // we update the doe
//!     doe = concatenate![Axis(0), doe, x_suggested];
//!     y_doe = xsinx(&doe.view());
//! }
//!
//! let best = egor.best_index(&doe, &y_doe);
//! println!("xsinx min result = {} at {}", y_doe.row(best), doe.row(best));
//! ```
//!
use std::marker::PhantomData;

use crate::{EgorConfig, EgorSolver, errors::Result, to_xtypes, types::*};
use egobox_moe::{GpMixtureParams, MixintGpMixtureParams, to_continuous_space, to_discrete_space};
use ndarray::{Array2, ArrayBase, Data, Ix2};

use serde::{Serialize, de::DeserializeOwned};

/// EGO optimizer service builder allowing to use Egor optimizer
/// as a service.
///
pub struct EgorServiceFactory<C: CstrFn = Cstr> {
    config: EgorConfig,
    phantom: PhantomData<C>,
}

impl<C: CstrFn> EgorServiceFactory<C> {
    /// Function to be minimized domain should be basically R^nx -> R^ny
    /// where nx is the dimension of input x and ny the output dimension
    /// equal to `n_obj` objectives (1 by default) + `n_cstr` constraints.
    /// With several objectives (see `EgorConfig::n_obj`), suggested points are chosen with the
    /// configured multi-objective strategy (see `EgorConfig::configure_moo`).
    /// But function has to be able to evaluate several points in one go
    /// hence take an (p, nx) matrix and return an (p, ny) matrix
    pub fn optimize() -> Self {
        EgorServiceFactory {
            config: EgorConfig::default(),
            phantom: PhantomData,
        }
    }

    /// Configure the Egor optimizer with a closure
    /// taking and returning an EgorConfig structure.
    pub fn configure<F: FnOnce(EgorConfig) -> EgorConfig>(mut self, init: F) -> Self {
        self.config = init(self.config);
        self
    }

    /// Build an Egor optimizer to minimize the function within
    /// the continuous `xlimits` specified as [[lower, upper], ...] array where the
    /// number of rows gives the dimension of the inputs (continuous optimization)
    /// and the ith row is the interval of the ith component of the input x.
    pub fn min_within(
        self,
        xlimits: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<EgorServiceApi<GpMixtureParams<f64>, C>> {
        let config = self.config.xtypes(&to_xtypes(xlimits));
        Ok(EgorServiceApi {
            solver: EgorSolver::new(config.check()?),
        })
    }

    /// Build an Egor optimizer to minimize the function R^n -> R^p taking
    /// inputs specified with given xtypes where some of components may be
    /// discrete variables (mixed-integer optimization).
    pub fn min_within_mixint_space(
        self,
        xtypes: &[XType],
    ) -> Result<EgorServiceApi<MixintGpMixtureParams, C>> {
        let config = self.config.xtypes(xtypes);
        Ok(EgorServiceApi {
            solver: EgorSolver::new(config.check()?),
        })
    }
}

/// Egor optimizer service API.
#[derive(Clone)]
pub struct EgorServiceApi<SB: SurrogateBuilder + Serialize + DeserializeOwned, C: CstrFn = Cstr> {
    solver: EgorSolver<SB, C>,
}

impl<SB: SurrogateBuilder + Serialize + DeserializeOwned, C: CstrFn> EgorServiceApi<SB, C> {
    /// Given an evaluated doe (x, y) data, return the next promising x point
    /// where optimum may be located with regard to the infill criterion.
    /// This function inverses the control of the optimization and can be used
    /// for an ask-and-tell interface to the Egor optimizer.
    pub fn suggest(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Array2<f64> {
        let xtypes = &self.solver.config.xtypes;
        let x_data = to_continuous_space(xtypes, x_data);
        let x = self.solver.suggest(&x_data, y_data);
        to_discrete_space(xtypes, &x).to_owned()
    }

    /// Given an evaluated doe (x, y) data, return the index of the best evaluated point:
    /// the minimum wrt constraints, or the compromise point of the Pareto front with
    /// several objectives.
    ///
    /// When GP models a noise on function values (see [`crate::GpConfig::noise_ratio`]),
    /// the best point is the one with the lowest estimated value (surrogate models posterior
    /// means) instead of the lowest observed value, which is likely a lucky noisy draw.
    /// In that case surrogate models are trained, as done by [`suggest`](Self::suggest).
    pub fn best_index(
        &self,
        x_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> usize {
        let xtypes = &self.solver.config.xtypes;
        let x_data = to_continuous_space(xtypes, x_data);
        self.solver.best_index(&x_data, y_data)
    }
}

/// Egor Service
pub type EgorServiceBuilder = EgorServiceFactory<Cstr>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_moe::{CorrelationSpec, RegressionSpec};
    use ndarray::{ArrayView2, Axis, array, concatenate};

    use ndarray_stats::QuantileExt;

    fn xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        (x - 3.5) * ((x - 3.5) / std::f64::consts::PI).mapv(|v| v.sin())
    }

    #[test]
    fn test_xsinx_egor_builder() {
        let ego = EgorServiceBuilder::optimize()
            .configure(|conf| {
                conf.configure_gp(|gp| {
                    gp.regression_spec(RegressionSpec::CONSTANT)
                        .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
                })
                .infill_strategy(InfillStrategy::EI)
                .seed(42)
            })
            .min_within(&array![[0., 25.]])
            .expect("Egor configured");

        let mut doe = array![[0.], [7.], [20.], [25.]];
        let mut y_doe = xsinx(&doe.view());
        for _i in 0..10 {
            let x_suggested = ego.suggest(&doe, &y_doe);

            doe = concatenate![Axis(0), doe, x_suggested];
            y_doe = xsinx(&doe.view());
        }

        let expected = -15.1;
        let y_opt = y_doe.min().unwrap();
        assert_abs_diff_eq!(expected, *y_opt, epsilon = 1e-1);
        // Without noise modeling, the best point is the observed minimum
        assert_eq!(ego.best_index(&doe, &y_doe), y_doe.argmin().unwrap().0);
    }

    /// xsinx with a deterministic pseudo-noise of amplitude 0.5
    fn noisy_xsinx(x: &ArrayView2<f64>) -> Array2<f64> {
        xsinx(x) + x.mapv(|v| 0.5 * (1000. * v).sin())
    }

    fn noisy_service() -> EgorServiceApi<GpMixtureParams<f64>> {
        EgorServiceBuilder::optimize()
            .configure(|conf| {
                conf.configure_gp(|gp| {
                    gp.noise_ratio(crate::ParamTuning::Optimized {
                        init: 1e-2,
                        bounds: (1e-10, 10.),
                    })
                })
                .seed(42)
            })
            .min_within(&array![[0., 25.]])
            .expect("Egor configured")
    }

    #[test]
    fn test_best_index_with_noisy_outlier() {
        use egobox_doe::{Lhs, SamplingMethod};
        use ndarray_rand::rand::SeedableRng;
        use rand_xoshiro::Xoshiro256Plus;

        let doe = Lhs::new(&array![[0., 25.]])
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .sample(20);
        let mut y_doe = noisy_xsinx(&doe.view());
        // A large negative noise far from the optimum (x ~ 18.9) makes it the observed minimum
        let spike = doe.column(0).mapv(|x| (x - 5.).abs()).argmin().unwrap();
        y_doe[[spike, 0]] = -20.;

        let egor = EgorServiceBuilder::optimize()
            .configure(|conf| conf.seed(42))
            .min_within(&array![[0., 25.]])
            .expect("Egor configured");
        assert_eq!(egor.best_index(&doe, &y_doe), spike);

        let best = noisy_service().best_index(&doe, &y_doe);
        println!("best x = {}, spike x = {}", doe.row(best), doe.row(spike));
        assert_ne!(best, spike);
        assert_abs_diff_eq!(doe[[best, 0]], 18.9, epsilon = 1.);
    }

    #[test]
    fn test_noisy_xsinx_egor_service() {
        let egor = noisy_service();
        let mut doe = array![[0.], [7.], [20.], [25.]];
        let mut y_doe = noisy_xsinx(&doe.view());
        for _i in 0..10 {
            let x_suggested = egor.suggest(&doe, &y_doe);
            doe = concatenate![Axis(0), doe, x_suggested];
            y_doe = noisy_xsinx(&doe.view());
        }
        let best = egor.best_index(&doe, &y_doe);
        println!("best x = {}, y = {}", doe.row(best), y_doe.row(best));
        assert_abs_diff_eq!(doe[[best, 0]], 18.9, epsilon = 0.5);
    }

    #[test]
    fn test_best_index_with_several_objectives() {
        let egor = EgorServiceBuilder::optimize()
            .configure(|conf| conf.n_obj(2).seed(42))
            .min_within(&array![[0., 1.]])
            .expect("Egor configured");
        let doe = array![[0.], [0.25], [0.5], [0.75], [1.]];
        let y_doe = concatenate![
            Axis(1),
            doe.mapv(|x| x * x),
            doe.mapv(|x| (x - 1.) * (x - 1.))
        ];
        let expected = crate::find_compromise_index(
            &y_doe,
            &Array2::zeros((doe.nrows(), 0)),
            2,
            &ndarray::Array1::zeros(0),
        );
        assert_eq!(Some(egor.best_index(&doe, &y_doe)), expected);
    }
}

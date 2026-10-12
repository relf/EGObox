//! Disciplinary Gaussian process surrogates.
//!
//! Each scalar coupling variable computed by a discipline is modelled by its own GP,
//! with a constant mean and a squared exponential kernel, trained on the disciplinary
//! inputs `[z_i, y_c(i)]` (specification, section 2).

use crate::errors::{EgmdoError, Result};
use crate::problem::MdoProblem;
use egobox_doe::{Lhs, SamplingMethod};
use egobox_gp::correlation_models::SquaredExponentialCorr;
use egobox_gp::mean_models::ConstantMean;
use egobox_gp::{GaussianProcess, ThetaTuning};
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2, array, concatenate};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// The GP used to model a scalar disciplinary output
pub type DisciplinaryGp = GaussianProcess<f64, ConstantMean, SquaredExponentialCorr>;

/// The distance, in inputs normalized by the discipline input bounds, under which
/// a new training point is considered as a duplicate of an existing one (decision D14).
pub const DUPLICATE_TOLERANCE: f64 = 1e-10;

/// Settings of the disciplinary GPs
#[derive(Clone, Debug)]
pub struct GpSettings {
    /// Initial value of the `theta` hyperparameters (inverse length scales)
    pub theta_init: f64,
    /// Bounds of the `theta` hyperparameters
    pub theta_bounds: (f64, f64),
}

impl Default for GpSettings {
    /// Length scales in `[0.3, 100]` on normalized inputs, as in the EGMDO paper,
    /// i.e. `theta = 1 / length scale` in `[0.01, 3.33]`.
    fn default() -> Self {
        GpSettings {
            theta_init: 0.1,
            theta_bounds: (1. / 100., 1. / 0.3),
        }
    }
}

impl GpSettings {
    /// Trains a GP on inputs `x` and scalar outputs `y`
    pub fn train(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &Array1<f64>,
    ) -> Result<DisciplinaryGp> {
        let gp = DisciplinaryGp::params(ConstantMean::default(), SquaredExponentialCorr::default())
            .theta_tuning(ThetaTuning::Full {
                init: array![self.theta_init],
                bounds: array![self.theta_bounds],
            })
            .fit(&Dataset::new(x.to_owned(), y.to_owned()))?;
        Ok(gp)
    }
}

/// Samples an initial design of experiments of `n` points for discipline `i`
/// by Latin hypercube sampling over `Z_i x C(i)`.
pub fn initial_doe(problem: &MdoProblem, i: usize, n: usize, seed: u64) -> Array2<f64> {
    Lhs::new(&problem.discipline_limits(i))
        .with_rng(Xoshiro256Plus::seed_from_u64(seed))
        .sample(n)
}

/// The GP surrogates of all disciplines, with their training data
#[derive(Clone)]
pub struct DisciplinaryModels {
    settings: GpSettings,
    /// Per discipline: training inputs `(n, input_dim)` and outputs `(n, output_dim)`
    data: Vec<(Array2<f64>, Array2<f64>)>,
    /// Per discipline: widths of the input bounds, used to normalize distances
    scales: Vec<Array1<f64>>,
    /// Per discipline: one GP per output
    gps: Vec<Vec<DisciplinaryGp>>,
}

impl DisciplinaryModels {
    /// Trains the GPs of every discipline of `problem` on the given data:
    /// for discipline `i`, `data[i] = (x, y)` with `x` of shape `(n, input_dim)`
    /// and `y` of shape `(n, output_dim)`.
    pub fn fit(
        problem: &MdoProblem,
        data: Vec<(Array2<f64>, Array2<f64>)>,
        settings: GpSettings,
    ) -> Result<Self> {
        if data.len() != problem.disciplines().len() {
            return Err(EgmdoError::InvalidData(format!(
                "training data given for {} disciplines, expected {}",
                data.len(),
                problem.disciplines().len()
            )));
        }
        for (i, (x, y)) in data.iter().enumerate() {
            check_data(problem, i, x, y)?;
        }
        let scales = (0..problem.disciplines().len())
            .map(|i| {
                let limits = problem.discipline_limits(i);
                &limits.column(1) - &limits.column(0)
            })
            .collect();
        let mut models = DisciplinaryModels {
            settings,
            data,
            scales,
            gps: vec![],
        };
        models.gps = (0..models.data.len())
            .map(|i| models.train(i))
            .collect::<Result<_>>()?;
        Ok(models)
    }

    fn train(&self, i: usize) -> Result<Vec<DisciplinaryGp>> {
        let (x, y) = &self.data[i];
        y.columns()
            .into_iter()
            .map(|yq| self.settings.train(x, &yq.to_owned()))
            .collect()
    }

    /// Adds training points `(x, y)` to discipline `i` and retrains its GPs.
    ///
    /// Points closer than [DUPLICATE_TOLERANCE] to an existing training point (in inputs
    /// normalized by the discipline input bounds) are skipped. Returns the number of
    /// points actually added; the GPs are retrained only if it is not zero.
    pub fn add_points(
        &mut self,
        problem: &MdoProblem,
        i: usize,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<usize> {
        check_data(problem, i, x, y)?;
        let mut added = 0;
        for (xk, yk) in x.rows().into_iter().zip(y.rows()) {
            let (xt, yt) = &self.data[i];
            let duplicate = xt.rows().into_iter().any(|xr| {
                ((&xr - &xk) / &self.scales[i]).mapv(|v| v * v).sum().sqrt() < DUPLICATE_TOLERANCE
            });
            if duplicate {
                log::warn!(
                    "Discipline '{}': training point {xk} already exists, skipped",
                    problem.disciplines()[i].name()
                );
                continue;
            }
            let xt = concatenate![Axis(0), *xt, xk.insert_axis(Axis(0))];
            let yt = concatenate![Axis(0), *yt, yk.insert_axis(Axis(0))];
            self.data[i] = (xt, yt);
            added += 1;
        }
        if added > 0 {
            self.gps[i] = self.train(i)?;
        }
        Ok(added)
    }

    /// Predicts the means and standard deviations of the outputs of discipline `i`
    /// at inputs `x` of shape `(n, input_dim)`, as two `(n, output_dim)` matrices.
    pub fn predict(&self, i: usize, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let gps = &self.gps[i];
        let mut mean = Array2::zeros((x.nrows(), gps.len()));
        let mut std = Array2::zeros((x.nrows(), gps.len()));
        for (q, gp) in gps.iter().enumerate() {
            mean.column_mut(q).assign(&gp.predict(x)?);
            std.column_mut(q)
                .assign(&gp.predict_var(x)?.mapv(|v| v.max(0.).sqrt()));
        }
        Ok((mean, std))
    }

    /// The GPs of discipline `i`, one per output
    pub fn gps(&self, i: usize) -> &[DisciplinaryGp] {
        &self.gps[i]
    }

    /// The training inputs and outputs of discipline `i`
    pub fn training_data(&self, i: usize) -> (&Array2<f64>, &Array2<f64>) {
        (&self.data[i].0, &self.data[i].1)
    }

    /// The number of training points of discipline `i`, i.e. its number of true evaluations
    pub fn n_evaluations(&self, i: usize) -> usize {
        self.data[i].0.nrows()
    }

    /// The GP settings
    pub fn settings(&self) -> &GpSettings {
        &self.settings
    }
}

fn check_data(problem: &MdoProblem, i: usize, x: &Array2<f64>, y: &Array2<f64>) -> Result<()> {
    let d = problem.disciplines().get(i).ok_or_else(|| {
        EgmdoError::InvalidData(format!(
            "discipline {i} does not exist ({} disciplines)",
            problem.disciplines().len()
        ))
    })?;
    if x.nrows() != y.nrows() || x.ncols() != d.input_dim() || y.ncols() != d.output_dim() {
        return Err(EgmdoError::InvalidData(format!(
            "discipline '{}' expects ({n}, {}) inputs and ({n}, {}) outputs, got {:?} and {:?}",
            d.name(),
            d.input_dim(),
            d.output_dim(),
            x.shape(),
            y.shape(),
            n = x.nrows(),
        )));
    }
    if x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err(EgmdoError::InvalidData(format!(
            "discipline '{}': training data must be finite",
            d.name()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems;
    use approx::assert_abs_diff_eq;

    fn toy_models(n: usize) -> (problems::TestProblem, DisciplinaryModels) {
        let tp = problems::toy();
        let data = (0..2)
            .map(|i| {
                let x = initial_doe(&tp.problem, i, n, 42 + i as u64);
                let y = tp.eval_discipline(i, &x);
                (x, y)
            })
            .collect();
        let models = DisciplinaryModels::fit(&tp.problem, data, GpSettings::default()).unwrap();
        (tp, models)
    }

    #[test]
    fn test_initial_doe() {
        let tp = problems::compound_cylinder();
        let doe = initial_doe(&tp.problem, 1, 8, 0);
        let limits = tp.problem.discipline_limits(1);
        assert_eq!(doe.dim(), (8, 4));
        for row in doe.rows() {
            for (v, l) in row.iter().zip(limits.rows()) {
                assert!(l[0] <= *v && *v <= l[1]);
            }
        }
        assert_eq!(doe, initial_doe(&tp.problem, 1, 8, 0), "reproducible");
        assert_ne!(doe, initial_doe(&tp.problem, 1, 8, 1));
    }

    #[test]
    fn test_models_interpolate_training_data() {
        let (_, models) = toy_models(6);
        for i in 0..2 {
            let (x, y) = models.training_data(i);
            let (mean, std) = models.predict(i, x).unwrap();
            assert_abs_diff_eq!(mean, *y, epsilon = 1e-4 * y.mapv(f64::abs).sum());
            assert!(
                std.iter().all(|&s| s < 1e-3),
                "std at training points: {std}"
            );
            assert_eq!(models.n_evaluations(i), 6);
            assert_eq!(models.gps(i).len(), 1);
        }
    }

    #[test]
    fn test_add_points_skips_duplicates() {
        let (tp, mut models) = toy_models(5);
        let x_new = array![[-3., 7.], [4., 20.]];
        let y_new = tp.eval_discipline(0, &x_new);
        assert_eq!(
            models.add_points(&tp.problem, 0, &x_new, &y_new).unwrap(),
            2
        );
        assert_eq!(models.n_evaluations(0), 7);
        // The same points again: nothing is added
        assert_eq!(
            models.add_points(&tp.problem, 0, &x_new, &y_new).unwrap(),
            0
        );
        assert_eq!(models.n_evaluations(0), 7);
        assert_eq!(models.n_evaluations(1), 5);
        let (mean, _) = models.predict(0, &x_new).unwrap();
        assert_abs_diff_eq!(mean, y_new, epsilon = 1e-3);
    }

    #[test]
    fn test_invalid_training_data() {
        let tp = problems::toy();
        let x = initial_doe(&tp.problem, 0, 5, 0);
        let y = tp.eval_discipline(0, &x);
        let bad = vec![(x.clone(), y.clone())];
        assert!(DisciplinaryModels::fit(&tp.problem, bad, GpSettings::default()).is_err());
        let bad = vec![(x.clone(), y.clone()), (x.clone(), Array2::zeros((5, 2)))];
        assert!(DisciplinaryModels::fit(&tp.problem, bad, GpSettings::default()).is_err());
        let mut y_nan = y.clone();
        y_nan[[0, 0]] = f64::NAN;
        let bad = vec![(x.clone(), y_nan), (x, y)];
        assert!(DisciplinaryModels::fit(&tp.problem, bad, GpSettings::default()).is_err());
    }
}

use crate::IaeAlphaPlotData;
use crate::errors::Result;
use crate::surrogates::*;
use crate::types::*;
use ndarray::{Array1, Array2, ArrayView2};

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

use crate::MoeError;
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;

/// A wrapper surrogate applying an affine transform to another surrogate's predictions.
///
/// Given an inner surrogate predicting `f(x)` this surrogate returns `scale * f(x) + offset`.
///
/// This avoids training a duplicate GP for the second internal constraint
/// produced by equality (`Eq`) or between (`Btw`) constraint specifications.
///
/// For `Eq(z)`: inner predicts `c - z`, transform with `scale=-1, offset=0` gives `z - c`.
/// For `Btw(lo, hi)`: inner predicts `lo - c`, transform with `scale=-1, offset=lo-hi` gives `c - hi`.
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct AffinedSurrogate {
    /// The underlying trained surrogate
    inner: Box<dyn MixtureGpSurrogate>,
    /// Multiplicative scale factor applied to inner predictions
    scale: f64,
    /// Constant offset added after scaling
    offset: f64,
    /// Cached transformed training data
    training_data: (Array2<f64>, Array1<f64>),
}

impl AffinedSurrogate {
    /// Create a surrogate that predicts `scale * inner(x) + offset`.
    ///
    /// The training data is derived from the inner surrogate's training data
    /// by applying the same affine transform to the target values.
    pub fn new(inner: Box<dyn MixtureGpSurrogate>, scale: f64, offset: f64) -> Self {
        let (xt, yt) = inner.training_data();
        let transformed_yt = yt * scale + offset;
        AffinedSurrogate {
            training_data: (xt.clone(), transformed_yt),
            inner,
            scale,
            offset,
        }
    }
}

impl std::fmt::Display for AffinedSurrogate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Affine({} * {} + {})",
            self.scale, self.inner, self.offset
        )
    }
}

impl Clustered for AffinedSurrogate {
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters()
    }

    fn recombination(&self) -> Recombination<f64> {
        self.inner.recombination()
    }

    fn to_clustering(&self) -> Clustering {
        self.inner.to_clustering()
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpSurrogate for AffinedSurrogate {
    fn dims(&self) -> (usize, usize) {
        self.inner.dims()
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let pred = self.inner.predict(x)?;
        Ok(pred * self.scale + self.offset)
    }

    fn predict_var(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        // Var(a*f + b) = a² * Var(f)
        let var = self.inner.predict_var(x)?;
        Ok(var * (self.scale * self.scale))
    }

    fn predict_valvar(&self, x: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let (val, var) = self.inner.predict_valvar(x)?;
        Ok((
            val * self.scale + self.offset,
            var * (self.scale * self.scale),
        ))
    }

    #[cfg(feature = "persistent")]
    fn save(&self, path: &str, format: GpFileFormat) -> Result<()> {
        let mut file = fs::File::create(path).unwrap();
        let bytes = match format {
            GpFileFormat::Json => {
                serde_json::to_vec(self as &dyn GpSurrogate).map_err(MoeError::SaveJsonError)?
            }
            GpFileFormat::Binary => {
                bincode::serde::encode_to_vec(self as &dyn GpSurrogate, bincode::config::standard())
                    .map_err(MoeError::SaveBinaryError)?
            }
        };
        file.write_all(&bytes)?;
        Ok(())
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpSurrogateExt for AffinedSurrogate {
    fn predict_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let grad = self.inner.predict_gradients(x)?;
        Ok(grad * self.scale)
    }

    fn predict_var_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        // d/dx Var(a*f + b) = a² * d/dx Var(f)
        let var_grad = self.inner.predict_var_gradients(x)?;
        Ok(var_grad * (self.scale * self.scale))
    }

    fn predict_valvar_gradients(&self, x: &ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let (val_grad, var_grad) = self.inner.predict_valvar_gradients(x)?;
        Ok((val_grad * self.scale, var_grad * (self.scale * self.scale)))
    }

    fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> Result<Array2<f64>> {
        let samples = self.inner.sample(x, n_traj)?;
        Ok(samples * self.scale + self.offset)
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpQualityAssurance for AffinedSurrogate {
    fn training_data(&self) -> &(Array2<f64>, Array1<f64>) {
        &self.training_data
    }

    // Quality metrics are invariant under negation + offset (linear transform),
    // so we delegate to the inner surrogate.
    fn q2_k(&self, kfold: usize) -> f64 {
        self.inner.q2_k(kfold)
    }

    fn q2(&self) -> f64 {
        self.inner.q2()
    }

    fn pva_k(&self, kfold: usize) -> f64 {
        self.inner.pva_k(kfold)
    }

    fn pva(&self) -> f64 {
        self.inner.pva()
    }

    fn iae_alpha_k(&self, kfold: usize) -> f64 {
        self.inner.iae_alpha_k(kfold)
    }

    fn iae_alpha_k_score_with_plot(&self, kfold: usize, plot_data: &mut IaeAlphaPlotData) -> f64 {
        self.inner.iae_alpha_k_score_with_plot(kfold, plot_data)
    }

    fn iae_alpha(&self) -> f64 {
        self.inner.iae_alpha()
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl MixtureGpSurrogate for AffinedSurrogate {
    fn experts(&self) -> &Vec<Box<dyn FullGpSurrogate>> {
        self.inner.experts()
    }

    /// Update the affined mixture with new data points
    ///
    /// Note: `y_new` is given in the affined (derived) output space while the inner
    /// surrogate is trained in the primary output space: the affine transform is
    /// inverted before delegating so that the inner surrogate training data stays
    /// consistent.
    fn update(
        &self,
        x_new: &ndarray::ArrayView2<f64>,
        y_new: &ndarray::ArrayView1<f64>,
    ) -> crate::errors::Result<Box<dyn MixtureGpSurrogate>> {
        if self.scale == 0. {
            return Err(MoeError::InvalidValueError(
                "Cannot update an affined surrogate with a null scale".to_string(),
            ));
        }
        // Invert the affine transform to get primary-space target values as the
        // inner surrogate is trained on primary-space data.
        let y_inner = (y_new.to_owned() - self.offset) / self.scale;
        let updated_inner = self.inner.update(x_new, &y_inner.view())?;
        Ok(Box::new(AffinedSurrogate::new(
            updated_inner,
            self.scale,
            self.offset,
        )))
    }
}

impl Clone for AffinedSurrogate {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            scale: self.scale,
            offset: self.offset,
            training_data: self.training_data.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpMixtureParams, NbClusters, SurrogateBuilder};
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    /// Train a single-expert mixture on primary-space data: y = 0.5 * x - 1
    fn train_inner() -> Box<dyn MixtureGpSurrogate> {
        let xt = array![[0.], [1.], [2.], [3.], [4.], [5.]];
        let yt = array![-1., -0.5, 0., 0.5, 1., 1.5];
        GpMixtureParams::new()
            .n_clusters(NbClusters::fixed(1))
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .train(xt.view(), yt.view())
            .expect("inner mixture training")
    }

    #[test]
    fn test_affined_update_inverse_transform() {
        // Eq-like derived constraint: derived value = -1 * primary value
        let scale = -1.;
        let offset = 0.;
        let affined = AffinedSurrogate::new(train_inner(), scale, offset);

        // New point with its value given in the derived space
        let x_new = array![[2.5]];
        let y_primary_new = 0.5 * 2.5 - 1.;
        let y_derived_new = scale * y_primary_new + offset;
        let updated = affined
            .update(&x_new.view(), &array![y_derived_new].view())
            .expect("affined update");

        // The updated surrogate interpolates the new point in the derived space
        let pred = updated.predict(&x_new.view()).expect("prediction");
        assert_abs_diff_eq!(pred[0], y_derived_new, epsilon = 1e-2);

        // ... and still predicts previously trained points in the derived space
        let pred_old = updated.predict(&array![[1.]].view()).expect("prediction");
        assert_abs_diff_eq!(pred_old[0], scale * (0.5 * 1. - 1.), epsilon = 1e-2);

        // The updated training data contains the derived-space target values
        let (xt_m, yt_m) = updated.training_data();
        assert_eq!(xt_m.nrows(), 7);
        assert_abs_diff_eq!(yt_m[6], y_derived_new, epsilon = 1e-10);
    }

    #[test]
    fn test_affined_clone_predictions() {
        let affined = AffinedSurrogate::new(train_inner(), -1., 0.);
        let x = array![[0.5], [3.5]];
        let models = vec![Box::new(affined) as Box<dyn MixtureGpSurrogate>];
        let pred = models[0].predict(&x.view()).expect("prediction");
        // Clone the models vector as done for model persistency in the solver state
        let cloned_models: Vec<Box<dyn MixtureGpSurrogate>> = models.clone();
        let pred_cloned = cloned_models[0].predict(&x.view()).expect("prediction");
        assert_abs_diff_eq!(pred, pred_cloned, epsilon = 1e-12);
    }
}

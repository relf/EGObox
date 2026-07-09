use crate::IaeAlphaPlotData;
use crate::errors::Result;
use crate::surrogates::*;
use crate::types::*;
use ndarray::{Array1, Array2, ArrayView2};

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "persistent")]
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
}

/// Clone a surrogate via serialization round-trip.
///
/// This produces an independent owned copy suitable for wrapping
/// in [`AffinedSurrogate`].
#[cfg(feature = "persistent")]
pub fn clone_surrogate(surrogate: &dyn MixtureGpSurrogate) -> Box<dyn MixtureGpSurrogate> {
    let json = serde_json::to_string(surrogate).expect("surrogate serialization");
    serde_json::from_str(&json).expect("surrogate deserialization")
}

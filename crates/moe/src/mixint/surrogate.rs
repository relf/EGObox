//! Mixed-integer mixture of Gaussian processes

use crate::errors::{MoeError, Result};
use crate::persistence::{read_model, write_model};
use crate::xtypes::{XType, discrete};
use crate::{
    Clustered, Clustering, FullGpSurrogate, GpMetrics, GpMixture, GpMixtureParams,
    GpQualityAssurance, GpSurrogate, GpSurrogateExt, IaeAlphaPlotData, MixtureGpSurrogate,
    Recombination,
};
use linfa::traits::PredictInplace;
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, CowArray, Data, Ix1, Ix2, concatenate};

use serde::{Deserialize, Serialize};

use super::space::cast_to_model_space;
use super::{MixintGpMixtureParams, MixintGpMixtureValidParams};
use crate::GpFileFormat;

/// The Moe model that takes into account XType specifications
#[derive(Serialize, Deserialize, Clone)]
pub struct MixintGpMixture {
    /// the decorated Moe
    pub(super) moe: GpMixture,
    /// The input specifications (None means all continuous variables, no casting needed)
    pub(super) xtypes: Option<Vec<XType>>,
    /// whether training input data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    pub(super) work_in_folded_space: bool,
    /// Training inputs
    pub(super) training_data: (Array2<f64>, Array1<f64>),
    /// Parameters used to trin this model
    pub(super) params: MixintGpMixtureValidParams,
}

impl std::fmt::Display for MixintGpMixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if let Some(xtypes) = &self.xtypes {
            if discrete(xtypes) { "MixInt" } else { "" }
        } else {
            ""
        };
        write!(f, "{}{}", prefix, self.moe)
    }
}

impl Clustered for MixintGpMixture {
    fn n_clusters(&self) -> usize {
        self.moe.n_clusters()
    }

    fn recombination(&self) -> Recombination<f64> {
        self.moe.recombination()
    }

    /// Convert to clustering
    fn to_clustering(&self) -> Clustering {
        Clustering::new(self.moe.gmx().clone(), self.moe.recombination())
    }
}

impl MixintGpMixture {
    /// Cast input data to the space expected by the underlying MoE model.
    ///
    /// If xtypes is None, returns the input as-is without any transformation.
    /// If working in folded space, unfolds enum dimensions to one-hot masks.
    /// Then projects continuous values to valid discrete values.
    fn cast_to_model_space<'a>(&self, x: &'a CowArray<'a, f64, Ix2>) -> CowArray<'a, f64, Ix2> {
        cast_to_model_space(
            self.xtypes.as_deref(),
            self.work_in_folded_space,
            CowArray::from(x),
        )
    }
}

#[typetag::serde]
impl GpSurrogate for MixintGpMixture {
    fn dims(&self) -> (usize, usize) {
        self.moe.dims()
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.predict(&xcast)
    }

    fn predict_var(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.predict_var(&xcast)
    }

    fn predict_valvar(&self, x: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.predict_valvar(&xcast)
    }

    /// Save Moe model in given file.
    fn save(&self, path: &str, format: GpFileFormat) -> Result<()> {
        write_model(path, self, format)
    }
}

impl MixintGpMixture {
    /// Constructor of mixture of experts parameters, automatically choosing between typed and continuous-only versions based on the presence of `xtypes`.
    pub fn params(xtypes: Option<&[XType]>) -> MixintGpMixtureParams {
        if let Some(xtypes) = xtypes {
            Self::params_with_types(xtypes)
        } else {
            Self::params_continuous()
        }
    }

    /// Constructor of mixture of experts parameters
    pub fn params_with_types(xtypes: &[XType]) -> MixintGpMixtureParams {
        MixintGpMixtureParams::new(Some(xtypes), &GpMixtureParams::new())
    }

    /// Constructor of mixture of experts parameters for continuous-only variables
    pub fn params_continuous() -> MixintGpMixtureParams {
        MixintGpMixtureParams::new(None, &GpMixtureParams::new())
    }

    /// Update the mixint mixture of experts with new data points.
    ///
    /// Casts/unfolds the new points into the space the inner moe is trained in
    /// (mirroring `_train`/`_train_on_clusters`) before delegating to the
    /// underlying `GpMixture::update`, and extends the (original-space)
    /// `training_data` bookkeeping accordingly, so callers relying on
    /// `training_data()` (e.g. incremental "fast path" updates) observe the
    /// true, up-to-date point count instead of the last full-training one.
    ///
    /// # Arguments
    ///
    /// * `x_new` - New input data points as a (n_new, nx) matrix
    /// * `y_new` - New output data values as a (n_new,) vector
    ///
    /// # Returns
    ///
    /// A new `MixintGpMixture` instance updated with the new data
    pub fn update(
        self,
        x_new: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_new: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    ) -> Result<MixintGpMixture> {
        let xcast = cast_to_model_space(
            self.xtypes.as_deref(),
            self.work_in_folded_space,
            CowArray::from(x_new),
        );

        // Update the underlying moe (consume self.moe)
        let updated_moe = self.moe.update(&xcast, y_new)?;

        // Extend training data bookkeeping (kept in the original, un-cast space,
        // like `_train`/`_train_on_clusters` do) instead of dropping the new points.
        let x_combined = concatenate(Axis(0), &[self.training_data.0.view(), x_new.view()])
            .map_err(|e| MoeError::GpError(egobox_gp::GpError::InvalidValueError(e.to_string())))?;
        let y_combined = concatenate(Axis(0), &[self.training_data.1.view(), y_new.view()])
            .map_err(|e| MoeError::GpError(egobox_gp::GpError::InvalidValueError(e.to_string())))?;

        // Rebuild MixintGpMixture with updated moe, moving fields
        Ok(MixintGpMixture {
            params: self.params, // Moved
            moe: updated_moe,
            xtypes: self.xtypes, // Moved
            training_data: (x_combined, y_combined),
            work_in_folded_space: self.work_in_folded_space,
        })
    }

    /// Load MixintGpMixture from given file.
    pub fn load(path: &str, format: GpFileFormat) -> Result<Box<MixintGpMixture>> {
        read_model(path, format).map(Box::new)
    }
}

#[typetag::serde]
impl GpSurrogateExt for MixintGpMixture {
    fn predict_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.predict_gradients(&xcast)
    }

    fn predict_var_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.predict_var_gradients(&xcast)
    }

    fn predict_valvar_gradients(&self, x: &ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.predict_valvar_gradients(&xcast)
    }

    fn predict_valvar_with_gradients(
        &self,
        x: &ArrayView2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>)> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.predict_valvar_with_gradients(&xcast)
    }

    fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> Result<Array2<f64>> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.sample(&xcast, n_traj)
    }

    fn predict_covariance(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let cow_x = CowArray::from(x);
        let xcast = self.cast_to_model_space(&cow_x);
        self.moe.predict_covariance(&xcast.view())
    }
}

impl GpMetrics<MoeError, MixintGpMixtureParams, Self> for MixintGpMixture {
    fn params(&self) -> MixintGpMixtureParams {
        self.params.clone().into()
    }

    fn training_data(&self) -> &(Array2<f64>, Array1<f64>) {
        &self.training_data
    }
}

#[typetag::serde]
impl GpQualityAssurance for MixintGpMixture {
    fn training_data(&self) -> &(Array2<f64>, Array1<f64>) {
        (self as &dyn GpMetrics<_, _, _>).training_data()
    }

    fn q2_k(&self, kfold: usize) -> f64 {
        (self as &dyn GpMetrics<_, _, _>).q2_k_score(kfold)
    }
    fn q2(&self) -> f64 {
        (self as &dyn GpMetrics<_, _, _>).q2_score()
    }

    fn pva_k(&self, kfold: usize) -> f64 {
        (self as &dyn GpMetrics<_, _, _>).pva_k_score(kfold)
    }
    fn pva(&self) -> f64 {
        (self as &dyn GpMetrics<_, _, _>).pva_score()
    }

    fn iae_alpha_k(&self, kfold: usize) -> f64 {
        (self as &dyn GpMetrics<_, _, _>).iae_alpha_k_score(kfold, None)
    }
    fn iae_alpha_k_score_with_plot(&self, kfold: usize, plot_data: &mut IaeAlphaPlotData) -> f64 {
        (self as &dyn GpMetrics<_, _, _>).iae_alpha_k_score(kfold, Some(plot_data))
    }
    fn iae_alpha(&self) -> f64 {
        (self as &dyn GpMetrics<_, _, _>).iae_alpha_score(None)
    }
}

#[typetag::serde]
impl MixtureGpSurrogate for MixintGpMixture {
    fn experts(&self) -> &Vec<Box<dyn FullGpSurrogate>> {
        self.moe.experts()
    }

    /// Update the mixint mixture with new data points
    fn update(
        &self,
        x_new: &ndarray::ArrayView2<f64>,
        y_new: &ndarray::ArrayView1<f64>,
    ) -> crate::errors::Result<Box<dyn MixtureGpSurrogate>> {
        // Clone self to call the consuming update method
        Ok(Box::new(MixintGpMixture::update(
            self.clone(),
            x_new,
            y_new,
        )?))
    }
}

impl<D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array1<f64>> for MixintGpMixture {
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<f64>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let values = self.moe.predict(x).expect("MixintGpMixture prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<f64> {
        Array1::zeros((x.nrows(),))
    }
}

#[cfg(test)]
mod tests {
    use crate::mixint::*;
    use crate::xtypes::XType;
    use crate::{CorrelationSpec, GpSurrogate};
    use approx::assert_abs_diff_eq;
    use egobox_doe::{LhsKind, SamplingMethod};

    use linfa::Dataset;
    use linfa::traits::Fit;

    use ndarray::{Array, Array1, Array2, Axis, array};

    #[test]
    fn test_mixint_moe_1d() {
        let xtypes = vec![XType::Int(0, 4)];

        let mixi = MixintContext::new(&xtypes);

        let surrogate_builder = MoeBuilder::new();
        let xt = array![[0.], [2.], [3.0], [4.]];
        let yt = array![0., 1.5, 0.9, 1.];
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let num = 5;
        let xtest = Array::linspace(0.0, 4.0, num).insert_axis(Axis(1));
        let ytest = mixi_moe.predict(&xtest.view()).expect("Predict val fail");
        let yvar = mixi_moe
            .predict_var(&xtest.view())
            .expect("Predict var fail");
        println!("{ytest:?}");
        assert_abs_diff_eq!(
            array![0., 0.7872696212255119, 1.5, 0.9, 1.],
            ytest,
            epsilon = 1e-3
        );
        println!("{yvar:?}");
        assert_abs_diff_eq!(
            array![0., 0.2852695228568877, 0., 0., 0.],
            yvar,
            epsilon = 1e-3
        );
        //println!("LOOCV = {}", mixi_moe.loocv_score());
    }

    fn ftest(x: &Array2<f64>) -> Array1<f64> {
        let mut y = x.column(0).to_owned() * x.column(0);
        y = &y + (x.column(1).to_owned() * x.column(1));
        y = &y * (x.column(2).mapv(|v| v + 1.));
        y
    }

    #[test]
    fn test_mixint_moe_3d() {
        let xtypes = vec![XType::Int(0, 5), XType::Float(0., 4.), XType::Enum(4)];

        let mixi = MixintContext::new(&xtypes);
        let mixi_lhs = mixi.create_lhs_sampling(LhsKind::default(), Some(0));

        let n = mixi.get_unfolded_dim() * 10;
        let xt = mixi_lhs.sample(n);
        let yt = ftest(&xt);

        let surrogate_builder =
            MoeBuilder::new().correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL);
        let ds = Dataset::new(xt, yt);
        let mixi_moe = mixi
            .create_surrogate(&surrogate_builder, &ds)
            .expect("Mixint surrogate creation");

        let ntest = 10;
        let mixi_lhs = mixi.create_lhs_sampling(LhsKind::default(), Some(42));

        let xtest = mixi_lhs.sample(ntest);
        let ytest = mixi_moe.predict(&xtest.view()).expect("Predict val fail");
        let ytrue = ftest(&xtest);
        assert_abs_diff_eq!(ytrue, ytest, epsilon = 2.0);
    }

    #[test]
    fn test_continuous_only_no_xtypes() {
        // Test that when xtypes is None, no casting is applied
        let surrogate_builder = MoeBuilder::new();
        let params = MixintGpMixtureParams::new_continuous(&surrogate_builder);

        // Verify xtypes is None
        assert!(params.xtypes().is_none());

        // Create some continuous training data
        let xt = array![[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let yt = array![1.0, 2.0, 3.0, 4.0];
        let ds = Dataset::new(xt.clone(), yt.clone());

        // Train the model
        let model = params.fit(&ds).expect("Failed to train model");

        // Predict should work without any issues
        let xtest = array![[1.5, 2.5]];
        let ytest = model.predict(&xtest.view()).expect("Predict failed");

        // Should get a reasonable prediction (linear interpolation)
        assert!(ytest[0] > 0.0);
    }
}

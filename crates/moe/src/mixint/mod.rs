//! Mixed-integer mixture of Gaussian processes
//!
//! This module exposes a GP mixture model featuring continuous relaxation to handle mixed-interger variables

mod context;
mod sampling;
mod space;
mod surrogate;

pub use context::MixintContext;
pub use sampling::MixintSampling;
pub use space::{as_continuous_limits, to_continuous_space, to_discrete_space};
pub use surrogate::MixintGpMixture;

use crate::errors::{MoeError, Result};
use crate::xtypes::XType;
use crate::{
    Clustering, CorrelationSpec, GpMixtureParams, MixtureGpSurrogate, NbClusters, Recombination,
    RegressionSpec,
};
use crate::{GpType, SurrogateBuilder};
use egobox_gp::{ParamTuning, ThetaTuning};
use linfa::traits::Fit;
use linfa::{DatasetBase, ParamGuard};
use ndarray::{ArrayBase, ArrayView1, ArrayView2, CowArray, Data, Ix1, Ix2};
use rand_xoshiro::Xoshiro256Plus;

use serde::{Deserialize, Serialize};
use space::cast_to_model_space;

/// Moe type builder for mixed-integer Egor optimizer
pub type MoeBuilder = GpMixtureParams<f64>;
/// A decorator of Moe surrogate builder that takes into account XType specifications
///
/// It allows to implement continuous relaxation over continuous Moe builder.
#[derive(Clone, Serialize, Deserialize)]
pub struct MixintGpMixtureValidParams {
    /// The surrogate factory
    gpmix_params: GpMixtureParams<f64>,
    /// The input specifications (None means all continuous variables)
    xtypes: Option<Vec<XType>>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    work_in_folded_space: bool,
}

impl MixintGpMixtureValidParams {
    /// Sets whether we want to work in folded space that is whether
    /// If set, input training data has to be given in folded space
    pub fn work_in_folded_space(&self) -> bool {
        self.work_in_folded_space
    }

    /// Sets the specification
    pub fn xtypes(&self) -> Option<&Vec<XType>> {
        self.xtypes.as_ref()
    }
}

/// Parameters for mixture of experts surrogate model
#[derive(Clone, Serialize, Deserialize)]
pub struct MixintGpMixtureParams(MixintGpMixtureValidParams);

impl MixintGpMixtureParams {
    /// Constructor given optional `xtypes` specifications and given surrogate builder
    ///
    /// Pass `None` for continuous-only variables (no mixed-integer handling)
    pub fn new(xtypes: Option<&[XType]>, surrogate_builder: &MoeBuilder) -> Self {
        MixintGpMixtureParams(MixintGpMixtureValidParams {
            gpmix_params: surrogate_builder.clone(),
            xtypes: xtypes.map(|xt| xt.to_vec()),
            work_in_folded_space: false,
        })
    }

    /// Creates a new instance without xtypes (all continuous variables)
    ///
    /// This is equivalent to `new(None, surrogate_builder)`
    pub fn new_continuous(surrogate_builder: &MoeBuilder) -> Self {
        MixintGpMixtureParams(MixintGpMixtureValidParams {
            gpmix_params: surrogate_builder.clone(),
            xtypes: None,
            work_in_folded_space: false,
        })
    }

    /// Sets whether we want to work in folded space
    pub fn work_in_folded_space(&mut self, wfs: bool) -> &mut Self {
        self.0.work_in_folded_space = wfs;
        self
    }

    /// Gets the domain specification
    pub fn xtypes(&self) -> Option<&Vec<XType>> {
        self.0.xtypes.as_ref()
    }

    /// Sets the number of clusters
    pub fn gp_type(mut self, gp_type: GpType<f64>) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.gp_type(gp_type);
        self
    }

    /// Sets the number of clusters
    pub fn n_clusters(mut self, n_clusters: NbClusters) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.n_clusters(n_clusters);
        self
    }

    /// Sets the recombination mode
    pub fn recombination(mut self, recombination: Recombination<f64>) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.recombination(recombination);
        self
    }

    /// Sets the regression models used in the mixture.
    ///
    /// Only GP models with regression models allowed by this specification
    /// will be used in the mixture.  
    pub fn regression_spec(mut self, regression_spec: RegressionSpec) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.regression_spec(regression_spec);
        self
    }

    /// Sets the correlation models used in the mixture.
    ///
    /// Only GP models with correlation models allowed by this specification
    /// will be used in the mixture.  
    pub fn correlation_spec(mut self, correlation_spec: CorrelationSpec) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.correlation_spec(correlation_spec);
        self
    }

    /// Sets the number of componenets retained during PLS dimension reduction.
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.kpls_dim(kpls_dim);
        self
    }

    /// Set theta hyper parameter tuning
    pub fn theta_tunings(mut self, theta_tunings: &[ThetaTuning<f64>]) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.theta_tunings(theta_tunings);
        self
    }

    /// Sets the number of hyperparameters optimization restarts
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.n_start(n_start);
        self
    }

    /// Sets The max number of likelihood optimization during hyperparameters optimization
    pub fn max_eval(mut self, max_eval: usize) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.max_eval(max_eval);
        self
    }

    /// Sets the noise-to-signal ratio tuning of full GP experts
    pub fn noise_ratio(mut self, noise_ratio: ParamTuning<f64>) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.noise_ratio(noise_ratio);
        self
    }

    /// Sets the random seed for the surrogate model training
    pub fn with_rng(mut self, rng: Xoshiro256Plus) -> Self {
        self.0.gpmix_params = self.0.gpmix_params.with_rng(rng);
        self
    }
}

impl MixintGpMixtureValidParams {
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

    fn _train(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    ) -> Result<MixintGpMixture> {
        let cow_xt = CowArray::from(xt);
        let xcast = self.cast_to_model_space(&cow_xt);
        let mixmoe = MixintGpMixture {
            moe: self
                .gpmix_params
                .clone()
                .check()?
                .train(&xcast, &yt.to_owned())?,
            xtypes: self.xtypes.clone(),
            work_in_folded_space: self.work_in_folded_space,
            training_data: (xt.to_owned(), yt.to_owned()),
            params: self.clone(),
        };
        Ok(mixmoe)
    }

    fn _train_on_clusters(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        clustering: &Clustering,
    ) -> Result<MixintGpMixture> {
        let cow_xt = CowArray::from(xt);
        let xcast = self.cast_to_model_space(&cow_xt);
        let mixmoe = MixintGpMixture {
            moe: self.gpmix_params.clone().check_ref()?.train_on_clusters(
                &xcast,
                &yt.to_owned(),
                clustering,
            )?,
            xtypes: self.xtypes.clone(),
            work_in_folded_space: self.work_in_folded_space,
            training_data: (xt.to_owned(), yt.to_owned()),
            params: self.clone(),
        };
        Ok(mixmoe)
    }
}

impl SurrogateBuilder for MixintGpMixtureParams {
    fn new_with_xtypes(xtypes: &[XType]) -> Self {
        MixintGpMixtureParams::new(Some(xtypes), &GpMixtureParams::new())
    }

    /// Sets the allowed regression models used in gaussian processes.
    fn set_regression_spec(&mut self, regression_spec: RegressionSpec) {
        self.0.gpmix_params.set_regression_spec(regression_spec);
    }

    /// Sets the allowed correlation models used in gaussian processes.
    fn set_correlation_spec(&mut self, correlation_spec: CorrelationSpec) {
        self.0.gpmix_params.set_correlation_spec(correlation_spec);
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>) {
        self.0.gpmix_params.set_kpls_dim(kpls_dim);
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: NbClusters) {
        self.0.gpmix_params.set_n_clusters(n_clusters);
    }

    fn set_recombination(&mut self, recombination: Recombination<f64>) {
        self.0.gpmix_params.set_recombination(recombination);
    }

    /// Sets the theta hyperparameter tuning strategy
    fn set_theta_tunings(&mut self, theta_tunings: &[ThetaTuning<f64>]) {
        self.0.gpmix_params.set_theta_tunings(theta_tunings);
    }

    fn set_optim_params(&mut self, n_start: usize, max_eval: usize) {
        self.0.gpmix_params.set_optim_params(n_start, max_eval);
    }

    /// Sets the noise-to-signal ratio tuning of full GP experts
    fn set_noise_ratio(&mut self, noise_ratio: ParamTuning<f64>) {
        self.0.gpmix_params.set_noise_ratio(noise_ratio);
    }

    /// Sets the random generator used during training (e.g. clustering)
    fn set_rng(&mut self, rng: Xoshiro256Plus) {
        self.0.gpmix_params.set_rng(rng);
    }

    fn train(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
    ) -> Result<Box<dyn MixtureGpSurrogate>> {
        let mixmoe = self.check_ref()?._train(&xt, &yt)?;
        Ok(mixmoe).map(|mixmoe| Box::new(mixmoe) as Box<dyn MixtureGpSurrogate>)
    }

    fn train_on_clusters(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn MixtureGpSurrogate>> {
        let mixmoe = self.check_ref()?._train_on_clusters(&xt, &yt, clustering)?;
        Ok(mixmoe).map(|mixmoe| Box::new(mixmoe) as Box<dyn MixtureGpSurrogate>)
    }
}

impl<D: Data<Elem = f64>> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>, MoeError>
    for MixintGpMixtureValidParams
{
    type Object = MixintGpMixture;

    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
    ) -> Result<Self::Object> {
        let x = dataset.records();
        let y = dataset.targets();
        self._train(x, y)
    }
}

impl ParamGuard for MixintGpMixtureParams {
    type Checked = MixintGpMixtureValidParams;
    type Error = MoeError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl From<MixintGpMixtureValidParams> for MixintGpMixtureParams {
    fn from(item: MixintGpMixtureValidParams) -> Self {
        MixintGpMixtureParams(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xtypes::XType;
    use crate::{
        CorrelationSpec, GpMixtureParams, GpSurrogate, NbClusters, Recombination, RegressionSpec,
        SurrogateBuilder,
    };

    use egobox_doe::{Lhs, SamplingMethod};
    use egobox_gp::{ParamTuning, ThetaTuning};
    use linfa::Dataset;
    use linfa::ParamGuard;
    use linfa::traits::Fit;
    use ndarray::{Array1, array};
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn test_surrogate_builder_setters_keep_xtypes() {
        let xtypes = vec![XType::Float(0., 1.), XType::Enum(3)];
        let mut builder = MixintGpMixtureParams::new(Some(&xtypes), &GpMixtureParams::new());
        builder.work_in_folded_space(true);
        let theta_tunings = [ThetaTuning::Fixed(array![0.5])];
        builder.set_regression_spec(RegressionSpec::LINEAR);
        builder.set_correlation_spec(CorrelationSpec::MATERN52);
        builder.set_kpls_dim(Some(1));
        builder.set_n_clusters(NbClusters::fixed(2));
        builder.set_recombination(Recombination::Smooth(Some(0.5)));
        builder.set_theta_tunings(&theta_tunings);
        builder.set_optim_params(3, 42);
        builder.set_noise_ratio(ParamTuning::Fixed(0.1));

        assert_eq!(
            format!("{:?}", builder.xtypes()),
            format!("{:?}", Some(&xtypes))
        );
        assert!(builder.check_ref().unwrap().work_in_folded_space());
        let inner = builder.0.gpmix_params.check_ref().unwrap();
        assert_eq!(inner.regression_spec(), RegressionSpec::LINEAR);
        assert_eq!(inner.correlation_spec(), CorrelationSpec::MATERN52);
        assert_eq!(inner.kpls_dim(), Some(1));
        assert_eq!(inner.n_clusters(), NbClusters::fixed(2));
        assert_eq!(inner.recombination(), Recombination::Smooth(Some(0.5)));
        assert_eq!(inner.theta_tunings(), &theta_tunings.to_vec());
        assert_eq!((inner.n_start(), inner.max_eval()), (3, 42));
        assert_eq!(inner.noise_ratio(), &ParamTuning::Fixed(0.1));
    }

    #[test]
    fn test_params_with_optional_xtypes() {
        // Test MixintGpMixtureParams::new with Some xtypes
        let xtypes = vec![XType::Float(0.0, 1.0), XType::Int(0, 5)];
        let builder = MoeBuilder::new();
        let params_with_xtypes = MixintGpMixtureParams::new(Some(&xtypes), &builder);
        assert!(params_with_xtypes.xtypes().is_some());
        assert_eq!(params_with_xtypes.xtypes().unwrap().len(), 2);

        // Test MixintGpMixtureParams::new with None (continuous only)
        let params_continuous = MixintGpMixtureParams::new(None::<&[XType]>, &builder);
        assert!(params_continuous.xtypes().is_none());

        // Test that params_continuous works for training
        let xt = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]];
        let yt = array![0.0, 0.5, 1.0];
        let ds = Dataset::new(xt, yt);
        let model = params_continuous.fit(&ds).expect("Training failed");

        let xtest = array![[0.25, 0.0]];
        let ytest = model.predict(&xtest.view()).expect("Prediction failed");
        assert!(ytest[0] >= 0.0 && ytest[0] <= 1.0);
    }

    #[test]
    fn test_params_continuous_helper() {
        // Test the params_continuous() helper method
        let builder = MoeBuilder::new();
        let params = MixintGpMixtureParams::new_continuous(&builder);
        assert!(params.xtypes().is_none());

        // Should be able to train and predict
        let xt = array![[0.0], [1.0], [2.0]];
        let yt = array![0.0, 1.0, 2.0];
        let ds = Dataset::new(xt, yt);
        let model = params.fit(&ds).expect("Training failed");

        let xtest = array![[1.5]];
        let ytest = model.predict(&xtest.view()).expect("Prediction failed");
        assert!((ytest[0] - 1.5).abs() < 0.5);
    }

    #[test]
    fn test_mixint_builder_noise_ratio() {
        use ndarray_rand::{RandomExt, rand_distr::Normal};

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(&array![[0., 10.]])
            .with_rng(rng.clone())
            .sample(30);
        let noise = Array1::random_using(30, Normal::new(0., 0.1).unwrap(), &mut rng);
        let yt = xt.column(0).mapv(f64::sin) + noise;

        let mut builder = MixintGpMixtureParams::new_with_xtypes(&[XType::Float(0., 10.)]);
        builder.set_n_clusters(NbClusters::fixed(1));
        builder.set_noise_ratio(ParamTuning::Optimized {
            init: 1e-2,
            bounds: (1e-8, 10.),
        });
        let model = builder.train(xt.view(), yt.view()).expect("Training");
        assert!(model.experts()[0].noise_variance() > 0.);
    }
}

//! Mixture of Gaussian process models used by the Egor solver

pub mod spec;

pub use egobox_moe::gpmix::mixint::{
    MixintGpMixture, MixintGpMixtureParams, MixintGpMixtureValidParams,
    as_continuous_limits, to_continuous_space, to_discrete_space,
};

use egobox_gp::ThetaTuning;
use egobox_moe::{
    Clustering, CorrelationSpec, GpMixtureParams, MixtureGpSurrogate, NbClusters, RegressionSpec, XType,
};
use ndarray::{ArrayView1, ArrayView2};

use linfa::ParamGuard;

use crate::Result;
use crate::SurrogateBuilder;

impl SurrogateBuilder for GpMixtureParams<f64> {
    /// Constructor from domain space specified with types
    /// **panic** if xtypes contains other types than continuous type `Float`
    fn new_with_xtypes(xtypes: &[XType]) -> Self {
        if crate::utils::discrete(xtypes) {
            panic!("GpMixtureParams cannot be created with discrete types!");
        }
        GpMixtureParams::new()
    }

    /// Sets the allowed regression models used in gaussian processes.
    fn set_regression_spec(&mut self, regression_spec: RegressionSpec) {
        *self = self.clone().regression_spec(regression_spec);
    }

    /// Sets the allowed correlation models used in gaussian processes.
    fn set_correlation_spec(&mut self, correlation_spec: CorrelationSpec) {
        *self = self.clone().correlation_spec(correlation_spec);
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>) {
        *self = self.clone().kpls_dim(kpls_dim);
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: NbClusters) {
        *self = self.clone().n_clusters(n_clusters);
    }

    /// Sets the mode of recombination to get the output prediction from experts prediction
    /// Onlyused if nb clusters is greater than one
    fn set_recombination(&mut self, recombination: egobox_moe::Recombination<f64>) {
        *self = self.clone().recombination(recombination);
    }

    /// Sets the theta tuning used by the expert during training.
    /// When only one element tuning is used for all clusters
    /// When several elements, the length should match the number of clusters
    fn set_theta_tunings(&mut self, theta_tunings: &[ThetaTuning<f64>]) {
        *self = self.clone().theta_tunings(theta_tunings);
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_optim_params(&mut self, n_start: usize, max_eval: usize) {
        *self = self.clone().n_start(n_start).max_eval(max_eval);
    }

    fn train(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
    ) -> Result<Box<dyn MixtureGpSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train(&xt, &yt)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn MixtureGpSurrogate>)
    }

    fn train_on_clusters(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn MixtureGpSurrogate>> {
        let checked = self.check_ref()?;
        let moe = checked.train_on_clusters(&xt, &yt, clustering)?;
        Ok(moe).map(|moe| Box::new(moe) as Box<dyn MixtureGpSurrogate>)
    }
}

impl SurrogateBuilder for MixintGpMixtureParams {
    fn new_with_xtypes(xtypes: &[XType]) -> Self {
        MixintGpMixtureParams::new(xtypes, &GpMixtureParams::new())
    }

    /// Sets the allowed regression models used in gaussian processes.
    fn set_regression_spec(&mut self, regression_spec: RegressionSpec) {
        self.0 = egobox_moe::gpmix::mixint::MixintGpMixtureValidParams {
            surrogate_builder: self
                .0
                .surrogate_builder
                .clone()
                .regression_spec(regression_spec),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    /// Sets the allowed correlation models used in gaussian processes.
    fn set_correlation_spec(&mut self, correlation_spec: CorrelationSpec) {
        self.0 = egobox_moe::gpmix::mixint::MixintGpMixtureValidParams {
            surrogate_builder: self
                .0
                .surrogate_builder
                .clone()
                .correlation_spec(correlation_spec),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>) {
        self.0 = egobox_moe::gpmix::mixint::MixintGpMixtureValidParams {
            surrogate_builder: self.0.surrogate_builder.clone().kpls_dim(kpls_dim),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: NbClusters) {
        self.0 = egobox_moe::gpmix::mixint::MixintGpMixtureValidParams {
            surrogate_builder: self.0.surrogate_builder.clone().n_clusters(n_clusters),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    fn set_recombination(&mut self, recombination: egobox_moe::Recombination<f64>) {
        self.0 = egobox_moe::gpmix::mixint::MixintGpMixtureValidParams {
            surrogate_builder: self
                .0
                .surrogate_builder
                .clone()
                .recombination(recombination),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    /// Sets the theta hyperparameter tuning strategy
    fn set_theta_tunings(&mut self, theta_tunings: &[ThetaTuning<f64>]) {
        self.0 = egobox_moe::gpmix::mixint::MixintGpMixtureValidParams {
            surrogate_builder: self
                .0
                .surrogate_builder
                .clone()
                .theta_tunings(theta_tunings),
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
    }

    fn set_optim_params(&mut self, n_start: usize, max_eval: usize) {
        let builder = self
            .0
            .surrogate_builder
            .clone()
            .n_start(n_start)
            .max_eval(max_eval);
        self.0 = egobox_moe::gpmix::mixint::MixintGpMixtureValidParams {
            surrogate_builder: builder,
            xtypes: self.0.xtypes.clone(),
            work_in_folded_space: self.0.work_in_folded_space,
        }
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

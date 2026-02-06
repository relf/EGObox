//! Surrogate model builder trait
//!
//! This module defines the `SurrogateBuilder` trait which is used by the Egor optimizer
//! to create and configure surrogate models.

use crate::{
    Clustering, CorrelationSpec, MixtureGpSurrogate, NbClusters, Recombination, RegressionSpec,
    XType, errors::Result,
};
use egobox_gp::ThetaTuning;
use ndarray::{ArrayView1, ArrayView2};

/// A trait for mixture of gp surrogate builder (aka gp configuration and training).
pub trait SurrogateBuilder: Clone + Sync {
    /// Constructor from domain space specified with types.
    fn new_with_xtypes(xtypes: &[XType]) -> Self;

    /// Sets the allowed regression models used in gaussian processes.
    fn set_regression_spec(&mut self, regression_spec: RegressionSpec);

    /// Sets the allowed correlation models used in gaussian processes.
    fn set_correlation_spec(&mut self, correlation_spec: CorrelationSpec);

    /// Sets the number of components to be used specifiying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>);

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: NbClusters);

    /// Sets the mode of recombination to get the output prediction from experts prediction
    fn set_recombination(&mut self, recombination: Recombination<f64>);

    /// Sets the hyperparameters tuning strategy
    fn set_theta_tunings(&mut self, theta_tunings: &[ThetaTuning<f64>]);

    /// Set likelihood optimization parameters
    fn set_optim_params(&mut self, n_start: usize, max_eval: usize);

    /// Train the surrogate with given training dataset (x, y)
    fn train(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
    ) -> Result<Box<dyn MixtureGpSurrogate>>;

    /// Train the surrogate with given training dataset (x, y) and given clustering
    fn train_on_clusters(
        &self,
        xt: ArrayView2<f64>,
        yt: ArrayView1<f64>,
        clustering: &Clustering,
    ) -> Result<Box<dyn MixtureGpSurrogate>>;
}

use crate::{GpMixtureParams, xtypes::discrete};
use linfa::ParamGuard;

impl SurrogateBuilder for GpMixtureParams<f64> {
    /// Constructor from domain space specified with types
    /// **panic** if xtypes contains other types than continuous type `Float`
    fn new_with_xtypes(xtypes: &[XType]) -> Self {
        if discrete(xtypes) {
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

    /// Sets the number of components to be used specifying PLS projection is used (a.k.a KPLS method).
    fn set_kpls_dim(&mut self, kpls_dim: Option<usize>) {
        *self = self.clone().kpls_dim(kpls_dim);
    }

    /// Sets the number of clusters used by the mixture of surrogate experts.
    fn set_n_clusters(&mut self, n_clusters: NbClusters) {
        *self = self.clone().n_clusters(n_clusters);
    }

    /// Sets the mode of recombination to get the output prediction from experts prediction
    /// Only used if nb clusters is greater than one
    fn set_recombination(&mut self, recombination: Recombination<f64>) {
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

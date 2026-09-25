use crate::errors::{MoeError, Result};
use crate::gaussian_mixture::GaussianMixture;
use crate::types::*;

use egobox_gp::GP_COBYLA_MAX_EVAL;
#[allow(unused_imports)]
use egobox_gp::correlation_models::{
    AbsoluteExponentialCorr, Matern32Corr, Matern52Corr, SquaredExponentialCorr,
};
#[allow(unused_imports)]
use egobox_gp::mean_models::{ConstantMean, LinearMean, QuadraticMean};
use linfa::{Float, ParamGuard};
use linfa_clustering::GaussianMixtureModel;
use ndarray::{Array1, Array2, Array3};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use serde::{Deserialize, Serialize};

pub use egobox_gp::{Inducings, SparseMethod, ThetaTuning};

/// Number of clusters specification
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NbClusters {
    /// Use a fixed number of clusters
    Fixed {
        /// Number of clusters
        nb: usize,
    },
    /// Find best number of clusters automatically optionally limited to max
    /// otherwize max is determined from number of samples
    Auto {
        /// Optional max number of clusters
        max: Option<usize>,
    },
}

impl Default for NbClusters {
    /// Default number of clusters is 1
    fn default() -> Self {
        Self::fixed(1)
    }
}

impl NbClusters {
    /// Return the number of clusters or 1 if auto
    pub fn n_or_else_one(&self) -> usize {
        match self {
            Self::Fixed { nb } => *nb,
            Self::Auto { max: _ } => 1,
        }
    }
    /// Constructor for fixed number of clusters
    pub fn fixed(nb: usize) -> Self {
        Self::Fixed { nb }
    }
    /// Constructor for automatic number of clusters
    pub fn auto() -> Self {
        Self::Auto { max: None }
    }
    /// Constructor for automatic number of clusters with a max
    pub fn automax(max: usize) -> Self {
        Self::Auto { max: Some(max) }
    }
    /// Is the number of clusters fixed
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed { nb: _ })
    }
    /// Is the number of clusters automatic
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto { max: _ })
    }
    /// Is the number of clusters mono (i.e. fixed to 1)
    pub fn is_mono(&self) -> bool {
        if let Self::Fixed { nb } = self
            && *nb == 1
        {
            return true;
        }
        false
    }
    /// Is the number of clusters multi (i.e. fixed to >1 or automatic)
    pub fn is_multi(&self) -> bool {
        if let Self::Fixed { nb } = self
            && *nb > 1
        {
            return true;
        }
        false
    }
}

/// Type of Gaussian Process
#[derive(Clone, Serialize, Deserialize)]
pub enum GpType<F: Float> {
    /// Full Gaussian Process
    FullGp,
    /// Sparse Gaussian Process used with large datasets (not iterpolating anymore)
    SparseGp {
        /// Used sparse method
        sparse_method: SparseMethod,
        /// Inducings
        inducings: Inducings<F>,
    },
}

/// Mixture of experts checked parameters
#[derive(Clone, Serialize, Deserialize)]
pub struct GpMixtureValidParams<F: Float> {
    /// Gp Type
    gp_type: GpType<F>,
    /// Number of clusters (i.e. number of experts)
    /// If 0 the number of clusters is deterined automatically
    n_clusters: NbClusters,
    /// [Recombination] mode
    recombination: Recombination<F>,
    /// Specification of GP regression models to be used
    regression_spec: RegressionSpec,
    /// Specification of GP correlation models to be used
    correlation_spec: CorrelationSpec,
    /// Theta hyperparameter tuning
    theta_tunings: Vec<ThetaTuning<F>>,
    /// Number of PLS components, should be used when problem size
    /// is over ten variables or so.
    kpls_dim: Option<usize>,
    /// Number of GP hyperparameters optimization restarts
    n_start: usize,
    /// Max number of likelihood evaluations during GP hyperparameters optimization
    max_eval: usize,
    /// Gaussian Mixture model used to cluster
    gmm: Option<GaussianMixtureModel<F>>,
    /// GaussianMixture preset
    gmx: Option<GaussianMixture<F>>,
    /// Random number generator
    rng: Xoshiro256Plus,
}

impl<F: Float> Default for GpMixtureValidParams<F> {
    fn default() -> GpMixtureValidParams<F> {
        GpMixtureValidParams {
            gp_type: GpType::FullGp,
            n_clusters: NbClusters::default(),
            recombination: Recombination::Hard,
            regression_spec: RegressionSpec::CONSTANT,
            correlation_spec: CorrelationSpec::SQUAREDEXPONENTIAL,
            theta_tunings: vec![ThetaTuning::default()],
            kpls_dim: None,
            n_start: 10,
            max_eval: GP_COBYLA_MAX_EVAL,
            gmm: None,
            gmx: None,
            rng: Xoshiro256Plus::from_entropy(),
        }
    }
}

impl<F: Float> GpMixtureValidParams<F> {
    /// The optional number of PLS components
    pub fn gp_type(&self) -> &GpType<F> {
        &self.gp_type
    }

    /// The number of clusters, hence the number of experts of the mixture.
    pub fn n_clusters(&self) -> NbClusters {
        self.n_clusters.clone()
    }

    /// The recombination mode
    pub fn recombination(&self) -> Recombination<F> {
        self.recombination
    }

    /// The allowed GP regression models in the mixture
    pub fn regression_spec(&self) -> RegressionSpec {
        self.regression_spec
    }

    /// The allowed GP correlation models in the mixture
    pub fn correlation_spec(&self) -> CorrelationSpec {
        self.correlation_spec
    }

    /// The speified tuning of theta hyperparameter
    pub fn theta_tunings(&self) -> &Vec<ThetaTuning<F>> {
        &self.theta_tunings
    }

    /// The optional number of PLS components
    pub fn kpls_dim(&self) -> Option<usize> {
        self.kpls_dim
    }

    /// The number of hyperparameters optimization restarts
    pub fn n_start(&self) -> usize {
        self.n_start
    }

    /// The max number of likelihood optimization during hyperparameters optimization
    pub fn max_eval(&self) -> usize {
        self.max_eval
    }

    /// An optional gaussian mixture to be fitted to generate multivariate normal
    /// in turns used to cluster
    pub fn gmm(&self) -> Option<&GaussianMixtureModel<F>> {
        self.gmm.as_ref()
    }

    /// An optional multivariate normal used to cluster (take precedence over gmm)
    pub fn gmx(&self) -> Option<&GaussianMixture<F>> {
        self.gmx.as_ref()
    }

    /// The random generator
    pub fn rng(&self) -> Xoshiro256Plus {
        self.rng.clone()
    }
}

/// Mixture of experts parameters
#[derive(Clone, Serialize, Deserialize)]
pub struct GpMixtureParams<F: Float>(GpMixtureValidParams<F>);

impl<F: Float> Default for GpMixtureParams<F> {
    fn default() -> GpMixtureParams<F> {
        GpMixtureParams(GpMixtureValidParams::default())
    }
}

impl<F: Float> GpMixtureParams<F> {
    /// Constructor of GP parameters.
    pub fn new() -> GpMixtureParams<F> {
        Self::new_with_rng(GpType::FullGp, Xoshiro256Plus::from_entropy())
    }
}

impl<F: Float> GpMixtureParams<F> {
    /// Constructor of Gp mixture parameters specifying random number generator for reproducibility
    ///
    /// See [`new`](GpMixtureParams::new) for default parameters.
    pub fn new_with_rng(gp_type: GpType<F>, rng: Xoshiro256Plus) -> GpMixtureParams<F> {
        Self(GpMixtureValidParams {
            gp_type,
            recombination: Recombination::Smooth(Some(F::one())),
            rng,
            ..Default::default()
        })
    }

    /// Sets the number of clusters
    pub fn gp_type(mut self, gp_type: GpType<F>) -> Self {
        self.0.gp_type = gp_type;
        self
    }

    /// Sets the number of clusters
    pub fn n_clusters(mut self, n_clusters: NbClusters) -> Self {
        self.0.n_clusters = n_clusters;
        self
    }

    /// Sets the recombination mode
    pub fn recombination(mut self, recombination: Recombination<F>) -> Self {
        self.0.recombination = recombination;
        self
    }

    /// Sets the regression models used in the mixture.
    ///
    /// Only GP models with regression models allowed by this specification
    /// will be used in the mixture.  
    pub fn regression_spec(mut self, regression_spec: RegressionSpec) -> Self {
        self.0.regression_spec = regression_spec;
        self
    }

    /// Sets the correlation models used in the mixture.
    ///
    /// Only GP models with correlation models allowed by this specification
    /// will be used in the mixture.  
    pub fn correlation_spec(mut self, correlation_spec: CorrelationSpec) -> Self {
        self.0.correlation_spec = correlation_spec;
        self
    }

    /// Sets the number of componenets retained during PLS dimension reduction.
    pub fn kpls_dim(mut self, kpls_dim: Option<usize>) -> Self {
        self.0.kpls_dim = kpls_dim;
        self
    }

    /// Set theta hyper parameter tuning
    pub fn theta_tunings(mut self, theta_tunings: &[ThetaTuning<F>]) -> Self {
        self.0.theta_tunings = theta_tunings.to_vec();
        self
    }

    /// Sets the number of hyperparameters optimization restarts
    pub fn n_start(mut self, n_start: usize) -> Self {
        self.0.n_start = n_start;
        self
    }

    /// Sets The max number of likelihood optimization during hyperparameters optimization
    pub fn max_eval(mut self, max_eval: usize) -> Self {
        self.0.max_eval = max_eval;
        self
    }

    #[doc(hidden)]
    /// Sets the gaussian mixture (used to find the optimal number of clusters)
    pub fn gmm(mut self, gmm: GaussianMixtureModel<F>) -> Self {
        self.0.gmm = Some(gmm);
        self
    }

    #[doc(hidden)]
    /// Sets the gaussian mixture (used to find the optimal number of clusters)
    /// Warning: no consistency check is done on the given initialization data
    /// *Panic* if multivariate normal init data not sound
    pub fn gmx(mut self, weights: Array1<F>, means: Array2<F>, covariances: Array3<F>) -> Self {
        self.0.gmx = Some(GaussianMixture::new(weights, means, covariances).unwrap());
        self
    }

    /// Sets the random number generator for reproducibility
    pub fn with_rng(mut self, rng: Xoshiro256Plus) -> GpMixtureParams<F> {
        self.0.rng = rng;
        self
    }
}

impl<F: Float> ParamGuard for GpMixtureParams<F> {
    type Checked = GpMixtureValidParams<F>;
    type Error = MoeError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if let Some(d) = self.0.kpls_dim
            && d == 0
        {
            return Err(MoeError::InvalidValueError(
                "`kpls_dim` canot be 0!".to_string(),
            ));
        }

        if self.0.n_clusters.is_multi() && self.0.theta_tunings.len() == 1 {
        } else if let NbClusters::Fixed { nb } = self.0.n_clusters
            && nb != self.0.theta_tunings.len()
        {
            panic!(
                "Number of clusters (={}) and theta init size (={}) not compatible, should be equal",
                nb,
                self.0.theta_tunings.len()
            );
        }
        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl<F: Float> From<GpMixtureValidParams<F>> for GpMixtureParams<F> {
    fn from(item: GpMixtureValidParams<F>) -> Self {
        GpMixtureParams(item)
    }
}

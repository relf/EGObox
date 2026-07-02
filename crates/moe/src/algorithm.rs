use super::gaussian_mixture::GaussianMixture;
use crate::clustering::{find_best_number_of_clusters, sort_by_cluster};
use crate::errors::MoeError;
use crate::errors::Result;
use crate::parameters::{GpMixtureParams, GpMixtureValidParams};
use crate::{GpMetrics, IaeAlphaPlotData, types::*};
use crate::{GpType, expertise_macros::*};
use crate::{NbClusters, surrogates::*};

use egobox_gp::{GaussianProcess, SparseGaussianProcess, correlation_models::*, mean_models::*};
use linfa::dataset::Records;
use linfa::traits::{Fit, Predict, PredictInplace};
use linfa::{Dataset, DatasetBase, Float, ParamGuard};
use linfa_clustering::GaussianMixtureModel;
use log::{debug, info, trace};
use paste::paste;
use std::cmp::Ordering;
use std::ops::Sub;

#[cfg(not(feature = "blas"))]
use linfa_linalg::norm::*;
use ndarray::{
    Array1, Array2, Array3, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip, concatenate, s,
};

#[cfg(feature = "blas")]
use ndarray_linalg::Norm;
use ndarray_rand::rand::Rng;
use ndarray_stats::QuantileExt;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;

macro_rules! check_allowed {
    ($spec:ident, $model_kind:ident, $model:ident, $list:ident) => {
        paste! {
            if $spec.contains([< $model_kind Spec>]::[< $model:upper >]) {
                $list.push(stringify!($model));
            }
        }
    };
}

impl<D: Data<Elem = f64>> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>, MoeError>
    for GpMixtureValidParams<f64>
{
    type Object = GpMixture;

    /// Fit Moe parameters using maximum likelihood
    ///
    /// # Errors
    ///
    /// * [MoeError::ClusteringError]: if there is not enough points regarding the clusters,
    /// * [MoeError::GpError]: if gaussian process fitting fails
    ///
    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
    ) -> Result<Self::Object> {
        let x = dataset.records();
        let y = dataset.targets();
        self.train(x, y)
    }
}

impl GpMixtureValidParams<f64> {
    /// Train a Mixture of Experts model on the given training data (xt, yt)
    pub fn train(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    ) -> Result<GpMixture> {
        trace!("Moe training...");
        let nx = xt.ncols();
        let data = concatenate(
            Axis(1),
            &[xt.view(), yt.to_owned().insert_axis(Axis(1)).view()],
        )
        .unwrap();

        let (n_clusters, recomb) = match self.n_clusters() {
            NbClusters::Auto { max } => {
                // automatic mode
                let max_nb_clusters = max.unwrap_or(xt.nrows() / 10 + 1);
                info!("Find best number of clusters up to {max_nb_clusters}");
                find_best_number_of_clusters(
                    xt,
                    yt,
                    max_nb_clusters,
                    self.kpls_dim(),
                    self.regression_spec(),
                    self.correlation_spec(),
                    self.rng(),
                )
            }
            NbClusters::Fixed { nb: nb_clusters } => (nb_clusters, self.recombination()),
        };
        if let NbClusters::Auto { max: _ } = self.n_clusters() {
            info!("Automatic settings {n_clusters} {recomb:?}");
        }

        let training = if recomb == Recombination::Smooth(None) && self.n_clusters().is_multi() {
            // Extract 5% of data for validation to find best heaviside factor
            // TODO: Better use cross-validation... but performances impact?
            let (_, training_data) = extract_part(&data, 5);
            training_data
        } else {
            data.to_owned()
        };
        let dataset = Dataset::from(training);

        let gmx = if self.gmx().is_some() {
            self.gmx().unwrap().clone()
        } else {
            trace!("GMM training...");
            let gmm = GaussianMixtureModel::params(n_clusters)
                .n_runs(20)
                .with_rng(self.rng())
                .fit(&dataset)?;

            // GMX for prediction
            let weights = gmm.weights().to_owned();
            let means = gmm.means().slice(s![.., ..nx]).to_owned();
            let covariances = gmm.covariances().slice(s![.., ..nx, ..nx]).to_owned();
            let factor = match recomb {
                Recombination::Smooth(Some(f)) => f,
                Recombination::Smooth(_) => 1.,
                Recombination::Hard => 1.,
            };
            GaussianMixture::new(weights, means, covariances)?.heaviside_factor(factor)
        };

        trace!("Train on clusters...");
        let clustering = Clustering::new(gmx, recomb);
        self.train_on_clusters(&xt.view(), &yt.view(), &clustering)
    }

    /// Using the current state of the clustering, select and train the experts
    /// Returns the fitted mixture of experts model
    pub fn train_on_clusters(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
        clustering: &Clustering,
    ) -> Result<GpMixture> {
        let gmx = clustering.gmx();
        let recomb = clustering.recombination();
        let nx = xt.ncols();
        let data = concatenate(
            Axis(1),
            &[xt.view(), yt.to_owned().insert_axis(Axis(1)).view()],
        )
        .unwrap();

        let dataset_clustering = gmx.predict(xt);
        let clusters = sort_by_cluster(gmx.n_clusters(), &data, &dataset_clustering);

        check_number_of_points(&clusters, xt.ncols(), self.regression_spec())?;

        // Fit GPs on clustered data
        let mut experts = Vec::new();
        let nb_clusters = clusters.len();
        for (nc, cluster) in clusters.iter().enumerate() {
            if nb_clusters > 1 && cluster.nrows() < 3 {
                return Err(MoeError::ClusteringError(format!(
                    "Not enough points in cluster, requires at least 3, got {}",
                    cluster.nrows()
                )));
            }
            debug!("nc={} theta_tuning={:?}", nc, self.theta_tunings());
            let expert = self.find_best_expert(nc, nx, cluster)?;
            experts.push(expert);
        }

        if recomb == Recombination::Smooth(None) && self.n_clusters().is_multi() {
            // Extract 5% of data for validation to find best heaviside factor
            // TODO: Better use cross-validation... but performances impact?
            let (test, _) = extract_part(&data, 5);
            let xtest = test.slice(s![.., ..nx]).to_owned();
            let ytest = test.slice(s![.., nx..]).to_owned().remove_axis(Axis(1));
            let factor = self.optimize_heaviside_factor(&experts, gmx, &xtest, &ytest);
            info!("Retrain mixture with optimized heaviside factor={factor}");

            let moe = GpMixtureParams::from(self.clone())
                .n_clusters(NbClusters::fixed(gmx.n_clusters()))
                .recombination(Recombination::Smooth(Some(factor)))
                .check()?
                .train(xt, yt)?; // needs to train the gaussian mixture on all data (xt, yt) as it was
            // previously trained on data excluding test data (see train method)
            Ok(moe)
        } else {
            Ok(GpMixture {
                gp_type: self.gp_type().clone(),
                recombination: recomb,
                experts,
                gmx: gmx.clone(),
                training_data: (xt.to_owned(), yt.to_owned()),
                params: self.clone(),
            })
        }
    }

    /// Select the surrogate which gives the smallest prediction error on the given data
    /// The error is computed using cross-validation
    fn find_best_expert(
        &self,
        nc: usize,
        nx: usize,
        data: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Box<dyn FullGpSurrogate>> {
        let xtrain = data.slice(s![.., ..nx]).to_owned();
        let ytrain = data.slice(s![.., nx..]).to_owned();
        let mut dataset = Dataset::from((xtrain.clone(), ytrain.clone().remove_axis(Axis(1))));
        let regression_spec = self.regression_spec();
        let mut allowed_means = vec![];
        check_allowed!(regression_spec, Regression, Constant, allowed_means);
        check_allowed!(regression_spec, Regression, Linear, allowed_means);
        check_allowed!(regression_spec, Regression, Quadratic, allowed_means);
        let correlation_spec = self.correlation_spec();
        let mut allowed_corrs = vec![];
        check_allowed!(
            correlation_spec,
            Correlation,
            SquaredExponential,
            allowed_corrs
        );
        check_allowed!(
            correlation_spec,
            Correlation,
            AbsoluteExponential,
            allowed_corrs
        );
        check_allowed!(correlation_spec, Correlation, Matern32, allowed_corrs);
        check_allowed!(correlation_spec, Correlation, Matern52, allowed_corrs);

        debug!("Find best expert");
        let best = if allowed_means.len() == 1 && allowed_corrs.len() == 1 {
            (format!("{}_{}", allowed_means[0], allowed_corrs[0]), None) // shortcut
        } else {
            let mut map_error = Vec::new();
            compute_errors!(self, allowed_means, allowed_corrs, dataset, map_error);
            let errs: Vec<f64> = map_error.iter().map(|(_, err)| *err).collect();
            debug!("Accuracies {map_error:?}");
            let argmin = errs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap();
            (map_error[argmin].0.clone(), Some(map_error[argmin].1))
        };
        debug!("after Find best expert");

        let expert = match self.gp_type() {
            GpType::FullGp => {
                let best_expert_params: std::result::Result<Box<dyn GpSurrogateParams>, MoeError> =
                    match best.0.as_str() {
                        "Constant_SquaredExponential" => {
                            Ok(make_surrogate_params!(Constant, SquaredExponential))
                        }
                        "Constant_AbsoluteExponential" => {
                            Ok(make_surrogate_params!(Constant, AbsoluteExponential))
                        }
                        "Constant_Matern32" => Ok(make_surrogate_params!(Constant, Matern32)),
                        "Constant_Matern52" => Ok(make_surrogate_params!(Constant, Matern52)),
                        "Linear_SquaredExponential" => {
                            Ok(make_surrogate_params!(Linear, SquaredExponential))
                        }
                        "Linear_AbsoluteExponential" => {
                            Ok(make_surrogate_params!(Linear, AbsoluteExponential))
                        }
                        "Linear_Matern32" => Ok(make_surrogate_params!(Linear, Matern32)),
                        "Linear_Matern52" => Ok(make_surrogate_params!(Linear, Matern52)),
                        "Quadratic_SquaredExponential" => {
                            Ok(make_surrogate_params!(Quadratic, SquaredExponential))
                        }
                        "Quadratic_AbsoluteExponential" => {
                            Ok(make_surrogate_params!(Quadratic, AbsoluteExponential))
                        }
                        "Quadratic_Matern32" => Ok(make_surrogate_params!(Quadratic, Matern32)),
                        "Quadratic_Matern52" => Ok(make_surrogate_params!(Quadratic, Matern52)),
                        _ => {
                            return Err(MoeError::ExpertError(format!(
                                "Unknown expert {}",
                                best.0
                            )));
                        }
                    };
                let mut expert_params = best_expert_params?;
                expert_params.n_start(self.n_start());
                expert_params.max_eval(self.max_eval());
                expert_params.kpls_dim(self.kpls_dim());
                if nc > 0 && self.theta_tunings().len() == 1 {
                    expert_params.theta_tuning(self.theta_tunings()[0].clone());
                } else {
                    debug!("Training with theta_tuning = {:?}.", self.theta_tunings());
                    expert_params.theta_tuning(self.theta_tunings()[nc].clone());
                }
                debug!("Train best expert...");
                expert_params.train(&xtrain.view(), &ytrain.view())
            }
            GpType::SparseGp {
                inducings,
                sparse_method,
                ..
            } => {
                let inducings = inducings.to_owned();
                let best_expert_params: std::result::Result<Box<dyn SgpSurrogateParams>, MoeError> =
                    match best.0.as_str() {
                        "Constant_SquaredExponential" => {
                            Ok(make_sgp_surrogate_params!(SquaredExponential, inducings))
                        }
                        "Constant_AbsoluteExponential" => {
                            Ok(make_sgp_surrogate_params!(AbsoluteExponential, inducings))
                        }
                        "Constant_Matern32" => Ok(make_sgp_surrogate_params!(Matern32, inducings)),
                        "Constant_Matern52" => Ok(make_sgp_surrogate_params!(Matern52, inducings)),
                        _ => {
                            return Err(MoeError::ExpertError(format!(
                                "Unknown expert {}",
                                best.0
                            )));
                        }
                    };
                let mut expert_params = best_expert_params?;
                let seed = self.rng().r#gen();
                debug!("Theta tuning = {:?}", self.theta_tunings());
                expert_params.sparse_method(*sparse_method);
                expert_params.seed(seed);
                expert_params.n_start(self.n_start());
                expert_params.kpls_dim(self.kpls_dim());
                expert_params.theta_tuning(self.theta_tunings()[0].clone());
                debug!("Train best expert...");
                expert_params.train(&xtrain.view(), &ytrain.view())
            }
        };

        debug!("...after best expert training");
        if let Some(v) = best.1 {
            info!("Best expert {} accuracy={}", best.0, v);
        }
        expert
    }

    /// Take the best heaviside factor from 0.1 to 2.1 (step 0.1).
    /// Mixture (`gmx` and experts`) is already trained only the continuous recombination is changed
    /// and the factor giving the smallest prediction error on the given test data  
    /// Used only in case of smooth recombination
    fn optimize_heaviside_factor(
        &self,
        experts: &[Box<dyn FullGpSurrogate>],
        gmx: &GaussianMixture<f64>,
        xtest: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        ytest: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    ) -> f64 {
        if self.recombination() == Recombination::Hard || self.n_clusters().is_mono() {
            1.
        } else {
            let scale_factors = Array1::linspace(0.1, 2.1, 20);
            let errors = scale_factors.map(move |&factor| {
                let gmx2 = gmx.clone();
                let gmx2 = gmx2.heaviside_factor(factor);
                let pred = predict_smooth(experts, &gmx2, xtest).unwrap();
                pred.sub(ytest).mapv(|x| x * x).sum().sqrt() / xtest.mapv(|x| x * x).sum().sqrt()
            });

            let min_error_index = errors.argmin().unwrap();
            if *errors.max().unwrap() < 1e-6 {
                1.
            } else {
                scale_factors[min_error_index]
            }
        }
    }
}

fn check_number_of_points<F>(
    clusters: &[ArrayBase<impl Data<Elem = F>, Ix2>],
    dim: usize,
    regr: RegressionSpec,
) -> Result<()> {
    if clusters.len() > 1 {
        let min_number_point = if regr.contains(RegressionSpec::QUADRATIC) {
            (dim + 1) * (dim + 2) / 2
        } else if regr.contains(RegressionSpec::LINEAR) {
            dim + 1
        } else {
            1
        };
        for cluster in clusters {
            if cluster.len() < min_number_point {
                return Err(MoeError::ClusteringError(format!(
                    "Not enough points in training set. Need {} points, got {}",
                    min_number_point,
                    cluster.len()
                )));
            }
        }
    }
    Ok(())
}

/// Predict outputs at given points with `experts` and gaussian mixture `gmx`.
/// `gmx` is used to get the probability of x to belongs to one cluster
/// or another (ie responsabilities). Those responsabilities are used to combine
/// output values predict by each cluster experts.
fn predict_smooth(
    experts: &[Box<dyn FullGpSurrogate>],
    gmx: &GaussianMixture<f64>,
    points: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Result<Array1<f64>> {
    let probas = gmx.predict_probas(points);
    let preds: Array1<f64> = experts
        .iter()
        .enumerate()
        .map(|(i, gp)| gp.predict(&points.view()).unwrap() * probas.column(i))
        .fold(Array1::zeros((points.nrows(),)), |acc, pred| acc + pred);
    Ok(preds)
}

/// Mixture of gaussian process experts
/// Implementation note: the structure is not generic over 'F: Float' to be able to
/// implement use serde easily as deserialization of generic impls is not supported yet
/// See <https://github.com/dtolnay/typetag/issues/1>
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct GpMixture {
    /// The mode of recombination to get the output prediction from experts prediction
    recombination: Recombination<f64>,
    /// The list of the best experts trained on each cluster
    experts: Vec<Box<dyn FullGpSurrogate>>,
    /// The gaussian mixture allowing to predict cluster responsabilities for a given point
    gmx: GaussianMixture<f64>,
    /// Gp type
    gp_type: GpType<f64>,
    /// Training inputs
    training_data: (Array2<f64>, Array1<f64>),
    /// Params used to fit this model
    params: GpMixtureValidParams<f64>,
}

impl std::fmt::Display for GpMixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let recomb = match self.recombination() {
            Recombination::Hard => "Hard".to_string(),
            Recombination::Smooth(Some(f)) => format!("Smooth({f})"),
            Recombination::Smooth(_) => "Smooth".to_string(),
        };
        let experts = self
            .experts
            .iter()
            .map(|expert| expert.to_string())
            .reduce(|acc, s| acc + ", " + &s)
            .unwrap();
        write!(f, "Mixture[{}]({})", &recomb, &experts)
    }
}

impl Clustered for GpMixture {
    /// Number of clusters
    fn n_clusters(&self) -> usize {
        self.gmx.n_clusters()
    }

    /// Clustering Recombination
    fn recombination(&self) -> Recombination<f64> {
        self.recombination()
    }

    /// Convert to clustering
    fn to_clustering(&self) -> Clustering {
        Clustering {
            recombination: self.recombination(),
            gmx: self.gmx.clone(),
        }
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpSurrogate for GpMixture {
    fn dims(&self) -> (usize, usize) {
        self.experts[0].dims()
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_hard(x),
            Recombination::Smooth(_) => self.predict_smooth(x),
        }
    }

    fn predict_var(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_var_hard(x),
            Recombination::Smooth(_) => self.predict_var_smooth(x),
        }
    }

    fn predict_valvar(&self, x: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        match self.recombination {
            Recombination::Hard => self.predict_valvar_hard(x),
            Recombination::Smooth(_) => self.predict_valvar_smooth(x),
        }
    }

    /// Save Moe model in given file.
    #[cfg(feature = "persistent")]
    fn save(&self, path: &str, format: GpFileFormat) -> Result<()> {
        let mut file = fs::File::create(path).unwrap();

        let bytes = match format {
            GpFileFormat::Json => serde_json::to_vec(self).map_err(MoeError::SaveJsonError)?,
            GpFileFormat::Binary => {
                bincode::serde::encode_to_vec(self, bincode::config::standard())
                    .map_err(MoeError::SaveBinaryError)?
            }
        };
        file.write_all(&bytes)?;

        Ok(())
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpSurrogateExt for GpMixture {
    fn predict_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_gradients_hard(x),
            Recombination::Smooth(_) => self.predict_gradients_smooth(x),
        }
    }

    fn predict_var_gradients(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.recombination {
            Recombination::Hard => self.predict_var_gradients_hard(x),
            Recombination::Smooth(_) => self.predict_var_gradients_smooth(x),
        }
    }

    fn predict_valvar_gradients(&self, x: &ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        match self.recombination {
            Recombination::Hard => self.predict_valvar_gradients_hard(x),
            Recombination::Smooth(_) => self.predict_valvar_gradients_smooth(x),
        }
    }

    fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> Result<Array2<f64>> {
        if self.n_clusters() != 1 {
            return Err(MoeError::SampleError(format!(
                "Can not sample when several clusters {}",
                self.n_clusters()
            )));
        }
        self.sample_expert(0, x, n_traj)
    }
}

impl GpMetrics<MoeError, GpMixtureParams<f64>, Self> for GpMixture {
    fn training_data(&self) -> &(Array2<f64>, Array1<f64>) {
        &self.training_data
    }

    fn params(&self) -> GpMixtureParams<f64> {
        GpMixtureParams::<f64>::from(self.params.clone())
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpQualityAssurance for GpMixture {
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

#[cfg_attr(feature = "serializable", typetag::serde)]
impl MixtureGpSurrogate for GpMixture {
    /// Selected experts in the mixture
    fn experts(&self) -> &Vec<Box<dyn FullGpSurrogate>> {
        &self.experts
    }
}

impl GpMixture {
    /// Constructor of mixture of experts parameters
    pub fn params() -> GpMixtureParams<f64> {
        GpMixtureParams::new()
    }

    /// Retrieve output dimensions from
    pub fn gp_type(&self) -> &GpType<f64> {
        &self.gp_type
    }

    /// Recombination mode
    pub fn recombination(&self) -> Recombination<f64> {
        self.recombination
    }

    /// Gaussian mixture
    pub fn gmx(&self) -> &GaussianMixture<f64> {
        &self.gmx
    }

    /// Sets recombination mode
    pub fn set_recombination(mut self, recombination: Recombination<f64>) -> Self {
        self.recombination = match recombination {
            Recombination::Hard => recombination,
            Recombination::Smooth(Some(_)) => recombination,
            Recombination::Smooth(_) => Recombination::Smooth(Some(1.)),
        };
        self
    }

    /// Set the gaussian mixture to use given weights, means and covariances
    pub fn set_gmx(
        mut self,
        weights: Array1<f64>,
        means: Array2<f64>,
        covariances: Array3<f64>,
    ) -> Self {
        self.gmx = GaussianMixture::new(weights, means, covariances).unwrap();
        self
    }

    /// Set the model experts to use in the mixture
    pub fn set_experts(mut self, experts: Vec<Box<dyn FullGpSurrogate>>) -> Self {
        self.experts = experts;
        self
    }

    /// Predict outputs at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the probability of the point to belongs to one cluster
    /// or another (ie responsabilities).     
    /// The smooth recombination of each cluster expert responsabilty is used to get the result.
    pub fn predict_smooth(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Result<Array1<f64>> {
        predict_smooth(&self.experts, &self.gmx, x)
    }

    /// Predict variances at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the probability of the point to belongs to one cluster
    /// or another (ie responsabilities).
    /// The smooth recombination of each cluster expert responsabilty is used to get the result.
    pub fn predict_var_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array1<f64>> {
        let probas = self.gmx.predict_probas(x);
        let preds: Array1<f64> = self
            .experts
            .iter()
            .enumerate()
            .map(|(i, gp)| {
                let p = probas.column(i);
                gp.predict_var(&x.view()).unwrap() * p * p
            })
            .fold(Array1::zeros(x.nrows()), |acc, var| acc + var);
        Ok(preds)
    }

    /// Predict derivatives of the output at a set of points `x` specified as (n, nx) matrix.
    /// Return derivatives as a (n, nx) matrix where the ith row contain the partial derivatives of
    /// of the output wrt the nx components of `x` valued at the ith x point.
    /// The smooth recombination of each cluster expert responsability is used to get the result.
    pub fn predict_gradients_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(x);
        let probas_drv = self.gmx.predict_probas_derivatives(x);
        let mut drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));

        Zip::from(drv.rows_mut())
            .and(x.rows())
            .and(probas.rows())
            .and(probas_drv.outer_iter())
            .for_each(|mut y, x, p, pprime| {
                let x = x.insert_axis(Axis(0));
                let preds: Array1<f64> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict(&x).unwrap()[0])
                    .collect();
                let drvs: Vec<Array1<f64>> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_gradients(&x).unwrap().row(0).to_owned())
                    .collect();

                let preds = preds.insert_axis(Axis(1));
                let mut preds_drv = Array2::zeros((self.experts.len(), x.len()));
                Zip::indexed(preds_drv.rows_mut()).for_each(|i, mut jc| jc.assign(&drvs[i]));

                let mut term1 = Array2::zeros((self.experts.len(), x.len()));
                Zip::from(term1.rows_mut())
                    .and(&p)
                    .and(preds_drv.rows())
                    .for_each(|mut t, p, der| t.assign(&(der.to_owned().mapv(|v| v * p))));
                let term1 = term1.sum_axis(Axis(0));

                let term2 = pprime.to_owned() * preds;
                let term2 = term2.sum_axis(Axis(0));

                y.assign(&(term1 + term2));
            });
        Ok(drv)
    }

    /// Predict derivatives of the variance at a set of points `x` specified as (n, nx) matrix.
    /// Return derivatives as a (n, nx) matrix where the ith row contain the partial derivatives of
    /// of the vairance wrt the nx components of `x` valued at the ith x point.
    /// The smooth recombination of each cluster expert responsability is used to get the result.
    pub fn predict_var_gradients_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let probas = self.gmx.predict_probas(x);
        let probas_drv = self.gmx.predict_probas_derivatives(x);

        let mut drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));

        Zip::from(drv.rows_mut())
            .and(x.rows())
            .and(probas.rows())
            .and(probas_drv.outer_iter())
            .for_each(|mut y, xi, p, pprime| {
                let xii = xi.insert_axis(Axis(0));
                let preds: Array1<f64> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_var(&xii).unwrap()[0])
                    .collect();
                let drvs: Vec<Array1<f64>> = self
                    .experts
                    .iter()
                    .map(|gp| gp.predict_var_gradients(&xii).unwrap().row(0).to_owned())
                    .collect();

                let preds = preds.insert_axis(Axis(1));
                let mut preds_drv = Array2::zeros((self.experts.len(), xi.len()));
                Zip::indexed(preds_drv.rows_mut()).for_each(|i, mut jc| jc.assign(&drvs[i]));

                let mut term1 = Array2::zeros((self.experts.len(), xi.len()));
                Zip::from(term1.rows_mut())
                    .and(&p)
                    .and(preds_drv.rows())
                    .for_each(|mut t, p, der| t.assign(&(der.to_owned().mapv(|v| v * p * p))));
                let term1 = term1.sum_axis(Axis(0));

                let term2 = (p.to_owned() * pprime * preds).mapv(|v| 2. * v);
                let term2 = term2.sum_axis(Axis(0));

                y.assign(&(term1 + term2));
            });

        Ok(drv)
    }

    /// Predict outputs and variances at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the probability of the point to belongs to one cluster
    /// or another (ie responsabilities).
    /// The smooth recombination of each cluster expert responsabilty is used to get the result.
    pub fn predict_valvar_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let probas = self.gmx.predict_probas(x);
        let valvar: (Array1<f64>, Array1<f64>) = self
            .experts
            .iter()
            .enumerate()
            .map(|(i, gp)| {
                let p = probas.column(i);
                let (pred, var) = gp.predict_valvar(&x.view()).unwrap();
                (pred * p, var * p * p)
            })
            .fold(
                (Array1::zeros((x.nrows(),)), Array1::zeros((x.nrows(),))),
                |acc, (pred, var)| (acc.0 + pred, acc.1 + var),
            );

        Ok(valvar)
    }

    fn predict_valvar_gradients_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let probas = self.gmx.predict_probas(x);
        let probas_drv = self.gmx.predict_probas_derivatives(x);

        let mut val_drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let mut var_drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));

        Zip::from(val_drv.rows_mut())
            .and(var_drv.rows_mut())
            .and(x.rows())
            .and(probas.rows())
            .and(probas_drv.outer_iter())
            .for_each(|mut val_y, mut var_y, xi, p, pprime| {
                let xii = xi.insert_axis(Axis(0));
                let (preds, vars): (Vec<f64>, Vec<f64>) = self
                    .experts
                    .iter()
                    .map(|gp| {
                        let (pred, var) = gp.predict_valvar(&xii).unwrap();
                        (pred[0], var[0])
                    })
                    .unzip();
                let preds: Array2<f64> = Array1::from(preds).insert_axis(Axis(1));
                let vars: Array2<f64> = Array1::from(vars).insert_axis(Axis(1));
                let (drvs, var_drvs): (Vec<Array1<f64>>, Vec<Array1<f64>>) = self
                    .experts
                    .iter()
                    .map(|gp| {
                        let (predg, varg) = gp.predict_valvar_gradients(&xii).unwrap();
                        (predg.row(0).to_owned(), varg.row(0).to_owned())
                    })
                    .unzip();

                let mut preds_drv = Array2::zeros((self.experts.len(), xi.len()));
                let mut vars_drv = Array2::zeros((self.experts.len(), xi.len()));
                Zip::indexed(preds_drv.rows_mut()).for_each(|i, mut jc| jc.assign(&drvs[i]));
                Zip::indexed(vars_drv.rows_mut()).for_each(|i, mut jc| jc.assign(&var_drvs[i]));

                let mut val_term1 = Array2::zeros((self.experts.len(), xi.len()));
                Zip::from(val_term1.rows_mut())
                    .and(&p)
                    .and(preds_drv.rows())
                    .for_each(|mut t, p, der| t.assign(&(der.to_owned().mapv(|v| v * p))));
                let val_term1 = val_term1.sum_axis(Axis(0));
                let val_term2 = pprime.to_owned() * preds;
                let val_term2 = val_term2.sum_axis(Axis(0));
                val_y.assign(&(val_term1 + val_term2));

                let mut var_term1 = Array2::zeros((self.experts.len(), xi.len()));
                Zip::from(var_term1.rows_mut())
                    .and(&p)
                    .and(vars_drv.rows())
                    .for_each(|mut t, p, der| t.assign(&(der.to_owned().mapv(|v| v * p * p))));
                let var_term1 = var_term1.sum_axis(Axis(0));
                let var_term2 = (p.to_owned() * pprime * vars).mapv(|v| 2. * v);
                let var_term2 = var_term2.sum_axis(Axis(0));
                var_y.assign(&(var_term1 + var_term2));
            });
        Ok((val_drv, var_drv))
    }

    /// Predict outputs at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// Then the expert of the cluster is used to predict the output value.
    /// Returns the ouputs as a (n, 1) column vector
    pub fn predict_hard(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Result<Array1<f64>> {
        let clustering = self.gmx.predict(x);
        let mut preds = Array1::zeros((x.nrows(),));
        Zip::from(&mut preds)
            .and(x.rows())
            .and(&clustering)
            .for_each(|y, x, &c| *y = self.experts[c].predict(&x.insert_axis(Axis(0))).unwrap()[0]);
        Ok(preds)
    }

    /// Predict variance at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    /// Returns the variances as a (n,) vector
    pub fn predict_var_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array1<f64>> {
        let clustering = self.gmx.predict(x);
        let mut variances = Array1::zeros(x.nrows());
        Zip::from(&mut variances)
            .and(x.rows())
            .and(&clustering)
            .for_each(|y, x, &c| {
                *y = self.experts[c]
                    .predict_var(&x.insert_axis(Axis(0)))
                    .unwrap()[0];
            });
        Ok(variances)
    }

    /// Predict outputs and variances at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    pub fn predict_valvar_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let clustering = self.gmx.predict(x);
        let mut preds = Array1::zeros((x.nrows(),));
        let mut variances = Array1::zeros(x.nrows());
        Zip::from(&mut preds)
            .and(&mut variances)
            .and(x.rows())
            .and(&clustering)
            .for_each(|y, v, x, &c| {
                let (pred, var) = self.experts[c]
                    .predict_valvar(&x.insert_axis(Axis(0)))
                    .unwrap();
                *y = pred[0];
                *v = var[0];
            });
        Ok((preds, variances))
    }

    /// Predict derivatives of the output at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    /// Returns derivatives as a (n, nx) matrix where the ith row contain the partial derivatives of
    /// of the output wrt the nx components of `x` valued at the ith x point.
    pub fn predict_gradients_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let mut drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let clustering = self.gmx.predict(x);
        Zip::from(drv.rows_mut())
            .and(x.rows())
            .and(&clustering)
            .for_each(|mut drv_i, xi, &c| {
                let x = xi.to_owned().insert_axis(Axis(0));
                let x_drv: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
                    self.experts[c].predict_gradients(&x.view()).unwrap();
                drv_i.assign(&x_drv.row(0))
            });
        Ok(drv)
    }

    /// Predict derivatives of the variances at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    /// Returns derivatives as a (n, nx) matrix where the ith row contain the partial derivatives of
    /// of the output wrt the nx components of `x` valued at the ith x point.
    pub fn predict_var_gradients_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        let mut vardrv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let clustering = self.gmx.predict(x);
        Zip::from(vardrv.rows_mut())
            .and(x.rows())
            .and(&clustering)
            .for_each(|mut vardrv_i, xi, &c| {
                let x = xi.to_owned().insert_axis(Axis(0));
                let x_vardrv: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
                    self.experts[c].predict_var_gradients(&x.view()).unwrap();
                vardrv_i.assign(&x_vardrv.row(0))
            });
        Ok(vardrv)
    }

    /// Predict derivatives of the outputs and variances at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict variance value.
    pub fn predict_valvar_gradients_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let mut val_drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let mut var_drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let clustering = self.gmx.predict(x);
        Zip::from(val_drv.rows_mut())
            .and(var_drv.rows_mut())
            .and(x.rows())
            .and(&clustering)
            .for_each(|mut val_y, mut var_y, xi, &c| {
                let x = xi.to_owned().insert_axis(Axis(0));
                let (x_val_drv, x_var_drv) =
                    self.experts[c].predict_valvar_gradients(&x.view()).unwrap();
                val_y.assign(&x_val_drv.row(0));
                var_y.assign(&x_var_drv.row(0));
            });
        Ok((val_drv, var_drv))
    }

    /// Sample `n_traj` trajectories at a set of points `x` specified as (n, nx) matrix.
    /// using the expert `ith` of the mixture.
    /// Returns the samples as a (n, n_traj) matrix where the ith row
    pub fn sample_expert(
        &self,
        ith: usize,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        n_traj: usize,
    ) -> Result<Array2<f64>> {
        self.experts[ith].sample(&x.view(), n_traj)
    }

    /// Predict outputs at a set of points `x` specified as (n, nx) matrix.
    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Result<Array1<f64>> {
        <GpMixture as GpSurrogate>::predict(self, &x.view())
    }

    /// Predict variances at a set of points `x` specified as (n, nx) matrix.
    pub fn predict_var(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Result<Array1<f64>> {
        <GpMixture as GpSurrogate>::predict_var(self, &x.view())
    }

    /// Predict outputs and variances at a set of points `x` specified as (n, nx) matrix.
    pub fn predict_valvar(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        <GpMixture as GpSurrogate>::predict_valvar(self, &x.view())
    }

    /// Predict derivatives of the output at a set of points `x` specified as (n, nx) matrix.
    pub fn predict_gradients(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        <GpMixture as GpSurrogateExt>::predict_gradients(self, &x.view())
    }

    /// Predict derivatives of the variance at a set of points `x` specified as (n, nx) matrix.
    pub fn predict_var_gradients(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<Array2<f64>> {
        <GpMixture as GpSurrogateExt>::predict_var_gradients(self, &x.view())
    }

    /// Predict derivatives of the outputs and variances at a set of points `x` specified as (n, nx) matrix.
    pub fn predict_valvar_gradients(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        <GpMixture as GpSurrogateExt>::predict_valvar_gradients(self, &x.view())
    }

    /// Sample `n_traj` trajectories at a set of points `x` specified as (n, nx) matrix.
    /// Returns the samples as a (n, n_traj) matrix where the ith row
    /// contain the samples of the output at the ith point.
    pub fn sample(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        n_traj: usize,
    ) -> Result<Array2<f64>> {
        <GpMixture as GpSurrogateExt>::sample(self, &x.view(), n_traj)
    }

    // pub fn cv_quality(&self) -> f64 {
    //     let dataset = Dataset::new(self.xtrain.to_owned(), self.ytrain.to_owned());
    //     let mut error = 0.;
    //     for (train, valid) in dataset.fold(self.xtrain.nrows()).into_iter() {
    //         if let Ok(mixture) = GpMixtureParams::default()
    //             .kpls_dim(self.kpls_dim)
    //             .gmx(
    //                 self.gmx.weights().to_owned(),
    //                 self.gmx.means().to_owned(),
    //                 self.gmx.covariances().to_owned(),
    //             )
    //             .fit(&train)
    //         {
    //             let pred = mixture.predict(valid.records()).unwrap();
    //             error += (valid.targets() - pred).norm_l2();
    //         } else {
    //             error += f64::INFINITY;
    //         }
    //     }
    //     error / self.ytrain.std(1.)
    // }

    /// Load Moe from the given file.
    #[cfg(feature = "persistent")]
    pub fn load(path: &str, format: GpFileFormat) -> Result<Box<GpMixture>> {
        let data = fs::read(path)?;
        let moe = match format {
            GpFileFormat::Json => serde_json::from_slice(&data)?,
            GpFileFormat::Binary => {
                bincode::serde::decode_from_slice(&data, bincode::config::standard())
                    .map(|(surrogate, _)| surrogate)?
            }
        };
        Ok(Box::new(moe))
    }
}

/// Take one out of `quantile` in a set of data rows
/// Returns the selected part and the remaining data.
fn extract_part<F: Float>(
    data: &ArrayBase<impl Data<Elem = F>, Ix2>,
    quantile: usize,
) -> (Array2<F>, Array2<F>) {
    let nsamples = data.nrows();
    let indices = Array1::range(0., nsamples as f32, quantile as f32).mapv(|v| v as usize);
    let data_test = data.select(Axis(0), indices.as_slice().unwrap());
    let indices2: Vec<usize> = (0..nsamples).filter(|i| i % quantile != 0).collect();
    let data_train = data.select(Axis(0), &indices2);
    (data_test, data_train)
}

impl<D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array1<f64>> for GpMixture {
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<f64>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let values = self.predict(x).expect("MoE prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<f64> {
        Array1::zeros(x.nrows())
    }
}

/// Adaptator to implement `linfa::Predict` for variance prediction
#[allow(dead_code)]
pub struct MoeVariancePredictor<'a>(&'a GpMixture);
impl<D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array1<f64>>
    for MoeVariancePredictor<'_>
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<f64>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let values = self.0.predict_var(x).expect("MoE variances prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<f64> {
        Array1::zeros(x.nrows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use argmin_testfunctions::rosenbrock;
    use egobox_doe::{Lhs, SamplingMethod};
    use ndarray::{Array, Array2, Zip, array};
    use ndarray_npy::write_npy;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256Plus;

    fn f_test_1d(x: &Array2<f64>) -> Array1<f64> {
        let mut y = Array1::zeros(x.len());
        let x = Array::from_iter(x.iter().cloned());
        Zip::from(&mut y).and(&x).for_each(|yi, xi| {
            if *xi < 0.4 {
                *yi = xi * xi;
            } else if (0.4..0.8).contains(xi) {
                *yi = 3. * xi + 1.;
            } else {
                *yi = f64::sin(10. * xi);
            }
        });
        y
    }

    fn df_test_1d(x: &Array2<f64>) -> Array2<f64> {
        let mut y = Array2::zeros(x.dim());
        Zip::from(y.rows_mut())
            .and(x.rows())
            .for_each(|mut yi, xi| {
                if xi[0] < 0.4 {
                    yi[0] = 2. * xi[0];
                } else if (0.4..0.8).contains(&xi[0]) {
                    yi[0] = 3.;
                } else {
                    yi[0] = 10. * f64::cos(10. * xi[0]);
                }
            });
        y
    }

    #[test]
    fn test_moe_hard() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt.to_owned());
        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(3))
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Hard)
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        let x = Array1::linspace(0., 1., 30).insert_axis(Axis(1));
        let preds = moe.predict(&x).expect("MOE prediction");
        let dpreds = moe.predict_gradients(&x).expect("MOE drv prediction");
        println!("dpred = {dpreds}");
        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();
        write_npy(format!("{test_dir}/x_hard.npy"), &x).expect("x saved");
        write_npy(format!("{test_dir}/preds_hard.npy"), &preds).expect("preds saved");
        write_npy(format!("{test_dir}/dpreds_hard.npy"), &dpreds).expect("dpreds saved");
        assert_abs_diff_eq!(
            0.39 * 0.39,
            moe.predict(&array![[0.39]]).unwrap()[0],
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(
            f64::sin(10. * 0.82),
            moe.predict(&array![[0.82]]).unwrap()[0],
            epsilon = 1e-4
        );
        println!("LOOQ2 = {}", moe.q2_score());
    }

    #[test]
    fn test_moe_smooth() {
        let test_dir = "target/tests";
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Array2::random_using((60, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt);
        let ds = Dataset::new(xt.to_owned(), yt.to_owned());
        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(3))
            .recombination(Recombination::Smooth(Some(0.5)))
            .with_rng(rng.clone())
            .fit(&ds)
            .expect("MOE fitted");
        let x = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict(&x).expect("MOE prediction");
        write_npy(format!("{test_dir}/xt.npy"), &xt).expect("x saved");
        write_npy(format!("{test_dir}/yt.npy"), &yt).expect("preds saved");
        write_npy(format!("{test_dir}/x_smooth.npy"), &x).expect("x saved");
        write_npy(format!("{test_dir}/preds_smooth.npy"), &preds).expect("preds saved");

        // Predict with smooth 0.5 which is not good
        println!("Smooth moe {moe}");
        assert_abs_diff_eq!(
            0.2623, // test we are not good as the true value = 0.37*0.37 = 0.1369
            moe.predict(&array![[0.37]]).unwrap()[0],
            epsilon = 1e-3
        );

        // Predict with smooth adjusted automatically which is better
        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(3))
            .recombination(Recombination::Smooth(None))
            .with_rng(rng.clone())
            .fit(&ds)
            .expect("MOE fitted");
        println!("Smooth moe {moe}");

        std::fs::create_dir_all(test_dir).ok();
        let x = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let preds = moe.predict(&x).expect("MOE prediction");
        write_npy(format!("{test_dir}/x_smooth2.npy"), &x).expect("x saved");
        write_npy(format!("{test_dir}/preds_smooth2.npy"), &preds).expect("preds saved");
        assert_abs_diff_eq!(
            0.37 * 0.37, // true value of the function
            moe.predict(&array![[0.37]]).unwrap()[0],
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_moe_auto() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Array2::random_using((60, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt);
        let ds = Dataset::new(xt, yt.to_owned());
        let moe = GpMixture::params()
            .n_clusters(NbClusters::auto())
            .with_rng(rng.clone())
            .fit(&ds)
            .expect("MOE fitted");
        println!(
            "Moe auto: nb clusters={}, recomb={:?}",
            moe.n_clusters(),
            moe.recombination()
        );
        assert_abs_diff_eq!(
            0.37 * 0.37, // true value of the function
            moe.predict(&array![[0.37]]).unwrap()[0],
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_moe_variances_smooth() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Array2::random_using((100, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt);
        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(3))
            .recombination(Recombination::Smooth(None))
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .with_rng(rng.clone())
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        // Smoke test: prediction is pretty good hence variance is very low
        let x = Array1::linspace(0., 1., 20).insert_axis(Axis(1));
        let variances = moe.predict_var(&x).expect("MOE variances prediction");
        assert_abs_diff_eq!(*variances.max().unwrap(), 0., epsilon = 1e-10);
    }

    fn xsinx(x: &[f64]) -> f64 {
        (x[0] - 3.5) * f64::sin((x[0] - 3.5) / std::f64::consts::PI)
    }

    #[test]
    fn test_find_best_expert() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = Array2::random_using((10, 1), Uniform::new(0., 1.), &mut rng);
        let yt = xt.mapv(|x| xsinx(&[x]));
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();
        let moe = GpMixture::params().with_rng(rng).check_unwrap();
        let best_expert = &moe.find_best_expert(0, 1, &data).unwrap();
        println!("Best expert {best_expert}");
    }

    #[test]
    fn test_find_best_heaviside_factor() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt);
        let _moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(3))
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
    }

    #[cfg(feature = "persistent")]
    #[test]
    fn test_save_load_moe() {
        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = Array2::random_using((50, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt);
        let ds = Dataset::new(xt, yt);
        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(3))
            .with_rng(rng)
            .fit(&ds)
            .expect("MOE fitted");
        let xtest = array![[0.6]];
        let y_expected = moe.predict(&xtest).unwrap();
        let filename = format!("{test_dir}/saved_moe.json");
        moe.save(&filename, GpFileFormat::Json).expect("MoE saving");
        let new_moe = GpMixture::load(&filename, GpFileFormat::Json).expect("MoE loading");
        assert_abs_diff_eq!(y_expected, new_moe.predict(&xtest).unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_moe_drv_smooth() {
        let rng = Xoshiro256Plus::seed_from_u64(0);
        // Use regular evenly spaced data to avoid numerical issue
        // and getting a smooth surrogate modeling
        // Otherwise with Lhs and bad luck this test fails from time to time
        // when surrogate modeling happens to be wrong
        let xt = Array1::linspace(0., 1., 100).insert_axis(Axis(1));
        let yt = f_test_1d(&xt);

        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(3))
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Smooth(Some(0.5)))
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        let x = Array1::linspace(0., 1., 50).insert_axis(Axis(1));
        let preds = moe.predict(&x).expect("MOE prediction");
        let dpreds = moe.predict_gradients(&x).expect("MOE drv prediction");

        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();
        write_npy(format!("{test_dir}/x_moe_smooth.npy"), &x).expect("x saved");
        write_npy(format!("{test_dir}/preds_moe_smooth.npy"), &preds).expect("preds saved");
        write_npy(format!("{test_dir}/dpreds_moe_smooth.npy"), &dpreds).expect("dpreds saved");

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        for _ in 0..100 {
            let x1: f64 = rng.gen_range(0.1..0.9);
            let h = 1e-8;
            let xtest = array![[x1]];

            let x = array![[x1], [x1 + h], [x1 - h]];
            let preds = moe.predict(&x).unwrap();
            let fdiff = (preds[1] - preds[2]) / (2. * h);

            let drv = moe.predict_gradients(&xtest).unwrap();
            let df = df_test_1d(&xtest);

            // Check only computed derivatives against fdiff of computed prediction
            // and fdiff can be wrong wrt to true derivatives due to bad surrogate modeling
            // specially at discontinuities hence no check against true derivative here
            let err = if drv[[0, 0]] < 1e-2 {
                (drv[[0, 0]] - fdiff).abs()
            } else {
                (drv[[0, 0]] - fdiff).abs() / drv[[0, 0]] // check relative error
            };
            println!(
                "Test predicted derivatives at {xtest}: drv {drv}, true df {df}, fdiff {fdiff}"
            );
            println!("preds(x, x+h, x-h)={preds}");
            assert_abs_diff_eq!(err, 0.0, epsilon = 1e-1);
        }
    }

    fn norm1(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.abs())
            .sum_axis(Axis(1))
            .insert_axis(Axis(1))
            .to_owned()
    }

    fn rosenb(x: &Array2<f64>) -> Array2<f64> {
        let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
        Zip::from(y.rows_mut())
            .and(x.rows())
            .par_for_each(|mut yi, xi| yi.assign(&array![rosenbrock(&xi.to_vec())]));
        y
    }

    #[allow(clippy::excessive_precision)]
    fn test_variance_derivatives(f: fn(&Array2<f64>) -> Array2<f64>) {
        let rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = egobox_doe::FullFactorial::new(&array![[-1., 1.], [-1., 1.]]).sample(100);
        let yt = f(&xt);

        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(2))
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Smooth(Some(1.)))
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt.remove_axis(Axis(1))))
            .expect("MOE fitted");

        for _ in 0..20 {
            let mut rng = Xoshiro256Plus::seed_from_u64(42);
            let x = Array::random_using((2,), Uniform::new(0., 1.), &mut rng);
            let xa: f64 = x[0];
            let xb: f64 = x[1];
            let e = 1e-4;

            println!("Test derivatives at [{xa}, {xb}]");

            let x = array![
                [xa, xb],
                [xa + e, xb],
                [xa - e, xb],
                [xa, xb + e],
                [xa, xb - e]
            ];
            let y_pred = moe.predict(&x).unwrap();
            let y_deriv = moe.predict_gradients(&x).unwrap();

            let diff_g = (y_pred[1] - y_pred[2]) / (2. * e);
            let diff_d = (y_pred[3] - y_pred[4]) / (2. * e);

            assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
            assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);

            let y_pred = moe.predict_var(&x).unwrap();
            let y_deriv = moe.predict_var_gradients(&x).unwrap();

            let diff_g = (y_pred[1] - y_pred[2]) / (2. * e);
            let diff_d = (y_pred[3] - y_pred[4]) / (2. * e);

            assert_rel_or_abs_error(y_deriv[[0, 0]], diff_g);
            assert_rel_or_abs_error(y_deriv[[0, 1]], diff_d);
        }
    }

    /// Test prediction valvar derivatives against derivatives and variance derivatives
    #[test]
    fn test_valvar_predictions() {
        let rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = egobox_doe::FullFactorial::new(&array![[-1., 1.], [-1., 1.]]).sample(100);
        let yt = rosenb(&xt).remove_axis(Axis(1));

        for corr in [
            CorrelationSpec::SQUAREDEXPONENTIAL,
            CorrelationSpec::MATERN32,
            CorrelationSpec::MATERN52,
        ] {
            println!("Test valvar derivatives with correlation {corr:?}");
            for recomb in [
                Recombination::Hard,
                Recombination::Smooth(Some(0.5)),
                Recombination::Smooth(None),
            ] {
                println!("Testing valvar derivatives with recomb={recomb:?}");

                let moe = GpMixture::params()
                    .n_clusters(NbClusters::fixed(2))
                    .regression_spec(RegressionSpec::CONSTANT)
                    .correlation_spec(corr)
                    .recombination(recomb)
                    .with_rng(rng.clone())
                    .fit(&Dataset::new(xt.to_owned(), yt.to_owned()))
                    .expect("MOE fitted");

                for _ in 0..10 {
                    let mut rng = Xoshiro256Plus::seed_from_u64(42);
                    let x = Array::random_using((2,), Uniform::new(0., 1.), &mut rng);
                    let xa: f64 = x[0];
                    let xb: f64 = x[1];
                    let e = 1e-4;

                    let x = array![
                        [xa, xb],
                        [xa + e, xb],
                        [xa - e, xb],
                        [xa, xb + e],
                        [xa, xb - e]
                    ];
                    let (y_pred, v_pred) = moe.predict_valvar(&x).unwrap();
                    let (y_deriv, v_deriv) = moe.predict_valvar_gradients(&x).unwrap();

                    let pred = moe.predict(&x).unwrap();
                    let var = moe.predict_var(&x).unwrap();
                    assert_abs_diff_eq!(y_pred, pred, epsilon = 1e-12);
                    assert_abs_diff_eq!(v_pred, var, epsilon = 1e-12);

                    let deriv = moe.predict_gradients(&x).unwrap();
                    let vardrv = moe.predict_var_gradients(&x).unwrap();
                    assert_abs_diff_eq!(y_deriv, deriv, epsilon = 1e-12);
                    assert_abs_diff_eq!(v_deriv, vardrv, epsilon = 1e-12);
                }
            }
        }
    }

    fn assert_rel_or_abs_error(y_deriv: f64, fdiff: f64) {
        println!("analytic deriv = {y_deriv}, fdiff = {fdiff}");
        if fdiff.abs() < 1e-2 {
            assert_abs_diff_eq!(y_deriv, 0.0, epsilon = 1e-1); // check absolute when close to zero
        } else {
            let drv_rel_error1 = (y_deriv - fdiff).abs() / fdiff; // check relative
            assert_abs_diff_eq!(drv_rel_error1, 0.0, epsilon = 1e-1);
        }
    }

    #[test]
    fn test_moe_var_deriv_norm1() {
        test_variance_derivatives(norm1);
    }
    #[test]
    fn test_moe_var_deriv_rosenb() {
        test_variance_derivatives(rosenb);
    }

    #[test]
    fn test_moe_display() {
        let rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = Lhs::new(&array![[0., 1.]])
            .with_rng(rng.clone())
            .sample(100);
        let yt = f_test_1d(&xt);

        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(3))
            .regression_spec(RegressionSpec::CONSTANT)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .recombination(Recombination::Hard)
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        // Values may vary depending on the platforms and linalg backends
        // assert_eq!("Mixture[Hard](Constant_SquaredExponentialGP(mean=ConstantMean, corr=SquaredExponential, theta=[0.03871601282054056], variance=[0.276011431746834], likelihood=454.17113736397033), Constant_SquaredExponentialGP(mean=ConstantMean, corr=SquaredExponential, theta=[0.07903503494417609], variance=[0.0077182164672893756], likelihood=436.39615700140183), Constant_SquaredExponentialGP(mean=ConstantMean, corr=SquaredExponential, theta=[0.050821466014058826], variance=[0.32824998062969973], likelihood=193.19339252734846))", moe.to_string());
        println!("Display moe: {moe}");
    }

    fn griewank(x: &Array2<f64>) -> Array1<f64> {
        let dim = x.ncols();
        let d = Array1::linspace(1., dim as f64, dim).mapv(|v| v.sqrt());
        let mut y = Array1::zeros((x.nrows(),));
        Zip::from(&mut y).and(x.rows()).for_each(|y, x| {
            let s = x.mapv(|v| v * v).sum() / 4000.;
            let p = (x.to_owned() / &d)
                .mapv(|v| v.cos())
                .fold(1., |acc, x| acc * x);
            *y = s - p + 1.;
        });
        y
    }

    #[test]
    fn test_kpls_griewank() {
        let dims = [100];
        let nts = [100];
        let lim = array![[-600., 600.]];

        let test_dir = "target/tests";
        std::fs::create_dir_all(test_dir).ok();

        (0..1).for_each(|i| {
            let dim = dims[i];
            let nt = nts[i];
            let xlimits = lim.broadcast((dim, 2)).unwrap();

            let prefix = "griewank";
            let xfilename = format!("{test_dir}/{prefix}_xt_{nt}x{dim}.npy");
            let yfilename = format!("{test_dir}/{prefix}_yt_{nt}x1.npy");

            let rng = Xoshiro256Plus::seed_from_u64(42);
            let xt = Lhs::new(&xlimits).with_rng(rng).sample(nt);
            write_npy(xfilename, &xt).expect("cannot save xt");
            let yt = griewank(&xt);
            write_npy(yfilename, &yt).expect("cannot save yt");

            let gp = GpMixture::params()
                .n_clusters(NbClusters::default())
                .regression_spec(RegressionSpec::CONSTANT)
                .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
                .kpls_dim(Some(3))
                .fit(&Dataset::new(xt, yt))
                .expect("GP fit error");

            // To see file size : 100D => json ~ 1.2Mo, bin ~ 0.6Mo
            // gp.save("griewank.json", GpFileFormat::Json).unwrap();
            // gp.save("griewank.bin", GpFileFormat::Binary).unwrap();

            let rng = Xoshiro256Plus::seed_from_u64(0);
            let xtest = Lhs::new(&xlimits).with_rng(rng).sample(100);
            let ytest = gp.predict(&xtest).expect("prediction error");
            let ytrue = griewank(&xtest);

            let nrmse = (ytrue.to_owned() - &ytest).norm_l2() / ytrue.norm_l2();
            println!(
                "diff={}  ytrue={} nrsme={}",
                (ytrue.to_owned() - &ytest).norm_l2(),
                ytrue.norm_l2(),
                nrmse
            );
            assert_abs_diff_eq!(nrmse, 0., epsilon = 1e-2);
        });
    }

    fn sphere(x: &Array2<f64>) -> Array1<f64> {
        (x * x)
            .sum_axis(Axis(1))
            .into_shape_with_order((x.nrows(),))
            .expect("Cannot reshape sphere output")
    }

    #[test]
    fn test_moe_smooth_vs_hard_one_cluster() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Array2::random_using((50, 2), Uniform::new(0., 1.), &mut rng);
        let yt = sphere(&xt);
        let ds = Dataset::new(xt, yt.to_owned());

        // Fit hard
        let moe_hard = GpMixture::params()
            .n_clusters(NbClusters::fixed(1))
            .recombination(Recombination::Hard)
            .with_rng(rng.clone())
            .fit(&ds)
            .expect("MOE hard fitted");

        // Fit smooth
        let moe_smooth = GpMixture::params()
            .n_clusters(NbClusters::fixed(1))
            .recombination(Recombination::Smooth(Some(1.0)))
            .with_rng(rng)
            .fit(&ds)
            .expect("MOE smooth fitted");

        // Predict
        let mut rng = Xoshiro256Plus::seed_from_u64(43);
        let x = Array2::random_using((1, 2), Uniform::new(0., 1.), &mut rng);
        let preds_hard = moe_hard.predict(&x).expect("MOE hard prediction");
        let preds_smooth = moe_smooth.predict(&x).expect("MOE smooth prediction");
        println!("predict hard = {preds_hard} smooth = {preds_smooth}");
        assert_abs_diff_eq!(preds_hard, preds_smooth, epsilon = 1e-5);

        // Predict var
        let preds_hard = moe_hard.predict_var(&x).expect("MOE hard prediction");
        let preds_smooth = moe_smooth.predict_var(&x).expect("MOE smooth prediction");
        assert_abs_diff_eq!(preds_hard, preds_smooth, epsilon = 1e-5);

        // Predict gradients
        println!("Check pred gradients at x = {x}");
        let preds_smooth = moe_smooth
            .predict_gradients(&x)
            .expect("MOE smooth prediction");
        println!("smooth gradients = {preds_smooth}");
        let preds_hard = moe_hard.predict_gradients(&x).expect("MOE hard prediction");
        assert_abs_diff_eq!(preds_hard, preds_smooth, epsilon = 1e-5);

        // Predict var gradients
        let preds_hard = moe_hard
            .predict_var_gradients(&x)
            .expect("MOE hard prediction");
        let preds_smooth = moe_smooth
            .predict_var_gradients(&x)
            .expect("MOE smooth prediction");
        assert_abs_diff_eq!(preds_hard, preds_smooth, epsilon = 1e-5);
    }
}

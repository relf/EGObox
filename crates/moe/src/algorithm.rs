use super::gaussian_mixture::GaussianMixture;
use crate::GpType;
use crate::clustering::{find_best_number_of_clusters, sort_by_cluster};
use crate::errors::MoeError;
use crate::errors::Result;
use crate::experts::{
    Expert, allowed_experts, cross_validation_error, gp_expert_params, sgp_expert_params,
};
use crate::parameters::{GpMixtureParams, GpMixtureValidParams};
use crate::persistence::{read_model, write_model};
use crate::{GpMetrics, IaeAlphaPlotData, types::*};
use crate::{NbClusters, surrogates::*};

use linfa::traits::{Fit, Predict, PredictInplace};
use linfa::{Dataset, DatasetBase, Float, ParamGuard};
use linfa_clustering::GaussianMixtureModel;
use log::{debug, info, trace};
use std::cmp::Ordering;
use std::ops::Sub;

use ndarray::{
    Array1, Array2, Array3, ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2, Zip, concatenate, s,
};

use ndarray_rand::rand::Rng;
use ndarray_stats::QuantileExt;

use serde::{Deserialize, Serialize};

impl<D: Data<Elem = f64>> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>, MoeError>
    for GpMixtureValidParams<f64>
{
    type Object = GpMixture;

    /// Fit Moe parameters using maximum likelihood
    ///
    /// # Errors
    ///
    /// * [MoeError::ClusteringError]: if a cluster of a multi-cluster mixture has fewer than
    ///   3 points, or too few points for every allowed regression trend,
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
        for (nc, cluster) in clusters.iter().enumerate() {
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
        let candidates = allowed_experts(self.regression_spec(), self.correlation_spec());

        debug!("Find best expert");
        let best = match candidates.as_slice() {
            [] => {
                return Err(MoeError::ExpertError(
                    "No expert allowed by regression and correlation specs".to_string(),
                ));
            }
            [expert] => (*expert, None), // shortcut
            _ => {
                let errors: Vec<(Expert, f64)> = candidates
                    .iter()
                    .map(|&expert| {
                        let err = cross_validation_error(
                            expert,
                            self.kpls_dim(),
                            self.noise_ratio(),
                            &mut dataset,
                        );
                        (expert, err)
                    })
                    .collect();
                debug!(
                    "Accuracies {:?}",
                    errors
                        .iter()
                        .map(|(expert, err)| (expert.to_string(), *err))
                        .collect::<Vec<_>>()
                );
                let (expert, err) = errors
                    .iter()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap();
                (*expert, Some(*err))
            }
        };
        debug!("after Find best expert");

        let expert = match self.gp_type() {
            GpType::FullGp => {
                let mut expert_params = gp_expert_params(best.0);
                expert_params.n_start(self.n_start());
                expert_params.max_eval(self.max_eval());
                expert_params.kpls_dim(self.kpls_dim());
                expert_params.noise_ratio(self.noise_ratio().clone());
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
                let mut expert_params = sgp_expert_params(best.0, inducings.to_owned())
                    .ok_or_else(|| MoeError::ExpertError(format!("Unknown expert {}", best.0)))?;
                // a seed has to be drawn: drawing an Option may give None (entropy)
                let seed = Some(self.rng().r#gen::<u64>());
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

/// Minimum number of points in each cluster of a multi-cluster mixture
pub(crate) const MIN_CLUSTER_POINTS: usize = 3;

/// Number of points needed to fit the least demanding trend allowed by `regr`
/// in dimension `dim`
fn min_trend_points(dim: usize, regr: RegressionSpec) -> usize {
    if regr.contains(RegressionSpec::CONSTANT) {
        1
    } else if regr.contains(RegressionSpec::LINEAR) {
        dim + 1
    } else if regr.contains(RegressionSpec::QUADRATIC) {
        (dim + 1) * (dim + 2) / 2
    } else {
        1 // no trend allowed: reported when selecting the expert
    }
}

/// Check that each cluster (rows of `[x, y]` data) of a multi-cluster mixture
/// has enough points to train an expert with one of the allowed trends
fn check_number_of_points<F>(
    clusters: &[ArrayBase<impl Data<Elem = F>, Ix2>],
    dim: usize,
    regr: RegressionSpec,
) -> Result<()> {
    if clusters.len() > 1 {
        let min_points = MIN_CLUSTER_POINTS.max(min_trend_points(dim, regr));
        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.nrows() < min_points {
                return Err(MoeError::ClusteringError(format!(
                    "Not enough points in cluster {i}: requires at least {min_points}, got {}",
                    cluster.nrows()
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
#[derive(Serialize, Deserialize, Clone)]
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
        write!(f, "Mixture[{}]({})", recomb, experts)
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

#[typetag::serde]
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
    fn save(&self, path: &str, format: GpFileFormat) -> Result<()> {
        write_model(path, self, format)
    }
}

#[typetag::serde]
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

    fn predict_valvar_with_gradients(
        &self,
        x: &ArrayView2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>)> {
        match self.recombination {
            Recombination::Hard => self.predict_valvar_with_gradients_hard(x),
            Recombination::Smooth(_) => self.predict_valvar_with_gradients_smooth(x),
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

    fn predict_covariance(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if self.n_clusters() != 1 {
            return Err(MoeError::SampleError(format!(
                "Can not compute posterior covariance when several clusters {}",
                self.n_clusters()
            )));
        }
        self.experts[0].predict_covariance(x)
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

#[typetag::serde]
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

#[typetag::serde]
impl MixtureGpSurrogate for GpMixture {
    /// Selected experts in the mixture
    fn experts(&self) -> &Vec<Box<dyn FullGpSurrogate>> {
        &self.experts
    }

    /// Update the mixture with new data points
    fn update(
        &self,
        x_new: &ndarray::ArrayView2<f64>,
        y_new: &ndarray::ArrayView1<f64>,
    ) -> crate::errors::Result<Box<dyn MixtureGpSurrogate>> {
        // Clone self to call the consuming update method
        Ok(Box::new(GpMixture::update(self.clone(), x_new, y_new)?))
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
        let (_, _, _, var_drv) = self.predict_valvar_with_gradients_smooth(x)?;
        Ok(var_drv)
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
        let (_, _, val_drv, var_drv) = self.predict_valvar_with_gradients_smooth(x)?;
        Ok((val_drv, var_drv))
    }

    /// Predict outputs, variances and their derivatives at a set of points `x` specified as (n, nx) matrix.
    /// The smooth recombination of each cluster expert responsability is used to get the result:
    /// with p_k the responsability of the kth expert predicting (mu_k, var_k),
    /// mu = sum_k p_k mu_k and var = sum_k p_k^2 var_k.
    #[allow(clippy::type_complexity)]
    fn predict_valvar_with_gradients_smooth(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>)> {
        let probas = self.gmx.predict_probas(x);
        // (n, n_clusters, nx)
        let probas_drv = self.gmx.predict_probas_derivatives(x);

        let mut vals = Array1::<f64>::zeros(x.nrows());
        let mut vars = Array1::<f64>::zeros(x.nrows());
        let mut val_drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let mut var_drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        for (i, xi) in x.rows().into_iter().enumerate() {
            let xii = xi.insert_axis(Axis(0));
            for (k, gp) in self.experts.iter().enumerate() {
                let (val, var, dval, dvar) = gp.predict_valvar_with_gradients(&xii)?;
                let (val, var) = (val[0], var[0]);
                let p = probas[[i, k]];
                let dp = probas_drv.slice(s![i, k, ..]);
                // dmu = sum_k p_k dmu_k + dp_k mu_k
                vals[i] += p * val;
                val_drv.row_mut(i).scaled_add(p, &dval.row(0));
                val_drv.row_mut(i).scaled_add(val, &dp);
                // dvar = sum_k p_k^2 dvar_k + 2 p_k dp_k var_k
                vars[i] += p * p * var;
                var_drv.row_mut(i).scaled_add(p * p, &dvar.row(0));
                var_drv.row_mut(i).scaled_add(2. * p * var, &dp);
            }
        }
        Ok((vals, vars, val_drv, var_drv))
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

    /// Predict outputs, variances and their derivatives at a set of points `x` specified as (n, nx) matrix.
    /// Gaussian Mixture is used to get the cluster where the point belongs (highest responsability)
    /// The expert of the cluster is used to predict the values.
    #[allow(clippy::type_complexity)]
    pub fn predict_valvar_with_gradients_hard(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>)> {
        let mut vals = Array1::<f64>::zeros(x.nrows());
        let mut vars = Array1::<f64>::zeros(x.nrows());
        let mut val_drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let mut var_drv = Array2::<f64>::zeros((x.nrows(), x.ncols()));
        let clustering = self.gmx.predict(x);
        for (i, (xi, &c)) in x.rows().into_iter().zip(&clustering).enumerate() {
            let (val, var, dval, dvar) =
                self.experts[c].predict_valvar_with_gradients(&xi.insert_axis(Axis(0)))?;
            vals[i] = val[0];
            vars[i] = var[0];
            val_drv.row_mut(i).assign(&dval.row(0));
            var_drv.row_mut(i).assign(&dvar.row(0));
        }
        Ok((vals, vars, val_drv, var_drv))
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

    /// Predict outputs, variances and their derivatives at a set of points `x` specified as (n, nx) matrix.
    #[allow(clippy::type_complexity)]
    pub fn predict_valvar_with_gradients(
        &self,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>)> {
        <GpMixture as GpSurrogateExt>::predict_valvar_with_gradients(self, &x.view())
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

    /// Update the mixture of experts with new data points.
    ///
    /// # Strategy based on number of clusters:
    ///
    /// - **Single expert (n_clusters == 1)**: Uses the efficient GP update method
    ///   with Cholesky rank-1 updates for fast incremental learning.
    /// - **Multiple experts (n_clusters > 1)**: Assigns new points to clusters based on
    ///   current Gaussian mixture, then refits each expert with its assigned points
    ///   using fixed theta (current theta as initialization) to preserve model structure.
    ///
    /// # Arguments
    ///
    /// * `x_new` - New input data points as a (n_new, nx) matrix
    /// * `y_new` - New output data values as a (n_new,) vector
    ///
    /// # Returns
    ///
    /// A new `GpMixture` instance updated with the new data
    ///
    /// # Notes
    ///
    /// - For single expert: truly incremental, fast update
    /// - For multiple experts: preserves cluster assignments and theta initialization
    /// - The number of clusters remains unchanged
    pub fn update(
        self,
        x_new: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y_new: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    ) -> Result<GpMixture> {
        let n_clusters = self.n_clusters();

        if n_clusters == 1 {
            // Single expert: use efficient GP update (consume self)
            debug!(
                "Updating single-expert mixture with {} new points",
                x_new.nrows()
            );
            // Extract training data before moving experts
            let (xt_old, yt_old) = <GpMixture as GpMetrics<_, _, _>>::training_data(&self).clone();

            let expert = self.experts.into_iter().next().unwrap();
            let y_new_col = y_new.to_owned().insert_axis(Axis(1));
            let updated_expert = expert.update(&x_new.view(), &y_new_col.view())?;
            let xt_combined =
                concatenate(Axis(0), &[xt_old.view(), x_new.view()]).map_err(|e| {
                    MoeError::GpError(egobox_gp::GpError::InvalidValueError(e.to_string()))
                })?;
            let yt_combined =
                concatenate(Axis(0), &[yt_old.view(), y_new.view()]).map_err(|e| {
                    MoeError::GpError(egobox_gp::GpError::InvalidValueError(e.to_string()))
                })?;

            // Create new GpMixture with updated data
            let moe = GpMixture {
                recombination: self.recombination,
                experts: vec![updated_expert],
                gmx: self.gmx,
                gp_type: self.gp_type,
                training_data: (xt_combined.to_owned(), yt_combined.to_owned()),
                params: self.params,
            };
            Ok(moe)
        } else {
            // Multiple experts: assign new points to clusters and update each expert
            debug!(
                "Updating multi-expert mixture ({} clusters) with {} new points",
                n_clusters,
                x_new.nrows()
            );

            // Assign NEW points to clusters using current Gaussian mixture
            let new_resp = self.gmx.predict_probas(x_new);

            // Collect new data for each cluster
            let mut new_cluster_x = vec![Vec::new(); n_clusters];
            let mut new_cluster_y = vec![Vec::new(); n_clusters];

            for i in 0..x_new.nrows() {
                let resp = new_resp.row(i);
                let cluster_idx = resp.argmax().unwrap_or(0);
                new_cluster_x[cluster_idx].extend(x_new.row(i).iter().cloned());
                new_cluster_y[cluster_idx].push(y_new[i]);
            }

            // Extract training data before moving experts
            let (xt_old, yt_old) = <GpMixture as GpMetrics<_, _, _>>::training_data(&self).clone();

            // Update each expert with its newly assigned points
            // Move experts vector out of self
            let experts = self.experts;
            let mut updated_experts: Vec<Box<dyn FullGpSurrogate>> = Vec::with_capacity(n_clusters);
            for cluster_idx in 0..n_clusters {
                if new_cluster_x[cluster_idx].is_empty() {
                    // No new data for this cluster, keep the old expert
                    let expert = experts[cluster_idx].clone();
                    updated_experts.push(expert);
                    info!(
                        "  Cluster {}: no new points, keeping old expert",
                        cluster_idx
                    );
                } else {
                    let n_points = new_cluster_y[cluster_idx].len();
                    let x_new_cluster: Array2<f64> = Array2::from_shape_vec(
                        (n_points, x_new.ncols()),
                        new_cluster_x[cluster_idx].clone(),
                    )
                    .map_err(|e| {
                        MoeError::GpError(egobox_gp::GpError::InvalidValueError(e.to_string()))
                    })?;
                    let y_new_cluster: Array1<f64> =
                        Array1::from_vec(new_cluster_y[cluster_idx].clone());

                    // Use efficient GP update for this expert
                    info!(
                        "  Cluster {}: updating expert with {} new points",
                        cluster_idx, n_points
                    );
                    let expert = experts[cluster_idx].clone();
                    let updated_expert = expert.update(
                        &x_new_cluster.view(),
                        &y_new_cluster.insert_axis(Axis(1)).view(),
                    )?;
                    updated_experts.push(updated_expert);
                }
            }

            // Combine training data (already extracted earlier)
            let xt_combined =
                concatenate(Axis(0), &[xt_old.view(), x_new.view()]).map_err(|e| {
                    MoeError::GpError(egobox_gp::GpError::InvalidValueError(e.to_string()))
                })?;
            let yt_combined =
                concatenate(Axis(0), &[yt_old.view(), y_new.view()]).map_err(|e| {
                    MoeError::GpError(egobox_gp::GpError::InvalidValueError(e.to_string()))
                })?;

            // Create new GpMixture with updated data, moving fields (no clone)
            let moe = GpMixture {
                recombination: self.recombination, // Moved
                experts: updated_experts,
                gmx: self.gmx,         // Moved (no GMM retraining)
                gp_type: self.gp_type, // Moved
                training_data: (xt_combined.to_owned(), yt_combined.to_owned()),
                params: self.params, // Moved
            };
            Ok(moe)
        }
    }

    /// Load Moe from the given file.
    pub fn load(path: &str, format: GpFileFormat) -> Result<Box<GpMixture>> {
        read_model(path, format).map(Box::new)
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
    use egobox_gp::ParamTuning;
    #[cfg(not(feature = "blas"))]
    use linfa_linalg::norm::*;
    use ndarray::{Array, Array2, Zip, array};
    #[cfg(feature = "blas")]
    use ndarray_linalg::Norm;
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
    fn test_moe_predict_covariance() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = Array2::random_using((20, 1), Uniform::new(0., 1.), &mut rng);
        let yt = f_test_1d(&xt.to_owned());
        let x = array![[0.1], [0.12], [0.5], [0.9]];
        // one cluster: the covariance of the expert
        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(1))
            .with_rng(rng.clone())
            .fit(&Dataset::new(xt.clone(), yt.clone()))
            .expect("MOE fitted");
        let cov =
            <GpMixture as GpSurrogateExt>::predict_covariance(&moe, &x.view()).expect("covariance");
        assert_eq!(cov.dim(), (4, 4));
        let var = moe.predict_var(&x).expect("variance");
        assert_abs_diff_eq!(cov.diag(), var, epsilon = 1e-9);
        // affine transformation: the covariance is scaled by scale^2
        let affined = crate::AffinedSurrogate::new(
            Box::new(moe.clone()) as Box<dyn MixtureGpSurrogate>,
            -2.,
            1.,
        );
        let cov_affined = affined.predict_covariance(&x.view()).expect("covariance");
        assert_abs_diff_eq!(cov_affined, &cov * 4., epsilon = 1e-9);
        // several clusters: not supported
        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(2))
            .with_rng(rng)
            .fit(&Dataset::new(xt, yt))
            .expect("MOE fitted");
        assert!(<GpMixture as GpSurrogateExt>::predict_covariance(&moe, &x.view()).is_err());
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

    fn lhs_data(
        xlimits: &Array2<f64>,
        n: usize,
        f: impl Fn(ndarray::ArrayView1<f64>) -> f64,
    ) -> (Array2<f64>, Xoshiro256Plus) {
        let rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(xlimits).with_rng(rng.clone()).sample(n);
        let yt = xt.map_axis(Axis(1), f).insert_axis(Axis(1));
        // row major layout as for clustered data
        let data = concatenate(Axis(1), &[xt.view(), yt.view()]).unwrap();
        (data.as_standard_layout().to_owned(), rng)
    }

    #[test]
    fn test_find_best_expert_quadratic_2d() {
        // Exact quadratic function: the quadratic trend is the best expert
        let xlimits = array![[-2., 2.], [-2., 2.]];
        let (data, rng) = lhs_data(&xlimits, 30, |x| {
            1. + x[0] - 2. * x[1] + 3. * x[0] * x[0] + x[0] * x[1] + 2. * x[1] * x[1]
        });
        let moe = GpMixture::params()
            .regression_spec(RegressionSpec::ALL)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .with_rng(rng)
            .check_unwrap();
        let best_expert = moe.find_best_expert(0, 2, &data).unwrap();
        assert!(
            best_expert.to_string().starts_with("Quadratic"),
            "Best expert {best_expert}"
        );
    }

    #[test]
    fn test_find_best_expert_quadratic_not_enough_fold_points() {
        // 4 * dim samples: quadratic trend allowed while cross-validation folds
        // have less training points than the 21 quadratic terms in 5-D
        let xlimits = Array2::from_shape_fn((5, 2), |(_, j)| if j == 0 { -1. } else { 1. });
        let (data, rng) = lhs_data(&xlimits, 20, |x| x.dot(&x));
        let moe = GpMixture::params()
            .regression_spec(RegressionSpec::ALL)
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL)
            .with_rng(rng)
            .check_unwrap();
        let best_expert = moe.find_best_expert(0, 5, &data).unwrap();
        assert!(
            !best_expert.to_string().starts_with("Quadratic"),
            "Best expert {best_expert}"
        );
    }

    #[test]
    fn test_check_number_of_points_counts_rows() {
        let clusters = |rows: &[usize], dim: usize| -> Vec<Array2<f64>> {
            rows.iter().map(|&n| Array2::zeros((n, dim + 1))).collect()
        };
        let linear_or_quadratic = RegressionSpec::LINEAR | RegressionSpec::QUADRATIC;
        for (dim, regr, rows, ok) in [
            // a linear trend needs dim+1 points
            (3, RegressionSpec::LINEAR, vec![3, 10], false),
            (3, RegressionSpec::LINEAR, vec![4, 10], true),
            // a quadratic trend needs (dim+1)(dim+2)/2 points
            (2, RegressionSpec::QUADRATIC, vec![5, 10], false),
            (2, RegressionSpec::QUADRATIC, vec![6, 10], true),
            // the least demanding allowed trend sets the requirement
            (6, RegressionSpec::ALL, vec![3, 10], true),
            (6, linear_or_quadratic, vec![6, 10], false),
            (6, linear_or_quadratic, vec![7, 10], true),
            // at least 3 points per cluster
            (1, RegressionSpec::CONSTANT, vec![2, 10], false),
            (1, RegressionSpec::CONSTANT, vec![3, 10], true),
            // a single cluster is not checked
            (6, RegressionSpec::QUADRATIC, vec![1], true),
        ] {
            let res = check_number_of_points(&clusters(&rows, dim), dim, regr);
            assert_eq!(
                res.is_ok(),
                ok,
                "dim={dim} regr={regr:?} rows={rows:?}: {res:?}"
            );
        }
    }

    /// 6-D data in two well-separated clusters, of `n_small` and 30 points,
    /// with the Hard clustering separating them
    fn two_separated_clusters(n_small: usize) -> (Array2<f64>, Array1<f64>, Clustering) {
        let dim = 6;
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let small = Array2::random_using((n_small, dim), Uniform::new(0., 1.), &mut rng);
        let large = Array2::random_using((30, dim), Uniform::new(10., 11.), &mut rng);
        let xt = concatenate(Axis(0), &[small.view(), large.view()]).unwrap();
        let yt = xt.sum_axis(Axis(1));
        let means = Array2::from_shape_fn((2, dim), |(i, _)| if i == 0 { 0.5 } else { 10.5 });
        let covariances =
            Array3::from_shape_fn((2, dim, dim), |(_, i, j)| if i == j { 0.1 } else { 0. });
        let gmx = GaussianMixture::new(array![0.5, 0.5], means, covariances).unwrap();
        (xt, yt, Clustering::new(gmx, Recombination::Hard))
    }

    #[test]
    fn test_train_on_clusters_small_cluster_with_constant_trend() {
        // 3 points: too few for a linear (7) or quadratic (28) trend in 6-D,
        // enough for the constant one, which is allowed
        let (xt, yt, clustering) = two_separated_clusters(3);
        let moe = GpMixture::params()
            .regression_spec(RegressionSpec::ALL)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .check_unwrap()
            .train_on_clusters(&xt, &yt, &clustering)
            .expect("mixture trained");
        assert_eq!(moe.n_clusters(), 2);
    }

    #[test]
    fn test_train_on_clusters_small_cluster_with_linear_trend() {
        // 3 points: too few for the only allowed trend, linear (7 points in 6-D)
        let (xt, yt, clustering) = two_separated_clusters(3);
        let res = GpMixture::params()
            .regression_spec(RegressionSpec::LINEAR)
            .with_rng(Xoshiro256Plus::seed_from_u64(42))
            .check_unwrap()
            .train_on_clusters(&xt, &yt, &clustering);
        assert!(
            matches!(res, Err(MoeError::ClusteringError(_))),
            "{:?}",
            res.map(|moe| moe.to_string())
        );
    }

    #[test]
    fn test_sparse_mixture_seeded_training_is_reproducible() {
        // seed 42 used to draw no seed for the sparse GP expert (entropy)
        let xlimits = array![[0., 25.]];
        let (data, _) = lhs_data(&xlimits, 100, |x| {
            (x[0] - 3.5) * ((x[0] - 3.5) / std::f64::consts::PI).sin()
        });
        let xt = data.slice(s![.., ..1]).to_owned();
        let yt = data.column(1).to_owned();
        let xtest = array![[1.], [7.5], [12.3], [20.]];
        let train = || {
            GpMixture::params()
                .gp_type(GpType::SparseGp {
                    sparse_method: egobox_gp::SparseMethod::Fitc,
                    inducings: egobox_gp::Inducings::Randomized(10),
                })
                .with_rng(Xoshiro256Plus::seed_from_u64(42))
                .fit(&Dataset::new(xt.clone(), yt.clone()))
                .unwrap()
                .predict(&xtest)
                .unwrap()
        };
        assert_eq!(train(), train());
    }

    #[test]
    fn test_no_allowed_expert_is_an_error() {
        let xt = array![[0.], [1.], [2.], [3.], [4.], [5.]];
        let yt = xt.column(0).mapv(f64::sin);
        for (regr, corr) in [
            (RegressionSpec::empty(), CorrelationSpec::ALL),
            (RegressionSpec::ALL, CorrelationSpec::empty()),
        ] {
            let res = GpMixture::params()
                .regression_spec(regr)
                .correlation_spec(corr)
                .fit(&Dataset::new(xt.clone(), yt.clone()));
            assert!(
                matches!(res, Err(MoeError::ExpertError(_))),
                "{regr:?}/{corr:?}"
            );
        }
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

    fn sin_mix(x: &Array2<f64>) -> Array1<f64> {
        let x0 = x.column(0);
        let x1 = x.column(1);
        let x2 = x.column(2);
        &x0 * &x0 + x1.mapv(f64::sin) * x2 + &x0 * &x2
    }

    /// Check analytic derivatives `deriv` (n, nx) at `x` against central finite differences of `f`
    fn assert_fdiff_close(
        f: impl Fn(&Array2<f64>) -> Array1<f64>,
        x: &Array2<f64>,
        deriv: &Array2<f64>,
        scale: f64,
    ) {
        let e = 1e-6;
        for (i, xi) in x.rows().into_iter().enumerate() {
            for k in 0..x.ncols() {
                let mut xp = xi.to_owned().insert_axis(Axis(0));
                let mut xm = xp.clone();
                xp[[0, k]] += e;
                xm[[0, k]] -= e;
                let fdiff = (f(&xp)[0] - f(&xm)[0]) / (2. * e);
                assert_abs_diff_eq!(
                    deriv[[i, k]],
                    fdiff,
                    epsilon = 1e-2 * fdiff.abs() + 1e-9 * scale
                );
            }
        }
    }

    /// Recombinations with more input dimensions than clusters
    #[test]
    fn test_valvar_with_gradients_3d() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let xt = Lhs::new(&array![[-3., 3.], [-3., 3.], [-3., 3.]])
            .with_rng(rng.clone())
            .sample(60);
        let yt = sin_mix(&xt);
        let x = Array::random_using((5, 3), Uniform::new(-3., 3.), &mut rng);

        for recomb in [Recombination::Hard, Recombination::Smooth(Some(1.))] {
            let moe = GpMixture::params()
                .n_clusters(NbClusters::fixed(2))
                .recombination(recomb)
                .with_rng(rng.clone())
                .fit(&Dataset::new(xt.to_owned(), yt.to_owned()))
                .expect("MOE fitted");

            let (vals, vars, dvals, dvars) = moe.predict_valvar_with_gradients(&x).unwrap();
            let val_scale = 1. + vals.mapv(f64::abs).sum();
            let var_scale = 1. + vars.mapv(f64::abs).sum();
            assert_abs_diff_eq!(vals, moe.predict(&x).unwrap(), epsilon = 1e-10 * val_scale);
            assert_abs_diff_eq!(
                vars,
                moe.predict_var(&x).unwrap(),
                epsilon = 1e-10 * var_scale
            );
            let (val_grads, var_grads) = moe.predict_valvar_gradients(&x).unwrap();
            assert_abs_diff_eq!(
                dvals,
                moe.predict_gradients(&x).unwrap(),
                epsilon = 1e-10 * val_scale
            );
            assert_abs_diff_eq!(dvals, val_grads, epsilon = 1e-10 * val_scale);
            assert_abs_diff_eq!(
                dvars,
                moe.predict_var_gradients(&x).unwrap(),
                epsilon = 1e-10 * var_scale
            );
            assert_abs_diff_eq!(dvars, var_grads, epsilon = 1e-10 * var_scale);

            if let Recombination::Smooth(_) = recomb {
                // hard recombination is discontinuous at cluster boundaries
                assert_fdiff_close(|x| moe.predict(x).unwrap(), &x, &dvals, val_scale);
                assert_fdiff_close(|x| moe.predict_var(x).unwrap(), &x, &dvars, var_scale);
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

    /// Noisy observations of sin(x) on [0, 10] with a gaussian noise of std `noise_std`
    fn noisy_sin(nt: usize, noise_std: f64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let xt = Lhs::new(&array![[0., 10.]])
            .with_rng(rng.clone())
            .sample(nt);
        let noise = Array1::random_using(
            nt,
            ndarray_rand::rand_distr::Normal::new(0., noise_std).unwrap(),
            &mut rng,
        );
        let yt = xt.column(0).mapv(f64::sin) + noise;
        (xt, yt)
    }

    #[test]
    fn test_moe_noise_ratio() {
        let noise_std = 0.1;
        let (xt, yt) = noisy_sin(40, noise_std);
        let ds = Dataset::new(xt, yt);

        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(1))
            .with_rng(Xoshiro256Plus::seed_from_u64(0))
            .fit(&ds)
            .expect("MOE fitted");
        assert_eq!(moe.experts()[0].noise_variance(), 0.);

        // Noise ratio is also used when selecting the best expert by cross-validation
        let moe = GpMixture::params()
            .n_clusters(NbClusters::fixed(1))
            .correlation_spec(CorrelationSpec::SQUAREDEXPONENTIAL | CorrelationSpec::MATERN52)
            .noise_ratio(ParamTuning::Optimized {
                init: 1e-2,
                bounds: (1e-8, 10.),
            })
            .with_rng(Xoshiro256Plus::seed_from_u64(0))
            .fit(&ds)
            .expect("MOE fitted");
        let noise_var = noise_std * noise_std;
        let estimated = moe.experts()[0].noise_variance();
        println!("{moe}");
        assert!(
            estimated > noise_var / 4. && estimated < 4. * noise_var,
            "noise variance {estimated} should be close to {noise_var}"
        );
    }
}

//! Mixed-integer mixture of Gaussian processes
//!
//! This module exposes a GP mixture model featuring continuous relaxation to handle mixed-interger variables

#![allow(dead_code)]
use crate::errors::{MoeError, Result};
use crate::gpmix::types::XType;
use egobox_doe::{FullFactorial, Lhs, LhsKind, Random};
use crate::{
    Clustered, Clustering, FullGpSurrogate, GpMetrics, GpMixture, GpMixtureParams,
    GpQualityAssurance, GpSurrogate, GpSurrogateExt, IaeAlphaPlotData, MixtureGpSurrogate,
};
use linfa::traits::{Fit, PredictInplace};
use linfa::{DatasetBase, Float, ParamGuard};
use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, DataMut, Ix1, Ix2, Zip, s,
};
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;
use std::marker::PhantomData;

#[cfg(feature = "serializable")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "persistent")]
use crate::GpFileFormat;
#[cfg(feature = "persistent")]
use std::fs;
#[cfg(feature = "persistent")]
use std::io::Write;

/// Expand xlimits to add continuous dimensions for enumeration x features.
///
/// Each level of an enumerate gives a new continuous dimension in [0, 1].
/// Each integer dimensions are relaxed continuously.
pub fn as_continuous_limits<F: Float>(xtypes: &[XType]) -> Array2<F> {
    let mut xlimits: Vec<F> = vec![];
    let mut dim = 0;
    xtypes.iter().for_each(|xtype| match xtype {
        XType::Float(lower, upper) => {
            dim += 1;
            xlimits.extend([F::cast(*lower), F::cast(*upper)]);
        }
        XType::Int(lower, upper) => {
            dim += 1;
            xlimits.extend([F::cast(*lower), F::cast(*upper)]);
        }
        XType::Ord(v) => {
            dim += 1;
            xlimits.extend([
                (v.iter()
                    .fold(F::infinity(), |a, &b| F::cast(a).min(F::cast(b)))),
                (v.iter()
                    .fold(-F::infinity(), |a, &b| F::cast(a).max(F::cast(b)))),
            ]);
        }
        XType::Enum(v) => {
            dim += v;
            (1..=*v).for_each(|_| {
                xlimits.extend([F::zero(), F::one()]);
            })
        }
    });
    Array::from_shape_vec((dim, 2), xlimits).unwrap()
}

/// Reduce categorical inputs from discrete unfolded space to
/// initial x dimension space where categorical x dimensions are valued by the index
/// in the corresponding enumerate list.
///
/// For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
/// the input x may contain the mask [..., 0, 0, 1, ...] which will be contracted in [..., 2, ...]
/// meaning the "green" value.
/// This function is the opposite of unfold_with_enum_mask().
pub(crate) fn fold_with_enum_index<F: Float>(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> Array2<F> {
    let mut xfold = Array::zeros((x.nrows(), xtypes.len()));
    let mut unfold_index = 0;
    Zip::indexed(xfold.columns_mut()).for_each(|j, mut col| match &xtypes[j] {
        XType::Float(_, _) | XType::Int(_, _) | XType::Ord(_) => {
            col.assign(&x.column(unfold_index));
            unfold_index += 1;
        }
        XType::Enum(v) => {
            let xenum = x.slice(s![.., j..j + v]);
            let argmaxx = xenum.map_axis(Axis(1), |row| F::cast(row.argmax().unwrap()));
            col.assign(&argmaxx);
            unfold_index += v;
        }
    });
    xfold
}

/// Compute dimension when all variables are continuously relaxed
fn compute_continuous_dim(xtypes: &[XType]) -> usize {
    xtypes
        .iter()
        .map(|s| match s {
            XType::Enum(v) => *v,
            _ => 1,
        })
        .reduce(|acc, l| -> usize { acc + l })
        .unwrap()
}

/// Expand categorical inputs from initial x dimension space where categorical x dimensions
/// are valued by the index in the corresponding enumerate list to the discrete unfolded space.
///
/// For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
/// the input x may contain [..., 2, ...] which will be expanded in [..., 0, 0, 1, ...].
/// This function is the opposite of fold_with_enum_index().
pub(crate) fn unfold_with_enum_mask(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let mut xunfold = Array::zeros((x.nrows(), compute_continuous_dim(xtypes)));
    let mut unfold_index = 0;
    xtypes.iter().enumerate().for_each(|(i, s)| match s {
        XType::Float(_, _) | XType::Int(_, _) | XType::Ord(_) => {
            xunfold
                .column_mut(unfold_index)
                .assign(&x.column(unfold_index));
            unfold_index += 1;
        }
        XType::Enum(v) => {
            let mut unfold = Array::zeros((x.nrows(), *v));
            Zip::from(unfold.rows_mut())
                .and(x.rows())
                .for_each(|mut row, xrow| {
                    let index = xrow[i] as usize;
                    row[index] = 1.;
                });
            xunfold
                .slice_mut(s![.., unfold_index..unfold_index + v])
                .assign(&unfold);
            unfold_index += v;
        }
    });
    xunfold
}

/// Continuous relaxation of x given possibly discrete types
/// Alias of `unfold_with_enum_mask`
pub fn to_continuous_space(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    unfold_with_enum_mask(xtypes, x)
}

/// Find closest value to `val` in given slice `v`.
fn take_closest<F: Float>(v: &[F], val: F) -> F {
    let idx = Array::from_vec(v.to_vec())
        .map(|refval| (val - *refval).abs())
        .argmin()
        .unwrap();
    v[idx]
}

/// Project continuously relaxed values to their closer assessable values.
///
/// See cast_to_discrete_values
fn cast_to_discrete_values_mut<F: Float>(
    xtypes: &[XType],
    x: &mut ArrayBase<impl DataMut<Elem = F>, Ix2>,
) {
    let mut xcol = 0;
    xtypes.iter().for_each(|s| match s {
        XType::Float(_, _) => xcol += 1,
        XType::Int(_, _) => {
            let xround = x.column(xcol).mapv(|v| v.round()).to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        XType::Ord(v) => {
            let vals: Vec<F> = v.iter().map(|&v| F::cast(v)).collect();
            let xround = x
                .column(xcol)
                .mapv(|val| take_closest(&vals, val))
                .to_owned();
            x.column_mut(xcol).assign(&xround);
            xcol += 1;
        }
        XType::Enum(v) => {
            let mut xenum = x.slice_mut(s![.., xcol..xcol + *v]);
            let argmaxx = xenum.map_axis(Axis(1), |row| row.argmax().unwrap());
            Zip::from(xenum.rows_mut())
                .and(&argmaxx)
                .for_each(|mut row, &m| {
                    let mut xcast = Array::zeros(*v);
                    xcast[m] = F::one();
                    row.assign(&xcast);
                });
            xcol += *v;
        }
    });
}

/// Project continuously relaxed values to their closer assessable values.
///
/// Note: categorical (or enum) x dimensions are still expanded that is
/// there are still as many columns as categorical possible values for the given x dimension.
/// For instance, if an input dimension is typed ["blue", "red", "green"] in xlimits a sample/row of
/// the input x may contain the values (or mask) [..., 0, 0, 1, ...] to specify "green" for
/// this original dimension.
pub(crate) fn cast_to_discrete_values(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let mut xcast = x.to_owned();
    cast_to_discrete_values_mut(xtypes, &mut xcast);
    xcast
}

/// Convenient method to pass from continuous unfolded space to discrete folded space
pub fn to_discrete_space(
    xtypes: &[XType],
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let x = cast_to_discrete_values(xtypes, x);
    fold_with_enum_index(xtypes, &x)
}

enum Method {
    Lhs,
    FullFactorial,
    Random,
}

/// A decorator of LHS sampling that takes into account XType specifications
/// casting continuous LHS result from floats to discrete types.
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct MixintSampling<F: Float, S: egobox_doe::SamplingMethod<F>> {
    /// The continuous sampling method
    method: S,
    /// The input specifications
    xtypes: Vec<XType>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    output_in_folded_space: bool,
    phantom: PhantomData<F>,
}

impl<F: Float, S: egobox_doe::SamplingMethod<F>> MixintSampling<F, S> {
    /// Constructor using `xtypes` specifications
    pub fn new(method: S, xtypes: Vec<XType>) -> Self {
        MixintSampling {
            method,
            xtypes: xtypes.clone(),
            output_in_folded_space: false,
            phantom: PhantomData,
        }
    }

    /// Sets whether we want to work in folded space
    /// If set, sampling data will be provided in folded space
    pub fn work_in_folded_space(&mut self, output_in_folded_space: bool) -> &mut Self {
        self.output_in_folded_space = output_in_folded_space;
        self
    }
}

impl<F: Float, S: egobox_doe::SamplingMethod<F>> egobox_doe::SamplingMethod<F>
    for MixintSampling<F, S>
{
    fn sampling_space(&self) -> &Array2<F> {
        self.method.sampling_space()
    }

    fn normalized_sample(&self, ns: usize) -> Array2<F> {
        self.method.normalized_sample(ns)
    }

    fn sample(&self, ns: usize) -> Array2<F> {
        let mut doe = self.method.sample(ns);
        cast_to_discrete_values_mut(&self.xtypes, &mut doe);
        if self.output_in_folded_space {
            fold_with_enum_index(&self.xtypes, &doe.view())
        } else {
            doe
        }
    }
}

/// Moe type builder for mixed-integer Egor optimizer
pub type MoeBuilder = GpMixtureParams<f64>;
/// A decorator of Moe surrogate builder that takes into account XType specifications
///
/// It allows to implement continuous relaxation over continuous Moe builder.
#[derive(Clone)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct MixintGpMixtureValidParams {
    /// The surrogate factory
    surrogate_builder: GpMixtureParams<f64>,
    /// The input specifications
    xtypes: Vec<XType>,
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
    pub fn xtypes(&self) -> &[XType] {
        &self.xtypes
    }
}

/// Parameters for mixture of experts surrogate model
#[derive(Clone)]
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct MixintGpMixtureParams(MixintGpMixtureValidParams);

impl MixintGpMixtureParams {
    /// Constructor given `xtypes` specifications and given surrogate builder
    pub fn new(xtypes: &[XType], surrogate_builder: &MoeBuilder) -> Self {
        MixintGpMixtureParams(MixintGpMixtureValidParams {
            surrogate_builder: surrogate_builder.clone(),
            xtypes: xtypes.to_vec(),
            work_in_folded_space: false,
        })
    }

    /// Sets whether we want to work in folded space
    pub fn work_in_folded_space(&mut self, wfs: bool) -> &mut Self {
        self.0.work_in_folded_space = wfs;
        self
    }

    /// Gets the domain specification
    pub fn xtypes(&self) -> &[XType] {
        &self.0.xtypes
    }
}

impl MixintGpMixtureValidParams {
    fn _train(
        &self,
        xt: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        yt: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    ) -> Result<MixintGpMixture> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, &xt.view())
        } else {
            xt.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        let mixmoe = MixintGpMixture {
            moe: self
                .surrogate_builder
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
        clustering: &crate::Clustering,
    ) -> Result<MixintGpMixture> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, &xt.view())
        } else {
            xt.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        let mixmoe = MixintGpMixture {
            moe: self
                .surrogate_builder
                .clone()
                .check_ref()?
                .train_on_clusters(&xcast, &yt.to_owned(), clustering)
                .unwrap(),
            xtypes: self.xtypes.clone(),
            work_in_folded_space: self.work_in_folded_space,
            training_data: (xt.to_owned(), yt.to_owned()),
            params: self.clone(),
        };
        Ok(mixmoe)
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

/// The Moe model that takes into account XType specifications
#[cfg_attr(feature = "serializable", derive(Serialize, Deserialize))]
pub struct MixintGpMixture {
    /// the decorated Moe
    moe: GpMixture,
    /// The input specifications
    xtypes: Vec<XType>,
    /// whether training input data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    work_in_folded_space: bool,
    /// Training inputs
    training_data: (Array2<f64>, Array1<f64>),
    /// Parameters used to trin this model
    params: MixintGpMixtureValidParams,
}

impl std::fmt::Display for MixintGpMixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = if crate::gpmix::types::discrete(&self.xtypes) {
            "MixInt"
        } else {
            ""
        };
        write!(f, "{}{}", prefix, &self.moe)
    }
}

impl Clustered for MixintGpMixture {
    fn n_clusters(&self) -> usize {
        self.moe.n_clusters()
    }

    fn recombination(&self) -> crate::Recombination<f64> {
        self.moe.recombination()
    }

    /// Convert to clustering
    fn to_clustering(&self) -> Clustering {
        Clustering::new(self.moe.gmx().clone(), self.moe.recombination())
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpSurrogate for MixintGpMixture {
    fn dims(&self) -> (usize, usize) {
        self.moe.dims()
    }

    fn predict(&self, x: &ArrayView2<f64>) -> crate::Result<Array1<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict(&xcast)
    }

    fn predict_var(&self, x: &ArrayView2<f64>) -> crate::Result<Array1<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_var(&xcast)
    }

    fn predict_valvar(
        &self,
        x: &ArrayView2<f64>,
    ) -> crate::Result<(Array1<f64>, Array1<f64>)> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_valvar(&xcast)
    }

    /// Save Moe model in given file.
    #[cfg(feature = "persistent")]
    fn save(&self, path: &str, format: GpFileFormat) -> crate::Result<()> {
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

impl MixintGpMixture {
    /// Load MixintGpMixture from given file.
    #[cfg(feature = "persistent")]
    pub fn load(path: &str, format: GpFileFormat) -> Result<Box<MixintGpMixture>> {
        let data = fs::read(path)?;
        let moe = match format {
            GpFileFormat::Json => serde_json::from_slice(&data).unwrap(),
            GpFileFormat::Binary => {
                bincode::serde::decode_from_slice(&data, bincode::config::standard())
                    .map(|(surrogate, _)| surrogate)?
            }
        };
        Ok(Box::new(moe))
    }
}

#[cfg_attr(feature = "serializable", typetag::serde)]
impl GpSurrogateExt for MixintGpMixture {
    fn predict_gradients(&self, x: &ArrayView2<f64>) -> crate::Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_gradients(&xcast)
    }

    fn predict_var_gradients(&self, x: &ArrayView2<f64>) -> crate::Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_var_gradients(&xcast)
    }

    fn predict_valvar_gradients(
        &self,
        x: &ArrayView2<f64>,
    ) -> crate::Result<(Array2<f64>, Array2<f64>)> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.predict_valvar_gradients(&xcast)
    }

    fn sample(&self, x: &ArrayView2<f64>, n_traj: usize) -> crate::Result<Array2<f64>> {
        let mut xcast = if self.work_in_folded_space {
            unfold_with_enum_mask(&self.xtypes, x)
        } else {
            x.to_owned()
        };
        cast_to_discrete_values_mut(&self.xtypes, &mut xcast);
        self.moe.sample(&xcast.view(), n_traj)
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

#[cfg_attr(feature = "serializable", typetag::serde)]
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

#[cfg_attr(feature = "serializable", typetag::serde)]
impl MixtureGpSurrogate for MixintGpMixture {
    fn experts(&self) -> &Vec<Box<dyn FullGpSurrogate>> {
        self.moe.experts()
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

struct MoeVariancePredictor<'a>(&'a GpMixture);
impl<D: Data<Elem = f64>> PredictInplace<ArrayBase<D, Ix2>, Array1<f64>>
    for MoeVariancePredictor<'_>
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<f64>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let values = self
            .0
            .predict_var(x)
            .expect("MixintGpMixture variances prediction");
        *y = values;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<f64> {
        Array1::zeros(x.nrows())
    }
}

/// A factory to build consistent sampling method and surrogate regarding
/// XType specifications
pub struct MixintContext {
    /// The input specifications
    xtypes: Vec<XType>,
    /// whether data are in given in folded space (enum indexes) or not (enum masks)
    /// i.e for "blue" in ["red", "green", "blue"] either \[2\] or [0, 0, 1]
    /// For sampling data refers to DOE data. For surrogate data refers to training input data
    work_in_folded_space: bool,
}

impl MixintContext {
    /// Constructor with given `xtypes` specification
    /// where working in folded space is the default
    pub fn new(xtypes: &[XType]) -> Self {
        MixintContext {
            xtypes: xtypes.to_vec(),
            work_in_folded_space: true,
        }
    }

    /// Compute input dim once unfolded due to continupous relaxation
    pub fn get_unfolded_dim(&self) -> usize {
        compute_continuous_dim(&self.xtypes)
    }

    /// Create a mixed integer LHS
    pub fn create_lhs_sampling<F: Float>(
        &self,
        kind: LhsKind,
        seed: Option<u64>,
    ) -> MixintSampling<F, Lhs<F, Xoshiro256Plus>> {
        let lhs = seed
            .map_or(Lhs::new(&as_continuous_limits(&self.xtypes)), |seed| {
                let rng = Xoshiro256Plus::seed_from_u64(seed);
                Lhs::new(&as_continuous_limits(&self.xtypes)).with_rng(rng)
            })
            .kind(kind);
        MixintSampling {
            method: lhs,
            xtypes: self.xtypes.clone(),
            output_in_folded_space: self.work_in_folded_space,
            phantom: PhantomData,
        }
    }

    /// Create a mixed integer full factorial
    pub fn create_ffact_sampling<F: Float>(&self) -> MixintSampling<F, FullFactorial<F>> {
        MixintSampling {
            method: FullFactorial::new(&as_continuous_limits(&self.xtypes)),
            xtypes: self.xtypes.clone(),
            output_in_folded_space: self.work_in_folded_space,
            phantom: PhantomData,
        }
    }

    /// Create a mixed integer random sampling
    pub fn create_rand_sampling<F: Float>(
        &self,
        seed: Option<u64>,
    ) -> MixintSampling<F, Random<F, Xoshiro256Plus>> {
        let rand = seed.map_or(Random::new(&as_continuous_limits(&self.xtypes)), |seed| {
            let rng = Xoshiro256Plus::seed_from_u64(seed);
            Random::new(&as_continuous_limits(&self.xtypes)).with_rng(rng)
        });
        MixintSampling {
            method: rand,
            xtypes: self.xtypes.clone(),
            output_in_folded_space: self.work_in_folded_space,
            phantom: PhantomData,
        }
    }

    /// Create a mixed integer mixture of experts surrogate
    pub fn create_surrogate(
        &self,
        surrogate_builder: &MoeBuilder,
        dataset: &DatasetBase<Array2<f64>, Array1<f64>>,
    ) -> Result<MixintGpMixture> {
        let mut params = MixintGpMixtureParams::new(&self.xtypes, surrogate_builder);
        let params = params.work_in_folded_space(self.work_in_folded_space);
        params.fit(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use egobox_doe::SamplingMethod;
    use crate::CorrelationSpec;
    use linfa::Dataset;
    use ndarray::array;

    #[test]
    fn test_mixint_lhs() {
        let xtypes = vec![
            XType::Float(-10.0, 10.0),
            XType::Enum(3),
            XType::Int(-10, 10),
            XType::Ord(vec![1., 3., 5., 8.]),
        ];

        let mixi = MixintContext::new(&xtypes);
        let mixi_lhs = mixi.create_lhs_sampling(LhsKind::default(), Some(0));

        let actual = mixi_lhs.sample(10);
        let expected = array![
            [-4.049003815966328, 0.0, -1.0, 1.0],
            [-3.3764166379738008, 2.0, 10.0, 5.0],
            [4.132857767184872, 2.0, 1.0, 1.0],
            [7.302048772024065, 0.0, 4.0, 8.0],
            [-7.614543694046457, 1.0, -7.0, 5.0],
            [0.028865479407640393, 1.0, 8.0, 3.0],
            [-1.4943993567665679, 0.0, -5.0, 8.0],
            [-8.291614427265058, 0.0, 5.0, 3.0],
            [9.712890742138065, 1.0, -4.0, 5.0],
            [3.392359215362074, 0.0, -9.0, 3.0]
        ];
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_mixint_ffact() {
        let xtypes = vec![XType::Float(-10.0, 10.0), XType::Int(-10, 10)];

        let mixi = MixintContext::new(&xtypes);
        let mixi_ff = mixi.create_ffact_sampling();

        let actual = mixi_ff.sample(16);
        let expected = array![
            [-10.0, -10.0],
            [-10.0, -3.0],
            [-10.0, 3.0],
            [-10.0, 10.0],
            [-3.333333333333334, -10.0],
            [-3.333333333333334, -3.0],
            [-3.333333333333334, 3.0],
            [-3.333333333333334, 10.0],
            [3.333333333333332, -10.0],
            [3.333333333333332, -3.0],
            [3.333333333333332, 3.0],
            [3.333333333333332, 10.0],
            [10.0, -10.0],
            [10.0, -3.0],
            [10.0, 3.0],
            [10.0, 10.0]
        ];
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

    #[test]
    fn test_mixint_random() {
        let xtypes = vec![
            XType::Float(-10.0, 10.0),
            XType::Enum(3),
            XType::Int(-10, 10),
            XType::Ord(vec![1., 3., 5., 8.]),
        ];

        let mixi = MixintContext::new(&xtypes);
        let mixi_rand = mixi.create_rand_sampling(Some(0));

        let actual = mixi_rand.sample(10);
        let expected = array![
            [7.08385572734942, 1.0, -5.0, 1.0],
            [3.923592153620703, 2.0, -3.0, 3.0],
            [1.6925857875746217, 1.0, 10.0, 3.0],
            [-0.12628232178356846, 1.0, 4.0, 3.0],
            [-4.333973977708889, 2.0, -7.0, 3.0],
            [-0.31189887669548, 2.0, 2.0, 8.0],
            [5.274476356096036, 0.0, -5.0, 3.0],
            [0.21749742902273717, 2.0, 6.0, 5.0],
            [6.267468405479235, 0.0, -3.0, 8.0],
            [-5.444093848698666, 0.0, 7.0, 8.0]
        ];
        assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
    }

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
}

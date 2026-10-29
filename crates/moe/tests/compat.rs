//! Persisted model compatibility: models saved by a previous version (fixtures in
//! `tests/compat/`) must still load, in json and binary formats, and give the same
//! predictions. This guards serde field names and typetag tags of the surrogates,
//! including the polymorphic list of mixtures saved by the Egor optimizer.
//!
//! Record (or update) fixtures with
//! `EGOBOX_UPDATE_COMPAT_FIXTURES=1 cargo test --release -p egobox-moe --test compat`.
//! A fixture update means saved models of users can no longer be loaded:
//! it has to be committed separately and explicitly reviewed.

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use egobox_moe::{
    AffinedSurrogate, CorrelationSpec, GpFileFormat, GpMixture, GpMixtureParams, GpType, Inducings,
    MixintContext, MixintGpMixture, MixtureGpSurrogate, NbClusters, Recombination, RegressionSpec,
    SparseMethod, XType,
};
use linfa::prelude::{Dataset, Fit};
use ndarray::{Array1, Array2, ArrayView1, Axis, array, stack};
use ndarray_npy::{read_npy, write_npy};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::path::PathBuf;

const UPDATE_ENV: &str = "EGOBOX_UPDATE_COMPAT_FIXTURES";
const ABS_TOL: f64 = 1e-10;
const REL_TOL: f64 = 1e-8;

fn updating() -> bool {
    std::env::var(UPDATE_ENV).is_ok()
}

fn fixture(name: &str) -> String {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("compat");
    if updating() {
        std::fs::create_dir_all(&dir).expect("compat fixture dir creation");
    }
    dir.join(name).to_str().expect("fixture path").to_string()
}

fn seeded() -> Xoshiro256Plus {
    Xoshiro256Plus::seed_from_u64(42)
}

fn f2d(x: &ArrayView1<f64>) -> f64 {
    (6. * x[0]).sin() + x[1] * x[1]
}

/// Training data and test points of the continuous mixtures
fn continuous_data() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let xlimits = array![[0., 1.], [0., 1.]];
    let xt = Lhs::new(&xlimits).with_rng(seeded()).sample(20);
    let yt = xt.map_axis(Axis(1), |x| f2d(&x));
    let xtest = Lhs::new(&xlimits)
        .with_rng(Xoshiro256Plus::seed_from_u64(7))
        .sample(7);
    (xt, yt, xtest)
}

fn mixint_types() -> Vec<XType> {
    vec![XType::Float(0., 1.), XType::Int(0, 4), XType::Enum(3)]
}

/// Training data and test points (folded space) of the mixed-integer mixture
fn mixint_data() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let ctx = MixintContext::new(&mixint_types());
    let xt: Array2<f64> = ctx
        .create_lhs_sampling(LhsKind::Maximin, Some(42))
        .sample(20);
    let yt = xt.map_axis(Axis(1), |x| (6. * x[0]).sin() + 0.1 * x[1] + 0.5 * x[2]);
    let xtest: Array2<f64> = ctx.create_lhs_sampling(LhsKind::Maximin, Some(7)).sample(7);
    (xt, yt, xtest)
}

fn train_mixture(params: GpMixtureParams<f64>) -> GpMixture {
    let (xt, yt, _) = continuous_data();
    params
        .with_rng(seeded())
        .fit(&Dataset::new(xt, yt))
        .expect("mixture training")
}

fn single_cluster(regr: RegressionSpec, corr: CorrelationSpec) -> GpMixture {
    train_mixture(
        GpMixture::params()
            .regression_spec(regr)
            .correlation_spec(corr),
    )
}

fn two_clusters() -> GpMixture {
    train_mixture(
        GpMixture::params()
            .n_clusters(NbClusters::fixed(2))
            .recombination(Recombination::Hard),
    )
}

fn sparse() -> GpMixture {
    train_mixture(GpMixture::params().gp_type(GpType::SparseGp {
        sparse_method: SparseMethod::Fitc,
        inducings: Inducings::Randomized(6),
    }))
}

fn mixint() -> MixintGpMixture {
    let (xt, yt, _) = mixint_data();
    MixintContext::new(&mixint_types())
        .create_surrogate(
            &GpMixtureParams::new_with_rng(GpType::FullGp, seeded()),
            &Dataset::new(xt, yt),
        )
        .expect("mixed-integer mixture training")
}

/// Predicted values and variances as columns
fn predictions(model: &dyn MixtureGpSurrogate, x: &Array2<f64>) -> Array2<f64> {
    let (mean, var) = model.predict_valvar(&x.view()).expect("prediction");
    stack![Axis(1), mean, var]
}

fn assert_close(name: &str, actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "{name}: predictions shape");
    for ((i, j), v) in actual.indexed_iter() {
        let e = expected[[i, j]];
        assert!(
            (v - e).abs() <= ABS_TOL + REL_TOL * e.abs(),
            "{name}: prediction[[{i}, {j}]] = {v} differs from {e}"
        );
    }
}

fn formats() -> [(&'static str, GpFileFormat); 2] {
    [("json", GpFileFormat::Json), ("bin", GpFileFormat::Binary)]
}

/// Save (when updating) or load the `name` model in both formats and check its predictions
fn check_model<M: MixtureGpSurrogate>(
    name: &str,
    xtest: &Array2<f64>,
    train: impl FnOnce() -> M,
    load: impl Fn(&str, GpFileFormat) -> Box<dyn MixtureGpSurrogate>,
) {
    let reference = fixture(&format!("{name}_predictions.npy"));
    if updating() {
        let model = train();
        for (ext, format) in formats() {
            model
                .save(&fixture(&format!("{name}.{ext}")), format)
                .expect("model save");
        }
        write_npy(&reference, &predictions(&model, xtest)).expect("predictions save");
    }
    let expected: Array2<f64> = read_npy(&reference).expect("reference predictions");
    for (ext, format) in formats() {
        let model = load(&fixture(&format!("{name}.{ext}")), format);
        assert_close(
            &format!("{name}.{ext}"),
            &predictions(model.as_ref(), xtest),
            &expected,
        );
    }
}

fn load_mixture(path: &str, format: GpFileFormat) -> Box<dyn MixtureGpSurrogate> {
    GpMixture::load(path, format).expect("mixture load")
}

fn load_mixint(path: &str, format: GpFileFormat) -> Box<dyn MixtureGpSurrogate> {
    MixintGpMixture::load(path, format).expect("mixed-integer mixture load")
}

#[test]
fn compat_gp_mixture_experts() {
    let (_, _, xtest) = continuous_data();
    let specs = [
        (
            "gp_constant_squared_exponential",
            RegressionSpec::CONSTANT,
            CorrelationSpec::SQUAREDEXPONENTIAL,
        ),
        (
            "gp_linear_matern32",
            RegressionSpec::LINEAR,
            CorrelationSpec::MATERN32,
        ),
        (
            "gp_quadratic_absolute_exponential",
            RegressionSpec::QUADRATIC,
            CorrelationSpec::ABSOLUTEEXPONENTIAL,
        ),
        (
            "gp_constant_matern52",
            RegressionSpec::CONSTANT,
            CorrelationSpec::MATERN52,
        ),
    ];
    for (name, regr, corr) in specs {
        check_model(name, &xtest, || single_cluster(regr, corr), load_mixture);
    }
}

#[test]
fn compat_gp_mixture_two_clusters() {
    let (_, _, xtest) = continuous_data();
    check_model("gp_two_clusters", &xtest, two_clusters, load_mixture);
}

#[test]
fn compat_sparse_gp_mixture() {
    let (_, _, xtest) = continuous_data();
    check_model("sgp_fitc", &xtest, sparse, load_mixture);
}

#[test]
fn compat_mixint_gp_mixture() {
    let (_, _, xtest) = mixint_data();
    check_model("mixint", &xtest, mixint, load_mixint);
}

/// List of mixtures as saved by the Egor optimizer (GP models of objective and constraints),
/// including an affine transform of a mixture (derived constraint)
#[test]
fn compat_mixture_list() {
    let (_, _, xtest) = continuous_data();
    let reference = fixture("mixture_list_predictions.npy");
    let json = fixture("mixture_list.json");
    let bin = fixture("mixture_list.bin");
    if updating() {
        let gp = single_cluster(
            RegressionSpec::CONSTANT,
            CorrelationSpec::SQUAREDEXPONENTIAL,
        );
        let models: Vec<Box<dyn MixtureGpSurrogate>> = vec![
            Box::new(gp.clone()),
            Box::new(two_clusters()),
            Box::new(AffinedSurrogate::new(Box::new(gp), -1., 0.5)),
        ];
        std::fs::write(&json, serde_json::to_vec(&models).unwrap()).unwrap();
        std::fs::write(
            &bin,
            bincode::serde::encode_to_vec(&models, bincode::config::standard()).unwrap(),
        )
        .unwrap();
        let preds = models
            .iter()
            .map(|m| predictions(m.as_ref(), &xtest))
            .collect::<Vec<_>>();
        let views = preds.iter().map(|p| p.view()).collect::<Vec<_>>();
        write_npy(&reference, &ndarray::concatenate(Axis(1), &views).unwrap()).unwrap();
    }
    let expected: Array2<f64> = read_npy(&reference).expect("reference predictions");
    let from_json: Vec<Box<dyn MixtureGpSurrogate>> =
        serde_json::from_slice(&std::fs::read(&json).unwrap()).expect("json mixture list");
    let from_bin: Vec<Box<dyn MixtureGpSurrogate>> = bincode::serde::decode_from_slice(
        &std::fs::read(&bin).unwrap(),
        bincode::config::standard(),
    )
    .map(|(models, _)| models)
    .expect("binary mixture list");
    for (format, models) in [("json", from_json), ("bin", from_bin)] {
        assert_eq!(models.len(), 3, "mixture_list.{format}: number of models");
        let preds = models
            .iter()
            .map(|m| predictions(m.as_ref(), &xtest))
            .collect::<Vec<_>>();
        let views = preds.iter().map(|p| p.view()).collect::<Vec<_>>();
        let actual = ndarray::concatenate(Axis(1), &views).unwrap();
        assert_close(&format!("mixture_list.{format}"), &actual, &expected);
    }
}

//! Shared service functions used across `gpx` command workflows.
//!
//! This module centralizes model loading/selection, metrics computation,
//! and format inference helpers.

use anyhow::{Context, Result, anyhow};
use egobox_moe::{GpMetric, MixtureGpSurrogate};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use crate::{DataFormat, ModelFormat};

#[derive(Debug, Clone)]
pub struct Metrics {
    pub q2: f64,
    pub pva: f64,
    pub iae_alpha: f64,
}

pub fn compute_metrics(
    gp_models: &[Box<dyn MixtureGpSurrogate>],
    indices: &[usize],
    kfold: usize,
) -> Vec<(usize, Metrics)> {
    let mut res: Vec<_> = indices
        .par_iter()
        .enumerate()
        .map(|(pos, idx)| {
            let i = *idx;
            let gp = gp_models[i].as_ref();
            let scores: Vec<_> = [GpMetric::Q2, GpMetric::Pva, GpMetric::IAEAlphaWithPlot]
                .par_iter()
                .map(|m| {
                    let score = gp.score(*m, kfold);
                    (m, score)
                })
                .collect();
            let scores: HashMap<_, _> = scores.into_iter().collect();

            if pos == 0
                && let Some(data) = &scores.get(&GpMetric::IAEAlphaWithPlot).unwrap().plot_data
            {
                println!("\nIAEalpha plot data for selected GP model index {}:", i);
                println!("Alpha | Empirical coverage | Target coverage | Delta");
                println!("---------------------------------------------------");
                for i in 0..data.alphas.len() {
                    let alpha = data.alphas[i];
                    let delta = data.deltas[i];

                    println!(
                        "{:5.2}% |       {:5.2}%      |     {:5.2}%    | {:5.2}%",
                        alpha * 100.,
                        delta * 100.,
                        (1. - alpha) * 100.,
                        (delta - (1. - alpha)).abs() * 100.
                    );
                }
                println!();
            }

            (
                i,
                Metrics {
                    q2: scores.get(&GpMetric::Q2).unwrap().value,
                    pva: scores.get(&GpMetric::Pva).unwrap().value,
                    iae_alpha: scores.get(&GpMetric::IAEAlphaWithPlot).unwrap().value,
                },
            )
        })
        .collect();
    res.sort_by_key(|(i, _)| *i);
    res
}

/// Load the models saved in the given file, see [egobox_moe::load_models]
pub fn load_models(path: &str) -> Result<Vec<Box<dyn MixtureGpSurrogate>>> {
    egobox_moe::load_models(path).with_context(|| format!("cannot load model file {path}"))
}

pub fn select_model(
    gp_models: &[Box<dyn MixtureGpSurrogate>],
    model_index: usize,
) -> Result<&dyn MixtureGpSurrogate> {
    gp_models
        .get(model_index)
        .map(|m| m.as_ref())
        .ok_or_else(|| {
            anyhow!(
                "model index {} is out of range [0, {})",
                model_index,
                gp_models.len()
            )
        })
}

pub fn infer_fit_input_format(input: &str) -> DataFormat {
    let is_npy = Path::new(input)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("npy"));

    if is_npy {
        DataFormat::Npy
    } else {
        DataFormat::Csv
    }
}

pub fn infer_fit_model_format(output: &str) -> ModelFormat {
    let is_json = Path::new(output)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));

    if is_json {
        ModelFormat::Json
    } else {
        ModelFormat::Binary
    }
}

pub fn infer_predict_input_format(input: &str) -> DataFormat {
    let is_npy = Path::new(input)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("npy"));

    if is_npy {
        DataFormat::Npy
    } else {
        DataFormat::Csv
    }
}

pub fn infer_predict_output_format(output: &str) -> DataFormat {
    let is_npy = Path::new(output)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("npy"));

    if is_npy {
        DataFormat::Npy
    } else {
        DataFormat::Csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egobox_moe::{GpFileFormat, GpMixtureParams, GpSurrogate, MixintGpMixtureParams, XType};
    use linfa::ParamGuard;
    use linfa::traits::Fit;
    use ndarray::{Array1, array};
    use std::fs;

    #[test]
    fn test_load_mixint_model_saved_alone() {
        // Python Gpx.save() writes a single MixintGpMixture
        let xt = array![
            [0., 0.],
            [1., 1.],
            [2., 0.],
            [0., 1.],
            [1., 0.],
            [2., 1.],
            [0.5, 0.],
            [1.5, 1.]
        ];
        let yt: Array1<f64> = xt.column(0).mapv(|v| v * v) + xt.column(1);
        let xtypes = [XType::Float(0., 2.), XType::Enum(2)];
        let mut params = MixintGpMixtureParams::new(Some(&xtypes), &GpMixtureParams::new());
        params.work_in_folded_space(true);
        let gp = params
            .check()
            .unwrap()
            .fit(&linfa::Dataset::new(xt, yt))
            .unwrap();
        let x = array![[1.2, 1.], [0.3, 0.]];
        let expected = gp.predict(&x.view()).unwrap();

        let dir = std::env::temp_dir();
        for (name, format) in [
            ("gpx_test_mixint.json", GpFileFormat::Json),
            ("gpx_test_mixint.bin", GpFileFormat::Binary),
        ] {
            let path = dir.join(name);
            let path = path.to_str().unwrap();
            gp.save(path, format).unwrap();
            let models = load_models(path);
            let _ = fs::remove_file(path);
            let models = models.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(models.len(), 1);
            // JSON floats are not round-tripped exactly
            let pred = models[0].predict(&x.view()).unwrap();
            assert!((pred - &expected).abs().iter().all(|d| *d < 1e-4), "{name}");
        }
    }

    #[test]
    fn test_load_model_vector_and_single_gp_mixture() {
        let xt = array![[0.], [1.], [2.], [3.], [4.]];
        let yt = array![0., 1., 4., 9., 16.];
        let gp = GpMixtureParams::new()
            .check()
            .unwrap()
            .fit(&linfa::Dataset::new(xt, yt))
            .unwrap();
        let x = array![[0.5], [2.5]];
        let expected = gp.predict(&x.view()).unwrap();
        let models: Vec<Box<dyn MixtureGpSurrogate>> = vec![Box::new(gp.clone()), Box::new(gp)];

        let dir = std::env::temp_dir();
        let vec_bin = dir.join("gpx_test_vec.bin");
        let vec_json = dir.join("gpx_test_vec.json");
        let single_bin = dir.join("gpx_test_single.bin");
        let single_json = dir.join("gpx_test_single.json");
        fs::write(
            &vec_bin,
            bincode::serde::encode_to_vec(&models, bincode::config::standard()).unwrap(),
        )
        .unwrap();
        fs::write(&vec_json, serde_json::to_vec(&models).unwrap()).unwrap();
        models[0]
            .save(single_bin.to_str().unwrap(), GpFileFormat::Binary)
            .unwrap();
        models[0]
            .save(single_json.to_str().unwrap(), GpFileFormat::Json)
            .unwrap();

        for (path, n) in [
            (vec_bin, 2),
            (vec_json, 2),
            (single_bin, 1),
            (single_json, 1),
        ] {
            let loaded = load_models(path.to_str().unwrap());
            let _ = fs::remove_file(&path);
            let loaded = loaded.unwrap_or_else(|e| panic!("{path:?}: {e}"));
            assert_eq!(loaded.len(), n, "{path:?}");
            let pred = loaded[n - 1].predict(&x.view()).unwrap();
            // JSON floats are not round-tripped exactly
            let err = (&pred - &expected).abs();
            assert!(err.iter().all(|d| *d < 1e-4), "{path:?}: {err}");
        }
    }

    #[test]
    fn test_load_invalid_model_file_is_an_error() {
        let path = std::env::temp_dir().join("gpx_test_invalid.bin");
        fs::write(&path, b"not a model").unwrap();
        let res = load_models(path.to_str().unwrap());
        let _ = fs::remove_file(&path);
        assert!(res.is_err());
    }
}

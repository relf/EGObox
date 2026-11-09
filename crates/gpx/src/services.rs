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
    use std::fs;

    #[test]
    fn test_load_invalid_model_file_is_an_error() {
        let path = std::env::temp_dir().join("gpx_test_invalid.bin");
        fs::write(&path, b"not a model").unwrap();
        let res = load_models(path.to_str().unwrap());
        let _ = fs::remove_file(&path);
        assert!(res.is_err());
    }
}

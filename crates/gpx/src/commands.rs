//! Subcommand execution layer for `gpx`.
//!
//! Each public function in this module implements one CLI subcommand
//! workflow (`fit`, `qa`, `spec`, `predict`, `py`).

use anyhow::{Result, anyhow, bail};
use egobox_moe::{GpMixture, MixtureGpSurrogate, NbClusters, Recombination};
use linfa::Dataset;
use linfa::traits::Fit;
use ndarray::Array2;
use std::fs;

use crate::io::{read_input_data, read_training_data, write_predictions_table_data};
use crate::services::{
    compute_metrics, infer_fit_input_format, infer_fit_model_format, infer_predict_input_format,
    infer_predict_output_format, load_models, select_model,
};
use crate::{CorrelationChoice, DataFormat, ModelFormat, RecombinationChoice, RegressionChoice};

pub fn run_qa(model: &str, model_index: Option<usize>, kfold: usize) -> Result<()> {
    let gp_models = load_models(model)?;

    let selected_indices: Vec<usize> = if let Some(index) = model_index {
        vec![index]
    } else {
        (0..gp_models.len()).collect()
    };
    if selected_indices.is_empty() {
        bail!("no model selected for QA");
    }
    for &index in &selected_indices {
        let _ = select_model(&gp_models, index)?;
    }

    if let Some(index) = model_index {
        println!("Loaded GP model: {}", gp_models[index]);
    } else {
        gp_models.iter().for_each(|gp| {
            println!("Loaded GP model: {}", gp);
        });
    }

    let ref_index = selected_indices[0];
    let (xt, _yt) = gp_models[ref_index].training_data();
    println!(
        "Training data (reference model {}): {} samples ({}-dim)",
        ref_index,
        xt.nrows(),
        xt.ncols()
    );

    let k = if kfold == 0 { xt.nrows() } else { kfold };
    let res = compute_metrics(&gp_models, &selected_indices, k);

    println!(
        "\nMetric interpretation reminder (cf. Marrel2024 https://cea.hal.science/cea-04322810v2/document):"
    );
    println!("- Q2 (to maximize, <= 1):");
    println!("  high and close to 1 -> good predictive capability.");
    println!("  low (Q2 <= 0.5, e.g.) -> poor predictive capability.");
    println!("- PVA (to minimize, >= 0):");
    println!(
        "  close to 0 -> predictive variances have the right order of magnitude vs prediction errors."
    );
    println!("  high -> unreliable intervals (over- or under-confident model).");
    println!("- IAEalpha (to minimize, in [0, 0.5]):");
    println!("  close to 0 -> reliable predicted intervals only when Q2 is also high.");
    println!(
        "  close to 0.5 -> unreliable intervals; interpret jointly with Q2, PVA, and the alpha-PI plot."
    );
    println!();

    if let Some(index) = model_index {
        println!("QA mode: selected surrogate index {index}");
    } else {
        println!("QA mode: all surrogates ({} model(s))", gp_models.len());
    }

    for (i, m) in res {
        println!(
            "GP({}): Q2 = {:.2}, PVA = {:.2}, IAEalpha = {:.2}",
            i, m.q2, m.pva, m.iae_alpha
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_fit(
    input: &str,
    outputs: usize,
    regression_spec: RegressionChoice,
    correlation_spec: CorrelationChoice,
    kpls_dim: Option<usize>,
    n_clusters: usize,
    recombination: RecombinationChoice,
    smooth_factor: Option<f64>,
    output: &str,
) -> Result<()> {
    if n_clusters == 0 {
        bail!("n_clusters must be >= 1");
    }
    if matches!(recombination, RecombinationChoice::Hard) && smooth_factor.is_some() {
        bail!("smooth_factor can only be used with recombination=smooth");
    }

    let recombination = match recombination {
        RecombinationChoice::Hard => Recombination::Hard,
        RecombinationChoice::Smooth => Recombination::Smooth(smooth_factor),
    };

    let input_format = infer_fit_input_format(input);
    let format = infer_fit_model_format(output);
    let (x, y_all) = read_training_data(input, input_format, outputs)?;
    let mut gp_models: Vec<Box<dyn MixtureGpSurrogate>> = Vec::with_capacity(y_all.ncols());

    for iy in 0..y_all.ncols() {
        let y = y_all.column(iy).to_owned();
        let ds = Dataset::new(x.clone(), y);
        let gp = GpMixture::params()
            .n_clusters(NbClusters::fixed(n_clusters))
            .recombination(recombination)
            .regression_spec(regression_spec.into())
            .correlation_spec(correlation_spec.into())
            .kpls_dim(kpls_dim)
            .fit(&ds)
            .map_err(|e| anyhow!("default GP training failed for output {}: {e}", iy))?;
        gp_models.push(Box::new(gp) as Box<dyn MixtureGpSurrogate>);
    }

    match format {
        ModelFormat::Binary => {
            let bytes = bincode::serde::encode_to_vec(&gp_models, bincode::config::standard())
                .map_err(|e| anyhow!("cannot serialize trained model(s) to binary: {e}"))?;
            fs::write(output, bytes)?;
        }
        ModelFormat::Json => {
            let text = serde_json::to_string_pretty(&gp_models)
                .map_err(|e| anyhow!("cannot serialize trained model(s) to JSON: {e}"))?;
            fs::write(output, text)?;
        }
    }

    let nx = x.ncols();
    let ny = y_all.ncols();
    println!("Trained default GP surrogate model(s) from {input}");
    println!("Training samples: {}", x.nrows());
    println!("Input dimension: {nx}");
    println!("Output dimension: {ny}");
    println!("Generated surrogate models: {}", gp_models.len());
    println!("Output columns used: last {outputs}");
    println!("Number of clusters: {}", n_clusters);
    println!("Recombination: {:?}", recombination);
    println!("Smooth factor: {:?}", smooth_factor);
    println!("Regression spec: {:?}", regression_spec);
    println!("Correlation spec: {:?}", correlation_spec);
    println!("KPLS dim: {:?}", kpls_dim);
    println!("Input format: {:?}", input_format);
    println!("Output format: {:?}", format);
    println!("Saved model to {output}");

    Ok(())
}

pub fn run_spec(model: &str, model_index: Option<usize>) -> Result<()> {
    let gp_models = load_models(model)?;
    let gp0 = gp_models
        .first()
        .ok_or_else(|| anyhow!("no GP model found in file {model}"))?;
    let (nx, _ny0) = gp0.dims();
    let total_outputs = gp_models.len();

    let selected_indices: Vec<usize> = if let Some(index) = model_index {
        vec![index]
    } else {
        (0..gp_models.len()).collect()
    };
    for &index in &selected_indices {
        let _ = select_model(&gp_models, index)?;
    }
    let reference_index = selected_indices[0];
    let (xt, _yt) = gp_models[reference_index].training_data();
    let selected_count = selected_indices.len();

    println!("Model file: {model}");
    println!("Model count in file: {}", gp_models.len());
    match model_index {
        Some(index) => println!("Spec mode: selected surrogate index {index}"),
        None => println!("Spec mode: all surrogates"),
    }
    println!("Surrogate models:");
    for &i in &selected_indices {
        let surrogate = gp_models[i].as_ref();
        let (sx, sy) = surrogate.dims();
        println!("  - model[{i}]: {surrogate}");
        println!("    input dimension: {sx}");
        println!("    output dimension: {sy}");
    }
    println!("Input specification:");
    println!("  - supported formats: csv, npy");
    println!("  - csv row layout: x1,x2,...,x{nx}");
    println!("  - npy array shape: (n_samples, {nx})");
    println!("  - expected input dimension: {nx}");
    println!("Output specification:");
    println!("  - output dimension (number of surrogates): {total_outputs}");
    if selected_count == 1 {
        println!("  - predict csv columns: prediction");
        println!("  - predict --with-variance csv columns: prediction,variance");
        println!("  - predict npy shape: (n_samples, 1)");
        println!("  - predict --with-variance npy shape: (n_samples, 2)");
    } else {
        println!(
            "  - predict csv columns: y_pred1..y_pred{}, y_var1..y_var{} (with --with-variance)",
            selected_count, selected_count
        );
        println!("  - predict npy shape: (n_samples, {})", selected_count);
        println!(
            "  - predict --with-variance npy shape: (n_samples, {})",
            2 * selected_count
        );
    }
    println!("Training data summary:");
    println!("  - reference model: {reference_index}");
    println!("  - samples: {}", xt.nrows());
    println!("  - input dimension: {}", xt.ncols());

    Ok(())
}

pub fn run_predict(
    model: &str,
    input: &str,
    output: &str,
    with_variance: bool,
    model_index: Option<usize>,
) -> Result<()> {
    let input_format = infer_predict_input_format(input);
    let output_format = infer_predict_output_format(output);

    let gp_models = load_models(model)?;
    let (nx, _ny) = gp_models
        .first()
        .ok_or_else(|| anyhow!("no GP model found in file {model}"))?
        .dims();

    let x = read_input_data(input, nx, input_format)?;

    let selected_indices: Vec<usize> = if let Some(index) = model_index {
        vec![index]
    } else {
        (0..gp_models.len()).collect()
    };

    if selected_indices.is_empty() {
        bail!("no model selected for prediction");
    }

    let mut pred_flat = Vec::with_capacity(x.nrows() * selected_indices.len());
    let mut var_flat = if with_variance {
        Some(Vec::with_capacity(x.nrows() * selected_indices.len()))
    } else {
        None
    };

    for &idx in &selected_indices {
        let gp = select_model(&gp_models, idx)?;
        let (mx, _my) = gp.dims();
        if mx != nx {
            bail!(
                "model {} input dimension mismatch: expected {}, got {}",
                idx,
                nx,
                mx
            );
        }

        let pred = gp.predict(&x.view())?;
        pred_flat.extend(pred.iter().copied());

        if let Some(var_values) = var_flat.as_mut() {
            let var = gp.predict_var(&x.view())?;
            var_values.extend(var.iter().copied());
        }
    }

    let preds = Array2::from_shape_vec((selected_indices.len(), x.nrows()), pred_flat)
        .map_err(|e| anyhow!("cannot build prediction matrix: {e}"))?
        .reversed_axes()
        .to_owned();

    let variances = if let Some(values) = var_flat {
        Some(
            Array2::from_shape_vec((selected_indices.len(), x.nrows()), values)
                .map_err(|e| anyhow!("cannot build variance matrix: {e}"))?
                .reversed_axes()
                .to_owned(),
        )
    } else {
        None
    };

    write_predictions_table_data(output, output_format, &x, &preds, variances.as_ref())?;

    println!(
        "Predicted {} sample(s) from {} and wrote {}",
        x.nrows(),
        input,
        output
    );
    if let Some(index) = model_index {
        println!("Model file: {model} (index {index})");
    } else {
        println!(
            "Model file: {} (all {} models)",
            model,
            selected_indices.len()
        );
    }
    println!("Input format: {:?}", input_format);
    println!("Output format: {:?}", output_format);
    match output_format {
        DataFormat::Csv => {
            if with_variance {
                println!(
                    "Output columns: x1..x{}, y_pred1..y_pred{}, y_var1..y_var{}",
                    x.ncols(),
                    preds.ncols(),
                    preds.ncols()
                );
            } else {
                println!(
                    "Output columns: x1..x{}, y_pred1..y_pred{}",
                    x.ncols(),
                    preds.ncols()
                );
            }
        }
        DataFormat::Npy => {
            let output_cols = if with_variance {
                x.ncols() + 2 * preds.ncols()
            } else {
                x.ncols() + preds.ncols()
            };
            if with_variance {
                println!(
                    "Output array shape: (n_samples, {}) [inputs, predictions, variances]",
                    output_cols
                );
            } else {
                println!(
                    "Output array shape: (n_samples, {}) [inputs, predictions]",
                    output_cols
                );
            }
        }
    }

    Ok(())
}

pub fn run_py(model: &str, output: &str) -> Result<()> {
    let gp_models = load_models(model)?;
    let first = gp_models
        .first()
        .ok_or_else(|| anyhow!("no GP model found in file {model}"))?;
    let (input_dim, _first_output_dim) = first.dims();
    for (idx, gp) in gp_models.iter().enumerate() {
        let (mx, _my) = gp.dims();
        if mx != input_dim {
            bail!(
                "model {} input dimension mismatch: expected {}, got {}",
                idx,
                input_dim,
                mx
            );
        }
    }
    let output_dim_all = gp_models.len();

    let model_bytes =
        fs::read(model).map_err(|e| anyhow!("cannot read model file {model}: {e}"))?;
    if model_bytes.is_empty() {
        bail!("model file {model} is empty");
    }

    let model_hex = to_hex_lines(&model_bytes, 120);
    let script = render_python_script(model, &model_hex, input_dim, output_dim_all);
    fs::write(output, script)
        .map_err(|e| anyhow!("cannot write generated Python file {output}: {e}"))?;

    println!("Generated Python helper script: {output}");
    println!("Embedded model source: {model}");
    println!("Embedded bytes: {}", model_bytes.len());
    println!("Input dimension: {}", input_dim);
    println!("Output dimension (all models): {}", output_dim_all);

    Ok(())
}

fn to_hex_lines(bytes: &[u8], chars_per_line: usize) -> String {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let mut out = String::new();
    let mut start = 0;
    while start < hex.len() {
        let end = (start + chars_per_line).min(hex.len());
        out.push_str("    ");
        out.push('"');
        out.push_str(&hex[start..end]);
        out.push('"');
        out.push('\n');
        start = end;
    }
    out
}

fn render_python_script(
    source_model_path: &str,
    model_hex: &str,
    input_dim: usize,
    output_dim_all: usize,
) -> String {
    format!(
        r#"#!/usr/bin/env python3
"""Generated by `gpx py`.

This module embeds a GP model and provides:
    predict(x, with_variance=False, model_index=None)

Model-specific shape contract:
- input shape: (n_samples, {input_dim}) or ({input_dim},)
- output shape when model_index is None: (n_samples, {output_dim_all})
- output shape when model_index is set: (n_samples, 1)

Requirements:
- Python 3.8+
- numpy
- `gpx` executable available in PATH
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

INPUT_DIM = {input_dim}
OUTPUT_DIM_ALL = {output_dim_all}
OUTPUT_DIM_SINGLE = 1


def _materialize_model(path: Path) -> None:
    path.write_bytes(bytes.fromhex(_MODEL_HEX))


def _as_matrix(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("x must be 1D or 2D and convertible to a float matrix")
    return arr


def _write_csv(path: Path, x: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(x.tolist())


def _read_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if row]
    if not rows:
        raise ValueError("prediction output CSV is empty")
    return np.array([[float(v) for v in row] for row in rows], dtype=float)


def predict(x, with_variance: bool = False, model_index: Optional[int] = None):
    """Predict values using the embedded GP model.

    Parameters
    ----------
    x:
        Input samples as array-like with shape (n_samples, INPUT_DIM)
        or a single sample with shape (INPUT_DIM,).
    with_variance:
        If True, also return predictive variances.
    model_index:
        Optional surrogate index passed to `gpx predict --model-index`.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        If model_index is None, predictions have shape (n_samples, OUTPUT_DIM_ALL).
        Otherwise predictions have shape (n_samples, OUTPUT_DIM_SINGLE).
        Variances (when requested) always have the same shape as predictions.
    """
    x_mat = _as_matrix(x)
    n_features = x_mat.shape[1]
    if n_features != INPUT_DIM:
        raise ValueError(
            f"invalid input shape {{x_mat.shape}}: expected second axis = {{INPUT_DIM}}"
        )

    expected_output_dim = OUTPUT_DIM_SINGLE if model_index is not None else OUTPUT_DIM_ALL

    with tempfile.TemporaryDirectory(prefix="gpx_py_") as tmpdir:
        tmp = Path(tmpdir)
        model_path = tmp / "model.gpx"
        input_path = tmp / "input.csv"
        output_path = tmp / "output.csv"

        _materialize_model(model_path)
        _write_csv(input_path, x_mat)

        cmd = [
            "gpx",
            "predict",
            str(input_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
        ]
        if with_variance:
            cmd.append("--with-variance")
        if model_index is not None:
            cmd.extend(["--model-index", str(model_index)])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("gpx executable not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            stdout = exc.stdout.strip() if exc.stdout else ""
            details = stderr or stdout or str(exc)
            raise RuntimeError(f"gpx predict failed: {{details}}") from exc

        table = _read_csv(output_path)

    if table.shape[1] <= n_features:
        raise RuntimeError("unexpected gpx output: no prediction columns found")

    trailing = table[:, n_features:]
    expected_cols = 2 * expected_output_dim if with_variance else expected_output_dim
    if trailing.shape[1] != expected_cols:
        raise RuntimeError(
            f"unexpected gpx output shape {{trailing.shape}}: expected (n_samples, {{expected_cols}})"
        )

    if with_variance:
        split = expected_output_dim
        preds = trailing[:, :split]
        variances = trailing[:, split:]
        return preds, variances

    return trailing


if __name__ == "__main__":
    print(
        "This module was generated from model: {source_model_path}. "
        "Import it and call predict(x, with_variance=False, model_index=None)."
    )


_MODEL_HEX = (
{model_hex})
"#
    )
}

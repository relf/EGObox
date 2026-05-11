//! Tabular data I/O helpers for `gpx`.
//!
//! Provides CSV/NPY readers and writers for training, prediction inputs,
//! and prediction outputs.

use anyhow::{Context, Result, anyhow, bail};
use ndarray::Array2;
use ndarray_npy::{read_npy, write_npy};
use std::fs;

use crate::DataFormat;

fn parse_csv_row(line: &str) -> Result<Vec<f64>> {
    line.split(',')
        .map(str::trim)
        .map(|cell| {
            cell.parse::<f64>()
                .map_err(|e| anyhow!("invalid float value '{cell}': {e}"))
        })
        .collect()
}

fn is_header_row(line: &str) -> bool {
    let cells: Vec<_> = line.split(',').map(str::trim).collect();
    !cells.is_empty() && cells.iter().all(|cell| cell.parse::<f64>().is_err())
}

fn read_input_csv(path: &str, nx: usize) -> Result<Array2<f64>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("cannot read input CSV {path}"))?;

    let mut nrows = 0usize;
    let mut flat_values = Vec::new();
    let mut first_non_empty = true;

    for (line_index, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let row = match parse_csv_row(line) {
            Ok(row) => row,
            Err(_e) if first_non_empty && is_header_row(line) => {
                first_non_empty = false;
                continue;
            }
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("invalid CSV row at line {}", line_index + 1));
            }
        };
        first_non_empty = false;

        if row.len() != nx {
            bail!(
                "invalid input dimension at line {}: expected {} values, got {}",
                line_index + 1,
                nx,
                row.len()
            );
        }

        flat_values.extend(row);
        nrows += 1;
    }

    if nrows == 0 {
        bail!("input CSV {path} does not contain any sample row");
    }

    Array2::from_shape_vec((nrows, nx), flat_values)
        .map_err(|e| anyhow!("cannot build input matrix from CSV {path}: {e}"))
}

fn read_input_npy(path: &str, nx: usize) -> Result<Array2<f64>> {
    let x: Array2<f64> = read_npy(path).with_context(|| format!("cannot read input NPY {path}"))?;

    if x.ncols() != nx {
        bail!(
            "invalid input dimension in NPY {}: expected {} columns, got {}",
            path,
            nx,
            x.ncols()
        );
    }

    if x.nrows() == 0 {
        bail!("input NPY {path} does not contain any sample row");
    }

    Ok(x)
}

pub fn read_input_data(path: &str, nx: usize, format: DataFormat) -> Result<Array2<f64>> {
    match format {
        DataFormat::Csv => read_input_csv(path, nx),
        DataFormat::Npy => read_input_npy(path, nx),
    }
}

fn read_training_csv(path: &str, n_outputs: usize) -> Result<(Array2<f64>, Array2<f64>)> {
    if n_outputs == 0 {
        bail!("number of outputs must be >= 1");
    }

    let content =
        fs::read_to_string(path).with_context(|| format!("cannot read training CSV {path}"))?;

    let mut nrows = 0usize;
    let mut nx = 0usize;
    let mut x_flat_values = Vec::new();
    let mut y_flat_values = Vec::new();
    let mut ncols_expected = 0usize;
    let mut first_non_empty = true;

    for (line_index, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let row = match parse_csv_row(line) {
            Ok(row) => row,
            Err(_e) if first_non_empty && is_header_row(line) => {
                first_non_empty = false;
                continue;
            }
            Err(e) => {
                return Err(e).with_context(|| {
                    format!("invalid training CSV row at line {}", line_index + 1)
                });
            }
        };
        first_non_empty = false;

        if row.len() <= n_outputs {
            bail!(
                "invalid training row at line {}: expected at least {} columns (features + {} output(s)), got {}",
                line_index + 1,
                n_outputs + 1,
                n_outputs,
                row.len()
            );
        }

        if ncols_expected == 0 {
            ncols_expected = row.len();
            nx = ncols_expected - n_outputs;
        } else if row.len() != ncols_expected {
            bail!(
                "inconsistent training row width at line {}: expected {} columns, got {}",
                line_index + 1,
                ncols_expected,
                row.len()
            );
        }

        let y_start = row.len() - n_outputs;
        x_flat_values.extend_from_slice(&row[..y_start]);
        y_flat_values.extend_from_slice(&row[y_start..]);
        nrows += 1;
    }

    if nrows == 0 {
        bail!("training CSV {path} does not contain any sample row");
    }

    let x = Array2::from_shape_vec((nrows, nx), x_flat_values)
        .map_err(|e| anyhow!("cannot build training input matrix from CSV {path}: {e}"))?;
    let y = Array2::from_shape_vec((nrows, n_outputs), y_flat_values)
        .map_err(|e| anyhow!("cannot build training output matrix from CSV {path}: {e}"))?;
    Ok((x, y))
}

fn read_training_npy(path: &str, n_outputs: usize) -> Result<(Array2<f64>, Array2<f64>)> {
    if n_outputs == 0 {
        bail!("number of outputs must be >= 1");
    }

    let xy: Array2<f64> =
        read_npy(path).with_context(|| format!("cannot read training NPY {path}"))?;

    if xy.ncols() <= n_outputs {
        bail!(
            "invalid training NPY {}: expected at least {} columns (features + {} output(s)), got {}",
            path,
            n_outputs + 1,
            n_outputs,
            xy.ncols()
        );
    }

    if xy.nrows() == 0 {
        bail!("training NPY {path} does not contain any sample row");
    }

    let nx = xy.ncols() - n_outputs;
    let x = xy.slice(ndarray::s![.., ..nx]).to_owned();
    let y = xy.slice(ndarray::s![.., nx..]).to_owned();
    Ok((x, y))
}

pub fn read_training_data(
    path: &str,
    format: DataFormat,
    n_outputs: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    match format {
        DataFormat::Csv => read_training_csv(path, n_outputs),
        DataFormat::Npy => read_training_npy(path, n_outputs),
    }
}

fn write_predictions_table_csv(
    path: &str,
    x: &Array2<f64>,
    preds: &Array2<f64>,
    variances: Option<&Array2<f64>>,
) -> Result<()> {
    if x.nrows() != preds.nrows() {
        bail!(
            "row count mismatch between inputs ({}) and predictions ({})",
            x.nrows(),
            preds.nrows()
        );
    }

    if let Some(var) = variances
        && (var.nrows() != preds.nrows() || var.ncols() != preds.ncols())
    {
        bail!(
            "variance matrix shape mismatch: expected ({}, {}), got ({}, {})",
            preds.nrows(),
            preds.ncols(),
            var.nrows(),
            var.ncols()
        );
    }

    let mut out = String::new();
    for i in 0..x.nrows() {
        let mut values = Vec::with_capacity(x.ncols() + preds.ncols());
        for j in 0..x.ncols() {
            values.push(format!("{:.16e}", x[(i, j)]));
        }
        for j in 0..preds.ncols() {
            values.push(format!("{:.16e}", preds[(i, j)]));
        }
        if let Some(var) = variances {
            for j in 0..var.ncols() {
                values.push(format!("{:.16e}", var[(i, j)]));
            }
        }
        out.push_str(&values.join(","));
        out.push('\n');
    }

    fs::write(path, out).with_context(|| format!("cannot write output CSV {path}"))
}

fn write_predictions_table_npy(
    path: &str,
    x: &Array2<f64>,
    preds: &Array2<f64>,
    variances: Option<&Array2<f64>>,
) -> Result<()> {
    if x.nrows() != preds.nrows() {
        bail!(
            "row count mismatch between inputs ({}) and predictions ({})",
            x.nrows(),
            preds.nrows()
        );
    }

    if let Some(var) = variances
        && (var.nrows() != preds.nrows() || var.ncols() != preds.ncols())
    {
        bail!(
            "variance matrix shape mismatch: expected ({}, {}), got ({}, {})",
            preds.nrows(),
            preds.ncols(),
            var.nrows(),
            var.ncols()
        );
    }

    let out_ncols = x.ncols() + preds.ncols() + variances.map_or(0, |v| v.ncols());
    let mut flat = Vec::with_capacity(x.nrows() * out_ncols);
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            flat.push(x[(i, j)]);
        }
        for j in 0..preds.ncols() {
            flat.push(preds[(i, j)]);
        }
        if let Some(var) = variances {
            for j in 0..var.ncols() {
                flat.push(var[(i, j)]);
            }
        }
    }

    let out = Array2::from_shape_vec((x.nrows(), out_ncols), flat)
        .map_err(|e| anyhow!("cannot build prediction output matrix: {e}"))?;
    write_npy(path, &out).with_context(|| format!("cannot write output NPY {path}"))
}

pub fn write_predictions_table_data(
    path: &str,
    format: DataFormat,
    x: &Array2<f64>,
    preds: &Array2<f64>,
    variances: Option<&Array2<f64>>,
) -> Result<()> {
    match format {
        DataFormat::Csv => write_predictions_table_csv(path, x, preds, variances),
        DataFormat::Npy => write_predictions_table_npy(path, x, preds, variances),
    }
}

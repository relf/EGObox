---
name: egobox-gpx-cli
description: >
  Use this skill whenever the user is working with the `gpx` command-line tool from the
  EGObox project. Triggers on any mention of: the gpx CLI, `gpx fit`, `gpx predict`,
  `gpx qa`, `gpx spec`, `gpx py`, training a surrogate model from CSV or npy files on
  the command line, assessing GP model quality from the terminal, generating a Python
  helper from a saved .gpx/.bin/.json model, or using gpx to run inference on tabular data
  without writing Python code. Also use when the user wants to inspect or use a .gpx,
  .bin, or .json model file saved by the Egor optimizer or the Python Gpx.save().
---

# EGObox `gpx` CLI Skill

`gpx` is a standalone command-line tool bundled with the `egobox` Python package (≥ 0.37.6).
It lets you fit, inspect, evaluate and run GP surrogate models entirely from the shell —
no Python code required.

## Installation

`gpx` ships with the Python package:

```bash
pip install egobox          # gpx binary lands in PATH automatically
gpx --version               # gpx 0.5.0
```

Alternatively, install the standalone binary directly:

```bash
# macOS (Homebrew)
brew install relf/tap/gpx

# Linux / macOS (shell installer)
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/relf/egobox/releases/download/gpx-0.4.0/gpx-installer.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c \
  "irm https://github.com/relf/egobox/releases/download/gpx-0.4.0/gpx-installer.ps1 | iex"
```

---

## Subcommands Overview

| Subcommand | Purpose |
|---|---|
| `fit` | Train a GP surrogate from a CSV or npy data file |
| `spec` | Inspect a saved model's input/output dimensions and configuration |
| `qa` | Assess predictive quality (Q2, PVA, IAEalpha) via LOO or k-fold CV |
| `predict` | Run inference on new input samples; writes CSV or npy |
| `py` | Generate a self-contained Python helper script with the model embedded |

---

## Data File Formats

`gpx` accepts two tabular formats, auto-detected by file extension:

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | One row per sample, comma-separated, optional header row (skipped) |
| NumPy | `.npy` | 2D float64 array saved with `np.save()` |

**Training file layout** (`fit`): columns are `[x1, x2, ..., xn, y1, y2, ..., ym]`.
By default the **last 1 column** is treated as output; use `--outputs N` for multi-output.

**Predict input file**: columns are just inputs `[x1, x2, ..., xn]`.

---

## `gpx fit` — Train a Surrogate

```
gpx fit [OPTIONS] <INPUT>
```

**Arguments:**
- `<INPUT>` — training data file (CSV or npy)

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--outputs N` | `1` | Number of output columns taken from the end of each row |
| `--regression-spec` | `constant` | `constant`, `linear`, `quadratic`, `all` |
| `--correlation-spec` | `squared-exponential` | `squared-exponential`, `absolute-exponential`, `matern32`, `matern52`, `all` |
| `--kpls-dim N` | (none) | PLS dimension reduction to N components; recommended when input dim ≥ 9 |
| `--n-clusters N` | `1` | Number of local expert clusters (mixture of GPs) |
| `--recombination` | `smooth` | `smooth` or `hard` — how experts are combined |
| `-o, --output FILE` | `surrogate_model.gpx` | Output model file path |

**Examples:**

```bash
# Minimal — single output, default settings
gpx fit data.csv

# Multi-output with Matérn 5/2 kernel
gpx fit data.csv --outputs 2 --correlation-spec matern52 -o model.gpx

# High-dimensional input with PLS reduction
gpx fit data.csv --kpls-dim 3 -o model_kpls.gpx

# Try all regression × correlation combinations (cross-validation picks best)
gpx fit data.csv --regression-spec all --correlation-spec all -o model_best.gpx

# Mixture of 3 local experts
gpx fit data.csv --n-clusters 3 --recombination smooth -o model_moe.gpx
```

---

## `gpx spec` — Inspect a Model

```
gpx spec [--model FILE] [-m INDEX]
```

Shows: surrogate type, input/output dimensions, supported I/O formats, expected array shapes, training data summary.

```bash
gpx spec                              # inspect surrogate_model.gpx
gpx spec --model my_model.gpx
gpx spec --model model.gpx -m 0      # single surrogate in a multi-output file
```

**Model file sources** — `gpx` can read models saved by:
- `gpx fit` → `.gpx` binary
- Python `egx.Gpx.save("model.json")` → `.json`
- Egor optimizer `outdir` → `egor_gp.bin`, `egor_initial_gp.bin`

---

## `gpx qa` — Quality Assessment

```
gpx qa [--model FILE] [-m INDEX] [-k KFOLD]
```

Reports three metrics (computed on training data via leave-one-out by default, or k-fold):

| Metric | Goal | Meaning |
|---|---|---|
| **Q2** | Maximise, close to 1 | Predictive accuracy (like R²). Q2 ≤ 0.5 is poor. |
| **PVA** | Minimise, close to 0 | Predictive Variance Adequacy — are uncertainty estimates well-calibrated? |
| **IAEalpha** | Minimise, in [0, 0.5] | Interval reliability — trustworthiness of prediction intervals |

Also prints an alpha/empirical coverage table (the α-PI plot).

```bash
gpx qa                          # LOO cross-validation on surrogate_model.gpx
gpx qa --model model.gpx
gpx qa --model model.gpx -k 5  # 5-fold cross-validation (better for larger datasets)
gpx qa --model model.gpx -m 0  # assess only surrogate 0 of a multi-output model
```

---

## `gpx predict` — Run Inference

```
gpx predict [OPTIONS] <INPUT>
```

**Arguments:**
- `<INPUT>` — input samples file (CSV or npy, shape `(n_samples, input_dim)`)

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model FILE` | `surrogate_model.gpx` | Model file |
| `-o, --output FILE` | `surrogate_predictions.csv` | Output file; extension sets format (`.csv` or `.npy`) |
| `--with-variance` | off | Also output predictive variance alongside the mean |
| `-m, --model-index N` | (all) | Predict only surrogate N from a multi-output model |

**Output layout:**

| Format | Columns (CSV) / Shape (npy) |
|---|---|
| CSV, no variance | `x1,...,xn, y_pred1,...,y_predm` |
| CSV, with variance | `x1,...,xn, y_pred1,...,y_predm, y_var1,...,y_varm` |
| npy, no variance | `(n_samples, n_outputs)` |
| npy, with variance | `(n_samples, 2 * n_outputs)` — predictions then variances interleaved |

```bash
# Predict from CSV, output CSV
gpx predict xtest.csv --model model.gpx -o predictions.csv

# Include predictive variance
gpx predict xtest.csv --model model.gpx --with-variance -o predictions.csv

# NumPy in, NumPy out
gpx predict xtest.npy --model model.gpx -o predictions.npy

# Single output from a 2-output model
gpx predict xtest.csv --model model2.gpx -m 1 -o pred_y2.csv
```

---

## `gpx py` — Generate a Python Helper Script

Embeds the model as a hex-encoded string and generates a pure-Python `predict()` function.
The generated script calls the `gpx` binary under the hood, so `gpx` must be in PATH.

```
gpx py [--model FILE] [-o OUTPUT_SCRIPT]
```

```bash
gpx py --model model.gpx -o surrogate.py   # generates surrogate.py
```

The generated script exposes:

```python
predict(x, with_variance=False, model_index=None)
# x: np.ndarray of shape (n_samples, input_dim) or (input_dim,)
# returns: np.ndarray of shape (n_samples, n_outputs)
```

Import and use it like any Python function — no need to manage model files separately:

```python
import numpy as np
from surrogate import predict

xtest = np.array([[0.5], [1.5], [2.5]])
y = predict(xtest)
y_mean, y_var = predict(xtest, with_variance=True)
```

---

## Typical Workflows

### Train → Inspect → Predict

```bash
# 1. Prepare training data as CSV: [inputs | outputs]
#    e.g. data.csv: x1,x2,y

# 2. Fit
gpx fit data.csv --correlation-spec matern52 -o model.gpx

# 3. Check model spec
gpx spec --model model.gpx

# 4. Assess quality
gpx qa --model model.gpx -k 5

# 5. Predict new points
gpx predict xtest.csv --model model.gpx --with-variance -o results.csv
```

### Inspect a Model from Egor Optimizer

When `Egor` is run with `outdir`, it saves GP models alongside the DOE:

```bash
# After: egx.Egor(..., outdir="./run").minimize(f, ...)
gpx spec --model ./run/egor_gp.bin
gpx predict xtest.csv --model ./run/egor_gp.bin -o predictions.csv
```

### Export Surrogate for Deployment

```bash
gpx fit data.csv -o surrogate.gpx
gpx py --model surrogate.gpx -o surrogate.py
# ship surrogate.py + gpx binary; call predict() directly in your app
```

---

## Common Pitfalls

- **Header row in CSV**: `gpx` skips the first row if it contains non-numeric data (auto-detected). No explicit flag needed.
- **Column order matters**: outputs must be the **last** N columns. `--outputs N` counts from the end.
- **npy files must be 2D**: use `np.save()` with an array of shape `(n_samples, n_dims)`, not 1D.
- **Model file extension doesn't matter**: `gpx` detects format by content, not extension (`.gpx`, `.bin`, `.json` all work).
- **`gpx py` needs `gpx` in PATH at runtime**: the generated script shells out to the `gpx` binary.

---

## Quick Reference

```bash
gpx fit data.csv [-o model.gpx] [--outputs N] [--kpls-dim N] [--correlation-spec matern52]
gpx spec [--model model.gpx] [-m INDEX]
gpx qa [--model model.gpx] [-k KFOLD] [-m INDEX]
gpx predict xtest.csv [--model model.gpx] [-o out.csv] [--with-variance] [-m INDEX]
gpx py [--model model.gpx] [-o script.py]
```

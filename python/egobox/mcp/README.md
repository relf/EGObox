# egobox MCP Server

This MCP (Model Context Protocol) server exposes the GPX CLI commands as tools for AI agents like Cline.

## Installation

1. Ensure egobox is installed:
   ```bash
   pip install egobox
   ```

2. Install the MCP Python package dependency:
   ```bash
   pip install mcp
   ```

## Configuration

Add the following to your MCP client configuration (e.g., Cline settings):

```json
{
  "mcpServers": {
    "egobox-gpx": {
      "command": "python",
      "args": ["-m", "egobox.mcp.server"],
      "cwd": "${workspaceFolder}/python"
    }
  }
}
```

## Available Tools

### gpx_fit

Fit GP surrogates from tabular data.

**Parameters:**
- `input_file` (required): Path to training data file (CSV or NPY format)
- `output_model`: Path for output model file (default: "surrogate_model.gpx")
- `outputs`: Number of output columns (default: 1)
- `regression_spec`: Regression model choice (constant, linear, quadratic, all)
- `correlation_spec`: Correlation model choice (squared_exponential, absolute_exponential, matern32, matern52, all)
- `kpls_dim`: Number of PLS components for KPLS dimension reduction
- `n_clusters`: Number of clusters/experts (default: 1)
- `recombination`: Recombination mode (hard or smooth)
- `smooth_factor`: Smooth recombination factor

**Example:**
```
gpx_fit(input_file="training_data.csv", output_model="my_model.gpx", n_clusters=3)
```

### gpx_qa

Assess quality metrics of GP model(s) using cross-validation.

**Parameters:**
- `model_file` (required): Path to GP model file
- `model_index`: Optional model index to assess single surrogate
- `kfold`: Number of folds for K-fold cross-validation (0 for LOO)

**Example:**
```
gpx_qa(model_file="surrogate_model.gpx", kfold=5)
```

### gpx_spec

Display GP model input/output specifications.

**Parameters:**
- `model_file` (required): Path to GP model file
- `model_index`: Optional model index to inspect single surrogate

**Example:**
```
gpx_spec(model_file="surrogate_model.gpx")
```

### gpx_predict

Predict outputs for tabular input samples using a trained GP model.

**Parameters:**
- `model_file` (required): Path to GP model file
- `input_file` (required): Path to input samples file (CSV or NPY format)
- `output_file`: Path for output predictions file (default: "surrogate_predictions.csv")
- `with_variance`: Include predictive variance in output (default: false)
- `model_index`: Optional model index for single surrogate prediction

**Example:**
```
gpx_predict(model_file="surrogate_model.gpx", input_file="test_data.csv", with_variance=true)
```

### gpx_py_generate

Generate a Python helper script with embedded GP model.

**Parameters:**
- `model_file` (required): Path to GP model file
- `output_script`: Path for output Python script (default: "gpx.py")

**Example:**
```
gpx_py_generate(model_file="surrogate_model.gpx", output_script="my_predictor.py")
```

## Usage with Cline

Once configured, you can ask Cline to:
- Train GP models from your data
- Assess model quality
- Make predictions with trained models
- Generate standalone Python prediction scripts

Example prompt:
> "Train a GP model on training_data.csv and save it as model.gpx"

Cline will use the `gpx_fit` tool to execute this task.
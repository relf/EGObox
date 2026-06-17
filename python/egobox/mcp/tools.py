"""MCP tools for egobox GPX CLI."""

from __future__ import annotations

import subprocess
import sys
from typing import Any


def _run_gpx_command(args: list[str]) -> tuple[str, str, int]:
    """Run a gpx CLI command and return stdout, stderr, and return code."""
    cmd = [sys.executable, "-m", "egobox.gpx_cli"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout, result.stderr, result.returncode


def gpx_fit(
    input_file: str,
    output_model: str = "surrogate_model.gpx",
    outputs: int = 1,
    regression_spec: str = "constant",
    correlation_spec: str = "squared_exponential",
    kpls_dim: int | None = None,
    n_clusters: int = 1,
    recombination: str = "smooth",
    smooth_factor: float | None = None,
) -> dict[str, Any]:
    """Fit GP surrogates from tabular data."""
    args = [
        "fit", input_file,
        "--outputs", str(outputs),
        "--regression-spec", regression_spec,
        "--correlation-spec", correlation_spec,
        "--n-clusters", str(n_clusters),
        "--recombination", recombination,
        "--output", output_model,
    ]
    if kpls_dim is not None:
        args.extend(["--kpls-dim", str(kpls_dim)])
    if smooth_factor is not None:
        args.extend(["--smooth-factor", str(smooth_factor)])
    stdout, stderr, returncode = _run_gpx_command(args)
    if returncode != 0:
        return {"success": False, "error": stderr or "Unknown error", "stdout": stdout}
    return {"success": True, "message": stdout, "model_path": output_model}


def gpx_qa(model_file: str, model_index: int | None = None, kfold: int = 0) -> dict[str, Any]:
    """Assess quality metrics of GP model(s)."""
    args = ["qa", "--model", model_file]
    if model_index is not None:
        args.extend(["--model-index", str(model_index)])
    if kfold != 0:
        args.extend(["--kfold", str(kfold)])
    stdout, stderr, returncode = _run_gpx_command(args)
    if returncode != 0:
        return {"success": False, "error": stderr or "Unknown error", "stdout": stdout}
    return {"success": True, "message": stdout}


def gpx_spec(model_file: str, model_index: int | None = None) -> dict[str, Any]:
    """Display GP model input/output specifications."""
    args = ["spec", "--model", model_file]
    if model_index is not None:
        args.extend(["--model-index", str(model_index)])
    stdout, stderr, returncode = _run_gpx_command(args)
    if returncode != 0:
        return {"success": False, "error": stderr or "Unknown error", "stdout": stdout}
    return {"success": True, "message": stdout}


def gpx_predict(
    model_file: str,
    input_file: str,
    output_file: str = "surrogate_predictions.csv",
    with_variance: bool = False,
    model_index: int | None = None,
) -> dict[str, Any]:
    """Predict outputs for tabular input samples."""
    args = ["predict", input_file, "--model", model_file, "--output", output_file]
    if with_variance:
        args.append("--with-variance")
    if model_index is not None:
        args.extend(["--model-index", str(model_index)])
    stdout, stderr, returncode = _run_gpx_command(args)
    if returncode != 0:
        return {"success": False, "error": stderr or "Unknown error", "stdout": stdout}
    return {"success": True, "message": stdout, "output_path": output_file}


def gpx_py_generate(model_file: str, output_script: str = "gpx.py") -> dict[str, Any]:
    """Generate a Python helper script with embedded GP model."""
    args = ["py", "--model", model_file, "--output", output_script]
    stdout, stderr, returncode = _run_gpx_command(args)
    if returncode != 0:
        return {"success": False, "error": stderr or "Unknown error", "stdout": stdout}
    return {"success": True, "message": stdout, "script_path": output_script}
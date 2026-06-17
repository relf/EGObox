"""MCP server for egobox GPX CLI.

This server exposes GPX CLI subcommands as MCP tools:
- gpx_fit: Train GP surrogates from CSV/NPY data
- gpx_qa: Assess model quality (Q2, PVA, IAEalpha)
- gpx_spec: Display model specifications
- gpx_predict: Predict outputs with optional variance
- gpx_py_generate: Generate Python helper script
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from . import tools


def create_tools() -> list[Tool]:
    """Create MCP tool definitions for GPX CLI commands."""
    return [
        Tool(
            name="gpx_fit",
            description="Fit GP surrogates from tabular data (CSV or NPY format).",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to training data file (CSV or NPY format)",
                    },
                    "output_model": {
                        "type": "string",
                        "description": "Path for output model file",
                        "default": "surrogate_model.gpx",
                    },
                    "outputs": {
                        "type": "integer",
                        "description": "Number of output columns",
                        "default": 1,
                    },
                    "regression_spec": {
                        "type": "string",
                        "description": "Regression model choice",
                        "enum": ["constant", "linear", "quadratic", "all"],
                        "default": "constant",
                    },
                    "correlation_spec": {
                        "type": "string",
                        "description": "Correlation model choice",
                        "enum": [
                            "squared_exponential",
                            "absolute_exponential",
                            "matern32",
                            "matern52",
                            "all",
                        ],
                        "default": "squared_exponential",
                    },
                    "kpls_dim": {
                        "type": "integer",
                        "description": "Number of PLS components for KPLS dimension reduction",
                        "default": None,
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters/experts (must be >= 1)",
                        "default": 1,
                    },
                    "recombination": {
                        "type": "string",
                        "description": "Recombination mode",
                        "enum": ["hard", "smooth"],
                        "default": "smooth",
                    },
                    "smooth_factor": {
                        "type": "number",
                        "description": "Smooth recombination factor (only used with recombination=smooth)",
                        "default": None,
                    },
                },
                "required": ["input_file"],
            },
        ),
        Tool(
            name="gpx_qa",
            description="Assess quality metrics of GP model(s) using cross-validation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_file": {
                        "type": "string",
                        "description": "Path to GP model file",
                    },
                    "model_index": {
                        "type": "integer",
                        "description": "Optional model index to assess single surrogate",
                        "default": None,
                    },
                    "kfold": {
                        "type": "integer",
                        "description": "Number of folds for K-fold cross-validation (0 for LOO)",
                        "default": 0,
                    },
                },
                "required": ["model_file"],
            },
        ),
        Tool(
            name="gpx_spec",
            description="Display GP model input/output specifications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_file": {
                        "type": "string",
                        "description": "Path to GP model file",
                    },
                    "model_index": {
                        "type": "integer",
                        "description": "Optional model index to inspect single surrogate",
                        "default": None,
                    },
                },
                "required": ["model_file"],
            },
        ),
        Tool(
            name="gpx_predict",
            description="Predict outputs for tabular input samples using a trained GP model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_file": {
                        "type": "string",
                        "description": "Path to GP model file",
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Path to input samples file (CSV or NPY format)",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path for output predictions file",
                        "default": "surrogate_predictions.csv",
                    },
                    "with_variance": {
                        "type": "boolean",
                        "description": "Include predictive variance in output",
                        "default": False,
                    },
                    "model_index": {
                        "type": "integer",
                        "description": "Optional model index for single surrogate prediction",
                        "default": None,
                    },
                },
                "required": ["model_file", "input_file"],
            },
        ),
        Tool(
            name="gpx_py_generate",
            description="Generate a Python helper script with embedded GP model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_file": {
                        "type": "string",
                        "description": "Path to GP model file",
                    },
                    "output_script": {
                        "type": "string",
                        "description": "Path for output Python script",
                        "default": "gpx.py",
                    },
                },
                "required": ["model_file"],
            },
        ),
    ]


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle an MCP tool call."""
    tool_map = {
        "gpx_fit": tools.gpx_fit,
        "gpx_qa": tools.gpx_qa,
        "gpx_spec": tools.gpx_spec,
        "gpx_predict": tools.gpx_predict,
        "gpx_py_generate": tools.gpx_py_generate,
    }

    if name not in tool_map:
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

    try:
        result = tool_map[name](**arguments)
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, indent=2)}
            ]
        }
    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Error executing {name}: {str(e)}"}
            ]
        }


async def main() -> None:
    """Run the MCP server."""
    server = Server("egobox-gpx")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return create_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await handle_tool_call(name, arguments)

    async with stdio_server() as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
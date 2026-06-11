+++
title = "EGObox"
+++

# EGObox

EGObox is an Efficient Global Optimization toolbox with Rust crates for surrogate modeling and Bayesian optimization plus Python bindings for the main user-facing workflows.

The companion website is intentionally wired to repository sources instead of hand-maintained snippets. The examples page pulls code directly from versioned files, and the website workflow exercises the Python examples that are shown here before publishing the site.

[Get started](getting-started.md){ .btn .btn-primary }
[Browse examples](examples.md){ .btn .btn-secondary }

## What is here

- Efficient global optimization with `Egor`
- Gaussian-process and mixture surrogate models with `Gpx` and `GpMix`
- Rust crates for DOE, GP, MOE, and EGO workflows
- Notebook-based tutorials for Python and CLI usage

## Project surfaces

| Surface | Best entry point | Notes |
| --- | --- | --- |
| Python package | `pip install egobox` | Main end-user API for optimization and surrogate modeling |
| Rust crates | `cargo add egobox-ego` and related crates | Lower-level building blocks for optimization workflows |
| Tutorials | [Tutorials](tutorials.md) | Existing notebooks in the repository and Colab links |
| Examples | [Examples](examples.md) | Website snippets sourced from tested repository files |

## Why Zola here

This site is meant to stay lightweight:

- Markdown-first pages for quick edits
- Direct inclusion of repository example files
- Static hosting on GitHub Pages
- A small docs workflow that builds the site and verifies the showcased Python examples
- Built with Zola - a fast static site generator written in Rust
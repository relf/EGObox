# Gaussian Process Quality Assurance

`gpx` is a companion executable of the `egobox` library.
It can assess surrogate quality, display model input/output specification,
and run predictions from CSV inputs.
CSV files for `fit` and `predict` may include an optional header row.

## Installation

```bash
cargo install gpx
```

## Usage

```bash
gpx --help
gpx fit --help
gpx qa --help
gpx spec --help
gpx predict --help
```

`gpx fit` supports training input as `csv` (default) or `npy` via `--input-format`.
`gpx fit` supports model output in `binary` (default) or `json` via `--format`.
`gpx predict` supports input and output data as `csv` (default) or `npy` via
`--input-format` and `--output-format`.

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

`gpx fit` auto-detects training input format from file extension:
- `.npy` -> `npy`
- otherwise -> `csv`
`gpx fit` uses the last `N` columns as outputs via `--outputs N` (default `1`) and
trains one surrogate model per output column.
`gpx fit` auto-detects model output format from output file extension:
- `.json` -> `json`
- otherwise -> `binary`
`gpx predict` supports input and output data as `csv` (default) or `npy` via
`--input-format` and `--output-format`.

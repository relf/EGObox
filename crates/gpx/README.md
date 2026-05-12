# Gaussian Process Quality Assurance

`gpx` is a companion executable of the `egobox` library.
It can fit GP surrogates to given data, assess surrogate quality, display model input/output specification, and run predictions from CSV inputs.
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
gpx py --help
```

`gpx fit` auto-detects training input format from file extension:
- `.npy` -> `npy`
- otherwise -> `csv`
`gpx fit` takes training input as a required positional argument.
`gpx fit` uses the last `N` columns as outputs via `--outputs N` (default `1`) and
trains one surrogate model per output column.
`gpx fit` supports expert-model configuration options:
- `--regression-spec <constant|linear|quadratic|all>` (default `constant`)
- `--correlation-spec <squared-exponential|absolute-exponential|matern32|matern52|all>` (default `squared-exponential`)
- `--kpls-dim <N>` (optional)
- `--n-clusters <N>` (default `1`, must be >= `1`)
- `--recombination <hard|smooth>` (default `smooth`)
- `--smooth-factor <F>` (optional, valid only when recombination is `smooth`)
`gpx fit` writes to `surrogate_model.gpx` by default; use `-o/--output` to change it.
`gpx fit` auto-detects model output format from output file extension:
- `.json` -> `json`
- otherwise -> `binary`
`gpx qa` and `gpx spec` use all surrogates by default; use `-m/--model-index`
to target a single surrogate.
`gpx predict` takes input as a required positional argument.
`gpx predict` uses `surrogate_model.gpx` by default; use `--model` to change it.
By default `gpx predict` uses all surrogate models from the model file and writes
one output file containing inputs followed by predicted outputs (training-like layout).
Use `-m/--model-index` to predict a single output model.
`gpx predict` auto-detects input format from input file extension:
- `.npy` -> `npy`
- otherwise -> `csv`
`gpx predict` writes to `surrogate_predictions.csv` by default; use `-o/--output` to change it.
`gpx predict` auto-detects output format from output file extension:
- `.npy` -> `npy`
- otherwise -> `csv`
`gpx py` generates a Python helper script from an existing model file.
Defaults:
- model: `surrogate_model.gpx` (`--model`)
- output script: `gpx.py` (`-o/--output`)
The generated script embeds the model bytes and provides:
- `predict(x, with_variance=False, model_index=None)`
At runtime, the generated Python helper invokes `gpx predict` via subprocess.

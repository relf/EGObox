# Changes

## gpx 0.3.0 - unreleased

Rename `gpqa` to `gpx`.

Add executable subcommands to:
- assess model quality (`qa`),
- display input/output specifications (`spec`),
- fit a default GP surrogate from tabular training data (`fit`),
- predict outputs from tabular inputs (`predict`).

Add model output format selection for `fit` by output file extension:
- `.json` -> `json`,
- otherwise -> `binary`.

Add tabular input/output data format support:
- `fit` input format inferred from extension (`.npy` -> `npy`, otherwise `csv`),
- `predict` input/output formats inferred from extension (`.npy` -> `npy`, otherwise `csv`).

Add optional CSV header row support for `fit` and `predict`.

Add multi-output fit support:
- `fit --outputs N` uses the last `N` columns as outputs (default `N=1`),
- one surrogate model is trained and saved per output column.

Add fit-time surrogate configuration options:
- `fit --regression-spec <constant|linear|quadratic|all>`,
- `fit --correlation-spec <squared-exponential|absolute-exponential|matern32|matern52|all>`,
- `fit --kpls-dim <N>`,
- `fit --n-clusters <N>`,
- `fit --recombination <hard|smooth>`,
- `fit --smooth-factor <F>` (smooth recombination only).

Make `fit` training input a required positional argument (not `-i/--input`).

Make `fit` output optional with default file `surrogate_model.gpx`.

Make `predict` input a required positional argument and `predict` output optional with
default file `surrogate_predictions.csv`.

Make `predict` model optional with default file `surrogate_model.gpx`.

Make `predict` use all surrogates by default (unless `--model-index` is set) and
write a single output containing input columns followed by predicted output columns.

## gpx 0.2.0 - 04/12/2025

Add Integrated Absolute Error metric

## gpx 0.1.0 - 25/09/2025

Initial experimental release of `gpx` utility used to assess GP surrogate quality.

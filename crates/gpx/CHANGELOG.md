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
- `predict --input-format csv|npy`,
- `predict --output-format csv|npy`.

Add optional CSV header row support for `fit` and `predict`.

Add multi-output fit support:
- `fit --outputs N` uses the last `N` columns as outputs (default `N=1`),
- one surrogate model is trained and saved per output column.

## gpx 0.2.0 - 04/12/2025

Add Integrated Absolute Error metric

## gpx 0.1.0 - 25/09/2025

Initial experimental release of `gpx` utility used to assess GP surrogate quality.

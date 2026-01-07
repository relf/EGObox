# COCO Benchmark for Egor

This directory contains a script to benchmark the Egor optimizer using the COCO (COmparing Continuous Optimizers) framework.

## Installation

First, install the required COCO package:

```bash
pip install coco-experiment
```

## Quick Test

To run a quick test on a small subset of problems (5 functions in 5 dimensions):

```bash
python coco_benchmark_egor.py --quick
```

This will:
- Test 5 bbob functions in 5 dimensions
- Use 10 initial DOE points
- Run 40 optimization iterations
- Save results to `exdata/egor_quick_test/`

## Full Benchmark

To run the full benchmark on all bbob functions up to 40 dimensions:

```bash
python coco_benchmark_egor.py
```

This will:
- Test all 24 bbob functions
- Test dimensions 2, 3, 5, 10, 20, 40
- Use `n_doe = 2 × dimension` initial points
- Use `max_iters = 10 × dimension - n_doe` optimization iterations
- Save results to `exdata/egor_default_bbob/`

The total number of problems is: 24 functions × 6 dimensions × 15 instances = 2160 problems.

⚠️ **Warning**: The full benchmark can take several hours to complete!

## Configuration

You can modify the benchmark parameters by editing the `run_egor_on_coco()` call in the script:

```python
run_egor_on_coco(
    suite_name="bbob",          # COCO suite name
    budget_multiplier=10,       # Total budget = dimension × multiplier
    n_doe_multiplier=2,         # Initial DOE = dimension × multiplier
    max_dimension=40,           # Maximum dimension to test
)
```

## Results

Results are saved in the `exdata/` directory with detailed logs of:
- Function evaluations
- Best values found
- Target achievement status

## Post-Processing

To analyze and visualize the results, use COCO's post-processing tools:

```bash
# Install post-processing tools
pip install cocopp

# Generate performance plots and tables
python -m cocopp exdata/egor_default_bbob/

# Compare multiple algorithms
python -m cocopp exdata/algorithm1/ exdata/algorithm2/
```

This will generate:
- Performance profiles
- Convergence plots
- Statistical tables
- HTML report

See the output directory (typically `ppdata/`) for the generated reports.

## COCO bbob Test Suite

The bbob (Black-Box Optimization Benchmarking) suite includes:
- **24 noiseless functions** covering various problem types:
  - Separable functions (f1-f5)
  - Functions with low/moderate conditioning (f6-f9)
  - Functions with high conditioning (f10-f14)
  - Multi-modal functions (f15-f19)
  - Weak structure functions (f20-f24)
- **Multiple dimensions**: 2, 3, 5, 10, 20, 40
- **15 instances** per function for statistical significance

## Egor Configuration

The benchmark uses Egor with **default configuration**:
- `infill_strategy`: LOG_EI (default)
- `gp_config`: Default GP configuration
- `n_start`: 20 (default)
- No TREGO or CoEGO
- Initial DOE: Latin Hypercube Sampling

## References

- COCO framework: https://github.com/numbbo/coco
- Hansen et al. (2021): "COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting"
- Egobox documentation: https://github.com/relf/egobox

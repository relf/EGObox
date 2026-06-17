# EGOR tuning heuristics

## Dimension

Low-dimensional problems:
- More initial samples
- More exploration

High-dimensional problems:
- Fewer initial samples
- Stronger surrogate guidance


## Evaluation cost

Cheap objective:
- Larger budget
- More exploration

Expensive objective:
- Smaller budget
- Maximize information gain


## Convergence

If progress is poor:

1. Try TREGO.
2. Replace SquaredExponential with Matern52.
3. Increase exploration.

## Gaussian process model

For high dimensions:

d > 10:
- use KPLS

Recommended:
- d=20 -> kpls_dim=5
- d=100 -> kpls_dim=10

## Very high dimension

d > 50:
use CoEGO.

Example:
n_coop_comp=5

## Parallel execution

When multiple objective evaluations are available:

Use qEIConfig.

Examples:

d=50:
batch=5

d=100:
batch=10

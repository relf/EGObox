# EGOR Parameter Tuning Skill

## Goal

Automatically select and adapt EGOR optimization parameters based on:
- problem dimension
- objective evaluation runtime
- available optimization budget
- parallel evaluation availability
- convergence progress

## Workflow

1. Estimate problem characteristics.
2. Select initial EGOR strategy.
3. Monitor optimization progress.
4. Adapt exploration/exploitation.

## Basic rules

- More dimensions -> reduce direct sampling, use surrogate strategies.
- Expensive evaluations -> maximize information gain.
- Poor progress -> increase exploration.

## Advanced strategies

### Poor convergence

If optimization stagnates:
- enable TREGO
- switch kernel from SquaredExponential to Matern52
- increase exploration

### Dimension > 10

Use KPLS for Gaussian process fitting.

Examples:
- dimension 20: kpls_dim=5
- dimension 100: kpls_dim=10

### Dimension > 50

Use CoEGO.

Example:
- dimension 100: n_coop_comp=5

### Parallel evaluations

If evaluations can run in parallel:
use qEIConfig.

Examples:
- dimension 50: batch=5
- dimension 100: batch=10

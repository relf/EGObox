#!/usr/bin/env python
"""
COCO (COmparing Continuous Optimizers) benchmark for Egor optimizer.

This script benchmarks the Egor optimizer with default configuration
on the COCO bbob (Black-Box Optimization Benchmarking) test suite.

Requirements:
    pip install coco-experiment

Usage:
    python coco_benchmark_egor.py

The results will be saved in the 'exdata' directory and can be
post-processed using the COCO post-processing tools.

For more information about COCO:
    https://github.com/numbbo/coco
"""

import cocoex
import numpy as np
import egobox as egx


def run_egor_on_coco(
    suite_name="bbob",
    budget_multiplier=10,
    n_doe_multiplier=2,
    max_dimension=40,
    observer_options=None,
):
    """
    Run Egor optimizer on COCO benchmark suite.

    Parameters
    ----------
    suite_name : str
        COCO suite name (default: 'bbob')
    budget_multiplier : int
        Budget multiplier relative to dimension (default: 10)
        Total evaluations = n_doe + max_iters
    n_doe_multiplier : int
        Initial DOE size multiplier (default: 2)
        n_doe = n_doe_multiplier * dimension
    max_dimension : int
        Maximum problem dimension to test (default: 40)
    observer_options : str, optional
        Additional observer options
    """

    # Create observer for logging results
    observer_name = f"egor_default_{suite_name}"
    if observer_options is None:
        observer_options = f"result_folder: exdata/{observer_name}"
    else:
        observer_options = f"result_folder: exdata/{observer_name} {observer_options}"

    observer = cocoex.Observer(suite_name, observer_options)

    # Create suite
    suite = cocoex.Suite(suite_name, "", f"dimensions: 2-{max_dimension}")

    print(f"Running Egor on {suite_name} benchmark")
    print(f"Suite contains {len(suite)} problems")
    print(f"Budget: dimension × {budget_multiplier}")
    print(f"Initial DOE: dimension × {n_doe_multiplier}")
    print("-" * 60)

    # Track statistics
    n_problems = 0
    n_success = 0

    # Iterate through all problems in the suite
    for problem in suite:
        problem.observe_with(observer)

        dimension = problem.dimension
        n_doe = n_doe_multiplier * dimension
        max_iters = budget_multiplier * dimension - n_doe

        # Skip if budget is too small
        if max_iters < 1:
            print(f"Skipping {problem.name}: budget too small")
            continue

        print(
            f"Problem: {problem.name} (dim={dimension}, "
            f"n_doe={n_doe}, max_iters={max_iters})"
        )

        # Get problem bounds
        bounds = [
            [lb, ub] for lb, ub in zip(problem.lower_bounds, problem.upper_bounds)
        ]

        # Define objective function wrapper
        def objective(x):
            """Wrapper for COCO problem evaluation."""
            x = np.atleast_2d(x)
            # Evaluate each row
            results = np.array([problem(xi) for xi in x])
            return results.reshape(-1, 1)

        try:
            # Create Egor optimizer with default configuration
            egor = egx.Egor(
                bounds,
                n_doe=n_doe,
                seed=None,  # Use different seed for each problem
            )

            # Run optimization
            result = egor.minimize(objective, max_iters=max_iters)

            # Check if target was reached
            if hasattr(problem, "final_target_hit"):
                if problem.final_target_hit:
                    n_success += 1
                    print(f"  ✓ Target reached! Best: {result.y_opt[0]:.6e}")
                else:
                    print(f"  ✗ Target not reached. Best: {result.y_opt[0]:.6e}")
            else:
                print(f"  Best value: {result.y_opt[0]:.6e}")

            n_problems += 1

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("-" * 60)
    print(f"Completed {n_problems} problems")
    if n_success > 0:
        print(
            f"Success rate: {n_success}/{n_problems} ({100 * n_success / n_problems:.1f}%)"
        )
    print(f"\nResults saved to: exdata/{observer_name}/")
    print("\nTo post-process results, use COCO's ppdata tools:")
    print("  python -m cocopp exdata/<folder>")


def run_quick_test():
    """
    Run a quick test on a subset of problems.
    Useful for testing the integration.
    """
    print("=" * 60)
    print("Quick COCO test with Egor (5-D, first 5 functions)")
    print("=" * 60)

    observer = cocoex.Observer("bbob", "result_folder: exdata/egor_quick_test")
    suite = cocoex.Suite(
        "bbob", "", "dimensions: 5 function_indices: 1-5 instance_indices: 1"
    )

    for problem in suite:
        problem.observe_with(observer)

        dimension = problem.dimension
        n_doe = 10
        max_iters = 40

        print(f"\n{problem.name} (dim={dimension})")

        bounds = [
            [lb, ub] for lb, ub in zip(problem.lower_bounds, problem.upper_bounds)
        ]

        def objective(x):
            x = np.atleast_2d(x)
            results = np.array([problem(xi) for xi in x])
            return results.reshape(-1, 1)

        egor = egx.Egor(bounds, n_doe=n_doe, seed=42)
        result = egor.minimize(objective, max_iters=max_iters)

        print(f"  Best value: {result.y_opt[0]:.6e}")
        print(f"  Evaluations: {len(result.y_doe)}")

    print("\nQuick test completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test mode
        run_quick_test()
    else:
        # Full benchmark
        run_egor_on_coco(
            suite_name="bbob",
            budget_multiplier=10,
            n_doe_multiplier=2,
            max_dimension=40,
        )

        print("\n" + "=" * 60)
        print("Benchmark completed!")
        print("=" * 60)

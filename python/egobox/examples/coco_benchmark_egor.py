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

import logging

logging.basicConfig(level=logging.ERROR)


def run_egor_on_coco(
    suite_name="bbob",
    budget_multiplier=20,
    n_doe_multiplier=4,
    max_dimension=2,
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
        observer_options = f"result_folder: {observer_name}"
    else:
        observer_options = f"result_folder: {observer_name} {observer_options}"

    observer = cocoex.Observer(suite_name, observer_options)

    repeater = cocoex.ExperimentRepeater(budget_multiplier)  # 0 == no repetitions
    minimal_print = cocoex.utilities.MiniPrint()

    # Create suite
    suite = cocoex.Suite(
        suite_name,
        "",
        "dimensions: 2 function_indices: 1 instance_indices: 1",
    )

    print(f"Running Egor on {suite_name} benchmark")
    print(f"Suite contains {len(suite)} problems")
    print(f"Budget: dimension × {budget_multiplier}")
    print(f"Initial DOE: dimension × {n_doe_multiplier}")
    print("-" * 60)

    # Track statistics
    n_problems = 0
    n_success = 0

    while not repeater.done():
        # Iterate through all problems in the suite
        for problem in suite:
            if repeater.done(problem):
                continue

            problem.observe_with(observer)

            dimension = problem.dimension
            n_doe = n_doe_multiplier * dimension
            max_iters = budget_multiplier * dimension - n_doe

            # print(
            #     f"Problem: {problem.name} (dim={dimension}, "
            #     f"n_doe={n_doe}, max_iters={max_iters})"
            # )

            # Get problem bounds
            bounds = [
                [lb, ub] for lb, ub in zip(problem.lower_bounds, problem.upper_bounds)
            ]

            # Define objective function wrapper
            def objective(x):
                """Wrapper for COCO problem evaluation."""
                x = np.atleast_2d(x).reshape(-1, dimension)
                # Evaluate each row
                results = np.array([problem(xi) for xi in x])
                return results.reshape(-1, 1)

            # Create Egor optimizer with default configuration
            egor = egx.Egor(
                bounds,
                infill_strategy=egx.InfillStrategy.WB2,
                n_doe=n_doe,
                seed=None,  # Use different seed for each problem
                trego=True,
            )

            # Run optimization
            result = egor.minimize(objective, max_iters=max_iters)
            problem(result.x_opt)

            repeater.track(problem)  # track evaluations and final_target_hit")
            minimal_print(problem, repeater)
            n_problems += 1

        print("\n" + "-" * 60)
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
        "bbob", "", "dimensions: 2 function_indices: 1 instance_indices: 1"
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
            budget_multiplier=20,
            n_doe_multiplier=4,
            max_dimension=2,
        )

        print("\n" + "=" * 60)
        print("Benchmark completed!")
        print("=" * 60)

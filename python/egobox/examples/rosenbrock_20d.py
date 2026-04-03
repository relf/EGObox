# =====================================================
# Egobox demo: Minimize Rosenbrock 20D with LOG_EI + TREGO
# Runs multiple replications and plots convergence history
# =====================================================

import argparse
import os
import numpy as np
import egobox as egx
import matplotlib.pyplot as plt


# -----------------------------------------------------
# Define the 20D Rosenbrock function
# -----------------------------------------------------
def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock function in nD: sum_{i=0}^{n-2} [100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2]
    Global minimum: f(1,...,1) = 0
    """
    x = np.atleast_2d(x)
    val = np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2, axis=1
    )
    return val.reshape(-1, 1)


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Minimize Rosenbrock 20D using EGObox with LOG_EI + TREGO"
    )
    parser.add_argument(
        "-n",
        "--n-runs",
        type=int,
        default=1,
        help="Number of optimization runs (default: 1)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="Maximum number of EGO iterations per run (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    args = parser.parse_args()

    dim = 20
    bounds = [[-2.0, 2.0]] * dim
    n_doe = dim + 1
    max_iters = args.max_iters
    base_seed = args.seed
    n_runs = args.n_runs
    base_outdir = "rosenbrock20D_001_out"

    all_histories = []

    for run in range(n_runs):
        seed = base_seed + run
        outdir = os.path.join(base_outdir, f"run_{run}") if n_runs > 1 else base_outdir
        os.makedirs(outdir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{n_runs}  (seed={seed}, outdir={outdir})")
        print(f"{'=' * 60}")

        opt = egx.Egor(
            bounds,
            n_doe=n_doe,
            # infill_strategy=egx.InfillStrategy.LOG_EI,
            gp_config=egx.GpConfig(kpls_dim=5, corr_spec=egx.CorrelationSpec.MATERN52),
            trego=True,
            outdir=outdir,
            seed=run,
            verbose=egx.Verbose.INFO,
            target=0.01,
        )

        optim = opt.minimize(
            rosenbrock,
            max_iters=max_iters,
            run_info=egx.RunInfo(fname="Rosenbrock20D", num=run + 1),
        )

        print(f"Best value (y*): {optim.result.y_opt}")
        print(
            f"Best point norm ||x* - 1||: {np.linalg.norm(optim.result.x_opt - 1.0):.4f}"
        )

        # Load convergence history saved by the optimizer
        hist_file = os.path.join(outdir, "egor_history.npy")
        history = np.load(hist_file)
        best_obj = history[:, 0]  # first column = best objective per iteration
        all_histories.append(best_obj)

    # -------------------------------------------------
    # Plot convergence
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))

    if n_runs == 1:
        iters = np.arange(1, len(all_histories[0]) + 1)
        plt.plot(iters, all_histories[0], "b-o", markersize=3, label="Best objective")
    else:
        # Pad histories to the same length for stats
        max_len = max(len(h) for h in all_histories)
        padded = np.full((n_runs, max_len), np.nan)
        for i, h in enumerate(all_histories):
            padded[i, : len(h)] = h

        iters = np.arange(1, max_len + 1)
        median = np.nanmedian(padded, axis=0)
        q25 = np.nanpercentile(padded, 25, axis=0)
        q75 = np.nanpercentile(padded, 75, axis=0)

        plt.plot(iters, median, "b-", linewidth=2, label="Median")
        plt.fill_between(
            iters, q25, q75, alpha=0.25, color="blue", label="IQR (25%-75%)"
        )
        for i, h in enumerate(all_histories):
            plt.plot(
                np.arange(1, len(h) + 1),
                h,
                alpha=0.3,
                linewidth=0.8,
                color="gray",
            )

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value (log scale)")
    plt.title(
        f"Rosenbrock 20D — LOG_EI + TREGO (n_doe={n_doe}, {n_runs} run{'s' if n_runs > 1 else ''})"
    )
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(base_outdir, "convergence.png")
    os.makedirs(base_outdir, exist_ok=True)
    plt.savefig(plot_file, dpi=150)
    print(f"\nConvergence plot saved to {plot_file}")
    plt.show()


if __name__ == "__main__":
    main()

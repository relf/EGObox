"""
Compare EFI_P vs EFI_FE on modified Branin-Hoo with Egor.

This example reproduces the hidden-failure setup from:
Tfaily et al. (2024), "Bayesian optimization with hidden constraints for aircraft design".

Both optimizations are run with Egor and differ only by
``feasible_infill_strategy``:
- ``FeasibleInfillStrategy.EFI_P`` (reference alpha=1.0)
- ``FeasibleInfillStrategy.EFI_FE`` (feasibility enhanced alpha=0.3)

The generated figure contains, for both runs:
- 2D Branin contour map
- hidden infeasible zone overlay
- initial DOE points
- iteration trajectory
- valid vs failed points
- best feasible minimum found
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import egobox as egx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Problem definition in scaled domain x in [0, 1]^2.
BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)


def branin_hoo_scaled(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    x1 = 15.0 * x[:, 0] - 5.0
    x2 = 15.0 * x[:, 1]

    term1 = x2 - 5.1 * x1**2 / (4.0 * np.pi**2) + 5.0 * x1 / np.pi - 6.0
    term2 = 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(x1) + 10.0
    return (term1**2 + term2 + 5.0 * x1).reshape(-1, 1)


def is_hidden_failure(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    in_hidden = (x1 < 0.4) & (x2 > 0.5)
    in_feasible_island = (x1 > 0.05) & (x1 < 0.2) & (x2 > 0.7) & (x2 < 0.9)
    return in_hidden & (~in_feasible_island)


def evaluate_hidden_branin(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    y = branin_hoo_scaled(x).reshape(-1, 1)
    y[is_hidden_failure(x)] = np.nan
    return y


@dataclass
class RunHistory:
    name: str
    x_all: np.ndarray
    initial_doe: np.ndarray
    valid_all: np.ndarray
    failed_points: np.ndarray
    n_doe: int
    x_best: np.ndarray
    y_best: float


def _get_feasible_infill_enum():
    if not hasattr(egx, "FeasibleInfillStrategy"):
        raise RuntimeError(
            "egobox.FeasibleInfillStrategy is not available in the currently installed module. "
            "Rebuild/reinstall the local package from this repo (for example with maturin develop) "
            "to use EFI_P and EFI_FE from Egor."
        )
    return egx.FeasibleInfillStrategy


def run_egor(
    name: str,
    feasible_strategy,
    x_doe: np.ndarray,
    n_doe: int,
    n_iters: int,
    seed: int,
    outdir: Path | None = None,
) -> RunHistory:
    egor = egx.Egor(
        BOUNDS.tolist(),
        doe=x_doe,
        infill_strategy=egx.InfillStrategy.EI,
        feasible_infill_strategy=feasible_strategy,
    )
    run_outdir = None if outdir is None else str(outdir)
    optim = egor.minimize(
        evaluate_hidden_branin,
        max_iters=n_iters,
        seed=seed,
        outdir=run_outdir,
        verbose=2,
    )
    x_all = np.asarray(optim.result.x_doe)
    y_all = evaluate_hidden_branin(x_all).reshape(-1)
    valid_all = np.isfinite(y_all)
    x_valid = x_all[valid_all]
    y_valid = y_all[valid_all]
    idx_best = int(np.argmin(y_valid))

    failed_points = x_all[~valid_all]
    if outdir is not None:
        failed_points_path = outdir / "egor_failed_points.npy"
        if failed_points_path.exists():
            failed_points = np.asarray(np.load(failed_points_path), dtype=float)
            if failed_points.ndim == 1:
                failed_points = failed_points.reshape(-1, 2)

    return RunHistory(
        name=name,
        x_all=x_all,
        initial_doe=np.asarray(x_doe, dtype=float),
        valid_all=valid_all,
        failed_points=failed_points,
        n_doe=n_doe,
        x_best=x_valid[idx_best],
        y_best=float(y_valid[idx_best]),
    )


def load_saved_history(
    name: str,
    saved_dir: Path,
    shared_initial_doe: np.ndarray,
    n_doe_fallback: int,
) -> RunHistory:
    doe_path = saved_dir / "egor_doe.npy"
    if not doe_path.exists():
        raise FileNotFoundError(f"Missing saved DOE file: {doe_path}")

    doe = np.load(doe_path)
    if doe.ndim != 2 or doe.shape[1] < 2:
        raise ValueError(f"Invalid DOE shape in {doe_path}: {doe.shape}")

    x_all = np.asarray(doe[:, :2], dtype=float)

    # If objective column is present, use it for valid/invalid mask;
    # otherwise recompute from x values.
    if doe.shape[1] >= 3:
        y_all = np.asarray(doe[:, 2], dtype=float)
    else:
        y_all = evaluate_hidden_branin(x_all).reshape(-1)

    valid_all = np.isfinite(y_all)
    if not np.any(valid_all):
        raise ValueError(f"No valid evaluations found in {doe_path}")

    if shared_initial_doe.size > 0:
        n_doe = int(shared_initial_doe.shape[0])
    else:
        initial_doe_path = saved_dir / "egor_initial_doe.npy"
        if initial_doe_path.exists():
            initial_doe = np.load(initial_doe_path)
            n_doe = int(initial_doe.shape[0])
            shared_initial_doe = np.asarray(initial_doe[:, :2], dtype=float)
        else:
            n_doe = min(int(n_doe_fallback), int(x_all.shape[0]))
            shared_initial_doe = np.asarray(x_all[:n_doe, :2], dtype=float)

    x_valid = x_all[valid_all]
    y_valid = y_all[valid_all]
    idx_best = int(np.argmin(y_valid))

    failed_points_path = saved_dir / "egor_failed_points.npy"
    if failed_points_path.exists():
        failed_points = np.asarray(np.load(failed_points_path), dtype=float)
        if failed_points.ndim == 1:
            failed_points = failed_points.reshape(-1, 2)
    else:
        failed_points = x_all[~valid_all]

    return RunHistory(
        name=name,
        x_all=x_all,
        initial_doe=shared_initial_doe,
        valid_all=valid_all,
        failed_points=failed_points,
        n_doe=n_doe,
        x_best=x_valid[idx_best],
        y_best=float(y_valid[idx_best]),
    )


def approximate_reference_optimum(n_grid: int = 500) -> tuple[np.ndarray, float]:
    x1 = np.linspace(0.0, 1.0, n_grid)
    x2 = np.linspace(0.0, 1.0, n_grid)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    y = evaluate_hidden_branin(grid).reshape(-1)
    valid = np.isfinite(y)
    idx = int(np.argmin(y[valid]))
    best_x = grid[valid][idx]
    best_y = float(y[valid][idx])
    return best_x, best_y


def plot_run(
    ax: plt.Axes,
    hist: RunHistory,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    hidden_mask: np.ndarray,
) -> None:
    contour = ax.contourf(xx, yy, zz, levels=50, cmap="viridis", alpha=0.7)

    ax.contourf(
        xx,
        yy,
        hidden_mask.astype(float),
        levels=[0.5, 1.5],
        colors=["red"],
        alpha=0.3,
    )
    ax.contour(xx, yy, zz, levels=12, colors="white", linewidths=0.4, alpha=0.5)

    x0 = np.asarray(hist.initial_doe, dtype=float)
    y0 = evaluate_hidden_branin(x0).reshape(-1)
    v0 = np.isfinite(y0)
    xt = hist.x_all[hist.n_doe :]
    vt = hist.valid_all[hist.n_doe :]

    if np.any(v0):
        ax.scatter(
            x0[v0, 0],
            x0[v0, 1],
            c="blue",
            s=100,
            marker="o",
            edgecolors="black",
            linewidths=1.5,
            label="Initial DOE (Valid)",
            zorder=5,
        )

    if np.any(vt):
        ax.scatter(
            xt[vt, 0],
            xt[vt, 1],
            c="orange",
            s=120,
            alpha=0.9,
            marker="o",
            edgecolors="black",
            linewidths=1.5,
            label="Iteration Points",
            zorder=5,
        )

    failed_points = hist.failed_points
    if failed_points.size == 0:
        failed_points = np.empty((0, 2), dtype=float)
    if failed_points.ndim == 1:
        failed_points = failed_points.reshape(-1, 2)

    if failed_points.shape[0] > 0:
        # Black underlay keeps red crosses visible over the red infeasible zone.
        ax.scatter(
            failed_points[:, 0],
            failed_points[:, 1],
            c="black",
            s=125,
            alpha=0.95,
            marker="x",
            linewidths=3.8,
            zorder=7,
        )
        ax.scatter(
            failed_points[:, 0],
            failed_points[:, 1],
            c="red",
            s=95,
            alpha=0.9,
            marker="x",
            linewidths=2.2,
            label="Failed Points",
            zorder=7,
        )

    ax.plot(
        hist.x_best[0],
        hist.x_best[1],
        "mD",
        markersize=13,
        markeredgecolor="black",
        markeredgewidth=1.5,
        label=f"Found Min: {hist.y_best:.4f}",
        zorder=8,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(hist.name)
    ax.grid(alpha=0.2)

    return contour


def make_comparison_figure(
    hist_ref: RunHistory, hist_fe: RunHistory, outpath: Path
) -> None:
    n_grid = 300
    x1 = np.linspace(0.0, 1.0, n_grid)
    x2 = np.linspace(0.0, 1.0, n_grid)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    zz = branin_hoo_scaled(grid).reshape(xx.shape)
    hidden_mask = is_hidden_failure(grid).reshape(xx.shape)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    contour = plot_run(axes[0], hist_ref, xx, yy, zz, hidden_mask)
    plot_run(axes[1], hist_fe, xx, yy, zz, hidden_mask)

    ref_x, ref_y = approximate_reference_optimum()
    for ax in axes:
        ax.scatter(
            ref_x[0],
            ref_x[1],
            marker="P",
            s=100,
            c="white",
            edgecolors="black",
            linewidths=1.0,
            label=f"Grid ref min: {ref_y:.3f}",
            zorder=8,
        )

    legend_handles = [
        Patch(facecolor="red", edgecolor="red", alpha=0.3, label="Infeasible zone"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=9,
            label="Initial DOE (Valid)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markeredgecolor="black",
            markersize=9,
            label="Iteration Points",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="red",
            linestyle="None",
            markersize=9,
            markeredgewidth=2.2,
            label="Failed Points",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="magenta",
            markeredgecolor="black",
            linestyle="None",
            markersize=9,
            label="Found Minimum",
        ),
        Line2D(
            [0],
            [0],
            marker="P",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="None",
            markersize=9,
            label="Grid Reference Minimum",
        ),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=4,
        frameon=True,
    )
    cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("Branin value")

    fig.suptitle(
        "Modified Branin-Hoo with hidden infeasible zone: EFI_P vs EFI_FE", fontsize=13
    )
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EFI_P and EFI_FE on hidden-constraint Branin-Hoo"
    )
    parser.add_argument(
        "--n-doe", type=int, default=7, help="Number of initial DOE points"
    )
    parser.add_argument(
        "--n-iters", type=int, default=50, help="Number of BO iterations"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="Load histories from saved files instead of running optimization",
    )
    parser.add_argument(
        "--saved-root",
        type=Path,
        default=Path("branin_efi_out"),
        help="Root directory containing efi_p/ and efi_fe/ saved outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("branin_hidden_constraints_efi_fe.png"),
        help="Output figure path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_saved:
        shared_initial_path = args.saved_root / "initial_doe.npy"
        if shared_initial_path.exists():
            shared_initial_doe = np.asarray(np.load(shared_initial_path), dtype=float)
            if shared_initial_doe.ndim == 1:
                shared_initial_doe = shared_initial_doe.reshape(-1, 2)
            if shared_initial_doe.shape[1] > 2:
                shared_initial_doe = shared_initial_doe[:, :2]
        else:
            efi_p_initial = args.saved_root / "efi_p" / "egor_initial_doe.npy"
            if efi_p_initial.exists():
                shared_initial_doe = np.asarray(np.load(efi_p_initial), dtype=float)
                if shared_initial_doe.ndim == 1:
                    shared_initial_doe = shared_initial_doe.reshape(-1, 2)
                if shared_initial_doe.shape[1] > 2:
                    shared_initial_doe = shared_initial_doe[:, :2]
            else:
                shared_initial_doe = np.empty((0, 2), dtype=float)

        hist_ref = load_saved_history(
            name="Reference optimizer (EFI_P)",
            saved_dir=args.saved_root / "efi_p",
            shared_initial_doe=shared_initial_doe,
            n_doe_fallback=args.n_doe,
        )
        hist_fe = load_saved_history(
            name="Feasibility enhanced optimizer (EFI_FE)",
            saved_dir=args.saved_root / "efi_fe",
            shared_initial_doe=shared_initial_doe,
            n_doe_fallback=args.n_doe,
        )
    else:
        feasible_enum = _get_feasible_infill_enum()
        x_doe = egx.lhs(BOUNDS.tolist(), args.n_doe, seed=args.seed)
        (args.saved_root / "efi_p").mkdir(parents=True, exist_ok=True)
        (args.saved_root / "efi_fe").mkdir(parents=True, exist_ok=True)
        np.save(args.saved_root / "initial_doe.npy", x_doe)

        hist_ref = run_egor(
            name="Reference optimizer (EFI_P)",
            feasible_strategy=feasible_enum.EFI_P,
            x_doe=x_doe,
            n_doe=args.n_doe,
            n_iters=args.n_iters,
            seed=args.seed,
            outdir=args.saved_root / "efi_p",
        )
        hist_fe = run_egor(
            name="Feasibility enhanced optimizer (EFI_FE)",
            feasible_strategy=feasible_enum.EFI_FE,
            x_doe=x_doe,
            n_doe=args.n_doe,
            n_iters=args.n_iters,
            seed=args.seed,
            outdir=args.saved_root / "efi_fe",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_comparison_figure(hist_ref, hist_fe, args.output)

    print(f"Saved comparison figure to: {args.output}")
    print(f"EFI_P best feasible y: {hist_ref.y_best:.6f} at x={hist_ref.x_best}")
    print(f"EFI_FE best feasible y: {hist_fe.y_best:.6f} at x={hist_fe.x_best}")


if __name__ == "__main__":
    main()

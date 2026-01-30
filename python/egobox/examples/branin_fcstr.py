"""
Constrained Branin function optimization as used in:
Forrester, A., Sobester, A., & Keane, A. (2008).
Engineering Design via Surrogate Modelling: A Practical Guide.
Chichester, UK: John Wiley & Sons.

The Branin function is modified with a constraint x1*x2 >= 0.2
"""

import numpy as np
import matplotlib.pyplot as plt
from egobox import Egor

import logging

logging.basicConfig(level=logging.INFO)


def branin_forrester(x):
    """
    Branin function as used by Forrester et al.
    x is in [0, 1]^2, transformed to proper domain internally.

    Returns: [objective, constraint]
    where constraint should be negative at optimum (x1*x2 - 0.2 < 0)
    """
    # Transform from [0,1]^2 to Forrester's domain
    x1 = x[:, 0] * 15 - 5
    x2 = x[:, 1] * 15

    # Branin function
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    obj = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    return obj.reshape(-1, 1)


CONST = 0.2
NDOE = 3


def constraint_branin(x, gradient=False):
    """
    Constraint function: x1 * x2 - 0.2 >= 0
    Returns positive value if constraint is violated.
    """
    if gradient:
        return np.array([-x[1], -x[0]])
    else:
        return CONST - x[0] * x[1]


def branin_grouped(point):
    p = np.atleast_2d(point)
    cstr = np.array([constraint_branin(pi) for pi in p]).reshape(-1, 1)
    return np.column_stack([branin_forrester(p), cstr])


def plot_constrained_branin(res):
    """
    Plot Branin function contours, constraint boundary, and optimum.
    """
    # Create mesh for contour plot
    x1 = np.linspace(0, 1, 200)
    x2 = np.linspace(0, 1, 200)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate Branin function
    points = np.column_stack([X1.ravel(), X2.ravel()])
    values = branin_forrester(points)
    Z = values[:, 0].reshape(X1.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Branin contours
    levels = np.linspace(Z.min(), Z.max(), 30)
    contour = ax.contourf(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.8)
    contour_lines = ax.contour(
        X1, X2, Z, levels=15, colors="white", linewidths=0.5, alpha=0.3
    )
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%1.0f")

    # Plot constraint boundary: x1*x2 = 0.2
    x1_constraint = np.linspace(0.01, 1, 500)
    x2_constraint = CONST / x1_constraint
    # Only plot where x2 is within [0, 1]
    mask = x2_constraint <= 1
    ax.plot(
        x1_constraint[mask],
        x2_constraint[mask],
        "r-",
        linewidth=2,
        label=f"Constraint: x₁·x₂ = {CONST}",
    )

    # Shade infeasible region (x1*x2 < CONST)
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    infeasible = X1_flat * X2_flat < CONST
    ax.scatter(
        X1_flat[infeasible],
        X2_flat[infeasible],
        c="red",
        s=1,
        alpha=0.1,
        label="Infeasible region",
    )

    # Plot optimization trajectory
    x_doe = res.x_doe
    ax.plot(
        x_doe[:, 0],
        x_doe[:, 1],
        "ko-",
        markersize=4,
        linewidth=1,
        alpha=0.5,
        label="Optimization path",
    )

    # Plot initial DOE
    # Assuming first points are initial DOE (before optimization iterations)
    n_initial = NDOE  # Adjust based on your DOE size
    if len(x_doe) >= n_initial:
        ax.scatter(
            x_doe[:n_initial, 0],
            x_doe[:n_initial, 1],
            c="cyan",
            s=100,
            marker="s",
            edgecolors="black",
            linewidths=1.5,
            label="Initial DOE",
            zorder=5,
        )

    # Plot optimum
    x_opt = res.x_opt
    y_opt = res.y_opt[0]
    ax.scatter(
        x_opt[0],
        x_opt[1],
        c="lime",
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label=f"Optimum: f={y_opt:.2f}",
        zorder=10,
    )

    # Add theoretical optimum location (approximately [0.9677, 0.2067])
    ax.scatter(
        0.9677,
        0.2067,
        c="white",
        s=200,
        marker="x",
        linewidths=3,
        label="Known optimum",
        zorder=10,
    )

    # Formatting
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_title(
        "Constrained Branin Function Optimization\n(Forrester et al., 2008)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Objective value", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.show()

    return fig


def main():
    """
    Main function to run constrained Branin optimization.
    """
    print("=" * 70)
    print("Constrained Branin Function Optimization")
    print(
        "Reference: Forrester et al. (2008), Engineering Design via Surrogate Modelling"
    )
    print("=" * 70)
    print()

    # Define problem: 2D input space in [0, 1]^2
    xspecs = [[0.0, 1.0], [0.0, 1.0]]

    # Create optimizer with constraint handling
    egor = Egor(
        xspecs,
        n_doe=NDOE,  # Initial DOE size
        seed=42,  # For reproducibility
    )

    print("Configuration:")
    print("  - Input space: [0, 1]² (transformed to Forrester's domain)")
    print(f"  - Constraint: x₁·x₂ ≥ {CONST}")
    print("  - Initial DOE: 15 points")
    print("  - Max iterations: 30")
    print()
    print("Running optimization...")
    print()

    # Run optimization
    res = egor.minimize(
        branin_forrester,
        fcstrs=[constraint_branin],
        max_iters=20,
    )

    # Display results
    print()
    print("=" * 70)
    print("Optimization Results")
    print("=" * 70)
    print(f"Optimum found at: x = [{res.x_opt[0]:.5f}, {res.x_opt[1]:.5f}]")
    print(f"Objective value: f(x*) = {res.y_opt[0]:.6f}")
    print(
        f"Constraint value: g(x*) = {constraint_branin(res.x_opt):.6f} (should be ≤ 0)"
    )
    print()

    # Transform to original domain for comparison
    x1_original = res.x_opt[0] * 15 - 5
    x2_original = res.x_opt[1] * 15
    print(f"In original domain: x = [{x1_original:.5f}, {x2_original:.5f}]")
    print(f"Product x₁·x₂ = {res.x_opt[0] * res.x_opt[1]:.6f} (should be ≥ {CONST})")
    print()

    # Known constrained optimum (approximately)
    x_known = np.array([0.9677, 0.2067])
    y_known = branin_forrester(x_known.reshape(1, -1))[0, 0]
    print(f"Known constrained optimum: x ≈ [{x_known[0]:.4f}, {x_known[1]:.4f}]")
    print(f"Known objective value: f(x) ≈ {y_known:.6f}")
    print(f"Known constraint value: g(x) ≈ {constraint_branin(x_known):.6f}")
    print()

    # Error from known optimum
    error_x = np.linalg.norm(res.x_opt - x_known)
    error_y = abs(res.y_opt[0] - y_known)
    print(f"Distance from known optimum: {error_x:.6f}")
    print(f"Objective error: {error_y:.6f}")
    print("=" * 70)

    # Plot results
    print("\nGenerating visualization...")
    plot_constrained_branin(res)


if __name__ == "__main__":
    main()

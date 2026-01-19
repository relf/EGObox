# =====================================================
# Egobox demo: Minimize Rosenbrock 2D with TREGO
# Visualize trust region evolution from saved states
# =====================================================

import os
import json
import numpy as np
import egobox as egx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import animation


# -----------------------------------------------------
# Define the 2D Rosenbrock function
# -----------------------------------------------------
def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return ((1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2).reshape(-1, 1)


# -----------------------------------------------------
# Define search space
# -----------------------------------------------------
bounds = [[-2.0, 2.0], [-1.0, 3.0]]

# -----------------------------------------------------
# Setup output directory for state files
# -----------------------------------------------------
outdir = "rosenbrock_output"
os.makedirs(outdir, exist_ok=True)

# Set environment variable to save EgorState at each iteration
os.environ["EGOR_USE_RUN_RECORDER"] = "WITH_ITER_STATE"

N_DOE = 10
DMIN = 1e-6  # TREGO min distance
DMAX = 1.0  # TREGO max distance for trust region

# -----------------------------------------------------
# Initialize optimizer with TREGO
# -----------------------------------------------------
print("Running Rosenbrock optimization with TREGO...")
opt = egx.Egor(
    bounds,
    n_doe=N_DOE,
    infill_strategy=egx.InfillStrategy.LOG_EI,  # default infill strategy
    trego=egx.TregoConfig(
        n_gl_steps=(1, 4), beta=0.9, alpha=1.0, d=(DMIN, DMAX)
    ),  # Enable TREGO with default parameter values
    outdir=outdir,
    seed=42,
)

# -----------------------------------------------------
# Run optimization
# -----------------------------------------------------
res = opt.minimize(rosenbrock, max_iters=20)

print("\n===== Optimization Result =====")
print("Best value (y*):", res.y_opt)
print("Best point (x*):", res.x_opt)
print("Known optimum: f(1, 1) = 0")


# -----------------------------------------------------
# Load EgorState files and extract trust region info
# -----------------------------------------------------
def load_state_files(outdir):
    """Load all egor_state_*.json files from output directory."""
    states = []
    iter_num = 0
    while True:
        state_file = os.path.join(outdir, f"egor_state_{iter_num:04d}.json")
        if not os.path.exists(state_file):
            break
        with open(state_file, "r") as f:
            state = json.load(f)
            states.append(state)
        iter_num += 1
    return states


def extract_trust_region_bounds(state, xlimits, delta_scale):
    """Extract trust region bounds from state."""
    # Trust region only used when local optimization is active
    if state.get("local_trego_iter") == 0:
        return None

    # Trust region is centered on current best x within dmax * sigma L1 distance
    if state.get("best_param") is None:
        return None

    param_json = state["best_param"]
    current_x = np.array(param_json["data"]).reshape(*param_json["dim"])

    sigma = state.get("sigma", 1.0)

    # Trust region bounds: current_x +/- delta_scale * sigma (L1 distance)
    delta = delta_scale * sigma

    tr_bounds = np.array(
        [[current_x[i] - delta, current_x[i] + delta] for i in range(len(current_x))]
    )

    # Clip to original bounds
    tr_bounds[:, 0] = np.maximum(tr_bounds[:, 0], np.array(xlimits)[:, 0])
    tr_bounds[:, 1] = np.minimum(tr_bounds[:, 1], np.array(xlimits)[:, 1])

    return tr_bounds


print(f"\nLoading state files from {outdir}...")
states = load_state_files(outdir)
print(f"Loaded {len(states)} state files")


# -----------------------------------------------------
# Create visualization grid
# -----------------------------------------------------
X = np.linspace(bounds[0][0], bounds[0][1], 300)
Y = np.linspace(bounds[1][0], bounds[1][1], 300)
XX, YY = np.meshgrid(X, Y)
XY = np.column_stack([XX.ravel(), YY.ravel()])
ZZ = rosenbrock(XY).reshape(XX.shape)


# -----------------------------------------------------
# Static plot: Show final state with all points
# -----------------------------------------------------
plt.figure(figsize=(10, 8))
levels = np.logspace(-1, 3, 30)
contour = plt.contour(XX, YY, ZZ, levels=levels, cmap="viridis", alpha=0.6)
plt.colorbar(contour, label="log(Rosenbrock value)")

# Plot all sampled points (distinguish DOE from iterative points)
X_data = np.array(res.x_doe)

# Plot initial DOE points
plt.scatter(
    X_data[:N_DOE, 0],
    X_data[:N_DOE, 1],
    c="blue",
    s=50,
    marker="s",
    alpha=0.6,
    edgecolors="darkblue",
    linewidths=1,
    label="Initial DOE points",
)

# Plot iterative points
if len(X_data) > N_DOE:
    plt.scatter(
        X_data[N_DOE:, 0],
        X_data[N_DOE:, 1],
        c="red",
        s=30,
        marker="o",
        alpha=0.5,
        label="Iterative points",
    )

# Plot optimum
plt.scatter(
    1.0,
    1.0,
    c="gold",
    s=200,
    marker="*",
    edgecolors="black",
    label="Global optimum (1,1)",
    zorder=10,
)

# Plot best found
plt.scatter(
    res.x_opt[0],
    res.x_opt[1],
    c="lime",
    s=150,
    marker="^",
    edgecolors="black",
    label="Best found",
    zorder=10,
)

# Show final trust region if available
if states:
    final_state = states[-1]
    tr_bounds = extract_trust_region_bounds(final_state, bounds, DMAX)
    if tr_bounds is not None:
        rect = Rectangle(
            (tr_bounds[0, 0], tr_bounds[1, 0]),
            tr_bounds[0, 1] - tr_bounds[0, 0],
            tr_bounds[1, 1] - tr_bounds[1, 0],
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
        )
        plt.gca().add_patch(rect)

plt.title("Rosenbrock Function - TREGO Optimization")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "rosenbrock_trego_final.png"), dpi=150)
plt.show()


# -----------------------------------------------------
# Animated plot: Show trust region evolution
# -----------------------------------------------------
print("\nCreating animation of trust region evolution...")

fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contour(XX, YY, ZZ, levels=levels, cmap="viridis", alpha=0.6)
plt.colorbar(contour, ax=ax, label="log(Rosenbrock value)")

# Global optimum marker (static)
ax.scatter(
    1.0,
    1.0,
    c="gold",
    s=200,
    marker="*",
    edgecolors="black",
    label="Global optimum (1,1)",
    zorder=10,
)

# Dynamic elements
(doe_points_plot,) = ax.plot(
    [],
    [],
    "bs",
    markersize=7,
    alpha=0.7,
    markeredgecolor="darkblue",
    markeredgewidth=1,
    label="Initial DOE points",
)
(iter_points_plot,) = ax.plot(
    [], [], "ro", markersize=5, alpha=0.7, label="Iterative points"
)
(best_plot,) = ax.plot([], [], "g^", markersize=10, label="Current best")
trust_region_patch = Rectangle(
    (0, 0), 0, 0, linewidth=2, edgecolor="black", facecolor="none", linestyle="--"
)
ax.add_patch(trust_region_patch)
ego_point = Circle(
    (0, 0), 0, linewidth=1, edgecolor="blue", facecolor="none", linestyle="-"
)
ax.add_patch(ego_point)

ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
title = ax.text(
    0.5,
    1.01,
    "Iteration 0",
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=ax.transAxes,
)


def init():
    """Initialize animation."""
    doe_points_plot.set_data([], [])
    iter_points_plot.set_data([], [])
    best_plot.set_data([], [])
    trust_region_patch.set_width(0)
    trust_region_patch.set_height(0)
    ego_point.set_radius(0)
    title.set_text("Iteration 0 - Best: N/A")
    return (
        doe_points_plot,
        iter_points_plot,
        best_plot,
        trust_region_patch,
        ego_point,
        title,
    )


def animate(frame):
    """Update animation frame."""
    if frame >= len(states):
        return (
            doe_points_plot,
            iter_points_plot,
            best_plot,
            trust_region_patch,
            ego_point,
            title,
        )

    state = states[frame]
    n_doe = N_DOE  # Initial DOE size

    # Get evaluated points up to this iteration
    if state.get("data") is not None and len(state["data"]) > 0:
        x_data_json = state["data"][
            0
        ]  # data is list [x, y, c] where each is ndarray JSON
        x_data = np.array(x_data_json["data"]).reshape(*x_data_json["dim"])

        # Separate DOE and iterative points
        doe_data = x_data[:n_doe] if len(x_data) >= n_doe else x_data
        iter_data = (
            x_data[n_doe:] if len(x_data) > n_doe else np.array([]).reshape(0, 2)
        )

        doe_points_plot.set_data(doe_data[:, 0], doe_data[:, 1])
        if len(iter_data) > 0:
            iter_points_plot.set_data(iter_data[:, 0], iter_data[:, 1])
        else:
            iter_points_plot.set_data([], [])

    # Get current best point
    if state.get("best_param") is not None:
        best_json = state["best_param"]
        best = np.array(best_json["data"]).reshape(*best_json["dim"])
        best_plot.set_data([best[0]], [best[1]])

    # Update trust region
    tr_bounds = extract_trust_region_bounds(state, bounds, DMAX)
    if tr_bounds is not None:
        trust_region_patch.set_xy((tr_bounds[0, 0], tr_bounds[1, 0]))
        trust_region_patch.set_width(tr_bounds[0, 1] - tr_bounds[0, 0])
        trust_region_patch.set_height(tr_bounds[1, 1] - tr_bounds[1, 0])
        trust_region_patch.set_visible(True)
        ego_point.set_visible(False)
    else:
        trust_region_patch.set_visible(False)
        # Show ego point instead (current x)
        if state.get("param") is not None:
            param_json = state["param"]
            current_x = np.array(param_json["data"]).reshape(*param_json["dim"])
            ego_point.center = (current_x[0], current_x[1])
            ego_point.set_radius(0.1)  # Fixed radius for visibility
            ego_point.set_visible(True)

    # Update iteration number in title
    best_cost_val = 0.0
    if state.get("best_cost") is not None:
        best_json = state["best_cost"]
        best_cost = np.array(best_json["data"]).reshape(*best_json["dim"])
        best_cost_val = best_cost[0]

    # print(
    #     f"Frame {frame}: Iteration {state.get('iter', frame)} - Best cost: {best_cost_val:.4f}"
    # )

    title.set_text(
        f"Iteration {state.get('iter', frame) + 1} - Best: {best_cost_val:.4f}"
    )
    return (
        doe_points_plot,
        iter_points_plot,
        best_plot,
        trust_region_patch,
        ego_point,
        title,
    )


# Create animation
anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(states),
    interval=500,
    blit=False,
    repeat=True,
)

# Save animation
anim_file = os.path.join(outdir, "rosenbrock_trego_animation.gif")
print(f"Saving animation to {anim_file}...")
anim.save(anim_file, writer="pillow", fps=2)
print("Animation saved!")

plt.tight_layout()
plt.show()

print(f"\nVisualization complete! Files saved in {outdir}/")

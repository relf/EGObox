# =====================================================
# Egobox demo: Minimize Rastrigin with Egor
# =====================================================

import numpy as np
import egobox as egx
import matplotlib.pyplot as plt


# -----------------------------------------------------
# Define the 2D Rastrigin function
# -----------------------------------------------------
def rastrigin(x: np.ndarray) -> np.ndarray:
    """Rastrigin function."""
    A = 10
    x = np.atleast_2d(x)
    return A * x.shape[1] + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1).reshape(
        -1, 1
    )


# -----------------------------------------------------
# Define search space
# -----------------------------------------------------
dim = 2
bounds = [[-5.12, 5.12]] * dim

# -----------------------------------------------------
# Available infill criterion
# -----------------------------------------------------
# Choose one of the following infill strategies: EI, LOG_EI, WB2, WB2S
criterion = egx.InfillStrategy.LOG_EI
print(f"Using infill strategy: {criterion}")

# -----------------------------------------------------
# Initialize optimizer
# -----------------------------------------------------
opt = egx.Egor(bounds, n_doe=20, infill_strategy=criterion)

# -----------------------------------------------------
# Run optimization
# -----------------------------------------------------
res = opt.minimize(rastrigin, max_iters=80, seed=42)

print("\n===== Optimization Result =====")
print("Best value (y*):", res.y_opt)
print("Best point (x*):", res.x_opt)

# -----------------------------------------------------
# Plot Rastrigin function and sample points
# -----------------------------------------------------
X = np.linspace(bounds[0][0], bounds[0][1], 200)
Y = np.linspace(bounds[1][0], bounds[1][1], 200)
XX, YY = np.meshgrid(X, Y)
XY = np.column_stack([XX.ravel(), YY.ravel()])
ZZ = rastrigin(XY).reshape(XX.shape)

plt.figure(figsize=(7, 6))
plt.contourf(XX, YY, ZZ, levels=50, cmap="viridis")
plt.colorbar(label="Rastrigin value")

# Plot training samples
X_data = np.array(res.x_doe)
plt.scatter(X_data[:, 0], X_data[:, 1], c="red", s=30, label="Sampled points")

plt.scatter(
    res.x_opt[0], res.x_opt[1], c="white", s=100, edgecolors="black", label="Best point"
)
plt.title("Rastrigin Function + Sampled Points")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.show()

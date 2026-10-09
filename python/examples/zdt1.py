# =====================================================
# Egobox demo: Multi-objective optimization of ZDT1 with Belfegor
# =====================================================

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import egobox as egx

# Comment out the following line to display the plot in a window
matplotlib.use("Agg")


# -----------------------------------------------------
# ZDT1 bi-objective test function with 2 variables in [0, 1]:
# the Pareto front is f2 = 1 - sqrt(f1), reached for x2 = 0
# -----------------------------------------------------
def zdt1(x: np.ndarray) -> np.ndarray:
    f1 = x[:, 0]
    g = 1.0 + 9.0 * x[:, 1]
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.column_stack([f1, f2])


# -----------------------------------------------------
# Approximate the Pareto front: the multi-objective strategy defaults to
# EHVI for 2 or 3 objectives, here batches of 3 points are selected with qEHVI
# -----------------------------------------------------
belfegor = egx.Belfegor(
    [[0.0, 1.0], [0.0, 1.0]],
    n_obj=2,
    moo_config=egx.MooConfig(strategy=egx.MooStrategy.QEHVI, batch=3),
    n_doe=10,
    seed=42,
)
res = belfegor.minimize(zdt1, max_iters=10)

print(f"Pareto front of {len(res.y_pareto)} points among {len(res.y_doe)} evaluations")
print(f"Compromise point f={res.y_opt} at x={res.x_opt}")
print(f"Status {res.status.exit} in {res.status.elapsed_time}s")

# -----------------------------------------------------
# Plot the front approximation against the true front
# -----------------------------------------------------
f1 = np.linspace(0.0, 1.0, 100)
plt.plot(f1, 1.0 - np.sqrt(f1), "k--", label="true front")
plt.plot(res.y_doe[:, 0], res.y_doe[:, 1], ".", color="grey", label="evaluations")
plt.plot(res.y_pareto[:, 0], res.y_pareto[:, 1], "o", label="Pareto front")
plt.plot(res.y_opt[0], res.y_opt[1], "r*", markersize=12, label="compromise")
plt.xlabel("f1")
plt.ylabel("f2")
plt.legend()
plt.title("ZDT1 optimized with Belfegor")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import egobox as egx

# Parameters
n_samples = 100
xlimits = np.array([[0.0, 1.0], [0.0, 1.0]])  # 2D bounds

# Available sampling methods in egobox
sampling_methods = {
    "Full Factorial": egx.Sampling.FULL_FACTORIAL,
    "Random": egx.Sampling.RANDOM,
    "LHS Classic": egx.Sampling.LHS_CLASSIC,
    "LHS Centered": egx.Sampling.LHS_CENTERED,
    "LHS Maximin": egx.Sampling.LHS_MAXIMIN,
    "LHS ESE": egx.Sampling.LHS,
}

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes = axes.flatten()

plt.suptitle(
    "Comparison of Egobox Sampling Methods in 2D", fontsize=16, fontweight="bold"
)

# Generate and plot samples for each method
for idx, method in enumerate(sampling_methods.items()):
    print(f"Generating samples using: {method}")
    # Generate samples using the sampling function
    samples = egx.sampling(method[1], xlimits, n_samples)

    # Plot
    ax = axes[idx]
    ax.scatter(
        samples[:, 0], samples[:, 1], alpha=0.6, s=50, edgecolors="black", linewidth=0.5
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(
        f"{method[0]}\n({len(samples)} samples)", fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()

# =====================================================
# Egobox demo: Minimize Rosenbrock with real-time observer plot
# Shows y_opt (best objective) vs iteration during optimization
# =====================================================

import time

import matplotlib.pyplot as plt
import numpy as np

import egobox as egx


# -----------------------------------------------------
# Delayed Rosenbrock function
# -----------------------------------------------------
def rosenbrock_delayed(x: np.ndarray, delay: float = 0.3) -> np.ndarray:
    """
    Rosenbrock function with artificial delay for visualization.
    
    Global minimum at x = [1, 1, ..., 1] with f(x) = 0.
    
    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n, dim) or (dim,)
    delay : float
        Artificial delay in seconds to slow down optimization for visualization
        
    Returns
    -------
    np.ndarray
        Function values of shape (n, 1)
    """
    # Add delay to simulate expensive function evaluation
    time.sleep(delay)
    
    x = np.atleast_2d(x)
    val = np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2, axis=1
    )
    return val.reshape(-1, 1)


# -----------------------------------------------------
# Observer class for real-time plotting
# -----------------------------------------------------
class RosenbrockObserver:
    """
    Observer callback that displays real-time optimization progress.
    
    Maintains a matplotlib figure that updates after each iteration,
    showing the best objective value (y_opt) vs iteration number.
    """
    
    def __init__(self, dim: int = 2, title: str = "Rosenbrock Optimization"):
        """
        Initialize the observer.
        
        Parameters
        ----------
        dim : int
            Dimension of the Rosenbrock function (for display purposes)
        title : str
            Title for the plot
        """
        self.iterations = []
        self.y_optima = []
        self.dim = dim
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel("Iteration", fontsize=12)
        self.ax.set_ylabel("Best objective value (y_opt)", fontsize=12)
        self.ax.set_title(f"{title} (dim={dim})", fontsize=14)
        self.ax.set_yscale("log")  # Log scale for better convergence visualization
        self.ax.grid(True, which="both", alpha=0.3, linestyle="--")
        
        # Initialize empty line
        self.line, = self.ax.plot([], [], "b-o", markersize=5, linewidth=2, label="y_opt")
        self.ax.legend()
        
        # Enable interactive mode
        plt.ion()
        plt.show()
        
        print(f"Observer initialized. Press Ctrl+C to stop optimization early.")
        print("-" * 60)
    
    def __call__(self, state: egx.EgorObservableState):
        """
        Called after each optimization iteration.
        
        Parameters
        ----------
        state : egx.EgorObservableState
            Current optimization state containing iter, x_opt, y_opt
        """
        # Store data
        self.iterations.append(state.iter)
        y_val = state.y_opt[0]
        self.y_optima.append(y_val)
        
        # Update plot
        self.line.set_data(self.iterations, self.y_optima)
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Add annotation with current best value
        if len(self.iterations) > 1:
            # Remove old annotation if exists
            for artist in self.ax.texts:
                artist.remove()
            # Add new annotation
            self.ax.annotate(
                f"y_opt = {y_val:.4e}",
                xy=(state.iter, y_val),
                xytext=(state.iter, y_val * 2),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
                color="red",
            )
        
        # Refresh the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Print progress to console
        print(f"Iter {state.iter:3d}: y_opt = {y_val:.6e}, x_opt = [{', '.join(f'{v:.4f}' for v in state.x_opt[:3])}{'...' if len(state.x_opt) > 3 else ''}]")
    
    def finalize(self):
        """Called when optimization completes."""
        plt.ioff()
        print("-" * 60)
        print(f"Optimization complete. Final y_opt = {self.y_optima[-1]:.6e}")
        plt.show()  # Keep figure open


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    """Run the Rosenbrock optimization with real-time observer plot."""
    # Problem configuration
    dim = 2
    bounds = [[-2.0, 2.0]] * dim
    n_doe = 5  # Initial design of experiments
    max_iters = 70  # Number of optimization iterations
    delay = 0.01  # Delay in seconds per function evaluation
    
    print("=" * 60)
    print(f"Rosenbrock Optimization Demo (dim={dim})")
    print(f"Initial DOE: {n_doe} points")
    print(f"Max iterations: {max_iters}")
    print(f"Evaluation delay: {delay}s")
    print("=" * 60)
    
    # Create observer for real-time visualization
    observer = RosenbrockObserver(dim=dim)
    
    # Configure optimizer
    opt = egx.Egor(
        bounds,
        n_doe=n_doe,
        infill_strategy=egx.InfillStrategy.LOG_EI,
        gp_config=egx.GpConfig(kpls_dim=1, corr_spec=egx.CorrelationSpec.MATERN52),
        trego=True,
    )
    
    # Run optimization with observer
    optim = opt.minimize(
        lambda x: rosenbrock_delayed(x, delay=delay),
        max_iters=max_iters,
        observers=[observer],
        verbose=egx.Verbose.ERROR,  # Minimal verbose from optimizer
        seed=42,
    )
    
    # Finalize observer (close figure properly)
    observer.finalize()
    
    # Print final results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Best y_opt: {optim.result.y_opt[0]:.6e}")
    print(f"Best x_opt: {optim.result.x_opt}")
    print(f"Total DOE points: {optim.result.x_doe.shape[0]}")
    print(f"Exit status: {optim.status.exit}")
    print("=" * 60)


if __name__ == "__main__":
    main()
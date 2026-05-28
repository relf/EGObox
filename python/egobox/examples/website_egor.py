import numpy as np

import egobox as egx


def objective(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return (x - 3.5) * np.sin((x - 3.5) / np.pi)


def optimize_example() -> egx.EgorOptim:
    return egx.Egor([[0.0, 25.0]]).minimize(objective, max_iters=20, seed=42)


def main() -> None:
    optim = optimize_example()
    print(f"Optimization f={optim.result.y_opt} at {optim.result.x_opt}")
    print(f"Status {optim.status.exit} in {optim.status.elapsed_time}s")


if __name__ == "__main__":
    main()

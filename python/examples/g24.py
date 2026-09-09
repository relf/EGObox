# To display optimization information (none by default)
import argparse
import logging

import numpy as np

import egobox as egx

logging.basicConfig(level=logging.INFO)

xspecs_g24 = [[0.0, 3.0], [0.0, 4.0]]
n_cstr_g24 = 2


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="G24 constrained optimization example")
    parser.add_argument(
        "--infill-opt",
        type=str,
        default="COBYLA",
        choices=["COBYLA", "SLSQP", "IPOPT"],
        help="Infill optimizer to use (default: COBYLA)",
    )
    return parser.parse_args()


# Objective
def G24(point):
    """
    Function g24
    1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
    """
    p = np.atleast_2d(point)
    return -p[:, 0] - p[:, 1]


# Constraints < 0
def G24_c1(point):
    p = np.atleast_2d(point)
    return (
        -2.0 * p[:, 0] ** 4.0
        + 8.0 * p[:, 0] ** 3.0
        - 8.0 * p[:, 0] ** 2.0
        + p[:, 1]
        - 2.0
    )


def G24_c2(point):
    p = np.atleast_2d(point)
    return (
        -4.0 * p[:, 0] ** 4.0
        + 32.0 * p[:, 0] ** 3.0
        - 88.0 * p[:, 0] ** 2.0
        + 96.0 * p[:, 0]
        + p[:, 1]
        - 36.0
    )


# Grouped evaluation
def g24(point):
    p = np.atleast_2d(point)
    return np.array([G24(p), G24_c1(p), G24_c2(p)]).T


def main():
    args = parse_args()

    # Map string argument to InfillOptimizer enum
    infill_optimizer_map = {
        "COBYLA": egx.InfillOptimizer.COBYLA,
        "SLSQP": egx.InfillOptimizer.SLSQP,
        "IPOPT": egx.InfillOptimizer.IPOPT,
    }
    infill_optimizer = infill_optimizer_map[args.infill_opt]

    # Configure the optimizer. See help(egor) for options
    egor = egx.Egor(
        xspecs_g24,
        n_doe=10,
        n_cstr=n_cstr_g24,
        cstr_tol=[1e-3] * n_cstr_g24,
        infill_strategy=egx.InfillStrategy.WB2,
        infill_optimizer=infill_optimizer,
        target=-5.50,  # known reference objective value
    )

    optim = egor.minimize(g24, max_iters=30)
    print(
        f"Optimization f={optim.result.y_opt} at {optim.result.x_opt} "
        f"(using {args.infill_opt} infill optimizer)"
    )


if __name__ == "__main__":
    main()

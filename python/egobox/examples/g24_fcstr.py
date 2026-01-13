import numpy as np
import egobox as egx

# To display optimization information (none by default)
import logging

logging.basicConfig(level=logging.INFO)

xspecs_g24 = [[0.0, 3.0], [0.0, 4.0]]


# Objective
def G24(point):
    """
    Function g24
    1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
    """
    p = np.atleast_2d(point)
    return -p[:, 0] - p[:, 1]


# Constraints < 0
def G24_c1(point, gradient=False):
    if gradient:
        raise NotImplementedError("G24_c1: Gradient not implemented")
    else:
        p = np.atleast_2d(point)
        return (
            -2.0 * p[:, 0] ** 4.0
            + 8.0 * p[:, 0] ** 3.0
            - 8.0 * p[:, 0] ** 2.0
            + p[:, 1]
            - 2.0
        )


def G24_c2(point, gradient=False):
    if gradient:
        raise NotImplementedError("G24_c2: Gradient not implemented")
    else:
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
    return np.array([G24(p)]).T


# Fonction constraints
fcstrs = [G24_c1, G24_c2]

# Configure the optimizer. See help(egor) for options
egor = egx.Egor(
    xspecs_g24,
    n_doe=10,
    cstr_tol=[1e-3] * len(fcstrs),  # Tolerance for function constraints
    infill_strategy=egx.InfillStrategy.WB2,
    target=-5.50,  # known reference objective value
)

res = egor.minimize(g24, max_iters=30, fcstrs=fcstrs)
print(f"Optimization f={res.y_opt} at {res.x_opt}")

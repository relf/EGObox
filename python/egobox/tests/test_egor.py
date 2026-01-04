import os
import unittest
import numpy as np
import egobox as egx
import time
import logging

logging.basicConfig(level=logging.INFO)


def sphere(x: np.ndarray) -> np.ndarray:
    """
    Sphere function
    Global optimum at x_opt = 0 with f_opt = 0
    """
    x = np.atleast_2d(x)
    y = np.sum(x**2, axis=1).reshape(-1, 1)
    print(f"obj={y} at {x}")
    return y


def xsinx(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    print(f"obj={y} at {x}")
    return y


# grad information is used when using SLSQP as infill optimizer
def cstr_xsinx(x, grad=False):
    if grad:
        return np.one(1.0)
    else:
        return (x - 18.0).item()


def G24(point):
    """
    Function G24
    1 global optimum y_opt = -5.5080 at x_opt =(2.3295, 3.1785)
    """
    p = np.atleast_2d(point)
    return -p[:, 0] - p[:, 1]


# Constraints < 0 to be metamodelized
def G24_c1(point):
    p = np.atleast_2d(point)
    return (
        -2.0 * p[:, 0] ** 4.0
        + 8.0 * p[:, 0] ** 3.0
        - 8.0 * p[:, 0] ** 2.0
        + p[:, 1]
        - 2.0
    )


# Used as fonction constraint
def g24_c1(x, grad=False):
    if grad:
        raise NotImplementedError("g24_c1: constraint gradient not available")
    return G24_c1(x).item()


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


# Used as fonction constraint
def g24_c2(x, grad=False):
    if grad:
        raise NotImplementedError("g24_c1: constraint gradient not available")
    return G24_c2(x).item()


# Grouped evaluation
def g24(point):
    p = np.atleast_2d(point)
    res = np.array([G24(p), G24_c1(p), G24_c2(p)]).T
    print(f"y={res}")
    return res


def g24_bare(point):
    p = np.atleast_2d(point)
    res = np.array([G24(p)]).T
    print(f"y={res}")
    return res


def six_humps(x):
    """
    Function Six-Hump Camel Back
    2 global optimum value =-1.0316 located at (0.089842, -0.712656) and  (-0.089842, 0.712656)
    https://www.sfu.ca/~ssurjano/camel6.html
    """
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    print(f"x={x}")
    sum1 = 4 * x1**2 - 2.1 * x1**4 + 1.0 / 3.0 * x1**6 + x1 * x2 - 4 * x2**2 + 4 * x2**4
    print(f"y={np.atleast_2d(sum1).T}")
    return np.atleast_2d(sum1).T


class TestEgor(unittest.TestCase):
    def test_sphere(self):
        dim = 5
        egor = egx.Egor(
            np.array([[-5.12, 5.12]] * dim),  # test ndarray API
            infill_strategy=egx.InfillStrategy.EI,
            seed=42,
        )
        res = egor.minimize(sphere, max_iters=100)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(0.0, res.y_opt[0], delta=5e-1)
        np.testing.assert_allclose(0.0, res.x_opt, atol=5e-1)

    def test_xsinx(self):
        egor = egx.Egor([[0.0, 25.0]], seed=42)  # test list of list api
        res = egor.minimize(xsinx, max_iters=20)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-2)

    def test_xsinx_with_reclustering(self):
        egor = egx.Egor([[0.0, 25.0]], seed=42, gp_config=egx.GpConfig(n_clusters=0))
        res = egor.minimize(xsinx, max_iters=20)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-2)

    def test_xsinx_with_warmstart(self):
        if os.path.exists("./test_dir/egor_initial_doe.npy"):
            os.remove("./test_dir/egor_initial_doe.npy")
        if os.path.exists("./test_dir/egor_doe.npy"):
            os.remove("./test_dir/egor_doe.npy")
        xlimits = [[0.0, 25.0]]
        doe = egx.lhs(xlimits, 10)
        egor = egx.Egor(
            xlimits,
            doe=doe,
            infill_strategy=egx.InfillStrategy.WB2,
            seed=42,
            outdir="./test_dir",
        )
        res = egor.minimize(xsinx, max_iters=15)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-3)

        egor = egx.Egor(xlimits, outdir="./test_dir", warm_start=True)
        res = egor.minimize(xsinx, max_iters=5)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-2)

        self.assertTrue(os.path.exists("./test_dir/egor_initial_doe.npy"))
        os.remove("./test_dir/egor_initial_doe.npy")
        self.assertTrue(os.path.exists("./test_dir/egor_doe.npy"))
        os.remove("./test_dir/egor_doe.npy")

    def test_g24(self):
        n_doe = 5
        max_iters = 30
        n_cstr = 2
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            cstr_tol=np.array([1e-3, 1e-3]),
            n_cstr=n_cstr,
            seed=42,
            q_optmod=2,
            n_doe=n_doe,
            cstr_strategy=egx.ConstraintStrategy.UTB,
        )
        start = time.process_time()
        res = egor.minimize(g24, max_iters=max_iters)
        end = time.process_time()
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end - start}s")
        self.assertAlmostEqual(-5.5080, res.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, res.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, res.x_opt[1], delta=1e-2)
        self.assertGreaterEqual(n_doe + max_iters, res.x_doe.shape[0])
        self.assertEqual(1 + n_cstr, res.y_doe.shape[1])

    def test_g24_kpls(self):
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            infill_strategy=egx.InfillStrategy.WB2,
            n_cstr=2,
            cstr_tol=np.array([5e-3, 5e-3]),
            gp_config=egx.GpConfig(
                regr_spec=egx.RegressionSpec.CONSTANT,
                corr_spec=egx.CorrelationSpec.SQUARED_EXPONENTIAL,
                kpls_dim=1,
            ),
            seed=1,
        )
        start = time.process_time()
        res = egor.minimize(g24, max_iters=30)
        end = time.process_time()
        self.assertAlmostEqual(-5.5080, res.y_opt[0], delta=5e-1)
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end - start}s")

    def test_g24_trego(self):
        n_doe = 5
        max_iters = 40
        n_cstr = 2
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            cstr_tol=np.array([1e-3, 1e-3]),
            n_cstr=n_cstr,
            seed=42,
            n_doe=n_doe,
            trego=egx.TregoConfig(),
        )
        start = time.process_time()
        res = egor.minimize(g24, max_iters=max_iters)
        end = time.process_time()
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end - start}s")
        self.assertAlmostEqual(-5.5080, res.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, res.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, res.x_opt[1], delta=1e-2)

    def test_six_humps(self):
        egor = egx.Egor(
            [[-3.0, 3.0], [-2.0, 2.0]],
            infill_strategy=egx.InfillStrategy.WB2,
            seed=42,
        )
        start = time.process_time()
        res = egor.minimize(six_humps, max_iters=45)
        end = time.process_time()
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end - start}s")
        # 2 global optimum value =-1.0316 located at (0.089842, -0.712656) and  (-0.089842, 0.712656)
        self.assertAlmostEqual(-1.0316, res.y_opt[0], delta=2e-1)

    def test_constructor(self):
        self.assertRaises(TypeError, egx.Egor)
        egx.Egor([[0.0, 25.0]], n_doe=10)

    def test_egor_service(self):
        xlimits = [[0.0, 25.0]]
        egor = egx.Egor(xlimits, infill_strategy=egx.InfillStrategy.WB2, seed=42)
        x_doe = egx.lhs(xlimits, 3, seed=42)
        y_doe = xsinx(x_doe)
        for _ in range(10):
            x = egor.suggest(x_doe, y_doe)
            x_doe = np.concatenate((x_doe, x))
            y_doe = np.concatenate((y_doe, xsinx(x)))
        res = egor.get_result(x_doe, y_doe)
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, res.x_opt[0], delta=1e-3)

    # Constraint function which prevent from reaching the
    # the unconstrained minimum located in x=18.9
    def test_egor_with_fcstrs(self):
        fcstrs = [cstr_xsinx]
        egor = egx.Egor(
            [[0.0, 25.0]], infill_strategy=egx.InfillStrategy.WB2, n_doe=5, seed=42
        )
        res = egor.minimize(xsinx, max_iters=20, fcstrs=fcstrs)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(18, res.x_opt[0], delta=2e-3)

    def test_g24_with_fcstrs(self):
        n_doe = 5
        max_iters = 20
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            seed=42,
            q_optmod=2,
            n_doe=n_doe,
        )
        start = time.process_time()
        fcstrs = [g24_c1, g24_c2]
        res = egor.minimize(g24_bare, max_iters=max_iters, fcstrs=fcstrs)
        end = time.process_time()
        print(f"Optimization f={res.y_opt} at {res.x_opt} in {end - start}s")
        self.assertAlmostEqual(-5.5080, res.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, res.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, res.x_opt[1], delta=1e-2)
        self.assertEqual((n_doe + max_iters, 2), res.x_doe.shape)
        self.assertEqual((n_doe + max_iters, 1), res.y_doe.shape)


if __name__ == "__main__":
    unittest.main(defaultTest=["TestOptimizer.test_g24_trego"], exit=False)

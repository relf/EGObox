import os
import unittest
import numpy as np
import egobox as egx
import time
import logging
import tempfile

logging.basicConfig(level=logging.INFO)


def sphere(x: np.ndarray) -> np.ndarray:
    """
    Sphere function
    Global optimum at x_opt = 0 with f_opt = 0
    """
    x = np.atleast_2d(x)
    y = np.sum(x**2, axis=1).reshape(-1, 1)
    return y


def xsinx(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
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


def branin(x):
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


BRANIN_CSTR_CONST = 0.2


def branin_constrained(x):
    """
    Branin function with a constraint y*y >= 0.2
    """
    x = np.atleast_2d(x)
    obj = branin(x)
    cstr = obj * obj
    return np.hstack((obj, cstr.reshape(-1, 1)))


class TestEgor(unittest.TestCase):
    def test_sphere(self):
        dim = 5
        egor = egx.Egor(
            np.array([[-5.12, 5.12]] * dim),  # test ndarray API
            infill_strategy=egx.InfillStrategy.EI,
        )
        optim = egor.minimize(sphere, max_iters=100, seed=42)
        print(f"Optimization f={optim.result.y_opt} at {optim.result.x_opt}")
        self.assertAlmostEqual(0.0, optim.result.y_opt[0], delta=5e-1)
        np.testing.assert_allclose(0.0, optim.result.x_opt, atol=5e-1)

    def test_xsinx(self):
        egor = egx.Egor([[0.0, 25.0]])  # test list of list api
        optim = egor.minimize(xsinx, max_iters=20, seed=42)
        print(f"Optimization f={optim.result.y_opt} at {optim.result.x_opt}")
        self.assertAlmostEqual(-15.125, optim.result.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, optim.result.x_opt[0], delta=5e-2)

    def test_xsinx_with_reclustering(self):
        egor = egx.Egor([[0.0, 25.0]], gp_config=egx.GpConfig(n_clusters=0))
        optim = egor.minimize(xsinx, max_iters=20, seed=42)
        print(f"Optimization f={optim.result.y_opt} at {optim.result.x_opt}")
        self.assertAlmostEqual(-15.125, optim.result.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, optim.result.x_opt[0], delta=5e-2)

    def test_xsinx_gp_as_rbf(self):
        theta_init = [3.14]  # fixed theta value
        outdir = "./test_dir"
        egor = egx.Egor(
            [[0.0, 25.0]],
            gp_config=egx.GpConfig(
                theta_init=theta_init, n_start=0
            ),  # no hyperparameter optimization
        )  # test list of list api
        _ = egor.minimize(
            xsinx, max_iters=5, verbose=egx.Verbose.INFO, outdir=outdir, seed=42
        )
        gps = egor.load_gp_models(os.path.join(outdir, "egor_gp.bin"))
        self.assertEqual(gps[0].thetas().item(), theta_init[0])
        gps = egor.load_gp_models(os.path.join(outdir, "egor_initial_gp.bin"))
        self.assertEqual(gps[0].thetas().item(), theta_init[0])

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
        )
        optim = egor.minimize(xsinx, max_iters=15, outdir="./test_dir", seed=42)
        print(f"Optimization f={optim.result.y_opt} at {optim.result.x_opt}")
        self.assertAlmostEqual(-15.125, optim.result.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, optim.result.x_opt[0], delta=1e-3)

        egor = egx.Egor(xlimits)
        optim = egor.minimize(xsinx, max_iters=5, outdir="./test_dir", warm_start=True)
        print(f"Optimization f={optim.result.y_opt} at {optim.result.x_opt}")
        self.assertAlmostEqual(-15.125, optim.result.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(18.935, optim.result.x_opt[0], delta=1e-2)

        self.assertTrue(os.path.exists("./test_dir/egor_initial_doe.npy"))
        os.remove("./test_dir/egor_initial_doe.npy")
        self.assertTrue(os.path.exists("./test_dir/egor_doe.npy"))
        os.remove("./test_dir/egor_doe.npy")

    def test_xsinx_with_hotstart_bool(self):
        xlimits = [[0.0, 25.0]]

        with tempfile.TemporaryDirectory() as outdir:
            egor = egx.Egor(xlimits)
            first = egor.minimize(
                xsinx, max_iters=1, outdir=outdir, hot_start=True, seed=42
            )
            second = egor.minimize(
                xsinx, max_iters=1, outdir=outdir, hot_start=True, seed=42
            )

            self.assertTrue(
                os.path.exists(os.path.join(outdir, "egor_checkpoint.json"))
            )
            self.assertEqual(first.result.x_doe.shape[0], second.result.x_doe.shape[0])

    def test_g24(self):
        n_doe = 5
        max_iters = 30
        n_cstr = 2
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            cstr_tol=np.array([1e-3, 1e-3]),
            n_cstr=n_cstr,
            n_doe=n_doe,
            cstr_strategy=egx.ConstraintStrategy.UTB,
        )
        start = time.process_time()
        optim = egor.minimize(
            g24, max_iters=max_iters, verbose=egx.Verbose.INFO, seed=42
        )
        end = time.process_time()
        print(
            f"Optimization f={optim.result.y_opt} at {optim.result.x_opt} in {end - start}s"
        )
        self.assertAlmostEqual(-5.5080, optim.result.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, optim.result.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, optim.result.x_opt[1], delta=1e-2)
        self.assertGreaterEqual(n_doe + max_iters, optim.result.x_doe.shape[0])
        self.assertEqual(1 + n_cstr, optim.result.y_doe.shape[1])

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
        )
        start = time.process_time()
        optim = egor.minimize(g24, max_iters=30, verbose=2, seed=1)
        end = time.process_time()
        self.assertAlmostEqual(-5.5080, optim.result.y_opt[0], delta=5e-1)
        print(
            f"Optimization f={optim.result.y_opt} at {optim.result.x_opt} in {end - start}s"
        )

    def test_g24_trego(self):
        n_doe = 5
        max_iters = 20
        n_cstr = 2
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            cstr_tol=np.array([1e-3, 1e-3]),
            n_cstr=n_cstr,
            n_doe=n_doe,
            trego=egx.TregoConfig((4, 1)),
        )
        start = time.process_time()
        optim = egor.minimize(g24, max_iters=max_iters, seed=42)
        end = time.process_time()
        print(
            f"Optimization f={optim.result.y_opt} at {optim.result.x_opt} in {end - start}s"
        )
        self.assertAlmostEqual(-5.5080, optim.result.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, optim.result.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, optim.result.x_opt[1], delta=1e-2)

        # Test with default TREGO parameters
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            cstr_tol=np.array([1e-3, 1e-3]),
            n_cstr=n_cstr,
            n_doe=n_doe,
            trego=True,
        )
        optim = egor.minimize(g24, max_iters=max_iters, seed=42)
        self.assertAlmostEqual(-5.5080, optim.result.y_opt[0], delta=1e-2)

    def test_six_humps(self):
        egor = egx.Egor(
            [[-3.0, 3.0], [-2.0, 2.0]],
            infill_strategy=egx.InfillStrategy.WB2,
        )
        start = time.process_time()
        optim = egor.minimize(six_humps, max_iters=45, seed=42)
        end = time.process_time()
        print(
            f"Optimization f={optim.result.y_opt} at {optim.result.x_opt} in {end - start}s"
        )
        # 2 global optimum value =-1.0316 located at (0.089842, -0.712656) and  (-0.089842, 0.712656)
        self.assertAlmostEqual(-1.0316, optim.result.y_opt[0], delta=2e-1)

    def test_constructor(self):
        self.assertRaises(TypeError, egx.Egor)
        egx.Egor([[0.0, 25.0]], n_doe=10)

    def test_egor_service(self):
        xlimits = [[0.0, 25.0]]
        egor = egx.Egor(xlimits, infill_strategy=egx.InfillStrategy.WB2)
        x_doe = egx.lhs(xlimits, 3, seed=42)
        y_doe = xsinx(x_doe)
        for _ in range(10):
            x = egor.suggest(x_doe, y_doe, seed=42)
            x_doe = np.concatenate((x_doe, x))
            y_doe = np.concatenate((y_doe, xsinx(x)))
        result = egor.get_result(x_doe, y_doe)
        self.assertAlmostEqual(-15.125, result.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, result.x_opt[0], delta=1e-3)

    # Constraint function which prevent from reaching the
    # the unconstrained minimum located in x=18.9
    def test_egor_with_fcstrs(self):
        fcstrs = [cstr_xsinx]
        egor = egx.Egor([[0.0, 25.0]], infill_strategy=egx.InfillStrategy.WB2, n_doe=5)
        optim = egor.minimize(xsinx, max_iters=20, fcstrs=fcstrs, seed=42)
        print(f"Optimization f={optim.result.y_opt} at {optim.result.x_opt}")
        self.assertAlmostEqual(18, optim.result.x_opt[0], delta=2e-3)

    def test_g24_with_fcstrs(self):
        n_doe = 5
        max_iters = 5
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            n_doe=n_doe,
        )
        start = time.process_time()
        fcstrs = [g24_c1, g24_c2]
        optim = egor.minimize(g24_bare, max_iters=max_iters, fcstrs=fcstrs, seed=42)
        end = time.process_time()
        print(
            f"Optimization f={optim.result.y_opt} at {optim.result.x_opt} in {end - start}s"
        )
        self.assertAlmostEqual(-5.5080, optim.result.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, optim.result.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, optim.result.x_opt[1], delta=1e-2)
        self.assertEqual((n_doe + max_iters, 2), optim.result.x_doe.shape)
        self.assertEqual((n_doe + max_iters, 1), optim.result.y_doe.shape)

    def test_g24_with_fcstrs_and_specs(self):
        n_doe = 5
        max_iters = 5
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            n_doe=n_doe,
        )
        start = time.process_time()
        fcstrs = [g24_c1, g24_c2]
        fcstr_specs = [egx.CstrSpec.leq(0.0), egx.CstrSpec.leq(0.0)]
        optim = egor.minimize(
            g24_bare,
            max_iters=max_iters,
            fcstrs=fcstrs,
            fcstr_specs=fcstr_specs,
            seed=42,
        )
        end = time.process_time()
        print(
            f"Optimization f={optim.result.y_opt} at {optim.result.x_opt} in {end - start}s"
        )
        self.assertAlmostEqual(-5.5080, optim.result.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, optim.result.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, optim.result.x_opt[1], delta=1e-2)

    def test_fcstr_specs_length_mismatch(self):
        egor = egx.Egor([[0.0, 3.0], [0.0, 4.0]], n_doe=5)
        fcstrs = [g24_c1, g24_c2]
        fcstr_specs = [egx.CstrSpec.leq(0.0)]
        with self.assertRaises(ValueError):
            _ = egor.minimize(
                g24_bare,
                max_iters=5,
                fcstrs=fcstrs,
                fcstr_specs=fcstr_specs,
                seed=42,
            )

    def test_g24_with_qei(self):
        n_doe = 5
        max_iters = 20
        n_cstr = 2
        egor = egx.Egor(
            [[0.0, 3.0], [0.0, 4.0]],
            cstr_tol=np.array([1e-3, 1e-3]),
            n_cstr=n_cstr,
            n_doe=n_doe,
            qei_config=egx.QEiConfig(batch=3, strategy=egx.QEiStrategy.KBLB, optmod=2),
        )
        start = time.process_time()
        optim = egor.minimize(g24, max_iters=max_iters, seed=42)
        end = time.process_time()
        print(
            f"Optimization f={optim.result.y_opt} at {optim.result.x_opt} in {end - start}s"
        )
        self.assertAlmostEqual(-5.5080, optim.result.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, optim.result.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, optim.result.x_opt[1], delta=1e-2)

    def test_g24_with_suggest(self):
        xlimits = [[0.0, 3.0], [0.0, 4.0]]
        egor = egx.Egor(
            xlimits,
            infill_strategy=egx.InfillStrategy.WB2,
            cstr_tol=np.array([1e-2, 1e-2]),
            n_cstr=2,
        )
        x_doe = egx.lhs(xlimits, 5, seed=42)
        y_doe = g24(x_doe)
        for _ in range(20):
            x = egor.suggest(x_doe, y_doe, seed=42)
            x_doe = np.concatenate((x_doe, x))
            y_doe = np.concatenate((y_doe, g24(x)))
        res_idx = egor.get_result_index(y_doe)
        result = egor.get_result(x_doe, y_doe)
        for xi, xii in zip(result.x_opt, x_doe[res_idx]):
            self.assertEqual(xi, xii)
        self.assertAlmostEqual(-5.5080, result.y_opt[0], delta=1e-2)
        self.assertAlmostEqual(2.3295, result.x_opt[0], delta=1e-2)
        self.assertAlmostEqual(3.1785, result.x_opt[1], delta=1e-2)

    def test_branin_constrained(self):
        xspecs = [[0.0, 1.0], [0.0, 1.0]]
        egor = egx.Egor(
            xspecs, cstr_specs=[egx.CstrSpec.geq(0.5)], n_doe=15, trego=True
        )
        optim = egor.minimize(
            branin_constrained,
            max_iters=70,
            seed=42,
            # verbose=egx.Verbose.INFO,
        )
        print(
            f"Optimum found at: x = {optim.result.x_opt}, f(x*) = {optim.result.y_opt[0]}"
        )
        self.assertAlmostEqual(0.707, optim.result.y_opt[0], delta=1e-1)

    def test_constrained_branin_with_nans(self):
        def branin_constrained_with_nans(x):
            def branin_constraint_nans(xi):
                if np.prod(xi) < BRANIN_CSTR_CONST:
                    return np.nan
                else:
                    return branin(np.atleast_2d(xi))[0, 0]

            res = np.apply_along_axis(branin_constraint_nans, 1, x).reshape(-1, 1)
            return res

        xspecs = [[0.0, 1.0], [0.0, 1.0]]
        egor = egx.Egor(
            xspecs,
            n_doe=15,
            failsafe_strategy=egx.FailsafeStrategy.IMPUTATION,
        )
        optim = egor.minimize(
            branin_constrained_with_nans,
            max_iters=30,
            seed=42,
        )
        print(
            f"Optimum found at: x = {optim.result.x_opt}, f(x*) = {optim.result.y_opt[0]}"
        )
        self.assertAlmostEqual(0.9677, optim.result.x_opt[0], delta=5e-2)
        self.assertAlmostEqual(0.2067, optim.result.x_opt[1], delta=6e-2)

    nb_calls = 0

    def test_fobj_crash(self):
        def fobj_crash(x):
            self.nb_calls = self.nb_calls + 1
            print(self.nb_calls)
            if self.nb_calls > 1 and np.prod(x) < BRANIN_CSTR_CONST:
                # Force a crash for points violating the constraint after the first call
                error = 1 / 0  # noqa
                print(error)  # to use the variable and avoid "unused variable" warning
            else:
                return branin(np.atleast_2d(x))

        xspecs = [[0.0, 1.0], [0.0, 1.0]]
        egor = egx.Egor(
            xspecs,
            n_doe=15,
            failsafe_strategy=egx.FailsafeStrategy.IMPUTATION,
        )
        optim = egor.minimize(
            fobj_crash,
            max_iters=30,
            seed=42,
        )
        print(
            f"Optimum found at: x = {optim.result.x_opt}, f(x*) = {optim.result.y_opt[0]}"
        )
        self.assertAlmostEqual(0.9677, optim.result.x_opt[0], delta=5e-2)
        self.assertAlmostEqual(0.2067, optim.result.x_opt[1], delta=6e-2)


if __name__ == "__main__":
    unittest.main(defaultTest=["TestEgor.test_fobj_crash"], exit=False)

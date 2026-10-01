"""Deprecated names still work with a DeprecationWarning until their removal after 0.38.

For each item: the old name warns and gives the same result as the new name,
giving both raises TypeError and the new name does not warn.
"""

import unittest
import warnings

import numpy as np

import egobox as egx


def xsinx(x: np.ndarray) -> np.ndarray:
    return (x - 3.5) * np.sin((x - 3.5) / np.pi)


class DeprecationTestCase(unittest.TestCase):
    def assertNoWarning(self, fn, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            return fn(*args, **kwargs)

    def assertDeprecated(self, old, new, fn, *args, **kwargs):
        with self.assertWarns(DeprecationWarning) as cm:
            res = fn(*args, **kwargs)
        msg = str(cm.warning)
        self.assertIn(old, msg)
        self.assertIn(new, msg)
        self.assertIn("deprecated since 0.38.0", msg)
        # warning points at the caller line, not inside egobox
        self.assertEqual(cm.filename, __file__)
        return res


class TestEgorOptimForwarding(unittest.TestCase):
    def setUp(self):
        egor = egx.Egor([[0.0, 25.0]], n_doe=5)
        self.res = egor.minimize(xsinx, max_iters=2, seed=42)

    def test_fields(self):
        r = self.res.result
        for name in ("x_opt", "y_opt", "x_doe", "y_doe"):
            np.testing.assert_array_equal(getattr(self.res, name), getattr(r, name))

    def test_unpacking(self):
        x_opt, y_opt = self.res
        np.testing.assert_array_equal(x_opt, self.res.result.x_opt)
        np.testing.assert_array_equal(y_opt, self.res.result.y_opt)

    def test_read_only(self):
        with self.assertRaises(AttributeError):
            self.res.x_opt = np.zeros(1)


class TestEgorNStart(DeprecationTestCase):
    def test_n_start(self):
        egor = self.assertDeprecated(
            "n_start", "infill_n_start", egx.Egor, [[0.0, 25.0]], n_start=5
        )
        self.assertIsInstance(egor, egx.Egor)

    def test_infill_n_start(self):
        self.assertNoWarning(egx.Egor, [[0.0, 25.0]], infill_n_start=5)

    def test_both(self):
        with self.assertRaises(TypeError):
            egx.Egor([[0.0, 25.0]], n_start=5, infill_n_start=5)

    def test_same_result(self):
        def run(**kwargs):
            egor = egx.Egor([[0.0, 25.0]], n_doe=5, **kwargs)
            return egor.minimize(xsinx, max_iters=3, seed=42).result

        new = run(infill_n_start=3)
        with self.assertWarns(DeprecationWarning):
            old = run(n_start=3)
        np.testing.assert_array_equal(new.x_doe, old.x_doe)


class TestGpConfigRenamed(DeprecationTestCase):
    def test_kwargs(self):
        cfg = self.assertDeprecated("n_start", "theta_n_start", egx.GpConfig, n_start=3)
        self.assertEqual(cfg.theta_n_start, 3)
        cfg = self.assertDeprecated(
            "max_eval", "theta_max_eval", egx.GpConfig, max_eval=30
        )
        self.assertEqual(cfg.theta_max_eval, 30)

    def test_new_kwargs(self):
        cfg = self.assertNoWarning(egx.GpConfig, theta_n_start=3, theta_max_eval=30)
        self.assertEqual((cfg.theta_n_start, cfg.theta_max_eval), (3, 30))
        cfg = self.assertNoWarning(egx.GpConfig)
        self.assertEqual((cfg.theta_n_start, cfg.theta_max_eval), (10, 50))

    def test_both_kwargs(self):
        with self.assertRaises(TypeError):
            egx.GpConfig(n_start=3, theta_n_start=3)
        with self.assertRaises(TypeError):
            egx.GpConfig(max_eval=30, theta_max_eval=30)

    def test_getters_setters(self):
        cfg = egx.GpConfig(theta_n_start=3, theta_max_eval=30)
        self.assertEqual(
            self.assertDeprecated("n_start", "theta_n_start", getattr, cfg, "n_start"),
            3,
        )
        self.assertEqual(
            self.assertDeprecated(
                "max_eval", "theta_max_eval", getattr, cfg, "max_eval"
            ),
            30,
        )
        with self.assertWarns(DeprecationWarning):
            cfg.n_start = 4
        with self.assertWarns(DeprecationWarning):
            cfg.max_eval = 40
        self.assertEqual((cfg.theta_n_start, cfg.theta_max_eval), (4, 40))

    def test_dict_keys(self):
        with self.assertWarns(DeprecationWarning):
            egor = egx.Egor([[0.0, 25.0]], gp_config={"n_start": 3, "max_eval": 30})
        self.assertIsInstance(egor, egx.Egor)
        self.assertNoWarning(
            egx.Egor,
            [[0.0, 25.0]],
            gp_config={"theta_n_start": 3, "theta_max_eval": 30},
        )
        with self.assertRaises(TypeError):
            egx.Egor([[0.0, 25.0]], gp_config={"n_start": 3, "theta_n_start": 3})


class TestGpMixRenamed(DeprecationTestCase):
    def setUp(self):
        self.xt = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
        self.yt = np.array([[0.0, 1.0, 1.5, 0.9, 1.0]]).T
        self.xtest = np.linspace(0, 4, 7).reshape(-1, 1)

    def test_gpmix(self):
        for builder in (egx.GpMix, egx.Gpx.builder):
            for old, new in (
                ("n_start", "theta_n_start"),
                ("max_eval", "theta_max_eval"),
            ):
                with self.subTest(builder=builder, old=old):
                    self.assertDeprecated(old, new, builder, **{old: 3})
                    self.assertNoWarning(builder, **{new: 3})
                    with self.assertRaises(TypeError):
                        builder(**{old: 3, new: 3})

    def test_gpmix_same_result(self):
        new = egx.GpMix(theta_n_start=2, theta_max_eval=30, seed=42)
        with self.assertWarns(DeprecationWarning):
            old = egx.GpMix(n_start=2, max_eval=30, seed=42)
        np.testing.assert_array_equal(
            new.fit(self.xt, self.yt).predict(self.xtest),
            old.fit(self.xt, self.yt).predict(self.xtest),
        )

    def test_sparse_gpmix(self):
        for builder in (egx.SparseGpMix, egx.SparseGpx.builder):
            with self.subTest(builder=builder):
                self.assertDeprecated(
                    "n_start", "theta_n_start", builder, n_start=3, n_inducing=2
                )
                self.assertNoWarning(builder, theta_n_start=3, n_inducing=2)
                with self.assertRaises(TypeError):
                    builder(n_start=3, theta_n_start=3, n_inducing=2)

    def test_sparse_gpmix_inducing(self):
        z = np.array([[1.0], [3.0]])
        for builder in (egx.SparseGpMix, egx.SparseGpx.builder):
            with self.subTest(builder=builder):
                self.assertDeprecated("nz", "n_inducing", builder, nz=2)
                self.assertDeprecated("z", "inducing", builder, z=z)
                self.assertNoWarning(builder, n_inducing=2)
                self.assertNoWarning(builder, inducing=z)
                with self.assertRaises(TypeError):
                    builder(nz=2, n_inducing=2)
                with self.assertRaises(TypeError):
                    builder(z=z, inducing=z)

    def test_sparse_gpmix_same_result(self):
        z = np.array([[1.0], [3.0]])
        new = egx.SparseGpMix(inducing=z, seed=42).fit(self.xt, self.yt)
        with self.assertWarns(DeprecationWarning):
            old = egx.SparseGpMix(z=z, seed=42).fit(self.xt, self.yt)
        np.testing.assert_array_equal(new.predict(self.xtest), old.predict(self.xtest))


class TestEgorBestResult(DeprecationTestCase):
    def setUp(self):
        self.egor = egx.Egor([[0.0, 25.0]])
        self.x_doe = np.array([[0.0], [7.0], [20.0], [25.0]])
        self.y_doe = xsinx(self.x_doe)

    def test_best_index(self):
        new = self.assertNoWarning(self.egor.best_index, self.y_doe)
        old = self.assertDeprecated(
            "get_result_index", "best_index", self.egor.get_result_index, self.y_doe
        )
        self.assertEqual(new, old)
        self.assertEqual(new, int(np.argmin(self.y_doe)))

    def test_best_result(self):
        new = self.assertNoWarning(self.egor.best_result, self.x_doe, self.y_doe)
        old = self.assertDeprecated(
            "get_result", "best_result", self.egor.get_result, self.x_doe, self.y_doe
        )
        np.testing.assert_array_equal(new.x_opt, old.x_opt)
        np.testing.assert_array_equal(new.y_opt, old.y_opt)


class TestCstrSpecBetween(DeprecationTestCase):
    def test_btw(self):
        old = self.assertDeprecated("btw", "between", egx.CstrSpec.btw, 1.0, 3.0)
        new = self.assertNoWarning(egx.CstrSpec.between, 1.0, 3.0)
        self.assertEqual(repr(old), repr(new))

    def test_dict_key(self):
        with self.assertWarns(DeprecationWarning):
            egx.Egor([[0.0, 25.0]], cstr_specs=[{"btw": (1.0, 3.0)}])
        self.assertNoWarning(
            egx.Egor, [[0.0, 25.0]], cstr_specs=[{"between": (1.0, 3.0)}]
        )
        with self.assertRaises((TypeError, ValueError)):
            egx.Egor(
                [[0.0, 25.0]], cstr_specs=[{"btw": (1.0, 3.0), "between": (1.0, 3.0)}]
            )


class TestQEiConfigRenamed(DeprecationTestCase):
    def test_kwarg(self):
        cfg = self.assertDeprecated("optmod", "optim_every", egx.QEiConfig, optmod=2)
        self.assertEqual(cfg.optim_every, 2)
        cfg = self.assertNoWarning(egx.QEiConfig, optim_every=2)
        self.assertEqual(cfg.optim_every, 2)
        with self.assertRaises(TypeError):
            egx.QEiConfig(optmod=2, optim_every=2)

    def test_getter_setter(self):
        cfg = egx.QEiConfig(optim_every=2)
        self.assertEqual(
            self.assertDeprecated("optmod", "optim_every", getattr, cfg, "optmod"), 2
        )
        with self.assertWarns(DeprecationWarning):
            cfg.optmod = 3
        self.assertEqual(cfg.optim_every, 3)

    def test_dict_key(self):
        with self.assertWarns(DeprecationWarning):
            egx.Egor([[0.0, 25.0]], qei_config={"optmod": 2})
        self.assertNoWarning(egx.Egor, [[0.0, 25.0]], qei_config={"optim_every": 2})
        with self.assertRaises(TypeError):
            egx.Egor([[0.0, 25.0]], qei_config={"optmod": 2, "optim_every": 2})


class TestTregoConfigRenamed(DeprecationTestCase):
    RENAMED = (
        ("n_gl_steps", "n_global_local_steps", (2, 3)),
        ("d", "radius_bounds", (1e-5, 0.5)),
    )

    def test_kwargs(self):
        for old, new, value in self.RENAMED:
            with self.subTest(old=old):
                cfg = self.assertDeprecated(old, new, egx.TregoConfig, **{old: value})
                self.assertEqual(getattr(cfg, new), value)
                cfg = self.assertNoWarning(egx.TregoConfig, **{new: value})
                self.assertEqual(getattr(cfg, new), value)
                with self.assertRaises(TypeError):
                    egx.TregoConfig(**{old: value, new: value})

    def test_getters_setters(self):
        for old, new, value in self.RENAMED:
            with self.subTest(old=old):
                cfg = egx.TregoConfig(**{new: value})
                self.assertEqual(
                    self.assertDeprecated(old, new, getattr, cfg, old), value
                )
                with self.assertWarns(DeprecationWarning):
                    setattr(cfg, old, value)

    def test_dict_keys(self):
        for old, new, value in self.RENAMED:
            with self.subTest(old=old):
                with self.assertWarns(DeprecationWarning):
                    egx.Egor([[0.0, 25.0]], trego={old: value})
                self.assertNoWarning(egx.Egor, [[0.0, 25.0]], trego={new: value})
                with self.assertRaises(TypeError):
                    egx.Egor([[0.0, 25.0]], trego={old: value, new: value})


class TestEgorDoe(DeprecationTestCase):
    def test_doe(self):
        x_doe = np.array([[0.0], [7.0], [25.0]])
        egor = self.assertDeprecated("doe", "x_doe", egx.Egor, [[0.0, 25.0]], doe=x_doe)
        self.assertIsInstance(egor, egx.Egor)
        self.assertNoWarning(egx.Egor, [[0.0, 25.0]], x_doe=x_doe)

    def test_both(self):
        x_doe = np.array([[0.0], [7.0], [25.0]])
        with self.assertRaises(TypeError):
            egx.Egor([[0.0, 25.0]], doe=x_doe, x_doe=x_doe)
        with self.assertRaises(TypeError):
            egx.Egor([[0.0, 25.0]], doe=x_doe, y_doe=xsinx(x_doe))

    def test_same_result(self):
        x_doe = np.array([[0.0], [7.0], [25.0]])
        y_doe = xsinx(x_doe)

        def run(**kwargs):
            egor = egx.Egor([[0.0, 25.0]], **kwargs)
            return egor.minimize(xsinx, max_iters=3, seed=42).result

        new = run(x_doe=x_doe, y_doe=y_doe)
        with self.assertWarns(DeprecationWarning):
            old = run(doe=np.hstack((x_doe, y_doe)))
        np.testing.assert_array_equal(new.x_doe, old.x_doe)
        np.testing.assert_array_equal(new.y_doe, old.y_doe)


class TestEgorCstrTol(DeprecationTestCase):
    """`cstr_tol` is not renamed: spec `tol` replaces it, it still warns and is used"""

    def setUp(self):
        # objective decreases with x, constraint c(x) = x - 1 <= 0: the best objective
        # violates the constraint by 0.2, accepted with a 0.5 tolerance only
        self.x_doe = np.array([[0.0], [0.5], [1.2]])
        self.y_doe = np.hstack((-self.x_doe, self.x_doe - 1.0))

    def fun(self, x):
        return np.hstack((-x, x - 1.0))

    def test_cstr_tol(self):
        egor = self.assertDeprecated(
            "cstr_tol",
            "CstrSpec(..., tol=...)",
            egx.Egor,
            [[0.0, 2.0]],
            n_cstr=1,
            cstr_tol=[0.5],
            x_doe=self.x_doe,
            y_doe=self.y_doe,
        )
        self.assertEqual(egor.best_index(self.y_doe), 2)
        res = egor.minimize(self.fun, max_iters=0)
        self.assertEqual(res.x_opt[0], self.x_doe[2, 0])

    def test_spec_tol(self):
        egor = self.assertNoWarning(
            egx.Egor,
            [[0.0, 2.0]],
            cstr_specs=[egx.CstrSpec.leq(0.0, tol=0.5)],
            x_doe=self.x_doe,
            y_doe=self.y_doe,
        )
        self.assertEqual(egor.best_index(self.y_doe), 2)
        res = egor.minimize(self.fun, max_iters=0)
        self.assertEqual(res.x_opt[0], self.x_doe[2, 0])


class TestSamplingOrder(DeprecationTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.XSPECS = [[0.0, 1.0], [-1.0, 1.0]]

    def test_old_order(self):
        new = egx.sampling(self.XSPECS, 5, method=egx.Sampling.RANDOM, seed=42)
        msg = "sampling(xspecs, n_samples, method=...)"
        for args, kwargs in (
            ((egx.Sampling.RANDOM, self.XSPECS, 5), {"seed": 42}),
            ((egx.Sampling.RANDOM, self.XSPECS, 5, 42), {}),
            ((3, self.XSPECS, 5), {"seed": 42}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                old = self.assertDeprecated(
                    "sampling(method", msg, egx.sampling, *args, **kwargs
                )
                np.testing.assert_array_equal(new, old)

    def test_keywords(self):
        # keywords only calls are not concerned by the order change
        np.testing.assert_array_equal(
            self.assertNoWarning(
                egx.sampling,
                method=egx.Sampling.RANDOM,
                xspecs=self.XSPECS,
                n_samples=5,
                seed=42,
            ),
            egx.sampling(self.XSPECS, 5, method=egx.Sampling.RANDOM, seed=42),
        )

    def test_new_order(self):
        new = self.assertNoWarning(
            egx.sampling, self.XSPECS, 5, egx.Sampling.RANDOM, 42
        )
        np.testing.assert_array_equal(
            new, egx.sampling(self.XSPECS, 5, method=egx.Sampling.RANDOM, seed=42)
        )
        np.testing.assert_array_equal(
            self.assertNoWarning(egx.sampling, self.XSPECS, 5, seed=42),
            egx.lhs(self.XSPECS, 5, seed=42),
        )


if __name__ == "__main__":
    unittest.main()

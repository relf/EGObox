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
                    "n_start", "theta_n_start", builder, n_start=3, nz=2
                )
                self.assertNoWarning(builder, theta_n_start=3, nz=2)
                with self.assertRaises(TypeError):
                    builder(n_start=3, theta_n_start=3, nz=2)


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


if __name__ == "__main__":
    unittest.main()

"""Names deprecated in 0.38.0 are removed in 1.0.0.

For each item: the old keyword argument raises TypeError, the old attribute or method
raises AttributeError and the old dict key raises ValueError, while the new name works.
"""

import unittest

import numpy as np

import egobox as egx


def xsinx(x: np.ndarray) -> np.ndarray:
    return (x - 3.5) * np.sin((x - 3.5) / np.pi)


XLIMITS = [[0.0, 25.0]]


class TestEgorRemoved(unittest.TestCase):
    def test_kwargs(self):
        x_doe = np.array([[0.0], [7.0], [25.0]])
        for old, value in (
            ("n_start", 5),
            ("doe", x_doe),
            ("cstr_tol", [0.5]),
        ):
            with self.subTest(old=old), self.assertRaises(TypeError):
                egx.Egor(XLIMITS, **{old: value})
        egx.Egor(XLIMITS, infill_n_start=5, x_doe=x_doe)
        egx.Egor(XLIMITS, cstr_specs=[egx.CstrSpec.leq(0.0, tol=0.5)])

    def test_methods(self):
        egor = egx.Egor(XLIMITS)
        for old in ("get_result", "get_result_index"):
            with self.subTest(old=old):
                self.assertFalse(hasattr(egor, old))
        x_doe = np.array([[0.0], [7.0], [20.0], [25.0]])
        y_doe = xsinx(x_doe)
        self.assertEqual(egor.best_index(y_doe), int(np.argmin(y_doe)))
        np.testing.assert_array_equal(
            egor.best_result(x_doe, y_doe).x_opt, x_doe[np.argmin(y_doe)]
        )


class TestGpConfigRemoved(unittest.TestCase):
    RENAMED = (("n_start", "theta_n_start"), ("max_eval", "theta_max_eval"))

    def test_kwargs(self):
        for old, new in self.RENAMED:
            with self.subTest(old=old):
                with self.assertRaises(TypeError):
                    egx.GpConfig(**{old: 3})
                self.assertEqual(getattr(egx.GpConfig(**{new: 3}), new), 3)

    def test_attributes(self):
        cfg = egx.GpConfig()
        for old, _ in self.RENAMED:
            with self.subTest(old=old):
                with self.assertRaises(AttributeError):
                    getattr(cfg, old)
                with self.assertRaises(AttributeError):
                    setattr(cfg, old, 3)

    def test_dict_keys(self):
        for old, new in self.RENAMED:
            with self.subTest(old=old):
                with self.assertRaises(ValueError):
                    egx.Egor(XLIMITS, gp_config={old: 3})
                egx.Egor(XLIMITS, gp_config={new: 3})


class TestGpMixRemoved(unittest.TestCase):
    def test_gpmix(self):
        for builder in (egx.GpMix, egx.Gpx.builder):
            for old in ("n_start", "max_eval"):
                with (
                    self.subTest(builder=builder, old=old),
                    self.assertRaises(TypeError),
                ):
                    builder(**{old: 3})
            builder(theta_n_start=3, theta_max_eval=30)

    def test_sparse_gpmix(self):
        z = np.array([[1.0], [3.0]])
        for builder in (egx.SparseGpMix, egx.SparseGpx.builder):
            for old, value in (("n_start", 3), ("nz", 2), ("z", z)):
                with (
                    self.subTest(builder=builder, old=old),
                    self.assertRaises(TypeError),
                ):
                    builder(**{old: value})
            builder(theta_n_start=3, n_inducing=2)
            builder(inducing=z)


class TestGpxProperties(unittest.TestCase):
    """`thetas`, `variances` and `likelihoods` are properties, no more methods"""

    def test_properties(self):
        xt = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
        yt = np.array([[0.0, 1.0, 1.5, 0.9, 1.0]]).T
        gpx = egx.Gpx.builder(seed=42).fit(xt, yt)
        sgpx = egx.SparseGpx.builder(n_inducing=3, seed=42).fit(xt, yt)
        for model in (gpx, sgpx):
            for name, shape in (
                ("thetas", (1, 1)),
                ("variances", (1,)),
                ("likelihoods", (1,)),
            ):
                with self.subTest(model=type(model).__name__, name=name):
                    value = getattr(model, name)
                    self.assertIsInstance(value, np.ndarray)
                    self.assertEqual(value.shape, shape)
                    with self.assertRaises(TypeError):
                        value()
                    with self.assertRaises(AttributeError):
                        setattr(model, name, value)


class TestCstrSpecRemoved(unittest.TestCase):
    def test_btw(self):
        self.assertFalse(hasattr(egx.CstrSpec, "btw"))
        egx.CstrSpec.between(1.0, 3.0)

    def test_dict_key(self):
        with self.assertRaises(ValueError):
            egx.Egor(XLIMITS, cstr_specs=[{"btw": (1.0, 3.0)}])
        egx.Egor(XLIMITS, cstr_specs=[{"between": (1.0, 3.0)}])


class TestQEiConfigRemoved(unittest.TestCase):
    def test_kwarg(self):
        with self.assertRaises(TypeError):
            egx.QEiConfig(optmod=2)
        self.assertEqual(egx.QEiConfig(optim_every=2).optim_every, 2)

    def test_attribute(self):
        cfg = egx.QEiConfig()
        with self.assertRaises(AttributeError):
            _ = cfg.optmod
        with self.assertRaises(AttributeError):
            cfg.optmod = 2

    def test_dict_key(self):
        with self.assertRaises(ValueError):
            egx.Egor(XLIMITS, qei_config={"optmod": 2})
        egx.Egor(XLIMITS, qei_config={"optim_every": 2})


class TestTregoConfigRemoved(unittest.TestCase):
    RENAMED = (
        ("n_gl_steps", "n_global_local_steps", (2, 3)),
        ("d", "radius_bounds", (1e-5, 0.5)),
    )

    def test_kwargs(self):
        for old, new, value in self.RENAMED:
            with self.subTest(old=old):
                with self.assertRaises(TypeError):
                    egx.TregoConfig(**{old: value})
                self.assertEqual(getattr(egx.TregoConfig(**{new: value}), new), value)

    def test_attributes(self):
        cfg = egx.TregoConfig()
        for old, _, value in self.RENAMED:
            with self.subTest(old=old):
                with self.assertRaises(AttributeError):
                    getattr(cfg, old)
                with self.assertRaises(AttributeError):
                    setattr(cfg, old, value)

    def test_dict_keys(self):
        for old, new, value in self.RENAMED:
            with self.subTest(old=old):
                with self.assertRaises(ValueError):
                    egx.Egor(XLIMITS, trego={old: value})
                egx.Egor(XLIMITS, trego={new: value})


class TestSamplingOrderRemoved(unittest.TestCase):
    def test_old_order(self):
        xspecs = [[0.0, 1.0], [-1.0, 1.0]]
        for args in (
            (egx.Sampling.RANDOM, xspecs, 5),
            (3, xspecs, 5),
        ):
            with self.subTest(args=args), self.assertRaises(TypeError):
                egx.sampling(*args, seed=42)


if __name__ == "__main__":
    unittest.main()

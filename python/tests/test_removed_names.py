"""Names deprecated in 0.38.0 are removed in 1.0.0.

Removed keyword arguments are checked by `test_stubs.py`: runtime signatures match
the stub (`test_signatures_match_runtime`), which no longer lists them
(`test_removed_names_not_in_stub`). Here are checked the runtime behaviors stubs
cannot tell: old attributes or methods raise AttributeError, old dict keys raise
ValueError while the new ones work.
"""

import unittest

import numpy as np

import egobox as egx

XLIMITS = [[0.0, 25.0]]


class TestRemovedNames(unittest.TestCase):
    def test_removed_attributes(self):
        for obj, old in (
            (egx.Egor(XLIMITS), "get_result"),
            (egx.Egor(XLIMITS), "get_result_index"),
            (egx.CstrSpec, "btw"),
            (egx.GpConfig(), "n_start"),
            (egx.GpConfig(), "max_eval"),
            (egx.QEiConfig(), "optmod"),
            (egx.TregoConfig(), "n_gl_steps"),
            (egx.TregoConfig(), "d"),
        ):
            with self.subTest(obj=type(obj).__name__, old=old):
                self.assertFalse(hasattr(obj, old))
                if not isinstance(obj, type):
                    with self.assertRaises(AttributeError):
                        setattr(obj, old, 3)

    def test_removed_dict_keys(self):
        for kwarg, old, new, value in (
            ("gp_config", "n_start", "theta_n_start", 3),
            ("gp_config", "max_eval", "theta_max_eval", 3),
            ("cstr_specs", "btw", "between", (1.0, 3.0)),
            ("qei_config", "optmod", "optim_every", 2),
            ("trego", "n_gl_steps", "n_global_local_steps", (2, 3)),
            ("trego", "d", "radius_bounds", (1e-5, 0.5)),
        ):
            wrap = (lambda d: [d]) if kwarg == "cstr_specs" else (lambda d: d)
            with self.subTest(kwarg=kwarg, old=old):
                with self.assertRaises(ValueError):
                    egx.Egor(XLIMITS, **{kwarg: wrap({old: value})})
                egx.Egor(XLIMITS, **{kwarg: wrap({new: value})})


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


if __name__ == "__main__":
    unittest.main()

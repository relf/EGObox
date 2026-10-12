import unittest

import numpy as np

import egobox as egx


class TestSampling(unittest.TestCase):
    def test_arguments(self):
        xspecs = [[0.0, 1.0], [-1.0, 1.0]]
        expected = egx.sampling(xspecs, 5, method=egx.Sampling.RANDOM, seed=42)
        # positional, keywords only, method as int
        for args, kwargs in (
            ((xspecs, 5, egx.Sampling.RANDOM, 42), {}),
            (
                (),
                {
                    "method": egx.Sampling.RANDOM,
                    "xspecs": xspecs,
                    "n_samples": 5,
                    "seed": 42,
                },
            ),
            ((xspecs, 5), {"method": 3, "seed": 42}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                np.testing.assert_array_equal(egx.sampling(*args, **kwargs), expected)
        # LHS by default
        np.testing.assert_array_equal(
            egx.sampling(xspecs, 5, seed=42), egx.lhs(xspecs, 5, seed=42)
        )

    def test_lhs(self):
        xtypes = [
            egx.XSpec(egx.XType.FLOAT, [-5.0, 5.0]),
            egx.XSpec(egx.XType.ENUM, tags=["blue", "red", "green"]),
            egx.XSpec(egx.XType.ENUM, xlimits=[2]),
            egx.XSpec(egx.XType.ORD, [0, 2, 3]),
        ]

        actual = egx.lhs(xtypes, 10, seed=42)
        expected = np.array(
            [
                [0.30306531, 2.0, 1.0, 2.0],
                [1.08760961, 0.0, 1.0, 0.0],
                [-4.9142444, 2.0, 1.0, 2.0],
                [-3.6895886, 0.0, 1.0, 3.0],
                [-2.93743022, 1.0, 0.0, 0.0],
                [-0.99995704, 1.0, 0.0, 3.0],
                [2.22703374, 2.0, 0.0, 3.0],
                [3.86279547, 0.0, 0.0, 2.0],
                [4.52325395, 1.0, 1.0, 3.0],
                [-1.69353868, 1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_all_lhs(self):
        xtypes = [
            egx.XSpec(egx.XType.FLOAT, [-5.0, 5.0]),
            egx.XSpec(egx.XType.ENUM, tags=["blue", "red", "green"]),
            egx.XSpec(egx.XType.ENUM, xlimits=[2]),
            egx.XSpec(egx.XType.ORD, [0, 2, 3]),
        ]

        for kind in [
            egx.Sampling.LHS_CLASSIC,
            egx.Sampling.LHS_CENTERED,
            egx.Sampling.LHS_MAXIMIN,
            egx.Sampling.LHS_CENTERED_MAXIMIN,
            egx.Sampling.LHS,
        ]:
            lhs = egx.sampling(xtypes, 10, method=kind, seed=42)
            print(lhs)

    def test_ffact(self):
        xtypes = [
            egx.XSpec(egx.XType.FLOAT, [-5.0, 5.0]),
            egx.XSpec(egx.XType.INT, [-10, 10]),
        ]

        actual = egx.sampling(xtypes, 10, method=egx.Sampling.FULL_FACTORIAL, seed=42)

        expected = np.array(
            [
                [-5.0, -10.0],
                [-5.0, 0.0],
                [-5.0, 10.0],
                [-1.66666667, -10.0],
                [-1.66666667, 0.0],
                [-1.66666667, 10.0],
                [1.66666667, -10.0],
                [1.66666667, 0.0],
                [1.66666667, 10.0],
                [5.0, -10.0],
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_random(self):
        xtypes = [
            egx.XSpec(egx.XType.FLOAT, [-5.0, 5.0]),
            egx.XSpec(egx.XType.ENUM, tags=["blue", "red", "green"]),
            egx.XSpec(egx.XType.ENUM, xlimits=[2]),
            egx.XSpec(egx.XType.ORD, [0, 2, 3]),
        ]

        actual = egx.sampling(xtypes, 10, method=egx.Sampling.RANDOM, seed=42)
        expected = np.array(
            [
                [-4.14244405, 0.0, 0.0, 0.0],
                [-2.72966261, 2.0, 1.0, 3.0],
                [-3.55769164, 1.0, 1.0, 0.0],
                [4.08641562, 0.0, 1.0, 3.0],
                [-2.5270829, 1.0, 1.0, 3.0],
                [0.45364361, 1.0, 0.0, 0.0],
                [0.14008757, 1.0, 1.0, 2.0],
                [-2.87309185, 2.0, 1.0, 0.0],
                [-2.83505015, 0.0, 1.0, 3.0],
                [2.84855288, 1.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()

import numpy as np
import egobox as egx
import unittest


class TestSampling(unittest.TestCase):
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
                [0.30306531, 0.0, 0.0, 2.0],
                [1.08760961, 1.0, 0.0, 0.0],
                [-4.9142444, 0.0, 0.0, 2.0],
                [-3.6895886, 2.0, 1.0, 2.0],
                [-2.93743022, 2.0, 1.0, 0.0],
                [-0.99995704, 1.0, 0.0, 3.0],
                [2.22703374, 0.0, 0.0, 2.0],
                [3.86279547, 2.0, 1.0, 2.0],
                [4.52325395, 0.0, 0.0, 2.0],
                [-1.69353868, 0.0, 0.0, 0.0],
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
            lhs = egx.sampling(kind, xtypes, 10, seed=42)
            print(lhs)

    def test_ffact(self):
        xtypes = [
            egx.XSpec(egx.XType.FLOAT, [-5.0, 5.0]),
            egx.XSpec(egx.XType.INT, [-10, 10]),
        ]

        actual = egx.sampling(egx.Sampling.FULL_FACTORIAL, xtypes, 10, seed=42)

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

        actual = egx.sampling(egx.Sampling.RANDOM, xtypes, 10, seed=42)
        expected = np.array(
            [
                [-4.14244405, 0.0, 0.0, 0.0],
                [-2.72966261, 0.0, 0.0, 3.0],
                [-3.55769164, 0.0, 0.0, 0.0],
                [4.08641562, 2.0, 1.0, 3.0],
                [-2.5270829, 1.0, 0.0, 3.0],
                [0.45364361, 2.0, 1.0, 0.0],
                [0.14008757, 0.0, 0.0, 2.0],
                [-2.87309185, 0.0, 0.0, 0.0],
                [-2.83505015, 2.0, 1.0, 2.0],
                [2.84855288, 2.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import egobox as egx
import logging

logging.basicConfig(level=logging.INFO)


def xsinx(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    print(f"obj={y} at {x}")
    return y


def mixobj(X):
    # float
    x1 = X[:, 0]
    #  XType.ENUM 1
    c1 = X[:, 1]
    x2 = c1 == 0
    x3 = c1 == 1
    x4 = c1 == 2
    #  XType.ENUM 2
    c2 = X[:, 2]
    x5 = c2 == 0
    x6 = c2 == 1
    # int
    i = X[:, 3]

    y = (x2 + 2 * x3 + 3 * x4) * x5 * x1 + (x2 + 2 * x3 + 3 * x4) * x6 * 0.95 * x1 + i
    return y.reshape(-1, 1)


class TestMixintEgx(unittest.TestCase):
    def test_int(self):
        xspecs = [egx.XSpec(egx.XType.INT, [0.0, 25.0])]

        egor = egx.Egor(
            xspecs,
            infill_strategy=egx.InfillStrategy.EI,
            seed=42,
            doe=np.array([[0.0], [7.0], [25.0]]),
        )
        res = egor.minimize(xsinx, max_iters=10)
        print(res.x_opt, res.y_opt)
        print(f"Optimization f={res.y_opt} at {res.x_opt}")
        self.assertAlmostEqual(-15.125, res.y_opt[0], delta=5e-3)
        self.assertAlmostEqual(19, res.x_opt[0], delta=1)

    def test_ord_enum(self):
        xspecs = [
            egx.XSpec(egx.XType.FLOAT, [-5.0, 5.0]),
            egx.XSpec(egx.XType.ENUM, tags=["blue", "red", "green"]),
            egx.XSpec(egx.XType.ENUM, xlimits=[2]),
            egx.XSpec(egx.XType.ORD, [0, 2, 3]),
        ]
        egor = egx.Egor(xspecs, infill_strategy=egx.InfillStrategy.WB2, seed=42)
        res = egor.minimize(mixobj, max_iters=10)
        self.assertAlmostEqual(-14.25, res.y_opt[0])
        self.assertAlmostEqual(-5, res.x_opt[0])
        self.assertAlmostEqual(2, res.x_opt[1])
        self.assertAlmostEqual(1, res.x_opt[2])
        self.assertAlmostEqual(0, res.x_opt[3])


if __name__ == "__main__":
    unittest.main()

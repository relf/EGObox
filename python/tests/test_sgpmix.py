import unittest
import numpy as np
import egobox as egx
import logging
import time

logging.basicConfig(level=logging.DEBUG)


def f_obj(x):
    return (
        np.sin(3 * np.pi * x)
        + 0.3 * np.cos(9 * np.pi * x)
        + 0.5 * np.sin(7 * np.pi * x)
    )


class TestSgp(unittest.TestCase):
    def setUp(self):
        # random generator for reproducibility
        self.rng = np.random.RandomState(0)

        # Generate training data
        self.nt = 200
        # Variance of the gaussian noise on our trainingg data
        eta2 = [0.01]
        gaussian_noise = self.rng.normal(
            loc=0.0, scale=np.sqrt(eta2), size=(self.nt, 1)
        )
        self.xt = 2 * self.rng.rand(self.nt, 1) - 1
        self.yt = f_obj(self.xt) + gaussian_noise

        # Pick inducing points randomly in training data
        self.n_inducing = 30

    def test_sgp(self):
        random_idx = self.rng.permutation(self.nt)[: self.n_inducing]
        Z = self.xt[random_idx].copy()

        start = time.time()
        sgp = egx.SparseGpMix(z=Z).fit(self.xt, self.yt)
        elapsed = time.time() - start
        print(elapsed)
        sgp.save("sgp.json")

    def test_sgp_random(self):
        start = time.time()
        sgp = egx.SparseGpMix(nz=self.n_inducing, seed=0).fit(self.xt, self.yt)
        elapsed = time.time() - start
        print(elapsed)
        print(sgp)

    def test_sgp_multi_outputs_exception(self):
        yt = np.hstack((self.yt, self.yt))

        with self.assertRaises(BaseException):
            egx.SparseGpx.builder(nz=self.n_inducing, seed=0).fit(self.xt, yt)

    def test_1d_training_data(self):
        self.xt1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.yt1 = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        self.sgpx = egx.SparseGpx.builder(nz=self.n_inducing, seed=0).fit(
            self.xt, self.yt
        )


if __name__ == "__main__":
    unittest.main()

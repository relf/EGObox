import logging
import time
import unittest

import numpy as np

import egobox as egx

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
        sgp = egx.SparseGpMix(inducing=Z).fit(self.xt, self.yt)
        elapsed = time.time() - start
        print(elapsed)
        sgp.save("sgp.json")

    def test_sgp_random(self):
        start = time.time()
        sgp = egx.SparseGpMix(n_inducing=self.n_inducing, seed=0).fit(self.xt, self.yt)
        elapsed = time.time() - start
        print(elapsed)
        print(sgp)

    def test_sgp_multi_outputs_exception(self):
        yt = np.hstack((self.yt, self.yt))

        with self.assertRaises(ValueError):
            egx.SparseGpx.builder(n_inducing=self.n_inducing, seed=0).fit(self.xt, yt)

    def test_1d_training_data(self):
        xt1 = self.xt.ravel()
        yt1 = self.yt.ravel()

        sgpx = egx.SparseGpx.builder(n_inducing=self.n_inducing, seed=0).fit(xt1, yt1)
        self.assertEqual(sgpx.dims(), (1, 1))

    def test_dims_and_training_data(self):
        sgpx = egx.SparseGpx.builder(n_inducing=self.n_inducing, seed=0).fit(
            self.xt, self.yt
        )
        self.assertEqual(sgpx.dims(), (1, 1))
        self.assertEqual((sgpx.nx, sgpx.ny), (1, 1))
        xdata, ydata = sgpx.training_data()
        np.testing.assert_array_equal(xdata, self.xt)
        np.testing.assert_array_equal(ydata, self.yt.ravel())

    def test_1d_input(self):
        sgpx = egx.SparseGpx.builder(n_inducing=self.n_inducing, seed=0).fit(
            self.xt, self.yt
        )
        x1d = np.array([-0.5, 0.0, 0.5])
        x2d = x1d[:, None]
        for method in [
            sgpx.predict,
            sgpx.predict_var,
            sgpx.predict_gradients,
            sgpx.predict_var_gradients,
        ]:
            np.testing.assert_array_equal(method(x1d), method(x2d))
        self.assertEqual(sgpx.sample(x1d, 2).shape, (3, 2))

    def test_predict_return_std(self):
        sgpx = egx.SparseGpx.builder(n_inducing=self.n_inducing, seed=0).fit(
            self.xt, self.yt
        )
        x = np.array([[-0.5], [0.0], [0.5]])
        mean, std = sgpx.predict(x, return_std=True)
        np.testing.assert_array_equal(mean, sgpx.predict(x))
        np.testing.assert_allclose(std, np.sqrt(sgpx.predict_var(x)))


if __name__ == "__main__":
    unittest.main()

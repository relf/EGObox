import os
import tempfile
import unittest

import numpy as np

import egobox as egx


def xsq(x):
    return (x - 0.3) ** 2


class TestErrors(unittest.TestCase):
    """Check that misuse raises standard Python exceptions (not PanicException)"""

    def setUp(self):
        self.xt = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
        self.yt = np.sin(6.0 * self.xt).ravel()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def assertRaisesStd(self, exc, fn, *args, **kwargs):
        # exc is a subclass of Exception, hence catchable with `except Exception`
        self.assertTrue(issubclass(exc, Exception))
        with self.assertRaises(exc):
            fn(*args, **kwargs)

    # Domain specifications

    def test_empty_domain(self):
        self.assertRaisesStd(ValueError, egx.Egor, [])
        self.assertRaisesStd(ValueError, egx.lhs, [], 5)

    def test_malformed_domain(self):
        self.assertRaisesStd(TypeError, egx.Egor, "abc")
        self.assertRaisesStd(ValueError, egx.Egor, [[0.0]])
        self.assertRaisesStd(ValueError, egx.lhs, np.zeros((2, 3)), 5)
        self.assertRaisesStd(
            ValueError, egx.Egor, [egx.XSpec(egx.XType.FLOAT, [0.0])]
        )
        self.assertRaisesStd(ValueError, egx.Egor, [egx.XSpec(egx.XType.INT, [1])])
        self.assertRaisesStd(ValueError, egx.Egor, [egx.XSpec(egx.XType.ORD)])
        self.assertRaisesStd(ValueError, egx.Egor, [egx.XSpec(egx.XType.ENUM)])
        self.assertRaisesStd(ValueError, egx.Gpx.builder, xspecs=[[0.0]])

    # GP configuration

    def test_bad_gp_config(self):
        self.assertRaisesStd(
            ValueError, egx.Egor, [[0.0, 1.0]], gp_config=egx.GpConfig(regr_spec=0)
        )
        self.assertRaisesStd(
            ValueError, egx.Egor, [[0.0, 1.0]], gp_config=egx.GpConfig(corr_spec=16)
        )
        self.assertRaisesStd(
            ValueError, egx.Gpx.builder(corr_spec=16).fit, self.xt, self.yt
        )
        self.assertRaisesStd(
            ValueError,
            egx.Gpx.builder(theta_bounds=[[1.0, 0.0]]).fit,
            self.xt,
            self.yt,
        )
        self.assertRaisesStd(
            ValueError,
            egx.SparseGpx.builder(nz=3, theta_bounds=[[1.0]]).fit,
            self.xt,
            self.yt,
        )

    def test_bad_trego(self):
        self.assertRaisesStd(TypeError, egx.Egor, [[0.0, 1.0]], trego=3.5)

    # Training data

    def test_bad_training_data(self):
        yt2 = np.hstack((self.yt[:, None], self.yt[:, None]))
        self.assertRaisesStd(ValueError, egx.Gpx.builder().fit, self.xt, yt2)
        self.assertRaisesStd(
            ValueError, egx.Gpx.builder().fit, self.xt, self.yt[:3]
        )
        self.assertRaisesStd(
            ValueError,
            egx.Gpx.builder(xspecs=[[0.0, 1.0], [0.0, 1.0]]).fit,
            self.xt,
            self.yt,
        )
        self.assertRaisesStd(ValueError, egx.SparseGpx.builder(nz=3).fit, self.xt, yt2)

    def test_sgp_without_inducing_points(self):
        self.assertRaisesStd(ValueError, egx.SparseGpMix().fit, self.xt, self.yt)

    # Trained model usage

    def test_predict_wrong_dim(self):
        gpx = egx.Gpx.builder(seed=42).fit(self.xt, self.yt)
        x = np.zeros((2, 3))
        for method in [
            gpx.predict,
            gpx.predict_var,
            gpx.predict_gradients,
            gpx.predict_var_gradients,
        ]:
            self.assertRaisesStd(ValueError, method, x)
        self.assertRaisesStd(ValueError, gpx.sample, x, 2)

    def test_update_mismatch(self):
        gpx = egx.Gpx.builder(seed=42).fit(self.xt, self.yt)
        self.assertRaisesStd(ValueError, gpx.update, np.zeros((2, 1)), np.zeros(3))

    def test_save_load(self):
        gpx = egx.Gpx.builder(seed=42).fit(self.xt, self.yt)

        # no extension means binary format
        filename = os.path.join(self.tmpdir.name, "noext")
        self.assertTrue(gpx.save(filename))
        self.assertEqual((1, 1), egx.Gpx.load(filename).dims())

        self.assertRaisesStd(
            FileNotFoundError, gpx.save, os.path.join(self.tmpdir.name, "no", "gp.json")
        )
        self.assertRaisesStd(
            FileNotFoundError, egx.Gpx.load, os.path.join(self.tmpdir.name, "gp.json")
        )

        garbage = os.path.join(self.tmpdir.name, "garbage.json")
        with open(garbage, "w") as f:
            f.write("not a gp")
        self.assertRaisesStd(ValueError, egx.Gpx.load, garbage)
        self.assertRaisesStd(ValueError, egx.SparseGpx.load, garbage)

    # Optimizer

    def test_fcstr_error_propagates(self):
        def cstr(x, return_grad):
            return 1 / 0

        egor = egx.Egor([[0.0, 1.0]], n_doe=3)
        self.assertRaisesStd(
            ZeroDivisionError, egor.minimize, xsq, fcstrs=[cstr], max_iters=2
        )

    def test_fcstr_bad_return_type(self):
        def cstr(x, return_grad):
            return "wrong"

        egor = egx.Egor([[0.0, 1.0]], n_doe=3)
        self.assertRaisesStd(TypeError, egor.minimize, xsq, fcstrs=[cstr], max_iters=2)

    def test_fun_bad_return_value(self):
        egor = egx.Egor([[0.0, 1.0]], n_doe=3)
        # 1D array instead of (n, 1)
        self.assertRaisesStd(
            TypeError, egor.minimize, lambda x: xsq(x).ravel(), max_iters=2
        )
        # wrong number of columns wrt n_cstr
        egor = egx.Egor([[0.0, 1.0]], n_cstr=1, n_doe=3)
        self.assertRaisesStd(ValueError, egor.minimize, xsq, max_iters=2)

    def test_fun_exception_in_initial_doe(self):
        # With stop_on_error, the optimization can not start without initial doe
        def fun(x):
            raise KeyError("boom")

        egor = egx.Egor([[0.0, 1.0]], n_doe=3)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            egor.minimize(fun, max_iters=2, stop_on_error=True)

    def test_fun_exception_during_iterations(self):
        # With stop_on_error, the optimization stops with a dedicated exit status
        n_calls = 0

        def fun(x):
            nonlocal n_calls
            n_calls += 1
            if n_calls > 1:  # first call is initial doe evaluation
                raise KeyError("boom")
            return xsq(x)

        egor = egx.Egor([[0.0, 1.0]], n_doe=3)
        res = egor.minimize(fun, max_iters=5, stop_on_error=True)
        self.assertEqual(egx.ExitStatus.OBJECTIVE_FUNCTION_ERROR, res.status.exit)

    def test_suggest_mismatch(self):
        egor = egx.Egor([[0.0, 1.0]])
        self.assertRaisesStd(
            ValueError, egor.suggest, np.zeros((3, 1)), np.zeros((2, 1))
        )
        self.assertRaisesStd(
            ValueError, egor.suggest, np.zeros((3, 2)), np.zeros((3, 1))
        )

    def test_result_empty(self):
        egor = egx.Egor([[0.0, 1.0]])
        self.assertRaisesStd(ValueError, egor.get_result_index, np.zeros((0, 1)))
        self.assertRaisesStd(
            ValueError, egor.get_result, np.zeros((3, 1)), np.zeros((2, 1))
        )


if __name__ == "__main__":
    unittest.main()

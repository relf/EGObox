import unittest

import numpy as np


class TestApiImports(unittest.TestCase):
    def test_all_public_symbols_are_importable(self):
        import egobox as egx
        from egobox import (
            ConstraintStrategy,
            CorrelationSpec,
            Egor,
            EgorOptim,
            ExitStatus,
            FailsafeStrategy,
            GpConfig,
            GpMix,
            Gpx,
            InfillOptimizer,
            InfillStrategy,
            OptimResult,
            QEiConfig,
            QEiStrategy,
            Recombination,
            RegressionSpec,
            RunInfo,
            RunStatus,
            Sampling,
            SparseGpMix,
            SparseGpx,
            SparseMethod,
            TregoConfig,
            Verbose,
            XSpec,
            XType,
            lhs,
            sampling,
        )

        imported_symbols = {
            "ConstraintStrategy": ConstraintStrategy,
            "CorrelationSpec": CorrelationSpec,
            "Egor": Egor,
            "EgorOptim": EgorOptim,
            "ExitStatus": ExitStatus,
            "FailsafeStrategy": FailsafeStrategy,
            "FeasibleInfillStrategy": egx.FeasibleInfillStrategy,
            "GpConfig": GpConfig,
            "GpMix": GpMix,
            "Gpx": Gpx,
            "InfillOptimizer": InfillOptimizer,
            "InfillStrategy": InfillStrategy,
            "OptimResult": OptimResult,
            "QEiConfig": QEiConfig,
            "QEiStrategy": QEiStrategy,
            "Recombination": Recombination,
            "RegressionSpec": RegressionSpec,
            "RunInfo": RunInfo,
            "RunStatus": RunStatus,
            "Sampling": Sampling,
            "SparseGpMix": SparseGpMix,
            "SparseGpx": SparseGpx,
            "SparseMethod": SparseMethod,
            "TregoConfig": TregoConfig,
            "Verbose": Verbose,
            "XSpec": XSpec,
            "XType": XType,
            "lhs": lhs,
            "sampling": sampling,
        }

        for name, symbol in imported_symbols.items():
            self.assertTrue(hasattr(egx, name), f"egobox is missing '{name}'")
            self.assertIs(symbol, getattr(egx, name))

    def test_egor_accepts_int_and_dict_arguments(self):
        import egobox as egx

        egor = egx.Egor(
            [[0.0, 1.0]],
            gp_config={"n_clusters": 0, "recombination": 1},
            qei_config={"batch": 2, "strategy": 2, "optmod": 1},
            infill_strategy=4,
            cstr_strategy=1,
            infill_optimizer=2,
            failsafe_strategy=3,
            trego={"alpha": 0.8},
        )
        self.assertIsNotNone(egor)

    def test_minimize_accepts_run_info_dict(self):
        import egobox as egx

        def fobj(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            return np.sum(x**2, axis=1).reshape(-1, 1)

        egor = egx.Egor([[0.0, 1.0]], infill_strategy=1)
        optim = egor.minimize(
            fobj, max_iters=1, seed=42, run_info={"fname": "fobj", "num": 7}
        )
        self.assertEqual(optim.status.info.fname, "fobj")
        self.assertEqual(optim.status.info.num, 7)

    def test_run_info_default_fname(self):
        import egobox as egx

        def fobj(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            return np.sum(x**2, axis=1).reshape(-1, 1)

        self.assertEqual(egx.RunInfo().fname, "objective_function")
        egor = egx.Egor([[0.0, 1.0]], infill_strategy=1)
        optim = egor.minimize(fobj, max_iters=1, seed=42)
        self.assertEqual(optim.status.info.fname, "objective_function")
        optim = egor.minimize(fobj, max_iters=1, seed=42, run_info={"num": 3})
        self.assertEqual(optim.status.info.fname, "objective_function")
        self.assertEqual(optim.status.info.num, 3)
        with self.assertRaises(TypeError):
            egor.minimize(fobj, max_iters=1, seed=42, run_info="fobj")

    def test_egor_target(self):
        import egobox as egx

        def fobj(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            return np.sum(x**2, axis=1).reshape(-1, 1)

        optim = egx.Egor([[-1.0, 1.0]], target=None).minimize(
            fobj, max_iters=2, seed=42
        )
        self.assertEqual(optim.status.exit, egx.ExitStatus.MAX_ITERS_REACHED)
        # any objective value reaches such a target
        optim = egx.Egor([[-1.0, 1.0]], target=1e9).minimize(fobj, max_iters=2, seed=42)
        self.assertEqual(optim.status.exit, egx.ExitStatus.TARGET_COST_REACHED)

    def test_gpmix_max_eval_is_used(self):
        import egobox as egx

        # likelihood evaluations per start: clamp(10 * nx, 25, theta_max_eval)
        # hence nx = 5 to get 50 evaluations by default, capped to 25 here
        xt = egx.lhs([[0.0, 4.0]] * 5, 30, seed=42)
        yt = (np.sin(xt[:, 0]) + np.cos(2 * xt[:, 1]) + xt[:, 2:].sum(axis=1)).reshape(
            -1, 1
        )

        def thetas(max_eval):
            gpx = egx.GpMix(theta_n_start=1, theta_max_eval=max_eval, seed=42).fit(
                xt, yt
            )
            return gpx.thetas()

        self.assertFalse(np.allclose(thetas(25), thetas(1000)))

    def test_sampling_accepts_int_method(self):
        import egobox as egx

        doe = egx.sampling(1, [[0.0, 1.0]], 4, seed=1)
        self.assertEqual(doe.shape, (4, 1))

    def test_xspec_and_sparse_method_accept_int(self):
        import egobox as egx

        xspec = egx.XSpec(1, [0.0, 1.0])
        self.assertEqual(xspec.xtype, egx.XType.FLOAT)

        sgp = egx.SparseGpMix(nz=3, method=2)
        self.assertIsNotNone(sgp)

    def test_cstr_spec_accepts_dict_form(self):
        import egobox as egx

        def fobj(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            obj = np.sum(x**2, axis=1).reshape(-1, 1)
            cst = x[:, 0].reshape(-1, 1)
            return np.hstack((obj, cst))

        egor = egx.Egor(
            [[0.0, 1.0]], n_cstr=1, cstr_specs=[{"leq": 0.8}], infill_strategy=1
        )
        optim = egor.minimize(fobj, max_iters=1, seed=42)
        self.assertEqual(optim.status.total_iters, 1)

    def test_classes_module_is_egobox(self):
        import egobox as egx

        for cls in (egx.GpConfig, egx.Egor, egx.Gpx, egx.SparseMethod, egx.XSpec):
            self.assertEqual(cls.__module__, "egobox")

    def test_configs_repr(self):
        import egobox as egx

        self.assertEqual(
            repr(egx.GpConfig(n_clusters=2, theta_bounds=[[0.1, 1.0]])),
            "GpConfig(regr_spec=1, corr_spec=1, kpls_dim=None, n_clusters=2, "
            "recombination=Recombination.HARD, theta_init=None, "
            "theta_bounds=[[0.1, 1.0]], theta_n_start=10, theta_max_eval=50)",
        )
        self.assertEqual(
            repr(egx.QEiConfig(batch=3)),
            "QEiConfig(batch=3, strategy=QEiStrategy.KB, optmod=1)",
        )
        self.assertEqual(
            repr(egx.TregoConfig()),
            "TregoConfig(n_gl_steps=(1, 4), d=(1e-06, 1.0), alpha=1.0, beta=0.9, sigma0=0.1)",
        )
        self.assertEqual(repr(egx.RunInfo("f", 2)), "RunInfo(fname='f', num=2)")

    def test_optim_result_repr(self):
        import egobox as egx

        def fobj(x: np.ndarray) -> np.ndarray:
            return (x - 0.3) ** 2

        optim = egx.Egor([[0.0, 1.0]]).minimize(fobj, max_iters=1, seed=42)
        self.assertTrue(
            repr(optim).startswith("EgorOptim(result=OptimResult(x_opt=array(")
        )
        self.assertIn("n_doe=", repr(optim.result))
        self.assertIn("exit=ExitStatus.MAX_ITERS_REACHED", repr(optim.status))

    def test_egor_config_arguments_accept_none(self):
        import egobox as egx

        def fobj(x: np.ndarray) -> np.ndarray:
            return (x - 0.3) ** 2

        egor = egx.Egor([[0.0, 1.0]], gp_config=None, qei_config=None)
        optim = egor.minimize(fobj, fcstrs=None, fcstr_specs=None, max_iters=1, seed=42)
        self.assertEqual(optim.status.total_iters, 1)


if __name__ == "__main__":
    unittest.main()

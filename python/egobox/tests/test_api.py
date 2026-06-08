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
        optim = egor.minimize(fobj, max_iters=1, seed=42, run_info={"fname": "fobj", "num": 7})
        self.assertEqual(optim.status.info.fname, "fobj")
        self.assertEqual(optim.status.info.num, 7)


if __name__ == "__main__":
    unittest.main()

import unittest


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


if __name__ == "__main__":
    unittest.main()

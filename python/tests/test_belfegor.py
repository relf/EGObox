import unittest

import numpy as np

import egobox as egx

ZDT1_XLIMITS = [[0.0, 1.0], [0.0, 1.0]]
# hypervolume of the ZDT1 true front wrt (1.1, 1.1)
ZDT1_HV_REF = 0.1 + 2.0 / 3.0 + 0.11


def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1 with 2 variables in [0, 1]: front f2 = 1 - sqrt(f1) for x2 = 0"""
    f1 = x[:, 0]
    g = 1.0 + 9.0 * x[:, 1]
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.column_stack([f1, f2])


def bnh_raw(x: np.ndarray) -> np.ndarray:
    """Binh and Korn problem: [f1, f2, c1, c2] with c1 <= 25 and c2 >= 7.7"""
    x1, x2 = x[:, 0], x[:, 1]
    f1 = 4.0 * x1**2 + 4.0 * x2**2
    f2 = (x1 - 5.0) ** 2 + (x2 - 5.0) ** 2
    c1 = (x1 - 5.0) ** 2 + x2**2
    c2 = (x1 - 8.0) ** 2 + (x2 + 3.0) ** 2
    return np.column_stack([f1, f2, c1, c2])


def hypervolume_2d(front: np.ndarray, ref=(1.1, 1.1)) -> float:
    pts = sorted((a, b) for a, b in front[:, :2] if a < ref[0] and b < ref[1])
    hv, current = 0.0, ref[1]
    for a, b in pts:
        if b < current:
            hv += (ref[0] - a) * (current - b)
            current = b
    return hv


def assert_non_dominated(test: unittest.TestCase, y: np.ndarray, n_obj: int = 2):
    objs = y[:, :n_obj]
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i != j:
                test.assertFalse(
                    np.all(b <= a) and np.any(b < a), f"{a} dominated by {b}"
                )


class TestBelfegor(unittest.TestCase):
    def test_zdt1_default_strategy(self):
        belfegor = egx.Belfegor(ZDT1_XLIMITS, n_doe=10, seed=42)
        self.assertEqual(belfegor.n_obj, 2)
        self.assertIsNone(belfegor.moo_config.strategy)
        res = belfegor.minimize(zdt1, max_iters=20)
        self.assertEqual(res.status.exit, egx.ExitStatus.MAX_ITERS_REACHED)
        self.assertEqual(res.x_doe.shape, (30, 2))
        self.assertEqual(res.y_doe.shape, (30, 2))
        assert_non_dominated(self, res.y_pareto)
        hv = hypervolume_2d(res.y_pareto)
        print(f"ZDT1 Belfegor front HV = {hv} ({100 * hv / ZDT1_HV_REF:.1f}%)")
        self.assertGreater(hv, 0.8 * ZDT1_HV_REF)
        # compromise point is a point of the front
        self.assertTrue(
            any(np.array_equal(res.x_opt, x) for x in res.x_pareto), res.x_opt
        )
        np.testing.assert_array_equal(res.y_opt, zdt1(res.x_opt[np.newaxis])[0])
        # unpacking and result shortcuts
        x_pareto, y_pareto = res
        np.testing.assert_array_equal(x_pareto, res.result.x_pareto)
        np.testing.assert_array_equal(y_pareto, res.result.y_pareto)
        self.assertIn("BelfegorOptim(result=ParetoResult(n_pareto=", repr(res))

    def test_zdt1_strategies(self):
        for strategy, batch in [
            (egx.MooStrategy.PAREGO, 1),
            (egx.MooStrategy.EIM, 1),
            (egx.MooStrategy.EHVI, 1),
            (egx.MooStrategy.QEHVI, 3),
        ]:
            with self.subTest(strategy=strategy):
                res = egx.Belfegor(
                    ZDT1_XLIMITS,
                    moo_config=egx.MooConfig(strategy=strategy, batch=batch),
                    n_doe=10,
                    seed=42,
                ).minimize(zdt1, max_iters=8)
                self.assertEqual(res.status.exit, egx.ExitStatus.MAX_ITERS_REACHED)
                self.assertGreater(res.x_doe.shape[0], 10 + 8 * (batch - 1))
                assert_non_dominated(self, res.y_pareto)
                hv = hypervolume_2d(res.y_pareto)
                print(f"ZDT1 {strategy}: HV = {hv} ({100 * hv / ZDT1_HV_REF:.1f}%)")
                self.assertGreater(hv, 0.5 * ZDT1_HV_REF)

    def test_bnh_with_cstr_specs(self):
        belfegor = egx.Belfegor(
            [[0.0, 5.0], [0.0, 3.0]],
            cstr_specs=[egx.CstrSpec.leq(25.0), egx.CstrSpec.geq(7.7)],
            n_doe=10,
            seed=42,
        )
        res = belfegor.minimize(bnh_raw, max_iters=20)
        # raw constraint values are returned
        self.assertEqual(res.y_pareto.shape[1], 4)
        assert_non_dominated(self, res.y_pareto)
        self.assertGreaterEqual(res.y_pareto.shape[0], 5)
        for y in res.y_pareto:
            self.assertLessEqual(y[2], 25.0 + 1e-4)
            self.assertGreaterEqual(y[3], 7.7 - 1e-4)
        # front helpers interpret the constraints as the optimizer does
        self.assertEqual(
            belfegor.pareto_result(res.x_doe, res.y_doe).y_pareto.tolist(),
            res.y_pareto.tolist(),
        )

    def test_mixed_integer(self):
        def f(x):
            xr = x.copy()
            xr[:, 1] = xr[:, 1] / 9.0
            return zdt1(xr)

        xspecs = [
            egx.XSpec(egx.XType.FLOAT, [0.0, 1.0]),
            egx.XSpec(egx.XType.INT, [0, 9]),
        ]
        res = egx.Belfegor(xspecs, n_doe=10, seed=42).minimize(f, max_iters=10)
        assert_non_dominated(self, res.y_pareto)
        np.testing.assert_array_equal(res.x_pareto[:, 1], np.round(res.x_pareto[:, 1]))

    def test_hv_stop(self):
        res = egx.Belfegor(
            ZDT1_XLIMITS,
            moo_config={"hv_stop": (1e-3, 5)},
            n_doe=10,
            seed=42,
        ).minimize(zdt1, max_iters=100)
        self.assertEqual(res.status.exit, egx.ExitStatus.SOLVER_CONVERGED)
        self.assertLess(res.status.total_iters, 100)

    def test_determinism(self):
        def run():
            return egx.Belfegor(ZDT1_XLIMITS, n_doe=10).minimize(
                zdt1, max_iters=5, seed=7
            )

        np.testing.assert_array_equal(run().x_doe, run().x_doe)

    def test_ask_and_tell_batch(self):
        for strategy in [egx.MooStrategy.QEHVI, egx.MooStrategy.EHVI]:
            with self.subTest(strategy=strategy):
                belfegor = egx.Belfegor(
                    ZDT1_XLIMITS,
                    moo_config=egx.MooConfig(strategy=strategy, batch=3),
                    seed=42,
                )
                x = egx.lhs(np.array(ZDT1_XLIMITS), 10, seed=42)
                x_new = belfegor.suggest(x, zdt1(x))
                self.assertEqual(x_new.shape, (3, 2))

    def test_ask_and_tell(self):
        belfegor = egx.Belfegor(ZDT1_XLIMITS, seed=42)
        x = egx.lhs(np.array(ZDT1_XLIMITS), 10, seed=42)
        for _ in range(10):
            x_new = belfegor.suggest(x, zdt1(x))
            self.assertEqual(x_new.shape, (1, 2))
            x = np.vstack([x, x_new])
        y = zdt1(x)
        res = belfegor.pareto_result(x, y)
        assert_non_dominated(self, res.y_pareto)
        self.assertEqual(
            sorted(belfegor.pareto_indices(y)),
            [i for i, yi in enumerate(y) if yi.tolist() in res.y_pareto.tolist()],
        )
        hv = hypervolume_2d(res.y_pareto)
        print(f"ZDT1 ask-and-tell: HV = {hv} ({100 * hv / ZDT1_HV_REF:.1f}%)")
        self.assertGreater(hv, 0.5 * ZDT1_HV_REF)

    def test_pareto_helpers_match_minimize(self):
        belfegor = egx.Belfegor(ZDT1_XLIMITS, n_doe=10, seed=42)
        res = belfegor.minimize(zdt1, max_iters=5)
        indices = belfegor.pareto_indices(res.y_doe)
        np.testing.assert_array_equal(res.y_doe[indices], res.y_pareto)
        helper = belfegor.pareto_result(res.x_doe, res.y_doe)
        np.testing.assert_array_equal(helper.x_pareto, res.x_pareto)
        np.testing.assert_array_equal(helper.x_opt, res.x_opt)
        np.testing.assert_array_equal(helper.y_opt, res.y_opt)
        with self.assertRaises(ValueError):
            belfegor.pareto_indices(res.y_doe[:, :1])

    def test_moo_config(self):
        self.assertEqual(egx.MooConfig().batch, 1)
        cfg = egx.MooConfig(
            strategy=egx.MooStrategy.EIM,
            batch=2,
            eim_aggregation=egx.EimAggregation.HYPERVOLUME,
            hv_stop=(1e-3, 4),
        )
        self.assertEqual(cfg.strategy, egx.MooStrategy.EIM)
        self.assertEqual(cfg.batch, 2)
        self.assertEqual(cfg.hv_stop, (1e-3, 4))
        self.assertEqual(
            repr(cfg),
            "MooConfig(strategy=MooStrategy.EIM, batch=2, "
            "eim_aggregation=EimAggregation.HYPERVOLUME, hv_stop=(0.001, 4), "
            "rho=0.05, n_divisions=None)",
        )
        cfg.n_divisions = 5
        self.assertEqual(cfg.n_divisions, 5)
        belfegor = egx.Belfegor(
            ZDT1_XLIMITS,
            n_obj=2,
            moo_config={
                "strategy": egx.MooStrategy.PAREGO,
                "batch": 3,
                "n_divisions": 6,
            },
        )
        self.assertEqual(belfegor.moo_config.strategy, egx.MooStrategy.PAREGO)
        self.assertEqual(belfegor.moo_config.batch, 3)
        self.assertEqual(belfegor.moo_config.n_divisions, 6)
        with self.assertRaisesRegex(ValueError, "unknown moo_config key 'stragety'"):
            egx.Belfegor(ZDT1_XLIMITS, moo_config={"stragety": egx.MooStrategy.EIM})

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "n_obj should be at least 1"):
            egx.Belfegor(ZDT1_XLIMITS, n_obj=0)
        # unsupported Egor options are not Belfegor options
        with self.assertRaises(TypeError):
            egx.Belfegor(ZDT1_XLIMITS, trego=True)
        with self.assertRaises(TypeError):
            egx.Belfegor(ZDT1_XLIMITS, target=0.0)
        # batches are configured with MooConfig
        with self.assertRaises(TypeError):
            egx.Belfegor(ZDT1_XLIMITS, qei_config=egx.QEiConfig(batch=2))
        with self.assertRaisesRegex(
            ValueError, "moo_config batch should be at least 1"
        ):
            egx.Belfegor(ZDT1_XLIMITS, moo_config={"batch": 0})
        # wrong number of objectives returned by the function
        with self.assertRaisesRegex(ValueError, r"\(3 objectives \+ 0 constraints\)"):
            egx.Belfegor(ZDT1_XLIMITS, n_obj=3).minimize(zdt1, max_iters=1)
        # configuration checked by the optimizer
        with self.assertRaises(ValueError):
            egx.Belfegor(
                ZDT1_XLIMITS,
                moo_config=egx.MooConfig(strategy=egx.MooStrategy.QEHVI, batch=5),
            ).minimize(zdt1, max_iters=1)


if __name__ == "__main__":
    unittest.main()

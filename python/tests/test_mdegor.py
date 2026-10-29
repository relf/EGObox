import unittest

import numpy as np

import egobox as egx
from egobox.egmdo import Discipline, EnrichmentStrategy, Mdegor

# Classic constrained Sellar problem (C-EGMDO paper): optimum z* = (1.9776, 0, 0), f* = 3.1834
SELLAR_DESIGN = {"z1": [-10.0, 10.0], "z2": [0.0, 10.0], "z3": [0.0, 10.0]}
SELLAR_COUPLINGS = {"y1": [0.0, 25.0], "y2": [0.0, 25.0]}
SELLAR_Z_REF = np.array([1.97763888, 0.0, 0.0])
SELLAR_F_REF = 3.183393951640613


def sellar_y1(x: np.ndarray) -> np.ndarray:
    """inputs [z1, z2, z3, y2]"""
    return (x[:, 0] ** 2 + x[:, 1] + x[:, 2] - 0.2 * x[:, 3])[:, None]


def sellar_y2(x: np.ndarray) -> np.ndarray:
    """inputs [z1, z2, y1]"""
    return (np.sqrt(np.maximum(x[:, 2], 0.0)) + x[:, 0] + x[:, 1])[:, None]


def sellar_qoi(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """objective, then constraints 3.16 - y1 <= 0 and y2 - 24 <= 0"""
    f = z[:, 2] ** 2 + z[:, 1] + y[:, 0] + np.exp(-y[:, 1])
    return np.column_stack([f, 3.16 - y[:, 0], y[:, 1] - 24.0])


def sellar_disciplines(with_funs: bool = True) -> list:
    return [
        Discipline(
            "d1",
            sellar_y1 if with_funs else None,
            inputs=["z1", "z2", "z3", "y2"],
            outputs=["y1"],
        ),
        Discipline(
            "d2",
            sellar_y2 if with_funs else None,
            inputs=["z1", "z2", "y1"],
            outputs=["y2"],
        ),
    ]


def sellar_mdegor(disciplines=None, qoi=sellar_qoi, **kwargs) -> Mdegor:
    return Mdegor(
        SELLAR_DESIGN,
        SELLAR_COUPLINGS,
        sellar_disciplines() if disciplines is None else disciplines,
        qoi,
        n_ineq=2,
        **kwargs,
    )


# 1-D toy problem (EGMDO paper): minimum z* = -3.0031, f* = -1.1497
def toy_y1(x: np.ndarray) -> np.ndarray:
    """inputs [z, y2]"""
    return (x[:, 0] ** 2 - np.cos(x[:, 1] / 2.0))[:, None]


def toy_y2(x: np.ndarray) -> np.ndarray:
    """inputs [z, y1]"""
    return (x[:, 0] + x[:, 1])[:, None]


def toy_qoi(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.cos((y[:, 0] + np.exp(-y[:, 1])) / np.pi) + z[:, 0] / 20.0)[:, None]


def toy_mdegor(**kwargs) -> Mdegor:
    disciplines = [
        Discipline("d1", toy_y1, inputs=["z", "y2"], outputs=["y1"]),
        Discipline("d2", toy_y2, inputs=["z", "y1"], outputs=["y2"]),
    ]
    options = {
        "n_doe": [5, 4],
        "doe_uq": np.array([[-4.3], [-2.5], [0.5], [3.7]]),
        "max_iters": 5,
        "seed": 0,
    }
    options.update(kwargs)
    return Mdegor(
        {"z": [-5.0, 5.0]},
        {"y1": [0.0, 25.0], "y2": [0.0, 25.0]},
        disciplines,
        toy_qoi,
        **options,
    )


class TestMdegor(unittest.TestCase):
    def test_constrained_sellar(self):
        mdegor = sellar_mdegor(n_doe=[5, 5], n_mc=200, seed=0)
        self.assertEqual(mdegor.design_names, ["z1", "z2", "z3"])
        self.assertEqual(mdegor.coupling_names, ["y1", "y2"])
        res = mdegor.minimize(max_iters=10)
        # The best point is the constrained optimum
        np.testing.assert_allclose(res.z_best, SELLAR_Z_REF, atol=1e-2)
        self.assertAlmostEqual(res.f_best, SELLAR_F_REF, delta=0.01 * SELLAR_F_REF)
        # Far fewer disciplinary evaluations than a direct optimization of the coupled problem
        self.assertLessEqual(sum(res.n_evaluations), 40)
        self.assertEqual(res.design_names, ["z1", "z2", "z3"])
        self.assertEqual(res.coupling_names, ["y1", "y2"])
        self.assertEqual(res.z_star.shape, (100, 3))
        self.assertEqual(res.y_min.shape, (100,))
        self.assertEqual(res.z_star_mean.shape, (3,))
        self.assertEqual(len(res.history), 10)
        self.assertEqual(
            set(res.history[0]),
            {
                "z_new",
                "max_criterion",
                "wb2_scale",
                "n_modes",
                "n_enrichments",
                "n_failed_mdas",
                "n_dropped_points",
            },
        )
        # The training data of each discipline are its true evaluations, in its inputs order
        for (x, y), n, d, fun in zip(
            res.disciplinary_doe,
            res.n_evaluations,
            sellar_disciplines(),
            [sellar_y1, sellar_y2],
        ):
            self.assertEqual(x.shape, (n, len(d.inputs)))
            np.testing.assert_allclose(y, fun(x))
        self.assertIn("MdegorResult(", repr(res))
        # Quantities of interest and couplings at the uncertainty quantification DoE
        n_uq = res.doe_uq.shape[0]
        self.assertEqual(res.qoi_mean.shape, (n_uq, 3))
        self.assertEqual(res.qoi_std.shape, (n_uq, 3))
        self.assertEqual(res.coupling_mean.shape, (n_uq, 2))
        self.assertTrue(np.all(res.qoi_std >= 0.0))
        best = np.flatnonzero(np.all(res.doe_uq == res.z_best, axis=1))
        self.assertEqual(len(best), 1)
        self.assertEqual(res.qoi_mean[best[0], 0], res.f_best)
        self.assertTrue(res.best_feasible)
        self.assertTrue(np.all(res.qoi_mean[best[0], 1:] <= 1e-4))
        # The mean couplings solve the true MDA closely at the best point
        z, y = res.z_best, res.coupling_mean[best[0]]
        np.testing.assert_allclose(
            y[0], sellar_y1(np.array([[*z, y[1]]]))[0, 0], rtol=1e-2
        )

    def test_inputs_in_any_order(self):
        # d1 receives its inputs as [y2, z3, z1, z2]: same problem, same run
        def y1_reordered(x: np.ndarray) -> np.ndarray:
            return sellar_y1(x[:, [2, 3, 1, 0]])

        disciplines = sellar_disciplines()
        disciplines[0] = Discipline(
            "d1", y1_reordered, inputs=["y2", "z3", "z1", "z2"], outputs=["y1"]
        )
        expected = sellar_mdegor(n_doe=[5, 5], max_iters=3, n_sim=20, seed=1)
        res = sellar_mdegor(
            disciplines, n_doe=[5, 5], max_iters=3, n_sim=20, seed=1
        ).minimize()
        expected = expected.minimize()
        np.testing.assert_allclose(res.z_star, expected.z_star)
        self.assertEqual(res.n_evaluations, expected.n_evaluations)
        # The training inputs follow the discipline inputs order
        x_expected = expected.disciplinary_doe[0][0]
        np.testing.assert_allclose(
            res.disciplinary_doe[0][0], x_expected[:, [3, 2, 0, 1]]
        )

    def test_linearized_mda(self):
        # Random MDAs linearized at the mean MDA solution: same optimum
        mdegor = sellar_mdegor(n_doe=[5, 5], n_mc=200, linearized_mda=True, seed=0)
        res = mdegor.minimize(max_iters=10)
        self.assertAlmostEqual(res.f_best, SELLAR_F_REF, delta=0.01 * SELLAR_F_REF)
        self.assertLessEqual(sum(res.n_evaluations), 40)

    def test_gp_correlation(self):
        # GP posterior correlation across the uncertainty quantification points
        for linearized in [False, True]:
            mdegor = sellar_mdegor(
                n_doe=[5, 5],
                n_mc=200,
                linearized_mda=linearized,
                gp_correlation=True,
                seed=0,
            )
            res = mdegor.minimize(max_iters=10)
            self.assertAlmostEqual(res.f_best, SELLAR_F_REF, delta=0.01 * SELLAR_F_REF)
            self.assertLessEqual(sum(res.n_evaluations), 40)

    def test_toy_problem_with_egmdo_paper_settings(self):
        mdegor = toy_mdegor(
            acquisition=egx.InfillStrategy.EI,
            enrichment=EnrichmentStrategy.ALL,
            p_threshold=None,
            n_mc=100,
            gp_kernel=egx.CorrelationSpec.SQUARED_EXPONENTIAL,
        )
        res = mdegor.minimize()
        self.assertEqual(len(res.history), 5)
        self.assertTrue(all(h["wb2_scale"] is None for h in res.history))
        # All disciplines are enriched together
        self.assertEqual(res.n_evaluations[0] - 5, res.n_evaluations[1] - 4)

    def test_acquisitions(self):
        for acquisition in [
            egx.InfillStrategy.WB2,
            egx.InfillStrategy.WB2S,
            egx.InfillStrategy.EI,
            egx.InfillStrategy.LOG_EI,
            4,  # integer value of LOG_EI
        ]:
            res = toy_mdegor(acquisition=acquisition, max_iters=2, n_sim=10).minimize()
            self.assertEqual(len(res.history), 2)

    def test_ask_tell_reproduces_minimize(self):
        funs = [toy_y1, toy_y2]
        mdegor = toy_mdegor()
        expected = mdegor.minimize()
        with self.assertRaises(ValueError):
            mdegor.tell([])  # no request yet
        n_requests = 0
        while (requests := mdegor.ask()) is not None:
            n_requests += 1
            for r in requests:
                self.assertIsInstance(r, egx.MdegorRequest)
                self.assertIn(r.discipline, (0, 1))
            mdegor.tell([funs[r.discipline](r.x) for r in requests])
        self.assertIsNone(mdegor.ask())
        res = mdegor.result()
        self.assertGreater(n_requests, 0)
        np.testing.assert_array_equal(res.z_star, expected.z_star)
        self.assertEqual(res.n_evaluations, expected.n_evaluations)
        # Restart
        mdegor.reset()
        with self.assertRaises(ValueError):
            mdegor.result()

    def test_ask_tell_without_functions(self):
        mdegor = sellar_mdegor(sellar_disciplines(with_funs=False))
        with self.assertRaises(ValueError):
            mdegor.minimize()
        requests = mdegor.ask()
        self.assertEqual([r.discipline for r in requests], [0, 1])
        # Request columns follow the discipline inputs: z1, z2, z3, y2 and z1, z2, y1
        self.assertEqual([r.x.shape[1] for r in requests], [4, 3])
        for r, bounds in zip(
            requests,
            [[[-10, 10], [0, 10], [0, 10], [0, 25]], [[-10, 10], [0, 10], [0, 25]]],
        ):
            bounds = np.array(bounds)
            self.assertTrue(np.all(r.x >= bounds[:, 0]) and np.all(r.x <= bounds[:, 1]))

    def test_initial_doe_in_inputs_order(self):
        doe = [
            np.array(
                [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [0.5, 1.0, 1.5, 2.0]]
            ),
            np.array([[1.0, 2.0, 5.0], [3.0, 1.0, 9.0], [0.0, 0.5, 2.0]]),
        ]
        mdegor = sellar_mdegor(sellar_disciplines(with_funs=False), doe=doe)
        requests = mdegor.ask()
        for r, x in zip(requests, doe):
            np.testing.assert_array_equal(r.x, x)
        with self.assertRaisesRegex(
            ValueError, "doe of discipline 'd2' should have 3 columns"
        ):
            sellar_mdegor(doe=[doe[0], doe[0]])

    def test_deterministic_constraint(self):
        # Constraint on the design variables only: z1 <= 1, which moves the optimum
        def qoi(z, y):
            return np.column_stack([sellar_qoi(z, y), z[:, 0] - 1.0])

        mdegor = Mdegor(
            SELLAR_DESIGN,
            SELLAR_COUPLINGS,
            sellar_disciplines(),
            qoi,
            n_ineq=3,
            deterministic_constraints=[2],
            cstr_tol=[1e-4, 1e-4, 1e-4],
            n_doe=[5, 5],
            max_iters=3,
            n_sim=20,
            seed=0,
        )
        res = mdegor.minimize()
        self.assertLessEqual(res.z_best[0], 1.0 + 1e-4)

    def test_callback_errors(self):
        def failing_qoi(z, y):
            raise RuntimeError("qoi failure")

        with self.assertRaisesRegex(RuntimeError, "qoi failure"):
            sellar_mdegor(qoi=failing_qoi, seed=0).minimize(max_iters=1)

        def bad_shape_qoi(z, y):
            return np.zeros((z.shape[0], 1))

        with self.assertRaisesRegex(ValueError, "qoi should return an array of shape"):
            sellar_mdegor(qoi=bad_shape_qoi, seed=0).minimize(max_iters=1)

        def failing_discipline(x):
            raise ValueError("discipline failure")

        disciplines = sellar_disciplines()
        disciplines[1] = Discipline(
            "d2", failing_discipline, inputs=["z1", "z2", "y1"], outputs=["y2"]
        )
        with self.assertRaisesRegex(ValueError, "discipline failure"):
            sellar_mdegor(disciplines, seed=0).minimize(max_iters=1)

    def test_invalid_variables(self):
        d = sellar_disciplines()
        with self.assertRaisesRegex(ValueError, "'z1' is declared twice"):
            Mdegor(SELLAR_DESIGN, {"z1": [0, 1], "y1": [0, 25]}, d, sellar_qoi)
        with self.assertRaisesRegex(ValueError, "design variable 'z1': bounds"):
            Mdegor({"z1": [1.0, 0.0]}, SELLAR_COUPLINGS, d, sellar_qoi)
        with self.assertRaisesRegex(TypeError, "coupling variable 'y1': bounds"):
            Mdegor(SELLAR_DESIGN, {"y1": "range"}, d, sellar_qoi)
        with self.assertRaisesRegex(ValueError, "at least one variable"):
            Mdegor({}, SELLAR_COUPLINGS, d, sellar_qoi)

    def test_invalid_wiring(self):
        def mdegor(*disciplines):
            return Mdegor(
                SELLAR_DESIGN, SELLAR_COUPLINGS, list(disciplines), sellar_qoi, n_ineq=2
            )

        d1, d2 = sellar_disciplines()
        with self.assertRaisesRegex(ValueError, "discipline 'd3': unknown input 'y3'"):
            mdegor(d1, d2, Discipline("d3", inputs=["y3"], outputs=["y1"]))
        with self.assertRaisesRegex(ValueError, "'y2' is computed by no discipline"):
            mdegor(d1)
        with self.assertRaisesRegex(
            ValueError, "'y1' is computed by both 'd1' and 'd3'"
        ):
            mdegor(d1, d2, Discipline("d3", inputs=["z1"], outputs=["y1"]))
        with self.assertRaisesRegex(ValueError, "output 'z1' is a design variable"):
            mdegor(d1, Discipline("d2", inputs=["y1"], outputs=["z1"]))
        with self.assertRaisesRegex(ValueError, "'y1' is both an input and an output"):
            mdegor(Discipline("d1", inputs=["z1", "y1"], outputs=["y1"]), d2)
        with self.assertRaisesRegex(ValueError, "input 'z1' is given twice"):
            mdegor(Discipline("d1", inputs=["z1", "z1"], outputs=["y1"]), d2)
        with self.assertRaisesRegex(ValueError, "discipline 'd1': no input"):
            mdegor(Discipline("d1", outputs=["y1"]), d2)
        with self.assertRaisesRegex(ValueError, "discipline 'd1': no output"):
            mdegor(Discipline("d1", inputs=["z1"]), d2)
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            mdegor(Discipline("d2", inputs=["z1", "y2"], outputs=["y1"]), d2)

    def test_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "gp_kernel"):
            sellar_mdegor(gp_kernel=egx.CorrelationSpec.MATERN32)
        with self.assertRaisesRegex(ValueError, "constraint tolerances"):
            sellar_mdegor(cstr_tol=[1e-4])
        with self.assertRaises(ValueError):
            sellar_mdegor(enrichment=3)

    def test_discipline(self):
        d = Discipline("d1", sellar_y1, inputs=["z1", "z2", "z3", "y2"], outputs=["y1"])
        self.assertEqual(d.name, "d1")
        self.assertEqual(d.inputs, ["z1", "z2", "z3", "y2"])
        self.assertEqual(d.outputs, ["y1"])
        self.assertEqual(
            repr(d),
            "Discipline(name='d1', inputs=['z1', 'z2', 'z3', 'y2'], outputs=['y1'])",
        )


if __name__ == "__main__":
    unittest.main()

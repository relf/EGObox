import unittest

import numpy as np

import egobox as egx
from egobox.egmdo import Discipline, Egmdo, EnrichmentStrategy

# Classic constrained Sellar problem (C-EGMDO paper): optimum z* = (1.9776, 0, 0), f* = 3.1834
SELLAR_XLIMITS = [[-10.0, 10.0], [0.0, 10.0], [0.0, 10.0]]
SELLAR_Z_REF = np.array([1.97763888, 0.0, 0.0])
SELLAR_F_REF = 3.183393951640613


def sellar_y1(x: np.ndarray) -> np.ndarray:
    """inputs [z1, z2, z3, y2]"""
    return (x[:, 0] ** 2 + x[:, 1] + x[:, 2] - 0.2 * x[:, 3])[:, None]


def sellar_y2(x: np.ndarray) -> np.ndarray:
    """inputs [z1, z2, z3, y1]"""
    return (np.sqrt(np.maximum(x[:, 3], 0.0)) + x[:, 0] + x[:, 1])[:, None]


def sellar_qoi(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """objective, then constraints 3.16 - y1 <= 0 and y2 - 24 <= 0"""
    f = z[:, 2] ** 2 + z[:, 1] + y[:, 0] + np.exp(-y[:, 1])
    return np.column_stack([f, 3.16 - y[:, 0], y[:, 1] - 24.0])


def sellar_disciplines(with_funs: bool = True) -> list:
    return [
        Discipline(
            "y1",
            sellar_y1 if with_funs else None,
            coupling_inputs=[1],
            coupling_bounds=[[0.0, 25.0]],
            coupling_outputs=[0],
        ),
        Discipline(
            "y2",
            sellar_y2 if with_funs else None,
            coupling_inputs=[0],
            coupling_bounds=[[0.0, 25.0]],
            coupling_outputs=[1],
        ),
    ]


# 1-D toy problem (EGMDO paper): minimum z* = -3.0031, f* = -1.1497
def toy_y1(x: np.ndarray) -> np.ndarray:
    """inputs [z, y2]"""
    return (x[:, 0] ** 2 - np.cos(x[:, 1] / 2.0))[:, None]


def toy_y2(x: np.ndarray) -> np.ndarray:
    """inputs [z, y1]"""
    return (x[:, 0] + x[:, 1])[:, None]


def toy_qoi(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.cos((y[:, 0] + np.exp(-y[:, 1])) / np.pi) + z[:, 0] / 20.0)[:, None]


def toy_egmdo(**kwargs) -> Egmdo:
    disciplines = [
        Discipline(
            "y1",
            toy_y1,
            coupling_inputs=[1],
            coupling_bounds=[[0, 25]],
            coupling_outputs=[0],
        ),
        Discipline(
            "y2",
            toy_y2,
            coupling_inputs=[0],
            coupling_bounds=[[0, 25]],
            coupling_outputs=[1],
        ),
    ]
    options = {
        "n_doe": [5, 4],
        "doe_uq": np.array([[-4.3], [-2.5], [0.5], [3.7]]),
        "max_iters": 5,
        "seed": 0,
    }
    options.update(kwargs)
    return Egmdo([[-5.0, 5.0]], disciplines, toy_qoi, **options)


class TestEgmdo(unittest.TestCase):
    def test_constrained_sellar(self):
        egmdo = Egmdo(
            SELLAR_XLIMITS,
            sellar_disciplines(),
            sellar_qoi,
            n_ineq=2,
            n_doe=[5, 5],
            n_mc=200,
            seed=0,
        )
        res = egmdo.minimize(max_iters=10)
        # The best point is the constrained optimum
        np.testing.assert_allclose(res.z_best, SELLAR_Z_REF, atol=1e-2)
        self.assertAlmostEqual(res.f_best, SELLAR_F_REF, delta=0.01 * SELLAR_F_REF)
        # Far fewer disciplinary evaluations than a direct optimization of the coupled problem
        self.assertLessEqual(sum(res.n_evaluations), 40)
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
        # The training data of each discipline are its true evaluations
        for (x, y), n, fun in zip(
            res.disciplinary_doe, res.n_evaluations, [sellar_y1, sellar_y2]
        ):
            self.assertEqual(x.shape, (n, 4))
            np.testing.assert_allclose(y, fun(x))
        self.assertIn("EgmdoResult(", repr(res))

    def test_toy_problem_with_egmdo_paper_settings(self):
        egmdo = toy_egmdo(
            acquisition=egx.InfillStrategy.EI,
            enrichment=EnrichmentStrategy.ALL,
            p_threshold=None,
            n_mc=100,
            gp_kernel=egx.CorrelationSpec.SQUARED_EXPONENTIAL,
        )
        res = egmdo.minimize()
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
            res = toy_egmdo(acquisition=acquisition, max_iters=2, n_sim=10).minimize()
            self.assertEqual(len(res.history), 2)

    def test_ask_tell_reproduces_minimize(self):
        funs = [toy_y1, toy_y2]
        egmdo = toy_egmdo()
        expected = egmdo.minimize()
        with self.assertRaises(ValueError):
            egmdo.tell([])  # no request yet
        n_requests = 0
        while (requests := egmdo.ask()) is not None:
            n_requests += 1
            for r in requests:
                self.assertIsInstance(r, egx.EgmdoRequest)
                self.assertIn(r.discipline, (0, 1))
            egmdo.tell([funs[r.discipline](r.x) for r in requests])
        self.assertIsNone(egmdo.ask())
        res = egmdo.result()
        self.assertGreater(n_requests, 0)
        np.testing.assert_array_equal(res.z_star, expected.z_star)
        self.assertEqual(res.n_evaluations, expected.n_evaluations)
        # Restart
        egmdo.reset()
        with self.assertRaises(ValueError):
            egmdo.result()

    def test_ask_tell_without_functions(self):
        egmdo = Egmdo(
            SELLAR_XLIMITS, sellar_disciplines(with_funs=False), sellar_qoi, n_ineq=2
        )
        with self.assertRaises(ValueError):
            egmdo.minimize()
        requests = egmdo.ask()
        self.assertEqual([r.discipline for r in requests], [0, 1])
        self.assertEqual([r.x.shape[1] for r in requests], [4, 4])

    def test_deterministic_constraint(self):
        # Constraint on the design variables only: z1 <= 1, which moves the optimum
        def qoi(z, y):
            return np.column_stack([sellar_qoi(z, y), z[:, 0] - 1.0])

        egmdo = Egmdo(
            SELLAR_XLIMITS,
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
        res = egmdo.minimize()
        self.assertLessEqual(res.z_best[0], 1.0 + 1e-4)

    def test_callback_errors(self):
        def failing_qoi(z, y):
            raise RuntimeError("qoi failure")

        with self.assertRaisesRegex(RuntimeError, "qoi failure"):
            Egmdo(
                SELLAR_XLIMITS, sellar_disciplines(), failing_qoi, n_ineq=2, seed=0
            ).minimize(max_iters=1)

        def bad_shape_qoi(z, y):
            return np.zeros((z.shape[0], 1))

        with self.assertRaisesRegex(ValueError, "qoi should return an array of shape"):
            Egmdo(
                SELLAR_XLIMITS, sellar_disciplines(), bad_shape_qoi, n_ineq=2, seed=0
            ).minimize(max_iters=1)

        def failing_discipline(x):
            raise ValueError("discipline failure")

        disciplines = sellar_disciplines()
        disciplines[1] = Discipline(
            "y2",
            failing_discipline,
            coupling_inputs=[0],
            coupling_bounds=[[0.0, 25.0]],
            coupling_outputs=[1],
        )
        with self.assertRaisesRegex(ValueError, "discipline failure"):
            Egmdo(SELLAR_XLIMITS, disciplines, sellar_qoi, n_ineq=2, seed=0).minimize(
                max_iters=1
            )

    def test_invalid_problems(self):
        d = sellar_disciplines()
        with self.assertRaisesRegex(ValueError, "computed by several disciplines"):
            Egmdo(SELLAR_XLIMITS, [d[0], d[0]], sellar_qoi, n_ineq=2)
        with self.assertRaisesRegex(ValueError, "coupling_bounds"):
            Discipline("y1", sellar_y1, coupling_inputs=[1], coupling_outputs=[0])
        with self.assertRaisesRegex(ValueError, "gp_kernel"):
            Egmdo(
                SELLAR_XLIMITS,
                d,
                sellar_qoi,
                n_ineq=2,
                gp_kernel=egx.CorrelationSpec.MATERN32,
            )
        with self.assertRaisesRegex(ValueError, "constraint tolerances"):
            Egmdo(SELLAR_XLIMITS, d, sellar_qoi, n_ineq=2, cstr_tol=[1e-4])
        with self.assertRaisesRegex(ValueError, "continuous"):
            Egmdo([egx.XSpec(egx.XType.INT, [0, 3])] * 3, d, sellar_qoi, n_ineq=2)
        with self.assertRaises(ValueError):
            Egmdo(SELLAR_XLIMITS, d, sellar_qoi, enrichment=3)

    def test_discipline(self):
        d = Discipline(
            "y1",
            sellar_y1,
            coupling_inputs=[1],
            coupling_bounds=np.array([[0.0, 25.0]]),
            coupling_outputs=[0],
        )
        self.assertEqual(d.name, "y1")
        self.assertEqual(d.coupling_inputs, [1])
        self.assertEqual(d.coupling_outputs, [0])
        np.testing.assert_array_equal(d.coupling_bounds, [[0.0, 25.0]])
        self.assertEqual(
            repr(d), "Discipline(name='y1', coupling_inputs=[1], coupling_outputs=[0])"
        )


if __name__ == "__main__":
    unittest.main()

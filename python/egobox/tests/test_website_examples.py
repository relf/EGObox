import runpy
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class TestWebsiteExamples(unittest.TestCase):
    def test_website_egor_example(self):
        namespace = runpy.run_path(str(EXAMPLES_DIR / "website_egor.py"))
        optim = namespace["optimize_example"]()

        self.assertAlmostEqual(-15.125, optim.result.y_opt[0], delta=1e-3)
        self.assertAlmostEqual(18.935, optim.result.x_opt[0], delta=5e-2)

    def test_website_gpx_example(self):
        namespace = runpy.run_path(str(EXAMPLES_DIR / "website_gpx.py"))
        model, xtest, ytest, fig = namespace["fit_surrogate_example"]()

        self.assertIsNotNone(model)
        self.assertEqual((100, 1), xtest.shape)
        self.assertEqual(100, ytest.shape[0])
        self.assertTrue(np.isfinite(ytest).all())
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()

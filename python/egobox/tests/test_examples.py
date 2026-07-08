import runpy
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class TestExamples(unittest.TestCase):
    def test_website_egor_example(self):
        runpy.run_path(str(EXAMPLES_DIR / "rastrigin.py"))

    def test_website_gpx_example(self):
        runpy.run_path(str(EXAMPLES_DIR / "kriging.py"))


if __name__ == "__main__":
    unittest.main()

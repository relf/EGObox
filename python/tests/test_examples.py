import importlib.util
import runpy
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class TestExamples(unittest.TestCase):
    def test_website_egor_example(self):
        runpy.run_path(str(EXAMPLES_DIR / "rastrigin.py"))

    def test_website_gpx_example(self):
        runpy.run_path(str(EXAMPLES_DIR / "kriging.py"))

    def test_belfegor_example(self):
        runpy.run_path(str(EXAMPLES_DIR / "zdt1.py"))

    @unittest.skipUnless(importlib.util.find_spec("pymoo"), "pymoo is not installed")
    def test_belfegor_pymoo_example(self):
        example = runpy.run_path(str(EXAMPLES_DIR / "belfegor_pymoo.py"))
        example["main"](["--list"])
        example["main"](["zdt1", "--n-var", "3", "--max-iters", "3", "--no-show"])
        example["main"](
            [
                "bnh",
                "--strategy",
                "qehvi",
                "--batch",
                "2",
                "--max-iters",
                "2",
                "--no-show",
            ]
        )


if __name__ == "__main__":
    unittest.main()

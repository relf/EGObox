import unittest
import egobox as egx


class TestUtils(unittest.TestCase):
    def test_lhs(self):
        xspecs = [[0.0, 25.0]]
        doe = egx.lhs(xspecs, 10)
        self.assertEqual(doe.shape, (10, 1))

    def test_mixint_lhs(self):
        xspecs = [
            egx.XSpec(egx.XType.FLOAT, [0.0, 25.0]),
            egx.XSpec(egx.XType.INT, [0, 25]),
        ]
        doe = egx.lhs(xspecs, 10)
        self.assertEqual(doe.shape, (10, 2))

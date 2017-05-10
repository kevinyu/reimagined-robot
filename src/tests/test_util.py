import unittest

from utils import cartesian


class TestUtil(unittest.TestCase):
    def test_cartesian(self):
        self.assertEqual(len(cartesian(2, 2)), 2 * 2)
        self.assertEqual(len(cartesian(4, 4, 5)), 4 * 4 * 5)


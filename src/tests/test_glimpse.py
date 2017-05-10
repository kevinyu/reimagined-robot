import numpy as np

import unittest

from glimpse import glimpse


class TestGlimpse(unittest.TestCase):

    def test_take_glimpse(self):
        img = np.array([
            [1, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0],
        ])
        result = glimpse(img, 2, 2, 3)
        np.testing.assert_array_almost_equal(
            result,
            np.array([
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
            ])
        )

    def test_take_glimpse_out_of_bounds(self):
        img = np.zeros((10, 10))
        with self.assertRaises(Exception):
            glimpse(img, 4, 1, 5)

        with self.assertRaises(Exception):
            glimpse(img, 1, 4, 5)

        with self.assertRaises(Exception):
            glimpse(img, 2, 8, 5)

        with self.assertRaises(Exception):
            glimpse(img, 8, 6, 5)

    def test_take_glimpse_in_bounds(self):
        img = np.zeros((10, 10))
        glimpse(img, 3, 2, 5)
        glimpse(img, 7, 2, 5)
        glimpse(img, 5, 6, 5)
        glimpse(img, 6, 4, 5)
        glimpse(img, 7, 5, 5)

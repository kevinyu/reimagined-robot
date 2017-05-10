import numpy as np

import unittest

from datasets.mnist_scene import (
        MNISTScene,
        generate_single_digit_scene,
        generate_n_digit_scene,
        generate_several_digit_scene,
)


class TestMnistScene(unittest.TestCase):

    def setUp(self):
        self.scene = MNISTScene((6, 6), label_radius=2.0)
        self.scene.DIGIT_SIZE = 3

        self.d1 = np.array([
            [0, 255, 0],
            [0, 255, 0],
            [0, 255, 0]
        ])
        self.d2 = np.array([
            [255, 255, 255],
            [0, 0, 255],
            [0, 0, 255]
        ])

    def test_add_outside_bounds(self):
        with self.assertRaises(Exception):
            self.scene.add_digit_at(-1, 1, self.d1, 0)

        with self.assertRaises(Exception):
            self.scene.add_digit_at(2, -2, self.d1, 0)

        with self.assertRaises(Exception):
            self.scene.add_digit_at(5, 2, self.d1, 0)

        with self.assertRaises(Exception):
            self.scene.add_digit_at(4, 1, self.d1, 0)

    def test_adding_at_valid_locations(self):
        """Test we can add digit at valid locations"""
        self.scene.add_digit_at(0, 0, self.d1, 0)
        self.scene.add_digit_at(3, 3, self.d1, 0)
        self.scene.add_digit_at(0, 3, self.d2, 1)
        self.scene.add_digit_at(3, 0, self.d2, 1)

    def test_near_digits(self):
        self.scene.add_digit_at(1, 1, self.d1, 0)
        near_digits = self.scene.near_digits(within=2.0)
        np.testing.assert_array_almost_equal(
                near_digits,
                [[   0.,   0.,   0.,   0.,   0.,   0.],
                 [   0.,   0.,   1.,   1.,   0.,   0.],
                 [   0.,   1.,   1.,   1.,   1.,   0.],
                 [   0.,   1.,   1.,   1.,   1.,   0.],
                 [   0.,   0.,   1.,   1.,   0.,   0.],
                 [   0.,   0.,   0.,   0.,   0.,   0.]]
        )
        self.scene.add_digit_at(3, 3, self.d1, 0)

        near_digits = self.scene.near_digits(within=2.0)
        np.testing.assert_array_almost_equal(
                near_digits,
                [[   0.,   0.,   0.,   0.,   0.,   0.],
                 [   0.,   0.,   1.,   1.,   0.,   0.],
                 [   0.,   1.,   1.,   1.,   1.,   0.],
                 [   0.,   1.,   1.,   1.,   1.,   1.],
                 [   0.,   0.,   1.,   1.,   1.,   1.],
                 [   0.,   0.,   0.,   1.,   1.,   1.]]
        )
        # TODO test the padding

    def test_sample_near_digits(self):
        self.scene.add_digit_at(1, 1, self.d1, 0)
        near_digits = self.scene.near_digits(within=1.0)
        sampled = self.scene.sample_near_digits(n=50, within=1.0)
        self.assertIn((2, 2), sampled)
        self.assertIn((2, 3), sampled)
        self.assertIn((3, 2), sampled)
        self.assertIn((3, 3), sampled)
        self.assertEqual(len(set(sampled)), 4)

    def test_add_digit(self):
        self.scene.add_digit_at(1, 1, self.d1, 0)
        np.testing.assert_array_almost_equal(
                self.scene.img,
                [[   0.,   0.,   0.,   0.,   0.,   0.],
                 [   0.,   0., 255.,   0.,   0.,   0.],
                 [   0.,   0., 255.,   0.,   0.,   0.],
                 [   0.,   0., 255.,   0.,   0.,   0.],
                 [   0.,   0.,   0.,   0.,   0.,   0.],
                 [   0.,   0.,   0.,   0.,   0.,   0.]]
        )
        self.assertEqual(self.scene.digit_locations, [(0, (1, 1))])

    def test_adding_overlapping(self):
        self.scene.add_digit_at(1, 1, self.d1, 0)
        np.testing.assert_array_almost_equal(
                self.scene.img,
                [[   0.,   0.,   0.,   0.,   0.,   0.],
                 [   0.,   0., 255.,   0.,   0.,   0.],
                 [   0.,   0., 255.,   0.,   0.,   0.],
                 [   0.,   0., 255.,   0.,   0.,   0.],
                 [   0.,   0.,   0.,   0.,   0.,   0.],
                 [   0.,   0.,   0.,   0.,   0.,   0.]]
        )
        self.assertEqual(self.scene.digit_locations, [(0, (1, 1))])

        self.scene.add_digit_at(2, 1, self.d2, 1)
        np.testing.assert_array_almost_equal(
                self.scene.img,
                [[   0.,   0.,   0.,   0.,   0.,   0.],
                 [   0.,   0., 255.,   0.,   0.,   0.],
                 [   0., 255., 255., 255.,   0.,   0.],
                 [   0.,   0., 255., 255.,   0.,   0.],
                 [   0.,   0.,   0., 255.,   0.,   0.],
                 [   0.,   0.,   0.,   0.,   0.,   0.]]
        )
        self.assertEqual(self.scene.digit_locations, [(0, (1, 1)), (1, (2, 1))])

    def test_get_label(self):
        self.scene.add_digit_at(1, 1, self.d1, 0)
        np.testing.assert_array_almost_equal(
            self.scene.get_label_at(2.0, 2.0),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self.scene.add_digit_at(2, 1, self.d2, 1)
        np.testing.assert_array_almost_equal(
            self.scene.get_label_at(2.0, 2.0),
            np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )


import matplotlib.pyplot as plt
class TestSceneGeneration(unittest.TestCase):
    PLOT_TESTS = True

    def test_single_digit_scene(self):
        scene = generate_single_digit_scene((100, 200))
        # plt.imshow(scene.img)
        # plt.show()

    def test_n_digit_scene(self):
        scene = generate_n_digit_scene((100, 100), 10)
        # plt.imshow(scene.img)
        # plt.show()

    def test_several_digit_scene(self):
        scene = generate_several_digit_scene((100, 100), (1, 10))


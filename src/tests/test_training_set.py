import numpy as np

import unittest

from datasets.mnist_scene import MNISTScene
from datasets.training_set import (
        generate_training_set,
        take_glimpses,
        take_samples,
)


class TestGlimpsingScene(unittest.TestCase):

    def setUp(self):
        self.scene = MNISTScene((6, 6), label_radius=1.0)
        self.scene.DIGIT_SIZE = 3

        self.d1 = np.array([
            [0, 255, 0],
            [0, 255, 0],
            [0, 255, 0]
        ])


    def test_take_glimpses(self):
        self.scene.add_digit_at(1, 1, self.d1, 0)
        self.assertTrue(np.all(
            [np.any(x) for x in take_glimpses(self.scene, n_glimpses=5, strategy="smart")[0]]
        ))


class TestSamplingScene(unittest.TestCase):

    def setUp(self):
        self.scene = MNISTScene((6, 6), label_radius=2.0)
        self.scene.DIGIT_SIZE = 3

        self.d1 = np.array([
            [0, 255, 0],
            [0, 255, 0],
            [0, 255, 0]
        ])

    def test_take_samples(self):
        self.scene.add_digit_at(1, 1, self.d1, 0)
        self.assertTrue(np.all(
            [x[0] == 1.0 for x in take_samples(self.scene, n_samples=10, strategy="smart")[0]]
        ), "All of the samples should come within the label radius")

        self.assertFalse(np.all(
            [x[0] == 1.0 for x in take_samples(self.scene, n_samples=10, strategy="smart", within=3.0)[0]]
        ), "Some samples should be outside the label radius now")

        # dont have a great test for the uniform case though
        take_samples(self.scene, n_samples=10, strategy="uniform")

    def test_generate_training_set(self):
        result = generate_training_set(
                img_shape=(64, 64),
                n_scenes=2,
                n_glimpses=2, 
                n_samples=2)
        # TODO generate plots here


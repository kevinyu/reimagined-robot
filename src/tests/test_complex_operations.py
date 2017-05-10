import unittest

import numpy as np
import theano
import theano.tensor as T

from position_encoding import L, L_inv
from glimpse import accumulate_glimpses
from utils.complex import complex_multiply, complex_dot, complex_map
    

class TestPositionEncoding(unittest.TestCase):

    def setUp(self):
        self.x = np.array([
            [0.0, -0.7, -1.0],
            [0.5, 0.0, -0.3],
        ])

    def test_encode_position(self):
        """Test encoding and decoding returns original value"""
        r_real, r_imag = L(self.x)
        np.testing.assert_array_almost_equal(self.x, L_inv(r_real, r_imag))

    def test_encode_big_position(self):
        # TODO: figure out what determines the bounds of the positions I can encode...
        # it must have to do with the magnitude of the encoding matrix K
        x = np.array([[0.8], [3.0]])
        r_real, r_imag = L(x)
        np.testing.assert_array_almost_equal(x, L_inv(r_real, r_imag))

    def test_vector_addition(self):
        """Test that adding two vectors corresponds to multiplication"""
        x0 = self.x[:, 0:1]
        x1 = self.x[:, 1:2]
        x2 = self.x[:, 2:3]

        r0 = L(x0)
        r1 = L(x1)
        r2 = L(x2)

        r01 = complex_multiply(r0, r1)
        r012 = complex_multiply(r01, r2)

        np.testing.assert_array_almost_equal(x0 + x1, L_inv(*r01))
        np.testing.assert_array_almost_equal(x0 + x1 + x2, L_inv(*r012))


class TestComplexOperation(unittest.TestCase):

    def test_multiply(self):
        v1 = T.fvectors("v1_real", "v1_imag")
        v2 = T.fvectors("v2_real", "v2_imag")

        fn = theano.function(inputs=v1 + v2, outputs=complex_multiply(v1, v2), allow_input_downcast=True)

        result = fn(
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]), 
            np.array([0.5, 0.2]),
            np.array([0.1, 0.8]))

        np.testing.assert_array_almost_equal(
                result[0],
                np.array([0.5, -0.8]))
        np.testing.assert_array_almost_equal(
                result[1],
                np.array([0.1, 0.2]))

    def test_dot_product(self):
        v1 = T.fvectors("v1_real", "v1_imag")
        v2 = T.fvectors("v2_real", "v2_imag")

        fn = theano.function(inputs=v1 + v2, outputs=complex_dot(v1, v2), allow_input_downcast=True)

        result = fn(
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]), 
            np.array([0.5, 0.2]),
            np.array([0.1, 0.8]))

        np.testing.assert_almost_equal(result[0], 1.3)
        np.testing.assert_almost_equal(result[1], 0.1)

    def test_map(self):
        m = T.fmatrices("m1", "m2")
        y = complex_map(m, lambda s: s.dimshuffle(1, 0, "x"))

        fn = theano.function(
                inputs=m,
                outputs=y,
                allow_input_downcast=True)
        
        result = fn(np.ones((3, 4)), np.ones((3, 4)))
        self.assertEqual(result[0].shape, (4, 3, 1))
        self.assertEqual(result[1].shape, (4, 3, 1))


class TestGlimpses(unittest.TestCase):

    def setUp(self):
        self.S_0 = (np.arange(5), np.zeros(5))
        self.glimpses = (
                np.array([
                    -1.0 * np.ones(5),
                    1.0 * np.ones(5),
                    2.0 * np.ones(5)
                ]),
                np.zeros((3, 5))
        )
        self.glimpse_positions = (
                np.array([
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0, 1.0],
                ]),
                np.zeros((3, 5))
        )

    def test_accumulate_glimpses(self):
        glimpse_vectors = T.fmatrices("glimpse_vectors_real", "glimpse_vectors_imag")
        glimpse_positions = T.fmatrices("glimpse_positions_real", "glimpse_positions_imag")
        S_0 = T.fvectors("S_0_real", "S_0_imag")

        S = accumulate_glimpses(S_0, glimpse_vectors, glimpse_positions)

        fn = theano.function(
            inputs=list(S_0) + list(glimpse_vectors) + list(glimpse_positions),
            outputs=S,
            allow_input_downcast=True)

        S_result = fn(*(self.S_0 + self.glimpses + self.glimpse_positions))

        np.testing.assert_array_almost_equal(
                S_result[0],
                np.array([
                    [-1.0, 0.0, 2.0, 3.0, 4.0],
                    [-1.0, 0.0, 3.0, 4.0, 4.0],
                    [-1.0, 0.0, 3.0, 2.0, 6.0],
                ]))



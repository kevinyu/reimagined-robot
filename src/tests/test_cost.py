import unittest

import config

import numpy as np
import theano
import theano.tensor as T

from cost import (
    apply_categorical_crossentropy,
    apply_softmax_to_digit_similarities,
    compute_cost_of_batch,
    evaluate_digit_similarities,
    unbind_sample_locations,
)


def vec(n=config.DIM):
    x = np.random.normal(size=n).astype(theano.config.floatX) + 1j * np.random.normal(size=n).astype(theano.config.floatX)
    return x / np.linalg.norm(x)


class TestCostFunctions(unittest.TestCase):

    def test_unbind_sample_locations(self):
        S = T.fmatrices("S_real", "S_imag")
        sample_locations = T.fmatrices("r_real", "r_imag")

        unbound = unbind_sample_locations(S, sample_locations)

        fn = theano.function(
                inputs=S + sample_locations,
                outputs=unbound,
                allow_input_downcast=True)

        s = (
            np.array([
                [1.0, 0.5, 0.5],
                [0.5, 1.0, 0.5]
            ]),
            np.zeros((2, 3))
        )
        r = (
            np.array([
                [1.0, 1.0, 0.0],
                [-1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0],
            ]),
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]),
        )

        result = fn(*(s + r))

        self.assertEqual(result[0].shape, (2, 3, 3))

        # test conjugate taken
        np.testing.assert_array_almost_equal(
                result[1][0, 0, :],
                np.array([-1.0, 0.0, 0.0]))

        # first glimpse by first sample
        np.testing.assert_array_almost_equal(
                result[0][0, 0, :],
                np.array([1.0, 0.5, 0.0]))

        # first glimpse by second sample
        np.testing.assert_array_almost_equal(
                result[0][0, 1, :],
                np.array([-1.0, -0.5, -0.5]))

        # second glimpse by first sample
        np.testing.assert_array_almost_equal(
                result[0][1, 0, :],
                np.array([0.5, 1.0, 0.0]))


    def test_evaluate_digit_similarities(self):
        S = [T.ftensor3("S_real"), T.ftensor3("S_imag")]
        digits = T.fmatrices("digits_real", "digits_imag")

        sim = evaluate_digit_similarities(S, digits)

        fn = theano.function(
            inputs=S + digits,
            outputs=sim,
            allow_input_downcast=True)

        # test imaginary 1 glimpse, 2 samples, 3 dimensional vectors
        test_s = (
            np.array([[
                [2.0, -1.0, 1.0],
                [0.2, 1.0, 0.1],
            ]]),
            np.zeros((1, 2, 3))
        )

        # imagine just 3 digits
        test_digits = (
            np.array([
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]),
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        )

        result = fn(*(test_s + test_digits))

        # commend out parts wouldve applied to the imaginary part

        self.assertEqual(result.shape, (1, 2, 3))
        # self.assertEqual(result[1].shape, (1, 2, 3))

        # similarity of glimpse 0 sample 0 to first digit
        self.assertEqual(result[0, 0, 0], 4.0)
        # self.assertEqual(result[1][0, 0, 0], -2.0)

        # similarity of glimpse 0 sample 1 to first digit
        self.assertEqual(result[0, 0, 1], -1.0)
        # self.assertEqual(result[1][0, 0, 1], 1.0)

    def test_softmax(self):
        sim = T.ftensor3("sim")
        softmax_output = apply_softmax_to_digit_similarities(sim)

        fn = theano.function(
                inputs=[sim],
                outputs=softmax_output,
                allow_input_downcast=True)

        test_sim = np.array([
            [[10.0, 1.0, 0.0],
             [5.0, 5.0, 6.0],
             [5.0, 5.0, 5.0]]
        ])
        result = fn(test_sim)

        row_1 = np.exp(test_sim[0, 0])
        np.testing.assert_array_almost_equal(result[0, 0], row_1 / row_1.sum())
        row_2 = np.exp(test_sim[0, 1])
        np.testing.assert_array_almost_equal(result[0, 1], row_2 / row_2.sum())
        row_3 = np.exp(test_sim[0, 2])
        np.testing.assert_array_almost_equal(result[0, 2], row_3 / row_3.sum())

    def test_categorical_crossentropy(self):
        softmax_output = T.ftensor3("softmax")
        labels = T.ftensor3("labels")
        xent = apply_categorical_crossentropy(softmax_output, labels)

        fn = theano.function(
                inputs=[softmax_output, labels],
                outputs=xent,
                allow_input_downcast=True)

        test_softmax_output = np.array([
            [[0.8, 0.1, 0.1],
             [0.3, 0.3, 0.4]]
        ])

        test_labels = np.array([
            [[1.0, 0.0, 0.0],
             [0.3, 0.3, 0.4]]
        ])

        result = fn(test_softmax_output, test_labels)

        np.testing.assert_array_almost_equal(result[0, 0], -np.log(0.8))
        np.testing.assert_array_almost_equal(result[0, 1], 1.08890009)


class TestBatch(unittest.TestCase):
    digits = theano.shared(np.real(np.array([vec(config.DIM) for _ in range(11)]).T))

    def test_everything(self):
        S0 = T.fvectors("S_real", "S_imag")
        digits = (self.digits, T.zeros_like(self.digits))
        batch_raw_glimpses = (T.ftensor3("raw_glimpses_real"), T.ftensor3("raw_glimpses_imag"))
        batch_glimpse_positions = (T.ftensor3("glimpse_positions_real"), T.ftensor3("glimpse_positions_imag"))
        batch_sample_labels = T.ftensor3("sample_labels")
        batch_sample_positions = (T.ftensor3("sample_positions_real"), T.ftensor3("sample_positions_imag"))

        cost = compute_cost_of_batch(S0, batch_raw_glimpses, batch_glimpse_positions,
                batch_sample_labels, batch_sample_positions, digits)

        inputs = (tuple(S0) + batch_raw_glimpses + batch_glimpse_positions + (batch_sample_labels,) +
                batch_sample_positions)

        fn = theano.function(
                inputs=inputs,
                outputs=cost,
                allow_input_downcast=True)

        a = (np.zeros(config.DIM), np.zeros(config.DIM))
        c = (
                np.random.choice([0, 1], size=(10, 3, config.DIM)),
                np.random.choice([0, 1], size=(10, 3, config.DIM))
                )
        d = (
                np.random.choice([0, 1], size=(10, 3, config.DIM)),
                np.random.choice([0, 1], size=(10, 3, config.DIM))
                )
        e = np.zeros((10, 50, 11))
        e[:, :, 0] = 1
        f = (
                np.random.choice([0, 1], size=(10, 50, config.DIM)),
                np.random.choice([0, 1], size=(10, 50, config.DIM))
                )

        result = fn(*(a + c + d + (e,) + f))



import numpy as np
import theano
import theano.tensor as T

import config
from glimpse import accumulate_glimpses_over_batch
from network import (
    glimpse_features,
    glimpse_features_batch,
    glimpse_network_output,
    network_params,
    networks
)
from optimizers import adam
from position_encoding import L
from tasks.mnist.parameters import S0, learn_params, D_table
from tasks.mnist.query_scene import D_directions
from utils.complex import ComplexTuple
from utils.unitary import theta

# This code is necessarily more complex than the basic code for glimpsing
# in order to allow for parallel processing of batches

# 2D glimpse positions (batch_size x n_glimpses x 2)
glimpse_positions = T.ftensor3("glimpse_positions")
glimpse_positions_hd = L.encode(config.POS_SCALE(glimpse_positions)k


# 2D sample positions (batch_size x n_samples x 2)
# batch_size x n_queries
query_directions = T.imatrix("query_direction_idx")
query_digits = T.imatrix("query_digits_idx")
query_colors = T.imatrix("query_colors_idx")

# One-hot labels at each sample position (batch_size x n_samples x 11)
query_labels = T.ftensor3("query_labels")

# FIXME dont dupilcate this somehow?
# One-hot labels at each sample position (batch_size x n_samples x 11)
query_labels2 = T.ftensor3("query_labels2")

_batch_size, _n_glimpses, _ = glimpse_positions.shape

# Convert to batch_size x n_glimpses x n_samples x DIM tensor
S_orig = glimpse_network_output.reshape((_batch_size, _n_glimpses, config.DIM))
S = accumulate_glimpses_over_batch(S0, S_orig, glimpse_positions_hd)
S = S.get_columns([-3, -2, -1])

# each of these are N x n_queries
_direction_vectors = D_directions.get_columns(query_directions).dimshuffle(1, 2, 0)
_digit_vectors = D_table["Digits"].get_columns(query_digits).dimshuffle(1, 2, 0)
_color_vectors = D_table["Color"].get_columns(query_colors).dimshuffle(1, 2, 0)


# all possible queries (batch_size x N x n_possible_queries
query_vectors = (_direction_vectors.conj * (_digit_vectors + _color_vectors))

# TODO we should split up here the training of the multipel sreams
S = S.dimshuffle(0, 1, "x", 2)
queried = S.conj * S * query_vectors.dimshuffle(0, "x", 1, 2)

# FIXME split this better?
digit_similarity = queried.dot(D_table["Digits"]).real
_batch_size, _n_glimpses, _n_samples, _n_digits = digit_similarity.shape

sampled_belief = T.nnet.softmax(
        digit_similarity.reshape((_batch_size * _n_glimpses * _n_samples, _n_digits))
).reshape((_batch_size, _n_glimpses, _n_samples, _n_digits ))

queried_labels = query_labels.dimshuffle(0, "x", 1, 2)

# Cross entropy calculation at all sampled points
cost = (-T.sum(queried_labels * T.log(sampled_belief), axis=1)).mean()

# FIXME do something else different
color_similarity = queried.dot(D_table["Color"]).real
_batch_size, _n_glimpses, _n_samples, _n_colors = color_similarity.shape

sampled_belief = T.nnet.softmax(
        color_similarity.reshape((_batch_size * _n_glimpses * _n_samples, _n_colors))
).reshape((_batch_size, _n_glimpses, _n_samples, _n_colors ))

queried_labels = query_labels2.dimshuffle(0, "x", 1, 2)
cost += (-T.sum(queried_labels * T.log(sampled_belief), axis=1)).mean()

from tasks.mnist.query_scene import learn_directions

params = network_params + learn_params + learn_directions + [theta]

updates = adam(cost, params)

train = theano.function(
        inputs=[
            glimpse_features,
            glimpse_positions,
            query_directions,
            query_digits,
            query_colors,
            query_labels,
            query_labels2
        ],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True)


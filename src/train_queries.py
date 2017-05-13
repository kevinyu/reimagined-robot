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
from parameters import S0, D, learn_params


# This code is necessarily more complex than the basic code for glimpsing
# in order to allow for parallel processing of batches

# 2D glimpse positions (batch_size x n_glimpses x 2)
glimpse_positions = T.ftensor3("glimpse_positions")
glimpse_positions_hd = L.encode(glimpse_positions / config.POS_SCALE)
# 2D sample positions (batch_size x n_samples x 2)
query_vectors = T.ftensor3("query_vectors")
# One-hot labels at each sample position (batch_size x n_samples x 11)
query_labels = T.ftensor3("query_labels")

# FIXME dont dupilcate this somehow?
# One-hot labels at each sample position (batch_size x n_samples x 11)
query_labels2 = T.ftensor3("query_labels")

_batch_size, _n_glimpses, _ = glimpse_positions.shape

# Convert to batch_size x n_glimpses x n_samples x DIM tensor
S = glimpse_network_output.reshape((_batch_size, _n_glimpses, config.DIM))
S = accumulate_glimpses_over_batch(S0, S, glimpse_positions_hd)


# TODO we should split up here the training of the multipel sreams
S = S.dimshuffle(0, 1, "x", 2) * query_vectors.dimshuffle(0, "x", 1, 2)

# FIXME split this better?
digit_similarity = S.dot(D_table["Digits"]).real
_batch_size, _n_glimpses, _n_samples, _n_digits = digit_similarity.shape

sampled_belief = T.nnet.softmax(
        digit_similarity.reshape((_batch_size * _n_glimpses * _n_samples, _n_digits))
).reshape((_batch_size, _n_glimpses, _n_samples, _n_digits ))

query_labels = query_labels.dimshuffle(0, "x", 1, 2)

# Cross entropy calculation at all sampled points
cost = (-T.sum(query_labels * T.log(sampled_belief), axis=1)).mean()

# FIXME do something else different
color_similarity = S.dot(D_table["Color"]).real
_batch_size, _n_glimpses, _n_samples, _n_colors = color_similarity.shape

sampled_belief = T.nnet.softmax(
        color_similarity.reshape((_batch_size * _n_glimpses * _n_samples, _n_colors))
).reshape((_batch_size, _n_glimpses, _n_samples, _n_colors ))

query_labels2 = query_labels2.dimshuffle(0, "x", 1, 2)
cost += (-T.sum(query_labels2 * T.log(sampled_belief), axis=1)).mean()

from tasks.mnist.query_scene import learn_directions

params = network_params + learn_params + learn_directions

updates = adam(cost, params)

train = theano.function(
        inputs=[
            glimpse_features,
            glimpse_positions,
            query_vectors,
            query_labels,
            query_labels2
        ],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True)


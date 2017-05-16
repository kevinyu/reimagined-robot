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
sample_positions = T.ftensor3("sample_positions")
sample_positions_hd = L.encode(sample_positions / config.POS_SCALE)
# One-hot labels at each sample position (batch_size x n_samples x 11)
sample_labels = T.ftensor3("sample_labels")

_batch_size, _n_glimpses, _ = glimpse_positions.shape




# Convert to batch_size x n_glimpses x n_samples x DIM tensor
S = glimpse_network_output.reshape((_batch_size, _n_glimpses, config.DIM))
S = accumulate_glimpses_over_batch(S0, S, glimpse_positions_hd)
S = S.get_columns([-3, -2, -1])


S = S.dimshuffle(0, 1, "x", 2) * sample_positions_hd.dimshuffle(0, "x", 1, 2).conj

similarity = S.dot(D).real
_batch_size, _n_glimpses, _n_samples, _n_digits = similarity.shape

sampled_belief = T.nnet.softmax(
        similarity.reshape((_batch_size * _n_glimpses * _n_samples, _n_digits))
).reshape((_batch_size, _n_glimpses, _n_samples, _n_digits ))

sampled_labels = sample_labels.dimshuffle(0, "x", 1, 2)

# Cross entropy calculation at all sampled points
cost = (-T.sum(sampled_labels * T.log(sampled_belief), axis=1)).mean()

params = network_params + learn_params

updates = adam(cost, params)

train = theano.function(
        inputs=[
            glimpse_features,
            glimpse_positions,
            sample_positions,
            sample_labels
        ],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True)


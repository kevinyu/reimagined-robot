import numpy as np
import theano
import theano.tensor as T

import config
from glimpse import accumulate_glimpses_over_batch
from network import (
    glimpse_network,
    glimpse_features,
    glimpse_features_batch,
    glimpse_network_output_batch,
)
from optimizers import adam
from parameters import S0, D
from position_encoding import L


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
S = glimpse_network.output.reshape((_batch_size, _n_glimpses, config.DIM))
S = accumulate_glimpses_over_batch(S0, S, glimpse_positions_hd)
S = S.dimshuffle(0, 1, "x", 2) * sample_positions_hd.dimshuffle(0, "x", 1, 2).conj

similarity = S.dot(D).real
_batch_size, _n_glimpses, _n_samples, _n_digits = similarity.shape

sampled_belief = T.nnet.softmax(
        similarity.reshape((_batch_size * _n_glimpses * _n_samples, _n_digits))
).reshape((_batch_size, _n_glimpses, _n_samples, _n_digits ))

sampled_labels = sample_labels.dimshuffle(0, "x", 1, 2)

digit_magnitudes = T.sqrt((D * D.conj).real.sum(axis=0))
digit_norms = T.outer(digit_magnitudes, digit_magnitudes)
mean_digit_similarity = (D.T.dot(D.conj).real / digit_norms).mean()

# TODO debut the digit similarity cost
cost = (
    (-T.sum(sampled_labels * T.log(sampled_belief), axis=1)).mean() +
    config.DICT_SIM_ALPHA * mean_digit_similarity
)

if config.LEARN_D:
    if config.SHAPES:
        params = glimpse_network.params + [S0.real, S0.imag]
        from parameters import D_table
        for k, v in D_table.items():
            params += [v.real, v.imag]
    else:
        params = glimpse_network.params + [S0.real, S0.imag, D.real, D.imag]
else:
    params = glimpse_network.params + [S0.real, S0.imag]


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


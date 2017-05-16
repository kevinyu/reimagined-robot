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
from tasks.mnist.query_scene import D_directions, learn_directions
from utils.complex import ComplexTuple
from utils.unitary import theta
from visualize import X, miniX, Xinv, mini_scale



# 2D glimpse positions (batch_size x n_glimpses x 2)
glimpse_positions = T.ftensor3("glimpse_positions")
glimpse_positions_hd = L.encode(glimpse_positions / config.POS_SCALE)




# One-hot labels at each sample position (batch_size x n_samples x 11)
query_labels = []
for stream in config.STREAMS:
    query_labels.append(T.ftensor3("query_labels"))

_batch_size, _n_glimpses, _ = glimpse_positions.shape




# Convert to batch_size x n_glimpses x n_samples x DIM tensor
S_orig = glimpse_network_output.reshape((_batch_size, _n_glimpses, config.DIM))
S = accumulate_glimpses_over_batch(S0, S_orig, glimpse_positions_hd)
S = S.get_columns([-3, -2, -1])
S = S.dimshuffle(0, 1, "x", 2)


# each of these are N x n_queries
if config.TRAIN_TYPE == "query-based":
    # 2D sample positions (batch_size x n_samples x 2)
    # batch_size x n_queries
    query_directions = T.imatrix("query_direction_idx")
    query_digits = T.imatrix("query_digits_idx")
    query_colors = T.imatrix("query_colors_idx")

    _direction_vectors = D_directions.get_columns(query_directions).dimshuffle(1, 2, 0)
    _digit_vectors = D_table["Digits"].get_columns(query_digits).dimshuffle(1, 2, 0)
    _color_vectors = D_table["Color"].get_columns(query_colors).dimshuffle(1, 2, 0)
    # this is where i first unbind digit+color. sharpen up in
    # spatial domain and pass thru sigmoid.
    # then go back to HD domain and bind
    _unbound_loc = S * (_digit_vectors + _color_vectors).conj.dimshuffle(0, "x", 1, 2)

    # (X Y N) dot (batch glimpses queries N)
    # (X Y N) dot (N batch glimpses queries)
    _unbound_map = miniX.dot(_unbound_loc.dimshuffle(0, 1, 3, 2)).real
    _unbound_belief = T.nnet.sigmoid(_unbound_map)
    # (X Y batch glimpses queries)

    a, b, c, d, e = _unbound_belief.shape
    _unbound_belief = ComplexTuple(_unbound_belief, T.zeros_like(_unbound_belief))

    _unbound_loc = Xinv.conj.reshape(
            (config.DIM, (config.IMG_WIDTH / mini_scale) * (config.IMG_HEIGHT / mini_scale))
    ).dot(
            _unbound_belief.reshape((a*b, c, d, e)).dimshuffle(1, 2, 0, 3)
    ).dimshuffle(1, 2, 3, 0)

    query_loc = _direction_vectors.dimshuffle(0, "x", 1, 2) * _unbound_loc
    # query_vectors = (_direction_vectors.conj * (_digit_vectors + _color_vectors))

    queried = S * query_loc.conj
else:
    # 2D sample positions (batch_size x n_samples x 2)
    sample_positions = T.ftensor3("sample_positions")
    sample_positions_hd = L.encode(sample_positions / config.POS_SCALE)

    queried = S * sample_positions_hd.dimshuffle(0, "x", 1, 2).conj


query_belief_fns = []
cost = 0.0
for stream, q_labels in zip(config.STREAMS, query_labels):
    # all possible queries (batch_size x N x n_possible_queries
    similarity = queried.dot(D_table[stream]).real
    _batch_size, _n_glimpses, _n_samples, _n_choices = similarity.shape

    sampled_belief = T.nnet.softmax(
            similarity.reshape((_batch_size * _n_glimpses * _n_samples, _n_choices))
    ).reshape((_batch_size, _n_glimpses, _n_samples, _n_choices ))

    if config.TRAIN_TYPE == "query-based":
        query_belief_fns.append(theano.function(
            [glimpse_features, glimpse_positions, query_directions, query_digits, query_colors],
            sampled_belief,
            allow_input_downcast=True
        ))

    queried_labels = q_labels.dimshuffle(0, "x", 1, 2)

    # Cross entropy calculation at all sampled points
    cost += (-T.sum(queried_labels * T.log(sampled_belief), axis=1)).mean()



if config.TRAIN_TYPE == "query-based":
    params = network_params + learn_params + learn_directions#  + [theta]
    updates = adam(cost, params)

    train = theano.function(
            inputs=[
                glimpse_features,
                glimpse_positions,
                query_directions,
                query_digits,
                query_colors,
            ] + query_labels,
            outputs=cost,
            updates=updates,
            allow_input_downcast=True)
else:
    params = network_params + learn_params#  + [theta]
    updates = adam(cost, params)
    train = theano.function(
            inputs=[
                glimpse_features,
                glimpse_positions,
                sample_positions,
            ] + query_labels,
            outputs=cost,
            updates=updates,
            allow_input_downcast=True)


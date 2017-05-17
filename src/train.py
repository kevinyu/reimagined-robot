import theano

import config
from cost import cost, glimpse_positions, query_labels
from network import glimpse_features, network_params
from optimizers import adam
from queries.location_queries import sample_positions
from queries.spatial_queries import (
        query_directions,
        query_digits,
        query_colors
)
from words import S0, D_table


learn_params = [
        S0.real,
        S0.imag,
        D_table["Digits"].real,
        D_table["Digits"].imag,
        D_table["Color"].real,
        D_table["Color"].imag
]


if config.TRAIN_TYPE == "query-based":
    learn_directions = [D_table["Directions"].real, D_table["Directions"].imag]
    # params = network_params + learn_params + learn_directions
    # TODO: we should be able to learn everything at once
    params = learn_directions
    updates = adam(cost, params)
    train = theano.function(
            inputs=[
                glimpse_features,
                glimpse_positions,
                query_directions,
                query_digits,
                query_colors
            ] + query_labels,
            outputs=cost,
            updates=updates,
            allow_input_downcast=True)
else:
    params = network_params + learn_params
    updates = adam(cost, params)
    train = theano.function(
            inputs=[
                glimpse_features,
                glimpse_positions,
                sample_positions
            ] + query_labels,
            outputs=cost,
            updates=updates,
            allow_input_downcast=True)

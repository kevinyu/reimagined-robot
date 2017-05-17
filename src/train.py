import theano

import config
from cost import cost, glimpse_positions, query_labels
from network import glimpse_features, network_params
from optimizers import adam
from parameters import learn_params
from queries.location_queries import sample_positions
from queries.spatial_queries import (
        query_directions,
        query_digits,
        query_colors
)


if config.TRAIN_TYPE == "query-based":
    params = network_params + learn_params + learn_directions
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

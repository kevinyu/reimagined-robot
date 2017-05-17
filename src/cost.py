"""Initialize the cost variable that will be used for training

The cost function is computed by querying the scene representation
for certain information after it has had a chance to gather
information about the scene from a series of glimpses.

This cost function is computed by evaluating the representation's
belief of information at particular locations and comparing them
to ground truth one-hot labels of what is at that location.
The locations queried can be part of the training data itself,
or can come from a more complex querying procedure implemented
in the `queries` module.
"""
import theano.tensor as T

import config
from glimpse import accumulate_glimpses_over_batch
from network import glimpse_network_output
from position_encoding import L
from utils import tensor_softmax
from words import S0, D_table

# (batch_size, n_queries, 2)
if config.TRAIN_TYPE == "query-based":
    from queries.spatial_queries import query_at_position
else:
    from queries.location_queries import query_at_position

# Glimpse positions (batch_size, n_glimpses, 2)
# Encoded positions in HD (batch_size, n_glimpses, N)
glimpse_positions = T.ftensor3("glimpse_positions")
glimpse_positions_hd = L.encode(config.POS_SCALE(glimpse_positions))
_batch_size, _n_glimpses, _ = glimpse_positions.shape

# Output of encoded glimpse data (batch_size, n_glimpses, N)
# Produce (batch_size, n_glimpses_seen, N) vectors representing the
# memory vectors after each glimpse per scene
glimpse_outputs = glimpse_network_output.reshape(
        (_batch_size, _n_glimpses, config.DIM)
)
S = accumulate_glimpses_over_batch(
        S0,
        glimpse_outputs,
        glimpse_positions_hd)

# Filter down to which scene memories will be learned from
# (i.e. can exclude scene vectors from early in training)
# Convert to (batch_size, n_glimpses, n_queries, N)
S = S.get_columns([-3, -2, -1])
S = S.dimshuffle(0, 1, "x", 2)

# One-hot labels at each sample position (batch_size x n_samples x 11)
# for each processing stream!
query_labels = []
query_answer_dicts = []
for stream in config.STREAMS:  # FIXME: should streams only refer to the encoding process?
    query_labels.append(T.ftensor3("query_labels"))
    # FIXME: this abstraction will leak once D_table doesnt have every property dictionary under the sun!
    query_answer_dicts.append(D_table[stream])

# Unbind positions from each scene memory
# (batch_size, n_glimpses, n_queries, N)
queried = query_at_position(S)

cost = 0.0
for query_label, query_dict in zip(
        query_labels, query_answer_dicts):

    # first compute the similarity using the dot product
    # (batch_size, n_glimpses, n_queries, n_choices)
    similarity = queried.dot(query_dict).real
    belief = tensor_softmax(similarity)

    # compute cross entropy between belief and one-hot labels
    query_label = query_label.dimshuffle(0, "x", 1, 2)
    cost += (-T.sum(query_label * T.log(belief), axis=1)).mean()

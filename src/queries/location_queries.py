import theano.tensor as T

import config
from position_encoding import L
from tasks.mnist.parameters import D_table


sample_positions = T.ftensor3("sample_positions")
# FIXME: put the POS_SCALE function into different place
sample_positions_hd = L.encode(config.POS_SCALE(sample_positions))


def query_at_position(S):
    """Returns queried positions

    Get a spatial map representing the reference object
    6. unbind scene with position vector to get answer
    """
    return S * sample_positions_hd.dimshuffle(0, "x", 1, 2).conj

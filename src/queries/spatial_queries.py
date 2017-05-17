import numpy as np
import theano.tensor as T

import config
from position_encoding.fft import fft_keepdims
from words import D_table


query_directions = T.imatrix("query_direction_idx")
query_digits = T.imatrix("query_digits_idx")
query_colors = T.imatrix("query_colors_idx")

_direction_vectors = (D_table["Directions"]
        .get_columns(query_directions)
        .dimshuffle(1, 2, 0))
_digit_vectors = (D_table["Digits"]
        .get_columns(query_digits)
        .dimshuffle(1, 2, 0))
_color_vectors = (D_table["Color"]
        .get_columns(query_colors)
        .dimshuffle(1, 2, 0))


def query_at_position(S):
    """Returns queried positions

    Get a spatial map representing the reference object
    1. generate spatial map corresponding to digit info
    2. generate spatial map corresponding to color info
    3. elementwise multiply in spatial domain to get reference location
    4. bind with relevant direction vector to get region of interest
    5. fft region of interest to get position-like vector
    6. unbind scene with new position vector to get answer
    """
    # batch_size, n_glimpses, n_queries, N
    _reference_digit = S * _digit_vectors.conj.dimshuffle(0, "x", 1, 2)
    _reference_color = S * _color_vectors.conj.dimshuffle(0, "x", 1, 2)

    batch_size, n_glimpses, n_queries, _ = _reference_digit.real.shape
    n = int(np.sqrt(config.DIM))

    reshaper = (batch_size * n_glimpses * n_queries, n, n)

    _reference_map = (
            fft_keepdims(_reference_digit.reshape(reshaper), inverse=True).real *
            fft_keepdims(_reference_color.reshape(reshaper), inverse=True).real
    )  # outcome is (_, n, n)
    # TODO: cleanup position vector here??
    _reference_vector = fft_keepdims(_reference_map).reshape(
            (batch_size, n_glimpses, n_queries, config.DIM)
    )

    # (batch_size, n_queries, N)
    query_positions_hd = (
            _direction_vectors.dimshuffle(0, "x", 1, 2) *
            _reference_vector
    )

    return S * query_positions_hd.conj

import numpy as np
import theano
import theano.tensor as T

import config
from utils import float_x


# Rotate the position encoding by this angle
theta = theano.shared(float_x(0.0))

U = T.stacklists([
    [T.cos(theta), -T.sin(theta)],
    [T.sin(theta), T.cos(theta)]
])

get_U = theano.function(
        [],
        U,
        allow_input_downcast=True)

import numpy as np
import theano

import config
from utils import float_x
from .encoder import PositionEncoder


rs = np.random.RandomState(seed=config.SEED)

K = theano.shared(float_x(
    rs.choice([-1, 1], size=(config.DIM, 2)) *
    rs.uniform(config.MIN_K, config.MAX_K, size=(config.DIM, 2))
))

L = PositionEncoder(K)

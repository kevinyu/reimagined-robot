import numpy as np
import theano
import theano.tensor as T

import config
from utils import float_x
from utils.complex import ComplexTuple


rs = np.random.RandomState(seed=config.SEED)

K = theano.shared(float_x(
    rs.choice([-1, 1], size=(config.DIM, 2)) *
    rs.uniform(config.MIN_K, config.MAX_K, size=(config.DIM, 2))
))


class PositionEncoder(object):
    def __init__(self, K):
        self.K = K

    def encode(self, X):
        phi = T.dot(X, self.K.T)
        return ComplexTuple(T.cos(phi), T.sin(phi))

    def encode_numeric(self, X):
        phi = np.dot(X, self.K.get_value().T)
        return ComplexTuple(np.cos(phi), np.sin(phi))


L = PositionEncoder(K)

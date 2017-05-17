import numpy as np
import theano
import theano.tensor as T

import config
from utils import float_x, tensor_softmax
from utils.complex import ComplexTuple


class PositionEncoder(object):
    def __init__(self, K):
        self.K = K
        # IMG_WIDTH x IMG_HEIGHT * DIM
        # FIXME: this class would be nicer if it
        # were independent of image size 
        X = self.encode_numeric(float_x(np.array(
            np.meshgrid(
                np.linspace(-1, 1, config.IMG_WIDTH),
                np.linspace(-1, 1, config.IMG_HEIGHT)
            )
        ).swapaxes(0, 2)))
        self.X = ComplexTuple(
                theano.shared(X.real),
                theano.shared(X.imag)
        )

    def encode(self, X):
        phi = T.dot(X, self.K.T)
        return ComplexTuple(T.cos(phi), T.sin(phi))

    def encode_numeric(self, X):
        phi = np.dot(X, self.K.get_value().T)
        return ComplexTuple(np.cos(phi), np.sin(phi))

    def FFT(self, X):
        """Convert a spatial map Y (n x n) into a hypervector"""
        raise NotImplementedError

    def IFFT(self, Y):
        """Convert a hypervector into a spatial map"""
        sim = self.X.dot(Y).real
        if Y.ndim == 1:
            return sim
        else:
            return tensor_softmax(sim)

